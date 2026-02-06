// SPDX-License-Identifier: MIT
/**
 * @file american_option.cpp
 * @brief American option pricing solver implementation
 *
 * Contains the CRTP PDE solvers (AmericanPutSolver, AmericanCallSolver) as
 * implementation details. Each solver owns its boundary conditions, spatial
 * operator, obstacle, and discrete dividend event handling.
 */

#include "mango/option/american_option.hpp"
#include "mango/pde/core/pde_solver.hpp"
#include "mango/pde/core/boundary_conditions.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/pde/core/time_domain.hpp"
#include "mango/pde/operators/operator_factory.hpp"
#include "mango/pde/operators/black_scholes_pde.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include <algorithm>
#include <cmath>
#include <variant>
#include <vector>
#include <format>
#include <cassert>
#include <functional>
#include <memory>
#include <span>

namespace mango {
namespace {

// Helper for std::visit with multiple lambdas
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };

// ============================================================================
// Grid resolution
// ============================================================================

std::pair<GridSpec<double>, TimeDomain> resolve_grid(
    const PricingParams& params,
    const std::optional<PDEGridSpec>& grid)
{
    if (!grid.has_value()) {
        return estimate_pde_grid(params);
    }
    return std::visit(overloaded{
        [&](const GridAccuracyParams& acc) {
            return estimate_pde_grid(params, acc);
        },
        [&](const PDEGridConfig& eg) {
            auto td = eg.mandatory_times.empty()
                ? TimeDomain::from_n_steps(0.0, params.maturity, eg.n_time)
                : TimeDomain::with_mandatory_points(0.0, params.maturity,
                    params.maturity / static_cast<double>(eg.n_time), eg.mandatory_times);
            return std::make_pair(eg.grid_spec, td);
        }
    }, *grid);
}

// ============================================================================
// Discrete dividend shift (shared implementation)
// ============================================================================

/// Core dividend shift: rebuilds spline with current solution, evaluates
/// u(x') where x' = ln(exp(x) - D/K) for each grid point.
TemporalEventCallback make_dividend_event(
    double dividend_amount, double strike, double intrinsic_fallback,
    CubicSpline<double>* spline)
{
    const double d = dividend_amount / strike;
    const double fallback = intrinsic_fallback;

    return [d, fallback, spline](double /*t*/, std::span<const double> x, std::span<double> u) {
        if (d <= 0.0) return;

        const size_t n = x.size();

        auto err = spline->rebuild_same_grid(std::span<const double>(u.data(), u.size()));
        if (err.has_value()) return;

        const double x_lo = x[0];
        const double x_hi = x[n - 1];

        for (size_t i = 0; i < n; ++i) {
            double S_over_K = std::exp(x[i]);
            double S_adj_over_K = S_over_K - d;

            if (S_adj_over_K > 1e-10) {
                double x_shifted = std::log(S_adj_over_K);
                if (x_shifted < x_lo) {
                    u[i] = fallback;
                } else {
                    u[i] = spline->eval(std::min(x_shifted, x_hi));
                }
            } else {
                u[i] = fallback;
            }
        }
    };
}

// ============================================================================
// American Put PDE Solver (CRTP)
// ============================================================================

class AmericanPutSolver : public PDESolver<AmericanPutSolver> {
public:
    using RateFn = std::function<double(double)>;
    using PDEType = operators::BlackScholesPDE<double, RateFn>;
    using SpatialOpType = operators::SpatialOperator<PDEType, double>;

    AmericanPutSolver(const PricingParams& params,
                     std::shared_ptr<Grid<double>> grid,
                     PDEWorkspace workspace)
        : PDESolver<AmericanPutSolver>(grid, workspace)
        , params_(params)
        , grid_(grid)
        , workspace_local_(workspace)
        , left_bc_(create_left_bc())
        , right_bc_(create_right_bc())
        , spatial_op_(create_spatial_op(workspace))
    {
        assert(grid_ != nullptr && "Grid cannot be null (programming error)");
    }

    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

    void obstacle(double /*t*/, std::span<const double> x, std::span<double> psi) const {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            psi[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    }

    size_t n_space() const { return grid_->n_space(); }
    size_t n_time() const { return grid_->time().n_steps(); }

    static void payoff(std::span<const double> x, std::span<double> u) {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    }

    /// Initialize dividend events. Must be called after the object is in its
    /// final memory location (e.g. after placement into a std::variant) because
    /// the event callbacks capture &dividend_spline_.
    void init_dividends() {
        if (params_.discrete_dividends.empty()) return;

        auto x = grid_->x();
        auto scratch = workspace_local_.reserved1();
        std::fill(scratch.begin(), scratch.end(), 0.0);
        [[maybe_unused]] auto err = dividend_spline_.build(
            x, std::span<const double>(scratch.data(), x.size()));

        for (const auto& div : params_.discrete_dividends) {
            double tau = params_.maturity - div.calendar_time;
            if (tau > 0.0 && tau < params_.maturity) {
                this->add_temporal_event(tau,
                    make_dividend_event(div.amount, params_.strike, 1.0, &dividend_spline_));
            }
        }
    }

    struct LeftBCFunction {
        double operator()(double /*t*/, double x) const {
            return std::max(1.0 - std::exp(x), 0.0);
        }
    };

    struct RightBCFunction {
        double operator()(double /*t*/, double /*x*/) const {
            return 0.0;
        }
    };

    static DirichletBC<LeftBCFunction> create_left_bc() {
        return DirichletBC(LeftBCFunction{});
    }

    static DirichletBC<RightBCFunction> create_right_bc() {
        return DirichletBC(RightBCFunction{});
    }

    SpatialOpType create_spatial_op(PDEWorkspace& workspace) const {
        auto pde = PDEType(
            params_.volatility,
            make_rate_fn(params_.rate, params_.maturity),
            params_.dividend_yield);
        auto spacing_ptr = std::make_shared<GridSpacing<double>>(grid_->spacing());
        return operators::create_spatial_operator(std::move(pde), spacing_ptr, workspace);
    }

    PricingParams params_;
    std::shared_ptr<Grid<double>> grid_;
    PDEWorkspace workspace_local_;
    DirichletBC<LeftBCFunction> left_bc_;
    DirichletBC<RightBCFunction> right_bc_;
    SpatialOpType spatial_op_;
    CubicSpline<double> dividend_spline_;
};

// ============================================================================
// American Call PDE Solver (CRTP)
// ============================================================================

class AmericanCallSolver : public PDESolver<AmericanCallSolver> {
public:
    using RateFn = std::function<double(double)>;
    using PDEType = operators::BlackScholesPDE<double, RateFn>;
    using SpatialOpType = operators::SpatialOperator<PDEType, double>;

    AmericanCallSolver(const PricingParams& params,
                      std::shared_ptr<Grid<double>> grid,
                      PDEWorkspace workspace)
        : PDESolver<AmericanCallSolver>(grid, workspace)
        , params_(params)
        , grid_(grid)
        , workspace_local_(workspace)
        , left_bc_(create_left_bc())
        , right_bc_(create_right_bc())
        , spatial_op_(create_spatial_op(workspace))
    {
        assert(grid_ != nullptr && "Grid cannot be null (programming error)");
    }

    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

    void obstacle(double /*t*/, std::span<const double> x, std::span<double> psi) const {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            psi[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    }

    size_t n_space() const { return grid_->n_space(); }
    size_t n_time() const { return grid_->time().n_steps(); }

    static void payoff(std::span<const double> x, std::span<double> u) {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    }

    /// Initialize dividend events. Must be called after the object is in its
    /// final memory location (e.g. after placement into a std::variant) because
    /// the event callbacks capture &dividend_spline_.
    void init_dividends() {
        if (params_.discrete_dividends.empty()) return;

        auto x = grid_->x();
        auto scratch = workspace_local_.reserved1();
        std::fill(scratch.begin(), scratch.end(), 0.0);
        [[maybe_unused]] auto err = dividend_spline_.build(
            x, std::span<const double>(scratch.data(), x.size()));

        for (const auto& div : params_.discrete_dividends) {
            double tau = params_.maturity - div.calendar_time;
            if (tau > 0.0 && tau < params_.maturity) {
                this->add_temporal_event(tau,
                    make_dividend_event(div.amount, params_.strike, 0.0, &dividend_spline_));
            }
        }
    }

    struct LeftBCFunction {
        double operator()(double /*t*/, double /*x*/) const {
            return 0.0;
        }
    };

    struct RightBCFunction {
        std::function<double(double)> forward_discount_fn;

        double operator()(double t, double x) const {
            return std::exp(x) - forward_discount_fn(t);
        }
    };

    static DirichletBC<LeftBCFunction> create_left_bc() {
        return DirichletBC(LeftBCFunction{});
    }

    DirichletBC<RightBCFunction> create_right_bc() const {
        return DirichletBC(RightBCFunction{make_forward_discount_fn(params_.rate, params_.maturity)});
    }

    SpatialOpType create_spatial_op(PDEWorkspace& workspace) const {
        auto pde = PDEType(
            params_.volatility,
            make_rate_fn(params_.rate, params_.maturity),
            params_.dividend_yield);
        auto spacing_ptr = std::make_shared<GridSpacing<double>>(grid_->spacing());
        return operators::create_spatial_operator(std::move(pde), spacing_ptr, workspace);
    }

    PricingParams params_;
    std::shared_ptr<Grid<double>> grid_;
    PDEWorkspace workspace_local_;
    DirichletBC<LeftBCFunction> left_bc_;
    DirichletBC<RightBCFunction> right_bc_;
    SpatialOpType spatial_op_;
    CubicSpline<double> dividend_spline_;
};

using AmericanSolverVariant = std::variant<AmericanPutSolver, AmericanCallSolver>;

}  // anonymous namespace

// ============================================================================
// AmericanOptionSolver public API
// ============================================================================

std::expected<AmericanOptionSolver, ValidationError>
AmericanOptionSolver::create(
    const PricingParams& params,
    PDEWorkspace workspace,
    std::optional<PDEGridSpec> grid,
    std::optional<std::span<const double>> snapshot_times)
{
    auto validation = validate_pricing_params(params);
    if (!validation) {
        return std::unexpected(validation.error());
    }

    auto grid_config = resolve_grid(params, grid);

    if (workspace.size() != grid_config.first.n_points()) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidGridSize,
            static_cast<double>(workspace.size()),
            grid_config.first.n_points()));
    }

    return AmericanOptionSolver(params, workspace, std::move(grid_config), snapshot_times);
}

AmericanOptionSolver::AmericanOptionSolver(
    const PricingParams& params,
    PDEWorkspace workspace,
    std::pair<GridSpec<double>, TimeDomain> grid_config,
    std::optional<std::span<const double>> snapshot_times)
    : params_(params)
    , workspace_(workspace)
    , grid_config_(std::move(grid_config))
{
    trbdf2_config_.rannacher_startup = true;

    if (snapshot_times.has_value()) {
        snapshot_times_.assign(snapshot_times->begin(), snapshot_times->end());
    }
}

std::expected<AmericanOptionResult, SolverError> AmericanOptionSolver::solve() {
    auto& [grid_spec, time_domain] = grid_config_;

    auto grid_result = Grid<double>::create(
        grid_spec, time_domain,
        snapshot_times_.empty() ? std::span<const double>() : std::span<const double>(snapshot_times_));

    if (!grid_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .iterations = 0,
            .residual = grid_result.error().value
        });
    }
    auto grid = grid_result.value();

    // Initialize dx in workspace from grid spacing
    auto dx_span = workspace_.dx();
    auto grid_points = grid->x();
    for (size_t i = 0; i < grid_points.size() - 1; ++i) {
        dx_span[i] = grid_points[i + 1] - grid_points[i];
    }

    // Construct variant solver
    AmericanSolverVariant solver = (params_.option_type == OptionType::PUT)
        ? AmericanSolverVariant{AmericanPutSolver(params_, grid, workspace_)}
        : AmericanSolverVariant{AmericanCallSolver(params_, grid, workspace_)};

    // Initialize dividends after variant construction so that the captured
    // &dividend_spline_ pointer refers to the object's final location.
    auto solve_result = std::visit([&](auto& pde_solver) {
        pde_solver.init_dividends();
        if (custom_ic_) {
            pde_solver.initialize(*custom_ic_);
        } else {
            pde_solver.initialize(std::remove_reference_t<decltype(pde_solver)>::payoff);
        }
        pde_solver.set_config(trbdf2_config_);
        return pde_solver.solve();
    }, solver);

    if (!solve_result.has_value()) {
        return std::unexpected(solve_result.error());
    }

    return AmericanOptionResult(grid, params_);
}

}  // namespace mango
