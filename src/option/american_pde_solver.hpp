/**
 * @file american_pde_solver.hpp
 * @brief American option PDE solvers with static dispatch via CRTP
 *
 * Uses Curiously Recurring Template Pattern (CRTP) to implement compile-time
 * polymorphism. AmericanSolverBase contains common solve() logic and calls
 * derived class methods via static_cast<Derived&>(*this).
 */

#pragma once

#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
#include "src/option/american_solver_workspace.hpp"
#include "src/option/option_spec.hpp"
#include "src/support/error_types.hpp"
#include <expected>
#include <variant>
#include <span>
#include <cmath>
#include <memory>
#include <functional>

namespace mango {

// Forward declarations
class AmericanPutSolver;
class AmericanCallSolver;

/**
 * Variant of American option solvers for runtime dispatch
 */
using AmericanSolverVariant = std::variant<AmericanPutSolver, AmericanCallSolver>;

/**
 * CRTP Base class for American option solvers
 *
 * Contains common solve() logic and delegates option-specific details
 * (boundary conditions, obstacles, initial conditions) to derived classes
 * via compile-time static dispatch.
 *
 * @tparam Derived The derived solver class (AmericanPutSolver or AmericanCallSolver)
 */
template<typename Derived>
class AmericanSolverBase {
public:
    AmericanSolverBase(const PricingParams& params,
                      std::shared_ptr<AmericanSolverWorkspace> workspace,
                      std::span<double> output_buffer = {})
        : params_(params)
        , workspace_(std::move(workspace))
        , output_buffer_(output_buffer)
    {
        if (!workspace_) {
            throw std::invalid_argument("Workspace cannot be null");
        }

        if (!output_buffer_.empty()) {
            const size_t required_size = (workspace_->n_time() + 1) * workspace_->n_space();
            if (output_buffer_.size() < required_size) {
                throw std::invalid_argument("Output buffer too small");
            }
        }
    }

    /**
     * Solve the PDE using CRTP to call derived class methods
     */
    std::expected<void, SolverError> solve() {
        auto& derived = static_cast<Derived&>(*this);

        // Setup grid and time domain
        std::span<const double> x_grid = workspace_->grid_span();
        std::shared_ptr<GridSpacing<double>> grid_spacing = workspace_->grid_spacing();
        TimeDomain time_domain(0.0, params_.maturity, params_.maturity / workspace_->n_time());

        // Get boundary conditions from derived class (CRTP call)
        auto left_bc = derived.left_boundary();
        auto right_bc = derived.right_boundary();

        // Black-Scholes operator
        auto pde = operators::BlackScholesPDE<double>(
            params_.volatility,
            params_.rate,
            params_.dividend_yield);
        auto spatial_op = operators::create_spatial_operator(std::move(pde), grid_spacing);

        // Get obstacle from derived class (CRTP call)
        auto obstacle = derived.obstacle_callback();

        // Create PDE solver
        PDESolver solver(
            x_grid, time_domain, TRBDF2Config{},
            left_bc, right_bc, spatial_op,
            obstacle,
            workspace_.get(),
            output_buffer_
        );

        // Initialize with derived's initial condition (CRTP call)
        solver.initialize([&derived](std::span<const double> x, std::span<double> u) {
            derived.initial_condition(x, u);
        });

        // Solve PDE
        auto solve_result = solver.solve();
        if (!solve_result) {
            return std::unexpected(solve_result.error());
        }

        // Extract solution
        auto solution_view = solver.solution();
        solution_.assign(solution_view.begin(), solution_view.end());

        return {};
    }

    std::span<const double> solution() const { return solution_; }

    // Grid info accessors
    double x_min() const { return workspace_->x_min(); }
    double x_max() const { return workspace_->x_max(); }
    size_t n_space() const { return workspace_->n_space(); }
    size_t n_time() const { return workspace_->n_time(); }

protected:
    PricingParams params_;
    std::shared_ptr<AmericanSolverWorkspace> workspace_;
    std::span<double> output_buffer_;
    std::vector<double> solution_;
};

/**
 * American Put Option PDE Solver
 *
 * Inherits from AmericanSolverBase via CRTP and implements put-specific:
 * - Left BC: V/K = max(1 - e^x, 0) as x → -∞ (deep ITM)
 * - Right BC: V/K = 0 as x → +∞ (deep OTM)
 * - Obstacle: V ≥ max(1 - e^x, 0) (early exercise)
 */
class AmericanPutSolver : public AmericanSolverBase<AmericanPutSolver> {
public:
    using AmericanSolverBase::AmericanSolverBase;  // Inherit constructors

    // CRTP interface - called by base class via static dispatch
    auto left_boundary() const {
        return DirichletBC([](double /*t*/, double x) {
            return std::max(1.0 - std::exp(x), 0.0);
        });
    }

    auto right_boundary() const {
        return DirichletBC([](double /*t*/, double /*x*/) {
            return 0.0;
        });
    }

    auto obstacle_callback() const {
        return [](double /*t*/, std::span<const double> x, std::span<double> psi) {
            for (size_t i = 0; i < x.size(); ++i) {
                psi[i] = std::max(1.0 - std::exp(x[i]), 0.0);
            }
        };
    }

    void initial_condition(std::span<const double> x, std::span<double> u) const {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    }
};

/**
 * American Call Option PDE Solver
 *
 * Inherits from AmericanSolverBase via CRTP and implements call-specific:
 * - Left BC: V/K = 0 as x → -∞ (deep OTM)
 * - Right BC: V/K = e^x - e^(-r*τ) as x → +∞ (deep ITM)
 * - Obstacle: V ≥ max(e^x - 1, 0) (early exercise)
 */
class AmericanCallSolver : public AmericanSolverBase<AmericanCallSolver> {
public:
    using AmericanSolverBase::AmericanSolverBase;  // Inherit constructors

    // CRTP interface - called by base class via static dispatch
    auto left_boundary() const {
        return DirichletBC([](double /*t*/, double /*x*/) {
            return 0.0;
        });
    }

    auto right_boundary() const {
        // Capture params_ for use in lambda
        return DirichletBC([this](double t, double x) {
            double discount = std::exp(-this->params_.rate * t);
            return std::exp(x) - discount;
        });
    }

    auto obstacle_callback() const {
        return [](double /*t*/, std::span<const double> x, std::span<double> psi) {
            for (size_t i = 0; i < x.size(); ++i) {
                psi[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
            }
        };
    }

    void initial_condition(std::span<const double> x, std::span<double> u) const {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    }
};

} // namespace mango
