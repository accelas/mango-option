/**
 * @file american_pde_solver.hpp
 * @brief American option PDE solvers inheriting from PDESolver via CRTP
 *
 * AmericanPutSolver and AmericanCallSolver inherit directly from
 * PDESolver<Derived> and implement the CRTP interface:
 * - left_boundary()
 * - right_boundary()
 * - spatial_operator()
 */

#pragma once

#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
#include "src/option/option_spec.hpp"
#include "src/support/error_types.hpp"
#include <cassert>
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
 * American Put Option PDE Solver
 *
 * Inherits from PDESolver via CRTP and implements put-specific:
 * - Left BC: V/K = max(1 - e^x, 0) as x → -∞ (deep ITM)
 * - Right BC: V/K = 0 as x → +∞ (deep OTM)
 * - Obstacle: V ≥ max(1 - e^x, 0) (early exercise)
 */
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
        , left_bc_(create_left_bc())
        , right_bc_(create_right_bc())
        , spatial_op_(create_spatial_op())
    {
        // Precondition: grid must not be null (comes from Grid::create which returns std::expected)
        // If callers check the std::expected, grid will never be null.
        // Null grid is a programming error, not a runtime condition.
        assert(grid_ != nullptr && "Grid cannot be null (programming error)");
    }

    // CRTP interface - called by PDESolver base class via static dispatch
    // Returns references to cached objects (no allocation per call!)
    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

    // Obstacle condition for American put: V ≥ max(1 - e^x, 0)
    void obstacle(double /*t*/, std::span<const double> x, std::span<double> psi) const {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            psi[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    }

    // Grid info accessors
    size_t n_space() const { return grid_->n_space(); }
    size_t n_time() const { return grid_->time().n_steps(); }

    /// Normalized put payoff: max(1 - exp(x), 0) where x = ln(S/K)
    static void payoff(std::span<const double> x, std::span<double> u) {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    }

private:
    // Function objects for boundary conditions (zero overhead vs lambdas)
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

    SpatialOpType create_spatial_op() const {
        auto pde = PDEType(
            params_.volatility,
            make_rate_fn(params_.rate),
            params_.dividend_yield);
        auto spacing_ptr = std::make_shared<GridSpacing<double>>(grid_->spacing());
        return operators::create_spatial_operator(std::move(pde), spacing_ptr);
    }

    PricingParams params_;
    std::shared_ptr<Grid<double>> grid_;

    // Cached BC and spatial operator (created once, reused many times)
    DirichletBC<LeftBCFunction> left_bc_;
    DirichletBC<RightBCFunction> right_bc_;
    SpatialOpType spatial_op_;
};

/**
 * American Call Option PDE Solver
 *
 * Inherits from PDESolver via CRTP and implements call-specific:
 * - Left BC: V/K = 0 as x → -∞ (deep OTM)
 * - Right BC: V/K = e^x - e^(-r*τ) as x → +∞ (deep ITM)
 * - Obstacle: V ≥ max(e^x - 1, 0) (early exercise)
 */
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
        , left_bc_(create_left_bc())
        , right_bc_(create_right_bc())
        , spatial_op_(create_spatial_op())
    {
        // Precondition: grid must not be null (comes from Grid::create which returns std::expected)
        // If callers check the std::expected, grid will never be null.
        // Null grid is a programming error, not a runtime condition.
        assert(grid_ != nullptr && "Grid cannot be null (programming error)");
    }

    // CRTP interface - called by PDESolver base class via static dispatch
    // Returns references to cached objects (no allocation per call!)
    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

    // Obstacle condition for American call: V ≥ max(e^x - 1, 0)
    void obstacle(double /*t*/, std::span<const double> x, std::span<double> psi) const {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            psi[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    }

    // Grid info accessors
    size_t n_space() const { return grid_->n_space(); }
    size_t n_time() const { return grid_->time().n_steps(); }

    /// Normalized call payoff: max(exp(x) - 1, 0) where x = ln(S/K)
    static void payoff(std::span<const double> x, std::span<double> u) {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    }

private:
    // Function objects for boundary conditions (zero overhead vs lambdas)
    struct LeftBCFunction {
        double operator()(double /*t*/, double /*x*/) const {
            return 0.0;
        }
    };

    struct RightBCFunction {
        std::function<double(double)> rate_fn;  // Capture rate function

        double operator()(double t, double x) const {
            double discount = std::exp(-rate_fn(t) * t);
            return std::exp(x) - discount;
        }
    };

    static DirichletBC<LeftBCFunction> create_left_bc() {
        return DirichletBC(LeftBCFunction{});
    }

    DirichletBC<RightBCFunction> create_right_bc() const {
        return DirichletBC(RightBCFunction{make_rate_fn(params_.rate)});
    }

    SpatialOpType create_spatial_op() const {
        auto pde = PDEType(
            params_.volatility,
            make_rate_fn(params_.rate),
            params_.dividend_yield);
        auto spacing_ptr = std::make_shared<GridSpacing<double>>(grid_->spacing());
        return operators::create_spatial_operator(std::move(pde), spacing_ptr);
    }

    PricingParams params_;
    std::shared_ptr<Grid<double>> grid_;

    // Cached BC and spatial operator (created once, reused many times)
    DirichletBC<LeftBCFunction> left_bc_;
    DirichletBC<RightBCFunction> right_bc_;
    SpatialOpType spatial_op_;
};

} // namespace mango
