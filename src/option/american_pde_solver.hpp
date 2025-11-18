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
 * American Put Option PDE Solver
 *
 * Inherits from PDESolver via CRTP and implements put-specific:
 * - Left BC: V/K = max(1 - e^x, 0) as x → -∞ (deep ITM)
 * - Right BC: V/K = 0 as x → +∞ (deep OTM)
 * - Obstacle: V ≥ max(1 - e^x, 0) (early exercise)
 */
class AmericanPutSolver : public PDESolver<AmericanPutSolver> {
public:
    AmericanPutSolver(const PricingParams& params,
                     std::shared_ptr<AmericanSolverWorkspace> workspace,
                     std::span<double> output_buffer = {})
        : PDESolver<AmericanPutSolver>(
              workspace->grid_span(),
              TimeDomain(0.0, params.maturity, params.maturity / workspace->n_time()),
              create_obstacle(),
              workspace->pde_workspace(),
              output_buffer)
        , params_(params)
        , workspace_(std::move(workspace))
        , grid_spacing_(workspace_->grid_spacing())
        , left_bc_(create_left_bc())
        , right_bc_(create_right_bc())
        , spatial_op_(create_spatial_op())
    {
        if (!workspace_) {
            throw std::invalid_argument("Workspace cannot be null");
        }
    }

    // CRTP interface - called by PDESolver base class via static dispatch
    // Returns references to cached objects (no allocation per call!)
    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

    // Grid info accessors
    double x_min() const { return workspace_->x_min(); }
    double x_max() const { return workspace_->x_max(); }
    size_t n_space() const { return workspace_->n_space(); }
    size_t n_time() const { return workspace_->n_time(); }

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

    static ObstacleCallback create_obstacle() {
        return [](double /*t*/, std::span<const double> x, std::span<double> psi) {
            for (size_t i = 0; i < x.size(); ++i) {
                psi[i] = std::max(1.0 - std::exp(x[i]), 0.0);
            }
        };
    }

    static DirichletBC<LeftBCFunction> create_left_bc() {
        return DirichletBC(LeftBCFunction{});
    }

    static DirichletBC<RightBCFunction> create_right_bc() {
        return DirichletBC(RightBCFunction{});
    }

    operators::SpatialOperator<operators::BlackScholesPDE<double>, double> create_spatial_op() const {
        auto pde = operators::BlackScholesPDE<double>(
            params_.volatility,
            params_.rate,
            params_.dividend_yield);
        return operators::create_spatial_operator(std::move(pde), grid_spacing_);
    }

    PricingParams params_;
    std::shared_ptr<AmericanSolverWorkspace> workspace_;
    std::shared_ptr<GridSpacing<double>> grid_spacing_;

    // Cached BC and spatial operator (created once, reused many times)
    DirichletBC<LeftBCFunction> left_bc_;
    DirichletBC<RightBCFunction> right_bc_;
    operators::SpatialOperator<operators::BlackScholesPDE<double>, double> spatial_op_;
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
    AmericanCallSolver(const PricingParams& params,
                      std::shared_ptr<AmericanSolverWorkspace> workspace,
                      std::span<double> output_buffer = {})
        : PDESolver<AmericanCallSolver>(
              workspace->grid_span(),
              TimeDomain(0.0, params.maturity, params.maturity / workspace->n_time()),
              create_obstacle(),
              workspace->pde_workspace(),
              output_buffer)
        , params_(params)
        , workspace_(std::move(workspace))
        , grid_spacing_(workspace_->grid_spacing())
        , left_bc_(create_left_bc())
        , right_bc_(create_right_bc())
        , spatial_op_(create_spatial_op())
    {
        if (!workspace_) {
            throw std::invalid_argument("Workspace cannot be null");
        }
    }

    // CRTP interface - called by PDESolver base class via static dispatch
    // Returns references to cached objects (no allocation per call!)
    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

    // Grid info accessors
    double x_min() const { return workspace_->x_min(); }
    double x_max() const { return workspace_->x_max(); }
    size_t n_space() const { return workspace_->n_space(); }
    size_t n_time() const { return workspace_->n_time(); }

private:
    // Function objects for boundary conditions (zero overhead vs lambdas)
    struct LeftBCFunction {
        double operator()(double /*t*/, double /*x*/) const {
            return 0.0;
        }
    };

    struct RightBCFunction {
        double rate;  // Capture risk-free rate

        double operator()(double t, double x) const {
            double discount = std::exp(-rate * t);
            return std::exp(x) - discount;
        }
    };

    static ObstacleCallback create_obstacle() {
        return [](double /*t*/, std::span<const double> x, std::span<double> psi) {
            for (size_t i = 0; i < x.size(); ++i) {
                psi[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
            }
        };
    }

    static DirichletBC<LeftBCFunction> create_left_bc() {
        return DirichletBC(LeftBCFunction{});
    }

    DirichletBC<RightBCFunction> create_right_bc() const {
        return DirichletBC(RightBCFunction{params_.rate});
    }

    operators::SpatialOperator<operators::BlackScholesPDE<double>, double> create_spatial_op() const {
        auto pde = operators::BlackScholesPDE<double>(
            params_.volatility,
            params_.rate,
            params_.dividend_yield);
        return operators::create_spatial_operator(std::move(pde), grid_spacing_);
    }

    PricingParams params_;
    std::shared_ptr<AmericanSolverWorkspace> workspace_;
    std::shared_ptr<GridSpacing<double>> grid_spacing_;

    // Cached BC and spatial operator (created once, reused many times)
    DirichletBC<LeftBCFunction> left_bc_;
    DirichletBC<RightBCFunction> right_bc_;
    operators::SpatialOperator<operators::BlackScholesPDE<double>, double> spatial_op_;
};

} // namespace mango
