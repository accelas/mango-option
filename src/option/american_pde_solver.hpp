/**
 * @file american_pde_solver.hpp
 * @brief American option PDE solvers with static dispatch via CRTP
 *
 * Replaces template PDESolver + strategy pattern with inheritance + variant.
 * Each solver type (Put/Call) implements boundary conditions and obstacles directly.
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
 * Variant of American option solvers for static dispatch
 */
using AmericanSolverVariant = std::variant<AmericanPutSolver, AmericanCallSolver>;

/**
 * American Put Option PDE Solver
 *
 * Implements put-specific:
 * - Left BC: V/K = max(1 - e^x, 0) as x → -∞ (deep ITM)
 * - Right BC: V/K = 0 as x → +∞ (deep OTM)
 * - Obstacle: V ≥ max(1 - e^x, 0) (early exercise)
 */
class AmericanPutSolver {
public:
    AmericanPutSolver(const PricingParams& params,
                     std::shared_ptr<AmericanSolverWorkspace> workspace,
                     std::span<double> output_buffer = {});

    std::expected<void, SolverError> solve();

    std::span<const double> solution() const;

    // Grid info (for result extraction)
    double x_min() const { return workspace_->x_min(); }
    double x_max() const { return workspace_->x_max(); }
    size_t n_space() const { return workspace_->n_space(); }
    size_t n_time() const { return workspace_->n_time(); }

private:
    PricingParams params_;
    std::shared_ptr<AmericanSolverWorkspace> workspace_;
    std::span<double> output_buffer_;
    std::vector<double> solution_;  // Final solution (extracted after solve)
};

/**
 * American Call Option PDE Solver
 *
 * Implements call-specific:
 * - Left BC: V/K = 0 as x → -∞ (deep OTM)
 * - Right BC: V/K = e^x - e^(-r*τ) as x → +∞ (deep ITM)
 * - Obstacle: V ≥ max(e^x - 1, 0) (early exercise)
 */
class AmericanCallSolver {
public:
    AmericanCallSolver(const PricingParams& params,
                      std::shared_ptr<AmericanSolverWorkspace> workspace,
                      std::span<double> output_buffer = {});

    std::expected<void, SolverError> solve();

    std::span<const double> solution() const;

    // Grid info (for result extraction)
    double x_min() const { return workspace_->x_min(); }
    double x_max() const { return workspace_->x_max(); }
    size_t n_space() const { return workspace_->n_space(); }
    size_t n_time() const { return workspace_->n_time(); }

private:
    PricingParams params_;
    std::shared_ptr<AmericanSolverWorkspace> workspace_;
    std::span<double> output_buffer_;
    std::vector<double> solution_;  // Final solution (extracted after solve)
};

} // namespace mango
