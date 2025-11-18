/**
 * @file american_pde_solver.cpp
 * @brief Implementation of American option PDE solvers
 */

#include "src/option/american_pde_solver.hpp"
#include "src/pde/core/time_domain.hpp"
#include <cmath>

namespace mango {

// ============================================================================
// AmericanPutSolver Implementation
// ============================================================================

AmericanPutSolver::AmericanPutSolver(
    const PricingParams& params,
    std::shared_ptr<AmericanSolverWorkspace> workspace,
    std::span<double> output_buffer)
    : params_(params)
    , workspace_(std::move(workspace))
    , output_buffer_(output_buffer)
{
    if (!workspace_) {
        throw std::invalid_argument("Workspace cannot be null");
    }

    // Validate output buffer size if provided
    if (!output_buffer_.empty()) {
        const size_t required_size = (workspace_->n_time() + 1) * workspace_->n_space();
        if (output_buffer_.size() < required_size) {
            throw std::invalid_argument("Output buffer too small");
        }
    }
}

std::expected<void, SolverError> AmericanPutSolver::solve() {
    // Setup grid and time domain
    std::span<const double> x_grid = workspace_->grid_span();
    std::shared_ptr<GridSpacing<double>> grid_spacing = workspace_->grid_spacing();
    TimeDomain time_domain(0.0, params_.maturity, params_.maturity / workspace_->n_time());

    // Put-specific boundary conditions
    auto left_bc = DirichletBC([](double /*t*/, double x) {
        // Deep ITM put: V/K = max(1 - e^x, 0)
        return std::max(1.0 - std::exp(x), 0.0);
    });

    auto right_bc = DirichletBC([](double /*t*/, double /*x*/) {
        // Deep OTM put: V → 0 as S → ∞
        return 0.0;
    });

    // Black-Scholes operator
    auto pde = operators::BlackScholesPDE<double>(
        params_.volatility,
        params_.rate,
        params_.dividend_yield);
    auto spatial_op = operators::create_spatial_operator(std::move(pde), grid_spacing);

    // Put obstacle: V ≥ max(1 - e^x, 0)
    auto obstacle = [](double /*t*/, std::span<const double> x, std::span<double> psi) {
        for (size_t i = 0; i < x.size(); ++i) {
            psi[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    };

    // Create PDE solver (local, auto-deduced types)
    PDESolver solver(
        x_grid, time_domain, TRBDF2Config{},
        left_bc, right_bc, spatial_op,
        obstacle,
        workspace_.get(),
        output_buffer_
    );

    // Initialize with put payoff at maturity
    solver.initialize([](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    });

    // Solve PDE
    auto solve_result = solver.solve();
    if (!solve_result) {
        return std::unexpected(solve_result.error());
    }

    // Extract solution before solver goes out of scope
    auto solution_view = solver.solution();
    solution_.assign(solution_view.begin(), solution_view.end());

    return {};  // Success
}

std::span<const double> AmericanPutSolver::solution() const {
    return solution_;
}

// ============================================================================
// AmericanCallSolver Implementation
// ============================================================================

AmericanCallSolver::AmericanCallSolver(
    const PricingParams& params,
    std::shared_ptr<AmericanSolverWorkspace> workspace,
    std::span<double> output_buffer)
    : params_(params)
    , workspace_(std::move(workspace))
    , output_buffer_(output_buffer)
{
    if (!workspace_) {
        throw std::invalid_argument("Workspace cannot be null");
    }

    // Validate output buffer size if provided
    if (!output_buffer_.empty()) {
        const size_t required_size = (workspace_->n_time() + 1) * workspace_->n_space();
        if (output_buffer_.size() < required_size) {
            throw std::invalid_argument("Output buffer too small");
        }
    }
}

std::expected<void, SolverError> AmericanCallSolver::solve() {
    // Setup grid and time domain
    std::span<const double> x_grid = workspace_->grid_span();
    std::shared_ptr<GridSpacing<double>> grid_spacing = workspace_->grid_spacing();
    TimeDomain time_domain(0.0, params_.maturity, params_.maturity / workspace_->n_time());

    // Call-specific boundary conditions
    auto left_bc = DirichletBC([](double /*t*/, double /*x*/) {
        // Deep OTM call: V → 0 as S → 0
        return 0.0;
    });

    auto right_bc = DirichletBC([this](double t, double x) {
        // Deep ITM call: V/K = e^x - e^(-r*τ)
        double discount = std::exp(-params_.rate * t);
        return std::exp(x) - discount;
    });

    // Black-Scholes operator
    auto pde = operators::BlackScholesPDE<double>(
        params_.volatility,
        params_.rate,
        params_.dividend_yield);
    auto spatial_op = operators::create_spatial_operator(std::move(pde), grid_spacing);

    // Call obstacle: V ≥ max(e^x - 1, 0)
    auto obstacle = [](double /*t*/, std::span<const double> x, std::span<double> psi) {
        for (size_t i = 0; i < x.size(); ++i) {
            psi[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    };

    // Create PDE solver (local, auto-deduced types)
    PDESolver solver(
        x_grid, time_domain, TRBDF2Config{},
        left_bc, right_bc, spatial_op,
        obstacle,
        workspace_.get(),
        output_buffer_
    );

    // Initialize with call payoff at maturity
    solver.initialize([](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    });

    // Solve PDE
    auto solve_result = solver.solve();
    if (!solve_result) {
        return std::unexpected(solve_result.error());
    }

    // Extract solution before solver goes out of scope
    auto solution_view = solver.solution();
    solution_.assign(solution_view.begin(), solution_view.end());

    return {};  // Success
}

std::span<const double> AmericanCallSolver::solution() const {
    return solution_;
}

} // namespace mango
