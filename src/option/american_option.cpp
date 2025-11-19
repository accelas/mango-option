/**
 * @file american_option.cpp
 * @brief American option pricing solver implementation
 */

#include "src/option/american_option.hpp"
#include "src/option/american_pde_solver.hpp"
#include "src/option/american_solver_workspace.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace mango {

AmericanOptionSolver::AmericanOptionSolver(
    const AmericanOptionParams& params,
    std::shared_ptr<AmericanSolverWorkspace> workspace,
    std::span<double> output_buffer)
    : params_(params)
    , workspace_(std::move(workspace))
    , output_buffer_(output_buffer)
{
    // Validate parameters using unified validation
    auto validation = validate_pricing_params(params_);
    if (!validation) {
        throw std::invalid_argument(validation.error());
    }

    // Validate workspace is not null
    if (!workspace_) {
        throw std::invalid_argument("Workspace cannot be null");
    }

    // Validate output buffer size if provided
    if (!output_buffer_.empty()) {
        const size_t required_size = (workspace_->n_time() + 1) * workspace_->n_space();
        if (output_buffer_.size() < required_size) {
            throw std::invalid_argument("Output buffer too small: need " +
                std::to_string(required_size) + " but got " +
                std::to_string(output_buffer_.size()));
        }
    }
}

// ============================================================================
// Factory methods with expected-based validation
// ============================================================================

std::expected<AmericanOptionSolver, std::string> AmericanOptionSolver::create(
    const AmericanOptionParams& params,
    std::shared_ptr<AmericanSolverWorkspace> workspace,
    std::span<double> output_buffer) {

    // Validate workspace first
    if (!workspace) {
        return std::unexpected("Workspace cannot be null");
    }

    // Chain validation and construction using monadic operations
    return validate_pricing_params(params)
        .and_then([&]() -> std::expected<AmericanOptionSolver, std::string> {
            try {
                return AmericanOptionSolver(params, workspace, output_buffer);
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Failed to create solver: ") + e.what());
            }
        });
}

// ============================================================================
// Public API
// ============================================================================

std::expected<AmericanOptionResult, SolverError> AmericanOptionSolver::solve() {
    // Create appropriate solver based on option type
    AmericanSolverVariant solver = [&]() -> AmericanSolverVariant {
        switch (params_.type) {
            case OptionType::CALL:
                return AmericanCallSolver(params_, workspace_, output_buffer_);
            case OptionType::PUT:
                return AmericanPutSolver(params_, workspace_, output_buffer_);
            default:
                throw std::runtime_error("Unknown option type");
        }
    }();

    // Solve using variant dispatch (static, zero-cost)
    auto solve_result = std::visit([](auto& s) { return s.solve(); }, solver);
    if (!solve_result) {
        return std::unexpected(solve_result.error());
    }

    // Extract solution and grid info from solver
    AmericanOptionResult result;
    result.converged = true;

    std::visit([&](auto& s) {
        auto solution_view = s.solution();
        solution_.assign(solution_view.begin(), solution_view.end());
        result.solution.assign(solution_view.begin(), solution_view.end());

        // Store grid information
        result.n_space = s.n_space();
        result.n_time = s.n_time();
        result.x_min = s.x_min();
        result.x_max = s.x_max();
        result.strike = params_.strike;

        // Compute value at current spot using actual grid
        double current_moneyness = std::log(params_.spot / params_.strike);
        auto grid = workspace_->grid();
        double normalized_value = interpolate_solution(current_moneyness, grid);
        result.value = normalized_value * params_.strike;  // Denormalize

        // Store full surface if buffer was provided
        if (!output_buffer_.empty()) {
            // Buffer layout: [u_old_initial][step0][step1]...[step(n_time-1)]
            // Extract only the time steps (skip initial scratch)
            result.surface_2d.assign(
                output_buffer_.begin() + s.n_space(),  // Skip u_old_initial
                output_buffer_.end()                    // Include all time steps
            );
        }
    }, solver);

    solved_ = true;
    return result;
}

std::expected<AmericanOptionGreeks, SolverError> AmericanOptionSolver::compute_greeks() const {
    if (!solved_) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidState,
            .message = "Cannot compute Greeks: solve() has not been called or did not converge",
            .iterations = 0
        });
    }

    AmericanOptionGreeks greeks;
    greeks.delta = compute_delta();
    greeks.gamma = compute_gamma();
    greeks.theta = compute_theta();

    return greeks;
}

double AmericanOptionResult::value_at(double spot) const {
    // Convert spot to log-moneyness
    double x_target = std::log(spot / strike);

    // Get final spatial solution (present value)
    if (solution.empty() || n_space == 0) {
        return 0.0;
    }

    // Use the final solution directly
    std::span<const double> final_surface(solution.data(), solution.size());

    // Compute grid spacing (uniform grid)
    const double dx = (x_max - x_min) / (n_space - 1);

    // Boundary cases
    if (x_target <= x_min) {
        return final_surface[0] * strike;  // Denormalize
    }
    if (x_target >= x_max) {
        return final_surface[n_space-1] * strike;  // Denormalize
    }

    // Find bracketing indices
    size_t i = 0;
    while (i < n_space-1 && x_min + (i+1)*dx < x_target) {
        i++;
    }

    // Linear interpolation
    double x_i = x_min + i * dx;
    double x_i1 = x_min + (i+1) * dx;
    double t = (x_target - x_i) / (x_i1 - x_i);
    double normalized_value = (1.0 - t) * final_surface[i] + t * final_surface[i+1];

    return normalized_value * strike;  // Denormalize
}

double AmericanOptionSolver::interpolate_solution(double x_target,
                                                   std::span<const double> x_grid) const {
    const size_t n = solution_.size();

    // Boundary cases
    if (x_target <= x_grid[0]) return solution_[0];
    if (x_target >= x_grid[n-1]) return solution_[n-1];

    // Find bracketing indices
    size_t i = 0;
    while (i < n-1 && x_grid[i+1] < x_target) {
        i++;
    }

    // Linear interpolation
    double t = (x_target - x_grid[i]) / (x_grid[i+1] - x_grid[i]);
    return (1.0 - t) * solution_[i] + t * solution_[i+1];
}

std::vector<double> AmericanOptionSolver::get_solution() const {
    if (!solved_) {
        throw std::runtime_error("Solver has not been run yet");
    }
    return solution_;
}

double AmericanOptionSolver::compute_delta() const {
    if (!solved_) {
        return 0.0;  // No solution available
    }

    const size_t n = solution_.size();
    const double x_min = workspace_->x_min();
    const double x_max = workspace_->x_max();
    const double dx = (x_max - x_min) / (n - 1);

    // Find current spot in grid
    double current_moneyness = std::log(params_.spot / params_.strike);

    // Find the grid point closest to current_moneyness
    // Use the same approach as interpolate_solution
    size_t i = 0;
    while (i < n-1 && x_min + (i+1)*dx < current_moneyness) {
        i++;
    }

    // Ensure we're in valid interior range for centered differences
    if (i == 0) i = 1;
    if (i >= n-1) i = n-2;

    // Compute ∂V/∂x using centered finite difference
    // Note: solution_ stores V/K (normalized)
    const double half_dx_inv = 1.0 / (2.0 * dx);
    double dVdx = (solution_[i+1] - solution_[i-1]) * half_dx_inv;

    // Transform from log-moneyness to spot
    // V_dollar = V_norm * K
    // Delta = ∂V_dollar/∂S = K * ∂V_norm/∂x * ∂x/∂S
    //       = K * dVdx * (1/S)
    //       = (K/S) * dVdx
    // Use FMA: (K/S) * dVdx
    const double K_over_S = params_.strike / params_.spot;
    double delta = K_over_S * dVdx;

    return delta;
}

double AmericanOptionSolver::compute_gamma() const {
    if (!solved_) {
        return 0.0;  // No solution available
    }

    const size_t n = solution_.size();
    const double x_min = workspace_->x_min();
    const double x_max = workspace_->x_max();
    const double dx = (x_max - x_min) / (n - 1);

    // Find current spot in grid
    double current_moneyness = std::log(params_.spot / params_.strike);

    // Find the grid point closest to current_moneyness
    size_t i = 0;
    while (i < n-1 && x_min + (i+1)*dx < current_moneyness) {
        i++;
    }

    // Ensure we're in valid interior range for centered differences
    if (i == 0) i = 1;
    if (i >= n-1) i = n-2;

    // Centered second derivative: [V(i+1) - 2*V(i) + V(i-1)] / dx²
    // Use FMA: (V(i+1) + V(i-1)) / dx² - 2*V(i) / dx²
    const double dx2_inv = 1.0 / (dx * dx);
    double d2Vdx2 = std::fma(solution_[i+1] + solution_[i-1], dx2_inv, -2.0*solution_[i]*dx2_inv);
    // Centered first derivative: [V(i+1) - V(i-1)] / (2*dx)
    double dVdx = (solution_[i+1] - solution_[i-1]) / (2.0 * dx);

    // Transform from log-moneyness to spot using chain rule
    // x = ln(S/K), so ∂x/∂S = 1/S and ∂²x/∂S² = -1/S²
    //
    // V_dollar(S) = K * V_norm(x(S))
    //
    // First derivative:
    // dV/dS = K * dV_norm/dx * dx/dS = K * dV_norm/dx * (1/S)
    //
    // Second derivative:
    // d²V/dS² = d/dS[K * dV_norm/dx * (1/S)]
    //         = K * d/dS[dV_norm/dx * (1/S)]
    //         = K * [d²V_norm/dx² * (dx/dS) * (1/S) + dV_norm/dx * d/dS(1/S)]
    //         = K * [d²V_norm/dx² * (1/S²) + dV_norm/dx * (-1/S²)]
    //         = (K/S²) * [d²V_norm/dx² - dV_norm/dx]
    //
    double S = params_.spot;
    double K = params_.strike;
    // Use FMA: (K/S²) * (d2Vdx2 - dVdx) = (K/S²)*d2Vdx2 - (K/S²)*dVdx
    const double K_over_S2 = K / (S * S);
    double gamma = std::fma(K_over_S2, d2Vdx2, -K_over_S2 * dVdx);

    return gamma;
}

double AmericanOptionSolver::compute_theta() const {
    // Theta is time decay: ∂V/∂t
    // For American options with no closed form, accurate theta requires:
    // 1. Re-solving at slightly different time, or
    // 2. Evaluating the PDE operator: ∂V/∂t = L(V)
    //
    // Both approaches are expensive and complex. For now, return 0.0 as stub.
    // Future enhancement could evaluate the BS operator on the solution surface.

    return 0.0;  // Stub implementation
}

}  // namespace mango
