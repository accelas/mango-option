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
    std::shared_ptr<AmericanSolverWorkspace> workspace)
    : params_(params)
    , workspace_(std::move(workspace))
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
}

// ============================================================================
// Factory methods with expected-based validation
// ============================================================================

std::expected<AmericanOptionSolver, std::string> AmericanOptionSolver::create(
    const AmericanOptionParams& params,
    std::shared_ptr<AmericanSolverWorkspace> workspace) {

    // Validate workspace first
    if (!workspace) {
        return std::unexpected("Workspace cannot be null");
    }

    // Chain validation and construction using monadic operations
    return validate_pricing_params(params)
        .and_then([&]() -> std::expected<AmericanOptionSolver, std::string> {
            try {
                return AmericanOptionSolver(params, workspace);
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Failed to create solver: ") + e.what());
            }
        });
}

// ============================================================================
// Public API
// ============================================================================

std::expected<AmericanOptionResult, SolverError> AmericanOptionSolver::solve() {
    // Create appropriate solver based on option type using new API
    AmericanSolverVariant solver = [&]() -> AmericanSolverVariant {
        switch (params_.type) {
            case OptionType::CALL:
                return AmericanCallSolver(params_,
                                         workspace_->grid_with_solution(),
                                         workspace_->workspace_spans());
            case OptionType::PUT:
                return AmericanPutSolver(params_,
                                        workspace_->grid_with_solution(),
                                        workspace_->workspace_spans());
            default:
                throw std::runtime_error("Unknown option type");
        }
    }();

    // Initialize with payoff at maturity (t=0 in PDE time)
    std::visit([&](auto& s) {
        using SolverType = std::decay_t<decltype(s)>;
        s.initialize(SolverType::payoff);
    }, solver);

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

        // Store grid information (including actual grid points)
        result.n_space = s.n_space();
        result.n_time = s.n_time();
        result.x_min = workspace_->x_min();
        result.x_max = workspace_->x_max();
        result.strike = params_.strike;

        // Store actual grid points for interpolation
        auto grid = workspace_->grid_with_solution()->x();
        result.x_grid.assign(grid.begin(), grid.end());

        // Compute value at current spot using actual grid
        double current_moneyness = std::log(params_.spot / params_.strike);
        double normalized_value = interpolate_solution(current_moneyness, grid);
        result.value = normalized_value * params_.strike;  // Denormalize

        // Note: surface_2d is no longer populated (new design only stores final solution)
        // For time-dependent data, users should implement custom surface collection
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
    if (solution.empty() || n_space == 0 || x_grid.empty()) {
        return 0.0;
    }

    // Use the final solution directly
    std::span<const double> final_surface(solution.data(), solution.size());

    // Boundary cases (use actual grid points)
    if (x_target <= x_grid[0]) {
        return final_surface[0] * strike;  // Denormalize
    }
    if (x_target >= x_grid[n_space-1]) {
        return final_surface[n_space-1] * strike;  // Denormalize
    }

    // Find bracketing indices using actual grid points
    size_t i = 0;
    while (i < n_space-1 && x_grid[i+1] < x_target) {
        i++;
    }

    // Linear interpolation using actual grid spacing
    double t = (x_target - x_grid[i]) / (x_grid[i+1] - x_grid[i]);
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

size_t AmericanOptionSolver::find_grid_index(double log_moneyness) const {
    const size_t n = solution_.size();
    auto grid = workspace_->grid_with_solution()->x();

    // Binary search for closest grid point
    size_t i = 0;
    while (i < n-1 && grid[i+1] < log_moneyness) {
        i++;
    }

    // Ensure we're in valid interior range for centered differences
    if (i == 0) i = 1;
    if (i >= n-1) i = n-2;

    return i;
}

const operators::CenteredDifference<double>&
AmericanOptionSolver::get_diff_operator() const {
    if (!diff_op_) {
        // Create GridSpacing from GridWithSolution's grid (NEW API)
        auto grid_view = GridView<double>(workspace_->grid_with_solution()->x());
        auto grid_spacing = GridSpacing<double>(grid_view);
        diff_op_ = std::make_unique<operators::CenteredDifference<double>>(
            grid_spacing);
    }
    return *diff_op_;
}

double AmericanOptionSolver::compute_delta() const {
    if (!solved_) {
        return 0.0;  // No solution available
    }

    // Find current spot in grid
    double current_moneyness = std::log(params_.spot / params_.strike);
    size_t i = find_grid_index(current_moneyness);

    // Compute ∂V/∂x using PDE operator (handles non-uniform grids + SIMD)
    std::vector<double> du_dx(solution_.size());
    get_diff_operator().compute_first_derivative(
        solution_, du_dx, i, i+1);

    // Transform: Delta = (K/S) * ∂V/∂x
    return (params_.strike / params_.spot) * du_dx[i];
}

double AmericanOptionSolver::compute_gamma() const {
    if (!solved_) {
        return 0.0;
    }

    // Find current spot in grid
    double current_moneyness = std::log(params_.spot / params_.strike);
    size_t i = find_grid_index(current_moneyness);

    // Compute ∂V/∂x and ∂²V/∂x² using PDE operator
    std::vector<double> du_dx(solution_.size());
    std::vector<double> d2u_dx2(solution_.size());

    get_diff_operator().compute_first_derivative(solution_, du_dx, i, i+1);
    get_diff_operator().compute_second_derivative(solution_, d2u_dx2, i, i+1);

    // Transform: Gamma = (K/S²) * [∂²V/∂x² - ∂V/∂x]
    double K_over_S2 = params_.strike / (params_.spot * params_.spot);
    return std::fma(K_over_S2, d2u_dx2[i], -K_over_S2 * du_dx[i]);
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
