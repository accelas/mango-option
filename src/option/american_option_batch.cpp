/**
 * @file american_option_batch.cpp
 * @brief Implementation of batch and normalized chain solvers
 */

#include "src/option/american_option_batch.hpp"
#include "common/ivcalc_trace.h"
#include <cmath>
#include <algorithm>
#include <ranges>

namespace mango {

// ============================================================================
// Normalized Solver Implementations
// ============================================================================

std::expected<void, std::string> NormalizedSolveRequest::validate() const {
    if (sigma <= 0.0) {
        return std::unexpected("Volatility must be positive");
    }
    // Note: rate can be negative (EUR, JPY markets)
    if (dividend < 0.0) {
        return std::unexpected("Dividend yield must be non-negative");
    }
    if (x_min >= x_max) {
        return std::unexpected("x_min must be < x_max");
    }
    if (n_space < 3) {
        return std::unexpected("n_space must be ≥ 3");
    }
    if (n_time < 1) {
        return std::unexpected("n_time must be ≥ 1");
    }
    if (T_max <= 0.0) {
        return std::unexpected("T_max must be positive");
    }
    if (tau_snapshots.empty()) {
        return std::unexpected("tau_snapshots must be non-empty");
    }

    // Validate snapshot times
    for (double tau : tau_snapshots) {
        if (tau <= 0.0 || tau > T_max) {
            return std::unexpected("Snapshot times must be in (0, T_max]");
        }
    }

    return {};
}

std::expected<NormalizedWorkspace, std::string> NormalizedWorkspace::create(
    const NormalizedSolveRequest& request)
{
    // Validate request
    auto validation = request.validate();
    if (!validation) {
        return std::unexpected(validation.error());
    }

    NormalizedWorkspace workspace;

    // Create PDE workspace
    auto grid_spec_result = GridSpec<double>::uniform(request.x_min, request.x_max, request.n_space);
    if (!grid_spec_result.has_value()) {
        return std::unexpected("Invalid grid specification: " + grid_spec_result.error());
    }

    auto pde_workspace = AmericanSolverWorkspace::create(
        grid_spec_result.value(), request.n_time, std::pmr::get_default_resource());
    if (!pde_workspace) {
        return std::unexpected("Failed to create PDE workspace: " + pde_workspace.error());
    }
    workspace.pde_workspace_ = std::move(pde_workspace.value());

    // Allocate x grid
    workspace.x_grid_.resize(request.n_space);
    double dx = (request.x_max - request.x_min) / (request.n_space - 1);
    for (size_t i = 0; i < request.n_space; ++i) {
        workspace.x_grid_[i] = request.x_min + i * dx;
    }

    // Allocate tau grid (copy from request)
    workspace.tau_grid_.assign(request.tau_snapshots.begin(), request.tau_snapshots.end());
    std::sort(workspace.tau_grid_.begin(), workspace.tau_grid_.end());

    // Allocate values array (Nx × Ntau)
    workspace.values_.resize(request.n_space * workspace.tau_grid_.size(), 0.0);

    return workspace;
}

NormalizedSurfaceView NormalizedWorkspace::surface_view() {
    return NormalizedSurfaceView(x_grid_, tau_grid_, values_);
}

std::optional<std::string> NormalizedSurfaceView::build_cache() {
    // Build 2D cubic spline from the surface data
    // Note: CubicSpline2D expects row-major layout: z[i*ny + j] = z(x[i], y[j])
    // Our layout: values[i*Ntau + j] = u(x[i], τ[j])
    return spline2d_.build(x_grid_, tau_grid_, values_);
}

double NormalizedSurfaceView::interpolate(double x, double tau) const {
    // Ensure cache is built
    if (!spline2d_.is_built()) {
        // Auto-build cache on first call (for convenience)
        // Note: This modifies mutable state, safe for const method
        auto error = const_cast<NormalizedSurfaceView*>(this)->build_cache();
        if (error.has_value()) {
            // Log cache build failure via USDT tracing
            MANGO_TRACE_RUNTIME_ERROR(MODULE_NORMALIZED_CHAIN,
                                     static_cast<int>(x_grid_.size()),
                                     static_cast<int>(tau_grid_.size()));
            // Fallback to boundary values if cache build fails
            return 0.0;
        }
    }

    // Use 2D cubic spline interpolation
    return spline2d_.eval(x, tau);
}

std::expected<void, SolverError> NormalizedChainSolver::solve(
    const NormalizedSolveRequest& request,
    NormalizedWorkspace& workspace,
    NormalizedSurfaceView& surface_view)
{
    // Create solver parameters (K=1, S=1 → x = ln(S/K) = 0 is ATM)
    AmericanOptionParams params;
    params.spot = 1.0;    // Normalized spot (ATM at x=0)
    params.strike = 1.0;  // Normalized strike
    params.maturity = request.T_max;
    params.rate = request.rate;
    params.dividend_yield = request.dividend;
    params.type = request.option_type;
    params.volatility = request.sigma;
    params.discrete_dividends = {};  // Normalized solver requires no discrete dividends

    // Create solver (surface is always collected for at_time() access)
    auto solver_result = AmericanOptionSolver::create(params, workspace.pde_workspace_);
    if (!solver_result) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = "Failed to create solver: " + solver_result.error(),
            .iterations = 0
        });
    }
    auto solver = std::move(solver_result.value());

    // Precompute step indices for each maturity
    double dt = request.T_max / request.n_time;
    std::vector<size_t> step_indices(workspace.tau_grid_.size());
    for (size_t j = 0; j < workspace.tau_grid_.size(); ++j) {
        // Compute step index: k = round(τ/dt) - 1
        double step_exact = workspace.tau_grid_[j] / dt - 1.0;
        long long step_rounded = std::llround(step_exact);

        // Clamp to valid range
        if (step_rounded < 0) {
            step_indices[j] = 0;
        } else if (step_rounded >= static_cast<long long>(request.n_time)) {
            step_indices[j] = request.n_time - 1;
        } else {
            step_indices[j] = static_cast<size_t>(step_rounded);
        }
    }

    // Solve PDE (surface collected to surface_buffer)
    auto solve_result = solver.solve();
    if (!solve_result) {
        return std::unexpected(solve_result.error());
    }

    // Extract solution from result
    const auto& result = solve_result.value();
    size_t Nx = workspace.x_grid_.size();
    size_t Ntau = workspace.tau_grid_.size();

    // Copy values from surface_2d to workspace (shape: Nx × Ntau)
    for (size_t j = 0; j < Ntau; ++j) {
        size_t step_idx = step_indices[j];
        std::span<const double> spatial_solution = result.at_time(step_idx);

        if (spatial_solution.empty() || spatial_solution.size() != Nx) {
            return std::unexpected(SolverError{
                .code = SolverErrorCode::InvalidState,
                .message = "Surface_2d returned wrong size for time step",
                .iterations = 0
            });
        }

        // Copy this time step's spatial solution to workspace
        // Workspace layout: values_[i * Ntau + j] = V(x_i, tau_j)
        for (size_t i = 0; i < Nx; ++i) {
            workspace.values_[i * Ntau + j] = spatial_solution[i];
        }
    }

    // Update surface view to reference workspace values
    surface_view = workspace.surface_view();

    // Build interpolation cache (x-direction cubic splines)
    auto cache_error = surface_view.build_cache();
    if (cache_error.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidState,
            .message = "Failed to build interpolation cache: " + cache_error.value(),
            .iterations = 0
        });
    }

    return {};
}

std::expected<void, std::string> NormalizedChainSolver::check_eligibility(
    const NormalizedSolveRequest& request,
    std::span<const double> moneyness_grid)
{
    // Check grid spacing
    double dx = (request.x_max - request.x_min) / (request.n_space - 1);
    if (dx > EligibilityLimits::MAX_DX) {
        return std::unexpected(
            "Grid spacing " + std::to_string(dx) +
            " exceeds limit " + std::to_string(EligibilityLimits::MAX_DX) +
            " (Von Neumann stability requirement)");
    }

    // Check domain width
    double width = request.x_max - request.x_min;
    if (width > EligibilityLimits::MAX_WIDTH) {
        return std::unexpected(
            "Domain width " + std::to_string(width) +
            " exceeds limit " + std::to_string(EligibilityLimits::MAX_WIDTH) +
            " (convergence degrades beyond 5.8 log-units)");
    }

    // Check margins
    // Moneyness convention: m = S/K, so x = ln(S/K) = ln(m)
    // x_min_data = ln(m_min), x_max_data = ln(m_max)
    if (moneyness_grid.empty()) {
        return std::unexpected("Moneyness grid is empty");
    }

    auto [m_min_it, m_max_it] = std::ranges::minmax_element(moneyness_grid);
    double m_min = *m_min_it;
    double m_max = *m_max_it;

    if (m_min <= 0.0 || m_max <= 0.0) {
        return std::unexpected("Moneyness values must be positive (m = S/K > 0)");
    }

    double x_min_data = std::log(m_min);
    double x_max_data = std::log(m_max);

    double margin_left = x_min_data - request.x_min;
    double margin_right = request.x_max - x_max_data;
    double min_margin = EligibilityLimits::min_margin(dx);

    if (margin_left < min_margin) {
        return std::unexpected(
            "Left margin " + std::to_string(margin_left) +
            " < required " + std::to_string(min_margin) +
            " (need ≥6 ghost cells to avoid boundary reflection)");
    }

    if (margin_right < min_margin) {
        return std::unexpected(
            "Right margin " + std::to_string(margin_right) +
            " < required " + std::to_string(min_margin) +
            " (need ≥6 ghost cells to avoid boundary reflection)");
    }

    // Check ratio (derived from width + margin constraints)
    double ratio = m_max / m_min;
    double max_ratio_limit = EligibilityLimits::max_ratio(dx);
    if (ratio > max_ratio_limit) {
        return std::unexpected(
            "Moneyness ratio " + std::to_string(ratio) +
            " exceeds limit " + std::to_string(max_ratio_limit) +
            " (derived from width=" + std::to_string(EligibilityLimits::MAX_WIDTH) +
            " and margin=" + std::to_string(min_margin) + ")");
    }

    return {};
}

}  // namespace mango
