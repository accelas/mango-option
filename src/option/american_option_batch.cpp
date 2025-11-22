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
    const NormalizedSolveRequest& request,
    std::span<double> pde_buffer,
    [[maybe_unused]] std::pmr::memory_resource* resource)
{
    // Validate request
    auto validation = request.validate();
    if (!validation) {
        return std::unexpected(validation.error());
    }

    NormalizedWorkspace workspace;

    // Create grid specification
    auto grid_spec_result = GridSpec<double>::uniform(request.x_min, request.x_max, request.n_space);
    if (!grid_spec_result.has_value()) {
        return std::unexpected("Invalid grid specification: " + grid_spec_result.error());
    }

    // Create Grid with solution storage
    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, request.T_max, request.n_time);
    auto grid_result = Grid<double>::create(grid_spec_result.value(), time_domain);
    if (!grid_result.has_value()) {
        return std::unexpected("Failed to create Grid: " + grid_result.error());
    }
    workspace.grid_ = grid_result.value();

    // Create workspace spans from caller-provided buffer
    size_t n = request.n_space;
    size_t required = PDEWorkspace::required_size(n);
    if (pde_buffer.size() < required) {
        return std::unexpected(std::format(
            "PDEWorkspace buffer too small: {} < {} required",
            pde_buffer.size(), required));
    }

    auto pde_workspace_result = PDEWorkspace::from_buffer(pde_buffer, n);
    if (!pde_workspace_result.has_value()) {
        return std::unexpected("Failed to create PDEWorkspace: " + pde_workspace_result.error());
    }
    workspace.pde_workspace_ = pde_workspace_result.value();

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

    // Create solver using PDEWorkspace API
    AmericanOptionSolver solver(params, workspace.workspace());

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

    // Solve PDE
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

bool BatchAmericanOptionSolver::is_normalized_eligible(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid) const
{
    // 1. Requires shared grid mode
    if (!use_shared_grid) {
        return false;
    }

    if (params.empty()) {
        return false;
    }

    const auto& first = params[0];

    // 2. All options must have consistent option type
    for (size_t i = 1; i < params.size(); ++i) {
        if (params[i].type != first.type) {
            return false;
        }
    }

    // 3. All options must have same maturity
    for (size_t i = 1; i < params.size(); ++i) {
        if (std::abs(params[i].maturity - first.maturity) > 1e-10) {
            return false;
        }
    }

    // 4. No discrete dividends
    for (const auto& p : params) {
        if (!p.discrete_dividends.empty()) {
            return false;
        }
    }

    // 5. Validate spot and strike are positive
    for (const auto& p : params) {
        if (p.spot <= 0.0 || p.strike <= 0.0) {
            return false;
        }
    }

    // 6. Grid constraints (dx, width, margins)
    auto [grid_spec, n_time] = estimate_grid_for_option(first, grid_accuracy_);
    double x_min = grid_spec.x_min();
    double x_max = grid_spec.x_max();
    size_t n_space = grid_spec.n_points();

    // Check grid spacing (Von Neumann stability)
    double dx = (x_max - x_min) / (n_space - 1);
    if (dx > MAX_DX) {
        return false;
    }

    // Check domain width (convergence constraint)
    double width = x_max - x_min;
    if (width > MAX_WIDTH) {
        return false;
    }

    // Check margins based on moneyness range
    std::vector<double> moneyness_values;
    moneyness_values.reserve(params.size());
    for (const auto& p : params) {
        double m = p.spot / p.strike;
        moneyness_values.push_back(m);
    }

    auto [m_min_it, m_max_it] = std::ranges::minmax_element(moneyness_values);
    double m_min = *m_min_it;
    double m_max = *m_max_it;

    double x_min_data = std::log(m_min);
    double x_max_data = std::log(m_max);

    double margin_left = x_min_data - x_min;
    double margin_right = x_max - x_max_data;
    double min_margin = std::max(MIN_MARGIN_ABS, 6.0 * dx);

    if (margin_left < min_margin || margin_right < min_margin) {
        return false;
    }

    return true;
}

void BatchAmericanOptionSolver::trace_ineligibility_reason(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid) const
{
    // Check forced disable first
    if (!use_normalized_) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::FORCED_DISABLE), 0);
        return;
    }

    // Check shared grid requirement
    if (!use_shared_grid) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::SHARED_GRID_DISABLED), 0);
        return;
    }

    // Check empty batch
    if (params.empty()) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::EMPTY_BATCH), 0);
        return;
    }

    const auto& first = params[0];

    // Check option type consistency
    for (size_t i = 1; i < params.size(); ++i) {
        if (params[i].type != first.type) {
            MANGO_TRACE_NORMALIZED_INELIGIBLE(
                static_cast<int>(NormalizedIneligibilityReason::MISMATCHED_OPTION_TYPE),
                static_cast<int>(params[i].type));
            return;
        }
    }

    // Check maturity consistency
    for (size_t i = 1; i < params.size(); ++i) {
        if (std::abs(params[i].maturity - first.maturity) > 1e-10) {
            MANGO_TRACE_NORMALIZED_INELIGIBLE(
                static_cast<int>(NormalizedIneligibilityReason::MISMATCHED_MATURITY),
                params[i].maturity);
            return;
        }
    }

    // Check discrete dividends
    for (const auto& p : params) {
        if (!p.discrete_dividends.empty()) {
            MANGO_TRACE_NORMALIZED_INELIGIBLE(
                static_cast<int>(NormalizedIneligibilityReason::DISCRETE_DIVIDENDS),
                p.discrete_dividends.size());
            return;
        }
    }

    // Check spot and strike validity
    for (const auto& p : params) {
        if (p.spot <= 0.0 || p.strike <= 0.0) {
            MANGO_TRACE_NORMALIZED_INELIGIBLE(
                static_cast<int>(NormalizedIneligibilityReason::INVALID_SPOT_OR_STRIKE),
                p.spot <= 0.0 ? p.spot : p.strike);
            return;
        }
    }

    // Check grid constraints
    auto [grid_spec, n_time] = estimate_grid_for_option(first, grid_accuracy_);
    double x_min = grid_spec.x_min();
    double x_max = grid_spec.x_max();
    size_t n_space = grid_spec.n_points();

    // Check grid spacing (Von Neumann stability)
    double dx = (x_max - x_min) / (n_space - 1);
    if (dx > MAX_DX) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::GRID_SPACING_TOO_LARGE), dx);
        return;
    }

    // Check domain width (convergence constraint)
    double width = x_max - x_min;
    if (width > MAX_WIDTH) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::DOMAIN_TOO_WIDE), width);
        return;
    }

    // Check margins based on moneyness range
    std::vector<double> moneyness_values;
    moneyness_values.reserve(params.size());
    for (const auto& p : params) {
        moneyness_values.push_back(p.spot / p.strike);
    }

    auto [m_min_it, m_max_it] = std::ranges::minmax_element(moneyness_values);
    double x_min_data = std::log(*m_min_it);
    double x_max_data = std::log(*m_max_it);

    double margin_left = x_min_data - x_min;
    double margin_right = x_max - x_max_data;
    double min_margin = std::max(MIN_MARGIN_ABS, 6.0 * dx);

    if (margin_left < min_margin) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::INSUFFICIENT_LEFT_MARGIN),
            margin_left);
        return;
    }

    if (margin_right < min_margin) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::INSUFFICIENT_RIGHT_MARGIN),
            margin_right);
        return;
    }
}

}  // namespace mango
