/**
 * @file american_option_batch.cpp
 * @brief Implementation of batch and normalized chain solvers
 */

#include "src/option/american_option_batch.hpp"
#include "src/support/ivcalc_trace.h"
#include "src/support/thread_workspace.hpp"
#include "src/pde/core/american_pde_workspace.hpp"
#include <cmath>
#include <algorithm>
#include <ranges>

namespace mango {

/// Group of options sharing same PDE parameters
struct PDEParameterGroup {
    double sigma;
    RateSpec rate;
    double dividend;
    OptionType option_type;
    double maturity;
    std::vector<size_t> option_indices;  ///< Indices into original params array
};

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
    auto [grid_spec, time_domain] = estimate_grid_for_option(first, grid_accuracy_);
    (void)time_domain;  // Not used in eligibility check
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
    auto [grid_spec, time_domain] = estimate_grid_for_option(first, grid_accuracy_);
    (void)time_domain;  // Not used in trace function
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

std::vector<PDEParameterGroup> BatchAmericanOptionSolver::group_by_pde_parameters(
    std::span<const AmericanOptionParams> params) const
{
    std::vector<PDEParameterGroup> groups;
    constexpr double TOL = 1e-10;

    for (size_t i = 0; i < params.size(); ++i) {
        const auto& p = params[i];

        // Find existing group with matching PDE parameters
        bool found = false;
        for (auto& group : groups) {
            // Compare rates: must be same variant type and value
            bool rate_match = false;
            if (group.rate.index() == p.rate.index()) {
                if (std::holds_alternative<double>(group.rate)) {
                    rate_match = std::abs(std::get<double>(group.rate) - std::get<double>(p.rate)) < TOL;
                } else {
                    // For yield curves, compare by value (tenor/discount vectors)
                    // This allows batching options that share the same curve data
                    rate_match = (std::get<YieldCurve>(group.rate) == std::get<YieldCurve>(p.rate));
                }
            }

            if (std::abs(group.sigma - p.volatility) < TOL &&
                rate_match &&
                std::abs(group.dividend - p.dividend_yield) < TOL &&
                std::abs(group.maturity - p.maturity) < TOL &&
                group.option_type == p.type)
            {
                group.option_indices.push_back(i);
                found = true;
                break;
            }
        }

        if (!found) {
            // Create new group
            PDEParameterGroup new_group{
                .sigma = p.volatility,
                .rate = p.rate,
                .dividend = p.dividend_yield,
                .option_type = p.type,
                .maturity = p.maturity,
                .option_indices = {i}
            };
            groups.push_back(new_group);
        }
    }

    return groups;
}

BatchAmericanOptionResult BatchAmericanOptionSolver::solve_batch(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid,
    SetupCallback setup,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid)
{
    // Ensure grid_accuracy_ is initialized
    if (grid_accuracy_.tol == 0.0) {
        grid_accuracy_ = GridAccuracyParams{};
    }

    // Disable normalized path if setup callback is provided
    // Reason: normalized solver creates one PDE for a group, but callback expects
    // to be invoked for each original option index. Supporting this would require
    // creating temporary solvers for all options or tracking index mapping.
    // For simplicity, fall back to regular batch when callback is needed.
    if (setup) {
        return solve_regular_batch(params, use_shared_grid, setup, custom_grid);
    }

    // Automatic routing based on eligibility
    if (use_normalized_ && is_normalized_eligible(params, use_shared_grid)) {
        MANGO_TRACE_NORMALIZED_SELECTED(params.size());
        return solve_normalized_chain(params, setup, custom_grid);
    } else {
        if (use_normalized_ && !is_normalized_eligible(params, use_shared_grid)) {
            trace_ineligibility_reason(params, use_shared_grid);
        }
        return solve_regular_batch(params, use_shared_grid, setup, custom_grid);
    }
}

BatchAmericanOptionResult BatchAmericanOptionSolver::solve_regular_batch(
    const std::vector<AmericanOptionParams>& params,
    bool use_shared_grid,
    SetupCallback setup,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid)
{
    return solve_regular_batch(std::span{params}, use_shared_grid, setup, custom_grid);
}

BatchAmericanOptionResult BatchAmericanOptionSolver::solve_normalized_chain(
    std::span<const AmericanOptionParams> params,
    SetupCallback setup,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid)
{
    // Group by PDE parameters: (σ, r, q, type, maturity)
    auto pde_groups = group_by_pde_parameters(params);

    // Process each PDE parameter group
    std::vector<std::expected<AmericanOptionResult, SolverError>> results;
    results.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        results.emplace_back(std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            // error code set above,
            .iterations = 0
        }));
    }
    size_t failed_count = 0;

    for (const auto& group : pde_groups) {
        // Solve normalized PDE once for this group (S=K=1)
        AmericanOptionParams normalized_params(
            1.0,                // spot (normalized)
            1.0,                // strike (normalized)
            group.maturity,     // maturity
            group.rate,         // rate (RateSpec)
            group.dividend,     // dividend_yield
            group.option_type,  // type
            group.sigma         // volatility
        );

        // Solve with shared grid to get full surface
        auto solve_result = solve_regular_batch(
            std::span{&normalized_params, 1},
            /*use_shared_grid=*/true,
            setup,
            custom_grid);

        if (!solve_result.results[0].has_value()) {
            // Mark all options in this group as failed
            for (size_t idx : group.option_indices) {
                results[idx] = std::unexpected(solve_result.results[0].error());
                ++failed_count;
            }
            continue;
        }

        // Extract normalized surface u(x,τ)
        const auto& normalized_result = solve_result.results[0].value();
        auto grid = normalized_result.grid();

        // For each option in this group, create result with actual params
        // The AmericanOptionResult will interpolate using the normalized grid
        // (uses linear interpolation internally)
        for (size_t idx : group.option_indices) {
            const auto& option = params[idx];

            // Create result wrapper with actual option params
            // Grid still contains normalized solution (S=K=1)
            // AmericanOptionResult::value_at() will handle interpolation
            AmericanOptionResult scaled_result(grid, option);

            // Move result into results vector (using placement new)
            results[idx].~expected();
            new (&results[idx]) std::expected<AmericanOptionResult, SolverError>(
                std::move(scaled_result));
        }
    }

    return BatchAmericanOptionResult{
        .results = std::move(results),
        .failed_count = failed_count
    };
}

BatchAmericanOptionResult BatchAmericanOptionSolver::solve_regular_batch(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid,
    SetupCallback setup,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid)
{
    if (params.empty()) {
        return BatchAmericanOptionResult{.results = {}, .failed_count = 0};
    }

    // Pre-allocate results vector with sentinel errors (parallel access requires pre-sized vector)
    // Since AmericanOptionResult is not copyable, we construct each element in-place
    std::vector<std::expected<AmericanOptionResult, SolverError>> results;
    results.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        results.emplace_back(std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            // error code set above,
            .iterations = 0
        }));
    }
    size_t failed_count = 0;

    // Precompute shared grid if needed
    std::optional<std::pair<GridSpec<double>, TimeDomain>> shared_grid;
    if (use_shared_grid) {
        if (custom_grid.has_value()) {
            // Use provided grid directly (bypass auto-estimation)
            shared_grid = custom_grid;
        } else {
            // Existing path: use grid_accuracy_ member to estimate grid
            shared_grid = compute_global_grid_for_batch(params, grid_accuracy_);
        }
    }

    // Precompute workspace size outside parallel region
    size_t workspace_size_bytes = 0;
    size_t shared_n_space = 0;

    if (use_shared_grid) {
        auto [grid_spec, time_domain] = shared_grid.value();
        shared_n_space = grid_spec.n_points();
        workspace_size_bytes = AmericanPDEWorkspace::required_bytes(shared_n_space);
    } else {
        // For per-option grids, estimate max workspace size across all options
        if (custom_grid.has_value()) {
            // Use custom grid size
            auto [grid_spec, time_domain] = custom_grid.value();
            size_t n = grid_spec.n_points();
            workspace_size_bytes = AmericanPDEWorkspace::required_bytes(n);
        } else {
            // Estimate based on all options
            for (const auto& p : params) {
                auto [grid_spec, time_domain] = estimate_grid_for_option(p, grid_accuracy_);
                size_t n = grid_spec.n_points();
                workspace_size_bytes = std::max(workspace_size_bytes, AmericanPDEWorkspace::required_bytes(n));
            }
        }
    }

    MANGO_PRAGMA_PARALLEL
    {
        // Per-thread workspace buffer (64-byte aligned)
        ThreadWorkspaceBuffer buffer(workspace_size_bytes);

        // Per-thread shared grid (only for shared grid strategy)
        std::shared_ptr<Grid<double>> thread_grid;

        if (use_shared_grid) {
            auto [grid_spec, time_domain] = shared_grid.value();

            // Create Grid with solution storage
            auto grid_result = Grid<double>::create(grid_spec, time_domain);
            if (grid_result.has_value()) {
                thread_grid = grid_result.value();
            }
            // If creation failed, thread_grid remains null and we'll fail in loop
        }

        // Use static scheduling to avoid false sharing on results vector
        // Each thread gets a contiguous block of iterations
        MANGO_PRAGMA_FOR_STATIC
        for (size_t i = 0; i < params.size(); ++i) {
            // Get or create grid and workspace for this iteration
            std::shared_ptr<Grid<double>> grid;
            PDEWorkspace* workspace_ptr = nullptr;
            std::optional<AmericanPDEWorkspace> workspace_storage;
            std::vector<std::byte> heap_buffer_storage;  // Heap fallback buffer (persists for iteration)

            // Track the grid config used for this solve (to pass to solver)
            std::optional<std::pair<GridSpec<double>, TimeDomain>> solver_grid_config;

            if (use_shared_grid) {
                // Shared grid: reuse thread grid and buffer
                grid = thread_grid;
                solver_grid_config = shared_grid;  // Use shared grid config
                if (grid) {
                    auto ws_result = AmericanPDEWorkspace::from_bytes(buffer.bytes(), shared_n_space);
                    if (ws_result.has_value()) {
                        workspace_storage = std::move(ws_result.value());
                        workspace_ptr = &workspace_storage->workspace();
                    }
                }
            } else {
                // Per-option grid: create workspace for this option
                auto [grid_spec, time_domain] = custom_grid.has_value()
                    ? custom_grid.value()
                    : estimate_grid_for_option(params[i], grid_accuracy_);

                solver_grid_config = std::make_pair(grid_spec, time_domain);

                // Create Grid with solution storage
                auto grid_result = Grid<double>::create(grid_spec, time_domain);
                if (grid_result.has_value()) {
                    grid = grid_result.value();

                    // Create workspace from buffer (zero allocations)
                    size_t n = grid_spec.n_points();
                    auto ws_result = AmericanPDEWorkspace::from_bytes(buffer.bytes(), n);
                    if (ws_result.has_value()) {
                        workspace_storage = std::move(ws_result.value());
                        workspace_ptr = &workspace_storage->workspace();
                    }
                }
            }

            // Fallback to heap if buffer allocation failed
            if (!grid || !workspace_ptr) {
                // Try allocating from default resource (heap) as fallback
                // Determine grid to use
                auto [grid_spec, time_domain] = use_shared_grid && shared_grid.has_value()
                    ? shared_grid.value()
                    : (custom_grid.has_value()
                        ? custom_grid.value()
                        : estimate_grid_for_option(params[i], grid_accuracy_));

                // Update solver_grid_config for fallback path
                solver_grid_config = std::make_pair(grid_spec, time_domain);

                auto grid_result = Grid<double>::create(grid_spec, time_domain);
                if (grid_result.has_value()) {
                    grid = grid_result.value();

                    size_t n = grid_spec.n_points();
                    // Allocate heap buffer that persists for the entire iteration
                    heap_buffer_storage.resize(AmericanPDEWorkspace::required_bytes(n));
                    auto ws_result = AmericanPDEWorkspace::from_bytes(heap_buffer_storage, n);
                    if (ws_result.has_value()) {
                        workspace_storage = std::move(ws_result.value());
                        workspace_ptr = &workspace_storage->workspace();
                    }
                }

                // If still failed after heap fallback, report error
                if (!grid || !workspace_ptr) {
                    results[i] = std::unexpected(SolverError{
                        .code = SolverErrorCode::InvalidConfiguration,
                        // error code set above,
                        .iterations = 0
                    });
                    MANGO_PRAGMA_ATOMIC
                    ++failed_count;
                    continue;
                }
            }

            // Create solver using PDEWorkspace API with explicit grid config
            // This ensures workspace size matches the grid that will be used
            AmericanOptionSolver solver(params[i], *workspace_ptr, std::nullopt, solver_grid_config);

            // Register snapshot times if configured (preserves normalized optimization)
            if (!snapshot_times_.empty()) {
                solver.set_snapshot_times(std::span{snapshot_times_});
            }

            // Invoke setup callback if provided
            if (setup) {
                setup(i, solver);
            }

            // Solve (use placement new to avoid copy/move assignment issues)
            auto solve_result = solver.solve();
            results[i].~expected();  // Destroy sentinel value
            new (&results[i]) std::expected<AmericanOptionResult, SolverError>(std::move(solve_result));

            if (!results[i].has_value()) {
                MANGO_PRAGMA_ATOMIC
                ++failed_count;
            }

            // ThreadWorkspaceBuffer handles memory automatically - no release() needed
        }
    }

    return BatchAmericanOptionResult{
        .results = std::move(results),
        .failed_count = failed_count
    };
}

}  // namespace mango
