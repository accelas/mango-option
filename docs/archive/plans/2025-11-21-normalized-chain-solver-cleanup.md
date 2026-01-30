<!-- SPDX-License-Identifier: MIT -->
# Normalized Chain Solver Cleanup

**Date:** 2025-11-21
**Status:** Design
**Authors:** Claude Code + User

## Executive Summary

This document describes the refactoring of the normalized chain solver from a standalone public API into an internal optimization within `BatchAmericanOptionSolver`. This simplification eliminates API duplication, removes unnecessary types, and makes the optimization transparent to users.

## Problem Statement

The current normalized chain solver implementation has two main issues:

### 1. NormalizedWorkspace API Inconsistency

- Hybrid storage + API object that owns data (`x_grid_`, `tau_grid_`, `values_`)
- Owns a `PDEWorkspace` internally
- Not aligned with codebase patterns:
  - `PDEWorkspace` uses spans (caller owns memory)
  - `AmericanOptionSolver` is lightweight (delegates to workspace)
- Duplicates memory management logic already in batch solver
- Adds unnecessary complexity for what should be a simple optimization

### 2. check_eligibility Uses Fake Request

```cpp
// Current implementation (ugly)
NormalizedSolveRequest test_request{
    .sigma = 0.20,  // Test value (ignored)
    .rate = 0.05,   // Test value (ignored)
    .dividend = config.dividend_yield,
    .option_type = config.option_type,
    .x_min = config.x_min,
    .x_max = config.x_max,
    .n_space = config.n_space,
    .n_time = config.n_time,
    .T_max = 1.0,  // Test value (ignored)
    .tau_snapshots = std::span<const double>{}  // Placeholder
};

auto eligibility = NormalizedChainSolver::check_eligibility(
    test_request, moneyness);
```

- Creates dummy `NormalizedSolveRequest` with placeholder PDE parameters
- Only checks geometric properties (dx, width, margins)
- Confusing API - looks like it needs PDE params but actually ignores them
- Forces caller to know about normalized solver to check eligibility

## Solution: Internal Batch Solver Optimization

Make normalized chain solving a **transparent internal optimization** inside `BatchAmericanOptionSolver`:

- Callers don't know about normalized solving - they just call `solve_batch()`
- Batch solver automatically routes to fast path when params are eligible
- No new types, no parallel APIs, no factory pattern needed
- Graceful fallback to regular batch solving when ineligible

**Key insight:** Normalized chain solving is just an optimization technique. Users care about solving batches of options efficiently, not about which algorithm is used internally.

## Architecture

### Public API (Unchanged)

```cpp
// Callers use this - same as before
BatchAmericanOptionSolver solver;
auto result = solver.solve_batch(params, use_shared_grid);

// New: Optional configuration to disable optimization
solver.set_use_normalized(false);  // For benchmarking/debugging
```

**PriceTableSolver simplifies to single call:**

```cpp
class PriceTableSolver {
public:
    explicit PriceTableSolver(const OptionSolverGrid& config)
        : config_(config) {}

    std::expected<void, std::string> solve(
        std::span<double> prices_4d,
        const PriceTableGrid& grid)
    {
        // Build batch params
        auto params = build_batch_params(grid, config_);

        // Batch solver handles routing internally
        BatchAmericanOptionSolver solver;
        solver.set_grid_accuracy(config_.grid_accuracy);
        auto result = solver.solve_batch(params, /*use_shared_grid=*/true);

        if (result.failed_count > 0) {
            return std::unexpected("Failed " + std::to_string(result.failed_count) + " solves");
        }

        // Extract (same for both paths)
        extract_batch_results_to_4d(result, prices_4d, grid, grid.K_ref);
        return {};
    }

private:
    OptionSolverGrid config_;
};
```

### Internal Implementation

```cpp
class BatchAmericanOptionSolver {
public:
    /// Disable normalized chain optimization (for benchmarking/debugging)
    void set_use_normalized(bool enable) {
        use_normalized_ = enable;
    }

    bool use_normalized() const {
        return use_normalized_;
    }

    BatchAmericanOptionResult solve_batch(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr)
    {
        // Ensure grid_accuracy_ is initialized with default if not set
        // (Needed for eligibility check which estimates grid configuration)
        if (grid_accuracy_.tol == 0.0) {
            grid_accuracy_ = GridAccuracyParams{};  // Use defaults
        }

        // Automatic routing based on eligibility
        if (use_normalized_ && is_normalized_eligible(params, use_shared_grid)) {
            MANGO_TRACE_NORMALIZED_SELECTED(params.size());
            return solve_normalized_chain(params, setup);
        } else {
            if (use_normalized_ && !is_normalized_eligible(params, use_shared_grid)) {
                trace_ineligibility_reason(params, use_shared_grid);
            }
            return solve_regular_batch(params, use_shared_grid, setup);
        }
    }

private:
    bool use_normalized_ = true;  ///< Enable normalized chain optimization

    // Check if batch qualifies for normalized solving
    bool is_normalized_eligible(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid) const;

    // Trace why normalized path wasn't used
    void trace_ineligibility_reason(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid) const;

    // Fast path: normalized chain solving with bracketing
    BatchAmericanOptionResult solve_normalized_chain(
        std::span<const AmericanOptionParams> params,
        SetupCallback setup);

    // Fallback: regular batch solving
    BatchAmericanOptionResult solve_regular_batch(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid,
        SetupCallback setup);

    // Eligibility constants
    static constexpr double MAX_WIDTH = 5.8;       ///< Convergence limit (log-units)
    static constexpr double MAX_DX = 0.05;         ///< Von Neumann stability
    static constexpr double MIN_MARGIN_ABS = 0.35; ///< 6-cell ghost zone minimum
};
```

## Eligibility Criteria

```cpp
bool BatchAmericanOptionSolver::is_normalized_eligible(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid) const
{
    // 1. Requires shared grid mode (normalized solver always uses shared grid)
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

    // 3. All options must have same maturity (for surface extraction)
    for (size_t i = 1; i < params.size(); ++i) {
        if (std::abs(params[i].maturity - first.maturity) > 1e-10) {
            return false;
        }
    }

    // 4. No discrete dividends (normalized solver doesn't support them)
    for (const auto& p : params) {
        if (!p.discrete_dividends.empty()) {
            return false;
        }
    }

    // 5. Validate spot and strike are positive (for log-moneyness)
    for (const auto& p : params) {
        if (p.spot <= 0.0 || p.strike <= 0.0) {
            return false;
        }
    }

    // 6. Grid constraints (dx, width, margins)
    // Use grid_accuracy_ to estimate grid configuration
    auto [grid_spec, time_domain] = estimate_grid_for_option(first, grid_accuracy_);
    double x_min = grid_spec.x_min();
    double x_max = grid_spec.x_max();
    size_t n_space = grid_spec.n_points();

    // Check grid spacing
    double dx = (x_max - x_min) / (n_space - 1);
    if (dx > MAX_DX) {
        return false;
    }

    // Check domain width
    double width = x_max - x_min;
    if (width > MAX_WIDTH) {
        return false;
    }

    // Check margins based on moneyness range (m = S/K)
    std::vector<double> moneyness_values;
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
```

**Eligibility requirements:**

1. **Shared grid mode enabled** - normalized solver always uses shared grid
2. **Consistent option type** - all calls or all puts (different payoffs/boundaries)
3. **Same maturity** - all options must have same expiry (for surface extraction)
4. **Varying strikes allowed** - this is the key use case! Multiple strikes with same maturity
5. **No discrete dividends** - normalized solver doesn't support them yet
6. **Positive spot/strike** - required for log-moneyness x = ln(S/K)
7. **Grid spacing** - `dx ≤ 0.05` for Von Neumann stability
8. **Domain width** - `width ≤ 5.8` for convergence
9. **Sufficient margins** - `≥ 6·dx` ghost cells to avoid boundary reflection based on moneyness range

## Normalized Chain Implementation

### Core Algorithm: Scale-Invariant PDE Solving

The normalized chain solver exploits the mathematical property that:
```
V(S, K, τ; σ, r, q) = K · u(ln(S/K), τ; σ, r, q)
```

where `u` is the solution to the dimensionless PDE with normalized coordinates (S=K=1).

**Key insight:** For options with the same (σ, r, q, type, maturity), we solve the PDE **once** with S=K=1, then scale and interpolate for all (S, K) combinations.

### Implementation

```cpp
BatchAmericanOptionResult BatchAmericanOptionSolver::solve_normalized_chain(
    std::span<const AmericanOptionParams> params,
    SetupCallback setup)
{
    // Group by PDE parameters: (σ, r, q, type, maturity)
    // Options in the same group share the same normalized surface u(x,τ)
    auto pde_groups = group_by_pde_parameters(params);

    // Process each PDE parameter group
    std::vector<std::expected<AmericanOptionResult, SolverError>> results;
    results.resize(params.size());
    size_t failed_count = 0;

    for (const auto& group : pde_groups) {
        // Solve normalized PDE once for this group
        // Use S=K=1 to get dimensionless solution u(x,τ)
        AmericanOptionParams normalized_params{
            .spot = 1.0,              // Normalized spot
            .strike = 1.0,            // Normalized strike
            .maturity = group.maturity,
            .volatility = group.sigma,
            .rate = group.rate,
            .dividend_yield = group.dividend,
            .type = group.option_type,
            .discrete_dividends = {}
        };

        // Solve with shared grid to get full surface
        auto solve_result = solve_regular_batch(
            std::span{&normalized_params, 1},
            /*use_shared_grid=*/true,
            setup);

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
        auto x_grid = grid->x();  // Spatial grid in log-moneyness

        // For each option in this group, interpolate and scale
        for (size_t idx : group.option_indices) {
            const auto& option = params[idx];

            // Compute log-moneyness: x = ln(S/K)
            double x = std::log(option.spot / option.strike);

            // Get today's option value (t = 0)
            // IMPORTANT: PDE solves backwards from t=T to t=0
            // - Step 0: t=T (terminal condition, payoff)
            // - Step n_time-1: t=0 (today, the price we want)
            // So we want the LAST step (final_step = n_time - 1)
            size_t final_step = grid->num_snapshots() - 1;
            auto spatial_solution = normalized_result.at_time(final_step);

            // Interpolate u(x) at query point
            CubicSpline<double> spline;
            auto build_error = spline.build(x_grid, spatial_solution);
            if (build_error.has_value()) {
                results[idx] = std::unexpected(SolverError{
                    .code = SolverErrorCode::InvalidState,
                    .message = "Failed to build spline: " + build_error.value(),
                    .iterations = 0
                });
                ++failed_count;
                continue;
            }

            double u_normalized = spline.eval(x);

            // Scale back to dimensional price: V = K·u
            double price = option.strike * u_normalized;

            // Compute Greeks from normalized surface using centered differences
            double du_dx = 0.0;  // First derivative at x
            double d2u_dx2 = 0.0;  // Second derivative at x

            // Find bracketing grid points for derivative computation
            // (simplified - production code should use CenteredDifference operator)
            size_t i_lower = 0;
            for (size_t i = 0; i < x_grid.size() - 1; ++i) {
                if (x >= x_grid[i] && x < x_grid[i+1]) {
                    i_lower = i;
                    break;
                }
            }

            if (i_lower > 0 && i_lower < x_grid.size() - 2) {
                // Use centered difference for derivatives
                double dx_grid = x_grid[i_lower+1] - x_grid[i_lower];
                du_dx = (spatial_solution[i_lower+1] - spatial_solution[i_lower-1]) / (2.0 * dx_grid);
                d2u_dx2 = (spatial_solution[i_lower+1] - 2.0*spatial_solution[i_lower] + spatial_solution[i_lower-1]) / (dx_grid * dx_grid);
            }

            // Scale Greeks back to dimensional form
            double K = option.strike;
            double S = option.spot;
            double delta = (K / S) * du_dx;           // ∂V/∂S
            double gamma = (K / (S*S)) * (d2u_dx2 - du_dx);  // ∂²V/∂S²

            // Create result with scaled price and Greeks
            // NOTE: This requires constructor that takes price/Greeks directly,
            // or we need to create a lightweight result type
            AmericanOptionResult scaled_result{
                .price = price,
                .delta = delta,
                .gamma = gamma,
                .converged = true
                // TODO: Add theta, grid reference, etc.
            };

            results[idx] = scaled_result;
        }
    }

    return BatchAmericanOptionResult{
        .results = std::move(results),
        .failed_count = failed_count
    };
}

// Helper: Group options by PDE parameters
struct PDEParameterGroup {
    double sigma;
    double rate;
    double dividend;
    OptionType option_type;
    double maturity;
    std::vector<size_t> option_indices;  // Indices into original params array
};

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
            if (std::abs(group.sigma - p.volatility) < TOL &&
                std::abs(group.rate - p.rate) < TOL &&
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

```

### Implementation Notes

1. **Result construction challenge:** The current implementation shows a simplified result construction. Production code needs to address:
   - `AmericanOptionResult` may require grid reference (for `at_time()` access)
   - We compute price/delta/gamma but result might need more fields (theta, vega, convergence info)
   - Options: lightweight result type, or shared grid ownership across results

2. **Greeks computation:** The shown centered difference approach is simplified. Production should:
   - Use `CenteredDifference` operator (unified with PDE solver)
   - Handle boundary cases (out-of-grid x values)
   - Optionally compute derivatives from spline directly (spline.eval_derivative())

3. **Snapshot extraction:** Code assumes last snapshot = t=0 (today). This needs verification against actual `Grid::at_time()` ordering:
   - If snapshots are stored forward in time (t=0 first), use step 0
   - If snapshots are stored backward (t=T first), use final step
   - Current assumption: last step = t=0 (most recent backward solve)

4. **Alternative: Bracketing for approximate grouping:** Current design groups by exact PDE parameters. We could optionally use `OptionBracketing` to group similar (σ,r) values within tolerance, reducing solves further while accepting small approximation errors.

## USDT Tracing

### Trace Point Definitions

Add to `common/ivcalc_trace.h`:

```cpp
// Normalized chain solver routing
#define MANGO_TRACE_NORMALIZED_SELECTED(batch_size) \
    DTRACE_PROBE2(MANGO_PROVIDER, normalized_selected, \
                  MODULE_BATCH_SOLVER, batch_size)

#define MANGO_TRACE_NORMALIZED_INELIGIBLE(reason_code, param_value) \
    DTRACE_PROBE3(MANGO_PROVIDER, normalized_ineligible, \
                  MODULE_BATCH_SOLVER, reason_code, param_value)
```

### Ineligibility Reason Codes

```cpp
enum class NormalizedIneligibilityReason {
    FORCED_DISABLE = 0,            ///< User disabled via set_use_normalized(false)
    SHARED_GRID_DISABLED = 1,      ///< use_shared_grid = false
    EMPTY_BATCH = 2,               ///< params.empty()
    MISMATCHED_OPTION_TYPE = 3,    ///< Mixed calls and puts
    MISMATCHED_MATURITY = 4,       ///< Options have different maturities
    DISCRETE_DIVIDENDS = 5,        ///< Options have discrete dividends
    INVALID_SPOT_OR_STRIKE = 6,    ///< Spot or strike <= 0 (can't compute log-moneyness)
    GRID_SPACING_TOO_LARGE = 7,    ///< dx > MAX_DX
    DOMAIN_TOO_WIDE = 8,           ///< width > MAX_WIDTH
    INSUFFICIENT_LEFT_MARGIN = 9,  ///< margin_left < min_margin
    INSUFFICIENT_RIGHT_MARGIN = 10 ///< margin_right < min_margin
};
```

### Tracing Implementation

```cpp
void BatchAmericanOptionSolver::trace_ineligibility_reason(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid) const
{
    if (!use_normalized_) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::FORCED_DISABLE), 0);
        return;
    }

    if (!use_shared_grid) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::SHARED_GRID_DISABLED), 0);
        return;
    }

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
    double x_min = grid_spec.x_min();
    double x_max = grid_spec.x_max();
    size_t n_space = grid_spec.n_points();

    double dx = (x_max - x_min) / (n_space - 1);
    if (dx > MAX_DX) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::GRID_SPACING_TOO_LARGE), dx);
        return;
    }

    double width = x_max - x_min;
    if (width > MAX_WIDTH) {
        MANGO_TRACE_NORMALIZED_INELIGIBLE(
            static_cast<int>(NormalizedIneligibilityReason::DOMAIN_TOO_WIDE), width);
        return;
    }

    // Check margins
    std::vector<double> moneyness_values;
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
```

### Monitoring with bpftrace

```bash
# Watch normalized solver routing decisions
sudo bpftrace -e '
usdt::mango:normalized_selected /arg0 == MODULE_BATCH_SOLVER/ {
    printf("Normalized path selected for batch of %d options\n", arg1);
}

usdt::mango:normalized_ineligible /arg0 == MODULE_BATCH_SOLVER/ {
    @reasons[arg1] = count();
}

END {
    printf("\nIneligibility reasons:\n");
    print(@reasons);
}
' -c './my_program'
```

## What Gets Removed

### Files to Remove

- `src/option/normalized_chain_solver.hpp`
- `src/option/normalized_chain_solver.cpp`
- `src/option/price_table_solver_factory.hpp`
- `src/option/price_table_solver_factory.cpp`
- `tests/normalized_chain_solver_test.cc`

### Classes/Types to Remove

- ❌ `NormalizedChainSolver` (public class)
- ❌ `NormalizedWorkspace`
- ❌ `NormalizedSolveRequest`
- ❌ `NormalizedSurfaceView`
- ❌ `EligibilityLimits` (move constants to `BatchAmericanOptionSolver`)
- ❌ `IPriceTableSolver` interface
- ❌ `PriceTableSolverFactory`
- ❌ `NormalizedPriceTableSolver`
- ❌ `BatchPriceTableSolver`

### What Remains

- ✅ `BatchAmericanOptionSolver` (with internal normalized optimization)
- ✅ `PriceTableSolver` (single class, no interface)
- ✅ `OptionBracketing` (used internally by normalized path)
- ✅ `extract_batch_results_to_4d()` (shared extraction logic)

## Implementation Plan

### Phase 1: Add Internal Normalized Path to BatchAmericanOptionSolver

**Files to modify:** `src/option/american_option_batch.hpp/cpp`

1. Add private methods to `BatchAmericanOptionSolver`:
   - `is_normalized_eligible()`
   - `trace_ineligibility_reason()`
   - `solve_normalized_chain()`
   - `merge_bracket_results()`

2. Rename current `solve_batch()` implementation to `solve_regular_batch()`

3. Create new `solve_batch()` that routes based on eligibility:
   ```cpp
   if (use_normalized_ && is_normalized_eligible(params, use_shared_grid)) {
       return solve_normalized_chain(params, setup);
   } else {
       return solve_regular_batch(params, use_shared_grid, setup);
   }
   ```

4. Add `use_normalized_` member and setter:
   ```cpp
   bool use_normalized_ = true;
   void set_use_normalized(bool enable);
   ```

5. Move eligibility constants to private members:
   - `MAX_WIDTH = 5.8`
   - `MAX_DX = 0.05`
   - `MIN_MARGIN_ABS = 0.35`

6. Add USDT trace points:
   - `MANGO_TRACE_NORMALIZED_SELECTED`
   - `MANGO_TRACE_NORMALIZED_INELIGIBLE`

**Testing:** All existing batch solver tests should pass unchanged.

### Phase 2: Simplify PriceTableSolver

**Files to modify:**
- Remove: `src/option/price_table_solver_factory.hpp/cpp`
- Update: `src/option/price_table_4d_builder.hpp/cpp`

1. Remove `IPriceTableSolver` interface

2. Remove `PriceTableSolverFactory` class

3. Merge `NormalizedPriceTableSolver` and `BatchPriceTableSolver` into single `PriceTableSolver`

4. Update `PriceTableSolver::solve()` to:
   ```cpp
   // Build params
   auto params = build_batch_params(grid, config_);

   // Just call batch solver - it handles routing internally
   BatchAmericanOptionSolver solver;
   solver.set_grid_accuracy(config_.grid_accuracy);
   auto result = solver.solve_batch(params, /*use_shared_grid=*/true);

   // Extract
   extract_batch_results_to_4d(result, prices_4d, grid, grid.K_ref);
   ```

5. Keep `extract_batch_results_to_4d()` as shared helper function

**Testing:** Price table tests should pass unchanged (API stays same).

### Phase 3: Remove Normalized Chain Solver

**Files to remove:**
- `src/option/normalized_chain_solver.hpp`
- `src/option/normalized_chain_solver.cpp`
- `tests/normalized_chain_solver_test.cc`

**Files to update:**
- `src/option/BUILD.bazel` - remove deleted targets
- `README.md` - update architecture docs
- `CLAUDE.md` - update architecture section

1. Delete normalized chain solver files

2. Remove BUILD.bazel targets:
   ```python
   # Remove these
   cc_library(name = "normalized_chain_solver", ...)
   cc_test(name = "normalized_chain_solver_test", ...)
   ```

3. Update documentation:
   - Remove "Normalized Chain Solver" section from CLAUDE.md
   - Update batch solver section to mention internal optimization
   - Remove references to `NormalizedSolveRequest`, `NormalizedWorkspace`

**Testing:** Full test suite should pass.

### Phase 4: Documentation Updates

**Files to update:**
- `CLAUDE.md`
- `README.md`
- `docs/plans/2025-01-12-normalized-solver-design.md` (add deprecation notice)

1. Update CLAUDE.md:
   - Remove normalized chain solver section
   - Add note about internal optimization in batch solver section
   - Update price table section to reflect simplified API

2. Update README.md:
   - Update example code to use simplified API
   - Remove references to normalized solver

3. Add deprecation notice to old design doc:
   ```markdown
   **DEPRECATED:** This design has been superseded by internal optimization
   in BatchAmericanOptionSolver. See `2025-11-21-normalized-chain-solver-cleanup.md`.
   ```

## Testing Strategy

### Existing Tests That Pass Unchanged

- `tests/american_option_batch_test.cc` - batch solver tests
- `tests/price_table_4d_test.cc` - price table API unchanged
- All integration tests using `BatchAmericanOptionSolver` or `PriceTableSolver`

### New Tests to Add

**Eligibility testing** (in `tests/american_option_batch_test.cc`):

```cpp
TEST(BatchAmericanOptionSolver, NormalizedEligibility) {
    // Test eligible batch: varying strikes with same maturity
    // This is the key use case for normalized solving!
    std::vector<AmericanOptionParams> eligible_params;
    double spot = 100.0;
    std::vector<double> strikes = {90, 95, 100, 105, 110};

    for (double K : strikes) {
        eligible_params.push_back({
            .spot = spot,
            .strike = K,           // Varying strikes
            .maturity = 1.0,       // Same maturity
            .volatility = 0.20,
            .rate = 0.05,
            .dividend_yield = 0.02,
            .type = OptionType::PUT,
            .discrete_dividends = {}
        });
    }

    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(eligible_params, true);

    // Should use normalized path: 1 PDE solve for 5 options
    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 5);
}

TEST(BatchAmericanOptionSolver, NormalizedIneligibleDividends) {
    // Test ineligible batch (discrete dividends)
    std::vector<AmericanOptionParams> ineligible_params;
    for (int i = 0; i < 10; ++i) {
        AmericanOptionParams p{
            .spot = 100.0,
            .strike = 100.0,
            .maturity = 1.0,
            .volatility = 0.20 + i * 0.01,
            .rate = 0.05,
            .dividend_yield = 0.02,
            .type = OptionType::PUT,
            .discrete_dividends = {{0.5, 2.0}}  // Has discrete dividend
        };
        ineligible_params.push_back(p);
    }

    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(ineligible_params, true);

    // Should fall back to regular path
    EXPECT_EQ(result.failed_count, 0);
}

TEST(BatchAmericanOptionSolver, DisableNormalizedOptimization) {
    // Test forcing regular path
    std::vector<AmericanOptionParams> params = create_eligible_batch();

    BatchAmericanOptionSolver solver;
    solver.set_use_normalized(false);

    auto result = solver.solve_batch(params, true);
    EXPECT_EQ(result.failed_count, 0);
}
```

### Tests to Remove

- `tests/normalized_chain_solver_test.cc` - functionality now internal

### Regression Testing

All existing tests should pass to ensure:
- Batch solver behavior unchanged for ineligible batches
- Price table results identical to before
- Performance characteristics maintained

## Migration Notes

### For Users of PriceTableSolver

**No changes needed** - API stays the same:

```cpp
// Before and after - identical
PriceTableSolver solver(config);
solver.solve(prices_4d, grid);
```

### For Users of BatchAmericanOptionSolver

**No changes needed** - API stays the same:

```cpp
// Before and after - identical
BatchAmericanOptionSolver solver;
solver.solve_batch(params, use_shared_grid);
```

**New capability** - disable optimization for benchmarking:

```cpp
// New: Force regular batch path
BatchAmericanOptionSolver solver;
solver.set_use_normalized(false);
solver.solve_batch(params, use_shared_grid);
```

### For Direct Users of NormalizedChainSolver

**Migration required** - use `BatchAmericanOptionSolver` instead:

```cpp
// Before
NormalizedSolveRequest request{...};
auto workspace = NormalizedWorkspace::create(request, buffer);
auto surface = workspace.surface_view();
NormalizedChainSolver::solve(request, workspace, surface);

// After
std::vector<AmericanOptionParams> params;
params.push_back({
    .spot = 1.0,    // Normalized
    .strike = 1.0,  // Normalized
    .maturity = request.T_max,
    .volatility = request.sigma,
    .rate = request.rate,
    .dividend_yield = request.dividend,
    .type = request.option_type,
    .discrete_dividends = {}
});

BatchAmericanOptionSolver solver;
auto result = solver.solve_batch(params, /*use_shared_grid=*/true);

// Access surface via result.results[0]->at_time(step_idx)
```

### For PriceTableSolverFactory Users

**Migration required** - use `PriceTableSolver` directly:

```cpp
// Before
auto solver_result = PriceTableSolverFactory::create(config, moneyness);
auto solver = std::move(solver_result.value());
solver->solve(prices_4d, grid);

// After
PriceTableSolver solver(config);
solver.solve(prices_4d, grid);
```

## Benefits

1. **Simpler architecture** - one solver, not two parallel implementations
2. **No API duplication** - `AmericanOptionParams` used everywhere
3. **Transparent optimization** - users get fast path automatically
4. **Better code reuse** - bracketing + batch solving shared
5. **Easier testing** - test one API, both paths covered
6. **Graceful fallback** - ineligible batches automatically use regular path
7. **No factory pattern** - direct instantiation, simpler code
8. **Debuggable** - can disable optimization via `set_use_normalized(false)`
9. **Observable** - USDT traces show routing decisions

## Performance Characteristics

### Expected Behavior

**Eligible batches (homogeneous, no discrete dividends):**
- Uses normalized chain solver with bracketing
- ~1500× reduction in PDE solves for large price tables
- Same accuracy as before (both use `BatchAmericanOptionSolver` internally)

**Ineligible batches:**
- Falls back to regular batch solver
- Same performance as before
- No overhead from eligibility check (simple comparisons)

### Benchmarking

```cpp
// Compare paths
std::vector<AmericanOptionParams> params = create_large_batch();

// Normalized path (automatic)
BatchAmericanOptionSolver solver_auto;
auto t1 = measure([&]{ solver_auto.solve_batch(params, true); });

// Force regular path
BatchAmericanOptionSolver solver_regular;
solver_regular.set_use_normalized(false);
auto t2 = measure([&]{ solver_regular.solve_batch(params, true); });

std::cout << "Normalized speedup: " << (t2 / t1) << "x\n";
```

## Open Questions

None - design is complete.

## References

- Original implementation: `src/option/normalized_chain_solver.cpp`
- Original design: `docs/plans/2025-01-12-normalized-solver-design.md`
- Batch bracketing: `src/option/batch_bracketing.hpp`
- Batch solver: `src/option/american_option_batch.hpp`

## Conclusion

This refactoring simplifies the normalized chain solver from a complex standalone API into a transparent internal optimization within `BatchAmericanOptionSolver`. Users benefit from automatic routing to the fast path without needing to understand the implementation details. The design eliminates unnecessary types, removes API duplication, and makes the codebase easier to maintain and test.
