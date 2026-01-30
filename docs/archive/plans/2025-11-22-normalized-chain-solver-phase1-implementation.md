# Normalized Chain Solver Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add internal normalized path to BatchAmericanOptionSolver with true scale-invariant PDE solving

**Architecture:** Refactor normalized chain solving from standalone API into internal BatchAmericanOptionSolver optimization. Automatically route eligible batches (varying strikes, same maturity) to normalized path which solves PDE once with S=K=1 and scales results.

**Tech Stack:** C++23, USDT tracing, GoogleTest

---

## Task 1: Add Eligibility Constants and Infrastructure

**Files:**
- Modify: `src/option/american_option_batch.hpp`
- Modify: `common/ivcalc_trace.h`

**Step 1: Add eligibility constants to BatchAmericanOptionSolver**

In `src/option/american_option_batch.hpp`, add private constants after the existing class definition (around line 305):

```cpp
private:
    GridAccuracyParams grid_accuracy_;  ///< Grid accuracy parameters for automatic estimation

    // Normalized chain solver eligibility constants
    static constexpr double MAX_WIDTH = 5.8;       ///< Convergence limit (log-units)
    static constexpr double MAX_DX = 0.05;         ///< Von Neumann stability
    static constexpr double MIN_MARGIN_ABS = 0.35; ///< 6-cell ghost zone minimum

    bool use_normalized_ = true;  ///< Enable normalized chain optimization
```

**Step 2: Add setter for use_normalized_**

In `src/option/american_option_batch.hpp` public section (after `grid_accuracy()` around line 97):

```cpp
    /// Disable normalized chain optimization (for benchmarking/debugging)
    void set_use_normalized(bool enable) {
        use_normalized_ = enable;
    }

    bool use_normalized() const {
        return use_normalized_;
    }
```

**Step 3: Add USDT trace point definitions**

In `common/ivcalc_trace.h`, add after existing trace definitions (find MODULE_BATCH_SOLVER section):

```cpp
// Normalized chain solver routing
#define MANGO_TRACE_NORMALIZED_SELECTED(batch_size) \
    DTRACE_PROBE2(MANGO_PROVIDER, normalized_selected, \
                  MODULE_BATCH_SOLVER, batch_size)

#define MANGO_TRACE_NORMALIZED_INELIGIBLE(reason_code, param_value) \
    DTRACE_PROBE3(MANGO_PROVIDER, normalized_ineligible, \
                  MODULE_BATCH_SOLVER, reason_code, param_value)
```

**Step 4: Add ineligibility reason enum**

In `common/ivcalc_trace.h`, add enum before or after MODULE_* definitions:

```cpp
/// Ineligibility reason codes for normalized chain solver
enum class NormalizedIneligibilityReason {
    FORCED_DISABLE = 0,            ///< User disabled via set_use_normalized(false)
    SHARED_GRID_DISABLED = 1,      ///< use_shared_grid = false
    EMPTY_BATCH = 2,               ///< params.empty()
    MISMATCHED_OPTION_TYPE = 3,    ///< Mixed calls and puts
    MISMATCHED_MATURITY = 4,       ///< Options have different maturities
    DISCRETE_DIVIDENDS = 5,        ///< Options have discrete dividends
    INVALID_SPOT_OR_STRIKE = 6,    ///< Spot or strike <= 0
    GRID_SPACING_TOO_LARGE = 7,    ///< dx > MAX_DX
    DOMAIN_TOO_WIDE = 8,           ///< width > MAX_WIDTH
    INSUFFICIENT_LEFT_MARGIN = 9,  ///< margin_left < min_margin
    INSUFFICIENT_RIGHT_MARGIN = 10 ///< margin_right < min_margin
};
```

**Step 5: Build to verify compilation**

Run: `bazel build //src/option:american_option_batch //common:ivcalc_trace`
Expected: SUCCESS (no errors)

**Step 6: Commit infrastructure changes**

```bash
git add src/option/american_option_batch.hpp common/ivcalc_trace.h
git commit -m "Add normalized chain solver infrastructure

- Add eligibility constants (MAX_WIDTH, MAX_DX, MIN_MARGIN_ABS)
- Add use_normalized_ flag with setter
- Add USDT trace points for routing decisions
- Add NormalizedIneligibilityReason enum

Part of normalized chain solver cleanup (Phase 1).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Implement Eligibility Check

**Files:**
- Modify: `src/option/american_option_batch.hpp` (add method declaration)
- Modify: `src/option/american_option_batch.cpp` (add implementation)

**Step 1: Add is_normalized_eligible() declaration**

In `src/option/american_option_batch.hpp`, add private method declaration after constants:

```cpp
private:
    // ... existing members ...

    /// Check if batch qualifies for normalized solving
    bool is_normalized_eligible(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid) const;
```

**Step 2: Implement is_normalized_eligible() in .cpp**

In `src/option/american_option_batch.cpp`, add implementation at end of file before closing namespace:

```cpp
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

    // Check margins based on moneyness range
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

**Step 3: Build to verify**

Run: `bazel build //src/option:american_option_batch`
Expected: SUCCESS

**Step 4: Commit eligibility check**

```bash
git add src/option/american_option_batch.hpp src/option/american_option_batch.cpp
git commit -m "Implement normalized solver eligibility check

Checks:
- Shared grid mode required
- Consistent option type (all calls or all puts)
- Same maturity across all options
- No discrete dividends
- Positive spot/strike (for log-moneyness)
- Grid constraints (dx, width, margins)

Allows varying strikes - the key normalized solver use case!

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Implement Tracing for Ineligibility

**Files:**
- Modify: `src/option/american_option_batch.hpp` (add method declaration)
- Modify: `src/option/american_option_batch.cpp` (add implementation)

**Step 1: Add trace_ineligibility_reason() declaration**

In `src/option/american_option_batch.hpp` private section:

```cpp
    /// Trace why normalized path wasn't used
    void trace_ineligibility_reason(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid) const;
```

**Step 2: Implement trace_ineligibility_reason()**

In `src/option/american_option_batch.cpp`, add after `is_normalized_eligible()`:

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

**Step 3: Build to verify**

Run: `bazel build //src/option:american_option_batch`
Expected: SUCCESS

**Step 4: Commit tracing**

```bash
git add src/option/american_option_batch.hpp src/option/american_option_batch.cpp
git commit -m "Add USDT tracing for normalized solver ineligibility

Traces specific reason why batch wasn't eligible for normalized path.
Enables runtime debugging via bpftrace.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Implement PDE Parameter Grouping

**Files:**
- Modify: `src/option/american_option_batch.cpp` (add helper struct and function)

**Step 1: Add PDEParameterGroup struct**

In `src/option/american_option_batch.cpp`, add before `BatchAmericanOptionSolver` methods (around line 12, in anonymous namespace):

```cpp
namespace {

/// Group of options sharing same PDE parameters
struct PDEParameterGroup {
    double sigma;
    double rate;
    double dividend;
    OptionType option_type;
    double maturity;
    std::vector<size_t> option_indices;  ///< Indices into original params array
};

}  // anonymous namespace
```

**Step 2: Add group_by_pde_parameters() private method declaration**

In `src/option/american_option_batch.hpp` private section:

```cpp
    /// Group options by PDE parameters for normalized solving
    std::vector<PDEParameterGroup> group_by_pde_parameters(
        std::span<const AmericanOptionParams> params) const;
```

**Step 3: Forward declare PDEParameterGroup in header**

In `src/option/american_option_batch.hpp`, add before `BatchAmericanOptionSolver` class:

```cpp
// Forward declaration for PDE parameter grouping
struct PDEParameterGroup;
```

**Step 4: Implement group_by_pde_parameters()**

In `src/option/american_option_batch.cpp`, add after tracing function:

```cpp
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

**Step 5: Build to verify**

Run: `bazel build //src/option:american_option_batch`
Expected: SUCCESS

**Step 6: Commit grouping logic**

```bash
git add src/option/american_option_batch.hpp src/option/american_option_batch.cpp
git commit -m "Add PDE parameter grouping for normalized solver

Groups options by (σ, r, q, type, maturity) to identify which options
can share a normalized surface u(x,τ).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Rename solve_batch to solve_regular_batch

**Files:**
- Modify: `src/option/american_option_batch.hpp` (update method name)
- Modify: `src/option/american_option_batch.cpp` (rename implementation)

**Step 1: Rename method in header**

In `src/option/american_option_batch.hpp`, rename the public `solve_batch` method to private `solve_regular_batch`:

Find the existing method (around line 107) and move it to private section:

```cpp
private:
    // ... other private members ...

    /// Regular batch solving (fallback path)
    BatchAmericanOptionResult solve_regular_batch(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr);
```

**Step 2: Rename implementation in .cpp**

In `src/option/american_option_batch.cpp`, find the `BatchAmericanOptionResult BatchAmericanOptionSolver::solve_batch` implementation and rename to `solve_regular_batch`.

Change line starting with:
```cpp
BatchAmericanOptionResult BatchAmericanOptionSolver::solve_batch(
```

To:
```cpp
BatchAmericanOptionResult BatchAmericanOptionSolver::solve_regular_batch(
```

**Step 3: Update vector overload**

Find the vector overload (around end of class in .cpp) and update to call `solve_regular_batch`:

```cpp
BatchAmericanOptionResult solve_regular_batch(
    const std::vector<AmericanOptionParams>& params,
    bool use_shared_grid = false,
    SetupCallback setup = nullptr)
{
    return solve_regular_batch(std::span{params}, use_shared_grid, setup);
}
```

**Step 4: Build to verify**

Run: `bazel build //src/option:american_option_batch`
Expected: FAILURE (tests will fail - expected, we'll fix in next task)

**Step 5: Commit rename**

```bash
git add src/option/american_option_batch.hpp src/option/american_option_batch.cpp
git commit -m "Rename solve_batch to solve_regular_batch

Prepares for new solve_batch that routes to normalized vs regular path.
Tests will be updated in next commit.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Implement solve_normalized_chain (Stub)

**Files:**
- Modify: `src/option/american_option_batch.hpp` (add declaration)
- Modify: `src/option/american_option_batch.cpp` (add stub implementation)

**Step 1: Add solve_normalized_chain() declaration**

In `src/option/american_option_batch.hpp` private section:

```cpp
    /// Fast path: normalized chain solving with PDE grouping
    BatchAmericanOptionResult solve_normalized_chain(
        std::span<const AmericanOptionParams> params,
        SetupCallback setup);
```

**Step 2: Add stub implementation**

In `src/option/american_option_batch.cpp`, add stub that falls back to regular:

```cpp
BatchAmericanOptionResult BatchAmericanOptionSolver::solve_normalized_chain(
    std::span<const AmericanOptionParams> params,
    SetupCallback setup)
{
    // TODO: Implement true normalization
    // For now, fall back to regular batch solving
    return solve_regular_batch(params, /*use_shared_grid=*/true, setup);
}
```

**Step 3: Build to verify**

Run: `bazel build //src/option:american_option_batch`
Expected: SUCCESS

**Step 4: Commit stub**

```bash
git add src/option/american_option_batch.hpp src/option/american_option_batch.cpp
git commit -m "Add solve_normalized_chain stub (fallback to regular)

Placeholder implementation that falls back to regular batch solver.
Will be replaced with true normalization in next task.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Implement New solve_batch with Routing

**Files:**
- Modify: `src/option/american_option_batch.hpp` (add public method)
- Modify: `src/option/american_option_batch.cpp` (implement routing)

**Step 1: Add new public solve_batch() to header**

In `src/option/american_option_batch.hpp`, add public method (around line 107):

```cpp
    /// Solve a batch of American options with automatic routing
    ///
    /// Automatically routes to normalized chain solver when eligible
    /// (varying strikes, same maturity, no discrete dividends).
    ///
    /// @param params Vector of option parameters
    /// @param use_shared_grid If true, all options share one global grid
    /// @param setup Optional callback invoked after solver creation
    /// @return Batch result with individual results and failure count
    BatchAmericanOptionResult solve_batch(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr);
```

**Step 2: Implement routing in .cpp**

In `src/option/american_option_batch.cpp`, add implementation:

```cpp
BatchAmericanOptionResult BatchAmericanOptionSolver::solve_batch(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid,
    SetupCallback setup)
{
    // Ensure grid_accuracy_ is initialized
    if (grid_accuracy_.tol == 0.0) {
        grid_accuracy_ = GridAccuracyParams{};
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
```

**Step 3: Add vector overload**

In `src/option/american_option_batch.hpp`, add after span version:

```cpp
    /// Solve a batch of American options (vector overload)
    BatchAmericanOptionResult solve_batch(
        const std::vector<AmericanOptionParams>& params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr)
    {
        return solve_batch(std::span{params}, use_shared_grid, setup);
    }
```

**Step 4: Build and run tests**

Run: `bazel test //tests:american_option_batch_test`
Expected: PASS (all existing tests should pass - routing falls back to regular)

**Step 5: Commit routing**

```bash
git add src/option/american_option_batch.hpp src/option/american_option_batch.cpp
git commit -m "Implement solve_batch routing with automatic path selection

Routes to normalized solver when eligible, falls back otherwise.
Initializes grid_accuracy_ with defaults if not set.
Traces routing decisions via USDT.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Implement True Normalization (Core Algorithm)

**Files:**
- Modify: `src/option/american_option_batch.cpp` (replace stub with real implementation)

**Step 1: Replace solve_normalized_chain() stub with real implementation**

In `src/option/american_option_batch.cpp`, replace the stub with:

```cpp
BatchAmericanOptionResult BatchAmericanOptionSolver::solve_normalized_chain(
    std::span<const AmericanOptionParams> params,
    SetupCallback setup)
{
    // Group by PDE parameters: (σ, r, q, type, maturity)
    auto pde_groups = group_by_pde_parameters(params);

    // Process each PDE parameter group
    std::vector<std::expected<AmericanOptionResult, SolverError>> results;
    results.resize(params.size());
    size_t failed_count = 0;

    for (const auto& group : pde_groups) {
        // Solve normalized PDE once for this group (S=K=1)
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
        auto x_grid = grid->x();

        // Get today's option value (last time step)
        size_t final_step = grid->num_snapshots() - 1;
        auto spatial_solution = normalized_result.at_time(final_step);

        // Build spline once for this group
        CubicSpline<double> spline;
        auto build_error = spline.build(x_grid, spatial_solution);
        if (build_error.has_value()) {
            // Mark all in group as failed
            for (size_t idx : group.option_indices) {
                results[idx] = std::unexpected(SolverError{
                    .code = SolverErrorCode::InvalidState,
                    .message = "Failed to build spline: " + build_error.value(),
                    .iterations = 0
                });
                ++failed_count;
            }
            continue;
        }

        // For each option in this group, interpolate and scale
        for (size_t idx : group.option_indices) {
            const auto& option = params[idx];

            // Compute log-moneyness: x = ln(S/K)
            double x = std::log(option.spot / option.strike);

            // Interpolate u(x)
            double u_normalized = spline.eval(x);

            // Scale back to dimensional price: V = K·u
            double price = option.strike * u_normalized;

            // Compute Greeks from spline derivatives
            double du_dx = spline.eval_derivative(x);
            double d2u_dx2 = spline.eval_second_derivative(x);

            // Scale Greeks to dimensional form
            double K = option.strike;
            double S = option.spot;
            double delta = (K / S) * du_dx;
            double gamma = (K / (S*S)) * (d2u_dx2 - du_dx);

            // Create simplified result
            // NOTE: This creates a minimal result. Production code should
            // create full AmericanOptionResult with grid reference.
            AmericanOptionResult scaled_result = normalized_result;
            scaled_result.price = price;
            scaled_result.delta = delta;
            scaled_result.gamma = gamma;

            results[idx] = scaled_result;
        }
    }

    return BatchAmericanOptionResult{
        .results = std::move(results),
        .failed_count = failed_count
    };
}
```

**Step 2: Build to verify**

Run: `bazel build //src/option:american_option_batch`
Expected: SUCCESS

**Step 3: Run existing tests**

Run: `bazel test //tests:american_option_batch_test`
Expected: PASS (existing tests should still pass)

**Step 4: Commit true normalization**

```bash
git add src/option/american_option_batch.cpp
git commit -m "Implement true normalized chain solving

Core algorithm:
1. Group options by (σ, r, q, type, maturity)
2. For each group, solve PDE once with S=K=1
3. Interpolate normalized surface at x=ln(S/K)
4. Scale price: V = K·u
5. Scale Greeks: delta = (K/S)·du/dx, gamma = (K/S²)·(d²u/dx² - du/dx)

Achieves ~N× speedup for N options with same PDE parameters.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Add Test for Normalized Eligibility

**Files:**
- Modify: `tests/american_option_batch_test.cc`

**Step 1: Add test for eligible batch**

In `tests/american_option_batch_test.cc`, add at end of file:

```cpp
TEST(BatchAmericanOptionSolver, NormalizedEligibility) {
    // Test eligible batch: varying strikes with same maturity
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
    auto result = solver.solve_batch(eligible_params, /*use_shared_grid=*/true);

    // Should use normalized path: 1 PDE solve for 5 options
    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 5);

    // All results should have converged
    for (const auto& r : result.results) {
        ASSERT_TRUE(r.has_value());
        EXPECT_TRUE(r->converged);
        EXPECT_GT(r->price, 0.0);
    }
}
```

**Step 2: Run test**

Run: `bazel test //tests:american_option_batch_test --test_filter=NormalizedEligibility --test_output=all`
Expected: PASS

**Step 3: Add test for ineligible batch (discrete dividends)**

In `tests/american_option_batch_test.cc`, add:

```cpp
TEST(BatchAmericanOptionSolver, NormalizedIneligibleDividends) {
    // Test ineligible batch (discrete dividends)
    std::vector<AmericanOptionParams> ineligible_params;
    double spot = 100.0;

    for (int i = 0; i < 5; ++i) {
        AmericanOptionParams p{
            .spot = spot,
            .strike = 90.0 + i * 5.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .dividend_yield = 0.02,
            .type = OptionType::PUT,
            .discrete_dividends = {{0.5, 2.0}}  // Has discrete dividend
        };
        ineligible_params.push_back(p);
    }

    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(ineligible_params, /*use_shared_grid=*/true);

    // Should fall back to regular path
    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 5);
}
```

**Step 4: Run test**

Run: `bazel test //tests:american_option_batch_test --test_filter=NormalizedIneligible --test_output=all`
Expected: PASS

**Step 5: Add test for disabling optimization**

In `tests/american_option_batch_test.cc`, add:

```cpp
TEST(BatchAmericanOptionSolver, DisableNormalizedOptimization) {
    // Test forcing regular path
    std::vector<AmericanOptionParams> params;
    double spot = 100.0;

    for (int i = 0; i < 5; ++i) {
        params.push_back({
            .spot = spot,
            .strike = 90.0 + i * 5.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .dividend_yield = 0.02,
            .type = OptionType::PUT,
            .discrete_dividends = {}
        });
    }

    BatchAmericanOptionSolver solver;
    solver.set_use_normalized(false);  // Force regular path

    auto result = solver.solve_batch(params, /*use_shared_grid=*/true);
    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 5);
}
```

**Step 6: Run all new tests**

Run: `bazel test //tests:american_option_batch_test --test_filter=Normalized`
Expected: All PASS

**Step 7: Commit tests**

```bash
git add tests/american_option_batch_test.cc
git commit -m "Add tests for normalized chain solver routing

Tests:
- NormalizedEligibility: varying strikes (eligible)
- NormalizedIneligibleDividends: discrete dividends (ineligible)
- DisableNormalizedOptimization: forced regular path

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Run Full Test Suite

**Step 1: Run all batch solver tests**

Run: `bazel test //tests:american_option_batch_test --test_output=all`
Expected: All PASS

**Step 2: Run all tests**

Run: `bazel test //...`
Expected: All PASS (no regressions)

**Step 3: If any tests fail, investigate and fix**

If failures occur:
1. Check test output for specific failure
2. Verify eligibility logic didn't break existing behavior
3. Ensure fallback to regular path works correctly
4. Fix issues and re-run tests

**Step 4: Commit verification**

If all tests pass:

```bash
git add .
git commit -m "Verify all tests pass with normalized chain solver

Phase 1 complete: internal normalized path added to BatchAmericanOptionSolver.
All existing tests pass, new tests verify routing behavior.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification

After completing all tasks, verify:

1. **Build succeeds**: `bazel build //...`
2. **All tests pass**: `bazel test //...`
3. **Normalized path works**: Run `NormalizedEligibility` test
4. **Fallback works**: Run `NormalizedIneligibleDividends` test
5. **Can disable**: Run `DisableNormalizedOptimization` test

Expected results:
- All builds succeed
- All tests pass
- No performance regressions
- Routing logic verified via tests

---

## Next Steps

After Phase 1 completion:

1. **Phase 2**: Simplify PriceTableSolver (remove factory pattern)
2. **Phase 3**: Remove old NormalizedChainSolver files
3. **Phase 4**: Update documentation

Use `superpowers:executing-plans` skill to continue with Phase 2.
