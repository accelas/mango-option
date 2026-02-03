# Adaptive Grid for Segmented Path — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable `AdaptiveGrid` with `SegmentedIVPath` so users don't need to hand-pick grid density for discrete dividend IV.

**Architecture:** Extract the refinement loop from `AdaptiveGridBuilder::build()` into a callback-based helper `run_refinement()`. Add `build_segmented()` which probes 2-3 K_refs using segmented PDE surfaces, takes per-axis max grid sizes, then builds all segments with a uniform grid. A final multi-K_ref validation pass catches cross-K_ref interpolation errors.

**Tech Stack:** C++23, Bazel, GoogleTest

**Design doc:** `docs/plans/2026-02-03-adaptive-segmented-grid-design.md`

---

### Task 1: Add `skip_moneyness_expansion` flag to SegmentedPriceTableBuilder

The segmented builder expands moneyness downward by `total_div / K_ref`. When the caller pre-expands, this must be skippable.

**Files:**
- Modify: `src/option/table/segmented_price_table_builder.hpp:19-29`
- Modify: `src/option/table/segmented_price_table_builder.cpp:130-154`
- Test: `tests/segmented_multi_kref_builder_test.cc`

**Step 1: Write the failing test**

Add to `tests/segmented_multi_kref_builder_test.cc`:

```cpp
TEST(SegmentedPriceTableBuilderTest, SkipMoneyExpansionFlag) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02,
                      .discrete_dividends = {{.calendar_time = 0.5, .amount = 5.0}}},
        .moneyness_grid = {0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
        .maturity = 1.0,
        .vol_grid = {0.10, 0.15, 0.20, 0.30},
        .rate_grid = {0.02, 0.03, 0.05, 0.07},
        .skip_moneyness_expansion = true,
    };

    auto surface = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(surface.has_value());

    // When expansion is skipped, the surface's moneyness range should
    // match the input grid exactly (no extra points below 0.6).
    // Query at the lowest input moneyness should work.
    double price = surface->price(100.0, 100.0 / 0.6, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:segmented_multi_kref_builder_test --test_output=all --test_filter='*SkipMoneyExpansion*'`
Expected: FAIL — `skip_moneyness_expansion` doesn't exist yet.

**Step 3: Implement**

In `src/option/table/segmented_price_table_builder.hpp`, add to Config:

```cpp
/// If true, skip internal moneyness expansion (caller pre-expanded).
bool skip_moneyness_expansion = false;
```

In `src/option/table/segmented_price_table_builder.cpp`, wrap the expansion block (lines 130-154) in a conditional:

```cpp
std::vector<double> expanded_m_grid = config.moneyness_grid;
if (!config.skip_moneyness_expansion) {
    // Existing expansion code (lines 132-154)
    double total_div = 0.0;
    for (const auto& div : dividends) {
        total_div += div.amount;
    }
    double m_min_expanded = config.moneyness_grid.front() - total_div / K_ref;
    if (m_min_expanded < 0.01) m_min_expanded = 0.01;

    if (m_min_expanded < expanded_m_grid.front()) {
        double step = (expanded_m_grid.front() - m_min_expanded) / 3.0;
        for (int i = 2; i >= 0; --i) {
            double val = m_min_expanded + step * static_cast<double>(i);
            if (val > 0.0 && val < expanded_m_grid.front()) {
                expanded_m_grid.insert(expanded_m_grid.begin(), val);
            }
        }
    }
    std::sort(expanded_m_grid.begin(), expanded_m_grid.end());
    expanded_m_grid.erase(
        std::unique(expanded_m_grid.begin(), expanded_m_grid.end()),
        expanded_m_grid.end());
}
```

**Step 4: Run tests**

Run: `bazel test //tests:segmented_multi_kref_builder_test --test_output=all`
Expected: ALL PASS (existing + new)

**Step 5: Commit**

```bash
git add src/option/table/segmented_price_table_builder.hpp \
        src/option/table/segmented_price_table_builder.cpp \
        tests/segmented_multi_kref_builder_test.cc
git commit -m "Add skip_moneyness_expansion flag to SegmentedPriceTableBuilder"
```

---

### Task 2: Extract `run_refinement()` from `AdaptiveGridBuilder::build()`

Refactor the ~300-line refinement loop into a callback-based private helper. The existing `build()` becomes a thin wrapper. No behavior change.

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.hpp`
- Modify: `src/option/table/adaptive_grid_builder.cpp`
- Test: `tests/adaptive_grid_builder_test.cc` (existing tests must still pass)

**Step 1: Run existing tests (baseline)**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all`
Expected: ALL PASS

**Step 2: Define private types and helper signature**

In `adaptive_grid_builder.cpp` (file scope, inside `namespace mango`), add these types **before** the `AdaptiveGridBuilder` methods:

```cpp
namespace {

/// Type-erased surface for validation
struct SurfaceHandle {
    std::function<double(double spot, double strike, double tau,
                         double sigma, double rate)> price;
};

/// Builds a surface from current grid sizes
using BuildFn = std::function<std::expected<SurfaceHandle, PriceTableError>(
    const std::vector<double>& moneyness,
    const std::vector<double>& vol,
    const std::vector<double>& rate,
    int tau_points)>;

/// Produces a fresh FD reference price for validation
using ValidateFn = std::function<std::expected<double, SolverError>(
    double spot, double strike, double tau,
    double sigma, double rate)>;

/// Context shared across all iterations
struct RefinementContext {
    double spot;
    double dividend_yield;
    OptionType option_type;
    double min_moneyness, max_moneyness;
    double min_tau, max_tau;
    double min_vol, max_vol;
    double min_rate, max_rate;
};

/// Result of grid sizing
struct GridSizes {
    std::vector<double> moneyness;
    std::vector<double> vol;
    std::vector<double> rate;
    int tau_points;
    double achieved_max_error;
    double achieved_avg_error;
    bool target_met;
};

}  // namespace
```

Add the private method declaration in `adaptive_grid_builder.hpp`:

```cpp
// In the private section of AdaptiveGridBuilder:
struct RefinementContext;
struct GridSizes;

GridSizes run_refinement(
    std::function<std::expected</* SurfaceHandle */ void*, PriceTableError>(
        const std::vector<double>&, const std::vector<double>&,
        const std::vector<double>&, int)> build_fn,
    std::function<std::expected<double, SolverError>(
        double, double, double, double, double)> validate_fn,
    /* context params */ double spot, double dividend_yield, OptionType type,
    double min_m, double max_m, double min_tau, double max_tau,
    double min_vol, double max_vol, double min_rate, double max_rate);
```

Actually, since the types are file-local, declare the helper as a **free function** in the .cpp rather than a class method. This avoids polluting the header. The approach:

1. Extract the loop body (lines 135-561) into a free function `run_refinement()` in the .cpp
2. `build()` calls `run_refinement()` passing lambdas for build and validate

**Step 3: Extract the loop**

Move lines 103-561 of `adaptive_grid_builder.cpp` (everything from seed grids through the loop) into:

```cpp
static GridSizes run_refinement(
    const AdaptiveGridParams& params,
    BuildFn build_fn,
    ValidateFn validate_fn,
    const RefinementContext& ctx)
{
    // linspace helper (moved from build)
    auto linspace = [](double lo, double hi, size_t n) { ... };

    // Seed grids
    std::vector<double> moneyness_grid = linspace(ctx.min_moneyness, ctx.max_moneyness, 5);
    std::vector<double> vol_grid = linspace(ctx.min_vol, ctx.max_vol, 5);
    std::vector<double> rate_grid = linspace(ctx.min_rate, ctx.max_rate, 4);
    int tau_points = 5;  // initial

    // Main loop (lines 135-561 adapted)
    // Replace surface building with build_fn(moneyness_grid, vol_grid, rate_grid, tau_points)
    // Replace validation FD solves with validate_fn(spot, strike, tau, sigma, rate)
    // Replace surface.value()->value({m,tau,sigma,rate}) with surface_handle.price(...)
    // ...

    // Return final grid sizes
    return GridSizes{
        .moneyness = std::move(moneyness_grid),
        .vol = std::move(vol_grid),
        .rate = std::move(rate_grid),
        .tau_points = tau_points,
        .achieved_max_error = max_error,
        .achieved_avg_error = avg_error,
        .target_met = converged,
    };
}
```

**Step 4: Rewrite `build()` as a wrapper**

```cpp
std::expected<AdaptiveResult, PriceTableError>
AdaptiveGridBuilder::build(const OptionGrid& chain,
                           GridSpec<double> grid_spec,
                           size_t n_time,
                           OptionType type)
{
    cache_.clear();

    // Validation (lines 36-47 unchanged)
    ...

    // Extract bounds (lines 49-101 unchanged)
    ...

    // Build callback: wraps PriceTableBuilder + SliceCache
    BuildFn build_fn = [&](const std::vector<double>& m_grid,
                           const std::vector<double>& v_grid,
                           const std::vector<double>& r_grid,
                           int tau_pts) -> std::expected<SurfaceHandle, PriceTableError>
    {
        // Use maturity_grid from tau_pts (for standard path, this is
        // the existing maturity grid derived from chain.maturities)
        // ... existing PriceTableBuilder + SliceCache logic ...
        // Return SurfaceHandle wrapping the built surface
    };

    // Validate callback: wraps solve_american_option
    ValidateFn validate_fn = [&](double spot, double strike, double tau,
                                  double sigma, double rate)
        -> std::expected<double, SolverError>
    {
        PricingParams params;
        params.spot = spot;
        params.strike = strike;
        params.maturity = tau;
        params.rate = rate;
        params.dividend_yield = chain.dividend_yield;
        params.option_type = type;
        params.volatility = sigma;
        auto fd = solve_american_option(params);
        if (!fd.has_value()) return std::unexpected(fd.error());
        return fd->value();
    };

    RefinementContext ctx{
        .spot = chain.spot,
        .dividend_yield = chain.dividend_yield,
        .option_type = type,
        .min_moneyness = min_moneyness, .max_moneyness = max_moneyness,
        .min_tau = min_tau, .max_tau = max_tau,
        .min_vol = min_vol, .max_vol = max_vol,
        .min_rate = min_rate, .max_rate = max_rate,
    };

    auto sizes = run_refinement(params_, build_fn, validate_fn, ctx);

    // Build final AdaptiveResult from sizes
    // (the last build_fn call during run_refinement stored the surface)
    ...
}
```

**Important nuance:** The standard path's `build_fn` closure is complex — it uses `PriceTableBuilder`, `SliceCache`, `BatchAmericanOptionSolver`, grid stability checks, tensor extraction, repair, and B-spline fitting (lines 148-329). This entire block becomes the `build_fn` body. The closure captures `grid_spec`, `n_time`, `&cache_`, the maturity grid, etc.

The key insight: for the standard path, `tau_points` maps to the maturity grid size. The maturity grid comes from `chain.maturities` and is linspaced between min/max tau. When tau is refined, `run_refinement` adds midpoints to the maturity grid (same as current behavior). So `tau_points` is really `maturity_grid.size()` for the standard path — the `build_fn` receives it and generates a linspace maturity grid.

Actually this is getting complex. Let me simplify: keep the maturity grid as a `std::vector<double>` inside `run_refinement` alongside the other grids. The `build_fn` receives all four grids (m, tau, vol, rate). For the standard path, the tau grid is the maturity grid. For the segmented path, the tau vector has just one element (the `tau_points_per_segment` count encoded as a vector size).

No — even simpler. Pass all four grid vectors to `build_fn`:

```cpp
using BuildFn = std::function<std::expected<SurfaceHandle, PriceTableError>(
    const std::vector<double>& moneyness,
    const std::vector<double>& tau_or_maturity,
    const std::vector<double>& vol,
    const std::vector<double>& rate)>;
```

For the standard path, `tau_or_maturity` is the actual maturity grid (linspace, refined with midpoints). For the segmented path, `tau_or_maturity` is ignored — the segmented `build_fn` closure captures `tau_points_per_segment` and increments it based on how many times dim 1 was refined.

Wait, that doesn't work because the refinement loop refines the tau grid by adding midpoints. For the segmented path, it needs to increment a scalar instead.

**Simplest approach:** Keep all 4 grid vectors in `run_refinement`. The segmented `build_fn` reads `tau_grid.size()` and uses it as `tau_points_per_segment`. The refinement loop adds midpoints to the tau vector as usual (which increases its size). The segmented `build_fn` just uses the **size** of the tau vector, not its actual values.

This is clean and requires no special-casing in the loop.

**Step 5: Run tests to verify no regression**

Run: `bazel test //tests:adaptive_grid_builder_test //tests:iv_solver_factory_test --test_output=all`
Expected: ALL PASS (behavior unchanged)

**Step 6: Commit**

```bash
git add src/option/table/adaptive_grid_builder.hpp \
        src/option/table/adaptive_grid_builder.cpp
git commit -m "Extract run_refinement() from AdaptiveGridBuilder::build()"
```

---

### Task 3: Add `build_segmented()` method

Implement the probe + max approach with segmented PDE and final multi-K_ref validation.

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.hpp`
- Modify: `src/option/table/adaptive_grid_builder.cpp`
- Test: `tests/adaptive_grid_builder_test.cc`

**Step 1: Write the failing test**

Add to `tests/adaptive_grid_builder_test.cc`:

```cpp
TEST(AdaptiveGridBuilderTest, BuildSegmentedBasic) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;  // 50 bps — relaxed for test speed
    params.max_iter = 2;
    params.validation_samples = 16;

    AdaptiveGrid grid{.params = params};

    AdaptiveGridBuilder builder(params);
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto result = builder.build_segmented(seg_config, grid);
    ASSERT_TRUE(result.has_value())
        << "build_segmented failed: " << static_cast<int>(result.error().code);

    // Should be able to query prices at various strikes
    double price = result->price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));

    // And at off-K_ref strikes
    double price2 = result->price(100.0, 90.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price2, 0.0);
    EXPECT_TRUE(std::isfinite(price2));
}

TEST(AdaptiveGridBuilderTest, BuildSegmentedSmallKRefList) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;

    AdaptiveGrid grid{.params = params};

    AdaptiveGridBuilder builder(params);
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {{.calendar_time = 0.25, .amount = 1.50}},
        .maturity = 0.5,
        .kref_config = {.K_refs = {95.0, 105.0}},  // < 3 K_refs
    };

    auto result = builder.build_segmented(seg_config, grid);
    ASSERT_TRUE(result.has_value());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all --test_filter='*BuildSegmented*'`
Expected: FAIL — `build_segmented` and `SegmentedAdaptiveConfig` don't exist.

**Step 3: Add types and declaration to header**

In `src/option/table/adaptive_grid_builder.hpp`, add includes and types:

```cpp
#include "mango/option/table/segmented_multi_kref_builder.hpp"
#include "mango/option/table/segmented_multi_kref_surface.hpp"

// Before AdaptiveGridBuilder class:
struct AdaptiveGrid;  // forward-declared; defined in iv_solver_factory.hpp

/// Configuration for segmented adaptive grid building
struct SegmentedAdaptiveConfig {
    double spot;
    OptionType option_type;
    double dividend_yield;
    std::vector<Dividend> discrete_dividends;
    double maturity;
    MultiKRefConfig kref_config;
};
```

Add method to `AdaptiveGridBuilder`:

```cpp
/// Build segmented multi-K_ref surface with adaptive grid refinement
///
/// Probes 2-3 representative K_refs, takes per-axis max grid sizes,
/// then builds all segments with a uniform grid.  Final validation
/// pass catches cross-K_ref interpolation errors.
///
/// @param config Segmented path configuration
/// @param grid Adaptive grid domain bounds and params
/// @return SegmentedMultiKRefSurface or error
[[nodiscard]] std::expected<SegmentedMultiKRefSurface, PriceTableError>
build_segmented(const SegmentedAdaptiveConfig& config,
                const AdaptiveGrid& grid);
```

Note: `AdaptiveGrid` is defined in `iv_solver_factory.hpp`. To avoid a circular include, either:
- Forward-declare `AdaptiveGrid` and include the factory header in the .cpp, or
- Move `AdaptiveGrid` to its own header (e.g., `adaptive_grid_types.hpp`)

The cleanest option: move `AdaptiveGrid` (and `ManualGrid`, `IVGridSpec`) to `adaptive_grid_types.hpp` alongside `AdaptiveGridParams`. But that's a bigger refactor. For now, forward-declare and include in the .cpp.

Actually, `AdaptiveGrid` has `AdaptiveGridParams params` and three vectors — the .hpp needs the full definition for the method signature. So we need the include. The simplest: include `iv_solver_factory.hpp` in the .cpp (not the .hpp), and pass `build_segmented` the pieces it needs:

```cpp
[[nodiscard]] std::expected<SegmentedMultiKRefSurface, PriceTableError>
build_segmented(const SegmentedAdaptiveConfig& config,
                const std::vector<double>& moneyness_domain,
                const std::vector<double>& vol_domain,
                const std::vector<double>& rate_domain);
```

The `AdaptiveGridParams` are already in `params_`. The caller extracts domain bounds from `AdaptiveGrid` and passes them. This keeps the header dependency-free.

**Step 4: Implement `build_segmented()`**

In `adaptive_grid_builder.cpp`:

```cpp
std::expected<SegmentedMultiKRefSurface, PriceTableError>
AdaptiveGridBuilder::build_segmented(
    const SegmentedAdaptiveConfig& config,
    const std::vector<double>& moneyness_domain,
    const std::vector<double>& vol_domain,
    const std::vector<double>& rate_domain)
{
    // 1. Determine full K_ref list
    std::vector<double> K_refs = config.kref_config.K_refs;
    if (K_refs.empty()) {
        // Auto-generate (same logic as SegmentedMultiKRefBuilder)
        const int count = config.kref_config.K_ref_count;
        const double span = config.kref_config.K_ref_span;
        const double log_lo = std::log(1.0 - span);
        const double log_hi = std::log(1.0 + span);
        for (int i = 0; i < count; ++i) {
            double t = (count == 1) ? 0.5 :
                static_cast<double>(i) / static_cast<double>(count - 1);
            K_refs.push_back(config.spot * std::exp(log_lo + t * (log_hi - log_lo)));
        }
    }
    std::sort(K_refs.begin(), K_refs.end());

    // 2. Select probe K_refs (ATM, lowest, highest)
    std::vector<double> probe_krefs;
    if (K_refs.size() <= 3) {
        probe_krefs = K_refs;
    } else {
        probe_krefs.push_back(K_refs.front());   // lowest
        probe_krefs.push_back(K_refs.back());    // highest
        // ATM: closest to spot
        auto atm_it = std::min_element(K_refs.begin(), K_refs.end(),
            [&](double a, double b) {
                return std::abs(a - config.spot) < std::abs(b - config.spot);
            });
        if (*atm_it != K_refs.front() && *atm_it != K_refs.back()) {
            probe_krefs.push_back(*atm_it);
        }
    }

    // 3. Pre-expand moneyness domain
    auto filtered = filter_dividends_for_expansion(
        config.discrete_dividends, config.maturity);
    double total_div = 0.0;
    for (const auto& d : filtered) total_div += d.amount;
    double K_ref_min = K_refs.front();
    double expansion = total_div / K_ref_min;

    double min_m = moneyness_domain.front();
    double max_m = moneyness_domain.back();
    double expanded_min_m = std::max(min_m - expansion, 0.01);

    // Extract vol/rate bounds
    double min_vol = vol_domain.front();
    double max_vol = vol_domain.back();
    double min_rate = rate_domain.front();
    double max_rate = rate_domain.back();

    // Expand bounds (same logic as build())
    // ... expand_bounds_positive / expand_bounds ...

    // 4. Run probes
    std::vector<GridSizes> probe_results;
    for (double K_ref : probe_krefs) {
        // Build callback: SegmentedPriceTableBuilder
        BuildFn build_fn = [&, K_ref](
            const std::vector<double>& m_grid,
            const std::vector<double>& tau_grid,
            const std::vector<double>& v_grid,
            const std::vector<double>& r_grid)
            -> std::expected<SurfaceHandle, PriceTableError>
        {
            int tau_pts = static_cast<int>(tau_grid.size());
            SegmentedPriceTableBuilder::Config seg_cfg{
                .K_ref = K_ref,
                .option_type = config.option_type,
                .dividends = {.dividend_yield = config.dividend_yield,
                              .discrete_dividends = config.discrete_dividends},
                .moneyness_grid = m_grid,
                .maturity = config.maturity,
                .vol_grid = v_grid,
                .rate_grid = r_grid,
                .tau_points_per_segment = tau_pts,
                .skip_moneyness_expansion = true,
            };
            auto surface = SegmentedPriceTableBuilder::build(seg_cfg);
            if (!surface.has_value()) {
                return std::unexpected(PriceTableError{
                    PriceTableErrorCode::InvalidConfig});
            }
            auto shared = std::make_shared<SegmentedPriceSurface>(
                std::move(*surface));
            double spot = config.spot;
            return SurfaceHandle{
                .price = [shared, spot, K_ref](double /*spot_arg*/, double strike,
                                                double tau, double sigma, double rate) {
                    return shared->price(spot, strike, tau, sigma, rate);
                },
            };
        };

        // Validate callback: solve_american_option with discrete dividends
        // Validate at strike = K_ref (probe validates at its own K_ref)
        ValidateFn validate_fn = [&](
            double spot, double strike, double tau,
            double sigma, double rate)
            -> std::expected<double, SolverError>
        {
            PricingParams params;
            params.spot = spot;
            params.strike = strike;
            params.maturity = tau;
            params.rate = rate;
            params.dividend_yield = config.dividend_yield;
            params.option_type = config.option_type;
            params.volatility = sigma;
            params.discrete_dividends = config.discrete_dividends;
            auto fd = solve_american_option(params);
            if (!fd.has_value()) return std::unexpected(fd.error());
            return fd->value();
        };

        RefinementContext ctx{
            .spot = config.spot,
            .dividend_yield = config.dividend_yield,
            .option_type = config.option_type,
            .min_moneyness = expanded_min_m,
            .max_moneyness = max_m,
            .min_tau = std::min(0.01, config.maturity * 0.5),
            .max_tau = config.maturity,
            .min_vol = min_vol, .max_vol = max_vol,
            .min_rate = min_rate, .max_rate = max_rate,
        };

        auto sizes = run_refinement(params_, build_fn, validate_fn, ctx);
        probe_results.push_back(std::move(sizes));
    }

    // 5. Per-axis max
    size_t max_m_size = 0, max_v_size = 0, max_r_size = 0;
    int max_tau_pts = 0;
    for (const auto& pr : probe_results) {
        max_m_size = std::max(max_m_size, pr.moneyness.size());
        max_v_size = std::max(max_v_size, pr.vol.size());
        max_r_size = std::max(max_r_size, pr.rate.size());
        max_tau_pts = std::max(max_tau_pts, pr.tau_points);
    }

    // 6. Generate uniform grids with max sizes
    auto linspace = [](double lo, double hi, size_t n) { ... };
    auto final_m = linspace(expanded_min_m, max_m, max_m_size);
    auto final_v = linspace(min_vol, max_vol, max_v_size);
    auto final_r = linspace(min_rate, max_rate, max_r_size);

    // 7. Build full SegmentedMultiKRefSurface
    SegmentedMultiKRefBuilder::Config mkref_config{
        .spot = config.spot,
        .option_type = config.option_type,
        .dividends = {.dividend_yield = config.dividend_yield,
                      .discrete_dividends = config.discrete_dividends},
        .moneyness_grid = final_m,
        .maturity = config.maturity,
        .vol_grid = final_v,
        .rate_grid = final_r,
        .kref_config = config.kref_config,
    };

    auto surface = SegmentedMultiKRefBuilder::build(mkref_config);
    if (!surface.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // 8. Final multi-K_ref validation at arbitrary strikes
    // ... LHS samples, compare surface->price() vs solve_american_option()
    // If error > target, bump grids by one refinement step and rebuild
    // Return best surface

    return std::move(*surface);
}
```

**Step 5: Run tests**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all`
Expected: ALL PASS (existing + new)

**Step 6: Commit**

```bash
git add src/option/table/adaptive_grid_builder.hpp \
        src/option/table/adaptive_grid_builder.cpp \
        tests/adaptive_grid_builder_test.cc
git commit -m "Add build_segmented() with probe + max approach"
```

---

### Task 4: Wire `build_segmented()` into the IV solver factory

Connect `AdaptiveGrid` + `SegmentedIVPath` through the factory.

**Files:**
- Modify: `src/option/iv_solver_factory.cpp:131-164`
- Test: `tests/iv_solver_factory_test.cc`

**Step 1: Write the failing test**

Add to `tests/iv_solver_factory_test.cc`:

```cpp
TEST(IVSolverFactorySegmented, AdaptiveGridDiscreteDividends) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;  // 50 bps for test speed
    params.max_iter = 2;
    params.validation_samples = 16;

    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.02,
        .grid = AdaptiveGrid{.params = params},
        .path = SegmentedIVPath{
            .maturity = 1.0,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}},
            .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value())
        << "Factory should succeed with AdaptiveGrid + SegmentedIVPath";

    // Solve IV for a known option
    PricingParams pricing_params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.5,
                   .rate = 0.05, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT},
        0.20);
    pricing_params.discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}};
    auto ref = solve_american_option(pricing_params);
    ASSERT_TRUE(ref.has_value());

    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.5,
                   .rate = 0.05, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT},
        ref->value());

    auto result = solver->solve(query);
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 3.0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:iv_solver_factory_test --test_output=all --test_filter='*AdaptiveGrid*'`
Expected: FAIL — factory rejects AdaptiveGrid with SegmentedIVPath.

**Step 3: Implement factory branch**

In `src/option/iv_solver_factory.cpp`, modify `build_segmented()`:

```cpp
static std::expected<AnyIVSolver, ValidationError>
build_segmented(const IVSolverFactoryConfig& config, const SegmentedIVPath& path) {
    return std::visit([&](const auto& grid) -> std::expected<AnyIVSolver, ValidationError> {
        using G = std::decay_t<decltype(grid)>;

        if constexpr (std::is_same_v<G, AdaptiveGrid>) {
            // Adaptive grid for segmented path
            AdaptiveGridBuilder builder(grid.params);
            SegmentedAdaptiveConfig seg_config{
                .spot = config.spot,
                .option_type = config.option_type,
                .dividend_yield = config.dividend_yield,
                .discrete_dividends = path.discrete_dividends,
                .maturity = path.maturity,
                .kref_config = path.kref_config,
            };
            auto surface = builder.build_segmented(
                seg_config, grid.moneyness, grid.vol, grid.rate);
            if (!surface.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidGridSize, 0.0});
            }

            auto solver = InterpolatedIVSolver<SegmentedMultiKRefSurface>::create(
                std::move(*surface), config.solver_config);
            if (!solver.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidGridSize, 0.0});
            }
            return AnyIVSolver(std::move(*solver));
        }

        // Manual grid: existing path (unchanged)
        SegmentedMultiKRefBuilder::Config seg_config{
            .spot = config.spot,
            .option_type = config.option_type,
            .dividends = {.dividend_yield = config.dividend_yield,
                          .discrete_dividends = path.discrete_dividends},
            .moneyness_grid = grid.moneyness,
            .maturity = path.maturity,
            .vol_grid = grid.vol,
            .rate_grid = grid.rate,
            .kref_config = path.kref_config,
        };

        auto surface = SegmentedMultiKRefBuilder::build(seg_config);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }

        auto solver = InterpolatedIVSolver<SegmentedMultiKRefSurface>::create(
            std::move(*surface), config.solver_config);
        if (!solver.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidGridSize, 0.0});
        }
        return AnyIVSolver(std::move(*solver));
    }, config.grid);
}
```

Also update the doc comment in `iv_solver_factory.hpp`:

```cpp
/// If grid holds AdaptiveGrid, uses AdaptiveGridBuilder
/// to automatically refine grid density (both standard and segmented paths).
```

**Step 4: Run tests**

Run: `bazel test //tests:iv_solver_factory_test --test_output=all`
Expected: ALL PASS

**Step 5: Full regression**

Run: `bazel test //...`
Expected: ALL PASS

Run: `bazel build //benchmarks/...`
Expected: BUILD OK

Run: `bazel build //src/python:mango_option`
Expected: BUILD OK

**Step 6: Commit**

```bash
git add src/option/iv_solver_factory.hpp \
        src/option/iv_solver_factory.cpp \
        tests/iv_solver_factory_test.cc
git commit -m "Wire AdaptiveGrid + SegmentedIVPath through factory"
```

---

### Task 5: Final validation pass and edge case tests

Add the multi-K_ref final validation step and comprehensive edge case tests.

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.cpp` (final validation in `build_segmented`)
- Test: `tests/adaptive_grid_builder_test.cc`

**Step 1: Write edge case tests**

```cpp
// Large discrete dividend (total_div/K_ref > 0.3, stresses moneyness expansion)
TEST(AdaptiveGridBuilderTest, BuildSegmentedLargeDividend) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 2;
    params.validation_samples = 16;

    AdaptiveGrid grid{.params = params};
    AdaptiveGridBuilder builder(params);

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {{.calendar_time = 0.25, .amount = 10.0},
                               {.calendar_time = 0.75, .amount = 10.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {70.0, 100.0, 130.0}},
    };

    auto result = builder.build_segmented(
        seg_config, grid.moneyness, grid.vol, grid.rate);
    ASSERT_TRUE(result.has_value());

    double price = result->price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));
}

// No dividends (single segment, degenerates to simple case)
TEST(AdaptiveGridBuilderTest, BuildSegmentedNoDividends) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;

    AdaptiveGrid grid{.params = params};
    AdaptiveGridBuilder builder(params);

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {},  // No discrete dividends
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto result = builder.build_segmented(
        seg_config, grid.moneyness, grid.vol, grid.rate);
    ASSERT_TRUE(result.has_value());
}
```

**Step 2: Implement final multi-K_ref validation**

In `build_segmented()`, after building the full surface (step 7), add:

```cpp
// 8. Final multi-K_ref validation at arbitrary strikes
auto final_samples = latin_hypercube_4d(
    params_.validation_samples, params_.lhs_seed + 999);

std::array<std::pair<double, double>, 4> final_bounds = {{
    {expanded_min_m, max_m},
    {std::min(0.01, config.maturity * 0.5), config.maturity},
    {min_vol, max_vol},
    {min_rate, max_rate},
}};
auto scaled = scale_lhs_samples(final_samples, final_bounds);

double final_max_error = 0.0;
size_t valid = 0;

for (const auto& sample : scaled) {
    double m = sample[0], tau = sample[1], sigma = sample[2], rate = sample[3];
    double strike = config.spot / m;

    double interp = surface->price(config.spot, strike, tau, sigma, rate);

    PricingParams params;
    params.spot = config.spot;
    params.strike = strike;
    params.maturity = tau;
    params.rate = rate;
    params.dividend_yield = config.dividend_yield;
    params.option_type = config.option_type;
    params.volatility = sigma;
    params.discrete_dividends = config.discrete_dividends;

    auto fd = solve_american_option(params);
    if (!fd.has_value()) continue;

    auto err = compute_error_metric(
        interp, fd->value(), config.spot, strike, tau,
        sigma, rate, config.dividend_yield);
    if (!err.has_value()) continue;

    final_max_error = std::max(final_max_error, *err);
    valid++;
}

if (valid > 0 && final_max_error > params_.target_iv_error) {
    // Bump grids by one refinement step and rebuild (one retry)
    size_t bumped_m = std::min(max_m_size + 2, params_.max_points_per_dim);
    size_t bumped_v = std::min(max_v_size + 1, params_.max_points_per_dim);
    size_t bumped_r = std::min(max_r_size + 1, params_.max_points_per_dim);
    int bumped_tau = std::min(max_tau_pts + 2,
        static_cast<int>(params_.max_points_per_dim));

    auto retry_m = linspace(expanded_min_m, max_m, bumped_m);
    auto retry_v = linspace(min_vol, max_vol, bumped_v);
    auto retry_r = linspace(min_rate, max_rate, bumped_r);

    SegmentedMultiKRefBuilder::Config retry_config = mkref_config;
    retry_config.moneyness_grid = retry_m;
    retry_config.vol_grid = retry_v;
    retry_config.rate_grid = retry_r;

    auto retry_surface = SegmentedMultiKRefBuilder::build(retry_config);
    if (retry_surface.has_value()) {
        return std::move(*retry_surface);
    }
}

return std::move(*surface);
```

**Step 3: Run all tests**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all`
Expected: ALL PASS

**Step 4: Full regression**

Run: `bazel test //...`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/option/table/adaptive_grid_builder.cpp \
        tests/adaptive_grid_builder_test.cc
git commit -m "Add final multi-K_ref validation and edge case tests"
```

---

## Verification Checklist

After all tasks are complete:

- [ ] `bazel test //...` — all tests pass
- [ ] `bazel build //benchmarks/...` — benchmarks compile
- [ ] `bazel build //src/python:mango_option` — Python bindings compile
- [ ] `AdaptiveGrid` + `SegmentedIVPath` builds and solves IV
- [ ] Existing `AdaptiveGrid` + `StandardIVPath` unchanged
- [ ] Existing `ManualGrid` + `SegmentedIVPath` unchanged
- [ ] No new compiler warnings
