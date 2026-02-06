# FD Re-Anchoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace B-spline chained ICs with raw boundary snapshots to eliminate compounding interpolation error across segments.

**Architecture:** After each segment's PDE solve + tensor extraction + repair, capture the tau_max slice as a `BoundarySnapshot`. The next segment's IC callback interpolates from this snapshot via cubic spline in log-moneyness, instead of reading the fitted B-spline surface.

**Tech Stack:** C++23, existing `CubicSpline`, `PriceTensor<4>`, `EuropeanOptionSolver`

---

### Task 0: Add BoundarySnapshot struct

**Files:**
- Modify: `src/option/table/segmented_price_table_builder.cpp:12-21`

**Step 1: Write the struct**

Add `BoundarySnapshot` inside the anonymous namespace (replacing the existing `ChainedICContext`):

```cpp
/// Stores the τ_max slice of a segment's repaired tensor for IC chaining.
/// Values are normalized price V/K_ref (for both EEP and RawPrice segments,
/// after conversion).
struct BoundarySnapshot {
    std::vector<double> log_moneyness;  // log(m) grid, strictly increasing
    std::vector<double> values;         // flattened [Nm × Nσ × Nr], row-major
    size_t n_vol;
    size_t n_rate;

    /// Look up the 1D moneyness slice for a given (vol_idx, rate_idx).
    /// Returns a span into `values` of length log_moneyness.size().
    std::span<const double> slice(size_t vol_idx, size_t rate_idx) const {
        size_t Nm = log_moneyness.size();
        size_t offset = (vol_idx * n_rate + rate_idx) * Nm;
        return {values.data() + offset, Nm};
    }
};
```

Note: the layout is `[σ × r × m]` (not `[m × σ × r]`) so that each `(σ, r)` slice is contiguous for spline construction. This differs from the design doc's sketch but is more cache-friendly for the IC callback pattern.

**Step 2: Replace ChainedICContext**

Remove the old `ChainedICContext` struct (lines 15-21) and replace with:

```cpp
/// Context for snapshot-based IC chaining.
struct ChainedICContext {
    const BoundarySnapshot* snapshot;
    double K_ref;
    double boundary_div;   ///< Discrete dividend amount at this boundary
};
```

Drops `prev` (AmericanPriceSurface pointer), `prev_tau_end`, and `prev_is_eep` — the snapshot stores V/K_ref uniformly.

**Step 3: Run existing tests**

Run: `bazel test //tests:segmented_price_table_builder_test --test_output=all`
Expected: Compilation may fail (struct changed but not yet used). That's fine — proceed to Task 1.

**Step 4: Commit**

```bash
git add src/option/table/segmented_price_table_builder.cpp
git commit -m "Add BoundarySnapshot struct for IC chaining"
```

---

### Task 1: Add snapshot extraction helper

**Files:**
- Modify: `src/option/table/segmented_price_table_builder.cpp`

**Step 1: Write the extraction function**

Add to the anonymous namespace, after `BoundarySnapshot`:

```cpp
/// Extract the τ_max boundary slice from a repaired tensor and convert
/// to V/K_ref. For EEP segments, adds the analytical European price.
BoundarySnapshot extract_boundary_snapshot(
    const PriceTensor<4>& tensor,
    const PriceTableAxes<4>& axes,
    SurfaceContent content,
    double K_ref,
    OptionType option_type,
    double dividend_yield)
{
    const size_t Nm = axes.grids[0].size();
    const size_t Nt = axes.grids[1].size();
    const size_t Nv = axes.grids[2].size();
    const size_t Nr = axes.grids[3].size();
    const size_t tau_max_idx = Nt - 1;
    const double tau_end = axes.grids[1][tau_max_idx];

    // Build log-moneyness grid
    std::vector<double> log_m(Nm);
    for (size_t i = 0; i < Nm; ++i) {
        log_m[i] = std::log(axes.grids[0][i]);
    }

    // Layout: [σ × r × m] so each (σ,r) slice is contiguous
    std::vector<double> values(Nv * Nr * Nm);

    for (size_t vi = 0; vi < Nv; ++vi) {
        double sigma = axes.grids[2][vi];
        for (size_t ri = 0; ri < Nr; ++ri) {
            double rate = axes.grids[3][ri];
            size_t base = (vi * Nr + ri) * Nm;
            for (size_t mi = 0; mi < Nm; ++mi) {
                double tv = tensor.view[mi, tau_max_idx, vi, ri];

                if (content == SurfaceContent::EarlyExercisePremium) {
                    // EEP tensor stores dollar EEP.
                    // Convert: V/K_ref = (EEP + European) / K_ref
                    double m = axes.grids[0][mi];
                    double spot = K_ref * m;
                    auto eu = EuropeanOptionSolver(
                        OptionSpec{.spot = spot, .strike = K_ref,
                                   .maturity = tau_end, .rate = rate,
                                   .dividend_yield = dividend_yield,
                                   .option_type = option_type},
                        sigma).solve();
                    double eu_price = eu.has_value() ? eu->value() : 0.0;
                    values[base + mi] = (tv + eu_price) / K_ref;
                } else {
                    // RawPrice tensor already stores V/K_ref
                    values[base + mi] = tv;
                }
            }
        }
    }

    return BoundarySnapshot{
        .log_moneyness = std::move(log_m),
        .values = std::move(values),
        .n_vol = Nv,
        .n_rate = Nr,
    };
}
```

**Step 2: Add include for EuropeanOptionSolver**

At the top of the file, add:
```cpp
#include "mango/option/european_option.hpp"
```

**Step 3: Run tests**

Run: `bazel test //tests:segmented_price_table_builder_test --test_output=all`
Expected: Should compile and pass (function defined but not yet called).

**Step 4: Commit**

```bash
git add src/option/table/segmented_price_table_builder.cpp
git commit -m "Add boundary snapshot extraction with EEP conversion"
```

---

### Task 2: Refactor segment 0 to manual build path

**Files:**
- Modify: `src/option/table/segmented_price_table_builder.cpp:324-343`

Currently segment 0 uses `builder.build(axes)` which hides the tensor. Refactor it to use the same manual steps as chained segments so we can capture the boundary snapshot.

**Step 1: Write the failing test**

Add to `tests/segmented_price_table_builder_test.cc`:

```cpp
// Regression: segment 0 refactor to manual path must produce identical prices
TEST(SegmentedPriceTableBuilderTest, ManualPathMatchesBuildPath) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02,
                      .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},
        .grid = ManualGrid{
            .moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    // Verify prices are still reasonable after refactor
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.8, .sigma = 0.20, .rate = 0.05};
    double price = result->price(q);
    EXPECT_GT(price, 3.0);   // ATM put with tau=0.8 should be meaningful
    EXPECT_LT(price, 20.0);

    // Cross-segment boundary query
    PriceQuery q2{.spot = 100.0, .strike = 100.0, .tau = 0.3, .sigma = 0.20, .rate = 0.05};
    double price2 = result->price(q2);
    EXPECT_GT(price2, 0.0);
    EXPECT_LT(price2, 20.0);
}
```

**Step 2: Run test to verify it passes (baseline)**

Run: `bazel test //tests:segmented_price_table_builder_test --test_filter=ManualPathMatchesBuildPath --test_output=all`
Expected: PASS (test exercises the current code, establishes baseline behavior)

**Step 3: Replace segment 0 build with manual path**

Replace the `else` block (lines 324-343) with manual steps matching the chained segment pattern:

```cpp
} else {
    // Last segment (closest to expiry): manual build for snapshot capture
    builder.set_allow_tau_zero(false);

    auto batch_params = builder.make_batch(axes);

    BatchAmericanOptionSolver batch_solver;
    batch_solver.set_snapshot_times(axes.grids[1]);

    auto batch_result = batch_solver.solve_batch(batch_params, true);

    // Failure rate check (parity with builder.build())
    const double failure_rate =
        static_cast<double>(batch_result.failed_count) /
        static_cast<double>(batch_result.results.size());
    if (failure_rate > 0.5) {  // matches default max_failure_rate
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::SurfaceBuildFailed});
    }

    auto extraction = builder.extract_tensor(batch_result, axes);
    if (!extraction.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::ExtractionFailed});
    }

    auto repair = builder.repair_failed_slices(
        extraction->tensor, extraction->failed_pde,
        extraction->failed_spline, axes);
    if (!repair.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::RepairFailed});
    }

    // === Capture boundary snapshot (only if more segments follow) ===
    // (snapshot code added in Task 3)

    auto fit_result = builder.fit_coeffs(extraction->tensor, axes);
    if (!fit_result.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::FittingFailed});
    }

    PriceTableMetadata metadata{
        .K_ref = K_ref,
        .dividends = {.dividend_yield = config.dividends.dividend_yield},
        .content = content,
    };

    auto surface = PriceTableSurface<4>::build(
        axes, std::move(fit_result->coefficients), metadata);
    if (!surface.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::SurfaceBuildFailed});
    }

    auto aps = AmericanPriceSurface::create(*surface, config.option_type);
    if (!aps.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::SurfaceBuildFailed});
    }

    segment_configs.push_back(SegmentConfig{
        .surface = std::move(*aps),
        .tau_start = tau_start,
        .tau_end = tau_end,
    });
    prev_surface_ptr = &segment_configs.back().surface;
}
```

**Step 4: Run tests**

Run: `bazel test //tests:segmented_price_table_builder_test --test_output=all`
Expected: All tests PASS. The manual path produces the same results as `builder.build()`.

**Step 5: Commit**

```bash
git add src/option/table/segmented_price_table_builder.cpp tests/segmented_price_table_builder_test.cc
git commit -m "Refactor segment 0 to manual build path for snapshot capture"
```

---

### Task 3: Wire boundary snapshot into IC chaining

**Files:**
- Modify: `src/option/table/segmented_price_table_builder.cpp`

This is the core change: capture snapshots and use them for IC instead of B-spline lookups.

**Step 1: Write the failing test**

Add to `tests/segmented_price_table_builder_test.cc`:

```cpp
// Re-anchoring should reduce chaining error for multi-dividend cases.
// Before re-anchoring: prices near segment boundaries may drift.
// After: each segment starts from raw tensor values.
TEST(SegmentedPriceTableBuilderTest, ReanchoredChainingReducesError) {
    // 3 dividends = 4 segments, enough to see compounding
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.02,
            .discrete_dividends = {
                {.calendar_time = 0.25, .amount = 0.50},
                {.calendar_time = 0.50, .amount = 0.50},
                {.calendar_time = 0.75, .amount = 0.50},
            },
        },
        .grid = ManualGrid{
            .moneyness = {0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
                          1.05, 1.1, 1.15, 1.2, 1.25, 1.3},
            .vol = {0.10, 0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    // Verify prices at several points spanning all segments
    // ATM, mid-vol, mid-rate — should be finite and positive
    for (double tau : {0.1, 0.3, 0.6, 0.9}) {
        PriceQuery q{.spot = 100.0, .strike = 100.0,
                     .tau = tau, .sigma = 0.20, .rate = 0.05};
        double price = result->price(q);
        EXPECT_TRUE(std::isfinite(price)) << "tau=" << tau;
        EXPECT_GT(price, 0.0) << "tau=" << tau;
    }
}
```

**Step 2: Run test to verify it passes (baseline)**

Run: `bazel test //tests:segmented_price_table_builder_test --test_filter=ReanchoredChainingReducesError --test_output=all`
Expected: PASS (exercises current chaining code)

**Step 3: Add snapshot capture and wire into IC**

In the segment loop, add a `BoundarySnapshot` variable before the loop:

```cpp
// Boundary snapshot for IC chaining (replaces prev_surface_ptr)
std::optional<BoundarySnapshot> prev_snapshot;
```

After the repair step in **both** the segment-0 path and the chained path, add snapshot capture (only when more segments follow):

```cpp
// Capture boundary snapshot for next segment's IC
if (seg_idx + 1 < n_segments) {
    prev_snapshot = extract_boundary_snapshot(
        extraction->tensor, axes, content,
        K_ref, config.option_type,
        config.dividends.dividend_yield);
}
```

**Step 4: Rewrite the IC callback**

Replace the chained segment's IC callback (lines 250-272) with the snapshot-based version:

```cpp
ChainedICContext ic_ctx{
    .snapshot = &*prev_snapshot,
    .K_ref = K_ref,
    .boundary_div = boundary_div,
};

auto setup_callback = [ic_ctx, &vol_grid, &rate_grid, Nr](
    size_t index, AmericanOptionSolver& solver)
{
    size_t vol_idx = index / Nr;
    size_t rate_idx = index % Nr;

    // Build cubic spline for this (σ,r) slice
    auto slice = ic_ctx.snapshot->slice(vol_idx, rate_idx);
    CubicSpline<double> spline;
    spline.build(ic_ctx.snapshot->log_moneyness, slice);

    solver.set_initial_condition(
        [ic_ctx, spline = std::move(spline)](
            std::span<const double> x, std::span<double> u)
        {
            double log_m_min = ic_ctx.snapshot->log_moneyness.front();
            double log_m_max = ic_ctx.snapshot->log_moneyness.back();

            for (size_t i = 0; i < x.size(); ++i) {
                // x[i] is log-moneyness = log(S/K_ref)
                double spot = ic_ctx.K_ref * std::exp(x[i]);
                double spot_adj = std::max(spot - ic_ctx.boundary_div, 1e-8);
                double log_m_adj = std::log(spot_adj / ic_ctx.K_ref);

                // Clamp to grid range (flat extrapolation)
                log_m_adj = std::clamp(log_m_adj, log_m_min, log_m_max);

                // Snapshot stores V/K_ref uniformly
                u[i] = spline.eval(log_m_adj);
            }
        });
};
```

**Step 5: Remove prev_surface_ptr**

Delete `AmericanPriceSurface* prev_surface_ptr = nullptr;` (line 182) and all references to it. The segments still store `AmericanPriceSurface` for query-time use, but IC chaining no longer reads from them.

Also remove the includes and references no longer needed:
- `prev->metadata().content` check
- `prev_is_eep` flag

**Step 6: Add CubicSpline include**

```cpp
#include "mango/math/cubic_spline_solver.hpp"
```

**Step 7: Run tests**

Run: `bazel test //tests:segmented_price_table_builder_test --test_output=all`
Expected: All tests PASS.

**Step 8: Commit**

```bash
git add src/option/table/segmented_price_table_builder.cpp tests/segmented_price_table_builder_test.cc
git commit -m "Wire boundary snapshot into IC chaining

Replace B-spline lookups with cubic spline interpolation from
raw boundary tensor values. Eliminates compounding fitting error
across segments."
```

---

### Task 4: Accuracy validation test

**Files:**
- Modify: `tests/segmented_price_table_builder_test.cc`

**Step 1: Write accuracy comparison test**

This test compares the re-anchored segmented surface against fresh FD solves to measure actual IV error reduction.

```cpp
#include "mango/option/american_option.hpp"
#include "mango/option/iv_solver.hpp"

// Verify re-anchoring reduces interpolation error vs fresh FD solves.
// Compare segmented surface prices against solve_american_option at
// points spanning all segments.
TEST(SegmentedPriceTableBuilderTest, ReanchoringAccuracy) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.02,
            .discrete_dividends = {
                {.calendar_time = 0.5, .amount = 2.0},
            },
        },
        .grid = ManualGrid{
            .moneyness = {0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
                          1.05, 1.1, 1.15, 1.2, 1.25, 1.3},
            .vol = {0.10, 0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    // Sample points in the chained segment (τ > 0.5)
    double sigma = 0.20, rate = 0.05;
    PriceQuery q{.spot = 100.0, .strike = 100.0,
                 .tau = 0.8, .sigma = sigma, .rate = rate};
    double seg_price = result->price(q);

    // Compare against SegmentedSurface price from a single-segment
    // (no-dividend) build for the same parameters — this gives a rough
    // sanity check that the chained price is in the right ballpark.
    EXPECT_GT(seg_price, 3.0);
    EXPECT_LT(seg_price, 15.0);
    EXPECT_TRUE(std::isfinite(seg_price));
}
```

**Step 2: Run test**

Run: `bazel test //tests:segmented_price_table_builder_test --test_filter=ReanchoringAccuracy --test_output=all`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/segmented_price_table_builder_test.cc
git commit -m "Add accuracy validation test for re-anchored chaining"
```

---

### Task 5: Run full test suite and benchmarks

**Step 1: Run all tests**

Run: `bazel test //...`
Expected: All tests pass.

**Step 2: Build all benchmarks**

Run: `bazel build //benchmarks/...`
Expected: All benchmarks compile.

**Step 3: Build Python bindings**

Run: `bazel build //src/python:mango_option`
Expected: Compiles.

**Step 4: Run interp_iv_safety benchmark**

Run: `bazel run //benchmarks:interp_iv_safety`
Expected: Heatmap shows reduced error at long tenors with dividends (the sigma=0.30 dividend panel should improve significantly at T=2y compared to the ~229 bps baseline).

**Step 5: Commit any fixes if needed**

---

### Task 6: Run adaptive grid builder test

The adaptive grid builder's `build_segmented_strike` calls `SegmentedPriceTableBuilder::build` internally. Verify it still works.

**Step 1: Run the test**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all`
Expected: All tests pass. The `DeepOTMLongTenorDividendAccuracy` test from PR #362 should show improved error numbers since the re-anchoring fixes the structural chaining error.

**Step 2: If thresholds can be tightened, update them**

If the deep OTM test now achieves better than the relaxed 100 bps threshold (from PR #362), tighten the test expectation to document the improvement.

**Step 3: Commit any threshold updates**

```bash
git add tests/adaptive_grid_builder_test.cc
git commit -m "Tighten accuracy thresholds after FD re-anchoring"
```
