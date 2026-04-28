# Remove Per-Strike Surface and Segmented EEP

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove per-strike surface strategy and EEP decomposition from the segmented price table builder. Rename `RawPrice` → `NormalizedPrice`.

**Architecture:** Multi-K_ref is the sole segmented strategy (benchmarked at 23 bps vs 50 bps RMS for per-strike at σ=15%). All segments use NormalizedPrice (V/K_ref) uniformly, eliminating the EEP/RawPrice representation mismatch at segment boundaries (benchmarked at 2.7 bps vs 5.1 bps for σ=30%).

**Tech Stack:** C++23, Bazel, GoogleTest

**Benchmark evidence (from `interp_iv_safety`):**

| Metric | Per-strike + EEP | Multi-K_ref + EEP | Multi-K_ref − EEP |
|--------|-----------------|-------------------|-------------------|
| σ=15% RMS (bps) | 50.4 | 23.0 | — |
| σ=30% RMS (bps) | 5.1 | 4.9 | 2.7 |
| Build time (8 mat) | 27.6s | 35.7s | — |
| Query latency | 170 µs | 289 µs | — |

---

### Task 1: Rename `RawPrice` → `NormalizedPrice`

Do this first so subsequent tasks use the correct name.

**Files:**
- Modify: `src/option/table/price_table_metadata.hpp:12`
- Modify: `src/option/table/price_table_workspace.hpp:196` (comment only)
- Modify: `src/option/table/price_table_builder.cpp:435-438`
- Modify: `src/option/table/american_price_surface.hpp:23` (comment)
- Modify: `src/option/table/american_price_surface.cpp:31,38,56-57,97-98`
- Modify: `src/option/table/segmented_price_table_builder.hpp:18,57`
- Modify: `src/option/table/segmented_price_table_builder.cpp:275,348`
- Modify: `src/option/table/spliced_surface.hpp:418-419,431`
- Modify: `src/option/table/spliced_surface_builder.cpp:40` (content push)
- Modify: `src/python/mango_bindings.cpp:702`

**Step 1: Rename the enum value**

In `src/option/table/price_table_metadata.hpp:12`, change:
```cpp
    NormalizedPrice = 0,           ///< V/K_ref as function of log-moneyness
```

**Step 2: Rename all references**

Search-and-replace `RawPrice` → `NormalizedPrice` across every file listed above. Also update comments that say "RawPrice" or "Raw American option prices" to say "NormalizedPrice" or "V/K_ref".

Key locations:
- `price_table_builder.cpp:435-436`: Change `SurfaceContent::RawPrice` and comment
- `american_price_surface.cpp:56-57`: Change enum check and assert message
- `american_price_surface.cpp:97-98`: Change enum check and comment
- `spliced_surface.hpp:419,431`: Change enum checks
- `segmented_price_table_builder.cpp:275,348`: Change enum value and comment
- `mango_bindings.cpp:702`: Change Python binding `.value("NormalizedPrice", ...)`

**Step 3: Build and test**

Run: `bazel test //...`
Expected: All 116 tests pass (pure rename, no behavior change)

**Step 4: Commit**

```bash
git add -u
git commit -m "Rename RawPrice to NormalizedPrice

The enum value stores V/K_ref (PDE-normalized output), not raw
dollar prices. The new name reflects what the B-spline actually
stores."
```

---

### Task 2: Hardcode NormalizedPrice in SegmentedPriceTableBuilder

Remove EEP decomposition from the segmented path. All segments store V/K_ref uniformly.

**Files:**
- Modify: `src/option/table/segmented_price_table_builder.cpp:17-23,265,273-275,326-327,348-349`
- Modify: `src/option/table/segmented_price_table_builder.hpp:17-19,56-57`

**Step 1: Remove `is_last_segment` and hardcode content**

In `segmented_price_table_builder.cpp`, change lines 265 and 273-275:

Before:
```cpp
        bool is_last_segment = (seg_idx == 0);
        ...
        SurfaceContent content = is_last_segment
            ? SurfaceContent::EarlyExercisePremium
            : SurfaceContent::RawPrice;
```

After:
```cpp
        SurfaceContent content = SurfaceContent::NormalizedPrice;
```

Delete the `is_last_segment` variable entirely (line 265). It's also used at lines 269, 288, 308, 360, 363 — check each:
- Line 269: `tau_points_per_segment` arg — keep using `seg_idx == 0` inline if needed
- Line 288: `builder.set_surface_content(content)` — unchanged
- Line 308: `if (!is_last_segment)` — change to `if (seg_idx > 0)` (chained segments)
- Line 360: strict mode check — `seg_idx == 0` for strict
- Line 363: failure rate — `seg_idx == 0` for strict

Replace all uses of `is_last_segment` with `(seg_idx == 0)` inline.

**Step 2: Remove `prev_is_eep` from ChainedICContext**

In `segmented_price_table_builder.cpp`, the anonymous struct at lines 17-23:

Before:
```cpp
struct ChainedICContext {
    const AmericanPriceSurface* prev;
    double K_ref;
    double prev_tau_end;
    double boundary_div;
    bool prev_is_eep;
};
```

After:
```cpp
struct ChainedICContext {
    const AmericanPriceSurface* prev;
    double K_ref;
    double prev_tau_end;
    double boundary_div;
};
```

Remove the `.prev_is_eep` initializer at lines 326-327.

**Step 3: Simplify IC handoff**

In `segmented_price_table_builder.cpp` line 348-349:

Before:
```cpp
                            // EEP returns actual price V; RawPrice returns V/K_ref
                            u[i] = ic_ctx.prev_is_eep ? raw / ic_ctx.K_ref : raw;
```

After:
```cpp
                            u[i] = raw;
```

The previous surface is always NormalizedPrice, so `AmericanPriceSurface::price()` returns `surface_->value(...)` which is V/K_ref. No division needed.

**Step 4: Update header comments**

In `segmented_price_table_builder.hpp`, update lines 17-19 and 56-57:

Before:
```
/// segment normally (EEP, payoff IC), and chains earlier segments backward
/// using RawPrice mode with initial conditions sourced from the previous
```

After:
```
/// All segments use NormalizedPrice mode (V/K_ref). The last segment uses
/// payoff IC; earlier segments chain from the previous segment's surface.
```

Before:
```
///   3. Build last segment (closest to expiry) with EEP and payoff IC.
///   4. Build earlier segments backward with RawPrice and chained IC.
```

After:
```
///   3. Build last segment (closest to expiry) with payoff IC.
///   4. Build earlier segments backward with chained IC.
```

**Step 5: Build and test**

Run: `bazel test //...`
Expected: All tests pass. Segmented surfaces now use uniform NormalizedPrice.

**Step 6: Commit**

```bash
git add -u
git commit -m "Remove EEP from segmented price table builder

All segments now use NormalizedPrice (V/K_ref) uniformly.
EEP decomposition in only the last segment created a
representation mismatch at segment boundaries, worsening
accuracy (5.1 → 2.7 bps RMS at σ=30%)."
```

---

### Task 3: Remove per-strike from factory and AnyIVSolver

Remove the `use_per_strike` branch, `strike_grid` field, and StrikeSurfaceWrapper variant.

**Files:**
- Modify: `src/option/interpolated_iv_solver.hpp:163,192,195-199`
- Modify: `src/option/interpolated_iv_solver.cpp:27,135-137,247-270,297-338,348-407`

**Step 1: Remove `strike_grid` from SegmentedIVPath**

In `interpolated_iv_solver.hpp:163`, delete:
```cpp
    std::vector<double> strike_grid;  ///< optional explicit strikes for per-strike surfaces
```

**Step 2: Remove StrikeSurfaceWrapper from AnyIVSolver**

In `interpolated_iv_solver.hpp`, delete:
- Line 192: `explicit AnyIVSolver(InterpolatedIVSolver<StrikeSurfaceWrapper<>> solver);`
- Lines 195-199: Remove `StrikeSurfaceWrapper<>` from the variant:

Before:
```cpp
    using SolverVariant = std::variant<
        InterpolatedIVSolver<AmericanPriceSurface>,
        InterpolatedIVSolver<MultiKRefSurfaceWrapper<>>,
        InterpolatedIVSolver<StrikeSurfaceWrapper<>>
    >;
```

After:
```cpp
    using SolverVariant = std::variant<
        InterpolatedIVSolver<AmericanPriceSurface>,
        InterpolatedIVSolver<MultiKRefSurfaceWrapper<>>
    >;
```

**Step 3: Remove per-strike code from .cpp**

In `interpolated_iv_solver.cpp`, delete:
- Line 27: `template class InterpolatedIVSolver<StrikeSurfaceWrapper<>>;`
- Lines 135-137: `AnyIVSolver::AnyIVSolver(InterpolatedIVSolver<StrikeSurfaceWrapper<>>)` constructor
- Lines 247-270: `wrap_strike_surface()` function
- Lines 297-338: `build_manual_strike_surface()` function

**Step 4: Simplify factory branching**

In `build_segmented()` (interpolated_iv_solver.cpp), remove all `use_per_strike` logic:

Before (lines 346-407):
```cpp
    MultiKRefConfig kref_config = path.kref_config;
    const auto& grid = config.grid;
    const bool use_per_strike = !path.strike_grid.empty();
    if (use_per_strike) {
        kref_config.K_refs = path.strike_grid;
    }
    ...
    if (config.adaptive.has_value()) {
        ...
        if (use_per_strike) {
            // per-strike adaptive branch
        }
        // multi-kref adaptive branch
    }
    // Manual grid path
    if (use_per_strike) {
        // per-strike manual branch
    }
    // multi-kref manual branch
```

After:
```cpp
    const auto& kref_config = path.kref_config;
    const auto& grid = config.grid;
    ...
    if (config.adaptive.has_value()) {
        // multi-kref adaptive branch only
    }
    // multi-kref manual branch only
```

Remove the `use_per_strike` variable, both per-strike branches (adaptive and manual), and the `kref_config.K_refs = path.strike_grid` override.

**Step 5: Build and test**

Run: `bazel test //...`
Expected: All tests pass. Compile errors from `strike_grid` users caught at build time.

**Step 6: Commit**

```bash
git add -u
git commit -m "Remove per-strike surface from IV solver factory

Multi-K_ref is the sole segmented strategy. Per-strike's
exact-strike lookup is less accurate than K_ref interpolation
(50 vs 23 bps RMS at σ=15%)."
```

---

### Task 4: Remove per-strike types and builders

Delete StrikeBracket, StrikeTransform, StrikeSurface, StrikeEntry, and their builders.

**Files:**
- Modify: `src/option/table/spliced_surface.hpp:315-376,470-482,527-533,592-593`
- Modify: `src/option/table/spliced_surface_builder.hpp:40-43,48-50`
- Modify: `src/option/table/spliced_surface_builder.cpp:115-146`
- Modify: `src/option/table/adaptive_grid_types.hpp:118-122`
- Modify: `src/option/table/adaptive_grid_builder.hpp:89-95`
- Modify: `src/option/table/adaptive_grid_builder.cpp:1335-1366`

**Step 1: Remove types from spliced_surface.hpp**

Delete these blocks:
- Lines 315-376: `StrikeBracket` class (entire class definition)
- Lines 470-482: `StrikeTransform` struct (entire struct)
- Lines 527-533: `StrikeSurface` type alias (template + using)
- Lines 592-593: `StrikeSurfaceWrapper` type alias

**Step 2: Remove StrikeEntry and build_strike_surface**

In `spliced_surface_builder.hpp`, delete:
- Lines 40-43: `StrikeEntry` struct
- Lines 48-50: `build_strike_surface()` declaration

In `spliced_surface_builder.cpp`, delete:
- Lines 115-146: `build_strike_surface()` implementation

**Step 3: Remove StrikeAdaptiveResult and build_segmented_strike**

In `adaptive_grid_types.hpp`, delete:
- Lines 118-122: `StrikeAdaptiveResult` struct

In `adaptive_grid_builder.hpp`, delete:
- Lines 89-95: `build_segmented_strike()` declaration and doc comment

In `adaptive_grid_builder.cpp`, delete:
- Lines 1335-1366: `build_segmented_strike()` implementation

**Step 4: Build and test**

Run: `bazel test //... && bazel build //src/python:mango_option`
Expected: All tests pass. Python bindings compile.

**Step 5: Commit**

```bash
git add -u
git commit -m "Delete per-strike surface types and builders

Remove StrikeBracket, StrikeTransform, StrikeSurface,
StrikeSurfaceWrapper, StrikeEntry, build_strike_surface,
build_segmented_strike, and StrikeAdaptiveResult."
```

---

### Task 5: Update benchmarks

Fix benchmarks that reference removed per-strike types.

**Files:**
- Modify: `benchmarks/interp_iv_safety.cc:165`
- Modify: `benchmarks/iv_interpolation_sweep.cc:388-597`

**Step 1: Fix safety benchmark**

In `interp_iv_safety.cc`, the `build_div_solvers()` function uses `strike_grid` at line 165.
Change from:
```cpp
            .path = SegmentedIVPath{
                .maturity = mat,
                .discrete_dividends = divs,
                .strike_grid = std::vector<double>(kStrikes.begin(), kStrikes.end()),
            },
```

To:
```cpp
            .path = SegmentedIVPath{
                .maturity = mat,
                .discrete_dividends = divs,
                .kref_config = {.K_refs = std::vector<double>(kStrikes.begin(), kStrikes.end())},
            },
```

**Step 2: Remove per-strike benchmark from sweep**

In `iv_interpolation_sweep.cc`, delete the entire per-strike segmented benchmark:
- Lines 388-395: `SegmentedSolverEntry` struct
- Lines 397-529: `get_segmented_solver()` function
- Lines 535-596: `BM_Adaptive_IV_Div_Scaled` benchmark function and registration

Also remove unused includes that were only needed for the per-strike path (`spliced_surface_builder.hpp`).

**Step 3: Build and run safety benchmark**

Run: `bazel build //benchmarks/... && bazel run //benchmarks:interp_iv_safety`
Expected: Builds clean. Safety benchmark runs with multi-K_ref for dividend cases.
Verify σ=30% dividend RMS ≈ 2.7 bps (matching our earlier all-RawPrice measurement).

**Step 4: Commit**

```bash
git add -u
git commit -m "Update benchmarks for multi-K_ref only path"
```

---

### Task 6: Final verification

**Step 1: Full CI check**

```bash
bazel test //...
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

Expected: All tests pass, all benchmarks compile, Python bindings compile.

**Step 2: Run safety benchmark for accuracy verification**

```bash
bazel run //benchmarks:interp_iv_safety
```

Expected: Dividend cases show NormalizedPrice accuracy (σ=30% RMS ≈ 2.7 bps).

**Step 3: Verify no leftover references**

Search for orphaned references:
```
grep -r "RawPrice\|StrikeSurface\|StrikeBracket\|StrikeTransform\|StrikeEntry\|build_strike_surface\|build_segmented_strike\|StrikeAdaptiveResult\|strike_grid\|prev_is_eep\|is_last_segment" src/ tests/ benchmarks/ --include='*.cpp' --include='*.hpp' --include='*.cc' --include='*.h'
```

Expected: No matches in source files (docs/plans may still reference these for historical context).
