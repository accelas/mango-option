# IV Benchmark Split & Adaptive Baseline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split iv_strike_sweep into FDM and interpolation benchmarks, replace hand-picked interpolation grid with AdaptiveGridBuilder baseline, add discrete dividend interpolation benchmark.

**Architecture:** AdaptiveGridBuilder calibrates the base interpolation grid once (2 bps target). Scaled variants (2x/4x/8x) refine those axes by inserting midpoints, then rebuild with PriceTableBuilder. Both standard (continuous div) and segmented (discrete div) paths are benchmarked. Exposing `SegmentedAdaptiveResult` from the adaptive builder enables extracting the base grid for scaling.

**Tech Stack:** C++23, Bazel, Google Benchmark, AdaptiveGridBuilder, PriceTableBuilder, SegmentedPriceTableBuilder. Reference prices use mango's own high-res FDM (not QuantLib) so IV error purely measures interpolation error.

---

### Task 1: Expose grid info from `build_segmented()` return type

**Files:**
- Modify: `src/option/table/adaptive_grid_types.hpp`
- Modify: `src/option/table/adaptive_grid_builder.hpp:76-84`
- Modify: `src/option/table/adaptive_grid_builder.cpp:1187-1307` and `1310-end`
- Modify: `src/option/iv_solver_factory.cpp:249-250` and `214-216`
- Modify: `tests/adaptive_grid_builder_test.cc` (14 call sites)

**Step 1: Add result structs to `adaptive_grid_types.hpp`**

Add before the closing `}  // namespace mango`:

```cpp
/// Result from adaptive segmented multi-K_ref build.
/// Exposes the chosen grid so callers can scale it for convergence studies.
struct SegmentedAdaptiveResult {
    MultiKRefSurface<> surface;
    ManualGrid grid;              ///< Grid axes adaptive chose
    int tau_points_per_segment;   ///< Tau density per segment
};

/// Result from adaptive segmented per-strike build.
struct StrikeAdaptiveResult {
    StrikeSurface<> surface;
    ManualGrid grid;
    int tau_points_per_segment;
};
```

Requires forward-declaring or including `spliced_surface.hpp`. Check if `adaptive_grid_types.hpp` already includes what's needed — it currently includes `price_table_surface.hpp` but not `spliced_surface.hpp`. Add the include.

**Step 2: Update `adaptive_grid_builder.hpp` return types**

Change:

```cpp
[[nodiscard]] std::expected<MultiKRefSurface<>, PriceTableError>
build_segmented(const SegmentedAdaptiveConfig& config,
                const ManualGrid& domain);

[[nodiscard]] std::expected<StrikeSurface<>, PriceTableError>
build_segmented_strike(const SegmentedAdaptiveConfig& config,
                       const std::vector<double>& strike_grid,
                       const ManualGrid& domain);
```

To:

```cpp
[[nodiscard]] std::expected<SegmentedAdaptiveResult, PriceTableError>
build_segmented(const SegmentedAdaptiveConfig& config,
                const ManualGrid& domain);

[[nodiscard]] std::expected<StrikeAdaptiveResult, PriceTableError>
build_segmented_strike(const SegmentedAdaptiveConfig& config,
                       const std::vector<double>& strike_grid,
                       const ManualGrid& domain);
```

**Step 3: Update `adaptive_grid_builder.cpp` implementations**

In `build_segmented()` (line 1187): change return type, wrap final returns:

At line ~1300 (retry path success):
```cpp
// Was: return std::move(*retry_surface);
return SegmentedAdaptiveResult{
    .surface = std::move(*retry_surface),
    .grid = {retry_m, retry_v, retry_r},
    .tau_points_per_segment = bumped_tau,
};
```

At line ~1306 (normal path):
```cpp
// Was: return std::move(*surface);
return SegmentedAdaptiveResult{
    .surface = std::move(*surface),
    .grid = build.seg_template.grid,
    .tau_points_per_segment = build.seg_template.tau_points_per_segment,
};
```

Apply equivalent changes to `build_segmented_strike()` (line 1310+).

**Step 4: Update `iv_solver_factory.cpp` callers**

Line 249-250 (adaptive multi-K_ref):
```cpp
// Was: auto surface = builder.build_segmented(...)
auto seg_result = builder.build_segmented(
    seg_config, {grid.moneyness, grid.vol, grid.rate});
if (!seg_result.has_value()) { ... }
// Was: surface->price(...)
// Now: seg_result->surface.price(...)
```

Use `std::move(seg_result->surface)` where the surface is passed to `MultiKRefSurfaceWrapper`. Apply same pattern at line 214 for `build_segmented_strike`.

**Step 5: Update test callers**

All 14 call sites in `tests/adaptive_grid_builder_test.cc` follow this pattern:

```cpp
auto result = builder.build_segmented(seg_config, {m, v, r});
ASSERT_TRUE(result.has_value());
double price = result->price(query);  // Was direct on MultiKRefSurface
```

Change to:

```cpp
auto result = builder.build_segmented(seg_config, {m, v, r});
ASSERT_TRUE(result.has_value());
double price = result->surface.price(query);  // Access via .surface
```

Apply `->surface.` prefix to all `price()`, `vega()`, `num_slices()` calls on the result.

**Step 6: Build and test**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all`
Expected: All tests pass

Run: `bazel test //...`
Expected: All 117 tests pass

**Step 7: Commit**

```bash
git add src/option/table/adaptive_grid_types.hpp \
        src/option/table/adaptive_grid_builder.hpp \
        src/option/table/adaptive_grid_builder.cpp \
        src/option/iv_solver_factory.cpp \
        tests/adaptive_grid_builder_test.cc
git commit -m "Expose grid info from adaptive segmented builder"
```

---

### Task 2: Rename `iv_strike_sweep` → `iv_fdm_sweep`, remove interpolation code

**Files:**
- Rename: `benchmarks/iv_strike_sweep.cc` → `benchmarks/iv_fdm_sweep.cc`
- Modify: `benchmarks/BUILD.bazel`

**Step 1: Rename file**

```bash
cd /home/kai/work/mango-option/.worktrees/bench-iv-sweep
git mv benchmarks/iv_strike_sweep.cc benchmarks/iv_fdm_sweep.cc
```

**Step 2: Remove interpolation code from `iv_fdm_sweep.cc`**

Remove these sections (keeping all FDM benchmarks intact):

- `InterpSolverEntry` struct (~line 704-708)
- `get_interp_solver()` function (~line 710-769)
- `BM_Interp_IV_Scaled` function and its BENCHMARK registration (~line 771-814)

Also remove includes only needed by interpolation:
- `#include "mango/option/interpolated_iv_solver.hpp"`
- `#include "mango/option/table/price_table_builder.hpp"`
- `#include "mango/option/table/price_table_surface.hpp"`
- `#include "mango/option/table/american_price_surface.hpp"`

Keep all other includes (QuantLib, benchmark, mango american_option, iv_solver, etc.).

**Step 3: Update `benchmarks/BUILD.bazel`**

Rename target `iv_strike_sweep` → `iv_fdm_sweep`. Update `srcs`. Remove deps no longer needed:

Remove from deps:
- `//src/option/table:price_table_builder`
- `//src/option/table:price_table_surface`
- `//src/option/table:american_price_surface`
- `//src/math:bspline_nd_separable`

Keep:
- `//src/option:american_option`
- `//src/option:iv_solver`
- `//src/option:interpolated_iv_solver` — check if still used. `IVSolver` is in `iv_solver.hpp`, not `interpolated_iv_solver.hpp`. Remove if unused after cleanup.
- `@google_benchmark//:benchmark`

**Step 4: Build**

Run: `bazel build //benchmarks:iv_fdm_sweep`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add benchmarks/iv_fdm_sweep.cc benchmarks/BUILD.bazel
git rm benchmarks/iv_strike_sweep.cc  # if git mv didn't auto-stage
git commit -m "Rename iv_strike_sweep to iv_fdm_sweep, remove interpolation"
```

---

### Task 3: Create `iv_interpolation_sweep.cc` — standard path

**Files:**
- Create: `benchmarks/iv_interpolation_sweep.cc`
- Modify: `benchmarks/BUILD.bazel`

**Step 1: Add BUILD target**

Add to `benchmarks/BUILD.bazel`:

```python
cc_binary(
    name = "iv_interpolation_sweep",
    srcs = ["iv_interpolation_sweep.cc"],
    copts = [
        "-Wall",
        "-Wextra",
        "-O3",
        "-march=native",
        "-ftree-vectorize",
        "-fopenmp",
        "-flto",
    ],
    linkopts = [
        "-fopenmp",
        "-flto",
    ],
    linkstatic = True,
    deps = [
        "//src/option:american_option",
        "//src/option:interpolated_iv_solver",
        "//src/option:iv_solver_factory",
        "//src/option/table:adaptive_grid_builder",
        "//src/option/table:american_price_surface",
        "//src/option/table:price_table_builder",
        "//src/option/table:price_table_surface",
        "//src/option/table:segmented_price_table_builder",
        "//src/option/table:spliced_surface_builder",
        "//src/math:bspline_nd_separable",
        "@google_benchmark//:benchmark",
    ],
    tags = ["benchmark", "manual"],
)
```

**Step 2: Create benchmark file with standard path**

Create `benchmarks/iv_interpolation_sweep.cc` with:

1. **Constants:** Same `kSpot`, `kRate`, `kDivYield`, `kScaledVol=0.20`, `kScaledMaturity=1.0`, `kScaledStrikes={80,100,120}` as iv_fdm_sweep.

2. **Reference prices:** Use mango's own high-res FDM via `solve_american_option()` with fine `GridAccuracyParams` (e.g., `GridAccuracyProfile::High` with 4x spatial scaling). This isolates interpolation error from FDM implementation differences. No QuantLib dependency needed for this benchmark.

3. **`refine_axis()` helper:**

```cpp
static std::vector<double> refine_axis(const std::vector<double>& base, int scale) {
    if (scale <= 1 || base.size() < 2) return base;
    size_t n = (base.size() - 1) * static_cast<size_t>(scale) + 1;
    std::vector<double> out(n);
    for (size_t i = 0; i + 1 < base.size(); ++i) {
        for (int j = 0; j < scale; ++j) {
            double t = static_cast<double>(j) / scale;
            out[i * static_cast<size_t>(scale) + static_cast<size_t>(j)] =
                base[i] + t * (base[i + 1] - base[i]);
        }
    }
    out[n - 1] = base.back();
    return out;
}
```

4. **`get_adaptive_solver()` cache:**

```cpp
struct AdaptiveSolverEntry {
    std::unique_ptr<DefaultInterpolatedIVSolver> solver;
    double build_time_ms = 0.0;
    size_t n_pde_solves = 0;
    std::array<size_t, 4> base_grid_sizes = {};
};

static const AdaptiveSolverEntry& get_adaptive_solver(int scale) {
    // Static cache: base_result computed once, scaled variants on demand
    static std::optional<AdaptiveResult> base_result;
    static std::map<int, AdaptiveSolverEntry> cache;

    auto it = cache.find(scale);
    if (it != cache.end()) return it->second;

    auto t0 = std::chrono::steady_clock::now();

    if (!base_result.has_value()) {
        // Calibrate base grid via adaptive builder
        OptionGrid chain;
        chain.spot = kSpot;
        chain.dividend_yield = kDivYield;
        for (double m : {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3})
            chain.strikes.push_back(kSpot / m);
        chain.maturities = {0.25, 0.5, 1.0, 1.5, 2.0};
        chain.implied_vols = {0.05, 0.10, 0.20, 0.30, 0.50};
        chain.rates = {0.01, 0.03, 0.05, 0.10};

        AdaptiveGridParams params;  // defaults: 2 bps target
        GridAccuracyParams accuracy = make_grid_accuracy(GridAccuracyProfile::High);

        AdaptiveGridBuilder builder(params);
        auto result = builder.build(chain, accuracy, OptionType::PUT);
        if (!result.has_value()) {
            std::fprintf(stderr, "AdaptiveGridBuilder::build failed\n");
            std::abort();
        }
        base_result = std::move(*result);
    }

    // For scale==1, use adaptive's surface directly
    // For scale>1, refine axes and rebuild
    std::shared_ptr<const PriceTableSurface<4>> surface;
    size_t pde_solves = 0;

    if (scale == 1) {
        surface = base_result->surface;
        pde_solves = base_result->total_pde_solves;
    } else {
        const auto& base_axes = base_result->axes;
        auto m_grid = refine_axis(base_axes.grids[0], scale);
        auto tau_grid = refine_axis(base_axes.grids[1], scale);
        auto vol_grid = refine_axis(base_axes.grids[2], scale);
        auto rate_grid = refine_axis(base_axes.grids[3], scale);

        double K_ref = base_result->surface->metadata().K_ref;
        GridAccuracyParams accuracy = make_grid_accuracy(GridAccuracyProfile::High);

        auto setup = PriceTableBuilder<4>::from_vectors(
            m_grid, tau_grid, vol_grid, rate_grid,
            K_ref, accuracy, OptionType::PUT, kDivYield);
        if (!setup) { std::abort(); }

        auto& [builder, axes] = *setup;
        auto table_result = builder.build(axes);
        if (!table_result) { std::abort(); }

        surface = table_result->surface;
        pde_solves = table_result->n_pde_solves;
    }

    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    if (!aps) { std::abort(); }

    auto solver = DefaultInterpolatedIVSolver::create(std::move(*aps));
    if (!solver) { std::abort(); }

    auto t1 = std::chrono::steady_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto shape = base_result->axes.shape();
    auto [pos, _] = cache.emplace(scale, AdaptiveSolverEntry{
        std::make_unique<DefaultInterpolatedIVSolver>(std::move(*solver)),
        build_ms, pde_solves, shape,
    });
    return pos->second;
}
```

5. **`BM_Adaptive_IV_Scaled` benchmark:**

Same structure as old `BM_Interp_IV_Scaled` — parametrized by (strike_idx, scale_idx). Scales: `{1, 2, 4, 8}`. Reports `iv_err_bps`, `build_ms`, `n_pde_solves`, `base_grid_m/tau/sig/r`.

6. **Brent IV solver:** Copy `brent_solve_iv()` template from iv_fdm_sweep. Reference price function uses mango FDM (`solve_american_option`) instead of QuantLib.

**Step 3: Build**

Run: `bazel build //benchmarks:iv_interpolation_sweep`
Expected: Compiles

**Step 4: Commit**

```bash
git add benchmarks/iv_interpolation_sweep.cc benchmarks/BUILD.bazel
git commit -m "Add adaptive-baseline interpolated IV benchmark"
```

---

### Task 4: Add segmented path to `iv_interpolation_sweep.cc`

**Files:**
- Modify: `benchmarks/iv_interpolation_sweep.cc`

**Step 1: Add dividend scenario infrastructure**

Add `make_div_schedule()` and `get_div_reference_prices()` using mango's own FDM (`solve_american_option` with discrete dividends) for reference. Single scaled scenario: σ=0.20, T=1.0, quarterly $0.50 dividends.

**Step 2: Add `get_adaptive_div_solver()` cache**

```cpp
struct AdaptiveDivSolverEntry {
    std::unique_ptr<AnyIVSolver> solver;
    double build_time_ms = 0.0;
    ManualGrid base_grid;
    int base_tau_points = 0;
};

static const AdaptiveDivSolverEntry& get_adaptive_div_solver(int scale) {
    static std::optional<SegmentedAdaptiveResult> base_result;
    static std::map<int, AdaptiveDivSolverEntry> cache;

    auto it = cache.find(scale);
    if (it != cache.end()) return it->second;

    auto t0 = std::chrono::steady_clock::now();

    auto divs = make_div_schedule(kScaledMaturity);

    if (!base_result.has_value()) {
        AdaptiveGridParams params;
        AdaptiveGridBuilder builder(params);
        SegmentedAdaptiveConfig seg_config{
            .spot = kSpot,
            .option_type = OptionType::PUT,
            .dividend_yield = kDivYield,
            .discrete_dividends = divs,
            .maturity = kScaledMaturity,
            .kref_config = {.K_ref_count = 11, .K_ref_span = 0.3},
        };
        ManualGrid domain{
            .moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
            .vol = {0.05, 0.10, 0.20, 0.30, 0.50},
            .rate = {0.01, 0.03, 0.05, 0.10},
        };
        auto result = builder.build_segmented(seg_config, domain);
        if (!result) { std::abort(); }
        base_result = std::move(*result);
    }

    // Build surface: scale==1 uses adaptive directly, scale>1 refines grid
    MultiKRefSurface<>* surface_ptr = nullptr;
    std::optional<MultiKRefSurface<>> scaled_surface;

    if (scale == 1) {
        surface_ptr = &base_result->surface;
    } else {
        // Refine base grid and rebuild all K_refs via SegmentedPriceTableBuilder
        auto m_grid = refine_axis(base_result->grid.moneyness, scale);
        auto v_grid = refine_axis(base_result->grid.vol, scale);
        auto r_grid = refine_axis(base_result->grid.rate, scale);
        int tau_pts = base_result->tau_points_per_segment * scale;

        // Generate same K_refs as adaptive
        MultiKRefConfig kref_config{.K_ref_count = 11, .K_ref_span = 0.3};
        std::vector<double> K_refs;
        /* ... same K_ref generation as adaptive ... */

        std::vector<MultiKRefEntry> entries;
        DividendSpec div_spec{.dividend_yield = kDivYield, .discrete_dividends = divs};
        ManualGrid refined{.moneyness = m_grid, .vol = v_grid, .rate = r_grid};

        for (double K_ref : K_refs) {
            SegmentedPriceTableBuilder::Config cfg{
                .K_ref = K_ref,
                .option_type = OptionType::PUT,
                .dividends = div_spec,
                .grid = refined,
                .maturity = kScaledMaturity,
                .tau_points_per_segment = tau_pts,
            };
            auto seg = SegmentedPriceTableBuilder::build(cfg);
            if (!seg) { std::abort(); }
            entries.push_back({.K_ref = K_ref, .surface = std::move(*seg)});
        }
        auto built = build_multi_kref_surface(std::move(entries));
        if (!built) { std::abort(); }
        scaled_surface = std::move(*built);
        surface_ptr = &*scaled_surface;
    }

    // Wrap in SplicedSurfaceWrapper → InterpolatedIVSolver → AnyIVSolver
    // (Use same wrapping as iv_solver_factory.cpp)
    /* ... compute bounds, create wrapper, create solver ... */

    auto t1 = std::chrono::steady_clock::now();
    /* ... cache and return ... */
}
```

**Step 3: Add `BM_Adaptive_IV_Div_Scaled` benchmark**

Same structure as standard path: parametrized by (strike_idx, scale_idx), reports IV accuracy vs mango high-res FDM reference prices (with discrete dividends).

**Step 4: Build**

Run: `bazel build //benchmarks:iv_interpolation_sweep`
Expected: Compiles

**Step 5: Commit**

```bash
git add benchmarks/iv_interpolation_sweep.cc
git commit -m "Add segmented interpolated IV benchmark"
```

---

### Task 5: Final verification and cleanup

**Step 1: Full build**

Run: `bazel test //...`
Expected: All tests pass

Run: `bazel build //benchmarks/...`
Expected: All benchmarks compile (including renamed iv_fdm_sweep and new iv_interpolation_sweep)

Run: `bazel build //src/python:mango_option`
Expected: Python bindings compile

**Step 2: Commit any final fixes, push, create PR**
