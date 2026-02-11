# Split adaptive_grid_builder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the 2296-line `AdaptiveGridBuilder` god class with free
functions per backend, moving implementations next to their interpolant code.

**Architecture:** Shared refinement loop in `adaptive_refinement.{hpp,cpp}`.
B-spline adaptive builder in `bspline/bspline_adaptive.{hpp,cpp}`.
Chebyshev adaptive builder in `chebyshev/chebyshev_adaptive.{hpp,cpp}`.
Old `adaptive_grid_builder.{hpp,cpp}` deleted.

**Tech Stack:** C++23, Bazel, GoogleTest

---

### Task 1: Create adaptive_refinement.hpp with shared types

Move shared types out of `adaptive_grid_builder.cpp` (internal) and
`adaptive_grid_builder.hpp` / `bspline_slice_cache.hpp` into one shared header.

**Files:**
- Create: `src/option/table/adaptive_refinement.hpp`
- Modify: `src/option/table/bspline/bspline_slice_cache.hpp` (remove ErrorBins)
- Modify: `src/option/table/BUILD.bazel` (add target)

**What goes in `adaptive_refinement.hpp`:**

Types (currently internal to .cpp):
- `RefinementContext` (lines 253-261)
- `GridSizes` (lines 264-274) — rename to `RefinementResult`
- `MaxGridSizes` (lines 277-280)
- `InitialGrids` (lines 943-951)
- `SegmentBoundaries` (lines 113-116)

Types (currently in other headers):
- `SurfaceHandle` (from adaptive_grid_builder.hpp:27-31)
- `ErrorBins` (from bspline_slice_cache.hpp:117-196)

Type aliases (currently internal to .cpp):
- `BuildFn` (lines 180-184)
- `ValidateFn` (lines 198-200)
- `RefineFn` (lines 189-195)

Function declarations for all shared helpers:
- `run_refinement()`
- `compute_iv_error()`
- `make_validate_fn()`
- `make_bs_vega_error_fn()`
- `select_probes()`
- `seed_grid()`
- `linspace()`
- `expand_domain_bounds()`
- `spline_support_headroom()`
- `extract_chain_domain()`
- `extract_initial_grids()`
- `compute_segment_boundaries()`
- `aggregate_max_sizes()`
- `total_discrete_dividends()`

**ErrorBins move:** Remove ErrorBins from `bspline_slice_cache.hpp` and have
it include `adaptive_refinement.hpp` instead (or just forward-declare).
ErrorBins is used by run_refinement() and all refine callbacks — it's shared
infrastructure, not B-spline specific.

**BUILD target:** `adaptive_refinement` (header-only for now — .cpp in next task).

**Verification:** `bazel build //src/option/table:adaptive_refinement`

---

### Task 2: Create adaptive_refinement.cpp

Move shared function implementations from `adaptive_grid_builder.cpp`.

**Files:**
- Create: `src/option/table/adaptive_refinement.cpp`
- Modify: `src/option/table/BUILD.bazel` (add srcs, deps)

**Functions to move (with original line ranges):**

```
expand_domain_bounds()           42-53
spline_support_headroom()        56-59
select_probes()                  63-77
total_discrete_dividends()       99-110
compute_segment_boundaries()     126-173
compute_iv_error()               203-212
make_bs_vega_error_fn()          215-227
make_validate_fn()               230-249
aggregate_max_sizes()            282-291
linspace()                       295-304
seed_grid()                      309-347
run_refinement()                 953-1209
extract_chain_domain()           1395-1436
extract_initial_grids()          1439-1449
```

Constants to move: `kMinPositive` (line 38), `kInset` (line 130).

**Dependencies:** `adaptive_grid_types.hpp`, `option_grid.hpp`,
`grid_spec_types.hpp`, `american_option.hpp`, `black_scholes_analytics.hpp`,
`latin_hypercube.hpp`, `error_types.hpp`.

**Do NOT move yet:** B-spline-specific or Chebyshev-specific functions. The
old `adaptive_grid_builder.cpp` still compiles with remaining functions +
class methods. Both old and new code coexist.

**Verification:**
```
bazel build //src/option/table:adaptive_refinement
bazel build //src/option/table:adaptive_grid_builder
```

Both must build. No test changes yet — old code still works.

---

### Task 3: Create bspline_adaptive.{hpp,cpp}

**Files:**
- Create: `src/option/table/bspline/bspline_adaptive.hpp`
- Create: `src/option/table/bspline/bspline_adaptive.cpp`
- Modify: `src/option/table/bspline/BUILD.bazel` (add target)

**Header declares:**

```cpp
struct BSplineAdaptiveResult {
    std::shared_ptr<const PriceTableSurface> surface;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
};

struct BSplineSegmentedAdaptiveResult {
    BSplineMultiKRefInner surface;
    IVGrid grid;
    int tau_points_per_segment;
};

std::expected<BSplineAdaptiveResult, PriceTableError>
build_adaptive_bspline(const AdaptiveGridParams& params,
                       const OptionGrid& chain,
                       PDEGridSpec pde_grid, OptionType type);

std::expected<BSplineSegmentedAdaptiveResult, PriceTableError>
build_adaptive_bspline_segmented(
    const AdaptiveGridParams& params,
    const SegmentedAdaptiveConfig& config,
    const IVGrid& domain);
```

**Implementation moves from adaptive_grid_builder.cpp:**

```
make_seg_config()                81-97      (B-spline segmented helper)
make_bspline_refine_fn()         354-433
build_segmented_surfaces()       1213-1231
probe_and_build()                1248-1380
solve_missing_slices()           1452-1532  (+ constants MAX_WIDTH etc.)
compute_error_metric()           1550-1554
merge_results()                  1556-1617
build_cached_surface()           1619-1759
```

Plus the bodies of `AdaptiveGridBuilder::build()` (lines 1761-1991) and
`AdaptiveGridBuilder::build_segmented()` (lines 1859-1981), rewritten as
free functions calling `run_refinement()` from adaptive_refinement.

Internal type `SegmentedBuildResult` (lines 1234-1243) stays in the .cpp.

**Dependencies:** `adaptive_refinement`, `bspline_surface`, `bspline_builder`,
`bspline_tensor_accessor`, `bspline_slice_cache`, `bspline_segmented_builder`,
`eep_decomposer`, `split_surface`, `tau_segment_split`, `multi_kref_split`,
`american_option_batch`, `cubic_spline_solver`.

**Verification:**
```
bazel build //src/option/table/bspline:bspline_adaptive
```

---

### Task 4: Create chebyshev_adaptive.{hpp,cpp}

**Files:**
- Create: `src/option/table/chebyshev/chebyshev_adaptive.hpp`
- Create: `src/option/table/chebyshev/chebyshev_adaptive.cpp`
- Modify: `src/option/table/chebyshev/BUILD.bazel` (add target)

**Header declares:**

```cpp
struct ChebyshevAdaptiveResult {
    std::shared_ptr<ChebyshevRawSurface> surface;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
};

struct ChebyshevSegmentedAdaptiveResult {
    std::function<double(double, double, double, double, double)> price_fn;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
};

std::expected<ChebyshevAdaptiveResult, PriceTableError>
build_adaptive_chebyshev(const AdaptiveGridParams& params,
                         const OptionGrid& chain,
                         OptionType type);

std::expected<ChebyshevSegmentedAdaptiveResult, PriceTableError>
build_adaptive_chebyshev_segmented(
    const AdaptiveGridParams& params,
    const SegmentedAdaptiveConfig& config,
    const IVGrid& domain);
```

**Implementation moves from adaptive_grid_builder.cpp:**

```
ChebyshevBuildConfig             440-444    (internal struct)
ChebyshevRefinementState         447-458    (internal struct)
SegmentedChebyshevBuildConfig    461-468    (internal struct)
make_chebyshev_build_fn()        472-606
make_segmented_chebyshev_build_fn()  609-822
make_chebyshev_refine_fn()       828-873
make_segmented_chebyshev_refine_fn() 877-936
```

Plus the bodies of `build_chebyshev()` (lines 1993-2083) and
`build_segmented_chebyshev()` (lines 2085-2294), rewritten as free functions.

**Dependencies:** `adaptive_refinement`, `chebyshev_surface`,
`chebyshev_nodes`, `pde_slice_cache`, `american_option_batch`,
`cubic_spline_solver`, `split_surface`, `tau_segment_split`,
`multi_kref_split`.

**Verification:**
```
bazel build //src/option/table/chebyshev:chebyshev_adaptive
```

---

### Task 5: Update interpolated_iv_solver.cpp

**Files:**
- Modify: `src/option/interpolated_iv_solver.cpp`
- Modify: `src/option/interpolated_iv_solver.hpp`
- Modify: `src/option/BUILD.bazel`

**Changes:**

Replace `#include "mango/option/table/adaptive_grid_builder.hpp"` with:
- `#include "mango/option/table/bspline/bspline_adaptive.hpp"`
- `#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"`

Replace each `AdaptiveGridBuilder builder(...)` + `builder.build*()` pattern
with the corresponding free function call. Remove `std::any_cast` — results
are now typed.

Three call sites in the .cpp:
1. Standard B-spline path (~line 216): `builder.build(chain, accuracy, type)`
   → `build_adaptive_bspline(params, chain, pde_grid, type)`
2. Segmented B-spline path (~line 318): `builder.build_segmented(seg_config, grid)`
   → `build_adaptive_bspline_segmented(params, config, grid)`
3. Segmented Chebyshev path (~line 414): `builder.build_segmented_chebyshev(seg_config, grid)`
   → `build_adaptive_chebyshev_segmented(params, config, grid)`

Update BUILD deps: replace `//src/option/table:adaptive_grid_builder` with
`//src/option/table/bspline:bspline_adaptive` and
`//src/option/table/chebyshev:chebyshev_adaptive`.

If the .hpp exposes `SegmentedAdaptiveConfig` or `SurfaceHandle`, update
those imports too.

**Verification:**
```
bazel build //src/option:interpolated_iv_solver
```

---

### Task 6: Update tests

**Files:**
- Modify: `tests/adaptive_grid_builder_test.cc`
- Modify: `tests/adaptive_grid_builder_integration_test.cc`
- Modify: `tests/BUILD.bazel`

Replace `AdaptiveGridBuilder` class usage with free function calls.

`adaptive_grid_builder_test.cc`: Uses `builder.build(chain, grid_spec, 200, type)`.
Replace with `build_adaptive_bspline(params, chain, PDEGridConfig{grid_spec, 200, {}}, type)`.

`adaptive_grid_builder_integration_test.cc`: Similar pattern.

Consider renaming test files to `bspline_adaptive_test.cc` /
`adaptive_integration_test.cc` if appropriate, or keep existing names.

Update BUILD deps to point at new targets.

**Verification:**
```
bazel test //tests:adaptive_grid_builder_test
bazel test //tests:adaptive_grid_builder_integration_test
```

---

### Task 7: Update benchmarks

**Files:**
- Modify: `benchmarks/iv_interpolation_sweep.cc`
- Modify: `benchmarks/interp_iv_safety.cc`
- Modify: `benchmarks/debug_vanilla_iv.cc`
- Modify: `benchmarks/BUILD.bazel`

Same pattern: replace class construction with free function calls.
Update BUILD deps.

**Verification:**
```
bazel build //benchmarks/...
```

---

### Task 8: Delete old files and clean up types

**Files:**
- Delete: `src/option/table/adaptive_grid_builder.hpp`
- Delete: `src/option/table/adaptive_grid_builder.cpp`
- Modify: `src/option/table/adaptive_grid_types.hpp` (remove AdaptiveResult,
  SegmentedAdaptiveResult — replaced by typed results per backend)
- Modify: `src/option/table/BUILD.bazel` (remove adaptive_grid_builder target)

Grep for any remaining references to `adaptive_grid_builder`,
`AdaptiveGridBuilder`, `AdaptiveResult`, `SegmentedAdaptiveResult` across
the entire repo. Fix any stragglers.

**Verification:**
```
bazel test //...
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

All 117 tests pass, all benchmarks and python bindings compile.

---

### Task 9: Final verification

```
bazel test //...
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

Verify no references to old types remain:
```
grep -r "AdaptiveGridBuilder\|adaptive_grid_builder" src/ tests/ benchmarks/
grep -r "AdaptiveResult\b" src/ tests/ benchmarks/
```

Check line counts — old file was 2296 lines. New files should total roughly
the same, distributed as ~600 shared + ~600 B-spline + ~600 Chebyshev.
