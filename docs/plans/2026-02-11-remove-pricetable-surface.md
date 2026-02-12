# Remove PriceTableSurfaceND Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the `PriceTableSurfaceND` abstraction layer so that `BSplineND<double, N>` is used directly in the concept-based layered architecture, matching Chebyshev's value-semantics structure.

**Architecture:** `PriceTableSurfaceND` is a middleman wrapping `BSplineND` + metadata (axes, K_ref, dividends). The concept-based layers (`TransformLeaf`, `EEPLayer`, `PriceTable`) already handle this metadata. We change `SharedBSplineInterp<N>` to wrap `shared_ptr<const BSplineND<double, N>>` directly, update all producers/consumers, then delete the class. `BSplineND` already has `eval()` and `eval_partial()` — we add a `partial()` alias to satisfy `SurfaceInterpolant`.

**Tech Stack:** C++23, Bazel, GoogleTest

**Working directory:** `/home/kai/work/mango-option/.worktrees/remove-pricetable`

---

## Background

### Current architecture (B-spline path)
```
PriceTableBuilder::build()
  → PriceTableSurfaceND::build(axes, coeffs, K_ref, dividends)
    → creates BSplineND internally, wraps in shared_ptr<const PriceTableSurfaceND>
  → SharedBSplineInterp<4>(shared_ptr<const PriceTableSurfaceND<4>>)
    → TransformLeaf<SharedBSplineInterp<4>, StandardTransform4D>
      → EEPLayer<..., AnalyticalEEP>
        → PriceTable<...>
```

### Target architecture
```
PriceTableBuilder::assemble_surface()
  → creates BSplineND directly, wraps in shared_ptr<const BSplineND<double, N>>
  → SharedBSplineInterp<4>(shared_ptr<const BSplineND<double, 4>>)
    → TransformLeaf<SharedBSplineInterp<4>, StandardTransform4D>
      → (rest unchanged)
```

### Key insight
`BSplineND::grid(dim)` provides all the bounds metadata that consumers previously got from `PriceTableSurfaceND::axes()`. K_ref and dividends are passed separately through the builder config.

### Files with `PriceTableSurfaceND` references

**Source files:**
- `src/option/table/bspline/bspline_surface.hpp` — class definition + `SharedBSplineInterp` + type aliases
- `src/option/table/bspline/bspline_surface.cpp` — `PriceTableSurfaceND::build()` + `make_bspline_surface()`
- `src/option/table/bspline/bspline_builder.hpp` — `PriceTableResult`, `AssembleSurfaceResult`
- `src/option/table/bspline/bspline_builder.cpp` — `assemble_surface()` calls `PriceTableSurfaceND::build()`
- `src/option/table/bspline/bspline_adaptive.hpp` — `BSplineAdaptiveResult`
- `src/option/table/bspline/bspline_adaptive.cpp` — `build_cached_surface()`, `build_adaptive_bspline()`
- `src/option/table/bspline/bspline_segmented_builder.hpp` — `BSplineSegmentConfig`, `build_segment()`
- `src/option/table/bspline/bspline_segmented_builder.cpp` — IC chaining, `build_segmented_surface()`
- `src/option/table/bspline/bspline_3d_surface.hpp` — 3D type aliases use `SharedBSplineInterp<3>`
- `src/option/table/dimensionless/dimensionless_builder.hpp` — `Segment::surface`
- `src/option/table/dimensionless/dimensionless_adaptive.cpp` — builds `PriceTableSurfaceND<3>`
- `src/option/interpolated_iv_solver.cpp` — `wrap_surface()`, 3D bspline path
- `src/python/mango_bindings.cpp` — Python `PriceTableSurface` class binding

**Test files:**
- `tests/price_table_surface_test.cc`
- `tests/price_table_4d_integration_test.cc`
- `tests/dimensionless_3d_surface_test.cc`
- `tests/interpolated_iv_solver_test.cc`
- `tests/test_bindings.py`

**Benchmark files:**
- `benchmarks/golden_surface_comparison.cc`
- `benchmarks/interpolation_greek_accuracy.cc`
- `benchmarks/iv_interpolation_profile.cc`
- `benchmarks/iv_interpolation_sweep.cc`
- `benchmarks/readme_benchmarks.cc`
- `benchmarks/real_data_benchmark.cc`
- `benchmarks/bspline_template_vs_hardcoded.cc`
- `benchmarks/component_performance.cc`

---

## Task 1: Add `partial()` to BSplineND

**Files:**
- Modify: `src/math/bspline_nd.hpp:176`
- Test: `tests/bspline_nd_test.cc`

Add a `partial()` method that forwards to `eval_partial()`. This makes `BSplineND<double, N>` satisfy the `SurfaceInterpolant<S, N>` concept which requires `s.partial(size_t{}, coords)`.

**Step 1: Add `partial()` method to BSplineND**

In `src/math/bspline_nd.hpp`, after `eval_partial()` (around line 207), add:

```cpp
    /// Alias for eval_partial — satisfies SurfaceInterpolant concept.
    T partial(size_t axis, const QueryPoint& query) const {
        return eval_partial(axis, query);
    }
```

**Step 2: Add concept check test**

In `tests/bspline_nd_test.cc`, add a static_assert or test:

```cpp
#include "mango/option/table/surface_concepts.hpp"

// Verify BSplineND satisfies SurfaceInterpolant
static_assert(mango::SurfaceInterpolant<mango::BSplineND<double, 4>, 4>);
static_assert(mango::SurfaceInterpolant<mango::BSplineND<double, 3>, 3>);
static_assert(mango::SurfaceInterpolant<mango::BSplineND<double, 2>, 2>);
```

**Step 3: Build and test**

```bash
bazel test //tests:bspline_nd_test --test_output=errors
```

**Step 4: Commit**

```bash
git add src/math/bspline_nd.hpp tests/bspline_nd_test.cc
git commit -m "Add partial() alias to BSplineND for SurfaceInterpolant"
```

---

## Task 2: Refactor SharedBSplineInterp to wrap BSplineND directly

**Files:**
- Modify: `src/option/table/bspline/bspline_surface.hpp:178-197`
- Modify: `src/option/table/bspline/bspline_surface.cpp:90-129`

Change `SharedBSplineInterp<N>` from wrapping `shared_ptr<const PriceTableSurfaceND<N>>` to wrapping `shared_ptr<const BSplineND<double, N>>`.

**Step 1: Change SharedBSplineInterp**

In `src/option/table/bspline/bspline_surface.hpp`, replace the `SharedBSplineInterp` class (lines 178-197):

```cpp
/// Adapter that wraps shared_ptr<const BSplineND<double, N>> to satisfy
/// SurfaceInterpolant. Preserves shared ownership semantics.
template <size_t N>
class SharedBSplineInterp {
public:
    explicit SharedBSplineInterp(std::shared_ptr<const BSplineND<double, N>> spline)
        : spline_(std::move(spline)) {}

    [[nodiscard]] double eval(const std::array<double, N>& coords) const {
        return spline_->eval(coords);
    }

    [[nodiscard]] double partial(size_t axis, const std::array<double, N>& coords) const {
        return spline_->partial(axis, coords);
    }

    /// Access underlying spline (for grid metadata, etc.)
    [[nodiscard]] const BSplineND<double, N>& spline() const { return *spline_; }

private:
    std::shared_ptr<const BSplineND<double, N>> spline_;
};
```

Note: The accessor changes from `.surface()` to `.spline()`. This will break callers — fixed in later steps.

**Step 2: Update `make_bspline_surface()` signature**

In `src/option/table/bspline/bspline_surface.hpp`, change the declaration (around line 228):

```cpp
/// Create a BSplinePriceTable from a pre-built BSplineND.
/// K_ref and dividend_yield are passed explicitly (no longer stored in surface).
[[nodiscard]] std::expected<BSplinePriceTable, std::string>
make_bspline_surface(
    std::shared_ptr<const BSplineND<double, 4>> spline,
    double K_ref,
    double dividend_yield,
    OptionType type);
```

**Step 3: Update `make_bspline_surface()` implementation**

In `src/option/table/bspline/bspline_surface.cpp`, replace the implementation (lines 90-131):

```cpp
std::expected<BSplinePriceTable, std::string>
make_bspline_surface(
    std::shared_ptr<const BSplineND<double, 4>> spline,
    double K_ref,
    double dividend_yield,
    OptionType type)
{
    if (!spline) {
        return std::unexpected(std::string("null spline"));
    }

    if (K_ref <= 0.0) {
        return std::unexpected(std::string("invalid K_ref"));
    }

    SharedBSplineInterp<4> interp(spline);
    StandardTransform4D xform;
    AnalyticalEEP eep(type, dividend_yield);
    BSplineTransformLeaf tleaf(std::move(interp), xform, K_ref);
    BSplineLeaf leaf(std::move(tleaf), eep);

    SurfaceBounds bounds{
        .m_min = spline->grid(0).front(),
        .m_max = spline->grid(0).back(),
        .tau_min = spline->grid(1).front(),
        .tau_max = spline->grid(1).back(),
        .sigma_min = spline->grid(2).front(),
        .sigma_max = spline->grid(2).back(),
        .rate_min = spline->grid(3).front(),
        .rate_max = spline->grid(3).back(),
    };

    return BSplinePriceTable(std::move(leaf), bounds, type, dividend_yield);
}
```

**Do not compile yet** — callers need updating in subsequent tasks.

---

## Task 3: Refactor builder result types

**Files:**
- Modify: `src/option/table/bspline/bspline_builder.hpp:380-393, 562-571`
- Modify: `src/option/table/bspline/bspline_builder.cpp:140-183`

Change the builder to produce `shared_ptr<const BSplineND<double, N>>` instead of `shared_ptr<const PriceTableSurfaceND<N>>`.

**Step 1: Update result types in bspline_builder.hpp**

Change `PriceTableResult<N>` (around line 380):

```cpp
template <size_t N>
struct PriceTableResult {
    std::shared_ptr<const BSplineND<double, N>> spline;  ///< Immutable B-spline
    PriceTableAxesND<N> axes;                             ///< Grid metadata
    double K_ref = 0.0;                                   ///< Reference strike
    DividendSpec dividends;                                ///< Dividend specification
    size_t n_pde_solves = 0;
    double precompute_time_seconds = 0.0;
    BSplineFittingStats<double, N> fitting_stats;
    size_t failed_pde_slices = 0;
    size_t failed_spline_points = 0;
    size_t repaired_full_slices = 0;
    size_t repaired_partial_points = 0;
    size_t total_slices = 0;
    size_t total_points = 0;
};
```

Change `AssembleSurfaceResult` (around line 562):

```cpp
struct AssembleSurfaceResult {
    std::shared_ptr<const BSplineND<double, N>> spline;
    BSplineFittingStats<double, N> fitting_stats;
    size_t failed_pde_slices = 0;
    size_t failed_spline_points = 0;
    size_t repaired_full_slices = 0;
    size_t repaired_partial_points = 0;
    size_t total_slices = 0;
};
```

**Step 2: Update `assemble_surface()` in bspline_builder.cpp**

Replace the PriceTableSurfaceND::build() call (around line 167-183) with direct BSplineND creation:

```cpp
    // Step 5: Build BSplineND directly
    typename BSplineND<double, N>::KnotArray knots;
    typename BSplineND<double, N>::GridArray grids_copy;
    for (size_t dim = 0; dim < N; ++dim) {
        knots[dim] = clamped_knots_cubic(axes.grids[dim]);
        grids_copy[dim] = axes.grids[dim];
    }

    auto spline_result = BSplineND<double, N>::create(
        std::move(grids_copy), std::move(knots),
        std::move(coeffs_result->coefficients));
    if (!spline_result.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
    }

    auto spline = std::make_shared<const BSplineND<double, N>>(
        std::move(spline_result.value()));

    return AssembleSurfaceResult{
        .spline = std::move(spline),
        .fitting_stats = coeffs_result->stats,
        .failed_pde_slices = extraction->failed_pde.size(),
        .failed_spline_points = extraction->failed_spline.size(),
        .repaired_full_slices = repair_stats.repaired_full_slices,
        .repaired_partial_points = repair_stats.repaired_partial_points,
        .total_slices = extraction->total_slices,
    };
```

Add include for `mango/math/bspline_basis.hpp` (for `clamped_knots_cubic`).

**Step 3: Update `build()` in bspline_builder.cpp**

The `build()` method (which calls `assemble_surface`) needs to populate the new `PriceTableResult` fields:

```cpp
    return PriceTableResult<N>{
        .spline = std::move(assembly->spline),
        .axes = axes,
        .K_ref = config_.K_ref,
        .dividends = config_.dividends,
        .n_pde_solves = batch.results.size(),
        .precompute_time_seconds = elapsed,
        .fitting_stats = assembly->fitting_stats,
        .failed_pde_slices = assembly->failed_pde_slices,
        .failed_spline_points = assembly->failed_spline_points,
        .repaired_full_slices = assembly->repaired_full_slices,
        .repaired_partial_points = assembly->repaired_partial_points,
        .total_slices = assembly->total_slices,
        .total_points = axes.total_points(),
    };
```

**Do not compile yet** — more consumers to update.

---

## Task 4: Update segmented builder

**Files:**
- Modify: `src/option/table/bspline/bspline_segmented_builder.hpp:19-23, 98-106`
- Modify: `src/option/table/bspline/bspline_segmented_builder.cpp:17-22, 278-402, 408-444`

**Step 1: Update BSplineSegmentConfig**

In `bspline_segmented_builder.hpp`:

```cpp
struct BSplineSegmentConfig {
    std::shared_ptr<const BSplineND<double, 4>> spline;
    double tau_start;
    double tau_end;
};
```

**Step 2: Update build_segment() signature**

Change `prev_surface` parameter type:

```cpp
    static std::expected<BSplineSegmentConfig, PriceTableError>
    build_segment(
        size_t seg_idx,
        const std::vector<double>& boundaries,
        const Config& config,
        const std::vector<double>& expanded_log_m_grid,
        double K_ref,
        const std::vector<Dividend>& dividends,
        std::shared_ptr<const BSplineND<double, 4>>& prev_spline);
```

**Step 3: Update ChainedICContext and IC lambda in bspline_segmented_builder.cpp**

Change `ChainedICContext` (line 17-22):

```cpp
struct ChainedICContext {
    std::shared_ptr<const BSplineND<double, 4>> prev_spline;
    double K_ref;
    double prev_tau_end;
    double boundary_div;
};
```

In the IC lambda (around line 361), change `prev_surface->value()` to `prev_spline->eval()`:

```cpp
                        double raw = ic_ctx.prev_spline->eval(
                            {x_adj, ic_ctx.prev_tau_end, sigma, rate});
```

**Step 4: Update build_segment() to use new result type**

Around lines 388-401, change assembly result access:

```cpp
    auto spline_ptr = assembly->spline;
    prev_spline = spline_ptr;

    return BSplineSegmentConfig{
        .spline = spline_ptr,
        .tau_start = tau_start,
        .tau_end = tau_end,
    };
```

Also update the main `build()` function's local variable (line 255):

```cpp
    std::shared_ptr<const BSplineND<double, 4>> prev_spline;
```

**Step 5: Update build_segmented_surface()**

In `build_segmented_surface()` (lines 408-444), change surface access:

```cpp
    for (auto& seg : config.segments) {
        tau_start.push_back(seg.tau_start);
        tau_end.push_back(seg.tau_end);
        tau_min.push_back(seg.spline->grid(1).front());
        tau_max.push_back(seg.spline->grid(1).back());

        SharedBSplineInterp<4> interp(seg.spline);
        StandardTransform4D xform;
        leaves.emplace_back(std::move(interp), xform, config.K_ref);
    }
```

---

## Task 5: Update adaptive builder

**Files:**
- Modify: `src/option/table/bspline/bspline_adaptive.hpp:17-24`
- Modify: `src/option/table/bspline/bspline_adaptive.cpp:309-428, 436-503`

**Step 1: Update BSplineAdaptiveResult**

In `bspline_adaptive.hpp`:

```cpp
struct BSplineAdaptiveResult {
    std::shared_ptr<const BSplineND<double, 4>> spline;
    PriceTableAxesND<4> axes;
    double K_ref = 0.0;
    double dividend_yield = 0.0;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
};
```

**Step 2: Update build_cached_surface()**

In `bspline_adaptive.cpp`, change the `build_cached_surface` signature (around line 309):

```cpp
static std::expected<SurfaceHandle, PriceTableError>
build_cached_surface(
    const AdaptiveGridParams& params,
    BSplinePDECache& cache,
    const std::vector<double>& m_grid,
    const std::vector<double>& tau_grid,
    const std::vector<double>& v_grid,
    const std::vector<double>& r_grid,
    double K_ref,
    double dividend_yield,
    const PDEGridSpec& pde_grid,
    OptionType type,
    size_t& build_iteration,
    std::shared_ptr<const BSplineND<double, 4>>& last_spline,
    PriceTableAxes& last_axes)
```

Inside the function body (around lines 408-428):

```cpp
    last_spline = assembly->spline;
    last_axes = axes;

    auto wrapper = make_bspline_surface(assembly->spline, K_ref, dividend_yield, type);
```

**Step 3: Update build_adaptive_bspline()**

Around lines 436-503, change local variables:

```cpp
    std::shared_ptr<const BSplineND<double, 4>> last_spline;
    PriceTableAxes last_axes;
```

And the lambda capture + result construction:

```cpp
    BuildFn build_fn = [&](/* ... */) {
        return build_cached_surface(
            /* ... */
            build_iteration, last_spline, last_axes);
    };
```

And the result:

```cpp
    BSplineAdaptiveResult result;
    result.spline = last_spline;
    result.axes = last_axes;
    result.K_ref = chain.spot;
    result.dividend_yield = chain.dividend_yield;
    // ... rest unchanged
```

---

## Task 6: Update dimensionless 3D path

**Files:**
- Modify: `src/option/table/dimensionless/dimensionless_builder.hpp:68-71`
- Modify: `src/option/table/dimensionless/dimensionless_adaptive.cpp:120-136`
- Modify: `src/option/interpolated_iv_solver.cpp:547-576`

**Step 1: Update SegmentedDimensionlessSurface::Segment**

In `dimensionless_builder.hpp`:

```cpp
    struct Segment {
        std::shared_ptr<const BSplineND<double, 3>> spline;
        double lk_min, lk_max;
    };
```

**Step 2: Update SegmentedDimensionlessSurface::value()**

In `dimensionless_adaptive.cpp`, change `surface->value(coords)` calls (around line 150) to `spline->eval(coords)`:

```cpp
    if (segments_.size() == 1) {
        return std::max(segments_[0].spline->eval(coords), 0.0);
    }
```

Search for all `segments_[i].surface->value(` in the file and change to `segments_[i].spline->eval(`.

**Step 3: Update segment building in dimensionless_adaptive.cpp**

Around lines 120-136, replace `PriceTableSurfaceND<3>::build()` with direct BSplineND creation:

```cpp
    // 4. Build BSplineND<double, 3> directly
    std::array<std::vector<double>, 3> spline_grids = {
        axes.log_moneyness, axes.tau_prime, axes.ln_kappa};

    std::array<std::vector<double>, 3> knots;
    for (size_t dim = 0; dim < 3; ++dim) {
        knots[dim] = clamped_knots_cubic(spline_grids[dim]);
    }

    auto spline_result = BSplineND<double, 3>::create(
        std::move(spline_grids), std::move(knots),
        std::move(fit_result->coefficients));
    if (!spline_result.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
    }

    auto spline = std::make_shared<const BSplineND<double, 3>>(
        std::move(spline_result.value()));

    return SegmentBuildResult{
        .segment = {.spline = std::move(spline), .lk_min = dom.lk_min_phys, .lk_max = dom.lk_max_phys},
        .n_pde_solves = pde->n_pde_solves,
    };
```

Add include for `mango/math/bspline_basis.hpp`.

**Step 4: Update 3D B-spline path in interpolated_iv_solver.cpp**

Around lines 547-576, replace `PriceTableSurfaceND<3>::build()` with direct BSplineND creation:

```cpp
    // 4. Build BSplineND<double, 3> directly
    std::array<std::vector<double>, 3> spline_grids = {
        axes.log_moneyness, axes.tau_prime, axes.ln_kappa};

    std::array<std::vector<double>, 3> knots;
    for (size_t dim = 0; dim < 3; ++dim) {
        knots[dim] = clamped_knots_cubic(spline_grids[dim]);
    }

    auto spline_result = BSplineND<double, 3>::create(
        std::move(spline_grids), std::move(knots),
        std::move(fit_result->coefficients));
    if (!spline_result.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    auto spline = std::make_shared<const BSplineND<double, 3>>(
        std::move(spline_result.value()));

    // 5. Wrap in layered PriceTable
    SharedBSplineInterp<3> interp(std::move(spline));
    // ... rest unchanged
```

---

## Task 7: Update interpolated IV solver factory (4D path)

**Files:**
- Modify: `src/option/interpolated_iv_solver.cpp:198-216, 298-347`

**Step 1: Update wrap_surface()**

Change signature and body (around lines 198-216):

```cpp
static std::expected<AnyInterpIVSolver, ValidationError>
wrap_surface(std::shared_ptr<const BSplineND<double, 4>> spline,
             double K_ref,
             double dividend_yield,
             OptionType option_type,
             const InterpolatedIVSolverConfig& solver_config) {
    auto wrapper = make_bspline_surface(spline, K_ref, dividend_yield, option_type);
    if (!wrapper.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    auto solver = InterpolatedIVSolver<BSplinePriceTable>::create(
        std::move(*wrapper), solver_config);
    if (!solver.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    return make_any_solver(std::move(*solver));
}
```

**Step 2: Update build_bspline_continuous()**

In the adaptive path (around line 319):

```cpp
        return wrap_surface(result->spline, chain.spot, chain.dividend_yield,
                            config.option_type, config.solver_config);
```

In the direct-build path (around line 346):

```cpp
    return wrap_surface(table_result->spline, config.spot, config.dividend_yield,
                        config.option_type, config.solver_config);
```

**Step 3: Build and test all source code**

```bash
bazel build //src/...
```

Fix any remaining compilation errors in source files. Common issues:
- `.surface` → `.spline` in result types
- `->value(` → `->eval(` for BSplineND
- `->axes().grids[i]` → `->grid(i)` for BSplineND
- `->m_min()` → `->grid(0).front()`
- `->K_ref()` → separate K_ref variable

**Step 4: Commit**

```bash
git add -A
git commit -m "Replace PriceTableSurfaceND with direct BSplineND usage

SharedBSplineInterp now wraps shared_ptr<const BSplineND> directly.
Builder returns BSplineND + metadata instead of PriceTableSurfaceND.
All consumers updated: segmented builder, adaptive builder,
dimensionless 3D path, and IV solver factory."
```

---

## Task 8: Remove PriceTableSurfaceND class

**Files:**
- Modify: `src/option/table/bspline/bspline_surface.hpp:104-174`
- Modify: `src/option/table/bspline/bspline_surface.cpp:1-88`

**Step 1: Remove PriceTableSurfaceND from header**

In `bspline_surface.hpp`, delete the entire `PriceTableSurfaceND` class template (lines 104-174) and the `PriceTableSurface` type alias (line 174).

Keep:
- `kPriceTableDim` constant
- `PriceTableAxesND<N>` struct (used by builder)
- `PriceTableAxes` alias
- `SharedBSplineInterp<N>` class
- All type aliases (`BSplineTransformLeaf`, `BSplineLeaf`, etc.)
- `make_bspline_surface()` declaration

Remove the `#include "mango/option/table/price_table.hpp"` if it was only needed for PriceTableSurfaceND (check).

**Step 2: Remove PriceTableSurfaceND from implementation**

In `bspline_surface.cpp`, delete:
- The `PriceTableSurfaceND` constructor (lines 10-18)
- The `PriceTableSurfaceND::build()` method (lines 21-67)
- The `value()`, `partial()`, `second_partial()` forwarding methods (lines 70-82)
- All explicit template instantiations for PriceTableSurfaceND (lines 85-88)

Keep only the `make_bspline_surface()` implementation.

**Step 3: Build**

```bash
bazel build //src/...
```

Fix any remaining references. There should be none if Tasks 2-7 were completed correctly.

**Step 4: Commit**

```bash
git add -A
git commit -m "Remove PriceTableSurfaceND class

No longer needed — BSplineND is used directly via SharedBSplineInterp."
```

---

## Task 9: Update tests

**Files:**
- Modify: `tests/price_table_surface_test.cc`
- Modify: `tests/price_table_4d_integration_test.cc`
- Modify: `tests/dimensionless_3d_surface_test.cc`
- Modify: `tests/interpolated_iv_solver_test.cc`

**Step 1: Update price_table_surface_test.cc**

This test directly creates `PriceTableSurfaceND<2>` and `<3>`. Rewrite to test `BSplineND` directly:

- `PriceTableSurfaceND<2>::build(axes, coeffs, K_ref)` → `BSplineND<double, 2>::create(grids, knots, coeffs)`
- `surface->value({x, y})` → `spline.eval({x, y})`
- `surface->partial(axis, {x, y})` → `spline.partial(axis, {x, y})`

You'll need to generate knots manually via `clamped_knots_cubic()` for each grid dimension.

**Step 2: Update price_table_4d_integration_test.cc**

Change `result->surface->value(...)` to `result->spline->eval(...)`. Update any references to `result->surface->axes()` to use `result->axes` (from PriceTableResult) or `result->spline->grid(dim)`.

**Step 3: Update dimensionless_3d_surface_test.cc**

Change `PriceTableSurfaceND<3>::build(...)` to direct `BSplineND<double, 3>::create(...)`. Update surface value queries from `->value()` to `->eval()`.

**Step 4: Update interpolated_iv_solver_test.cc**

These tests go through the factory API (`make_interpolated_iv_solver`), so they should mostly work unchanged. Fix any direct `PriceTableSurface` references if present.

**Step 5: Build and run all tests**

```bash
bazel test //tests/... --test_output=errors
```

Fix any remaining failures.

**Step 6: Commit**

```bash
git add tests/
git commit -m "Update tests for PriceTableSurfaceND removal"
```

---

## Task 10: Update Python bindings

**Files:**
- Modify: `src/python/mango_bindings.cpp:554-736`

**Step 1: Remove PriceTableSurface class binding**

Delete the entire `py::class_<mango::PriceTableSurface, ...>` block (lines 557-630).

**Step 2: Update factory functions**

The `build_price_table_surface_from_grid_auto_profile` and `build_price_table_surface_from_grid` functions (lines 635-736) return `shared_ptr<const PriceTableSurface>`. These need to be either:

a) **Removed** (if Python users should use the IV solver factory instead), or
b) **Changed** to return `shared_ptr<const BSplineND<double, 4>>` and expose `BSplineND` to Python.

Recommended: Remove both factory functions for now. They were convenience wrappers. Python users should use `make_interpolated_iv_solver()` instead.

If keeping raw surface access in Python is needed, add a `BSplineND4D` binding (future work).

**Step 3: Update test_bindings.py**

Remove any tests that use `PriceTableSurface` directly.

**Step 4: Build Python bindings**

```bash
bazel build //src/python:mango_option
```

**Step 5: Commit**

```bash
git add src/python/mango_bindings.cpp tests/test_bindings.py
git commit -m "Remove PriceTableSurface from Python bindings"
```

---

## Task 11: Update benchmarks

**Files:**
- Modify: `benchmarks/golden_surface_comparison.cc`
- Modify: `benchmarks/interpolation_greek_accuracy.cc`
- Modify: `benchmarks/iv_interpolation_profile.cc`
- Modify: `benchmarks/iv_interpolation_sweep.cc`
- Modify: `benchmarks/readme_benchmarks.cc`
- Modify: `benchmarks/real_data_benchmark.cc`
- Modify: `benchmarks/bspline_template_vs_hardcoded.cc`
- Modify: `benchmarks/component_performance.cc`

**Step 1: Fix benchmark compilation**

Common patterns to fix:
- `result->surface` → `result->spline` (PriceTableResult)
- `surface->value(coords)` → `spline->eval(coords)` (BSplineND)
- `surface->partial(axis, coords)` → `spline->partial(axis, coords)` or `spline->eval_partial(axis, coords)`
- `surface->axes()` → `result->axes` or `spline->grid(dim)`
- `surface->K_ref()` → `result->K_ref`
- `PriceTableSurfaceND<3>::build(...)` → `BSplineND<double, 3>::create(...)` + knot generation
- `make_bspline_surface(surface, type)` → `make_bspline_surface(spline, K_ref, div_yield, type)`

**Step 2: Build benchmarks**

```bash
bazel build //benchmarks/...
```

**Step 3: Commit**

```bash
git add benchmarks/
git commit -m "Update benchmarks for PriceTableSurfaceND removal"
```

---

## Task 12: Final verification

**Step 1: Full build + test**

```bash
bazel test //...
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

**Step 2: Verify no PriceTableSurfaceND references remain**

```bash
grep -r "PriceTableSurfaceND" src/ tests/ benchmarks/ --include="*.hpp" --include="*.cpp" --include="*.cc"
```

Should return zero results (except possibly comments, which should be cleaned up).

**Step 3: Squash commits if desired, or keep as-is**

The commit history should be:
1. Add partial() to BSplineND
2. Replace PriceTableSurfaceND with direct BSplineND usage
3. Remove PriceTableSurfaceND class
4. Update tests
5. Update Python bindings
6. Update benchmarks
