# Table Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up `src/option/table/` per the design in `docs/plans/2026-02-10-table-cleanup-design.md` — separate shared infrastructure from B-spline-specific types, remove legacy naming, establish symmetric backend structure.

**Architecture:** Three phases of increasing risk. Phase 1 does pure type/function renames (no interface changes). Phase 2 moves files into the correct directory. Phase 3 restructures metadata interfaces. Each task produces a green build and one commit.

**Tech Stack:** C++23, Bazel, GoogleTest

**Working directory:** `/home/kai/work/mango-option/.worktrees/cleanup`

**Verification after every task:**
```bash
bazel test //...
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

---

## Phase 1: Pure Renames (type + function names only)

These tasks use `replace_all` across the repo. No file moves, no interface changes. Each rename is atomic — the build stays green after each commit.

---

### Task 1: Rename PriceTable → PriceTable

Rename the class template and its file. This is the most-referenced type in the cleanup.

**Files:**
- Rename: `src/option/table/bounded_surface.hpp` → stays in place for now (file move in Phase 2)
- Modify: ~20 files across src/, tests/, benchmarks/, docs/

**Step 1: Rename the class**

In `src/option/table/bounded_surface.hpp`:
- `PriceTable` → `PriceTable` (class name, all occurrences)
- Keep `SurfaceBounds` for now (removed in Phase 3)
- Update comment: "Adds bounds and metadata to any inner surface" → "Top-level queryable price surface with runtime metadata"

**Step 2: Apply rename across repo**

Use `replace_all` in every file that references `PriceTable`:
- `src/option/table/standard_surface.hpp:28,41` — type aliases
- `src/option/table/chebyshev/chebyshev_surface.hpp:19,25` — type aliases
- `src/option/table/chebyshev/chebyshev_table_builder.cpp:138,145`
- `src/option/table/adaptive_grid_builder.cpp:603`
- `tests/surface_concepts_test.cc:79,82`
- `docs/` — all planning docs referencing PriceTable

**Step 3: Build and test**

```bash
bazel test //...
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

**Step 4: Commit**

```bash
git add -A
git commit -m "Rename PriceTable to PriceTable"
```

---

### Task 2: Rename B-spline type aliases

Rename all "Standard*" and "Segmented*" type aliases to use `BSpline` prefix.

**Files:**
- Modify: `src/option/table/standard_surface.hpp` (definitions)
- Modify: ~50 files across src/, tests/, benchmarks/, docs/

**Step 1: Rename type aliases in standard_surface.hpp**

| Old | New |
|-----|-----|
| `StandardLeaf` | `BSplineLeaf` |
| `StandardSurface` | `BSplinePriceTable` |
| `SegmentedLeaf` | `BSplineSegmentedLeaf` |
| `SegmentedPriceSurface` | `BSplineSegmentedSurface` |
| `MultiKRefInner` | `BSplineMultiKRefInner` |
| `MultiKRefPriceSurface` | `BSplineMultiKRefSurface` |

**Step 2: Apply each rename across repo with replace_all**

Key files for each rename (non-exhaustive — use replace_all on entire repo):

`StandardSurface` (35+ refs): interpolated_iv_solver.hpp/cpp, 7 test files, 8 benchmark files, docs/
`SegmentedPriceSurface` (40+ refs): spliced_surface_builder.hpp/cpp, bspline_segmented_builder.hpp/cpp, adaptive_grid_builder.cpp, adaptive_grid_types.hpp, interpolated_iv_solver.cpp, docs/
`MultiKRefPriceSurface` (15+ refs): interpolated_iv_solver.hpp/cpp, docs/
`MultiKRefInner` (10+ refs): spliced_surface_builder.hpp/cpp, interpolated_iv_solver.cpp, adaptive_grid_types.hpp

**Step 3: Build and test**

**Step 4: Commit**

```bash
git add -A
git commit -m "Rename Standard*/Segmented* aliases to BSpline* prefix"
```

---

### Task 3: Rename factory functions

**Files:**
- Modify: `src/option/table/standard_surface.hpp` and `.cpp` (declarations + definitions)
- Modify: ~40 files across tests/, benchmarks/, docs/

**Step 1: Rename functions**

| Old | New |
|-----|-----|
| `make_bspline_surface` | `make_bspline_surface` |

Note: `make_standard_wrapper` does not exist in current code — the only factory is `make_bspline_surface`.

**Step 2: Apply rename across repo with replace_all**

Heavy usage in: interpolated_iv_solver_test.cc (5 refs), eep_integration_test.cc (4 refs), debug_vanilla_iv.cc (5 refs), market_iv_e2e_benchmark.cc (4 refs), CLAUDE.md, docs/API_GUIDE.md

**Step 3: Build and test**

**Step 4: Commit**

```bash
git add -A
git commit -m "Rename make_bspline_surface to make_bspline_surface"
```

---

## Phase 2: File Moves and Merges

Each task moves files to their correct location, updates include paths and Bazel targets. Build stays green after each commit.

---

### Task 4: Rename bounded_surface.hpp → price_table.hpp

**Files:**
- Rename: `src/option/table/bounded_surface.hpp` → `src/option/table/price_table.hpp`
- Modify: `src/option/table/BUILD.bazel` (target name `bounded_surface` → `price_table`)
- Modify: all files that include `mango/option/table/bounded_surface.hpp`
- Modify: all BUILD.bazel files that dep on `//src/option/table:bounded_surface`

**Step 1: git mv the file**

```bash
git mv src/option/table/bounded_surface.hpp src/option/table/price_table.hpp
```

**Step 2: Update include path across repo**

`replace_all`: `mango/option/table/bounded_surface.hpp` → `mango/option/table/price_table.hpp`

Consumers: standard_surface.hpp, chebyshev_surface.hpp

**Step 3: Update BUILD.bazel target**

In `src/option/table/BUILD.bazel`:
- Rename target `bounded_surface` → `price_table`
- Update `hdrs`, `strip_include_prefix`, `include_prefix`

In all BUILD.bazel files that dep on it:
`replace_all`: `":bounded_surface"` → `":price_table"` and `"//src/option/table:bounded_surface"` → `"//src/option/table:price_table"`

Check: src/option/table/BUILD.bazel (standard_surface, adaptive_grid_types deps), chebyshev/BUILD.bazel

**Step 4: Build and test**

**Step 5: Commit**

```bash
git add -A
git commit -m "Rename bounded_surface.hpp to price_table.hpp"
```

---

### Task 5: Move standard_surface into bspline/

Merge `standard_surface.hpp/cpp` into `bspline/bspline_surface.hpp/cpp`. This moves the B-spline type aliases and `make_bspline_surface()` factory next to the interpolant they use.

**Files:**
- Delete: `src/option/table/standard_surface.hpp` and `.cpp`
- Modify: `src/option/table/bspline/bspline_surface.hpp` (add type aliases + includes)
- Modify: `src/option/table/bspline/bspline_surface.cpp` (add `make_bspline_surface()` implementation)
- Modify: `src/option/table/bspline/BUILD.bazel` (update `bspline_surface` deps)
- Remove: `standard_surface` target from `src/option/table/BUILD.bazel`
- Modify: ~20 files that include `mango/option/table/standard_surface.hpp`
- Modify: all BUILD.bazel that dep on `//src/option/table:standard_surface`

**Step 1: Add type aliases and factory to bspline_surface**

Append to `bspline/bspline_surface.hpp` (after the existing classes):
- All `using` declarations from `standard_surface.hpp` (BSplineLeaf, BSplinePriceTable, etc.)
- The `make_bspline_surface()` declaration
- Required includes: price_table.hpp, eep_surface_adapter.hpp, analytical_eep.hpp, identity_eep.hpp, split_surface.hpp, tau_segment.hpp, multi_kref.hpp, standard_4d.hpp

Append to `bspline/bspline_surface.cpp`:
- The `make_bspline_surface()` implementation from `standard_surface.cpp`

**Step 2: Update bspline/BUILD.bazel**

Add deps to `bspline_surface` target:
```
"//src/option/table:price_table",
"//src/option/table:eep_surface_adapter",
"//src/option/table:analytical_eep",
"//src/option/table:identity_eep",
"//src/option/table:split_surface",
"//src/option/table:tau_segment_split",
"//src/option/table:multi_kref_split",
"//src/option/table:standard_transform_4d",
```

**Step 3: Delete standard_surface files**

```bash
git rm src/option/table/standard_surface.hpp src/option/table/standard_surface.cpp
```

Remove `standard_surface` target from `src/option/table/BUILD.bazel`.

**Step 4: Update include paths across repo**

`replace_all`: `mango/option/table/standard_surface.hpp` → `mango/option/table/bspline/bspline_surface.hpp`

Consumers (~20 files): interpolated_iv_solver.hpp, adaptive_grid_types.hpp, spliced_surface_builder.hpp, bspline_segmented_builder.hpp, 7 test files, 8 benchmark files

**Step 5: Update Bazel dep labels**

`replace_all` in all BUILD.bazel files:
- `"//src/option/table:standard_surface"` → `"//src/option/table/bspline:bspline_surface"`
- `":standard_surface"` → `"//src/option/table/bspline:bspline_surface"` (within table/BUILD.bazel)

**Step 6: Build and test**

**Step 7: Commit**

```bash
git add -A
git commit -m "Move B-spline type aliases into bspline/bspline_surface"
```

---

### Task 6: Move dividend_utils to src/option/

**Files:**
- Move: `src/option/table/dividend_utils.hpp` → `src/option/dividend_utils.hpp`
- Modify: `src/option/table/BUILD.bazel` (remove target)
- Create: target in `src/option/BUILD.bazel`
- Modify: 2 consumers (adaptive_grid_builder.cpp, bspline_segmented_builder.cpp)

**Step 1: git mv**

```bash
git mv src/option/table/dividend_utils.hpp src/option/dividend_utils.hpp
```

**Step 2: Update BUILD files**

Remove `dividend_utils` target from `src/option/table/BUILD.bazel`.

Add to `src/option/BUILD.bazel`:
```python
cc_library(
    name = "dividend_utils",
    hdrs = ["dividend_utils.hpp"],
    deps = ["//src/option:option_spec"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option",
    include_prefix = "mango/option",
)
```

**Step 3: Update include paths**

`replace_all`: `mango/option/table/dividend_utils.hpp` → `mango/option/dividend_utils.hpp`

**Step 4: Update Bazel dep labels**

`replace_all`: `"//src/option/table:dividend_utils"` → `"//src/option:dividend_utils"`

**Step 5: Build and test**

**Step 6: Commit**

```bash
git add -A
git commit -m "Move dividend_utils to src/option/ (shared with FDM)"
```

---

### Task 7: Move spliced_surface_builder to bspline/

`spliced_surface_builder` constructs `BSplineSegmentedSurface` and `BSplineMultiKRefInner` — all B-spline types.

**Files:**
- Move: `src/option/table/spliced_surface_builder.{hpp,cpp}` → `src/option/table/bspline/`
- Modify: BUILD.bazel files
- Modify: 3 consumers (bspline_segmented_builder.hpp, adaptive_grid_builder.cpp, interpolated_iv_solver.cpp)

**Step 1: git mv**

```bash
git mv src/option/table/spliced_surface_builder.hpp src/option/table/bspline/
git mv src/option/table/spliced_surface_builder.cpp src/option/table/bspline/
```

**Step 2: Update BUILD files**

Move `spliced_surface_builder` target from `src/option/table/BUILD.bazel` to `src/option/table/bspline/BUILD.bazel`. Update `strip_include_prefix` and `include_prefix` to `bspline` variants. Update internal deps.

**Step 3: Update include paths**

`replace_all`: `mango/option/table/spliced_surface_builder.hpp` → `mango/option/table/bspline/spliced_surface_builder.hpp`

**Step 4: Update Bazel dep labels**

`replace_all`: `"//src/option/table:spliced_surface_builder"` → `"//src/option/table/bspline:spliced_surface_builder"`

**Step 5: Build and test**

**Step 6: Commit**

```bash
git add -A
git commit -m "Move spliced_surface_builder to bspline/ (B-spline only)"
```

---

### Task 8: Move B-spline-only config/helpers to bspline/

Move remaining B-spline-specific files and merge small ones.

**Files to move/merge:**
- `price_table_config.hpp` → merge into `bspline/bspline_builder.hpp`
- `price_table_grid_estimator.hpp` → merge into `bspline/bspline_builder.hpp`
- `recursion_helpers.hpp` → inline into `bspline/bspline_builder.cpp` (not in header)

**Step 1: Merge price_table_config into bspline_builder**

Copy the `PriceTableConfig` struct (and `GridAccuracyParams` if defined there — check; it may be in `grid_spec_types.hpp`) into `bspline/bspline_builder.hpp`. Add any missing includes.

Delete `src/option/table/price_table_config.hpp`. Remove its Bazel target. Update `bspline_builder` target deps to absorb what `price_table_config` depended on.

`replace_all` include path: `mango/option/table/price_table_config.hpp` → `mango/option/table/bspline/bspline_builder.hpp`

Consumers: bspline_builder.hpp (already there), 3 test files

`replace_all` Bazel dep: `"//src/option/table:price_table_config"` → `"//src/option/table/bspline:bspline_builder"`

**Step 2: Merge price_table_grid_estimator into bspline_builder**

Copy `PriceTableGridProfile` enum and `estimate_grid_for_price_table()` into `bspline/bspline_builder.hpp`.

Delete `src/option/table/price_table_grid_estimator.hpp`. Remove Bazel target.

`replace_all` include path: `mango/option/table/price_table_grid_estimator.hpp` → `mango/option/table/bspline/bspline_builder.hpp`

Consumers: bspline_builder.hpp (already), 1 test file, 2 benchmark files

`replace_all` Bazel dep: `"//src/option/table:price_table_grid_estimator"` → `"//src/option/table/bspline:bspline_builder"`

**Step 3: Inline recursion_helpers into bspline_builder.cpp**

Copy `for_each_axis_index()` template into an anonymous namespace in `bspline/bspline_builder.cpp`.

Delete `src/option/table/recursion_helpers.hpp`. Remove Bazel target.

The dedicated test `recursion_helpers_test.cc`: either delete it (behavior tested indirectly via builder tests) or keep it by including `bspline_builder.hpp` instead. Prefer keeping if it tests edge cases.

**Step 4: Build and test**

**Step 5: Commit**

```bash
git add -A
git commit -m "Merge B-spline config and helpers into bspline_builder"
```

---

### Task 9: Move slice_cache + error_attribution to bspline/

**Files:**
- Move: `src/option/table/slice_cache.hpp` → `src/option/table/bspline/bspline_slice_cache.hpp`
- Merge: `src/option/table/error_attribution.hpp` into `bspline_slice_cache.hpp`
- Modify: `adaptive_grid_builder.hpp` (update includes)

**Step 1: Create bspline_slice_cache.hpp**

Combine `SliceCache` from `slice_cache.hpp` and `ErrorBins` from `error_attribution.hpp` into one file: `src/option/table/bspline/bspline_slice_cache.hpp`. Keep both class names.

**Step 2: Delete old files and Bazel targets**

```bash
git rm src/option/table/slice_cache.hpp src/option/table/error_attribution.hpp
```

Remove `slice_cache` and `error_attribution` targets from `src/option/table/BUILD.bazel`.

Add `bspline_slice_cache` target to `bspline/BUILD.bazel`.

**Step 3: Update includes**

`replace_all`: `mango/option/table/slice_cache.hpp` → `mango/option/table/bspline/bspline_slice_cache.hpp`
`replace_all`: `mango/option/table/error_attribution.hpp` → `mango/option/table/bspline/bspline_slice_cache.hpp`

Consumer: `adaptive_grid_builder.hpp` (includes both)

**Step 4: Update Bazel dep labels**

`replace_all`: `"//src/option/table:slice_cache"` → `"//src/option/table/bspline:bspline_slice_cache"`
`replace_all`: `"//src/option/table:error_attribution"` → `"//src/option/table/bspline:bspline_slice_cache"`

**Step 5: Update dedicated tests**

`tests/slice_cache_test.cc` → update include
`tests/error_attribution_test.cc` → update include

**Step 6: Build and test**

**Step 7: Commit**

```bash
git add -A
git commit -m "Move SliceCache and ErrorBins to bspline/bspline_slice_cache"
```

---

### Task 10: Merge PriceTableAxes into bspline_surface

**Files:**
- Delete: `src/option/table/price_table_axes.hpp`
- Modify: `src/option/table/bspline/bspline_surface.hpp` (add PriceTableAxesND definition)
- Modify: ~15 files that include price_table_axes.hpp

**Step 1: Move type definition**

Copy `PriceTableAxesND<N>`, `kPriceTableDim`, and the `PriceTableAxes` type alias from `price_table_axes.hpp` into `bspline/bspline_surface.hpp` (before the surface class, since surface uses axes). Include the deps (`error_types`, `safe_math`).

**Step 2: Delete old file and Bazel target**

```bash
git rm src/option/table/price_table_axes.hpp
```

Remove `price_table_axes` target from `src/option/table/BUILD.bazel`. Add its deps (`error_types`, `safe_math`) to the `bspline_surface` target.

**Step 3: Update include paths**

`replace_all`: `mango/option/table/price_table_axes.hpp` → `mango/option/table/bspline/bspline_surface.hpp`

Consumers: price_tensor.hpp, eep_decomposer.hpp, adaptive_grid_types.hpp, bspline_builder.hpp, mango_bindings.cpp, 4 test files, 1 benchmark

**Step 4: Update Bazel dep labels**

`replace_all`: `"//src/option/table:price_table_axes"` → `"//src/option/table/bspline:bspline_surface"`

Check: price_tensor target deps, eep_decomposer deps, adaptive_grid_types deps, bspline_builder deps, bspline_surface deps (self-dep remove)

**Step 5: Build and test**

**Step 6: Commit**

```bash
git add -A
git commit -m "Merge PriceTableAxes into bspline/bspline_surface"
```

---

### Task 11: Merge PriceTensor into bspline_builder

**Files:**
- Delete: `src/option/table/price_tensor.hpp`
- Modify: `src/option/table/bspline/bspline_builder.hpp` (add PriceTensorND definition)
- Modify: eep_decomposer.hpp (update include)

**Step 1: Move type definition**

Copy `PriceTensorND<N>` and `PriceTensor` alias into `bspline/bspline_builder.hpp`. Include deps (`aligned_allocator`, `safe_math`, `mdspan`).

**Step 2: Delete old file and Bazel target**

```bash
git rm src/option/table/price_tensor.hpp
```

Remove `price_tensor` target. Add its deps to `bspline_builder` target.

**Step 3: Update include paths**

`replace_all`: `mango/option/table/price_tensor.hpp` → `mango/option/table/bspline/bspline_builder.hpp`

Consumers: eep_decomposer.hpp, 2 test files

**Step 4: Update Bazel dep labels**

`replace_all`: `"//src/option/table:price_tensor"` → `"//src/option/table/bspline:bspline_builder"`

**Step 5: Build and test**

**Step 6: Commit**

```bash
git add -A
git commit -m "Merge PriceTensor into bspline/bspline_builder"
```

---

## Phase 3: Interface Changes

These tasks modify type interfaces. Higher risk, requires updating constructor call sites.

---

### Task 12: Move EEPDecomposer to bspline/

`EEPDecomposer.decompose()` takes `PriceTensor&` and `PriceTableAxes&` — both now in bspline/. Move it there.

**Files:**
- Move: `src/option/table/eep/eep_decomposer.{hpp,cpp}` → `src/option/table/bspline/`
- Modify: BUILD.bazel files
- Modify: 2 consumers (adaptive_grid_builder.cpp, interpolated_iv_solver.cpp)

**Step 1: git mv**

```bash
git mv src/option/table/eep/eep_decomposer.hpp src/option/table/bspline/
git mv src/option/table/eep/eep_decomposer.cpp src/option/table/bspline/
```

**Step 2: Update BUILD files**

Move `eep_decomposer` target from `src/option/table/BUILD.bazel` to `bspline/BUILD.bazel`. Update strip/include prefix. Update deps to use `bspline_surface` (for axes) and `bspline_builder` (for tensor).

**Step 3: Update include paths and Bazel labels**

`replace_all`: `mango/option/table/eep/eep_decomposer.hpp` → `mango/option/table/bspline/eep_decomposer.hpp`
`replace_all`: `"//src/option/table:eep_decomposer"` → `"//src/option/table/bspline:eep_decomposer"`

**Step 4: Build and test**

**Step 5: Commit**

```bash
git add -A
git commit -m "Move EEPDecomposer to bspline/ (depends on B-spline types)"
```

---

### Task 13: Remove SurfaceContent enum

The type-level `AnalyticalEEP` vs `IdentityEEP` already encodes this. Remove the runtime enum.

**Files:**
- Modify: `src/option/table/price_table_metadata.hpp` (remove enum)
- Modify: `src/option/table/bspline/bspline_surface.cpp` (remove content check in make_bspline_surface)
- Modify: `src/option/table/bspline/bspline_builder.cpp` (remove content field usage)
- Modify: `src/option/table/bspline/bspline_workspace.cpp` (remove content field — workspace is broken anyway)
- Modify: `src/python/mango_bindings.cpp` (remove SurfaceContent binding)
- Modify: test/benchmark files that reference SurfaceContent

**Step 1: Remove enum from price_table_metadata.hpp**

Delete the `SurfaceContent` enum. Remove the `content` field from `PriceTableMetadata`.

**Step 2: Remove all references**

Use grep to find all `SurfaceContent` references and remove/update each:
- `make_bspline_surface()`: remove the `meta.content != SurfaceContent::EarlyExercisePremium` check (the type alias already enforces EEP)
- `bspline_builder.cpp`: remove `metadata.content = ...` assignments
- `bspline_workspace.cpp`: remove content-related code (persistence is broken)
- `mango_bindings.cpp`: remove `SurfaceContent` Python enum binding
- Tests/benchmarks: remove assertions on `.content`

**Step 3: Build and test**

**Step 4: Commit**

```bash
git add -A
git commit -m "Remove SurfaceContent enum (type-level via EEP strategy)"
```

---

### Task 14: Remove PriceTableMetadata

After SurfaceContent removal, PriceTableMetadata has: K_ref, dividends, m_min, m_max. These move to either PriceTable (K_ref, dividends) or the interpolant (bounds).

**Note:** This is the deepest interface change. The B-spline surface currently stores metadata internally and `make_bspline_surface()` reads from it. After this change:
- `PriceTableSurfaceND` stores only axes + spline (no metadata)
- `PriceTable<Inner>` gains K_ref and dividend_yield constructor params
- m_min/m_max come from axes (already available)

**Files:**
- Delete: `src/option/table/price_table_metadata.hpp`
- Modify: `src/option/table/bspline/bspline_surface.hpp` (remove PriceTableMetadata member)
- Modify: `src/option/table/bspline/bspline_surface.cpp` (update build() signature)
- Modify: `src/option/table/bspline/bspline_builder.cpp` (stop creating metadata)
- Modify: `src/option/table/bspline/bspline_workspace.hpp` (update — persistence broken anyway)
- Modify: `src/option/table/price_table.hpp` (add K_ref to PriceTable constructor)
- Modify: `src/option/table/bspline/bspline_surface.cpp` (update make_bspline_surface)
- Modify: `src/python/mango_bindings.cpp` (remove PriceTableMetadata binding, update PriceTableSurface)
- Modify: test files referencing metadata

**Step 1: Add K_ref to PriceTable**

In `price_table.hpp`, add `K_ref` parameter to `PriceTable` constructor and member:
```cpp
PriceTable(Inner inner, SurfaceBounds bounds,
           OptionType option_type, double dividend_yield, double K_ref)
```

**Step 2: Remove PriceTableMetadata from BSplineSurface**

In `bspline_surface.hpp`:
- Remove `PriceTableMetadata meta_` member
- Remove `metadata()` accessor
- Change `build()` signature: remove metadata param, keep axes + coeffs
- Surface stores only axes + spline

**Step 3: Update all construction sites**

- `bspline_builder.cpp`: pass K_ref directly to PriceTable instead of via metadata
- `make_bspline_surface()`: read K_ref and dividend_yield from caller params (function signature changes)
- `adaptive_grid_builder.cpp`: update surface construction
- `bspline_segmented_builder.cpp`: update construction

**Step 4: Delete price_table_metadata.hpp**

```bash
git rm src/option/table/price_table_metadata.hpp
```

Remove Bazel target. Update deps.

**Step 5: Update tests and bindings**

- Remove `PriceTableMetadata` from Python bindings
- Update all tests that reference `.metadata()` or construct `PriceTableMetadata`
- Update `PriceTableSurface` Python binding (remove metadata param from constructor)

**Step 6: Build and test**

**Step 7: Commit**

```bash
git add -A
git commit -m "Remove PriceTableMetadata (fields absorbed into PriceTable)"
```

---

## Phase 3b: Optional — Bounds Delegation

### Task 15 (Optional): Delegate bounds from PriceTable to interpolant

Remove `SurfaceBounds` struct. `PriceTable` queries bounds from the inner surface's interpolant. This requires adding bounds accessors to `EEPSurfaceAdapter` and the interpolant types.

**Deferred:** This can be a follow-up PR. The current design with `SurfaceBounds` works and Phase 1-2 already delivers the main cleanup value.

---

## Verification Checklist

After all tasks complete:

```bash
# All tests pass
bazel test //...

# All benchmarks compile
bazel build //benchmarks/...

# Python bindings compile
bazel build //src/python:mango_option

# No stale files in old locations
ls src/option/table/bounded_surface.hpp        # should not exist
ls src/option/table/standard_surface.hpp       # should not exist
ls src/option/table/price_table_config.hpp     # should not exist
ls src/option/table/price_table_grid_estimator.hpp  # should not exist
ls src/option/table/recursion_helpers.hpp      # should not exist
ls src/option/table/slice_cache.hpp            # should not exist
ls src/option/table/error_attribution.hpp      # should not exist
ls src/option/table/dividend_utils.hpp         # should not exist

# Confirm no references to old names
grep -r "PriceTable" src/ tests/ benchmarks/  # should be zero
grep -r "StandardLeaf" src/ tests/ benchmarks/     # should be zero
grep -r "StandardSurface" src/ tests/ benchmarks/  # should be zero
grep -r "make_bspline_surface" src/ tests/ benchmarks/  # should be zero
```
