# NaN Input Guards — Design (issues #425, #426, #466)

Date: 2026-08-31 (rev 2, after design review round 1)
Issues: #425 (CubicSpline), #426 (ChebyshevInterpolant), #466 (silent NaN→0.0 at surface queries)
Branch: one PR closes all three.

## Problem

Three interpolation layers silently accept non-finite data and convert it into
plausible-looking garbage instead of an error. All were traced during the #419
investigation, where PDE instability produced NaN values that flowed through
the whole table-build stack and surfaced as prices of exactly 0.0, hundreds of
splines away from the fault.

1. **#425** — `CubicSpline<T>::build()` / `rebuild_same_grid()`
   (`src/math/cubic_spline_solver.hpp`) validate sizes and x-monotonicity but
   not finiteness. NaN `y` flows through the Thomas solve into NaN
   coefficients; `build()` reports success.
2. **#426** — `ChebyshevInterpolant<N, Storage>::build_from_values()`
   (`src/math/chebyshev/chebyshev_interpolant.hpp`) does no input validation
   and returns the constructed object directly, so there is no error path.
   CI logs showed builds succeeding with 15–20% NaN input.
3. **#466** — a NaN query coordinate produces a price of exactly **0.0** from
   surface queries. **The issue's stated mechanism is wrong** (verified
   empirically during design review, see below): raw `BSplineND::eval`,
   `eval_partial`, and `eval_second_partial` already return NaN for NaN
   queries — the degree-0 indicators do go to 0, but the degree-1 recursion
   computes `(NaN - t)/den * 0 = NaN` and the NaN survives the tensor
   contraction. The actual NaN→0.0 conversions are the ordered `std::max`
   calls at the layer above:
   - `TransformLeaf::price()` — `std::max(0.0, raw)`
     (`src/option/table/transform_leaf.hpp:32`), the query-time mask;
   - `eep_floor()` — `std::max(0.0, eep_raw)`
     (`src/option/table/eep/eep_decomposer.hpp:18`), the build-time mask that
     also hides NaN from the new #426 guard;
   - `AmericanOptionResult::value_at` — `std::max(0.0, spline_.eval(x))`
     (`src/option/american_option_result.cpp:45`), same family in the FDM
     result path.
   (`std::max(0.0, NaN)` returns its first argument, `0.0`, because every
   comparison with NaN is false.)

### Additional gap surfaced in design review (part of the motivating story)

`ChebyshevPDECache::store_slice()` (`chebyshev_pde_cache.hpp:23`) discards the
`CubicSpline::build()` error and only clears a `valid` flag; both adaptive
extraction loops (`chebyshev_adaptive.cpp:301` and `:410`) then do
`if (!spline) continue` over zero-initialized tensors. So even with the #425
guard, the motivating PDE-instability scenario would still build a "successful"
surface out of finite zeros. The extraction loops must fail loudly instead.

## Decisions (brainstorm Q&A + review round 1)

- **D1 — #426 uses a runtime `std::expected` contract (Option B), not an
  assert.** User chose B. Asserts are elided in opt builds, which is exactly
  where production price tables are built; `std::expected` matches the project
  error model. *(Review refinement: the error type is the existing structured
  `InterpolationError` — see D5.)*
- **D2 — #466 propagates NaN; no eval API change.** User chose NaN propagation
  over an error return. *(Review refinement: no `BSplineND` code change is
  needed at all — raw eval already propagates NaN, verified empirically on the
  unmodified implementation (1D and 2D, all three eval methods returned NaN).
  The fix moves to the masking boundaries: swap `std::max(0.0, v)` to
  `std::max(v, 0.0)`, which returns identical results for all finite `v` and
  NaN for NaN, at zero cost. Regression tests lock the raw-eval behavior so an
  "optimization" cannot reintroduce zeroing there.)*
- **D3 — Single branch/PR for all three.** Same theme, one review cycle.
- **D4 — Performance constraint (user, explicit): no significant perf impact.**
  - #425/#426 guards are build-time-only O(n) scans over data those functions
    already traverse O(n); no query-path change.
  - #466 is now a pure operand-order swap in `std::max` — same instruction
    count, no new branches. The hot ~250ns eval path is untouched.
  - Due diligence: run `benchmarks/bspline_template_vs_hardcoded` (which
    actually calls `BSplineND<double,4>::eval`, unlike
    `bspline_nd_optimization_bench` which benchmarks fitting) before/after;
    timings must be within run-to-run noise.
- **D5 — Math-layer error type is `InterpolationError`** (from
  `src/support/error_types.hpp`, so no layering violation): the closest peer
  `BSplineND::create` already returns
  `std::expected<..., InterpolationError>`, and it carries a code plus the
  offending index. Use `NaNInput` and `InfInput` as distinct codes.
- **D6 — Fail-loud beats fail-partial in the adaptive cache path.** A needed
  slice that is missing or invalid at extraction time is
  `PriceTableErrorCode::ExtractionFailed`, never a silent zero region. The
  intentional gap-segment placeholder (literal zeros, `Nt_seg == 0` branch)
  remains a separate, explicit case.

## Design

### #425 — CubicSpline finiteness validation

In `build()`, after the existing size checks and before the monotonicity scan,
validate **both** `x` and `y`:

```cpp
for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(x[i])) return "X contains non-finite values (NaN or Inf)";
for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(y[i])) return "Y contains non-finite values (NaN or Inf)";
```

The issue asks only for `y`, but the `x` guard is required too: the
strictly-increasing check uses `x[i] <= x[i-1]`, and every comparison against
NaN is false, so `{0, NaN, 2}` currently *passes* monotonicity.

`rebuild_same_grid()` gets the same `y` scan (x is unchanged there, already
validated by the prior `build()`).

Both functions already return `std::optional<std::string_view>`; no API
change. Performance: `rebuild_same_grid` has exactly one production caller —
the discrete-dividend event callback (`american_option.cpp:106`), invoked once
per dividend date per solve — and the O(n) scan is the same order as the
`std::copy` and Thomas solve already there. That callback ignores rebuild
errors (`if (err) return;`, skipping the shift); with the new guard this
remains safe-by-accident — the NaN is in the solution vector `u` itself, so it
keeps propagating through the PDE and surfaces regardless. Documented here;
improving that callback's error handling is out of scope.

### #426 — ChebyshevInterpolant build validation via std::expected

New signatures (both factories; the sampling `build()` delegates to
`build_from_values`, and NaN from a sampled function is equally dangerous):

```cpp
template <typename... Args>
[[nodiscard]] static std::expected<ChebyshevInterpolant, InterpolationError>
build_from_values(std::span<const double> values, const Domain<N>& domain,
                  const std::array<size_t, N>& num_pts, Args&&... storage_args);

template <typename... Args>
[[nodiscard]] static std::expected<ChebyshevInterpolant, InterpolationError>
build(std::function<double(std::array<double, N>)> f, const Domain<N>& domain,
      const std::array<size_t, N>& num_pts, Args&&... storage_args);
```

Validation order in `build_from_values` (all before node generation /
`Storage::build`; the sampling overload validates domain/num_pts before
sampling, then delegates):

1. every `num_pts[d] >= 2` → `InsufficientGridPoints` (axis in error);
2. every `domain.lo[d]`, `domain.hi[d]` finite and `lo[d] < hi[d]` →
   `NaNInput`/`InfInput`/`ZeroWidthGrid` (axis in error);
3. `values.size() == ∏ num_pts[d]`, product computed with overflow checking →
   `ValueSizeMismatch`;
4. every `values[i]` finite → `NaNInput` or `InfInput` with the offending
   index.

These mirror the invariants `reconstruct.hpp` already enforces for the same
type. Cost is one O(total) scan next to an existing O(total) copy.

**Call-site inventory** (all updated in this PR):

| Site | Enclosing error model | Adaptation |
|---|---|---|
| `chebyshev_adaptive.cpp:283` (gap placeholder, literal zeros) | `build_segment_leaves` | propagate like any other error (no `.value()` — domain/shape validation can still fail in principle) |
| `chebyshev_adaptive.cpp:324` (per-segment values) | `build_segment_leaves` | return type becomes `std::expected<std::vector<ChebyshevSegmentedLeaf>, PriceTableError>`; map `InterpolationError` → `convert_to_price_table_error` |
| both `build_segment_leaves` callers: BuildFn lambda (`:499`, returns `std::expected<SurfaceHandle, PriceTableError>`) and `build_chebyshev_segmented_pieces` (`:595`, returns `std::expected<ChebyshevSegmentedPieces, PriceTableError>`) | expected-valued | unwrap-or-propagate |
| `chebyshev_adaptive.cpp:441` (4D EEP values) | lambda → `std::expected<SurfaceHandle, PriceTableError>` | map via `convert_to_price_table_error` |
| `chebyshev_table_builder.cpp:185` | `std::expected<ChebyshevTableResult, PriceTableError>` | map via `convert_to_price_table_error` |
| `price_table_factory.cpp:648` | `std::expected<Chebyshev3DPriceTable, ValidationError>` | `detail::to_validation_error(convert_to_price_table_error(...))` — exactly the neighboring `BSplineND::create` handling |
| `reconstruct.hpp:125` | `std::expected<..., PriceTableError>` | map via `convert_to_price_table_error` (its existing size checks stay; NaN in persisted data now fails the load — see below) |
| `benchmarks/latency_sweep.cc:416`, `benchmarks/greek_latency.cc:511` | benchmark code | `.value()` |
| `tests/chebyshev_interpolant_test.cc`, `tests/chebyshev_surface_test.cc`, `tests/parquet_io_test.cc` (2 sites) | test bodies | `.value()` / `ASSERT_TRUE` |

**Deserialization policy (explicit):** no serialized schema changes, but a
previously persisted table containing NaN/Inf values used to load and now
fails. That is intended — such a table produces garbage prices — and gets a
regression test (reconstruct a segment with a NaN value, assert failure).

Python/Rust bindings do not expose these types directly (verified by grep); no
binding change.

### #466 — stop masking NaN as 0.0 at the surface boundaries

No `BSplineND` change. Three one-token operand swaps, `std::max(0.0, v)` →
`std::max(v, 0.0)`:

1. `src/option/table/transform_leaf.hpp:32` (`TransformLeaf::price`) — the
   query-time mask; after the swap, a NaN raw eval reaches the caller as NaN.
2. `src/option/table/eep/eep_decomposer.hpp:18` (`eep_floor`) — the build-time
   mask. Required for the #426 guard to ever fire on the motivating scenario:
   without it, NaN PDE values are floored to 0.0 *before*
   `build_from_values` sees them.
3. `src/option/american_option_result.cpp:45` (`value_at`) — same family in
   the FDM result path; NaN spot queries now return NaN instead of 0.0.

For all finite inputs `std::max(v, 0.0) == std::max(0.0, v)`; only NaN
handling differs (`-0.0` vs `+0.0` ties are indistinguishable by value).
`clamp_bspline_query` and the ±Inf clamp-to-edge behavior are untouched, and
`ChebyshevInterpolant::eval` needs no change (`std::clamp` passes NaN through;
the barycentric formula yields NaN — locked by a regression test).

Higher-level guards are unaffected: `validate_iv_query` still rejects
non-finite queries at the `InterpolatedIVSolver` boundary; this change fixes
direct surface callers (`BSplinePriceTable::price()` etc.).

### Adaptive cache: fail loudly on missing slices (D6)

In both extraction loops (`chebyshev_adaptive.cpp:301`, `:410`):
`cache.get_slice(...) == nullptr` for a slice that should exist (after
`solve_missing_pde_pairs`) becomes
`std::unexpected(PriceTableError{PriceTableErrorCode::ExtractionFailed})`
instead of `continue` over a zero-filled tensor. Both enclosing functions are
expected-valued after the #426 changes, so propagation is mechanical.
`store_slice` keeps its signature; the invalid flag now has a loud consumer.
(A failed pair staying in the cache means `missing_pairs()` won't retry it —
acceptable: the build fails with a clear error rather than retrying
indefinitely.)

## Tests (regression-test format per CLAUDE.md)

`tests/thomas_cubic_spline_test.cc` (existing CubicSpline test home):
- `BuildRejectsNaNY`, `BuildRejectsInfY`, `BuildRejectsNaNX`,
  `BuildRejectsInfX` (the X cases are the monotonicity blind spot),
  `RebuildSameGridRejectsNaNY`.

`tests/chebyshev_interpolant_test.cc`:
- `BuildFromValuesRejectsNaN` / `RejectsInf` (checks code and index),
  `RejectsSizeMismatch`, `RejectsNumPtsBelowTwo`, `RejectsNonFiniteDomain`,
  `RejectsReversedDomain`;
- `BuildRejectsNaNSampledFunction` (sampling overload);
- `EvalPropagatesNaNQuery` (locks existing behavior).

`tests/bspline_nd_test.cc`:
- `EvalPropagatesNaNQuery` — `eval`, `eval_partial`, `eval_second_partial`
  return NaN for a NaN in any coordinate position (locks the verified
  behavior against future basis "optimizations");
- `EvalStillClampsInfQuery` — ±Inf keeps clamp-to-edge behavior.

Surface/price layer:
- `TransformLeaf::price` (or the cheapest wrapper test around it) returns NaN
  for a NaN query coordinate; finite queries unchanged.
- `AmericanOptionResult::value_at(NaN)` returns NaN.
- Adaptive path: a cache with an invalid (NaN-input) slice makes the segment
  build return `ExtractionFailed` — proving a rejected spline can no longer
  become a silently-zero surface (regression for the motivating incident).
- Reconstruction: persisted segment with a non-finite value fails to load.

## Acceptance criteria

1. All existing tests pass (`bazel test //...`, 148/148 at baseline) with
   updated call sites; new regression tests pass.
2. `bazel build //benchmarks/...` and `bazel build //src/python:mango_option`
   still build (CI parity) — includes the two benchmark call-site updates.
3. `bspline_template_vs_hardcoded` eval timings before/after within
   run-to-run noise (D4; expected trivially true — no eval-path code change).
4. NaN input now yields an error (build paths) or NaN (query paths) — never a
   silent 0.0 or a fake success — at every boundary listed above.
