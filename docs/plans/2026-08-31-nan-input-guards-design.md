# NaN Input Guards — Design (issues #425, #426, #466)

Date: 2026-08-31
Issues: #425 (CubicSpline), #426 (ChebyshevInterpolant), #466 (BSplineND query)
Branch: one PR closes all three.

## Problem

Three interpolation entry points silently accept non-finite input and convert it
into plausible-looking garbage instead of an error. All three were traced during
the #419 investigation, where PDE instability produced NaN values that flowed
through the whole table-build stack and surfaced as prices of exactly 0.0
(via `std::max(0.0, NaN)`), hundreds of splines deep from the actual fault.

1. **#425** — `CubicSpline<T>::build()` / `rebuild_same_grid()`
   (`src/math/cubic_spline_solver.hpp`) validate sizes and x-monotonicity but
   not finiteness. NaN `y` flows through the Thomas solve into NaN
   coefficients; `build()` reports success.
2. **#426** — `ChebyshevInterpolant<N, Storage>::build_from_values()`
   (`src/math/chebyshev/chebyshev_interpolant.hpp`) does no input validation at
   all and returns the constructed object directly, so there is no error path.
   CI logs showed builds succeeding with 15–20% NaN input.
3. **#466** — `BSplineND<T, N>::eval()` family
   (`src/math/bspline/bspline_nd.hpp`): a NaN query coordinate falls through
   `clamp_bspline_query` (both comparisons false), then the degree-0 indicator
   comparisons in `cubic_basis_nonuniform` are all false, so eval returns
   exactly **0.0** — a plausible price — instead of NaN.

## Decisions (brainstorm Q&A)

- **D1 — #426 uses Option B (`std::expected`), not an assert.**
  User chose B. Rationale: `assert` is elided in opt builds, which is exactly
  where production price tables are built, so Option A would not guard the
  motivating scenario. `std::expected` matches the project error model
  (CLAUDE.md). Call-site count is modest (5 production, 2 test files).
- **D2 — #466 propagates NaN; no eval API change.**
  User chose NaN propagation over an error return. `eval()` is the ~250ns hot
  query path; changing its signature to `std::expected` would ripple through
  every surface caller. A NaN return is honest (NaN in → NaN out) and cheap.
- **D3 — Single branch/PR for all three.** Same theme, small diffs, one
  review cycle. PR closes #425, #426, #466.
- **D4 — Performance constraint (user, explicit): no significant perf impact.**
  - #425/#426 guards are build-time-only O(n) scans over data the functions
    already copy O(n); no query-path change.
  - #466 adds one `std::isnan` check per dimension per eval — a predictable,
    never-taken branch against a ~250ns query. No allocation, no signature
    change.
  - Verified by running `benchmarks/bspline_nd_optimization_bench.cc`
    before/after (eval timings must be within run-to-run noise).

## Design

### #425 — CubicSpline finiteness validation

In `build()`, after the existing size checks, validate **both** `x` and `y`:

```cpp
for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(x[i])) return "X contains non-finite values (NaN or Inf)";
}
for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(y[i])) return "Y contains non-finite values (NaN or Inf)";
}
```

The issue asks only for `y`, but the `x` guard is required too: the existing
strictly-increasing check uses `x[i] <= x[i-1]`, and every comparison against
NaN is false, so `{0, NaN, 2}` currently *passes* the monotonicity check.
With the finiteness check first, the monotonicity check regains its intended
meaning.

`rebuild_same_grid()` gets the same `y` scan (x is unchanged there, already
validated by the prior `build()`).

Both functions already return `std::optional<std::string_view>`; no API change.
`rebuild_same_grid()` is called per-slice during table builds — the O(n) scan
is the same order as the `std::copy` and Thomas solve already in the function.

### #426 — ChebyshevInterpolant::build_from_values returns std::expected

New signature:

```cpp
template <typename... Args>
[[nodiscard]] static std::expected<ChebyshevInterpolant, std::string_view>
build_from_values(std::span<const double> values,
                  const Domain<N>& domain,
                  const std::array<size_t, N>& num_pts,
                  Args&&... storage_args);
```

Validation before `Storage::build`: every `values[i]` must be finite, else
`std::unexpected("build_from_values: input contains non-finite values")`.

Error type is `std::string_view`: `ChebyshevInterpolant` lives in `src/math`
and must not depend on option-layer error enums; this matches the
`CubicSpline::build` / `PDEWorkspace::from_buffer` convention.

The sampling overload `build(f, domain, num_pts, ...)` delegates to
`build_from_values`, so it becomes
`std::expected<ChebyshevInterpolant, std::string_view>` as well — NaN from a
sampled function is exactly as dangerous as NaN in a value array.

Additional validation while we are here: `values.size()` must equal
`∏ num_pts[d]` (currently a size mismatch reads out of bounds or silently
truncates inside `Storage::build`). Same error-path mechanics, negligible cost.

**Call-site inventory** (all updated in this PR):

| Site | Enclosing error model | Adaptation |
|---|---|---|
| `src/option/table/chebyshev/chebyshev_adaptive.cpp:283` (placeholder zeros leaf) | `build_segment_leaves` returns `std::vector<ChebyshevSegmentedLeaf>` | cannot fail (literal zeros); `.value()` with a comment, or propagate (see below) |
| `src/option/table/chebyshev/chebyshev_adaptive.cpp:324` (per-segment values from cached splines — the real NaN risk) | same | `build_segment_leaves` return type becomes `std::expected<std::vector<ChebyshevSegmentedLeaf>, PriceTableError>`; the BuildFn lambda callers already return `std::expected<SurfaceHandle, PriceTableError>` and propagate |
| `src/option/table/chebyshev/chebyshev_adaptive.cpp:441` (4D EEP values) | lambda returns `std::expected<SurfaceHandle, PriceTableError>` | map error → `PriceTableError{PriceTableErrorCode::SurfaceBuildFailed}` |
| `src/option/table/chebyshev/chebyshev_table_builder.cpp:185` | function returns an expected of `ChebyshevTableResult` | map → `SurfaceBuildFailed` |
| `src/option/price_table_factory.cpp:648` | factory returns expected | map → `SurfaceBuildFailed` (match the neighboring `BSplineND::create` handling) |
| `src/option/table/serialization/reconstruct.hpp:125` | returns `std::expected<..., PriceTableError>` | map → `PriceTableError{PriceTableErrorCode::InvalidConfig}` (consistent with the function's existing size-validation errors) |
| `tests/chebyshev_interpolant_test.cc`, `tests/chebyshev_surface_test.cc`, `tests/parquet_io_test.cc` (2 sites) | test bodies | `.value()` / `ASSERT_TRUE(result.has_value())` |

No new `PriceTableErrorCode` enum value: `SurfaceBuildFailed` and
`InvalidConfig` already carry the right meaning at their respective layers.
Python/Rust bindings do not expose these types directly (verified by grep);
no binding change.

### #466 — NaN query coordinates propagate through BSplineND eval

Key insight vs. the issue's suggested fix: `clamp_bspline_query` already
*returns* NaN unchanged (both comparisons false → `return x`). The zeroing
happens downstream, in the degree-0 indicator comparisons of
`cubic_basis_nonuniform`. So an `isnan` check inside the clamp alone changes
nothing — the guard must short-circuit eval itself.

Fix, in each of `eval()`, `eval_partial()`, `eval_second_partial()`
(the three users of `clamp_bspline_query`), inside the existing per-dimension
loop before the span lookup:

```cpp
if (std::isnan(query[dim])) [[unlikely]] {
    return std::numeric_limits<T>::quiet_NaN();
}
```

- ±Inf coordinates keep their current behavior (clamped to the domain edge) —
  that is consistent clamping semantics, and #466 is about NaN specifically.
- `ChebyshevInterpolant::eval` needs **no change**: `std::clamp` passes NaN
  through and the barycentric formula yields NaN naturally. A regression test
  locks this in so a future "optimization" cannot reintroduce the 0.0 masking
  there.
- Higher-level guards are unaffected: `validate_iv_query` still rejects
  non-finite queries at the `InterpolatedIVSolver` boundary; this change fixes
  direct surface callers (`BSplinePriceTable::price()` etc.).

## Tests (regression-test format per CLAUDE.md)

`tests/thomas_cubic_spline_test.cc` (existing CubicSpline test home):
- `BuildRejectsNaNY`, `BuildRejectsInfY` — error returned, per issue #425.
- `BuildRejectsNaNX` — regression for the monotonicity-check blind spot.
- `RebuildSameGridRejectsNaNY`.

`tests/chebyshev_interpolant_test.cc`:
- `BuildFromValuesRejectsNaN`, `BuildFromValuesRejectsInf` —
  `EXPECT_FALSE(result.has_value())`.
- `BuildFromValuesRejectsSizeMismatch`.
- `EvalPropagatesNaNQuery` — locks the existing NaN-propagation behavior.

`tests/bspline_nd_test.cc`:
- `EvalReturnsNaNForNaNQuery` — `eval`, `eval_partial`, `eval_second_partial`
  each return NaN when any single coordinate is NaN (test a NaN in each
  position for at least `eval`).
- `EvalStillClampsInfQuery` — ±Inf keeps clamp-to-edge behavior.

## Acceptance criteria

1. All existing tests pass (`bazel test //...`, 148/148 at baseline) with the
   updated call sites; new regression tests pass.
2. `bazel build //benchmarks/...` and `bazel build //src/python:mango_option`
   still build (CI parity).
3. `bspline_nd_optimization_bench` eval timings before/after within
   run-to-run noise (D4).
4. NaN input to any of the three entry points now yields an error
   (build paths) or NaN (query path) — never a silent 0.0 or success.
