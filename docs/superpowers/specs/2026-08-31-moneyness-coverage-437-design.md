# Design: Moneyness coverage for the adaptive cached build path (#437)

**Issue:** [#437](https://github.com/accelas/mango-option/issues/437) — Adaptive
cached build path skips `ensure_moneyness_coverage`; tensor tails can be
spatially extrapolated.

**Branch:** `fix/437-moneyness-coverage`

## Problem

The non-adaptive `PriceTableBuilderND<4>::solve_batch`
(`src/option/table/bspline/bspline_builder.cpp`) widens the PDE spatial
domain before solving, on both of its auto-grid paths:

- `estimate_pde_grid` (`:266`) calls `ensure_moneyness_coverage` before
  `estimate_batch_pde_grid`;
- the explicit-grid fallback (`:365`) calls it after deriving fallback
  `GridAccuracyParams`.

`ensure_moneyness_coverage` (`bspline_builder.cpp:237-257`) raises
`accuracy.n_sigma` so that `n_sigma · max σ√T ≥ max(|log m_min|, |log m_max|) · 1.1`.
The batch solves are normalized (spot = strike = K_ref, so x0 = 0), and the
per-param PDE half-width is `n_sigma · σ√T` (`grid_spec_types.cpp:45-46`,
default `n_sigma = 5`), so this guarantees the PDE domain covers the full
log-moneyness axis of the table.

The adaptive cached path never does this. `build_adaptive_bspline` →
`build_cached_surface` → `solve_missing_slices`
(`src/option/table/bspline/bspline_adaptive.cpp:173-253`) bypasses
`PriceTableBuilderND::solve_batch` (to reuse the `BSplinePDECache`) and
re-implements the grid dispatch **without** the coverage widening:

- the `GridAccuracyParams` branch (`:246-249`) passes the accuracy straight
  through to `BatchAmericanOptionSolver`;
- the `PDEGridConfig` fallback branch (`:221-243`) replicates the builder's
  fallback minus the `ensure_moneyness_coverage` call.

This path is what production hits: `build_bspline_continuous_table`
(`src/option/price_table_factory.cpp:299`) calls `build_adaptive_bspline`
with `make_grid_accuracy(GridAccuracyProfile::High)` — the
`GridAccuracyParams` branch.

**Consequence.** When `n_sigma · max σ√τ` undershoots the moneyness axis,
`extract_tensor` (`bspline_builder.cpp:466-468`) evaluates the per-slice
`CubicSpline` beyond its domain. `CubicSpline::eval`
(`src/math/cubic_spline_solver.hpp:204-214`) extrapolates the last
interval's cubic polynomial, so extreme-moneyness tensor points hold
polynomial-extrapolated garbage rather than PDE solutions. Adaptive
validation (fresh FDM solves) then sees persistent boundary error that
moneyness refinement can never fix — it is a domain problem, not a density
problem — producing refinement churn or a silently accepted error.

**Concrete failure scenario** (from the issue): chain with τ_max = 0.1y,
vols ≤ 0.20, strikes spanning ±40% of spot. Half-width = 5 · 0.2 · √0.1 ≈
0.316; required |log m| reaches ln(100/60) ≈ 0.51.

## Fix

Hoist `ensure_moneyness_coverage` into a shared free function, call it from
both `solve_missing_slices` branches that estimate a grid from
`GridAccuracyParams`, and add the non-adaptive path's upfront explicit-grid
coverage rejection to the adaptive path.

### 1. Hoist the helper

Move `ensure_moneyness_coverage` from a file-static template in
`bspline_builder.cpp` to a free function declared in
`src/option/table/bspline/bspline_builder.hpp` (namespace `mango::detail` —
shared builder machinery, not general public API; implementation stays in
`bspline_builder.cpp`):

```cpp
namespace detail {
/// Ensure accuracy.n_sigma is large enough that a normalized batch solve
/// (spot = strike = K_ref, x0 = 0) spans the whole log-moneyness axis.
/// The baseline PDE half-width is n_sigma * max(σ√T) over `batch`; if that
/// undershoots max(|m_front|, |m_back|), extract_tensor would extrapolate.
/// No-op when `batch` or `log_moneyness_grid` is empty; expects the grid
/// sorted ascending (as validated table axes are).
void ensure_moneyness_coverage(GridAccuracyParams& accuracy,
                               std::span<const PricingParams> batch,
                               std::span<const double> log_moneyness_grid);
}
```

The `template <size_t N>` on the current version exists only to read
`axes.grids[0]`; the hoisted function takes the log-moneyness grid directly
and drops the template. Existing call sites pass `axes.grids[0]`. Logic is
unchanged for nonempty inputs: max σ√T over the batch, floored at 1e-10;
`MARGIN = 1.1`; `n_sigma = max(n_sigma, required)`. The empty-input no-op is
the one addition: the exported function must not read `front()/back()` of an
empty span or divide a wide axis by the 1e-10 floor for an empty batch
(which would inflate `n_sigma` absurdly instead of doing nothing).

### 2. Call it in `solve_missing_slices`

`solve_missing_slices` gains a `std::span<const double> m_grid` parameter
(the log-moneyness axis, available in `build_cached_surface`), and calls
the helper in two places:

- **`GridAccuracyParams` branch:** copy the accuracy params, apply
  `ensure_moneyness_coverage(accuracy, missing_params, m_grid)`, then
  `set_grid_accuracy` + solve. (The variant member is `const`; the copy is
  required anyway.)
- **`PDEGridConfig` fallback branch:** after deriving the fallback
  accuracy (n_sigma from `max_abs_x`), also apply
  `ensure_moneyness_coverage(accuracy, missing_params, m_grid)` — mirroring
  `bspline_builder.cpp:365`.

The **constraints-met explicit-grid branch itself is left alone** (a
constraints-met, coverage-validated explicit grid is used verbatim, as in
the non-adaptive path) — but see §2b: the adaptive path must first gain the
upfront coverage *rejection* that the non-adaptive `build()` applies to
explicit grids, which is what actually keeps that branch safe.

### 2b. Upfront explicit-grid coverage rejection

The non-adaptive `build()` **rejects** a `PDEGridConfig` whose spatial
bounds do not cover the moneyness axis (`bspline_builder.cpp:73-84`,
returns `InvalidConfig`) before ever solving. The adaptive path bypasses
`build()` entirely (`from_vectors` + `assemble_surface` directly), so a
constraints-met explicit grid that undershoots the axis would still be
extrapolated — the original spec's claim that this branch was "symmetric on
both paths" was wrong.

Add the same check to `build_cached_surface`, before any solving: if
`pde_grid` holds a `PDEGridConfig` and `m_grid.front() < x_min` or
`m_grid.back() > x_max`, return
`PriceTableError{PriceTableErrorCode::InvalidConfig}`. Notes:

- Rejection happens per build iteration; since moneyness bounds never grow
  during refinement (midpoint insertion only), a grid accepted on iteration
  1 stays accepted — no mid-run refusal churn.
- With this check in place, any explicit grid *reaching*
  `solve_missing_slices` covers the axis; the fallback-branch
  `ensure_moneyness_coverage` call is still required because the fallback
  *discards* the explicit grid and re-estimates from `GridAccuracyParams`,
  whose `n_sigma · max σ√τ` width is independent of the validated explicit
  bounds — the same reason `build()`'s own fallback calls it
  (`bspline_builder.cpp:365`) despite the upfront check.

### 3. Compute over `missing_params`, not `all_params`

The solve is `solve_batch(missing_params, true)`, and
`estimate_batch_pde_grid` (`grid_spec_types.cpp:120-177`) unions per-param
estimates over **the batch it is given**. The realized half-width is
therefore `n_sigma · max σ√τ(missing_params)`. On refinement iterations
after the first, `missing_params` contains only newly inserted (interior)
σ values, so `max σ√τ(missing) ≤ max σ√τ(all)`; an `n_sigma` derived from
`all_params` would still undershoot. The coverage call must use
`missing_params`.

For the same reason, the fallback branch's existing
`required_n_sigma = max_abs_x / max_sigma_sqrt_tau` computation switches
its denominator from `all_params`' max to `missing_params`' max — with the
`all_params` value, the emulated explicit-grid width
`n_sigma · max σ√τ(missing)` falls short of `max_abs_x · 1.1` whenever the
missing batch's max σ√τ is below the full batch's. The
`grid_meets_constraints` check itself keeps using `all_params` (it
validates the caller's explicit grid against the whole problem, and a
per-iteration answer flipping between branches would be a behavior change
with no benefit).

## Why this is safe across refinement iterations

- The refiner (`make_bspline_refine_fn`) only inserts midpoints; axis
  *bounds* never change during a run, so the required half-width is
  constant across iterations.
- The `pde_grid` (`PDEGridSpec`) is immutable for the whole
  `build_adaptive_bspline` run — this is a load-bearing assumption: the
  `BSplinePDECache` key is `(σ, rate)` only (plus tau-grid invalidation,
  `bspline_pde_cache.hpp`), so cache reuse would be unsafe if accuracy
  params or domain bounds could change between iterations without
  invalidation. They cannot: `pde_grid` is captured by the `BuildFn` once,
  and the widened accuracy derives only from it, the fixed axis bounds, and
  the missing batch.
- Cached slices each carry their own PDE grid, and `extract_tensor`
  re-splines each slice from its own grid — slices solved with different
  widths coexist fine as long as each covers the moneyness axis. With the
  fix, every fresh solve does; mixed-resolution slices across iterations
  are pre-existing behavior (the missing batch already differs per
  iteration).
- A wider domain can push the fresh missing-batch `Nx` into the
  `max_spatial_points` clamp (with the estimator's +1 odd-point adjustment,
  `grid_spec_types.cpp:50-54`, so the cap is not a strict ceiling), trading
  interior resolution for coverage — exactly the trade-off the non-adaptive
  path already makes; no new policy.
- `n_sigma · max σ√T` is the *baseline symmetric* half-width
  (`grid_spec_types.cpp:45-46`); discrete-dividend handling can only
  *extend* the left boundary further (`:56-69`), so it is a lower bound on
  realized coverage. (The adaptive continuous path passes no discrete
  dividends anyway.)

## Regression tests

1. **Unit test for the hoisted helper** (fast, per-PR CI): given
   `n_sigma = 5`, a batch whose max σ√T is small, and a wide log-moneyness
   grid, assert `n_sigma` is raised to
   `max(|m_front|, |m_back|) / max σ√T · 1.1`; given ample σ√T, assert
   `n_sigma` is untouched; empty batch and empty grid are no-ops.
   Location: `tests/price_table_test.cc` (or wherever `PriceTableBuilder`
   unit tests live).

2. **Explicit-grid rejection test** (fast): `build_adaptive_bspline` with a
   `PDEGridConfig` whose bounds undershoot the chain's moneyness axis
   fails with `InvalidConfig`, matching non-adaptive `build()`.

3. **End-to-end adaptive regression** (issue's sketch): adaptive build via
   `build_adaptive_bspline` with `make_grid_accuracy(High)` on a chain with
   τ_max = 0.1, vols ≤ 0.20, strikes spanning ±40% of spot (spot 100,
   strikes down to 60 ⇒ required half-width ≈ 0.51 vs. unfixed
   5·0.2·√0.1 ≈ 0.316 — the short-maturity/highest-vol combination that
   establishes the unfixed width is part of the chain). Oracle details:
   - `BSplineAdaptiveResult::spline` stores normalized EEP-transformed
     values, not comparable prices; the test queries a wrapper created via
     `make_bspline_surface(result->spline, K_ref, q, type)` — the same
     reconstruction the factory uses — and compares full American prices
     against direct `solve_american_option` references at identical
     (S, K, τ, σ, r).
   - Query points sit at the *extreme fit-axis endpoints*
     (`result->axes.grids[0].front()/.back()`, which include adaptive
     B-spline support headroom beyond the user strikes) — directly
     verifying `extract_tensor` never extrapolated — plus the extreme user
     strikes.
   - Tolerance is pinned empirically during implementation, with the
     pre-fix error on the parent revision recorded in the test comment;
     at least one assertion must be demonstrated red on the parent
     revision (TDD), and the tolerance must sit well below that pre-fix
     error so the test discriminates domain coverage from ordinary
     interpolation error.
   - Placement: `tests:adaptive_surface_build_integration_test` (already
     `large` + sharded in `tests/BUILD.bazel`); move to the nightly `slow`
     split only if measured runtime warrants it (CI doctrine).

## Decisions

Triage skipped the brainstorm: the issue ships the fix location, the
mechanism, and acceptance criteria; no user-facing interface is touched.
The judgment calls below were made from code analysis and are what the
design review should validate:

- **D1 — Hoist vs. duplicate:** hoist `ensure_moneyness_coverage` to a
  declared free function in `bspline_builder.hpp`, namespace
  `mango::detail`, rather than copying the logic into
  `bspline_adaptive.cpp`. Rationale: the issue's own suggestion; a copy
  would re-create the asymmetry the next time the formula changes; `detail`
  signals shared builder machinery, not public API. Rejected alternative:
  moving it to `grid_spec_types.hpp` — it is price-table-specific
  (normalized-solve moneyness semantics), and `bspline_adaptive.cpp`
  already includes `bspline_builder.hpp`. (Revised in design review round
  1: added `detail` namespace + empty-input no-op contract.)
- **D2 — Which branches:** widen in the `GridAccuracyParams` branch (the
  production path) and the `PDEGridConfig` fallback branch (parity with
  `solve_batch:365`), and additionally add `build()`'s upfront
  explicit-grid coverage *rejection* (`bspline_builder.cpp:73-84`) to
  `build_cached_surface` (§2b). (Revised in design review round 1: the
  original claim that the constraints-met branch was symmetric on both
  paths was wrong — the adaptive path bypasses `build()`'s validation.)
- **D3 — Which batch:** compute coverage (and the fallback branch's
  `required_n_sigma` denominator) over `missing_params` — the batch the
  solver actually sizes the shared grid from — not `all_params`.
  Rejected alternative: `all_params`, which undershoots on
  post-first refinement iterations (see §3).
- **D4 — No cap on the widening:** parity with the non-adaptive path,
  which imposes none. The realistic widened width here (~1.1 log-units for
  ±40% strikes) is far below the `MAX_WIDTH = 5.8` convergence limit;
  inventing a cap would be new policy outside the issue's scope.
- **D5 — Test strategy:** fast unit tests on the helper + explicit-grid
  rejection test + one end-to-end adaptive regression reproducing the
  issue's scenario (oracle: `make_bspline_surface`-reconstructed prices at
  extreme fit-axis endpoints vs. direct FDM), red before the fix. (Revised
  in design review round 1: oracle pinned — the raw result spline holds
  normalized EEP values and is not directly comparable.)
