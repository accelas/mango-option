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
/// Ensure accuracy.n_sigma is large enough that a shared grid estimated by
/// estimate_batch_pde_grid(batch, accuracy) for a normalized batch
/// (spot = strike = K_ref, x0 = 0) spans the whole log-moneyness axis:
/// that grid's baseline half-width is n_sigma * max(σ√T) over `batch`, and
/// if it undershoots max(|m_front|, |m_back|), extract_tensor would
/// extrapolate. Callers must actually solve on such an estimated grid
/// (passed as a concrete custom grid); BatchAmericanOptionSolver's own
/// gridless routing estimates per normalized group instead and does NOT
/// realize this width. No-op when `batch` or `log_moneyness_grid` is
/// empty; expects the grid sorted ascending (as validated table axes are).
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

### 2. Materialize a covering shared grid in `solve_missing_slices`

**Widening `GridAccuracyParams` alone is NOT sufficient** (design review
round 2). `solve_batch(missing_params, true)` with no custom grid routes
through `solve_normalized_chain` (`american_option_batch.cpp:293-295`,
eligible here: normalized batch, no discrete dividends, no callback), which
groups by (σ, r, q, type, maturity) and solves **one normalized param per
group** via `solve_regular_batch({single param}, true, setup, custom_grid)`
(`:333-345`). With `custom_grid == nullopt`, each group's grid is
`estimate_batch_pde_grid({that param}, accuracy)` — half-width
`n_sigma · σᵢ√T`, proportional to *that slice's own σ*. A widened `n_sigma`
sized by `max σ√T` therefore fixes only the max-σ slice; every lower-σ
slice still undershoots by the factor σᵢ/σ_max. (The normalized eligibility
margin check guards coverage around x = 0, not the table's moneyness axis.)

The correct mechanism — the one the non-adaptive `GridAccuracyParams`
branch already uses (`bspline_builder.cpp:296-299`) — is to **materialize a
concrete `PDEGridConfig` and pass it as `custom_grid`**, which propagates
verbatim into every normalized group (`american_option_batch.cpp:345`,
`:433-435`). Both auto-estimation branches of `solve_missing_slices` do,
with `m_grid` (the log-moneyness axis) passed down from
`build_cached_surface`:

1. copy/derive the `GridAccuracyParams`;
2. `detail::ensure_moneyness_coverage(accuracy, missing_params, m_grid)`;
3. `estimate_batch_pde_grid(missing_params, accuracy)`;
4. wrap the result as `PDEGridConfig` with
   `TimeDomain::from_n_steps(0, tau_grid.back(), n_steps)`;
5. `solve_batch(missing_params, true, nullptr, custom_grid)`.

Concretely per branch:

- **`GridAccuracyParams` branch:** steps 1–5 replace the current
  `set_grid_accuracy` + gridless solve.
- **`PDEGridConfig` fallback branch:** keeps its existing fallback-accuracy
  derivation (n_sigma from `max_abs_x`), then applies steps 2–5 —
  strengthening the current gridless `set_grid_accuracy` + solve, which has
  the same per-group routing defect.

Since steps 2–5 are identical in both branches (and in spirit to the
non-adaptive primary branch), the plan may encapsulate them as a small
shared utility (e.g. `detail::solve_batch_with_covering_grid` or a
grid-materializing helper next to `ensure_moneyness_coverage`); exact shape
is an implementation detail.

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

The materialized grid comes from
`estimate_batch_pde_grid(missing_params, accuracy)`
(`grid_spec_types.cpp:120-177`), which unions per-param estimates over
**the batch it is given** — so the realized shared half-width is
`n_sigma · max σ√τ(missing_params)` (baseline, before any dividend
extension). On refinement iterations after the first, `missing_params`
contains only newly inserted (interior) σ values, so
`max σ√τ(missing) ≤ max σ√τ(all)`; an `n_sigma` derived from `all_params`
would still undershoot. The coverage call must use `missing_params`.

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

### 4. Non-adaptive fallback gets the same materialization

"Parity with the non-adaptive path" is not a soundness argument for the
gridless solve: `PriceTableBuilderND::solve_batch`'s own explicit-grid
fallback (`bspline_builder.cpp:365-368`, `set_grid_accuracy` + gridless
`solve_batch`) has the identical per-normalized-group routing defect. Since
this change establishes the invariant *every auto-estimated batch solve
materializes a concrete shared grid that covers the moneyness axis*, and
the fix is the same steps 2–5 three lines from code we already touch, the
non-adaptive fallback is corrected here too rather than left as a filed
follow-up. Behavior change is confined to the rarely-hit
explicit-grid-fails-stability-constraints path, and aligns it with the
builder's primary branch. (The non-adaptive primary branch already
materializes; it keeps its `estimate_pde_grid` + maturity-extension shape.)

## Why this is safe across refinement iterations

The real cache-safety invariants (the widened accuracy, estimated
resolution, and concrete domain **can** differ between iterations, because
they derive from each iteration's `missing_params` — that is fine):

- `pde_grid` (the `PDEGridSpec` input), option type, continuous yield,
  `K_ref`, and the fit-axis *bounds* are fixed for the whole
  `build_adaptive_bspline` run (the refiner only inserts midpoints), so the
  required half-width is constant across iterations.
- A tau-grid change invalidates the entire cache
  (`bspline_pde_cache.hpp`, `invalidate_if_tau_changed`).
- Each cached slice owns its concrete PDE grid, and `extract_tensor`
  re-splines each slice from its own grid — slices with different
  domains/resolutions across `(σ, rate)` cohorts coexist safely.
- The one requirement is therefore that **every newly solved slice
  independently covers the fixed moneyness bounds** — which the
  materialized covering grid guarantees for all slices of the batch,
  including every normalized group.
- A wider domain can push the fresh missing-batch `Nx` into the
  `max_spatial_points` clamp (with the estimator's +1 odd-point adjustment,
  `grid_spec_types.cpp:50-54`, so the cap is not a strict ceiling), trading
  interior resolution for coverage — exactly the trade-off the non-adaptive
  primary branch already makes; no new policy.
- `n_sigma · max σ√T` is the *baseline symmetric* half-width of the
  estimated grid (`grid_spec_types.cpp:45-46`); discrete-dividend handling
  can only *extend* the left boundary further (`:56-69`), so it is a lower
  bound on realized coverage. (The adaptive continuous path passes no
  discrete dividends anyway.)

## Regression tests

1. **Unit test for the hoisted helper** (fast, per-PR CI): given
   `n_sigma = 5`, a batch whose max σ√T is small, and a wide log-moneyness
   grid, assert `n_sigma` is raised to
   `max(|m_front|, |m_back|) / max σ√T · 1.1`; given ample σ√T, assert
   `n_sigma` is untouched; empty batch and empty grid are no-ops.
   Location: `tests/price_table_builder_test.cc`.

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
   - Endpoint comparisons run at **both the maximum-σ and the minimum-σ
     axis values** (and on-axis τ/r values, isolating PDE-domain/extraction
     behavior from cross-axis B-spline interpolation). The min-σ assertion
     is the guard against the per-normalized-group routing defect (§2): a
     max-σ-only test could pass under an incomplete `n_sigma`-widening fix
     while every lower-σ slice still extrapolates.
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
- **D2 — Which branches & mechanism:** in both `solve_missing_slices`
  auto-estimation branches, widen the accuracy AND **materialize the
  estimated grid as a concrete `PDEGridConfig` passed as `custom_grid`**
  (§2); add `build()`'s upfront explicit-grid coverage *rejection*
  (`bspline_builder.cpp:73-84`) to `build_cached_surface` (§2b); correct
  the non-adaptive fallback the same way (§4). (Revised in round 1: the
  constraints-met-branch "symmetry" premise was false. Revised in round 2:
  accuracy widening without a concrete custom grid is bypassed by
  per-normalized-group grid estimation in `solve_normalized_chain` — only
  a materialized grid propagates to every group.)
- **D3 — Which batch:** compute coverage (and the fallback branch's
  `required_n_sigma` denominator) over `missing_params` — the batch the
  materialized grid is estimated from — not `all_params`. Rejected
  alternative: `all_params`, which undershoots on post-first refinement
  iterations (see §3). (Round 2 note: this premise only holds because the
  estimated grid is now explicitly materialized and passed as
  `custom_grid`; gridless solving would not realize it.)
- **D4 — No cap on the widening:** parity with the non-adaptive path,
  which imposes none. The realistic widened width here (~1.1 log-units for
  ±40% strikes) is far below the `MAX_WIDTH = 5.8` convergence limit;
  inventing a cap would be new policy outside the issue's scope.
- **D5 — Test strategy:** fast unit tests on the helper + explicit-grid
  rejection test + one end-to-end adaptive regression reproducing the
  issue's scenario (oracle: `make_bspline_surface`-reconstructed prices at
  extreme fit-axis endpoints vs. direct FDM), red before the fix. (Revised
  in round 1: oracle pinned — the raw result spline holds normalized EEP
  values and is not directly comparable. Revised in round 2: endpoint
  assertions required at both max-σ and min-σ so an incomplete
  widening-only fix cannot pass.)
- **D6 — Non-adaptive fallback included (round 2):** correct
  `bspline_builder.cpp:365-368` with the same materialization rather than
  filing a follow-up — same defect class, same helper, three lines from
  touched code; establishes the covering-grid invariant across all
  auto-estimating builder branches. Rejected alternative: follow-up issue
  (leaves a known-unsound branch behind a "parity" claim this spec no
  longer makes).

Implementation notes carried from review: `bspline_builder.hpp` gains a
direct `<span>` include when the declaration lands.
