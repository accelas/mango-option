# Design: Moneyness coverage for the Chebyshev and dimensionless builders (#480)

**Issue:** [#480](https://github.com/accelas/mango-option/issues/480) — Chebyshev
and dimensionless adaptive builders solve gridless; same moneyness-coverage
defect as #437.

**Branch:** `fix/480-chebyshev-coverage`

**Parent fix:** PR #479 (spec
`docs/superpowers/specs/2026-08-31-moneyness-coverage-437-design.md`)
established the invariant for the B-spline builders: *every auto-estimated
table batch solve materializes one concrete PDE grid that covers the
log-moneyness axis, and passes it as `custom_grid`*. This change extends that
invariant to every remaining table builder.

## Problem

Every table builder solves a normalized batch (spot = strike = K_ref, so
x0 = 0), stores each slice's PDE solution as a `CubicSpline` over the
solver's own x-grid, and evaluates that spline at the table's log-moneyness
nodes. `CubicSpline::eval` (`src/math/cubic_spline_solver.hpp:204-214`)
extrapolates the last interval's cubic beyond its domain, so any node
outside the PDE domain holds polynomial garbage instead of a PDE value.

`BatchAmericanOptionSolver::solve_batch(batch, /*use_shared_grid=*/true)`
with no `custom_grid` sizes the PDE domain from `GridAccuracyParams` alone,
half-width `n_sigma · σ√T` with the default `n_sigma = 5`
(`src/option/grid_spec_types.cpp:43-46`). Nothing on the paths below relates
that width to the moneyness axis being sampled.

### Sites (all verified by reading the code)

| # | Site | Batch shape | Routing without `custom_grid` | Realized half-width |
|---|------|-------------|-------------------------------|---------------------|
| S1 | `chebyshev_adaptive.cpp:400-406` `make_chebyshev_build_fn` (continuous-yield adaptive) | missing (σ, r) pairs, no dividends, T = 1.01·τ_hi | eligible → `solve_normalized_chain`, per-(σ, r) group grid; **ineligible** → `solve_regular_batch`, batch-union grid | per group: `5·σᵢ√T`; union: `5·σ_max(batch)√T` |
| S2 | `chebyshev_adaptive.cpp:203-208` `solve_missing_pde_pairs` (segmented, discrete dividends) | missing pairs, dividends attached | `is_normalized_eligible` **rejects any batch with discrete dividends** (`american_option_batch.cpp:48-55`) → batch-union grid | `5·σ_max(batch)√T`, left edge extended by `ln(e^{x_min} − D/K)` |
| S3 | `chebyshev_table_builder.cpp:116-120` `build_chebyshev_table` (non-adaptive, not in the issue) | all (σ, r) CGL pairs, default `GridAccuracyParams` | as S1 | as S1 |
| S4 | `dimensionless_builder.cpp:49-55` `solve_dimensionless_pde` | one param per κ: σ = √2, r = κ, q = 0, T = 1.01·τ'_max | single-param batch: same grid either way | `5·√2·√T ≈ 7.1·√τ'_max` |
| S5 | `dimensionless_adaptive.cpp:40-57` `reference_eep` (adaptive loop's ground-truth probe) | one param, T = max(1.01·τ'₀, 0.02), spline read at probe x₀ | `use_shared_grid=false`, per-contract estimate | `7.1·√T ≥ 1.0` (maturity floor), reach needed is `|x₀|` |

The B-spline segmented builder (`bspline_segmented_builder.cpp:322-323`)
already materializes via `builder.estimate_pde_grid`, which calls
`ensure_moneyness_coverage`, and is **not** in scope.

### Required reach per site

- **S1** — the CC-extended node span `[m_lo, m_hi]` = user moneyness bounds
  ± `3·width/32` (`chebyshev_adaptive.cpp:687-698`, frozen at seed time and
  identical for every refinement level). The nodes are *evaluated* at
  those extremes, so the extension is part of the reach.
- **S2** — the CC-extended span from `compute_headroom` (`:846-851`) over
  the dividend-widened `domain_`; likewise frozen at `kInitLevels` for the
  adaptive path (`:919`) and level-dependent for the manual `build(...)`.
- **S3** — `config.domain.lo[0] .. hi[0]`.
- **S4** — `axes.log_moneyness.front() .. back()`.
- **S5** — the single probe point `x₀`.

### Premise corrections to the issue

1. **The Chebyshev batches are usually normalized-ineligible, not
   per-group.** `is_normalized_eligible` estimates the margin from the
   *first* batch param and requires `5·σ_first√T ≥ 0.35`
   (`american_option_batch.cpp:64-105`). `missing_pairs` enumerates
   σ-major (`pde_cache.hpp:89`), so on S1 the first missing pair carries
   the lowest *not-yet-cached* σ node: on the seed build that is `sigma_lo
   = max(σ_min − 3(σ_max−σ_min)/4, 0.01)` — 0.025 for a chain with vols
   0.10–0.20 — and on a σ-only refinement it is the lowest newly inserted
   node, which is still small. The margin is therefore far below 0.35
   unless T ≳ 8y. S2 is always ineligible because of the dividends. Both
   therefore solve on the batch-union grid, half-width `5·σ_max(batch)√T`,
   which covers every slice *only coincidentally*, whenever that product
   happens to exceed the node reach. The per-group routing the issue
   describes is the eligible corner (first-param σ large enough, or S2
   with an empty dividend list).
2. **Both routings are unsound and both are fixed by the same mechanism.**
   A concrete `PDEGridConfig` passed as `custom_grid` propagates verbatim
   into every normalized group (`american_option_batch.cpp:333-345`) and
   into the regular shared path (`:425-435`). Dividend mandatory times
   survive: `solve_regular_batch` rebuilds them per contract from the
   dividend schedule (`:448-457`) and `AmericanOptionSolver` merges them
   again (`american_option.cpp:70-76`), so the helper's empty
   `mandatory_times` is safe for S2.
3. **S3 exists.** `build_chebyshev_table` is a fifth gridless site with the
   default accuracy profile; it is in scope for the same reason #437's D6
   pulled in the non-adaptive B-spline fallback: same defect, same helper,
   and leaving it would break the invariant this change establishes.

### Why this is worse than #437

A Chebyshev interpolant is a global polynomial. A garbage value at an
endpoint node contaminates queries across the whole moneyness axis, not
just the tails, so the user-visible error appears at interior strikes too.
For S5, `reference_eep` is the adaptive loop's *ground truth*; an
extrapolated reference misdirects refinement silently. CLAUDE.md documents
`ChebyshevBackend` as **the** supported backend for adaptive
discrete-dividend surfaces (S2 is the production path behind it).

## Fix

### 1. Relocate the covering-grid helpers to a backend-neutral target

Move `detail::ensure_moneyness_coverage` and
`detail::materialize_covering_grid` (declared in `bspline_builder.hpp:236-253`,
defined in `bspline_builder.cpp:235-265`) to a new pair
`src/option/table/covering_grid.hpp` / `covering_grid.cpp`, Bazel target
`//src/option/table:covering_grid` with deps `//src/option:grid_spec_types`
and `//src/option:option_spec` only (the `PricingParams` and
`estimate_batch_pde_grid` it needs). Namespace stays `mango::detail`;
include path `mango/option/table/covering_grid.hpp`, following the
`strip_include_prefix`/`include_prefix` convention of the sibling targets in
`src/option/table/BUILD.bazel`.

`bspline_builder.hpp` replaces its two declarations with
`#include "mango/option/table/covering_grid.hpp"`, so every #479 call site
(`bspline_adaptive.cpp:263/272`, `bspline_builder.cpp:380`) compiles
unchanged. The helper's four existing unit tests
(`tests/price_table_builder_test.cc:429-495`) move with the code to a new
small `tests:covering_grid_test` (D7).

Bazel wiring, stated here so strict-deps failures are not left to
implementation discovery: the new target is `visibility =
["//visibility:public"]` like its siblings, and every target that includes
the header — directly or through `bspline_builder.hpp` — lists it as a
direct dep: `//src/option/table/bspline:bspline_builder`,
`//src/option/table/chebyshev:chebyshev_adaptive`,
`//src/option/table/chebyshev:chebyshev_table_builder`,
`//src/option/table/dimensionless:dimensionless_builder`,
`//src/option/table/dimensionless:dimensionless_adaptive`, and
`//tests:covering_grid_test`.

Contract widening (D3): the moneyness-axis argument is renamed
`log_moneyness_nodes` and its reach is computed as
`max(|min|, |max|)` over the span with `std::minmax_element`, dropping
#479's "sorted ascending" precondition. Chebyshev-Gauss-Lobatto nodes from
`chebyshev_nodes(n, lo, hi)` and CC-level nodes are handed in directly, so
the helper must not depend on `front()/back()` being the extremes. Empty
inputs stay a no-op.

### 2. Materialize a covering grid at every site

Each site keeps its accuracy profile, computes
`auto covering = detail::materialize_covering_grid(accuracy, batch, reach);`
and passes `covering` as the fourth `solve_batch` argument
(`solve_batch(batch, /*use_shared_grid=*/…, nullptr, covering)`). The
helper's estimator is driven by the `accuracy` value passed to it, not by
solver state; the existing `set_grid_accuracy` calls become redundant for
grid sizing but stay, since `grid_accuracy_` still feeds
`is_normalized_eligible` and therefore the routing choice. Routing no
longer affects the grid.

- **S1** `make_chebyshev_build_fn`: `reach = m_nodes` (already in scope of
  the lambda). Batch = the *missing* pairs, which is what the grid must be
  estimated over (same argument as #437 §3: on later iterations the missing
  batch's `max σ√T` can be smaller than the full batch's; estimating over
  the batch actually solved is what guarantees every new slice covers).
- **S2** `solve_missing_pde_pairs`: gains a
  `std::span<const double> m_nodes` parameter; both callers
  (`make_segmented_chebyshev_build_fn`, `build_chebyshev_segmented_pieces`)
  already hold it. `estimate_batch_pde_grid` sees the batch's dividends and
  applies its own left extension on top of the widened `n_sigma`; the
  union of the two is a superset of the reach either way.
- **S3** `build_chebyshev_table`: `reach = m_nodes`, accuracy =
  `GridAccuracyParams{}` (the solver's current default, made explicit).
- **S4** `solve_dimensionless_pde`: `reach = axes.log_moneyness`, batch =
  the single κ param; hoist the accuracy construction out of the κ loop.
- **S5** `reference_eep`: `reach = std::array{x0}`; the covering grid goes
  through the `use_shared_grid=false` path, where `resolved_custom_grid` is
  used per contract (`american_option_batch.cpp:466-470`). To make this
  site testable (review round 1), the file-static function becomes
  `double detail::dimensionless_reference_eep(double x0, double tau_prime_0,
  double ln_kappa_0, double K_ref, OptionType option_type)`, declared in
  `dimensionless_builder.hpp` (the header both dimensionless targets
  export) and defined where it is today; the adaptive loop calls it
  unchanged. Semantics are untouched: normalized EEP `max(American −
  European, 0)`, 0.0 on any solver failure.

### 3. Cache safety across refinement iterations (S1, S2)

Same invariants as #437: `ChebyshevPDECache::store_slice` builds each
slice's spline over the x-grid that slice was solved on
(`chebyshev_pde_cache.hpp:23-30`), so cohorts with different domains and
resolutions coexist; the node span `[m_lo, m_hi]` is frozen at seed time
(`build_adaptive_chebyshev` and `ChebyshevSegmentedBuilder::build_adaptive`
both size the headroom once from the initial CC levels), so the required
reach is constant across iterations and a slice that covered on iteration
1 still covers on iteration N. Widening only requires that *every newly
solved slice* covers the frozen span, which the materialized grid
guarantees for the whole batch under either routing.

The τ-axis guard is narrower than "any τ change clears the cache": both
build functions compare only `tau_nodes.size()` (`chebyshev_adaptive.cpp:377`,
`:523`). That is sufficient on the adaptive paths because the τ bounds are
frozen and a CC level's node count determines its nodes exactly, so equal
size implies equal τ vector. The manual `ChebyshevSegmentedBuilder::build
(cc_levels)` has level-dependent m-headroom, but it constructs a fresh
cache per call (`build_chebyshev_segmented_pieces`), so no slice is ever
reused across different node spans. This change relies on those two facts
and does not alter the guard.

### 4. Resolution and cost

For a dividend-free normalized param, `estimate_pde_grid` gives width
`2·n_sigma·σ√T` and `dx_target = σ·√tol` (`grid_spec_types.cpp:45-50`), so
`Nx = 2·n_sigma·√T/√tol` — **σ cancels**. Every param in an S1/S3/S4
batch therefore estimates the same `Nx`, and the batch union takes that
value: at Ultra and the 0.5y-floored τ axis (T ≈ 0.694) it is ≈ 3,725
points today. Widening `n_sigma` to `reach·1.1/(σ_max√T)` scales `Nx` by
the same factor: for the T2 chain below (reach ≈ 1.09, σ_hi ≈ 0.225) that
is ≈ 4,760 points, still under the 5,000 clamp, so the widened solves cost
roughly 28 % more spatial points (and proportionally more time steps,
since `dt` follows the smallest cell) than today's union grid. This is the
same resolution-for-coverage trade #437 D4 accepted for the B-spline path;
there is no new policy.

S2 differs: the dividend left-extension is applied per param *after* the
σ-scaled width, so low-σ params get a relatively larger extension and can
be the ones that set the union's `x_min`; that is pre-existing estimator
behaviour and the covering grid only adds the widened `n_sigma` on top.
The `MAX_WIDTH = 5.8` convergence limit remains unenforced against
materialized custom grids on every materializing path (pre-existing gap
noted in #437 D4; out of scope here).

## Regression tests

Every regression must be demonstrated **red on the parent revision** with
the pre-fix error recorded in the test comment, and its tolerance pinned
empirically well below that error (cross-toolchain loosening as in #479 is
allowed and must be justified in the comment). References are direct FDM
solves pinned to an explicit `GridAccuracyParams` profile, never the
solver default.

- **T1 — helper unit tests** (`tests:covering_grid_test`, small): the four
  #479 tests moved verbatim, plus one new case proving the reach is
  order-independent (a descending copy of a node array widens exactly as
  the ascending original).
- **T2 — S1 e2e** (`adaptive_surface_build_integration_test`). The chain
  must be derived against the full transformation chain, because
  `extract_chain_domain` (`adaptive_refinement.cpp:1147-1150`) first
  applies minimum spreads (m ≥ 0.10, τ ≥ 0.5y, σ ≥ 0.10, r ≥ 0.04, each
  centred on the chain's range and then clamped positive) and
  `build_adaptive_chebyshev` (`chebyshev_adaptive.cpp:679-698`) then adds
  CC headroom (`3·width/32` on m, `3·width/8` on τ, `3·width/4` on σ,
  clamped to `σ_lo ≥ 0.01`, `τ_lo ≥ 1e-4`). For any chain with maturities
  ≤ 0.5y that gives `τ_hi = 0.5 + 0.1875 = 0.6875`, PDE `T = 1.01·τ_hi`,
  `√T ≈ 0.833`; a single chain vol `v` gives sample σ `[v−0.05, v+0.05]`
  and `σ_hi = v + 0.125`. The old union half-width is thus
  `5·(v+0.125)·0.833 ≈ 4.17·(v+0.125)`, minimised by a small `v`. The
  node reach for strikes symmetric about spot, `K ∈ [S/a, S·a]`, is
  `ln a·(1 + 6/32) = 1.1875·ln a`. With `v = 0.10`: half-width ≈ 0.94,
  so `a > 2.2` undershoots; the spec's candidate is spot 100, strikes
  `{40, 60, 100, 160, 250}` (reach ≈ 1.09 vs 0.94 — both node endpoints
  ≈ 0.15 beyond the old domain), maturities `{0.05, 0.1}`, vols `{0.10}`,
  rates `{0.03, 0.05}`. The plan measures the pre-fix error on the parent
  revision and widens the strikes further if the extrapolated tail is not
  visibly wrong; the recorded number goes into the test comment.
  Assertions, at fixed K = K_ref and `S = K_ref·exp(m)` so that `m` is
  the surface's log-moneyness coordinate: (i) at both extreme *node*
  endpoints `m_lo/m_hi` read from the returned interpolant's `domain()`,
  and (ii) at the extreme user strikes (`m = ln(100/250)`, `ln(100/40)`);
  each at both the min-σ and max-σ sample bounds, on-axis τ and r. The
  user-strike assertions are what a global-polynomial contamination shows;
  the node assertions pin extraction. A put's deep-ITM (negative-m) end is
  where the extrapolation is expected to be visibly wrong; the deep-OTM end
  is asserted too but is not expected to be the red one.
- **T3 — S2 regression** through `build_chebyshev_segmented_manual` (fixed
  CC levels, no refinement, single `K_ref = spot` so no strike blend).
  `expand_segmented_domain` (`adaptive_refinement.cpp:1080-1121`) applies
  the same minimum spreads but keeps `τ_max = maturity`, so PDE
  `T = 1.01·maturity`. Candidate: maturity 0.25 (`√T ≈ 0.50`), one
  dividend of 1.0 at calendar time 0.1, vols `{0.10}` (σ_hi ≈ 0.225, old
  half-width ≈ 0.57), log-moneyness `{ln 0.5, 0, ln 2}` (reach ≈ 0.69,
  dividend-extended to ≈ 0.71 on the left, plus `3·1.41/32 ≈ 0.13` of
  headroom ≈ 0.84). Assert at the *queried* user moneyness endpoints
  (`S = 50` and `S = 200` at `K = 100`, both outside the old PDE domain
  by themselves, not only via hidden support nodes), at τ = maturity, at
  **both** σ sample endpoints, vs a discrete-dividend FDM solve built the
  way `make_validate_fn` builds its references (same `PricingParams`
  including `discrete_dividends`).
- **T4 — S4 unit** (`dimensionless_builder_test`): axes with
  `max|x| > 7.1·√τ'_max` (e.g. x ∈ [−0.7, 0.7], τ' ≤ 0.004, old
  half-width ≈ 0.45); compare `values` at the x extremes vs
  `solve_american_option` with spot = K·eˣ, strike = K, σ = √2, r = κ,
  q = 0, T = τ', divided by K.
- **T5 — S3 regression** (`chebyshev_surface_test`): `build_chebyshev_table`
  with a domain whose m reach exceeds `5·σ_hi·√τ_hi` (e.g. m ∈ [−0.7, 0.7],
  τ ∈ [0.01, 0.1], σ ∈ [0.10, 0.20]: old half-width ≈ 0.32); compare
  `surface.price` at the extreme m vs FDM at min-σ and max-σ.
- **T6 — S5 unit** (`dimensionless_adaptive_test` or
  `dimensionless_builder_test`): `detail::dimensionless_reference_eep` at a
  probe with `|x₀| > 1` (beyond the ≈ ±1.0 reach the 0.02 maturity floor
  gives today, e.g. x₀ = −1.3, τ'₀ = 0.005) vs the same normalized EEP
  computed directly: `solve_american_option` at spot = K·e^{x₀}, strike =
  K, σ = √2, r = κ, T = τ'₀, minus `dimensionless_european`, floored at 0.
  This also exercises the `use_shared_grid=false` custom-grid propagation
  path, which no other test covers.

Placement follows the CI doctrine (per-PR = short invariant tests; move a
test to the nightly `slow` split only if its measured runtime warrants it,
and never tag it `manual`).

## Expected behaviour changes and pinned numbers at risk

Node values that were extrapolated become PDE values; interior values may
shift at discretization level where the union grid was already covering,
because the domain and point placement change. Tests with pinned accuracy
numbers on affected paths must be re-run and, if they move, re-measured
with the new value recorded: `IVSolverFactorySegmented.DocumentedAdaptive
DiscreteDividendConfig` (549 bps, bound 0.10, nightly slow split),
`greeks_accuracy_test` (uses `build_chebyshev_table`),
`chebyshev_surface_test`, `dimensionless_*_test`, and
`price_table_data_test`'s Chebyshev round-trips. B-spline bit-identity
goldens are untouched (no B-spline builder code changes beyond the include
move).

## Decisions

Triage skipped the brainstorm (the issue ships location, mechanism, and
acceptance criteria). Three decisions were put to the user; the rest were
made from code analysis. All are for the design review to validate.

- **Q1 — Route.** Options: skip brainstorm and record decisions in the
  spec (recommended) / brainstorm first. **Chosen: skip.** Rationale: no
  user-facing interface changes; the open points are premise corrections
  and one placement choice.
- **Q2 / D1 — Helper home.** Options: (a) new backend-neutral
  `//src/option/table:covering_grid` with `bspline_builder.hpp` re-including
  it (recommended); (b) keep in `bspline_builder.hpp` and make the Chebyshev
  and dimensionless targets depend on the B-spline builder. **Chosen: (a).**
  Rationale: (b) is a cross-backend dependency seam the issue itself flags,
  and it drags the B-spline builder's transitive closure (collocation
  workspace, surface, tracing) into targets that need one free function.
  Rejected earlier (#437 D1) and still rejected: `grid_spec_types.hpp` —
  the helper encodes price-table semantics (normalized solve, moneyness
  axis), not solver semantics.
- **Q3 / D2 — Scope.** Options: fix all sites including `reference_eep`
  (recommended) / only the table-building sites. **Chosen: all.** S3
  (`build_chebyshev_table`) was found during the scope sweep and added
  under the same invariant. Rejected alternative: filing S3/S5 as
  follow-ups, which leaves known-unsound solves behind a fixed-invariant
  claim.
- **D3 — Order-independent reach.** Compute the reach with
  `std::minmax_element` instead of `front()/back()`. Rationale: CGL and CC
  node arrays are the natural argument at S1–S3 and their ordering is an
  implementation detail of the node generators; a silent precondition here
  would be exactly the kind of latent trap #441 hunted. Cost is O(n) on a
  few dozen nodes. Rejected: requiring callers to pass a sorted two-element
  `{lo, hi}` — pushes the precondition onto five call sites instead of
  removing it.
- **D4 — Estimate over the batch actually solved.** S1/S2 pass the
  *missing* batch, not the full node product (#437 D3 carried over).
- **D5 — Keep `set_grid_accuracy`.** It still governs routing eligibility
  and the estimator used inside `materialize_covering_grid`; removing it
  would silently change which path is taken. Rejected: dropping it as
  dead code.
- **D6 — No cap on the widening** (#437 D4 carried over). A cap would
  knowingly restore extrapolation; the `MAX_WIDTH` policy gap is
  pre-existing and out of scope.
- **D7 — Tests move with the code.** The #479 helper unit tests relocate to
  `tests:covering_grid_test`; `price_table_builder_test` loses them and its
  dependency surface is unchanged. Rejected: leaving them in
  `price_table_builder_test.cc` exercising a re-exported symbol, which ties
  the test to the include seam rather than the target.
- **D8 — Test strategy** as in *Regression tests*: one red-first regression
  per fixed site (S1–S5; S5 through a new `detail` seam, revised in review
  round 1 — the original "S5 by invariant only" left a site whose fix
  could be silently dropped, and the existing adaptive test cannot reach
  the defect); direct FDM oracles at extreme moneyness, at both σ extremes
  where the path has a σ axis (the min-σ assertion is what distinguishes a
  per-group undershoot from a union-grid undershoot). T2's chain is derived
  against the full domain-transformation chain (round 1: the first example
  was covered by the old union grid and would not have been red).
- **D9 — `reference_eep` becomes a `detail` function** rather than gaining
  a test-only hook. Rejected: a friend/test-access header — the function
  has no state to protect and the `detail` namespace is already the
  convention for shared-but-internal builder machinery (#437 D1).
