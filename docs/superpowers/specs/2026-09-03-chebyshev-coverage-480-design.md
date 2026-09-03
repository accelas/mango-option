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

> Superseded by §Rework (D12–D16); kept for the review history.

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
["//visibility:public"]` like its siblings; every translation unit that
calls the helpers includes `mango/option/table/covering_grid.hpp`
**directly** (no reliance on the `bspline_builder.hpp` re-include, which
is kept only so the public header's surface is unchanged), and its target
lists `//src/option/table:covering_grid` as a direct dep:
`//src/option/table/bspline:bspline_builder`,
`//src/option/table/bspline:bspline_adaptive`,
`//src/option/table/chebyshev:chebyshev_adaptive`,
`//src/option/table/chebyshev:chebyshev_table_builder`,
`//src/option/table/dimensionless:dimensionless_builder`,
`//src/option/table/dimensionless:dimensionless_adaptive`, and
`//tests:covering_grid_test`. `//tests:dimensionless_adaptive_test` gains
`//src/option:american_option` and
`//src/option/table/dimensionless:dimensionless_european` for T6's oracle.

Contract widening (D3): the moneyness-axis argument is renamed
`log_moneyness_nodes` and its reach is computed as
`max(|min|, |max|)` over the span with `std::minmax_element`, dropping
#479's "sorted ascending" precondition. `chebyshev_nodes` and
`cc_level_nodes` do promise ascending output (`chebyshev_nodes.hpp:10`,
`:59`), so this is a deliberate generalization of the helper's contract —
five new call sites hand in node arrays, and the coverage guarantee should
not rest on a documentation promise of an unrelated generator. Empty
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
  double ln_kappa_0, double K_ref, OptionType option_type)`, declared in a
  new header `src/option/table/dimensionless/dimensionless_adaptive_detail.hpp`
  that is owned by the `dimensionless_adaptive` target (added to its
  `hdrs`), and defined in `dimensionless_adaptive.cpp` where it lives
  today; the adaptive loop calls it unchanged. The declaration must not
  go into `dimensionless_builder.hpp` (review round 2): that header is
  also exported by the `dimensionless_builder` target, which does not
  define the symbol, so a consumer of that target alone could compile a
  call it cannot link. Semantics are untouched: normalized EEP
  `max(American − European, 0)`, 0.0 on any solver failure.

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

(Note: when the `max_spatial_points` clamp binds, `estimate_pde_grid` still
adds one point to keep the count odd, so a clamped grid has 5,001 points at
Ultra — the same pre-existing "+1 odd-point adjustment, so the cap is not a
strict ceiling" the #437 spec recorded; this change does not alter it.)

For a dividend-free normalized param, `estimate_pde_grid` gives width
`2·n_sigma·σ√T` and `dx_target = σ·√tol` (`grid_spec_types.cpp:45-50`), so
`Nx = 2·n_sigma·√T/√tol` — **σ cancels**. Every param in an S1/S3/S4
batch therefore estimates the same `Nx`, and the batch union takes that
value: at Ultra and the 0.5y-floored τ axis (T ≈ 0.694) it is ≈ 3,725
points today. Under D11's clearance rule, `max(1.1·reach, reach +
3·max σ√T)`, the T2 chain below (reach ≈ 1.09, σ_hi ≈ 0.225) resolves to a
half-width ≈ 1.65, so `Nx ≈ 6,550` — the Ultra `max_spatial_points = 5,000`
clamp **binds** for this chain (measured in Task 3b: errors did not
degrade against the covering oracle). The trade is therefore coarser
interior `dx` in exchange for coverage, not the ~28 % point-count growth
originally estimated; the spec accepts that trade. The time-step count
stays approximately constant regardless: width and `Nx` grow together
until the clamp binds, so with a fixed sinh concentration the smallest
cell — which sets `dt = c_t·dx_min` (`grid_spec_types.cpp:86`) — barely
moves. This is the same resolution-for-coverage trade #437 D4 accepted for
the B-spline path; there is no new policy.

S2 differs only in that the dividend left-extension
`x_min ← ln(e^{x_min} − D/K)` (`grid_spec_types.cpp:65`) is applied per
param after the σ-scaled width; it is monotone, so the highest-σ param
still sets the union's `x_min` and the extension only ever pushes the
left edge further out. That is pre-existing estimator behaviour and the
covering grid only adds the widened `n_sigma` on top.
The `MAX_WIDTH = 5.8` convergence limit remains unenforced against
materialized custom grids on every materializing path (pre-existing gap
noted in #437 D4; out of scope here).

## Regression tests

Every regression must be demonstrated **red on the parent revision** with
the pre-fix error recorded in the test comment, and its tolerance pinned
empirically well below that error (cross-toolchain loosening as in #479 is
allowed and must be justified in the comment). References are direct FDM
solves pinned to an explicit `GridAccuracyParams` profile, never the
solver default: every oracle below that says "American solve" means
`AmericanOptionSolver::create(params, PDEGridSpec{make_grid_accuracy(High)})`
followed by `solve()`, as `fdm_reference_price` in
`adaptive_surface_build_integration_test.cc` already does — not
`solve_american_option`, whose only overload uses the default grid.

- **T1 — helper unit tests** (`tests:covering_grid_test`, small): the four
  #479 tests moved, plus one new case proving the reach is
  order-independent. D11's boundary clearance rule updates three of the
  expected values (`WidensNSigmaWhenAxisUndershoots`,
  `LeavesNSigmaWhenCovered`, and the new order-independence case below):
  each now computes `max(1.1·reach/σ√T, reach/σ√T + 3)` instead of the
  flat `1.1·reach/σ√T`. A merely reversed array would pass against the old
  `front()/back()` code (review round 2), so the case uses a permutation
  whose largest-magnitude node is *interior*, e.g. `{0.0, -0.51, 0.10}`,
  and asserts it widens exactly as its sorted form does.
- **T2 — S1 e2e** (`adaptive_surface_build_integration_test`). The chain
  must be derived against the full transformation chain, because
  `extract_chain_domain` (`adaptive_refinement.cpp:1147-1150`) first
  applies minimum spreads (m ≥ 0.10, τ ≥ 0.5y, σ ≥ 0.10, r ≥ 0.04, each
  centred on the chain's range and then clamped positive) and
  `build_adaptive_chebyshev` (`chebyshev_adaptive.cpp:679-698`) then adds
  CC headroom (`3·width/32` on m, `3·width/8` on τ, `3·width/4` on σ,
  clamped to `σ_lo ≥ 0.01`, `τ_lo ≥ 1e-4`). The τ spread is centred on
  the chain's range and only shifted up when its lower end crosses the
  positive clamp, so for the short-maturity chain proposed here
  (`{0.05, 0.1}`) the sample τ axis becomes ≈ `[1e-6, 0.500001]`, giving
  `τ_hi = 0.5 + 0.1875 = 0.6875`, PDE `T = 1.01·τ_hi`, `√T ≈ 0.833`
  (a `{0.4, 0.5}` chain would instead land at ≈ `[0.2, 0.7]`); a single
  chain vol `v` gives sample σ `[v−0.05, v+0.05]`
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
  **both** σ sample endpoints. Two oracles (review round 2), because the
  table's PDE contract is not the user's: S2 solves with
  `maturity = 1.01·τ_max` and the dividend's calendar time is anchored to
  that padded maturity, so at the τ = 0.25 snapshot the event sits at
  τ = 0.1525 rather than 0.15 — a small, pre-existing timing skew of the
  segmented Chebyshev path that is out of scope here and gets a follow-up
  issue.
  - *Coverage oracle (tight tolerance):* an independent high-accuracy
    solve on the **same padded timeline** — one normalized batch param
    (spot = strike = K_ref, maturity 0.2525, the same dividend schedule),
    snapshot at 0.25, solved on an explicit wide `PDEGridConfig` that
    covers the queried x, and its snapshot spline evaluated at the queried
    x. This isolates spatial coverage from the timing skew.
  - *User-contract oracle (looser tolerance, pinned empirically):* the
    `make_validate_fn`-style direct pinned-profile solve at maturity 0.25 with the same
    `discrete_dividends`, which is what a user compares against; its
    tolerance must sit above the measured timing-skew discrepancy and well
    below the pre-fix extrapolation error.
- **T4 — S4 unit** (`dimensionless_builder_test`): axes with
  `max|x| > 7.1·√τ'_max` (e.g. x ∈ [−0.7, 0.7], τ' ≤ 0.004, old
  half-width ≈ 0.45); compare `values` at the x extremes vs a
  pinned-profile American solve with spot = K·eˣ, strike = K, σ = √2,
  r = κ, q = 0, T = τ', its dollar value divided by K.
- **T5 — S3 regression** (`chebyshev_surface_test`): `build_chebyshev_table`
  with a domain whose m reach exceeds `5·σ_hi·√τ_hi` (e.g. m ∈ [−0.7, 0.7],
  τ ∈ [0.01, 0.1], σ ∈ [0.10, 0.20]: old half-width ≈ 0.32); compare
  `surface.price` at the extreme m vs FDM at min-σ and max-σ.
- **T6 — S5 unit** (`dimensionless_adaptive_test`, the target that owns
  the symbol): `detail::dimensionless_reference_eep` at a
  probe with `|x₀| > 1` (beyond the ≈ ±1.0 reach the 0.02 maturity floor
  gives today, e.g. x₀ = −1.3, τ'₀ = 0.005; the plan moves the probe
  further out if the pre-fix error is not robustly visible) vs the same
  *normalized* EEP computed directly: a pinned-profile American solve at
  spot = K·e^{x₀}, strike = K, σ = √2, r = κ, T = τ'₀, whose dollar
  `value()` is **divided by K** before subtracting the normalized
  `dimensionless_european(x₀, τ'₀, κ, type)`, floored at 0 (review round
  3: `AmericanOptionResult::value()` is a dollar price, the reference is
  V/K). This also exercises the `use_shared_grid=false` custom-grid
  propagation path, which no other test covers.

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
`price_table_data_test`'s Chebyshev round-trips. D11 lives in the shared
`ensure_moneyness_coverage` helper, so it also reaches the B-spline path
through the unchanged #479 call sites: a B-spline build whose reach falls
in `(2·σ_max√T, 4.5·σ_max√T]` previously left `n_sigma` at its default 5
(the flat 10 % margin didn't clear that band) and now widens under the
`reach + 3·σ_max√T` floor. The B-spline bit-identity goldens and the #479
regressions still pass (re-run in Task 3b, Step 4) because their pinned
configurations fall outside that band, not because the B-spline builder
code is unaffected by D11.

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
  `std::minmax_element` instead of `front()/back()`. Rationale (corrected
  in round 2): the node generators do promise ascending output, so this
  is a deliberate generalization, not a bug fix — the coverage guarantee
  at five new call sites should not rest on another component's ordering
  promise; a silent precondition here would be exactly the kind of latent
  trap #441 hunted. Cost is O(n) on a few dozen nodes. Rejected: requiring
  callers to pass a sorted two-element `{lo, hi}` — pushes the
  precondition onto five call sites instead of removing it.
- **D4 — Estimate over the batch actually solved.** S1/S2 pass the
  *missing* batch, not the full node product (#437 D3 carried over).
- **D5 — Keep `set_grid_accuracy`.** It still governs routing eligibility
  and the traced route / rejection reason derived from it
  (`american_option_batch.cpp:166`); removing it would silently change
  which path is taken. It does not feed `materialize_covering_grid`, which
  takes its `accuracy` explicitly. Rejected: dropping it as dead code.
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
- **D9 — `reference_eep` becomes a `detail` function** declared in a new
  `dimensionless_adaptive_detail.hpp` owned by the `dimensionless_adaptive`
  target (revised in round 2: declaring it in the shared
  `dimensionless_builder.hpp` would let `dimensionless_builder`-only
  consumers compile a call that cannot link). Rejected: a
  friend/test-access header — the function has no state to protect and
  the `detail` namespace is already the convention for shared-but-internal
  builder machinery (#437 D1). Rejected: moving the definition into
  `dimensionless_builder.cpp` — it needs `dimensionless_european`, which
  that target does not depend on, and the probe belongs to the adaptive
  loop.
- **D10 — T3 gets two oracles** (round 2): a padded-timeline coverage
  oracle at tight tolerance and a user-contract oracle at a looser,
  empirically pinned tolerance. The segmented path's dividend-timing skew
  from the 1.01 maturity padding is pre-existing and filed as a follow-up,
  not fixed here.
- **D11 — Boundary clearance (execution, Task 3b):** the covering half-width
  is `max(1.1·reach, reach + 3·max σ√T)`, not `1.1·reach`. Found by Task 3's
  node sweep: with the flat 10 % margin the two edge nodes of the segmented
  Chebyshev fit carried boundary contamination growing with σ (0.84 per $100
  at σ ≈ 0.19) while interior nodes matched an independent wide-grid oracle
  to ≤ 1.4e-4. A clearance that is a fraction of the reach is thinner than a
  diffusion length whenever the reach is large relative to σ√T. Applies to
  the #479 B-spline path too (same helper). Rejected: fixing the boundary
  values themselves (a solver change, out of scope) and leaving it as a
  follow-up (the production discrete-dividend path would ship with a
  measured 0.84/$100 defect at its support nodes).

## Rework: composable coverage API (execution round 2)

**Why.** The user's review of PR #484: the existing API is composable — each
layer makes no assumption about anything outside it — and
`detail::materialize_covering_grid` breaks that. It re-derives the solver's
grid formula, knows about the normalized-chain routing, knows about the
point clamp, encodes a boundary-diffusion rule, and its own doc comment says
callers must understand the solver's routing to get a correct answer. Eight
materialization sites plus one `ensure_moneyness_coverage` call, across
the six builder paths named below, each have to remember to call it, and
the "`set_grid_accuracy` is routing-only" comments are a second symptom of
the solver having two sources of truth for its grid. The *approach* (cover
the nodes, clear the boundary by a few diffusion lengths, D11) is right;
only the shape is wrong.

**Principle.** The requirement travels *down* with the request. The table
layer states what it needs — "the solution must resolve this log-moneyness
range" — on the accuracy spec it already hands the solver; the estimator,
which owns `n_sigma`'s semantics, honours it; every routing inherits it
because every routing calls the estimator.

**Scope of the claim (narrowed in review round 1).** This rework removes the
*coverage-specific* solver knowledge from the table layer. The explicit-grid
validation and fallback policy the B-spline builders still duplicate
(`MAX_WIDTH`, `MAX_DX`, minimum width, point clamps, `required_n_sigma`
emulation in `bspline_builder.cpp` and `bspline_adaptive.cpp`) is the same
kind of smell and is out of scope here; it is filed as a follow-up at
finish.

### API

`src/option/grid_spec_types.hpp`:

```cpp
/// Closed interval of log-moneyness x = ln(S/K), relative to the contract's
/// strike (absolute in the solver's x coordinate, NOT an offset from spot).
struct LogMoneynessRange {
    double lo = 0.0;
    double hi = 0.0;
    /// Tight range of a node set given in any order; nullopt for an empty set.
    static std::optional<LogMoneynessRange> of(std::span<const double> nodes);
    /// Largest distance from x0 to either endpoint: the symmetric half-width
    /// about x0 that contains the whole range.
    [[nodiscard]] double reach_from(double x0) const;
};

struct GridAccuracyParams {
    ... existing fields ...
    /// Log-moneyness range the PDE solution must resolve BEYOND the
    /// contract's own spot, because the caller will read the solution there
    /// (price tables evaluate every slice at their moneyness nodes).  The
    /// estimated domain, which is symmetric about the contract's x0 =
    /// ln(spot/strike), is widened until the range sits at least
    /// `coverage_clearance_sigmas` diffusion lengths (sigma*sqrt(T)) inside
    /// the boundary, with a 10 % widening of the reach as a floor for tiny
    /// sigma*sqrt(T).  For contracts sharing one estimated grid the largest
    /// sigma*sqrt(T) among them sets the clearance and the largest required
    /// widening is applied to all, so the shared grid covers the range as a
    /// whole.  nullopt: only the contract's spot matters (the n_sigma domain).
    /// Coverage is disabled (the plain n_sigma domain is used) when either
    /// endpoint or the clearance is non-finite; a negative clearance counts
    /// as zero.  `LogMoneynessRange::of` returns nullopt when any node is
    /// non-finite.  An explicit PDEGridConfig always takes precedence: a
    /// caller supplying a concrete grid owns its domain.
    std::optional<LogMoneynessRange> log_moneyness_coverage;
    double coverage_clearance_sigmas = 3.0;
};

/// The grid estimate_batch_pde_grid would use for `batch`, as a concrete
/// PDEGridConfig for solve_batch's custom_grid (mandatory_times empty: the
/// batch solver rebuilds per-contract dividend times itself).
PDEGridConfig estimate_batch_pde_grid_config(std::span<const PricingParams> batch,
                                             const GridAccuracyParams& accuracy);
```

### Semantics

One private fold in `grid_spec_types.cpp`. For a contract with
`x0 = ln(spot/strike)` and `s = sigma*sqrt(T)`:

```
reach    = max(|lo − x0|, |hi − x0|)                 // symmetric half-width about x0 that contains [lo, hi]
required = max(reach·1.1, reach + clearance·s) / s   // in units of s
```

- `estimate_pde_grid(contract, accuracy)`: `n_sigma = max(n_sigma, required)`
  with the contract's own `s`, then coverage is cleared and the existing
  estimate runs. The domain `[x0 ± n_sigma·s]` contains `[lo − clearance·s,
  hi + clearance·s]` by construction, for any `x0`, inside or outside the
  range. (Review round 1 found the previous draft measured `reach` from 0
  and therefore failed for non-ATM contracts on the public
  `AmericanOptionSolver::create(params, GridAccuracyParams)` path.)
- `estimate_batch_pde_grid(batch, accuracy)`: `s_max = max s_i` over the
  batch; `n_sigma = max(n_sigma, max_i required_i(s_max))`, applied
  uniformly to every contract with coverage cleared, then the existing
  per-contract estimates are unioned. Correctness for an arbitrary batch:
  the contract with `s = s_max` alone has half-width `≥ reach_max +
  clearance·s_max` about its own `x0`, so its domain — and therefore the
  union — contains `[lo − clearance·s_max, hi + clearance·s_max]`. `Nx`
  stays σ-cancelled because one `n_sigma` is applied to all.
- **Numeric identity with D11.** Every table batch is normalized
  (`spot = strike`, so `x0 = 0` for all contracts) and every table site
  passes `LogMoneynessRange::of(nodes)`; then `reach_i = max(|lo|, |hi|)`
  for all `i` and the batch fold is exactly the expression
  `materialize_covering_grid` evaluates today, feeding the same
  `estimate_batch_pde_grid` (same clamp, same odd-point adjustment, same
  time domain). The realized edge is at least `reach + 3·σ_max√T` (the
  maximum of the three terms; `n_sigma`'s own domain can be wider).
- **Every routing of `solve_batch` honours it**, with one solver repair:
  - shared regular path: `estimate_batch_pde_grid(params, grid_accuracy_)`
    over the whole batch (unchanged code);
  - normalized chain: each single-contract group calls
    `estimate_batch_pde_grid({group}, grid_accuracy_)` — a batch of one,
    covering with its own clearance (unchanged code);
  - `use_shared_grid=false`: `estimate_pde_grid(params[i], grid_accuracy_)`
    per contract (unchanged code);
  - `custom_grid = GridAccuracyParams{...}` (**repaired, D15**): today
    `solve_regular_batch` resolves that accuracy against `params[0]` once
    and reuses the result for every contract, on both the shared and the
    per-contract path — a pre-existing violation of the shared-grid
    contract that would also defeat coverage. It now resolves a
    `GridAccuracyParams` custom grid with `estimate_batch_pde_grid(params)`
    on the shared path and `estimate_pde_grid(params[i])` per contract
    otherwise; a `PDEGridConfig` custom grid is still resolved once. No
    in-tree caller passes a `GridAccuracyParams` as `custom_grid`
    (verified by grep), so no existing number moves.
- `is_normalized_eligible` and `trace_ineligibility_reason` judge
  eligibility on a copy of `grid_accuracy_` with coverage cleared (D14):
  routing is decided on the contract's own kink-region grid exactly as
  today. Making eligibility judge the grid actually solved on is #487's
  business, unchanged here.

### Call sites

Every builder sets `accuracy.log_moneyness_coverage =
LogMoneynessRange::of(nodes)` on the accuracy object it already builds. To
keep every number on the branch bit-identical, the sites that today pass a
materialized grid keep passing an explicit shared grid — now obtained from
the public estimator, `estimate_batch_pde_grid_config(batch, accuracy)` —
because a builder that caches per-(σ, r) slices legitimately wants one grid
per cohort. That is a caller's choice expressed through the public API, not
a leaked internal: the same call with no `custom_grid` is correct too, and
solver-level tests pin that on every routing. The dimensionless reference
probe (S5, `use_shared_grid=false`, one contract) goes gridless — it has no
cohort.

**Routing preservation rule (D14, refined in rounds 1–2):** no
`set_grid_accuracy` call is added or removed at any site. S1, S2, S4 and
S5 retain their existing call and pass the same accuracy object with
coverage set, which eligibility clears; S3 and the B-spline sites, which
today leave the solver's accuracy at its default, keep doing so and hand
the coverage-bearing accuracy only to the estimator. Task 9 carries a per-site
before/after table so the reviewer can check this rather than trust it.
The "routing-only" comments at S1–S5 are deleted: the solver's accuracy is
the single source of truth again.

Sites: `chebyshev_adaptive.cpp` S1/S2, `chebyshev_table_builder.cpp` S3,
`dimensionless_builder.cpp` S4, `dimensionless_adaptive.cpp` S5,
`bspline_builder.cpp` (`PriceTableBuilderND::estimate_pde_grid` and the
explicit-grid fallback), `bspline_adaptive.cpp` (both auto-estimation
branches). `covering_grid.{hpp,cpp}`, its target, and `covering_grid_test`
are deleted; `bspline_builder.hpp` drops the re-include.

**Python (D16).** `CONTEXT.md`'s parity rule makes every supported C++
capability reachable from Python, and `GridAccuracyParams` /
`BatchAmericanOptionSolver.set_grid_accuracy_params` are already bound, so
`LogMoneynessRange` (fields `lo`, `hi`) and the two new fields are bound
too, with a round-trip reachability check in `tests/test_bindings.py`.

**ABI / ADR.** `GridAccuracyParams` grows; it is not serialized anywhere
(not part of the Parquet price-table artifacts), so binary consumers must
rebuild but nothing persisted changes. ADR 0001 (Python API parity) is what
requires D16; this change does not alter that decision and needs no new
ADR.

### Tests

- `tests/grid_spec_types_test.cc` (new, small, on `//src/option:grid_spec_types`):
  `LogMoneynessRange::of` order-independence and empty → nullopt;
  `reach_from` for x0 inside and outside the range; single-contract
  coverage at `x0 = 0` and at `x0 ≠ 0` (domain contains `[lo − 3s, hi +
  3s]`); the batch edge for a normalized batch equals the D11 expression;
  a heterogeneous-`x0` batch in which the farthest required reach belongs
  to a contract *other than* the `s_max` contract (the union must contain
  the clearance-expanded absolute range — this is what pins the novel part
  of the fold); a non-default `coverage_clearance_sigmas`; coverage well
  inside the `n_sigma` domain leaves the grid literally unchanged (bounds,
  type, every coordinate, time steps); NaN / ±∞ in an endpoint, in a node
  handed to `of`, or in the clearance → the plain grid (or nullopt); a
  negative clearance counts as zero;
  `estimate_batch_pde_grid_config` carries `n_steps` and an empty
  `mandatory_times`; and **exact grid goldens** (`x_min`, `x_max`,
  `n_points`, `n_time`) for a clamp-binding Ultra batch and a dividend
  batch, recorded from the helper before it is deleted.
- **Exact identity with the helper (Task 8, while it still exists):** for
  the T2-like Ultra chain batch, a High B-spline-like batch, and a dividend
  batch, `estimate_batch_pde_grid_config(batch, accuracy-with-coverage)`
  equals `detail::materialize_covering_grid(accuracy, batch, nodes)` in
  `x_min`, `x_max`, `n_points`, grid type/concentration, every generated
  coordinate (exact equality), and `n_time`. This temporary comparison is
  the proof of migration identity; the retained goldens guard the fold's
  bounds, point count and step count afterwards (the B-spline bit-identity
  goldens keep guarding full coordinates on that path).
- `tests/american_option_batch_test.cc`: (a) normalized chain with
  coverage set and reach 3.0, chosen so the coverage-widened first-contract
  grid is wider than `MAX_WIDTH` while its base grid is eligible — the
  batch must still route normalized (distinct per-group grids, each
  covering with its own clearance), which fails if eligibility is not
  judged with coverage cleared; (b) the shared regular route (two-σ batch
  with a discrete dividend, hence ineligible): all results share one grid
  whose edge follows `σ_max√T`; (c) `use_shared_grid=false` with a probe
  beyond the `n_sigma` domain; (d) `custom_grid = GridAccuracyParams` with
  coverage set, on the shared path (one grid for all, identical endpoints,
  edge from `σ_max`) and on the per-contract path (two different grids,
  each covering the range with its own clearance — pins that coverage is
  not lost while resolving a custom accuracy spec).
- T1's behaviour is covered by the estimator tests; T2–T6 are untouched
  and must pass with their recorded numbers unchanged.

### Decisions

- **D12 — Coverage is a property of the accuracy spec, honoured by the
  estimator, in absolute `ln(S/K)` with the reach measured from each
  contract's own `x0`.** Rejected: an offset-from-spot field (the table
  layer thinks in strike-relative moneyness, and the batch guarantee is
  cleanest as an absolute interval); a new `PDEGridSpec` variant; keeping
  the helper but moving it into the solver. (Round 1: the first draft
  measured reach from 0 and was wrong off the money.)
- **D13 — Explicit shared grid at cohort-caching sites, gridless at S5.**
  Numeric identity with the reviewed branch, and one grid per cache cohort
  is a legitimate caller intent. Rejected: gridless everywhere — on the
  normalized-eligible corner it would switch those batches to per-group
  covering grids (correct, but different numbers and, for low-σ groups,
  the point clamp), a behaviour change this API rework must not carry.
- **D14 — Eligibility judged with coverage cleared, and no
  `set_grid_accuracy` call added or removed.** Preserves today's routing at
  every site; the pre-existing gap that eligibility judges a grid never
  solved on stays with #487.
- **D15 — Repair `custom_grid = GridAccuracyParams` resolution** so the
  shared path estimates over the batch and the per-contract path per
  contract. Rejected: narrowing `custom_grid` to `PDEGridConfig` (a public
  API change needing its own compatibility pass) and documenting the
  first-contract behaviour (a contract violation, not a feature).
- **D16 — Bind the new fields in Python** per the parity rule. Rejected:
  leaving them C++-only (the type is public and user-configurable).
- **Out of scope, noted:** the B-spline explicit-grid policy duplication
  (follow-up); letting `build_adaptive_chebyshev` take a `PDEGridSpec` like
  the B-spline builder (a separate API gap).
