# Round-trip IV-error metric for adaptive validation (#500 remainder)

Issue: https://github.com/accelas/mango-option/issues/500 (the residual after
#504/#505). Research: `docs/research/2026-09-19-iv-inversion-conditioning.md`
(primary-source synthesis) and
`docs/research/2026-09-19-iv-inversion-conditioning-codex-review.md` (Codex
mathematical review of that synthesis; this design adopts its recommendation
in part: operational round trip plus reference brackets; forward-price
validation is recorded, not gated — see §3 and Q9).

Spec revision 5, 2026-09-21.
Rev 4 → 5 (execution, plan Task 10 measurements; user decision Q10):
acceptance scores on the published σ range widened by `target_iv_error` at
each end (the exact product bracket's refusals become the
`edge_band_rescues` diagnostic), and `δ̂` is floored at the oracle's
calibrated uncertainty scale `kReferenceUncertaintyFloor · K`. Reason: the
exact bracket refused the documented Pattern-4 configuration over a 2.7 bps
miss below σ_min, and identical discretizations produced `δ̂ = 0` at
near-intrinsic points, admitting brackets separated by microdollars.
Spec revision 4, 2026-09-19.
Rev 1 → 2 (review round 1): endpoint uncertainty in the admission test, an
explicitly empirical contract, the product's inversion policy, a controlled
grid family, consistent fresh/holdout failure rules.
Rev 2 → 3 (round 2): exact product bracket for acceptance (edge band
diagnostic only); full inversion policy in the shared function; nested grid
family; calibration with order stability and effective sensitivity;
partial-stencil and counter contracts; holdout-only ordering; enumerated
diagnostics and a refusal probe; #500 fixture measured before pinned.
Rev 3 → 4 (round 3, descendant-only — gate 1 passed by convergence): target
validity uses the product's full validation at preparation; segmented
Chebyshev sizing references live on the probe's contract; maturity support
is populated for the segmented Chebyshev loop and final validation; one
fixed grid rounding for production and calibration; classification order in
D8; extraction-boundary details; finite-only aggregation; a second refusal
probe; wording per L5.

## 0. Contract of this metric (read first)

Everything this design reports is an **empirical operational measurement on
the declared sample set**:

- The reference is a numerical oracle (D1). Its uncertainty `δ̂` is a
  grid-convergence *estimate*, never a bound; `p` is a *calibrated constant*.
- "Resolved" (D2) means the oracle's three stencil price intervals
  `[V̂ − δ̂, V̂ + δ̂]` are pairwise separated in the expected order and all
  three targets are valid queries. It is a statement about numbers, made
  before any candidate exists. It is not a monotonicity, uniqueness, or
  continuous-model localisation claim, and it does not bound the oracle's
  actual error.
- The score (D3) is the outcome of running the **shipped** inversion on the
  candidate surface at three target prices. Nothing is claimed about prices
  between them. This three-price test is the final uncertainty contract.
- "Universal" rejection (D4) means "no resolved failure on the declared fresh
  and holdout sets", not a domain-wide maximum guarantee.
- Forward-price accuracy is recorded as a residual and never gated;
  unresolved regions carry no price acceptance requirement and can contain
  finite price errors of any size (Q9: offered and not chosen).
- A "surface inversion failure" is an algorithmic outcome of the shipped
  solver, not proof of mathematical non-existence or non-uniqueness.

## 1. Problem

The adaptive builders (`bspline_adaptive.cpp`, `chebyshev_adaptive.cpp`, both
segmented paths) certify a price surface by scoring Latin-hypercube points
against an FDM reference. `make_iv_score_fn` (`adaptive_metrics.cpp`) turns a
price miss into an "IV error" by dividing by a finite-difference vega, after
two admission filters: skip if `(ref_price - intrinsic)/strike < 1e-4`, skip
if `|vega| < 1e-4`. `kViabilityBound = 0.20` then rejects any candidate whose
holdout max exceeds 2000 bps.

Measured on main (a1f38e77) with the issue's wide-band configuration:

- The build succeeds (788.4 bps max, target 5 bps unmet). The max is one
  deep-ITM put (K=113.72, S=100, τ=0.272, σ=0.14, r=0.055) with a $0.50
  dividend 0.022y ahead: reference 14.1263, surface 14.1090, FD vega 0.2196.
- The numerical reference is flat at 14.1249 for σ ≤ 0.12 at High accuracy
  (the holder waits for the ex-dividend drop, then exercises) and bends up
  after. The linearised number extrapolates 56 bump widths across that
  shoulder and is not an IV error. Whether the continuous model's plateau is
  exactly flat is not established and not needed: the numerical oracle cannot
  distinguish σ there.
- The same σ-shoulder exists everywhere deep ITM. Before the dividend, time
  value is exactly zero and the TV/K filter hides the point; just after, the
  deterministic dividend gain counts as time value and the filter admits it.
- The validation oracle (`make_validate_fn`) calls `solve_american_option`,
  which auto-estimates a grid with default `GridAccuracyParams` (`tol = 1e-2`,
  documented as ~1e-3 price accuracy). Measured against Ultra: 3.3e-3 at an
  ATM 1y put with three dividends, 4.6e-5 at the trigger point. Through a
  0.22 vega, 1e-3 of oracle noise is 45 bps by itself.
- The segmented Chebyshev sizing loop scores a single-`K_ref` leaf evaluated
  at `ln(S/K)` and scaled by `a = K/K_ref` (`chebyshev_adaptive.cpp:618`),
  i.e. `a·V(S/a, K_ref; D) = V(S, K; a·D)`, against a reference solved at
  `V(S, K; D)` (`:1043`). The B-spline probe already compensates for this
  (`bspline_adaptive.cpp:771`); Chebyshev does not, so its sizing errors
  include a dividend-scaling residual no refinement can remove.

Both admission constants, the vega floor, and the 0.20 bound are unanchored.
The user's requirement: the replacement must rest on sound mathematics.

What the mathematics says (the research note and three review rounds agree):

1. `|ΔV| / vega` is a first-order inverse image of a price discrepancy,
   meaningful only where the price is differentiable in σ with positive slope
   and the displacement stays where that slope is representative.
2. Conditioning is a first-order diagnostic: a small forward tolerance on
   `σ = f(V)` is attainable only where the input error times `1/vega` is
   small (Higham §1.6). It does not by itself give a finite-error certificate,
   and with a numerical oracle the input error must be estimated.
3. Local vega, bump-pair derivatives, and a two-grid Richardson number are
   diagnostics, not bounds. A finite bracket in σ with uncertainty estimates
   at every bracket point is the derivative-free way to state "the oracle
   separates σ0 from σ0 ± τ".
4. The quantity the product delivers is the surface's own inverse of a price
   under the product's validation, bracket and pre-check policy.
5. Admission must not depend on the candidate surface.
6. A scalar-or-nothing score cannot encode "error", "no inverse", "ambiguous
   inverse", "did not converge", and "reference cannot resolve".

## 2. Goals

- Replace the IV-error metric with one whose constants are the user's
  tolerance, measured or calibrated quantities, or explicitly listed
  operational policies (§4.11). No time-value threshold, no vega floor, no
  0.20 bound.
- Score what the product does, with the product's code and policy, on
  references that live on the contract the candidate prices.
- Make the reference's accuracy explicit and admission surface-independent.
- Report statuses and a price residual.
- Keep `run_refinement`'s role, candidate retention, the final validation
  step, the public builder APIs, and the C ABI layout.

## 3. Non-goals

- The σ-axis leaf resolution of the segmented Chebyshev builder. Follow-up
  issue, filed at PR time with this metric's measured numbers.
- Changing the product's query-time policy (`vega_threshold`,
  `adaptive_bounds`, edge behaviour). The edge-band diagnostic (D3) informs
  that follow-up; it decides nothing.
- Removing `AdaptiveGridParams::vega_floor` from the C ABI, Rust, Python (#463).
- A price-space acceptance gate (Q9).
- Statistical coverage guarantees or deterministic enclosures. See §0.
- Sampling, axis selection, headroom rules, and the reference loop's serial
  execution.

## 4. Design

Notation per validation point `p = (S, K, τ, σ0, r)` with the builder's
dividend schedule and reference maturity: `V̂(σ)` is the reference price on
the fine grid, `V̂½(σ)` on the coarse grid, `S(σ) = handle.price(S, K, τ, σ, r)`
the candidate surface, `τ_iv = params.target_iv_error`.

### D1. Reference oracle: declared profile, one controlled stencil

**Profile.** `kReferenceAccuracy = GridAccuracyProfile::High` (constant in
`adaptive_metrics.cpp`, with the measurement that chose it). Agreement with
Ultra is evidence of adequacy, not reference truth; D8 re-measures it.

**Nested grid family (one constructor, one rounding, for production and
calibration).** `make_reference_grid_family(params, accuracy, levels)`:

1. `estimate_pde_grid(params, accuracy)` gives `(GridSpec G0, TimeDomain T0)`
   with point count `n0`.
2. Spatial: `n` is the **largest** count `≡ 1 (mod 16)` with
   `n ≥ n0` and `n ≤ accuracy.max_spatial_points` (the cap is strict per the
   grid API); if none exists above `n0`, the largest such count in
   `[min_spatial_points, max_spatial_points]` is used and the family records
   `rounded_down = true` (never triggered by the shipped profiles, whose
   windows are ≥ 1000 wide; asserted in tests). The multi-sinh spec is
   re-sampled at `n` with the same domain, centres and weights. Level `k` is
   the strict subsequence of every `2^k`-th node of level 0; for `k ≤ 3`
   every level has an odd count and shares level 0's middle-index node
   (whatever physical point it is); the refinement ratio is exactly 2 per
   level. The same `n` is used whether one or three coarse levels are
   requested, so production's fine grid **is** calibration's finest grid.
3. Temporal: level 0 requests `n_time = T0.n_steps()`; level `k` requests
   `⌈n_time / 2^k⌉`. `mandatory_times` holds the **event times only**
   (`resolve_grid` merges dividend taus into any explicit config, verified at
   `american_option.cpp:68`); no fine time nodes are copied. Because
   `TimeDomain::with_mandatory_points` rounds per event segment, the achieved
   temporal ratio is 2 up to per-segment rounding and can be 1:1 inside a
   very short event interval; the calibrated `p` is therefore an *effective*
   order for this family under equal-coordinate refinement, not an exact
   joint Richardson exponent. Achieved step counts are recorded.

**Stencil.** Six solves per point, `G = level 0`, `G½ = level 1`, chosen once
per preparation for the contract at σ0 + τ_iv (the widest x-domain of the
three); dividends are rolled once per preparation:

| Solve | σ | Grid |
|---|---|---|
| `y`, `y½` | σ0 | G, G½ |
| `lo`, `lo½` | σ0 − τ_iv | G, G½ |
| `hi`, `hi½` | σ0 + τ_iv | G, G½ |

**Uncertainty estimates.** For each σ: `δ̂_k = max(F_s · |V̂_k − V̂½_k| /
(2^p − 1), kReferenceUncertaintyFloor · K)`, `F_s = 3` (Roache's two-grid
safety factor), `p = kReferenceConvergenceOrder` calibrated by D8, `K` the
strike of the contract actually solved (the probe strike on a probe
contract; the adapter's scaling then carries the floor to the user strike).
The floor is the oracle's calibrated accuracy scale: D8 measures
`max |V_High − V_Ultra| / K` over the calibration set and asserts the
constant is at least that (provisional value 1e-7 from the 2026-09-19
measurements: 4e-6 and 7e-6 on K = 100). Without it, two discretizations
that agree exactly (both on the obstacle at a near-intrinsic point) yield
`δ̂ = 0` and the stencil "resolves" a bracket separated by microdollars,
which the shipped inversion then cannot invert (rev 5, measured). `δ̂` is an
*estimate*: a two-grid difference cannot see bias shared by both grids, and a
profile-level `p` does not establish pointwise order.

**Types and partial stencils.**

```cpp
struct ErrorRefs {
    double ref_price;                            // y (always present)
    double bracket_lo_price, bracket_hi_price;   // NaN when unavailable
    double sigma_lo, sigma_hi;                   // σ0 ∓ τ_iv as solved
    double delta, delta_lo, delta_hi;            // δ̂ estimates; NaN when unavailable
    bool resolved;                               // D2
    uint32_t fine_steps, coarse_steps;           // achieved time steps (record only)
};
```

`PrepareRefsFn` keeps its outward signature
`expected<ErrorRefs, SolverError>(spot, strike, tau, sigma, rate)`. Contract:

- The base fine solve `y` failing → `unexpected` (the point is *invalid*,
  counted as today).
- `σ0 − τ_iv ≤ 0`, a non-finite stencil coordinate, any bracket/coarse solve
  failing, or any target failing validation (D2) → success with `resolved =
  false`, base price present, unavailable fields NaN. Such a point is
  `ReferenceUnresolved`; its residual (D3) is still computed.
- Preparation validity ("base price exists"), reference resolution (D2), and
  candidate non-finiteness (D4) are three separate facts and are never
  conflated; the holdout and final preparation paths accept a finite base
  price with an incomplete stencil (today they reject on non-finite `vega`;
  that check goes with the field).

`make_fd_vega_refs_fn` is replaced by `make_stencil_refs_fn(params, oracle,
counter)`: `oracle` is a value type owning the dividend schedule, reference
maturity, option type and yield and exposing `solve(spot, strike, tau, sigma,
rate, const PDEGridConfig&)`; `counter` is a `std::shared_ptr<ReferenceSolve
Counter>` (`fine_attempts`, `coarse_attempts`, `fine_failures`,
`coarse_failures`, atomics) that survives failed preparations. No mutable
grid state (L2). `make_validate_fn` remains for single-price callers, on the
same oracle at High.

**Probe adapters (both segmented sizing loops).** A sizing handle that
prices a single-`K_ref` probe at `(S/a, K_ref)` and scales by `a = K/K_ref`
approximates `V(S, K; a·D)`, not `V(S, K; D)`. The reference for such a
handle is therefore prepared on the **probe's own contract**: solve the whole
stencil at `(S/a, K_ref)` with the builder's schedule and scale `ref_price`,
both bracket prices and all three `δ̂` by `a`; σ coordinates and `resolved`
are unchanged (every term of D2 scales alike). The existing B-spline adapter
(`bspline_adaptive.cpp:771`) is kept and extended to the new fields; **the
segmented Chebyshev sizing loop gets the same adapter** (new; it has none
today). Final assembled-surface validation on both backends stays on the
user's contract `V(S, K; D)`. Pinned by non-ATM-strike dividend fixtures on
both backends (L6).

### D2. Resolution: surface-independent admission

A point is **resolved at τ_iv** iff all six stencil prices are finite, the
three estimated price intervals are separated in the expected order,

```
y − δ̂  >  lo + δ̂_lo        and        hi − δ̂_hi  >  y + δ̂ ,
```

and **each of the three targets `y − δ̂, y, y + δ̂` passes the product's
query validation** (`validate_iv_query`: finite, positive, `≥ intrinsic`,
`≤` the upper no-arbitrage bound — spot for calls, strike times the maximum
discount factor for puts, which exceeds 1 under negative rates). The
validation is run at preparation on the probe/user contract the reference
describes, with the point's rate; a rejected target makes the point
`ReferenceUnresolved` (a reference limitation that no candidate can repair).

This is the finite-bracket condition of the prior review (§3.B) applied to
the reference's *estimated* intervals. It compares numbers and claims
nothing else: not actual oracle error bounds, not monotonicity between the
evaluations, not uniqueness, not a continuous-model volatility enclosure
(Ekström's monotonicity is for a diffusion model and is not extended here to
the implemented cash-dividend jumps). Ordering violations are detected only
where they show at the three stencil points.

It **replaces** both old filters. In the exercise region `lo = y = hi` and
the point is unresolved; where vega is small the separation falls inside the
estimates; with `δ̂ = 0` an arbitrarily small positive separation passes,
which is the honest answer when two resolutions agree exactly.

Unresolved points: no IV statistic, no refinement bin, counted (D7), residual
recorded (D3). Resolution is decided at preparation, before any candidate
exists (L1).

**Coverage policy (chosen, not derived).** The loop keeps its rule "refuse
(`ValidationFailed`) when fewer than `max(4, validation_samples/4)` holdout
points have prepared references" (`validation_samples` = requested count;
unsupported-maturity samples are excluded **before** preparation, D4) and
applies the same threshold to **resolved** references, for the holdout and
for the final validation set. It is an operational coverage policy with no
spatial or statistical guarantee (§4.11). On refusal the `PriceTableError`
is unchanged (no ABI change) and the USDT probe
`mango:adaptive_validation_refused(set, requested, prepared, resolved,
unsupported)` fires (`set` ∈ {holdout, final}; `ivcalc_trace.h`).

### D3. Score: the shipped inversion, three targets, one diagnostic band

**Seam.**

```cpp
enum class PointStatus {
    Measured,              // inversion succeeded at all three targets
    ReferenceUnresolved,   // D2: not evidence
    SurfaceVegaTooSmall,   // product pre-check refused
    SurfaceNoRoot,         // BracketingFailed on the product bracket
    SurfaceAmbiguous,      // MultipleRoots (screen or post-Brent slope)
    SurfaceNonConvergent,  // MaxIterationsExceeded
    SurfaceNonFinite,      // NumericalInstability / NaN price or vega
};
struct PointScore {
    PointStatus status;
    double iv_error;          // max_k |σ̂_k − σ0| iff Measured
    double price_residual;    // |S(σ0) − y| / K when S(σ0) finite, else NaN
    bool   edge_band_rescue;  // diagnostic only
};
using ScoreErrorFn = std::function<PointScore(
    const SurfaceHandle& handle, const ErrorRefs& refs,
    double spot, double strike, double tau, double sigma, double rate)>;
```

Target validity is established at preparation (D2), so the scorer never
sees an invalid target; a validation rejection inside the shared function
on this path is an invariant violation and is treated like a non-finite
evaluation (clears `all_finite`, disqualifies the candidate) so it can never
pass silently. Tests assert it does not occur on valid fixtures.

`SurfaceHandle` gains `vega` (same signature as `price`). Every handle fills
it: the four named surface wrappers from their `vega()`; the segmented
Chebyshev sizing handle (a leaf-routing lambda, `chebyshev_adaptive.cpp:618`)
from `ChebyshevSegmentedLeaf::vega` (a `TransformLeaf`, analytic partial)
through **the same** segment routing, local-time origin, gap handling and
`a` scaling as its price lambda; the B-spline probe handle likewise. Both
adapters get price/vega tests (L4). Domain data and policy are captured in
the scoring factory `make_round_trip_score_fn(params, ctx, option_type,
dividend info)`; the scorer performs no PDE solves.

**One inversion component, no cycle.** New low-level target
`//src/option:surface_inversion` (`surface_inversion.{hpp,cpp}`; deps:
`root_finding`, `option_spec`, `error_types`, tracing header; no table or
builder deps) receives from `interpolated_iv_solver.{hpp,cpp}`:
`BracketScreen`, `screen_bracket` (keeping `ObjectiveRef`, no allocation on
the `noexcept` path), and:

```cpp
struct SurfaceInversionPolicy {
    double config_sigma_min, config_sigma_max;   // InterpolatedIVSolverConfig (0.01, 3.0)
    double published_sigma_min, published_sigma_max;
    double vega_threshold;                       // 1e-4
    bool   detect_multiple_roots;                // true
    double tolerance;                            // 1e-6 (Brent price residual and σ width)
    size_t max_iter;                             // 50
};
using PriceFn = function_ref<double(double sigma)>;   // non-owning views
using VegaFn  = function_ref<double(double sigma)>;

/// The effective σ bracket the product will search (adaptive cap ∩ config ∩
/// published, with the fallback to the published range when empty).
std::pair<double, double> effective_sigma_bracket(
    double spot, double strike, OptionType type, double target_price,
    const SurfaceInversionPolicy& policy) noexcept;

/// Pre-check, screen, Brent, post-check — current order, current codes.
std::expected<IVSuccess, IVError> invert_price_on_surface(
    PriceFn price, VegaFn vega, double target_price,
    std::pair<double, double> bracket,
    double spot, const SurfaceInversionPolicy& policy) noexcept;
```

`InterpolatedIVSolver<Surface>::solve` becomes: `validate_query` →
`effective_sigma_bracket` → `is_in_bounds` at both bracket ends →
`invert_price_on_surface` with `price = σ ↦ eval_price(moneyness, τ, σ, r,
K)` (which reconstructs spot as `(spot/strike)·strike` today — preserved
verbatim) and `vega = σ ↦ surface_.vega(spot, K, τ, σ, r)` (original spot,
preserved) → set `used_rate_approximation` for curve rates. Its behaviour is
unchanged (L4), verified bit-for-bit on the existing solver fixtures across
all instantiated surface families, plus custom `sigma_min/max`, disjoint
limits (fallback), a cap transition, and a yield-curve rate.

**Boundary for the loop.** Loop points lie inside the published domain by
construction (drawn from `ctx.sample_bounds`; τ passes
`maturity_is_supported`, now populated for every segmented path, D4), so
`is_in_bounds` is not re-run; `InvalidGridConfig` cannot arise. The loop
calls `effective_sigma_bracket` with `published = ctx.sample_bounds` σ range
and the default policy, then `invert_price_on_surface` — identical code,
identical thresholds, no policy duplication.

**Acceptance uses the product policy on the tolerance band (rev 5, user
decision Q10).** The scorer runs the shipped inversion (same pre-check,
screen, Brent, post-check, cap and config limits) with the published σ
limits widened by τ_iv on each side, clipped to the fit domain `ctx.bounds`.
A root within the user's own tolerance beyond a published edge is a
measurement, not a refusal. The metric is therefore named
**operational round trip (edge band τ)** in code, diagnostics and docs, and
the docs state that it is stronger than the shipped solver at the edges by
exactly τ_iv. The exact-bracket version was tried first (revs 3–4): it
refused the documented Pattern-4 configuration over a 2.7 bps miss below
σ_min, and any surface with a few bps of error near an edge has a coin-flip
chance per edge-adjacent sample of refusing the whole build. The query-time
follow-up gives the product the same band so the two coincide again.

**Exact-bracket diagnostic (never used for acceptance).** For a `Measured`
point, if any of the three recovered σ̂_k lies outside the exact product
bracket (published limits, no widening, after cap and config limits), the
point sets `edge_band_rescue = true`: the shipped solver would have refused
that query today. Counts are reported (D7) as evidence for the follow-up
(L3); they influence neither ranking nor viability.

**Three targets.** For a resolved point the inversion runs for
`y − δ̂, y, y + δ̂`. `Measured` iff all three succeed;
`iv_error = max_k |σ̂_k − σ0|`. Otherwise the status is the most severe
failure among the three (NonFinite > NonConvergent > Ambiguous > NoRoot >
VegaTooSmall). What this measures: the shipped solver's answer at three
prices the oracle could have meant, given its estimated uncertainty. What it
does not claim: anything about prices between them (the 17-point screen
documents folds it cannot detect; Brent stops on a residual/width condition,
not an exact inverse; endpoint extrema would need monotone inverse behaviour
the screen does not establish). When `τ_iv` is below Brent's stopping
tolerance mapped through the surface's slope, the reported error carries
that resolution floor; the docs say so.

**Price residual.** `|S(σ0) − y| / K` for every point with a prepared
reference whose surface price is finite, resolved or not.

### D4. Loop consumption: support, fresh and holdout, ranking, viability, refinement

**Maturity support.** `RefinementContext::maturity_is_supported` is
populated for **both** segmented paths (no builder sets it today; only tests
do): the Chebyshev loop and final validation from `seg_bounds_`/`seg_is_gap_`,
the B-spline loop and final validation from `compute_segment_boundaries` on
the same schedule, maturity and τ domain, so the predicate matches the
`TauSegmentSplit::contains_maturity` the product enforces (unsupported
inside a ±5e-4 gap). Unsupported samples are excluded before
preparation and counted (`unsupported`); they are not references, not
unresolved points, and not candidate defects. At supported maturities the
non-finite veto stands. (Moved from §3 non-goals: leaving it unset would make
`is_in_bounds` reject queries the loop had scored, contradicting L4.)

`SampleEval` and `FinalScore` gain `unresolved`, `unsupported`,
`surface_failures`, `max_price_residual`, `edge_band_rescues`; `measured`
counts `Measured` only; `filtered` is replaced by `unresolved`. Both the
fresh pass and the holdout pass produce these.

**Fresh-path order.** Support check → candidate surface price at the sample
(non-finite → veto, regardless of what follows) → reference preparation →
score. The current `continue` before the veto (`adaptive_refinement.cpp:384`)
is reordered.

**Viability of a candidate** (replaces the `kViabilityBound` clause):

```
all fresh and holdout surface prices finite  ∧  holdout.measured > 0  ∧
fresh.surface_failures == 0  ∧  holdout.surface_failures == 0
```

`fresh_converged = fresh.measured > 0 ∧ fresh.max ≤ τ_iv ∧
fresh.surface_failures == 0`. `target_met = picked.viable ∧
picked.holdout_max ≤ τ_iv ∧ picked.fresh_converged`. `FinalScore::viable() =
all_finite ∧ measured > 0 ∧ surface_failures == 0`; `needs_final_retry` and
`select_final_surface` keep their logic on it ("both failing" → `None`;
finite errors above the old 0.20 return as best effort with
`target_met = false`, covered by a test).

**Ordering** (exploration-base advance, retention pick, final pick), applied
after the viability filter where one exists: fewer **holdout**
`surface_failures` → lower holdout max → lower holdout avg → earlier
iteration. Fresh failures veto and attribute; they do not rank. A candidate
with a non-finite holdout statistic is not an exploration base. A candidate
with `measured == 0` and failures may be a base if no better exists, never a
returned candidate.

**Walk restart.** Fewer holdout failures than the base, or equal failures
with the existing 2 % relative improvement of the holdout max.

**Refinement bins.** `ErrorBins` gains `failure_counts[dim][bin]`, recorded
unconditionally for every surface failure (fresh and holdout);
`pick_refinement_axis` and `problematic_bins` use `bin_counts +
failure_counts`; `evaluate_holdout` returns bins for its failures;
`Candidate::bins` merges both. Measured errors above τ_iv are recorded as
today.

### D5. Monotonicity scan

Diagnostic only. Noise floor per point: the maximum over the **finite**
values among `delta, delta_lo, delta_hi`, floored at `1e-8 · spot`; if none
is finite the point is skipped by the scan. `vega_floor` parameter removed;
the doc comment names it a reporting threshold.

### D6. `vega_floor` deprecation

Stays in the struct, the C ABI (offset asserts untouched), Rust, Python. C
header comment, Rust docs (both layers), Python docstring and C++ comment:
deprecated, ignored since this change, removed in #463. No validation.
Benchmark mirrors `kVegaFloor` and `kTVKThreshold` in
`benchmarks/interp_iv_safety.cc` and their explanatory text go.

### D7. Diagnostics (enumerated)

`BuildDiagnostics` gains exactly:

```cpp
size_t holdout_points_unresolved = 0;   // oracle could not resolve target (D2)
size_t holdout_points_unsupported = 0;  // excluded by maturity support (D4)
size_t surface_failures = 0;            // returned surface on the holdout (0 by D4)
size_t edge_band_rescues = 0;           // D3 diagnostic
double max_price_residual = 0.0;        // |S − V̂|/K over prepared holdout points
double reference_uncertainty_max = 0.0; // max over finite δ̂ on the holdout (estimate); 0 if none
size_t reference_solves_fine = 0;       // ReferenceSolveCounter totals
size_t reference_solves_coarse = 0;
```

`IterationStats` gains `unresolved`, `surface_failures`, `edge_band_rescues`
(C++ only). The Python `build_diagnostics` property dict gains the eight keys.
Rust exposes no diagnostics; unchanged.

**Refusal probes.** Besides `adaptive_validation_refused` (D2), a second
probe `mango:adaptive_no_viable_surface(stage, candidates, failures_no_root,
failures_ambiguous, failures_nonconvergent, failures_nonfinite,
failures_vega, edge_band_rescues)` fires when retention or final selection
finds no viable candidate (`stage` ∈ {loop, final, retry}), so a universal
refusal keeps its status evidence without a `BuildDiagnostics` return or an
ABI change.

**Solve accounting.** All scattered `× 3` multipliers go; `total_pde_solves`
adds `counter.fine_attempts + counter.coarse_attempts` from the one counter
each path owns (holdout, fresh, final, retry reuse included).

### D8. Calibration of `p` and the profile

Nightly `slow` test `reference_oracle_calibration_test`, using
`make_reference_grid_family(levels = 3)` (levels 0..3), on six points at
High: ATM 1y put with three $0.50 dividends; the #500 trigger; OTM 30-day
put; deep-OTM 7-day put; ITM 2y put; ATM 6-month call. It calibrates
**complete production stencils**: for each point and each `τ_iv ∈ {5e-4,
1e-3}` all three stencil σ on the grid selected at σ0 + τ_iv, so the
endpoint uncertainties that control admission are the ones calibrated.

For each price series and each consecutive triple `(k, k+1, k+2)`:
`d_a = V_{k+2} − V_{k+1}`, `d_b = V_{k+1} − V_k`, threshold `θ = 2^-40 · K`
(a chosen classification threshold, §4.11):

1. **insufficient signal** if `|d_a| ≤ θ` or `|d_b| ≤ θ` (no order
   available; legitimate at exercise or deep-OTM points);
2. else **oscillatory** if `d_a · d_b < 0`;
3. else **usable**, `p_obs = ln(d_a/d_b)/ln 2`.

Assertions: no usable-or-oscillatory triple is oscillatory; every usable
`p_obs` is finite and strictly positive; for series with two usable triples
the two agree within 0.5 (chosen allowance); at least four of the six
points have two usable triples at the base σ (coverage rule);
`kReferenceConvergenceOrder ≤ min usable p_obs`; `|V_High − V_Ultra| ≤
δ̂_High` at every point.

**Effective sensitivity experiments** (the American path is a single-pass
projected Thomas solve, `pde_solver.hpp:568/885`; `TRBDF2Config::tolerance`
does not affect it and is not varied): (a) x-domain widened by one σ√T —
**assertion**: shift ≤ `δ̂`; (b) spatial-only and temporal-only refinement
to attribute error between axes — reported; (c) the solver's `LcpKktReport`
recorded per point — reported.

**If the calibration fails** (any assertion), the constant is **not** tuned
to pass: the next step is further controlled refinement (levels 4–5) and, if
the order remains unusable or the domain assertion fails, a revised oracle
family (profile or domain rule) before `p` is chosen. Numbers,
classifications and shifts are recorded in `docs/MATHEMATICAL_FOUNDATIONS.md`.
Per-point `p` is not measured at build time.

### D9. Regression coverage

- `tests/adaptive_refinement_unit_test.cc` (synthetic references and
  surfaces, no PDE): D2 with overlapping endpoint intervals (`y=10, lo=9.85,
  hi=10.15, δ=0.10` is unresolved), reversed ordering, `σ0 − τ_iv ≤ 0`,
  partial stencils through fresh, cached holdout, final scoring and the
  monotonicity scan, targets failing the intrinsic bound and the **upper
  bound** (call above spot; put above strike, and above `K·e^{−rT}` with a
  negative rate) → unresolved, scale invariance under the probe rescaling;
  every `PointStatus` from a purpose-built surface (exact, biased,
  edge-shifted → `NoRoot` with and without `edge_band_rescue`, `y ± δ̂`
  straddling the surface's attainable range, decreasing crossing, fold
  between two targets caught only if it falls on a screen point — documented
  limit, NaN interior, low surface vega, forced `max_iter`); three-target
  aggregation and severity order; the invariant-violation veto; fresh-path
  veto with failed preparation and NaN price; unsupported samples excluded
  before preparation; viability with fresh-only and holdout-only failures;
  zero-error successes mixed with failures; ordering with NaN statistics;
  `measured == 0` base eligibility; failure bins when τ_iv exceeds the
  bracket; restart on fewer failures; `select_final_surface` both failing /
  ties / failures-vs-max / finite error above 0.20 as best effort; exact
  attempt accounting across probes, failed preparation, final and retry;
  all-unresolved holdout refusal; τ_iv larger than the σ domain.
- Solver tests: `effective_sigma_bracket` + `invert_price_on_surface`
  reproduce `solve()` bit-for-bit across all instantiated surface families,
  plus custom limits, disjoint limits (fallback), a cap transition, a
  yield-curve rate (`used_rate_approximation` preserved).
- Adapters: price/vega tests for the segmented Chebyshev sizing handle and
  the B-spline probe handle (routing, local time, gaps, scaling); non-ATM
  dividend fixtures for both probe reference adapters (L6).
- Grid family: nodes nested, odd at every level ≤ 3, shared middle index;
  identical fine grid for `levels = 1` and `levels = 3`; event-only mandatory
  times; achieved step counts; cap interplay; a fake oracle asserts identical
  fine configs across the three fine solves, identical coarse configs across
  the three coarse solves, and exact 2:1 nesting (criterion 2).
- `tests/adaptive_grid_types_test.cc`: `vega_floor` accepted at any value.
  Python: the eight new keys present; `vega_floor` ignored.
- The existing fixed-expiry oracle test (`adaptive_refinement_unit_test.cc:
  1298`) compares against a solver built at the same High profile and keeps
  its dividend-rolling assertion.
- `tests/adaptive_surface_build_slow_test.cc`: `WideBandDividendBracket
  RemainsViable` becomes `WideBandDividendBracketRoundTrip` on the unchanged
  manual two-K_ref fixture. **The outcome is measured first** (resolution
  flag, status under the exact product policy, error if measured,
  `edge_band_rescue`), then pinned with provenance. A refusal, if that is
  what the shipped solver produces, is pinned as a refusal and becomes
  evidence for the leaf follow-up; no ceiling is adjusted. `// Bug:` line
  states the 788 bps artifact.
- `tests/adaptive_surface_build_integration_test.cc`,
  `tests/adaptive_grid_builder_test.cc`, `tests/iv_solver_factory_slow_test.cc`:
  direct users of the removed helpers/bound and retry-path tests updated;
  pins produced by the old metric's amplification re-measured and
  regenerated with the reason in the commit.
- A segmented Chebyshev case with an actual event gap confirms unsupported
  exclusion (not failure) and the non-finite veto at supported maturities.

### D10. Documentation

`docs/MATHEMATICAL_FOUNDATIONS.md` "Adaptive validation metric": stencil,
grid family, admission inequalities, the round trip and its three targets,
the edge-band diagnostic, the calibration record, §0's contract, §4.11's
inventory, citations. `docs/API_GUIDE.md`: statuses, diagnostics,
deprecation. `docs/ARCHITECTURE.md`: the shared inversion component.
`CONTEXT.md` gains *reference-resolved point*, *operational round trip*,
*surface inversion failure* (defined as an algorithmic outcome). ADR 0001's
split holds: C++ tests own numerical correctness, Python tests own
reachability.

### 4.11 Constant inventory (honest list)

| Constant | Kind | Where |
|---|---|---|
| `target_iv_error` | user | `AdaptiveGridParams` |
| `δ̂` | estimate (two-grid GCI) | D1 |
| `p` | calibrated constant (D8) | D1 |
| `F_s = 3` | literature convention (Roache, two-grid) | D1 |
| `kReferenceAccuracy = High` | chosen from measurement | D1 |
| grid rounding `n ≡ 1 (mod 16)` | construction rule (nesting to 3 levels) | D1 |
| `θ = 2^-40 · K`, order-stability allowance 0.5, four-of-six coverage | calibration classification/acceptance policy | D8 |
| coverage `max(4, N/4)` on prepared and on resolved | operational policy (pre-existing rule, applied twice) | D2 |
| inversion policy: config σ 0.01/3.0, `vega_threshold 1e-4`, 17 screen points, zero-tol `1e-9·spot`, Brent `1e-6` (residual and width) / 50, cap 1.5/2/3 | product policy, reused unchanged | D3 |
| edge band `τ_iv` | the user's tolerance, reused as the acceptance band (rev 5) | D3 |
| `kReferenceUncertaintyFloor` | calibrated constant (D8: ≥ max \|V_High − V_Ultra\| / K) | D1 |
| monotonicity-scan floor `1e-8·spot` | diagnostic floor, pre-existing | D5 |
| walk restart 2 % | pre-existing loop policy | D4 |

Nothing else numeric appears in the metric.

### Binding laws (govern every not-yet-enumerated instance)

- L1. Admission never reads the candidate surface; it validates every target
  the scorer will use.
- L2. Every stencil shares one fine grid from the family constructor; the
  coarse grid is its strict subsequence; nothing is re-estimated per σ.
- L3. A status is never converted into a number for acceptance or
  attribution; only `Measured` errors enter `max`/`avg`/thresholded bins;
  failures have their own counts; diagnostics decide nothing.
- L4. The build-time inversion is the product's inversion function with the
  product's default policy over the published σ domain, on the same support
  the product enforces; identical code, identical thresholds.
- L5. Every claim about `δ̂`, `p`, resolved points and statuses uses
  empirical language; no "bound", "guarantee", or "for every price".
- L6. A reference is prepared on the contract the handle it scores actually
  prices (probe contract for sizing probes, user contract for assembled
  surfaces), with every monetary quantity scaled alike.

## 5. Acceptance criteria

1. `make_iv_score_fn`, `make_fd_vega_refs_fn`, `compute_iv_error`, the TV/K
   constant, the vega-floor filter, and `kViabilityBound` no longer exist.
2. One factory prepares the stencil at High; a fake oracle asserts identical
   fine configs across the three fine solves, identical coarse configs across
   the three coarse solves, exact 2:1 nesting, and the same fine grid for
   `levels = 1` and `3`.
3. D2 admission is bitwise identical across candidates.
4. `effective_sigma_bracket` + `invert_price_on_surface` reproduce
   `InterpolatedIVSolver::solve` on existing and added fixtures across all
   instantiated surface families.
5. Every `PointStatus` has a unit fixture; ordering, viability, restart, veto,
   support and bin rules hold on fresh-only and holdout-only failures.
6. The #500 round-trip regression pins the measured outcome with provenance.
7. `bazel test //...` green; `//benchmarks/...` and `//src/python:mango_option`
   build; Rust layout test unchanged; Python diagnostics-keys test passes.
8. Calibration test passes with D8's classification and assertions; numbers
   and shifts are in the math doc.
9. Diagnostics report the eight D7 fields through C++ and Python; both
   refusal probes fire with their arguments.
10. Runtime of the adaptive test targets is measured before and after on the
    actual serial reference path and stated in the PR.
11. Both segmented sizing loops prepare references on the probe contract
    (non-ATM dividend fixtures on both backends).

## 6. Risks and assumptions

- **More refusals, by design.** Resolved sets shrink where vega is small;
  the coverage rule may refuse builds that pass today; edge-adjacent
  reference points can produce `SurfaceNoRoot` under the exact product
  policy and reject a candidate. Both are the shipped solver's behaviour
  measured honestly; `edge_band_rescues` quantifies the second. Nightly pins
  are re-measured, not loosened.
- **Cost.** Six High solves per point (≈ 3.75 fine-equivalents) instead of
  three default solves, on a serial reference path. Criterion 10 measures
  the real effect.
- **Calibration may fail** on the shipped oracle; D8 says what happens then.
- **Assumption:** `screen_bracket`, the pre-check and Brent are pure functions
  of the callables and the policy (criterion 4 checks it).
- **Assumption:** every handle can supply vega: the four named surfaces have
  `vega()`; the segmented Chebyshev sizing leaves are `TransformLeaf`s with
  analytic `vega`; the B-spline probe is a `BSplinePriceTable`.

## 7. Decisions (brainstorm record)

Each entry: question → options offered → choice → why.

**Q1. Target of this work.** Options: metric first; leaf first; both; close
with diagnosis. **Chosen: metric first.** Why: the metric admits points
where IV is barely identifiable and amplifies a $0.017 miss into 788 bps;
fixing it removes the false precision and stops the loop chasing
ill-conditioned points. The leaf oscillation becomes a separately filed,
honestly measured problem.

**Q2. Anchor for the criterion** (options: oracle conditioning `vega ≥
δ/(ρτ)`; relative vega threshold; match the query-time screen; conditioning
plus query-time alignment). **Withdrawn by the user**: "the metrics must be
based on sound math; possibly research a little first". A primary-source
research note followed, then a Codex mathematical review (both under
`docs/research/`).

**Q3. Shape of the score (after research, before Codex review).** Options:
three gates with per-point δ_ref and a secant verification; two gates with a
profile-level δ_ref; three gates with vomma from the bump pair. **Chosen at
the time: three gates, local δ_ref. Superseded** by Q5 after the review
showed the gates are diagnostics, not a certificate; bump-pair derivatives
cannot establish positive vega under the reference's own error; the secant
"true error" is the reference's inverse of the surface's price, not what the
product computes.

**Q4. Oracle accuracy profile.** Options: High; Ultra; Medium. **Chosen:
High**, provisional on D8. Why: matches Ultra to ~4e-6 at half the cost;
default was off by 3.3e-3 at ATM 1y; Medium's 3e-5 would push more points to
unresolved at a 5 bps target.

**Q5. Principle defining the IV-error metric.** Options: query-time round
trip with surface-independent bracket admission; reference bracket
certificate; price-space certification only. **Chosen: query-time round
trip.** Why: it measures the operation users consume with the product's own
algorithm; admission via the reference's own τ-bracket is derivative-free
and surface-independent; the price residual is kept as the forward-space
record.

**Q6. Resolved surface failures.** Options: universal; conditional domain;
budgeted fraction. **Chosen: universal** on the declared fresh and holdout
sets. Why: a max-error statement cannot ignore a resolved counterexample;
failures feed refinement; the alternatives add a contract or a constant.

**Q7. `vega_floor`.** Options: keep, ignore, deprecate; remove now.
**Chosen: keep and deprecate.** Why: no ABI churn in a metric change.

**Q8. The 0.20 viability constant.** Options: drop; keep. **Chosen: drop.**
Why: measured errors are genuine σ distances and resolved failures already
reject.

**Q9. Price-space acceptance.** Offered in Q5 as "price-space
certification only" and **not chosen**; the residual is recorded, never
gated. Consequence (review round 3, open question 3): unresolved regions
carry no forward-price acceptance requirement. Flagged for the go/no-go.

**Q10 (execution, 2026-09-21). Edge policy for acceptance.** Raised after
plan Task 10 measured the exact-bracket rule: six fixtures refused on a
single edge-adjacent sample, including the documented Pattern-4
configuration (2.7 bps miss below σ_min). Options: tolerance band for
acceptance with the exact bracket as a diagnostic; keep the exact bracket
and change the product's edge policy now; keep the exact bracket and accept
the refusals. **Chosen: tolerance band.** Why: a root within the user's own
tolerance beyond a published edge is a measurement, the shipped solver's
refusal there is a query-path policy to fix in the follow-up, and the
documented workflow must keep building. This reverses design-review rounds
1–2 on this one point, with the measured evidence as the reason.

**Design choices made without a question, with review verdicts:**

- One shared fine grid per stencil, strict-subsequence coarse grid (L2) —
  agreed (rounds 1–3); removes re-gridding differences, does not cancel bias.
- Inversion bracket — rev 2's τ_iv edge band was rejected twice; **rev 3+
  uses the exact product bracket; the band is a diagnostic only.**
  Consequence flagged for the go/no-go: edge-adjacent refusals reject
  candidates.
- Minimum-resolved threshold `max(4, N/4)` — agreed as an explicitly chosen
  coverage policy.
- Failure attribution — unconditional failure counts, no pseudo-error.
- `VegaTooSmall` pre-check — replicated via `SurfaceHandle::vega` and the
  shared inversion.
- `F_s = 3` with a calibrated profile-level `p` — agreed only as an
  empirical estimator; D8 carries classification, stability, sensitivity,
  and a failure policy.
- Three targets `y ± δ̂` — agreed as three-price testing; no interval claim.
- Monotonicity language in D2 — removed; D2 is a separation statement about
  estimated intervals plus target validity.
- Target validity (rev 4) — the product's full validation, including the
  upper no-arbitrage bound, at preparation (round 3 correction; descendant
  of L1).
- Segmented Chebyshev probe references on the probe contract (rev 4) — round
  3 correction; descendant of the B-spline adapter rule, now L6.
- Maturity support populated for segmented Chebyshev (rev 4) — round 3
  correction; moved from non-goals because leaving it unset contradicted L4.

**Gate 1 disposition.** Rounds 1 and 2 each reversed or completed a design
element. Round 3's critical findings were all deeper instances of laws the
spec already stated (L1, L4, and the probe-contract rule now L6), changed no
brainstorm decision, and came with prescribed fixes; they are folded here
and the gate is passed by convergence. The per-task and pre-merge reviews
re-check the same ground against real code.

**Design approval:** the user approved the five-section design on
2026-09-19 with "Yes, proceed". Revisions 2–4 change no user decision.
