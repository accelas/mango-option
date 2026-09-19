# Round-trip IV-error metric for adaptive validation (#500 remainder)

Issue: https://github.com/accelas/mango-option/issues/500 (the residual after
#504/#505). Research: `docs/research/2026-09-19-iv-inversion-conditioning.md`
(primary-source synthesis) and
`docs/research/2026-09-19-iv-inversion-conditioning-codex-review.md` (Codex
mathematical review of that synthesis; this design adopts its recommendation).

Spec revision 2, 2026-09-19. Revision 1 → 2 folds design-review round 1:
endpoint uncertainty in the admission test, an explicitly **empirical**
certification contract (no interval guarantee), the product's real inversion
policy for the round trip, a fully specified controlled grid family, and a
consistent fresh/holdout failure rule.

## 0. Contract of this metric (read first)

Everything this design reports is an **empirical operational measurement on
the declared sample set**:

- The reference is a numerical oracle (D1). Its uncertainty `δ̂` is a
  grid-convergence *estimate*, never a bound.
- "Resolved" (D2) means the oracle's three stencil prices are separated by
  more than their summed uncertainty estimates. It is a statement about the
  oracle, made before any candidate surface exists.
- The score (D3) is the outcome of running the product's inversion on the
  candidate surface at three target prices. It is not a proof about every
  price in an interval, and no such claim appears in code, diagnostics, or
  docs.
- "Universal" rejection (D4) means "no resolved failure on the declared
  fresh and holdout sets", not a statement about the whole domain.

This is the honest version of what the loop has always been: Latin-hypercube
sampling against a numerical reference. What changes is that every number
reported now means what it says, and no chosen constant hides inside the
metric except those listed in §4.11.

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
  after. The point sits on that shoulder; 788 bps is a $0.017 miss divided by
  a vega that is about to vanish. The linearised number extrapolates 56 bump
  widths across that shoulder and is not an IV error. (Whether the
  continuous model's plateau is exactly flat is not established and not
  needed: the numerical oracle cannot distinguish σ there, which is what
  matters for certification.)
- The same σ-shoulder exists everywhere deep ITM. Before the dividend, time
  value is exactly zero and the TV/K filter hides the point; just after, the
  deterministic dividend gain counts as time value and the filter admits it.
- The validation oracle (`make_validate_fn`) calls `solve_american_option`,
  which auto-estimates a grid with default `GridAccuracyParams` (`tol = 1e-2`,
  documented as ~1e-3 price accuracy). Measured against Ultra: 3.3e-3 at an
  ATM 1y put with three dividends, 4.6e-5 at the trigger point. Through a
  0.22 vega, 1e-3 of oracle noise is 45 bps by itself. The Chebyshev paths
  sample their tables at Ultra; segmented B-spline uses modified defaults and
  ordinary B-spline accepts caller grids, so in the common case the reference
  is noisier than the thing it grades.

Both admission constants, the vega floor, and the 0.20 bound are unanchored.
The user's requirement: the replacement must rest on sound mathematics, and
the choice was made after research and two independent mathematical reviews.

What the mathematics says (the research note and both reviews agree):

1. `|ΔV| / vega` is a first-order inverse image of a price discrepancy. It is
   meaningful only where the price is differentiable in σ with positive slope
   and the displacement stays where that slope is representative.
2. A derived quantity `σ = f(V)` can be certified to tolerance τ only where
   the input error times the condition number is below τ (Higham §1.6). With
   a numerical oracle the input error must be estimated, not assumed.
3. Local vega, bump-pair derivatives, and a two-grid Richardson number are
   diagnostics, not bounds. A finite bracket in σ, with uncertainty at
   **every** bracket point, is the derivative-free way to state "σ0 is
   distinguishable from σ0 ± τ".
4. The quantity the product delivers is the surface's own inverse of a price
   under the product's bracket and pre-check policy. Measuring anything else
   and calling it the product's error is a different experiment.
5. Admission must not depend on the candidate surface.
6. A scalar-or-nothing score cannot encode "error", "no inverse", "ambiguous
   inverse", "did not converge", and "reference cannot resolve".

## 2. Goals

- Replace the IV-error metric with one whose constants are the user's
  tolerance, measured quantities, or explicitly listed operational policies
  (§4.11). No time-value threshold, no vega floor, no 0.20 bound.
- Score what the product does, with the product's code and policy.
- Make the reference's accuracy explicit (declared profile, per-stencil
  uncertainty estimates) and make admission surface-independent.
- Report statuses and a price residual.
- Keep `run_refinement`'s role, candidate retention, the final validation
  step, the public builder APIs, and the C ABI layout.

## 3. Non-goals

- The σ-axis leaf resolution of the segmented Chebyshev builder (the real
  price error at the trigger). Follow-up issue, filed at PR time with this
  metric's honest numbers.
- Changing the product's query-time policy (`vega_threshold`,
  `adaptive_bounds`, edge behaviour). Follow-up; see D3 for the one place the
  metric deliberately deviates and says so.
- Removing `AdaptiveGridParams::vega_floor` from the C ABI, Rust, and Python
  surfaces (#463).
- A price-space acceptance gate. The residual is recorded, not gated.
- Statistical coverage guarantees or deterministic enclosures. See §0.
- Populating `RefinementContext::maturity_is_supported` for the segmented
  Chebyshev loop and final validation (today unset; Chebyshev handles return
  NaN inside event gaps and the existing non-finite veto handles a sample that
  lands there). Pre-existing; noted as a follow-up.
- Sampling, axis selection, and headroom rules.

## 4. Design

Notation per validation point `p = (S, K, τ, σ0, r)` with the builder's
dividend schedule and reference maturity: `V̂(σ)` is the reference price on
the fine grid, `V̂½(σ)` on the coarse grid, `S(σ) = handle.price(S, K, τ, σ, r)`
the candidate surface, `τ_iv = params.target_iv_error`.

### D1. Reference oracle: declared profile, one controlled stencil

**Profile.** The reference solves at `kReferenceAccuracy =
GridAccuracyProfile::High` (constant in `adaptive_metrics.cpp`, with the
measurement that chose it: High matches Ultra to ~4e-6 at half the cost;
default was off by 3.3e-3 at ATM 1y). "Matches Ultra" is evidence of the
profile's adequacy, not an error bound; D8 re-measures it.

**Stencil.** The reference for one point is six solves:

| Solve | σ | Grid |
|---|---|---|
| `y`, `y½` | σ0 | G, G½ |
| `lo`, `lo½` | σ0 − τ_iv | G, G½ |
| `hi`, `hi½` | σ0 + τ_iv | G, G½ |

`G` is chosen **once per stencil**: `estimate_pde_grid` at High for the
contract at σ0 + τ_iv (the widest x-domain of the three), then the point
count is rounded up to `n ≡ 1 (mod 4)` (at most two extra points) so that
the coarse grid below keeps the centre node and an odd count. All three fine
solves receive that `GridSpec` and `TimeDomain` as an explicit
`PDEGridConfig` (dividend event times are merged into any explicit config by
`resolve_grid`, `american_option.cpp:68`; verified).

**Controlled coarse grid `G½`.** Spatial: the strict subsequence of every
other node of `G` (the multi-sinh clustering, domain, centre node and
oddness are preserved exactly; refinement ratio 2 by construction).
Temporal: the same `mandatory_times`, `n_time = ⌈n_t/2⌉`. Because
`TimeDomain::with_mandatory_points` rounds per segment, the temporal ratio
is 2 up to per-segment rounding; D8 measures the achieved order with the same
construction, so `p` reflects the family actually used.

**Uncertainty estimates.** For each of the three σ:
`δ̂_k = F_s · |V̂_k − V̂½_k| / (2^p − 1)`, `F_s = 3` (Roache's two-grid
safety factor), `p = kReferenceConvergenceOrder` measured by D8. A zero
difference yields `δ̂_k = 0` legitimately (e.g. the price is intrinsic on
both grids). `δ̂` is an *estimate*: a two-grid difference cannot see bias
shared by both grids (domain truncation, obstacle handling, dividend
interpolation), and a profile-level `p` does not establish pointwise order.
Every consumer of `δ̂` in this design is empirical (D2's admission, D3's
three target prices, D5's diagnostic floor); none claims an enclosure.

**Types.**

```cpp
struct ErrorRefs {
    double ref_price;                    // y
    double bracket_lo_price, bracket_hi_price;
    double sigma_lo, sigma_hi;           // σ0 ∓ τ_iv as solved
    double delta, delta_lo, delta_hi;    // δ̂ at σ0, σ_lo, σ_hi (estimates)
    bool resolved;                       // D2, decided here
};
```

`PrepareRefsFn`'s outward signature is unchanged. `make_fd_vega_refs_fn` is
replaced by `make_stencil_refs_fn(params, oracle)` where `oracle` is a small
value type owning the dividend schedule, reference maturity, type and yield;
it exposes `solve(spot, strike, tau, sigma, rate, const PDEGridConfig&)`. The
factory rolls dividends once, estimates `G`, derives `G½`, and issues the six
solves; there is no mutable "current grid" state and no per-σ re-estimation
(L2). `make_validate_fn` remains for callers that want one price (tests,
diagnostics) and is implemented on the same oracle type at High.

**Segmented B-spline probe adapter** (`bspline_adaptive.cpp:771`): the probe
solves at `(spot/scale, K_ref)` and rescales monetary quantities by `scale =
K/K_ref`. The adapter must scale `ref_price`, both bracket prices, and all
three `δ̂` by `scale`; σ coordinates and `resolved` are unchanged (resolution
is scale-invariant because every term scales alike). A test with an
asymmetric reference strike pins this.

**Reference σ domain.** The oracle accepts any finite positive σ
(`option_spec.cpp:136`). If `σ0 − τ_iv ≤ 0`, or any stencil coordinate is
non-finite, the point is `ReferenceUnresolved` (two-sided admission only; no
one-sided rule).

### D2. Resolution: surface-independent admission

A point is **resolved at τ_iv** iff all six stencil prices are finite and

```
y − lo  >  δ̂ + δ̂_lo        and        hi − y  >  δ̂_hi + δ̂ .
```

With uncertainty at every bracket point this is the finite-bracket condition
of the prior review (§3.B) applied to the reference: the oracle's price at σ0
is separated from its prices at σ0 ± τ_iv by more than the summed
uncertainty estimates on each side. It requires no vega, no intrinsic value,
and no monotonicity assumption for the *statement* (it only compares three
numbers); interpreting a resolved bracket as "σ localised to ±τ_iv" does
assume the reference is non-decreasing in σ across the bracket, which holds
for American prices with convex payoff (Ekström 2004) and is what makes the
round trip meaningful. Where that assumption fails numerically (`lo > y` or
`y > hi`) the point is unresolved by the inequalities themselves.

It subsumes both old filters: in the exercise region `lo = y = hi` exactly;
where vega is small the separation falls inside the estimates.

Unresolved points have status `ReferenceUnresolved`. They enter no IV
statistic and no refinement bin, are counted (D7), and still record a price
residual (D3). Resolution is decided when references are prepared, before
any candidate exists, so it is identical for every candidate (L1).

**Coverage policy (chosen, not derived).** Today the loop refuses
(`ValidationFailed`) when fewer than `max(4, validation_samples/4)` holdout
points have usable references; `validation_samples` is the requested count,
and unsupported-maturity samples are skipped before counting. This design
keeps that rule on *prepared* references and adds the same threshold on
*resolved* references, for the holdout and for the final validation set. The
threshold is an operational coverage policy (§4.11): "a build that cannot
resolve its own tolerance on a quarter of the requested sample cannot
certify it". When the rule refuses, the returned `PriceTableError` cannot
carry counts (no ABI change); the refusal fires the existing USDT build
probes with the prepared/resolved/unsupported counts, and the log-free
library contract is kept.

### D3. Score: the operational round trip

**Seam.** The score changes from `std::optional<double>` to a status:

```cpp
enum class PointStatus {
    Measured,              // inversion succeeded at all three targets
    ReferenceUnresolved,   // D2: not evidence
    SurfaceVegaTooSmall,   // product pre-check refused
    SurfaceNoRoot,         // no sign change on the product bracket
    SurfaceAmbiguous,      // multi-root screen or post-Brent slope refused
    SurfaceNonConvergent,  // Brent hit max_iter
    SurfaceNonFinite,      // NaN/inf surface price or vega during inversion
};
struct PointScore {
    PointStatus status;
    double iv_error;        // max |σ̂_k − σ0| over the targets, iff Measured
    double price_residual;  // |S(σ0) − y| / K when S(σ0) is finite, else NaN
};
using ScoreErrorFn = std::function<PointScore(
    const SurfaceHandle& handle, const ErrorRefs& refs,
    double spot, double strike, double tau, double sigma, double rate)>;
```

`SurfaceHandle` gains `vega` (same signature as `price`); every builder's
handle fills it from the surface's `vega()` (all four surface families expose
one). The scorer receives the handle because the round trip evaluates the
surface along σ; it performs no PDE solves. Domain data (`ctx.sample_bounds`,
the inversion policy) is captured in the scoring factory, not passed per
call.

**The product's inversion, shared.** `InterpolatedIVSolver<Surface>::solve`
is refactored so that everything after query validation lives in a free
function template

```cpp
struct SurfaceInversionPolicy {   // defaults == InterpolatedIVSolverConfig defaults
    double sigma_min, sigma_max;  // published σ domain
    double vega_threshold;        // 1e-4
    bool   detect_multiple_roots; // true
    double tolerance;             // 1e-6
    size_t max_iter;              // 50
};
std::expected<IVSuccess, IVError> invert_price_on_surface(
    PriceFn price, VegaFn vega, const IVQuery& q, double rate,
    const SurfaceInversionPolicy& policy);
```

that performs `adaptive_bounds` (published domain ∩ config ∩ the
price-dependent 1.5/2.0/3.0 cap), the quartile-vega pre-check, the 17-point
screen (`detail::screen_bracket`) with its boundary-root return and narrowed
bracket, Brent, and the post-Brent slope check, in the current order and with
the current error codes. The solver calls it; the loop calls it with the
default policy over the candidate's published σ domain (`ctx.sample_bounds`).
Behaviour of the product path is unchanged (L4), verified by the existing
solver tests.

**One deliberate deviation, named.** The loop's bracket is the published σ
domain widened by exactly τ_iv on each side (clipped to the fit domain,
`ctx.bounds`). The product does not widen. Reason: the reference sample σ0 is
drawn from the published domain up to its edges, and a surface whose inverse
of the reference price lies within τ_iv beyond an edge is within the declared
tolerance of a point at that edge; the product would refuse it today
(`BracketingFailed`), which is a query-path edge policy worth fixing, not
evidence about the surface. The metric is therefore named
`operational round trip (edge band τ_iv)` in code, docs and diagnostics, and
the query-time alignment follow-up proposes giving the product the same band
(returning the edge-clamped σ with a flag). Every other policy element is the
product's. Widening to the whole fit domain was rejected: Chebyshev headroom
extends σ to 0.01 where the fit is support, not a surface, and its wiggles
would produce refusals the product never sees; ordinary B-spline has no
support beyond the published σ range at all, so backends would differ.

**Three targets.** For a resolved point the inversion runs for
`y − δ̂, y, y + δ̂`. The point is `Measured` iff all three succeed;
`iv_error = max_k |σ̂_k − σ0|`. Otherwise the status is the most severe
failure among the three (NonFinite > NonConvergent > Ambiguous > NoRoot >
VegaTooSmall). What this measures: the product's answer at three prices the
oracle could have meant, given its estimated uncertainty. What it does not
claim: anything about prices strictly between them; the 17-point screen
documents folds it cannot detect (`interpolated_iv_solver.hpp:62`), and
Brent's stop is a residual/bracket condition, not an exact inverse. When
`τ_iv` is below Brent's price tolerance mapped through the surface's slope,
the reported error carries that resolution floor; D7 reports the policy so a
reader can see it.

**Error-code mapping.** `IVErrorCode::VegaTooSmall → SurfaceVegaTooSmall`;
`BracketingFailed → SurfaceNoRoot`; `MultipleRoots → SurfaceAmbiguous`;
`MaxIterationsExceeded → SurfaceNonConvergent`; `NumericalInstability` and
any non-finite price/vega → `SurfaceNonFinite`. A `boundary_root` success is
`Measured`. Failure statuses are operational outcomes; `SurfaceNoRoot` on a
non-monotone surface is not proof the price is outside its range, and the
docs say so.

**Price residual.** `|S(σ0) − y| / K` for every referenced point whose
surface price is finite, resolved or not; NaN otherwise (and the non-finite
veto below applies).

### D4. Loop consumption: fresh and holdout, ranking, viability, refinement

`SampleEval` and `FinalScore` gain `unresolved`, `surface_failures`,
`max_price_residual`; `measured` counts `Measured` only; `filtered` is
replaced by `unresolved`. Both the fresh pass and the holdout pass produce
these.

**Viability of a candidate** (replaces the `kViabilityBound` clause):

```
all fresh and holdout surface prices finite (existing veto, kept even at
unresolved points)  ∧  holdout.measured > 0  ∧
fresh.surface_failures == 0  ∧  holdout.surface_failures == 0
```

`fresh_converged` is `fresh.measured > 0 ∧ fresh.max ≤ τ_iv ∧
fresh.surface_failures == 0`. The loop's convergence break and its
`target_met` keep their existing shape: `target_met = picked.viable ∧
picked.holdout_max ≤ τ_iv ∧ picked.fresh_converged`. `FinalScore::viable()`
is `all_finite ∧ measured > 0 ∧ surface_failures == 0`, and the segmented
final gates and the B-spline `needs_final_retry` / `select_final_surface`
keep their logic on top of it (both already route through `viable()`; a
comparator that orders by failures first, then max, then avg replaces the
bare `max_error <` in `select_final_surface`).

**Ordering** wherever candidates are compared (exploration-base advance,
retention pick, final pick): fewer `surface_failures` → lower holdout max →
lower holdout avg → earlier iteration. A candidate whose resolved holdout
points all fail (`measured == 0`, `surface_failures > 0`) remains eligible as
an *exploration base* (it carries bins) but never as a returned candidate.

**Walk restart.** The 2 % relative-improvement rule applies to the holdout
max as today; in addition, any reduction in `surface_failures` counts as a
measured improvement and restarts the axis walk.

**Refinement bins.** `ErrorBins` gains `failure_counts[dim][bin]`, recorded
unconditionally for every surface failure (no pseudo-error, no threshold).
`pick_refinement_axis` and `problematic_bins` use `bin_counts + failure_counts`.
Holdout failures are attributed too: `evaluate_holdout` now returns bins for
its failures (it still performs no solves), and `Candidate::bins` merges
fresh and holdout attribution. Measured errors above τ_iv are recorded as
today.

### D5. Monotonicity scan

`scan_monotonicity` stays a diagnostic. Its noise floor becomes the per-point
`max(refs.delta, refs.delta_lo, refs.delta_hi)` (floored at `1e-8 · spot`);
its `vega_floor` parameter goes. The doc comment names it a reporting
threshold, not monotonicity evidence.

### D6. `vega_floor` deprecation

`AdaptiveGridParams::vega_floor` stays in the struct, the C ABI (offset 56
asserts untouched), the Rust crate, and the Python binding. C header comment,
Rust doc, Python docstring and the C++ comment say: deprecated, ignored since
this change, removed in the ABI revision (#463). The loop no longer validates
it; `benchmarks/interp_iv_safety.cc`'s `kVegaFloor` mirror goes.

### D7. Diagnostics

`BuildDiagnostics` gains

```cpp
size_t holdout_points_unresolved = 0;   // oracle could not resolve target
size_t surface_failures = 0;            // returned surface on the holdout
double max_price_residual = 0.0;        // |S − V̂|/K over referenced points
double reference_uncertainty_max = 0.0; // max δ̂ over the holdout (estimate)
```

`IterationStats` gains `surface_failures` and `unresolved` (C++ only; the
Python converter exposes no per-iteration entries today and this does not
change). The Python `build_diagnostics` property's dict gains the four keys.
Rust exposes no diagnostics; nothing changes there. Segmented final gates
fill the same fields from `FinalScore`. A returned surface always has
`surface_failures == 0` by D4; the field exists so `FinalScore`, the
per-iteration record and the USDT refusal probe share one vocabulary, and so
a future non-universal policy has a home.

**Solve accounting** is centralised: `PrepareRefsFn` implementations report
their attempts through one counter type (`ReferenceSolveCount{fine, coarse}`)
that `prepare_final_validation`, the holdout preparation and the fresh pass
all add to, replacing the scattered `× 3` multipliers. `total_pde_solves`
counts fine and coarse solves separately in `BuildDiagnostics`
(`reference_solves_fine`, `reference_solves_coarse`); callers that today
multiply by 3 read the counter instead.

### D8. Calibration of `p` and the profile

A nightly `slow` test (`reference_oracle_calibration_test`) solves a fixed
six-point set at High on three controlled resolutions (G, G½, G¼, each
derived by the D1 subsequence rule): ATM 1y put with three $0.50 dividends,
the #500 trigger point, an OTM 30-day put, a deep-OTM 7-day put, an ITM 2y
put, an ATM 6-month call. Per point it computes `d1 = V¼ − V½`, `d2 = V½ −
V` and classifies: `d1·d2 ≤ 0` or `d2 == 0` → **degenerate** (oscillatory or
converged to resolution); else `p_obs = ln(d1/d2)/ln 2`. It asserts: no
point is degenerate at the High grid sizes; every `p_obs` is finite and
positive; `kReferenceConvergenceOrder ≤ min p_obs`; `|V_High − V_Ultra| ≤
δ̂_High` at every point. It additionally reruns two points with the x-domain
widened by one σ√T and with the solver's algebraic tolerance tightened one
decade, and reports the price shifts next to `δ̂`, so shared-bias
sensitivity is visible in the calibration record (reported, not asserted;
the doc states the numbers). If a future change makes a point degenerate, the
test fails and the constant is revisited; per-point `p` is not measured at
build time.

### D9. Regression coverage

- `tests/adaptive_refinement_unit_test.cc` (synthetic references and
  surfaces, no PDE): D2 admission with overlapping endpoint intervals (the
  reviewer's `y=10, lo=9.85, hi=10.15, δ=0.10` example is unresolved),
  non-monotone stencils, `σ0 − τ_iv ≤ 0`, scale-invariance under the B-spline
  probe rescaling; each `PointStatus` from a purpose-built surface (exact,
  biased, edge-shifted inside/outside the τ_iv band, decreasing crossing,
  fold between two targets caught only if it falls on a screen point — the
  test documents that limit, NaN interior, low surface vega, forced
  `max_iter` exhaustion); three-target aggregation and severity ordering;
  viability with fresh-only and holdout-only failures; `measured == 0` base
  eligibility; failure bins recorded when τ_iv exceeds the bracket width;
  walk restart on failure reduction; `select_final_surface` with both
  failing, ties on max, and failures vs max; solve counters.
- `tests/interpolated_iv_solver_test.cc` (or the existing solver test file):
  `invert_price_on_surface` reproduces `solve()` bit-for-bit on the existing
  fixtures (L4).
- `tests/adaptive_grid_types_test.cc`: `vega_floor` accepted at any value;
  Python test asserts the new diagnostic keys and that `vega_floor` is
  ignored.
- `tests/adaptive_surface_build_slow_test.cc`:
  `WideBandDividendBracketRemainsViable` becomes
  `WideBandDividendBracketRoundTrip` on the same manual two-K_ref fixture:
  the point must be **resolved** (asserted, so the test cannot pass vacuously)
  and **Measured** with error below a bound measured on that exact fixture
  during implementation and recorded in the test with its provenance (the
  manual and adaptive Chebyshev paths have different σ headroom, so the
  number is fixture-specific). `// Bug:` line states the 788 bps artifact.
- `tests/adaptive_grid_builder_test.cc`: retry-path tests updated for the
  status comparator.
- Existing pins that assert `kViabilityBound`, `filtered`, `vega_floor`, or
  numbers produced by the old metric's amplification are re-measured under
  the new one and regenerated with the reason in the commit message.
- A segmented case with an actual event gap confirms the non-finite veto and
  status accounting on the real builder.

### D10. Documentation

`docs/MATHEMATICAL_FOUNDATIONS.md` gets "Adaptive validation metric": the
stencil, the admission inequalities, the round trip and its three targets,
the edge band, the calibration numbers, the empirical contract of §0, and
the constant inventory of §4.11 with citations from the research note.
`docs/API_GUIDE.md` documents the statuses, the new diagnostics, and the
deprecation. `docs/ARCHITECTURE.md` names the shared inversion component.
`CONTEXT.md` gains three terms: *reference-resolved point*, *operational
round trip*, *surface inversion failure* (ADR 0001's split holds: C++ tests
own numerical correctness, Python tests own reachability).

### 4.11 Constant inventory (honest list)

| Constant | Kind | Where |
|---|---|---|
| `target_iv_error` | user | `AdaptiveGridParams` |
| `δ̂`, `p` | measured | D1, D8 |
| `F_s = 3` | literature convention (Roache, two-grid) | D1 |
| `kReferenceAccuracy = High` | chosen from measurement | D1 |
| coverage `max(4, N/4)` on prepared and on resolved | operational policy (pre-existing rule, now applied twice) | D2 |
| inversion policy: `vega_threshold 1e-4`, 17 screen points, zero-tol `1e-9·spot`, Brent `1e-6` / 50 iters, 1.5/2/3 cap | product policy, reused unchanged | D3 |
| edge band `τ_iv` | the user's tolerance, reused | D3 |
| monotonicity-scan floor `1e-8·spot` | diagnostic floor, pre-existing | D5 |
| walk restart 2 % | pre-existing loop policy | D4 |

Nothing else numeric appears in the metric.

### Binding laws (govern every not-yet-enumerated instance)

- L1. Admission never reads the candidate surface.
- L2. Every stencil shares one fine grid; the coarse grid is the strict
  every-other-node subsequence of it; nothing is re-estimated per σ.
- L3. A status is never converted into a number for acceptance or
  attribution; only `Measured` errors enter `max`/`avg`/thresholded bins;
  failures have their own counts.
- L4. The build-time inversion is the product's inversion function with the
  product's default policy; the only difference is the named edge band.
- L5. Every claim in code comments, diagnostics and docs about `δ̂`, resolved
  points and statuses uses empirical language ("estimate", "outcome",
  "on the declared sample set"); no "bound", "guarantee", or "for every
  price" appears.

## 5. Acceptance criteria

1. `make_iv_score_fn`, `make_fd_vega_refs_fn`, `compute_iv_error`, the TV/K
   constant, the vega-floor filter, and `kViabilityBound` no longer exist.
2. The stencil is prepared by one factory at High on one grid per point
   (test: a fake oracle records the `PDEGridConfig` of every call and asserts
   identity across the six solves and the exact 2:1 node subsequence).
3. D2 admission is bitwise identical across candidates (two different
   handles, same `resolved` flags).
4. `invert_price_on_surface` reproduces `InterpolatedIVSolver::solve` on the
   existing solver fixtures.
5. Every `PointStatus` has a unit fixture; ordering, viability, restart, and
   bin rules hold on fresh-only and holdout-only failures.
6. The #500 round-trip regression asserts resolved + measured with a
   fixture-measured bound.
7. `bazel test //...` green; `//benchmarks/...` and `//src/python:mango_option`
   build; Rust layout test unchanged; Python diagnostics keys test passes.
8. Calibration test measures `p` with the D8 classification; the constant is
   ≤ the measured minimum; numbers and sensitivity shifts are in the math doc.
9. Diagnostics report unresolved counts, failures, max residual, max δ̂, and
   fine/coarse solve counts through C++ and Python.
10. Runtime of the adaptive test targets is measured before and after on the
    actual serial reference path and stated in the PR; no prediction is
    asserted in the spec.

## 6. Risks and assumptions

- **More refusals, by design.** At τ_iv = 5 bps the resolved set shrinks
  where vega is small; the coverage rule may refuse builds that pass today.
  The USDT probe and the unresolved counts say why. Nightly viability pins
  are re-measured, not loosened.
- **Edge refusals.** Even with the τ_iv band, a surface whose inverse of an
  edge-adjacent reference price lands farther than τ_iv outside the published
  domain is a `SurfaceNoRoot` and rejects the candidate. This is the product's
  behaviour minus the band. If this dominates refusals in practice, that is
  the query-time edge-policy follow-up's evidence, not a reason to soften L3.
- **Cost.** Six High solves per point (three fine + three coarse ≈ 3.75
  fine-equivalents) instead of three default solves, on a serial reference
  path. Measured single-thread: ATM 1y with dividends 0.39 s at High. A
  64-point set ≈ 90 s serial per preparation. Acceptance criterion 10 measures
  the real effect; parallelising the reference loop is a separate change.
- **`p` per profile.** D8's classification catches degenerate convergence on
  the calibration set only. Between calibration points `δ̂` may be optimistic
  or pessimistic; both effects are visible in the unresolved counts and the
  residuals, and neither is claimed away.
- **Assumption:** `screen_bracket`, Brent and the pre-check are pure functions
  of the objective/vega callables and the policy, so extraction cannot change
  the product path (criterion 4 checks it).
- **Assumption:** every surface handed to the loop exposes `vega()`. True for
  `BSplinePriceTable`, `BSplineMultiKRefInner`, `ChebyshevSurface`,
  `ChebyshevMultiKRefSurface` today.

## 7. Decisions (brainstorm record)

Each entry: question → options offered → choice → why.

**Q1. Target of this work.** Options: metric first; leaf first; both; close
with diagnosis. **Chosen: metric first.** Why: the metric admits points
where IV is barely identifiable and amplifies a $0.017 miss into 788 bps;
fixing it removes the false precision and stops the loop chasing
ill-conditioned points. The leaf oscillation stays visible and becomes a
separately filed, honestly measured problem.

**Q2. Anchor for the criterion** (options: oracle conditioning `vega ≥
δ/(ρτ)`; relative vega threshold; match the query-time screen; conditioning
plus query-time alignment). **Withdrawn by the user** with the instruction
"the metrics must be based on sound math; possibly research a little
first". A primary-source research note was produced, then reviewed by Codex
(both under `docs/research/`).

**Q3. Shape of the score (after research, before Codex review).** Options:
three gates with per-point δ_ref and a secant verification; two gates with a
profile-level δ_ref; three gates with vomma from the bump pair. **Chosen at
the time: three gates, local δ_ref. Superseded** by Q5 after the Codex review
showed the three gates are diagnostics, not a certificate, that bump-pair
derivatives cannot establish positive vega under the reference's own error,
and that the secant "true error" is the reference's inverse of the surface's
price, not what the product computes.

**Q4. Oracle accuracy profile.** Options: High; Ultra; Medium. **Chosen:
High.** Why: matches Ultra to ~4e-6 at half the cost; default was off by
3.3e-3 at ATM 1y; Medium's 3e-5 would push more points to unresolved at a
5 bps target. Provisional on D8's calibration.

**Q5. Principle defining the IV-error metric.** Options: query-time round
trip with surface-independent bracket admission; reference bracket
certificate (pass/fail/unresolved, ladder for a scalar); price-space
certification only. **Chosen: query-time round trip.** Why: it measures the
operation users consume with the product's own algorithm, needs no
derivative and no vega; admission via the reference's own τ-bracket is
derivative-free and surface-independent; the price residual is kept as the
forward-space record.

**Q6. Resolved surface failures.** Options: universal (any resolved failure
rejects); conditional domain (publish unsupported regions); budgeted failure
fraction. **Chosen: universal** (on the declared fresh and holdout sets).
Why: a max-error statement cannot ignore a resolved counterexample; failures
feed refinement so the loop can repair them; the other two add a new
contract or a new constant.

**Q7. `vega_floor`.** Options: keep, ignore, deprecate; remove now.
**Chosen: keep and deprecate.** Why: no ABI or binding churn in a metric
change; removal belongs to #463.

**Q8. The 0.20 viability constant.** Options: drop; keep as sanity bound.
**Chosen: drop.** Why: measured errors are genuine σ distances and resolved
failures already reject; the constant was the last unanchored number in the
gate.

**Design choices made without a question (validated in review round 1;
reviewer verdicts folded):**

- One shared fine grid per stencil and a strict-subsequence coarse grid (L2)
  — agreed; the spec now says it removes re-gridding differences and does
  not establish error cancellation.
- Inversion bracket: the product's policy plus a τ_iv edge band, named as a
  deviation (D3) — reviewer preferred the exact product bracket; the band is
  kept because refusing a surface for a root within the user's own tolerance
  of an edge would reject most builds on a query-path policy defect, and the
  metric's name, docs and the follow-up state the difference. Flagged for the
  go/no-go.
- Minimum-resolved threshold reuses `max(4, N/4)` — kept as an explicitly
  chosen coverage policy (D2, §4.11), not a mathematical consequence.
- Failures in refinement bins — now unconditional failure counts (D4), no
  pseudo-error weight (reviewer's correction accepted).
- The product's `VegaTooSmall` pre-check — now replicated via `SurfaceHandle::
  vega` and the shared inversion (reviewer's correction accepted).
- `F_s = 3` with a profile-level measured `p` — kept as an empirical
  estimator with D8's degeneracy classification and sensitivity record; all
  certificate language removed (reviewer's correction accepted).
- Three targets `y ± δ̂` — kept as an empirical three-price test; the
  interval claim is withdrawn (reviewer's correction accepted).

**Design approval:** the user approved the five-section design on
2026-09-19 with "Yes, proceed". Revision 2 changes no user decision; it
tightens the mathematics and the product fidelity per review round 1.
