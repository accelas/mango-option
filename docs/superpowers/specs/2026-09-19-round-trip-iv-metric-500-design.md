# Round-trip IV-error metric for adaptive validation (#500 remainder)

Issue: https://github.com/accelas/mango-option/issues/500 (the residual after
#504/#505). Research: `docs/research/2026-09-19-iv-inversion-conditioning.md`
(primary-source synthesis) and
`docs/research/2026-09-19-iv-inversion-conditioning-codex-review.md` (Codex
mathematical review of that synthesis; this design adopts its recommendation
in part: operational round trip plus reference brackets; forward-price
validation is recorded, not gated — see §3).

Spec revision 3, 2026-09-19.
Rev 1 → 2 (review round 1): endpoint uncertainty in the admission test, an
explicitly empirical contract, the product's inversion policy, a controlled
grid family, consistent fresh/holdout failure rules.
Rev 2 → 3 (review round 2): the **exact** product bracket is the acceptance
target (the edge band becomes a diagnostic only); the shared inversion
function carries the full policy (configured limits, published limits,
adaptive cap and its fallback, target validation); a nested grid family that
survives three coarsenings; a calibration protocol with order-stability and
effective sensitivity checks; partial-stencil and solve-counter contracts;
holdout-only failure ordering; enumerated diagnostics and a refusal probe;
the #500 fixture is measured before it is pinned.

## 0. Contract of this metric (read first)

Everything this design reports is an **empirical operational measurement on
the declared sample set**:

- The reference is a numerical oracle (D1). Its uncertainty `δ̂` is a
  grid-convergence *estimate*, never a bound; `p` is a *calibrated constant*.
- "Resolved" (D2) means the oracle's three stencil price intervals
  `[V̂ − δ̂, V̂ + δ̂]` are pairwise separated in the expected order. It is a
  statement about three numbers, made before any candidate exists. It is not
  a monotonicity, uniqueness, or continuous-model localisation claim.
- The score (D3) is the outcome of running the **shipped** inversion on the
  candidate surface at three target prices. Nothing is claimed about prices
  between them.
- "Universal" rejection (D4) means "no resolved failure on the declared fresh
  and holdout sets", not a domain-wide maximum guarantee.
- Forward-price accuracy is recorded as a residual and never gated;
  unresolved regions carry no price acceptance requirement (partial adoption
  of the prior review's recommendation, stated as such).

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
  distinguish σ there, which is what matters.
- The same σ-shoulder exists everywhere deep ITM. Before the dividend, time
  value is exactly zero and the TV/K filter hides the point; just after, the
  deterministic dividend gain counts as time value and the filter admits it.
- The validation oracle (`make_validate_fn`) calls `solve_american_option`,
  which auto-estimates a grid with default `GridAccuracyParams` (`tol = 1e-2`,
  documented as ~1e-3 price accuracy). Measured against Ultra: 3.3e-3 at an
  ATM 1y put with three dividends, 4.6e-5 at the trigger point. Through a
  0.22 vega, 1e-3 of oracle noise is 45 bps by itself. Chebyshev paths sample
  their tables at Ultra; segmented B-spline uses modified defaults; ordinary
  B-spline accepts caller grids.

Both admission constants, the vega floor, and the 0.20 bound are unanchored.
The user's requirement: the replacement must rest on sound mathematics.

What the mathematics says (the research note and both reviews agree):

1. `|ΔV| / vega` is a first-order inverse image of a price discrepancy,
   meaningful only where the price is differentiable in σ with positive slope
   and the displacement stays where that slope is representative.
2. A derived quantity `σ = f(V)` can be certified to tolerance τ only where
   the input error times the condition number is below τ (Higham §1.6). With
   a numerical oracle the input error must be estimated, not assumed.
3. Local vega, bump-pair derivatives, and a two-grid Richardson number are
   diagnostics, not bounds. A finite bracket in σ with uncertainty at every
   bracket point is the derivative-free way to state "σ0 is distinguishable
   from σ0 ± τ" for the oracle.
4. The quantity the product delivers is the surface's own inverse of a price
   under the product's bracket and pre-check policy. Measuring anything else
   and calling it the product's error is a different experiment.
5. Admission must not depend on the candidate surface.
6. A scalar-or-nothing score cannot encode "error", "no inverse", "ambiguous
   inverse", "did not converge", and "reference cannot resolve".

## 2. Goals

- Replace the IV-error metric with one whose constants are the user's
  tolerance, measured or calibrated quantities, or explicitly listed
  operational policies (§4.11). No time-value threshold, no vega floor, no
  0.20 bound.
- Score what the product does, with the product's code and policy.
- Make the reference's accuracy explicit and admission surface-independent.
- Report statuses and a price residual.
- Keep `run_refinement`'s role, candidate retention, the final validation
  step, the public builder APIs, and the C ABI layout.

## 3. Non-goals

- The σ-axis leaf resolution of the segmented Chebyshev builder. Follow-up
  issue, filed at PR time with this metric's measured numbers.
- Changing the product's query-time policy (`vega_threshold`,
  `adaptive_bounds`, edge behaviour). The edge-band diagnostic (D3) exists to
  inform that follow-up; it decides nothing.
- Removing `AdaptiveGridParams::vega_floor` from the C ABI, Rust, Python (#463).
- A price-space acceptance gate.
- Statistical coverage guarantees or deterministic enclosures. See §0.
- Populating `RefinementContext::maturity_is_supported` for the segmented
  Chebyshev loop and final validation (today unset; Chebyshev handles return
  NaN inside event gaps and the non-finite veto handles a sample that lands
  there). Pre-existing; follow-up.
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

**Nested grid family (one constructor for production and calibration).**
`make_reference_grid_family(params, accuracy, levels)`:

1. `estimate_pde_grid(params, accuracy)` gives `(GridSpec G0, TimeDomain T0)`.
2. Spatial: the point count `n` of `G0` is rounded **up** to
   `n ≡ 1 (mod 2^(levels+1))` (production `levels = 1` → mod 4, at most two
   extra points; calibration `levels = 3` → mod 16) and the multi-sinh spec is
   re-sampled at that count with the same domain, centres and weights. Level
   `k` is the strict subsequence of every `2^k`-th node of level 0. By the
   rounding, every level has an odd count and shares level 0's middle-index
   node (whatever physical point it is); the refinement ratio is exactly 2
   per level. Even counts are never produced; the claim is tested on the
   generated nodes.
3. Temporal: level 0 requests `n_time = T0.n_steps()`; level `k` requests
   `⌈n_time / 2^k⌉`. `mandatory_times` holds the **event times only**
   (`resolve_grid` merges dividend taus into any explicit config, verified at
   `american_option.cpp:68`); no fine time nodes are copied. Because
   `TimeDomain::with_mandatory_points` rounds per event segment, the achieved
   temporal ratio is 2 up to per-segment rounding and can be 1:1 inside a
   very short event interval. D8 measures `p` with this same constructor, so
   the calibrated order reflects the family actually used; the family's
   achieved step counts are recorded in `ErrorRefs` for the calibration
   record (not used by the loop).

**Stencil.** Six solves per point, `G = level 0`, `G½ = level 1`, chosen once
for the contract at σ0 + τ_iv (the widest x-domain of the three):

| Solve | σ | Grid |
|---|---|---|
| `y`, `y½` | σ0 | G, G½ |
| `lo`, `lo½` | σ0 − τ_iv | G, G½ |
| `hi`, `hi½` | σ0 + τ_iv | G, G½ |

**Uncertainty estimates.** For each σ: `δ̂_k = F_s · |V̂_k − V̂½_k| /
(2^p − 1)`, `F_s = 3` (Roache's two-grid safety factor), `p =
kReferenceConvergenceOrder` calibrated by D8. A zero difference yields
`δ̂_k = 0` (legitimate where the price is intrinsic on both grids). `δ̂` is an
*estimate*: a two-grid difference cannot see bias shared by both grids
(domain truncation, obstacle handling, dividend interpolation), and a
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
- `σ0 − τ_iv ≤ 0`, a non-finite stencil coordinate, or any bracket/coarse
  solve failing → success with `resolved = false`, base price present,
  unavailable fields NaN. The residual (D3) is still computed for such a
  point; it is `ReferenceUnresolved`.

`make_fd_vega_refs_fn` is replaced by `make_stencil_refs_fn(params, oracle,
counter)`: `oracle` is a value type owning the dividend schedule, reference
maturity, option type and yield and exposing `solve(spot, strike, tau, sigma,
rate, const PDEGridConfig&)`; `counter` is a `std::shared_ptr<ReferenceSolve
Counter>` (`fine_attempts`, `coarse_attempts`, `fine_failures`,
`coarse_failures`, atomics) that survives failed preparations. The factory
rolls dividends once, builds the family once, and issues the six solves;
there is no mutable grid state (L2). `make_validate_fn` remains for
single-price callers and is implemented on the same oracle at High.

**Segmented B-spline probe adapter** (`bspline_adaptive.cpp:771`): scales
`ref_price`, both bracket prices, and all three `δ̂` by `scale = K/K_ref`; σ
coordinates and `resolved` are unchanged (every term of D2 scales alike, so
resolution is scale-invariant). Pinned by an asymmetric-reference-strike
test.

### D2. Resolution: surface-independent admission

A point is **resolved at τ_iv** iff all six stencil prices are finite and the
three estimated price intervals are separated in the expected order:

```
y − δ̂  >  lo + δ̂_lo        and        hi − δ̂_hi  >  y + δ̂ ,
```

and additionally `y − δ̂ > intrinsic(S, K)` (so every target price in D3 is a
valid, arbitrage-free query; for American prices `lo ≥ intrinsic` makes this
implied, but it is checked so the invariant is enforced, not assumed).

This is exactly the prior review's finite-bracket condition (§3.B) applied to
the reference. It compares three numbers with their uncertainty estimates and
claims nothing else: not monotonicity between the evaluations, not uniqueness,
not a continuous-model volatility enclosure (Ekström's monotonicity is for a
diffusion model and is not extended here to the implemented cash-dividend
jumps). Ordering violations are detected only where they show at the three
stencil points. The round trip in D3 needs no localisation interpretation.

It **replaces** both old filters (not "subsumes": with `δ̂ = 0` an
arbitrarily small positive separation passes, which is the honest answer when
two resolutions agree exactly). In the exercise region `lo = y = hi` and the
point is unresolved; where vega is small the separation falls inside the
estimates.

Unresolved points have status `ReferenceUnresolved`: no IV statistic, no
refinement bin, counted (D7), residual recorded (D3). Resolution is decided
at preparation, before any candidate exists (L1).

**Coverage policy (chosen, not derived).** The loop keeps its rule "refuse
(`ValidationFailed`) when fewer than `max(4, validation_samples/4)` holdout
points have prepared references" (`validation_samples` = requested count;
unsupported-maturity samples are skipped before counting) and applies the
same threshold to **resolved** references, for the holdout and for the final
validation set. It is an operational coverage policy (§4.11) with no spatial
coverage guarantee. On refusal the `PriceTableError` cannot carry counts (no
ABI change); the refusal fires a new USDT probe
`mango:adaptive_validation_refused(set, requested, prepared, resolved,
unsupported)` (`set` ∈ {holdout, final}) via `ivcalc_trace.h`, and the
Python/C++ error is unchanged.

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
    TargetRejected,        // validation refused a target price (expected never; see D9)
};
struct PointScore {
    PointStatus status;
    double iv_error;          // max_k |σ̂_k − σ0| iff Measured
    double price_residual;    // |S(σ0) − y| / K when S(σ0) finite, else NaN
    bool   edge_band_rescue;  // diagnostic only, see below
};
using ScoreErrorFn = std::function<PointScore(
    const SurfaceHandle& handle, const ErrorRefs& refs,
    double spot, double strike, double tau, double sigma, double rate)>;
```

`SurfaceHandle` gains `vega` (same signature as `price`), filled by every
builder from the surface's `vega()` (all four surface families have one).
Domain data and policy are captured in the scoring factory
`make_round_trip_score_fn(params, ctx, option_type, dividend info)`, not
passed per call. The scorer performs no PDE solves.

**One inversion component, no cycle.** A new low-level target
`//src/option:surface_inversion` (`surface_inversion.{hpp,cpp}`, deps:
`root_finding`, `option_spec`, `error_types`, tracing header; **no** table or
builder deps) receives from `interpolated_iv_solver.{hpp,cpp}`:
`BracketScreen`, `screen_bracket` (keeping `ObjectiveRef`, no allocation on
the `noexcept` path), and a new

```cpp
struct SurfaceInversionPolicy {
    double config_sigma_min, config_sigma_max;   // InterpolatedIVSolverConfig (0.01, 3.0)
    double published_sigma_min, published_sigma_max;
    double vega_threshold;                       // 1e-4
    bool   detect_multiple_roots;                // true
    double tolerance;                            // 1e-6
    size_t max_iter;                             // 50
};
std::expected<IVSuccess, IVError> invert_price_on_surface(
    PriceFn price, VegaFn vega, double spot, double strike, double tau,
    double rate, double target_price, OptionType type,
    const SurfaceInversionPolicy& policy) noexcept;
```

that performs, in the current order and with the current error codes:
`adaptive_bounds` (intrinsic-based 1.5/2.0/3.0 cap ∩ configured limits ∩
published limits, **with the existing fallback to the published range when
the intersection is empty**), the quartile-vega pre-check, the 17-point
screen with boundary-root return and bracket narrowing, Brent, and the
post-Brent slope check. `InterpolatedIVSolver<Surface>::solve` becomes:
`validate_query` → `is_in_bounds` at the policy's effective σ limits →
`invert_price_on_surface` (L4). Its behaviour is unchanged, verified bit-for-
bit on the existing solver fixtures, including custom `sigma_min/max`,
disjoint limits (fallback), a cap transition, and yield-curve rates.

**Boundary between validation and inversion, for the loop.** The loop's
points lie inside the published domain by construction (samples are drawn
from `ctx.sample_bounds`; τ passes `maturity_is_supported` where set), so
`is_in_bounds` cannot fail for them and is not re-run. Target validity
(`validate_iv_query`: positive price, price ≥ intrinsic) is guaranteed by D2's
third inequality for all three targets; the loop still calls the shared
function's target validation, and a rejection maps to `TargetRejected`,
which counts as a surface failure for viability (conservative) and is
asserted zero in tests. `InvalidGridConfig` cannot arise on this path.

**Acceptance uses the exact product policy.** `policy = defaults with
published = ctx.sample_bounds σ range`. No widening. A resolved reference
price whose only surface root lies outside the published domain is a
`SurfaceNoRoot`: that is what the shipped solver returns, and the metric
measures the shipped solver (review rounds 1 and 2; the user's Q5).

**Edge-band diagnostic (never used for acceptance).** After a
`SurfaceNoRoot`, the scorer re-runs the same function with the published
σ limits widened by τ_iv on each side (clipped to the fit domain
`ctx.bounds`). If that succeeds, `edge_band_rescue = true`; the status stays
`SurfaceNoRoot`. The count of rescues is reported (D7) so the query-time
edge-policy follow-up has evidence. It influences neither ranking nor
viability (L3).

**Three targets.** For a resolved point the inversion runs for
`y − δ̂, y, y + δ̂`. `Measured` iff all three succeed;
`iv_error = max_k |σ̂_k − σ0|`. Otherwise the status is the most severe
failure among the three (NonFinite > TargetRejected > NonConvergent >
Ambiguous > NoRoot > VegaTooSmall). What this measures: the shipped solver's
answer at three prices the oracle could have meant, given its estimated
uncertainty. What it does not claim: anything about prices between them
(the 17-point screen documents folds it cannot detect,
`interpolated_iv_solver.hpp:62`; Brent stops on a residual/bracket condition,
not an exact inverse; endpoint extrema would need monotone inverse behaviour
the screen does not establish). When `τ_iv` is below Brent's price tolerance
mapped through the surface's slope, the reported error carries that
resolution floor; the docs say so.

**Price residual.** `|S(σ0) − y| / K` for every point with a prepared
reference whose surface price is finite, resolved or not.

### D4. Loop consumption: fresh and holdout, ranking, viability, refinement

`SampleEval` and `FinalScore` gain `unresolved`, `surface_failures`,
`max_price_residual`, `edge_band_rescues`; `measured` counts `Measured`
only; `filtered` is replaced by `unresolved`. Both the fresh pass and the
holdout pass produce these.

**Fresh-path order.** The candidate's surface price at the sample is
evaluated **before** reference preparation, so the existing non-finite veto
applies even when preparation fails or the point is unresolved (the current
`continue` before the veto, `adaptive_refinement.cpp:384`, is reordered).

**Viability of a candidate** (replaces the `kViabilityBound` clause):

```
all fresh and holdout surface prices finite  ∧  holdout.measured > 0  ∧
fresh.surface_failures == 0  ∧  holdout.surface_failures == 0
```

`fresh_converged = fresh.measured > 0 ∧ fresh.max ≤ τ_iv ∧
fresh.surface_failures == 0`. `target_met = picked.viable ∧
picked.holdout_max ≤ τ_iv ∧ picked.fresh_converged` (existing shape).
`FinalScore::viable() = all_finite ∧ measured > 0 ∧ surface_failures == 0`;
`needs_final_retry` and `select_final_surface` keep their logic on top of it
(both already route through `viable()`; "both failing" still yields `None`;
the comparator below replaces the bare `max_error <`).

**Ordering** (exploration-base advance, retention pick, final pick), applied
**after** the viability filter where one exists: fewer **holdout**
`surface_failures` → lower holdout max → lower holdout avg → earlier
iteration. Holdout failures are the comparable progress measure because the
holdout is fixed; fresh failures veto viability and feed bins but do not
rank (fresh coordinates change per iteration). A candidate with a non-finite
holdout statistic is not an exploration base (existing rule). A candidate
whose resolved holdout points all fail (`measured == 0`, failures > 0) may
be an exploration base if no better one exists, never a returned candidate.

**Walk restart.** Restart on fewer holdout failures than the base, or on
equal failures with the existing 2 % relative improvement of the holdout max.

**Refinement bins.** `ErrorBins` gains `failure_counts[dim][bin]`, recorded
unconditionally for every surface failure (fresh and holdout).
`pick_refinement_axis` and `problematic_bins` use `bin_counts +
failure_counts`. `evaluate_holdout` returns bins for its failures (still no
solves); `Candidate::bins` merges fresh and holdout attribution. Measured
errors above τ_iv are recorded as today.

### D5. Monotonicity scan

Diagnostic only. Noise floor: per-point `max(refs.delta, refs.delta_lo,
refs.delta_hi)` floored at `1e-8 · spot`; `vega_floor` parameter removed; the
doc comment names it a reporting threshold, not monotonicity evidence.

### D6. `vega_floor` deprecation

Stays in the struct, the C ABI (offset asserts untouched), Rust, Python. C
header comment, Rust doc, Python docstring and C++ comment: deprecated,
ignored since this change, removed in #463. No validation. Benchmark mirrors
`kVegaFloor` and `kTVKThreshold` in `benchmarks/interp_iv_safety.cc` and their
explanatory text go.

### D7. Diagnostics (enumerated)

`BuildDiagnostics` gains exactly:

```cpp
size_t holdout_points_unresolved = 0;   // oracle could not resolve target (D2)
size_t surface_failures = 0;            // returned surface on the holdout (0 by D4)
size_t edge_band_rescues = 0;           // D3 diagnostic
double max_price_residual = 0.0;        // |S − V̂|/K over prepared holdout points
double reference_uncertainty_max = 0.0; // max δ̂ over the holdout (estimate)
size_t reference_solves_fine = 0;       // ReferenceSolveCounter totals
size_t reference_solves_coarse = 0;
```

`IterationStats` gains `unresolved`, `surface_failures`, `edge_band_rescues`
(C++ only; the Python converter exposes no per-iteration entries, unchanged).
The Python `build_diagnostics` property dict gains the seven keys above. Rust
exposes no diagnostics; unchanged. Segmented final gates fill the same fields
from `FinalScore`. `surface_failures` is 0 on any returned surface by D4; it
exists so `FinalScore`, `IterationStats` and the probe share one vocabulary.

**Solve accounting.** All scattered `× 3` multipliers go; `total_pde_solves`
adds `counter.fine_attempts + counter.coarse_attempts` from the one counter
each path owns (holdout, fresh, final, retry reuse included).

### D8. Calibration of `p` and the profile

Nightly `slow` test `reference_oracle_calibration_test`, using
`make_reference_grid_family(levels = 3)` (levels 0..3 = G, G½, G¼, G⅛), on
six points at High: ATM 1y put with three $0.50 dividends; the #500 trigger;
OTM 30-day put; deep-OTM 7-day put; ITM 2y put; ATM 6-month call. For each
point and each consecutive triple `(k, k+1, k+2)`: `d_a = V_{k+2} − V_{k+1}`,
`d_b = V_{k+1} − V_k`, classified as

- **oscillatory** if `d_a · d_b < 0`;
- **below resolution** if `|d_b| ≤ 2^-40 · K` (the two finest grids agree to
  double-precision noise on this scale; legitimate at exercise or deep-OTM
  points, provides no order);
- **usable** otherwise, `p_obs = ln(d_a/d_b)/ln 2`.

Assertions: no triple is oscillatory; every usable `p_obs` is finite and
positive; for points with two usable triples, the two `p_obs` agree within
0.5 (asymptotic-range stability; one triple alone is not accepted as
evidence, so at least four of the six points must have two usable triples);
`kReferenceConvergenceOrder ≤ min usable p_obs`; `|V_High − V_Ultra| ≤ δ̂_High`
at every point.

**Effective sensitivity experiments** (the American path is a single-pass
projected Thomas solve, `pde_solver.hpp:568/885`; `TRBDF2Config::tolerance`
does not affect it, so it is not varied): (a) x-domain widened by one σ√T;
(b) spatial-only refinement (level-1 space, level-0 time) and temporal-only
refinement, to separate the two error sources; (c) the solver's
`LcpKktReport` recorded per point. **Policy:** if the domain-widening shift
exceeds `δ̂` at any point, the test fails; the fix is the profile's domain
rule, not the constant. If either single-axis refinement shows a shift
larger than the joint `δ̂`, the calibration record states which axis
dominates (report only). Numbers and classifications are recorded in
`docs/MATHEMATICAL_FOUNDATIONS.md`. Per-point `p` is not measured at build
time; between calibration points `δ̂` may be optimistic or pessimistic, and
both show up in unresolved counts and residuals.

### D9. Regression coverage

- `tests/adaptive_refinement_unit_test.cc` (synthetic references and
  surfaces, no PDE): D2 with overlapping endpoint intervals (the reviewer's
  `y=10, lo=9.85, hi=10.15, δ=0.10` is unresolved), reversed ordering,
  `σ0 − τ_iv ≤ 0`, partial stencils (each bracket/coarse solve failing →
  unresolved with base present; base failing → invalid), the intrinsic
  guard, scale invariance under the probe rescaling; every `PointStatus`
  from a purpose-built surface (exact, biased, edge-shifted → `NoRoot` with
  and without `edge_band_rescue`, decreasing crossing, fold between two
  targets caught only if it falls on a screen point — documented limit, NaN
  interior, low surface vega, forced `max_iter`); three-target aggregation
  and severity order; `TargetRejected` asserted zero on valid fixtures;
  fresh-path veto with failed preparation and NaN surface price; viability
  with fresh-only and holdout-only failures; ordering with NaN statistics;
  `measured == 0` base eligibility; failure bins when τ_iv exceeds the
  bracket; restart on fewer failures; `select_final_surface` both failing /
  ties / failures-vs-max; counters surviving failures; all-unresolved
  holdout refusal; τ_iv larger than the σ domain.
- Solver tests: `invert_price_on_surface` reproduces `solve()` on the existing
  fixtures plus custom limits, disjoint limits (fallback), a cap transition,
  and a yield-curve rate (L4).
- Grid family: generated nodes are nested, odd at every level, share the
  middle index; event times are the only mandatory times; achieved step
  counts recorded; a fake oracle asserts identical fine configs across the
  three fine solves, identical coarse configs across the three coarse solves,
  and nesting between them (acceptance criterion 2).
- `tests/adaptive_grid_types_test.cc`: `vega_floor` accepted at any value.
  Python: the seven new keys present; `vega_floor` ignored.
- `tests/adaptive_surface_build_slow_test.cc`: `WideBandDividendBracket
  RemainsViable` becomes `WideBandDividendBracketRoundTrip` on the unchanged
  manual two-K_ref fixture. **The outcome is measured first** (resolution
  flag, status under the exact product policy, error if measured,
  `edge_band_rescue`), then pinned with provenance in the test. A refusal, if
  that is what the shipped solver produces, is pinned as a refusal and
  becomes evidence for the leaf follow-up; no ceiling is adjusted to force
  success. `// Bug:` line states the 788 bps artifact.
- `tests/adaptive_surface_build_integration_test.cc`,
  `tests/adaptive_grid_builder_test.cc`, `tests/iv_solver_factory_slow_test.cc`:
  direct users of the removed helpers/bound and retry-path tests updated for
  the status comparator; pins produced by the old metric's amplification
  re-measured and regenerated with the reason in the commit.
- A segmented case with an actual event gap confirms the non-finite veto and
  status accounting on the real builder.

### D10. Documentation

`docs/MATHEMATICAL_FOUNDATIONS.md` "Adaptive validation metric": stencil,
grid family, admission inequalities, the round trip and its three targets,
the edge-band diagnostic, the calibration record, §0's contract, §4.11's
inventory, citations from the research note. `docs/API_GUIDE.md`: statuses,
diagnostics, deprecation. `docs/ARCHITECTURE.md`: the shared inversion
component. `CONTEXT.md` gains *reference-resolved point*, *operational round
trip*, *surface inversion failure* (ADR 0001's split holds: C++ tests own
numerical correctness, Python tests own reachability).

### 4.11 Constant inventory (honest list)

| Constant | Kind | Where |
|---|---|---|
| `target_iv_error` | user | `AdaptiveGridParams` |
| `δ̂` | estimate (two-grid GCI) | D1 |
| `p` | calibrated constant (D8) | D1 |
| `F_s = 3` | literature convention (Roache, two-grid) | D1 |
| `kReferenceAccuracy = High` | chosen from measurement | D1 |
| `2^-40 · K` below-resolution threshold | calibration classification only | D8 |
| coverage `max(4, N/4)` on prepared and on resolved | operational policy (pre-existing rule, applied twice) | D2 |
| inversion policy: config σ 0.01/3.0, `vega_threshold 1e-4`, 17 screen points, zero-tol `1e-9·spot`, Brent `1e-6` / 50, cap 1.5/2/3 | product policy, reused unchanged | D3 |
| edge band `τ_iv` | diagnostic only | D3 |
| monotonicity-scan floor `1e-8·spot` | diagnostic floor, pre-existing | D5 |
| walk restart 2 % | pre-existing loop policy | D4 |

Nothing else numeric appears in the metric.

### Binding laws (govern every not-yet-enumerated instance)

- L1. Admission never reads the candidate surface.
- L2. Every stencil shares one fine grid from the family constructor; the
  coarse grid is its strict subsequence; nothing is re-estimated per σ.
- L3. A status is never converted into a number for acceptance or
  attribution; only `Measured` errors enter `max`/`avg`/thresholded bins;
  failures have their own counts; diagnostics (edge band, residual) decide
  nothing.
- L4. The build-time inversion is the product's inversion function with the
  product's default policy over the published σ domain; identical code,
  identical thresholds.
- L5. Every claim in code comments, diagnostics and docs about `δ̂`, `p`,
  resolved points and statuses uses empirical language ("estimate",
  "calibrated", "outcome", "on the declared sample set"); no "bound",
  "guarantee", or "for every price" appears.

## 5. Acceptance criteria

1. `make_iv_score_fn`, `make_fd_vega_refs_fn`, `compute_iv_error`, the TV/K
   constant, the vega-floor filter, and `kViabilityBound` no longer exist.
2. One factory prepares the stencil at High; a fake oracle asserts identical
   fine configs across the three fine solves, identical coarse configs across
   the three coarse solves, and exact 2:1 nesting between them.
3. D2 admission is bitwise identical across candidates (two different
   handles, same `resolved` flags).
4. `invert_price_on_surface` reproduces `InterpolatedIVSolver::solve` on the
   existing and the added solver fixtures.
5. Every `PointStatus` has a unit fixture; ordering, viability, restart, veto
   and bin rules hold on fresh-only and holdout-only failures.
6. The #500 round-trip regression pins the measured outcome with provenance.
7. `bazel test //...` green; `//benchmarks/...` and `//src/python:mango_option`
   build; Rust layout test unchanged; Python diagnostics-keys test passes.
8. Calibration test passes with D8's classification and assertions; numbers
   and sensitivity shifts are in the math doc.
9. Diagnostics report the seven D7 fields through C++ and Python; the refusal
   probe fires with its five arguments.
10. Runtime of the adaptive test targets is measured before and after on the
    actual serial reference path and stated in the PR.

## 6. Risks and assumptions

- **More refusals, by design.** Resolved sets shrink where vega is small;
  the coverage rule may refuse builds that pass today; edge-adjacent
  reference points can produce `SurfaceNoRoot` under the exact product
  policy and reject a candidate. Both are the shipped solver's behaviour
  measured honestly. `edge_band_rescues` quantifies the second for the
  follow-up. Nightly pins are re-measured, not loosened.
- **Cost.** Six High solves per point (≈ 3.75 fine-equivalents) instead of
  three default solves, on a serial reference path. Measured single-thread:
  ATM 1y with dividends 0.39 s at High. Criterion 10 measures the real
  effect.
- **`p` per profile.** D8's checks apply to the calibration set. Elsewhere
  `δ̂` is an estimate; L5 keeps the language honest.
- **Assumption:** `screen_bracket`, the pre-check and Brent are pure functions
  of the callables and the policy, so extraction cannot change the product
  path (criterion 4).
- **Assumption:** every surface handed to the loop exposes `vega()`
  (`BSplinePriceTable`, `BSplineMultiKRefInner`, `ChebyshevSurface`,
  `ChebyshevMultiKRefSurface` do).

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
record (partial adoption of the review's price-validation recommendation).

**Q6. Resolved surface failures.** Options: universal; conditional domain;
budgeted fraction. **Chosen: universal** on the declared fresh and holdout
sets. Why: a max-error statement cannot ignore a resolved counterexample;
failures feed refinement; the alternatives add a contract or a constant.

**Q7. `vega_floor`.** Options: keep, ignore, deprecate; remove now.
**Chosen: keep and deprecate.** Why: no ABI churn in a metric change.

**Q8. The 0.20 viability constant.** Options: drop; keep. **Chosen: drop.**
Why: measured errors are genuine σ distances and resolved failures already
reject.

**Design choices made without a question, with review verdicts:**

- One shared fine grid per stencil, strict-subsequence coarse grid (L2) —
  agreed (rounds 1–2); removes re-gridding differences, does not cancel bias.
- Inversion bracket — rev 2 proposed the product policy plus a τ_iv edge
  band; both rounds disagreed (it measures a modified solver, moves screen
  nodes and quartile probes, and differs by backend). **Rev 3 adopts the
  exact product bracket for acceptance; the edge band is a reported
  diagnostic only.** Consequence flagged for the go/no-go: edge-adjacent
  refusals are real and may reject candidates.
- Minimum-resolved threshold `max(4, N/4)` — agreed as an explicitly chosen
  coverage policy.
- Failure attribution — unconditional failure counts, no pseudo-error
  (round 1 correction).
- `VegaTooSmall` pre-check — replicated via `SurfaceHandle::vega` and the
  shared inversion (round 1 correction).
- `F_s = 3` with a calibrated profile-level `p` — agreed only as an
  empirical estimator; rev 3 adds order-stability, classification, and
  effective sensitivity checks (round 2 correction; the algebraic-tolerance
  experiment was a no-op on the projected solver and is replaced).
- Three targets `y ± δ̂` — agreed as three-price testing; the interval claim
  is withdrawn (round 1), endpoint-extrema reasoning not relied on (round 2).
- Monotonicity language in D2 — removed; D2 is a three-interval separation
  statement only (round 2 correction).

**Design approval:** the user approved the five-section design on
2026-09-19 with "Yes, proceed". Revisions 2 and 3 change no user decision;
they tighten the mathematics and product fidelity per review rounds 1–2. The
edge-band change (rev 3) narrows what the user approved as "the product's
own Brent" to exactly that.
