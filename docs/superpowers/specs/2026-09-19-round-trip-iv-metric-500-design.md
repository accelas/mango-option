# Round-trip IV-error metric for adaptive validation (#500 remainder)

Issue: https://github.com/accelas/mango-option/issues/500 (the residual after
#504/#505). Research: `docs/research/2026-09-19-iv-inversion-conditioning.md`
(primary-source synthesis) and
`docs/research/2026-09-19-iv-inversion-conditioning-codex-review.md` (Codex
mathematical review of that synthesis; this design adopts its recommendation).

Spec revision 1, 2026-09-19.

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
- The reference price is flat at 14.1249 for σ ≤ 0.12 (the holder waits for
  the ex-dividend drop, then exercises) and bends up after. The point sits on
  that shoulder; 788 bps is a $0.017 miss divided by a vega that is about to
  vanish. The surface price is below the σ→0 plateau, so no reference σ
  reproduces it and the linearised number is not an IV error at all.
- The same σ-kink exists everywhere deep ITM. Before the dividend, time value
  is exactly zero and the TV/K filter hides the point; just after, the
  deterministic dividend gain counts as time value and the filter admits it.
  The dividend's only role is to defeat the filter.
- The validation oracle (`make_validate_fn`) calls `solve_american_option`,
  which auto-estimates a grid with default `GridAccuracyParams` (`tol = 1e-2`,
  documented as ~1e-3 price accuracy). Measured against Ultra: 3.3e-3 at an
  ATM 1y put with three dividends, 4.6e-5 at the trigger point. Through a
  0.22 vega, 1e-3 of oracle noise is 45 bps by itself. The table's own PDE
  samples run at Ultra, so the reference is noisier than the thing it grades.

Both admission constants, the vega floor, and the 0.20 bound are unanchored.
The user's requirement for this work: the replacement must rest on sound
mathematics, and the choice was made after research and an independent
mathematical review.

What the mathematics says (both documents agree on these points):

1. `|ΔV| / vega` is a first-order inverse image of a price discrepancy. It is
   correct where the price is differentiable in σ with positive slope and the
   displacement stays inside the region where the slope is representative.
   At the trigger it extrapolates 56 bump widths across a plateau.
2. A derived quantity `σ = f(V)` can be certified to tolerance τ only where
   the input error times the condition number is below τ (Higham §1.6). With
   a numerical oracle, the input error is the oracle's own price error, which
   must be estimated, not assumed.
3. Local vega at one point, a bump-pair vega, and a two-grid Richardson
   number are diagnostics, not bounds. A finite bracket in σ is the
   derivative-free way to state "σ0 is distinguishable from σ0 ± τ".
4. The quantity the product delivers is the surface's *own* inverse of a
   price, not the reference's inverse of the surface's price. The two agree
   only to first order; an oscillating surface can differ.
5. Admission of a point to the statistics must not depend on the candidate
   surface, or a worse candidate improves its score by making hard points
   disappear.
6. A scalar-or-nothing score cannot honestly encode "error", "no inverse",
   "ambiguous inverse", and "reference cannot resolve". Statuses are needed.

## 2. Goals

- Replace the IV-error metric with one whose every constant is either the
  user's tolerance, a measured quantity, or a literature convention with a
  stated role. No time-value threshold, no vega floor, no 0.20 bound.
- Score what the product does: invert the reference price on the candidate
  surface with the product's own algorithm.
- Make the reference's accuracy explicit (declared profile, per-point
  uncertainty estimate) and make admission surface-independent.
- Report statuses and a price residual, so a non-invertible point is named
  as such instead of being converted into a bps number or silently dropped.
- Keep the loop contract (`run_refinement`, candidate retention, final
  validation) and the public builder APIs; no C ABI layout change.

## 3. Non-goals

- The σ-axis leaf resolution of the segmented Chebyshev builder (the real
  price error at the trigger). Filed as a follow-up once this metric reports
  it honestly.
- Aligning the query-time screens in `InterpolatedIVSolver` (`vega_threshold
  = 1e-4`, bracket = published σ domain) with the build-time criterion.
  Follow-up.
- Removing `AdaptiveGridParams::vega_floor` from the C ABI, Rust, and Python
  surfaces. Folded into the pending ABI revision (#463).
- A price-space acceptance gate. The price residual is recorded, not gated.
- Statistical coverage guarantees for the Latin-hypercube sample. The
  certificate remains "sampled performance on the declared sample set".
- Changing how samples are drawn, how axes are picked, or the CC/knot
  headroom rules.

## 4. Design

Notation per validation point `p = (S, K, τ, σ0, r)` with the builder's
dividend schedule: `V̂(σ)` is the reference price at full resolution,
`V̂½(σ)` at half resolution, `y = V̂(σ0)`, `τ_iv = params.target_iv_error`,
`S(σ) = handle.price(S, K, τ, σ, r)` the candidate surface.

### D1. Reference oracle: declared profile, shared stencil grid

`make_validate_fn` builds its solver with an explicit
`PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)}` instead of the
convenience wrapper's defaults. The profile is a named constant
`kReferenceAccuracy` in `adaptive_metrics.cpp` with the measurement that
chose it (High matches Ultra to ~4e-6 at half the cost; default was off by
3.3e-3 at ATM 1y).

The reference for one point is a **stencil of four solves on one grid**:

| Solve | σ | Grid |
|---|---|---|
| `y` | σ0 | G |
| `lo` | max(σ0 − τ_iv, σ_floor) | G |
| `hi` | σ0 + τ_iv | G |
| `y½` | σ0 | G½ |

`G` is `estimate_pde_grid` at High evaluated **once** for the widest member of
the stencil (σ0 + τ_iv; the x-domain scales with σ) and passed as an explicit
`PDEGridConfig` to all three full-resolution solves, so the bracket
differences are not contaminated by the estimator re-gridding per σ (the
Codex review's "grid-selection changes with volatility"). `G½` is the same
grid family (same x-domain, same sinh α, dividend event times retained) with
`(n_x − 1)/2 + 1` spatial points (kept odd) and `⌈n_t/2⌉` time steps.
`σ_floor` is the solver's own minimum volatility (1e-4); if σ0 ≤ σ_floor +
τ_iv the `lo` side is one-sided and the point is unresolved unless the `hi`
side alone resolves it (D2 treats a collapsed side as failing that side).

**Uncertainty estimate.** `δ̂ = F_s · |y − y½| / (2^p − 1)` (Roache's
generalised Richardson estimator, `F_s = 3` for a two-grid comparison). `p`
is `kReferenceConvergenceOrder`, a constant in `adaptive_metrics.cpp`
**measured** by the calibration test in D8 (three resolutions on a fixed
point set) and set to the lower end of the measured range; it is never
assumed to be 2 (Forsyth & Vetzal measure ≈1.5 for American puts with
constant steps). The spec calls `δ̂` an *estimate*; nothing in this design
treats it as a bound, and the diagnostics say "estimated".

`ErrorRefs` becomes:

```cpp
struct ErrorRefs {
    double ref_price;        // y
    double bracket_lo_price; // V̂(σ_lo)
    double bracket_hi_price; // V̂(σ_hi)
    double sigma_lo, sigma_hi;
    double delta;            // δ̂, estimated reference uncertainty at y
    bool resolved;           // D2, decided here, surface-independent
};
```

`PrepareRefsFn`'s signature is unchanged; `make_fd_vega_refs_fn` is replaced
by `make_bracket_refs_fn(params, validate_fn, validate_half_fn)`. The half
resolution oracle is `make_validate_fn`'s sibling built from the same
parameters (`make_half_resolution_validate_fn`), so the two cannot drift in
dividend handling. `vega` leaves `ErrorRefs`; nothing downstream needs it.

### D2. Resolution: surface-independent admission

A point is **resolved at τ_iv** iff every stencil price is finite and

```
y − lo > δ̂   and   hi − y > δ̂ .
```

This is the finite-bracket condition of the Codex review (§3.B) applied to
the reference itself: the oracle can tell σ0 from σ0 ± τ_iv by more than its
own estimated uncertainty. It subsumes both old filters: in the exercise
region `lo = y = hi` exactly; where vega is small the bracket collapses into
the noise. It uses no vega, no intrinsic value, and no chosen constant beyond
τ_iv and Roache's `F_s`.

Unresolved points have status `ReferenceUnresolved`. They enter no IV
statistic and no refinement bin, are counted (D7), and still record a price
residual (D3). Resolution is decided when references are prepared, before
any candidate exists, so it is identical for every candidate.

**Minimum resolved set.** Today the loop refuses (`ValidationFailed`) when
fewer than `max(4, validation_samples/4)` holdout points have usable
references. The same threshold now applies to *resolved* holdout points, and
the final validation set likewise. A build that cannot resolve τ_iv on a
quarter of its sample domain cannot certify τ_iv; refusing early is the
honest outcome and the diagnostics (D7) say how many points were unresolved.

### D3. Score: the query-time round trip

The score seam changes from `std::optional<double>` to

```cpp
enum class PointStatus {
    Measured,              // iv_error engaged
    ReferenceUnresolved,   // D2: excluded, not evidence
    SurfaceNoRoot,         // y outside the surface's price range on the bracket
    SurfaceAmbiguous,      // multi-root screen refused the bracket
    SurfaceNonFinite,      // NaN/inf price during screen or inversion
};
struct PointScore {
    PointStatus status;
    double iv_error;        // valid iff Measured
    double price_residual;  // |S(σ0) − y| / K, always set when S(σ0) finite
};
using ScoreErrorFn = std::function<PointScore(
    const SurfaceHandle& handle, const ErrorRefs& refs,
    double spot, double strike, double tau, double sigma, double rate)>;
```

The scorer receives the handle rather than one interpolated price because
the round trip evaluates the surface along σ. It performs no PDE solves; all
reference data is cached in `refs`.

**Round trip.** With `f(σ) = S(σ) − y_target` on the bracket

```
[σ_min − τ_iv, σ_max + τ_iv] ∩ [fit σ_min, fit σ_max]
```

(`σ_min, σ_max` = `ctx.sample_bounds`, the published domain; the fit domain
is `ctx.bounds`), run the product's inversion: `detail::screen_bracket(f,
lo, hi, spot, kInversionTolerance)` followed by `find_root` (Brent) with the
`InterpolatedIVSolverConfig` defaults (`tolerance = 1e-6`, `max_iter = 50`).
`screen_bracket` and the Brent call move to a header both the solver and the
loop include, so there is one inversion algorithm. Outcomes map to statuses:
refusal → `SurfaceAmbiguous`; no sign change → `SurfaceNoRoot`; non-finite
objective → `SurfaceNonFinite`; root σ̂ → `Measured`.

The bracket is widened by exactly τ_iv beyond the published domain because a
root within τ_iv of the edge is inside the tolerance band of a point at the
edge; anything farther is a genuine failure of the surface to represent the
reference price on its domain. Widening to the whole fit domain was rejected:
Chebyshev headroom extends σ down to 0.01, where the fit is support, not a
price surface, and its wiggles would trigger spurious `SurfaceAmbiguous`
refusals the product never sees.

**Uncertainty propagation.** For a resolved point the round trip runs for
three targets `y − δ̂, y, y + δ̂`. The reported error is
`max |σ̂_k − σ0|` over the three; if any of the three fails, the point takes
that failure status (the most severe of the three, ordering NonFinite >
Ambiguous > NoRoot). The certificate is therefore "for every price the
oracle could have meant, the surface's inverse lies within `iv_error` of σ0".
These are surface evaluations only (≈3 × 17 screen points + Brent).

**Price residual.** `|S(σ0) − y| / K` for every point whose surface price is
finite, resolved or not, and independent of any status.

The product's `VegaTooSmall` pre-check (surface vega at three quartiles) is
**not** replicated: `SurfaceHandle` exposes price only, and the check is an
operational policy of the query path, not part of the inverse. This is
recorded as the query-time alignment follow-up (§3).

### D4. Loop consumption: ranking, viability, refinement

`SampleEval` and `FinalScore` gain `unresolved`, `surface_failures`, and
`max_price_residual`; `measured` counts `Measured` only; `filtered` is
replaced by `unresolved`. Ordering is lexicographic everywhere a candidate
is compared (exploration base advancement at `adaptive_refinement.cpp`
~980, retention pick ~1052, `pick_final` for the B-spline retry ~684):

```
fewer surface_failures  →  lower holdout max  →  lower holdout avg
```

`viable()` becomes `all_finite && measured > 0 && surface_failures == 0`.
`kViabilityBound` is deleted. `target_met` is `viable && max ≤ τ_iv`. Fresh
convergence (`fresh_converged`) additionally requires zero fresh failures.

**Refinement bins.** Measured errors above τ_iv are recorded as today.
Surface failures are also recorded, with `iv_error` set to the inversion
bracket width (the largest σ distance the bracket admits). `ErrorBins` bin
counts drive axis selection; `dim_error_mass` is diagnostic only
(`worst_dimension`), so this weight affects no decision except attribution.
A failure therefore pulls refinement toward its region instead of vanishing.

**Universal guarantee.** A candidate with a surface failure at a resolved
holdout point is never returned as viable. This is the user's decision (Q6):
a maximum-error certificate cannot ignore a resolved counterexample. The
expected consequence is that some wide-band builds that pass today will
refuse until the leaf work lands; they were passing on a metric that hid the
defect.

### D5. Monotonicity scan

`scan_monotonicity` keeps its role (diagnostics only) and replaces the noise
floor `target_iv_error · vega_floor` with the per-point `refs.delta`
(floored at `1e-8 · spot` as today). Its `vega_floor` parameter is removed.

### D6. `vega_floor` deprecation

`AdaptiveGridParams::vega_floor` stays in the struct, the C ABI (offset 56
asserts untouched), `crates/mango-option`, and the Python binding. Its
comment, the Python docstring, and the Rust doc say it is deprecated and
ignored since this change and will be removed in the ABI revision (#463).
The loop no longer validates it (the `vega_floor == 0` / non-finite rejection
tests are removed); `benchmarks/interp_iv_safety.cc`'s `kVegaFloor` mirror
goes.

### D7. Diagnostics

`BuildDiagnostics` gains:

```cpp
size_t holdout_points_unresolved = 0;  // reference could not resolve target
size_t surface_failures = 0;           // returned surface, resolved holdout
double max_price_residual = 0.0;       // |S − V̂|/K over all referenced points
double reference_uncertainty_max = 0.0; // max δ̂ over the holdout (estimated)
```

`IterationStats` gains `surface_failures`. `holdout_points_measured` keeps
its meaning (resolved points that produced an error). The Python `build_
diagnostics()` dict gains the four keys; Rust exposes nothing new (it does
not expose diagnostics today). Segmented final validation (`Chebyshev
SegmentedBuilder::build_adaptive`, `bspline_adaptive.cpp` final gate) fills
the same fields from `FinalScore`.

Solve accounting: `FinalValidationSet::ref_attempts` and the loop's
`pde_solves_validation` multiplier become 4 per point (three full plus one
half-resolution solve, counted as one solve each; comments say the half
solve is ~¼ the cost).

### D8. Calibration of `p` and the oracle pin

A nightly `slow` test (`reference_oracle_calibration_test`, per the CI test
philosophy) solves a fixed six-point set at High on three controlled
resolutions (G, G½, G¼): ATM 1y put with three $0.50 dividends, the #500
trigger point, an OTM 30-day put, a deep-OTM 7-day put, an ITM 2y put, and
an ATM 6-month call. It reports `p_obs = ln[(V¼ − V½)/(V½ − V)]/ln 2` per
point and asserts (i) every `p_obs` is finite with consistent signs, (ii)
`kReferenceConvergenceOrder ≤ min p_obs`, and (iii) `|V_High − V_Ultra| ≤
δ̂` at every point (the profile choice is consistent with its own
uncertainty estimate). The measured values are recorded in
`docs/MATHEMATICAL_FOUNDATIONS.md`.

### D9. Regression coverage

- `tests/adaptive_refinement_unit_test.cc`: synthetic references (a monotone
  price curve, a plateau, a collapsed bracket) prove D2 admits/excludes
  independently of any surface; synthetic surfaces (exact, biased, edge-
  shifted within τ_iv, edge-shifted beyond τ_iv, non-monotone, NaN) produce
  each `PointStatus`; a candidate with one failure loses to one with a worse
  max; `viable()` without the bound; bins receive failures.
- `tests/adaptive_grid_types_test.cc`: `vega_floor` accepted at any value.
- `tests/adaptive_surface_build_slow_test.cc`: `WideBandDividendBracket
  RemainsViable` becomes `WideBandDividendBracketRoundTrip`: same two-K_ref
  surface and point; assert the status is `Measured` or
  `ReferenceUnresolved`, never a failure, and if measured the error is below
  200 bps. Expected: ≈70 bps (the surface's σ-profile crosses 14.1263 near
  σ ≈ 0.147). `// Bug:` line states the 788 bps artifact.
- Existing pins that assert `kViabilityBound`, `filtered`, `vega_floor`, or
  a specific `holdout_points_measured` are updated; any that pinned an
  error produced by the old metric's amplification are re-measured under the
  new one and their numbers regenerated with the reason in the commit.

### D10. Documentation

`docs/MATHEMATICAL_FOUNDATIONS.md` gets a section "Adaptive validation
metric" stating the criterion (stencil, resolution test, round trip,
uncertainty propagation) with the citations from the research note, and
listing which constants are user (τ_iv), measured (δ̂, p), or convention
(F_s). `docs/API_GUIDE.md` documents the new diagnostics and the deprecation.
`docs/ARCHITECTURE.md`'s adaptive-refinement paragraph names the statuses.

### Binding laws (govern every not-yet-enumerated instance)

- L1. Admission never reads the candidate surface.
- L2. Every reference stencil shares one grid; the half-resolution grid is
  derived from it, never re-estimated.
- L3. A status is never converted into a number for acceptance; only
  `Measured` errors enter `max`/`avg`, and `viable` is false with any
  surface failure.
- L4. The build-time inversion is the product's inversion code, not a copy.

## 5. Acceptance criteria

1. `make_iv_score_fn`, `make_fd_vega_refs_fn`, `compute_iv_error`, the TV/K
   constant, the vega-floor filter, and `kViabilityBound` no longer exist.
2. `make_validate_fn` solves at `kReferenceAccuracy = High` on an explicit
   grid; a test pins the profile and that the stencil shares one grid.
3. D2 resolution is computed in `PrepareRefsFn` and is bitwise identical
   across candidates (unit test with two different handles).
4. Each `PointStatus` is produced by the unit fixtures in D9, and the
   ordering/viability rules hold.
5. The #500 round-trip regression passes with the expected status.
6. `bazel test //...` green; `bazel build //benchmarks/...` and
   `//src/python:mango_option` build; Rust layout test unchanged.
7. The calibration test measures `p` and the constant is ≤ the measured
   minimum; the numbers appear in the math doc.
8. Diagnostics report unresolved counts, failures, max residual, and max δ̂
   through C++ and Python.
9. Full-suite runtime does not exceed the baseline by more than the
   validation cost model predicts (≈ +40% on adaptive-build tests; measured
   and stated in the PR).

## 6. Risks and assumptions

- **More refusals.** At τ_iv = 5 bps the resolved set shrinks where vega is
  small (short-dated deep OTM/ITM). Builds may hit the minimum-resolved
  threshold. This is the criterion working; the diagnostics say why. The
  nightly viability pins are re-measured, not loosened.
- **Half-resolution grid with dividends.** The plan must verify that an
  explicit `PDEGridConfig` retains dividend event times and that halving
  `n_time` does not drop a mandatory time. If the solver cannot honour a
  halved grid for a contract, the point is `ReferenceUnresolved`, never a
  crash.
- **`p` per profile, not per point.** Codex notes a profile-level order does
  not establish pointwise order near free boundaries. `F_s = 3` is the
  convention that absorbs this, and δ̂ is labelled an estimate. If the
  calibration test shows `p_obs` varying widely, the constant takes the
  minimum and the doc says so.
- **Edge band τ_iv.** A root just beyond `σ_min − τ_iv` is a failure while
  one just inside is measured. Any finite bracket has an edge; τ_iv is the
  only tolerance the user declared, so it is the only defensible width.
- **Cost.** Four High solves per point instead of three default solves.
  Measured single-thread: ATM 1y with dividends 0.39 s (High) vs ≈0 at
  default; a 64-point set ≈ 100 s single-thread, ≈ 7 s at 16 threads. The
  segmented paths already spend minutes on PDE sampling.
- **Assumption:** `detail::screen_bracket` and `find_root` are pure functions
  of an objective and can be shared between solver and loop without
  behaviour change.

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
the time: three gates, local δ_ref.** **Superseded** by Q5 after the Codex
review showed the three gates are diagnostics, not a certificate, that
bump-pair derivatives cannot even establish positive vega under the
reference's own error, and that the secant "true error" is the reference's
inverse of the surface's price, not what the product computes.

**Q4. Oracle accuracy profile.** Options: High; Ultra; Medium. **Chosen:
High.** Why: matches Ultra to ~4e-6 at half the cost; default was off by
3.3e-3 at ATM 1y; Medium's 3e-5 would push more points to unresolved at a
5 bps target.

**Q5. Principle defining the IV-error metric.** Options: query-time round
trip with surface-independent bracket admission; reference bracket
certificate (pass/fail/unresolved, ladder for a scalar); price-space
certification only. **Chosen: query-time round trip.** Why: it measures the
operation users consume with the product's own algorithm, needs no
derivative, no linearisation and no vega; admission via the reference's own
τ-bracket is derivative-free and surface-independent; the price residual is
kept as the forward-space record.

**Q6. Resolved surface failures.** Options: universal (any resolved failure
rejects); conditional domain (publish unsupported regions); budgeted failure
fraction. **Chosen: universal.** Why: a max-error guarantee cannot ignore a
resolved counterexample; failures feed refinement so the loop can repair
them; the other two add a new contract or a new constant.

**Q7. `vega_floor`.** Options: keep, ignore, deprecate; remove now.
**Chosen: keep and deprecate.** Why: no ABI or binding churn in a metric
change; removal belongs to #463.

**Q8. The 0.20 viability constant.** Options: drop; keep as sanity bound.
**Chosen: drop.** Why: measured errors are genuine σ distances and resolved
failures already reject; the constant was the last unanchored number in the
gate.

**Design choices made without a question (reviewer: validate these too):**
the stencil shares one grid (L2); the inversion bracket is the published
domain widened by τ_iv, not the fit domain (D3); the minimum-resolved
threshold reuses the existing `max(4, N/4)` rule (D2); failures enter bins
with the bracket width as attribution weight (D4); the product's
`VegaTooSmall` pre-check is not replicated (D3, follow-up); `F_s = 3` and a
profile-level measured `p` (D1, D8); uncertainty propagated through the
surface inverse at `y ± δ̂` (D3).

**Design approval:** the user approved the five-section design on
2026-09-19 with "Yes, proceed".
