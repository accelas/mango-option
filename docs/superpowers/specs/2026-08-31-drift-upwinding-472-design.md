# Il'in-Fitted Drift Discretization for M-Matrix Guarantee (#472)

**Date:** 2026-08-31
**Issue:** #472 — Drift upwinding: enforce monotone (M-matrix) discretization
**Parent work:** #439/#455 (PR #475, merged d75ceb1e) — sweep orientation fix +
full-KKT validator. This issue closes the remaining hole in one-pass
Brennan-Schwartz exactness: the M-matrix property of the stage Jacobian.

## Problem

`solve_thomas_projected2` is exact in one pass only when the stage matrix
`A = I − w·∂L/∂u` is an M-matrix (positive diagonal, non-positive
off-diagonals, diagonally dominant) and the active set is one interval
touching the sweep's starting side. Orientation was fixed in #475. The
M-matrix half is not guaranteed: the centered drift stencil flips an
off-diagonal sign at high cell Péclet number.

With the library's actual non-uniform stencils (weighted central first
derivative with `w_left = dx_r/(dx_l+dx_r)`, `w_right = dx_l/(dx_l+dx_r)`;
see `SpatialOperator::assemble_jacobian` and `CenteredDifference`), the
off-diagonals of the spatial operator `L(u) = a·u'' + b·u' + c·u`
(`a = σ²/2`, `b = r(t) − d − σ²/2`, `c = −r(t)`) reduce to:

```
lower_i = (a − b·dx_right/2) / (dx_left · dx_avg)
upper_i = (a + b·dx_left/2)  / (dx_right · dx_avg)      dx_avg = (dx_left+dx_right)/2
```

So the sign condition is **per-side**: for `b > 0` the lower entry binds
(`a ≥ b·dx_right/2`); for `b < 0` the upper entry binds
(`a ≥ |b|·dx_left/2`). Since both derivative stencils annihilate constants,
`diag_L = c − lower_i − upper_i`, so once the off-diagonals of `L` are ≥ 0
the stage matrix `A = I − w·L` is a **Z-matrix** (non-positive
off-diagonals) with row sum `1 + w·r(t)`. If additionally
`1 + w·r(t) > 0` (a **sufficient** condition, not a characterization —
a Z-matrix can be a nonsingular M-matrix without strict row dominance),
every non-identity row is strictly diagonally dominant with positive
diagonal, hence `A` is a nonsingular M-matrix. The condition is automatic
for `r ≥ 0` and fails only when `r(t) ≤ −1/w ≈ −2/Δt` (thousands of
percent negative for practical step sizes; unreachable in practice, but
the public API accepts any finite rate, so when it fails the proof simply
no longer applies — the matrix may or may not still be an M-matrix).
Off-diagonal non-negativity of `L` is the property this issue enforces;
the dominance condition is documented and covered by a direct matrix-row
test (note: `validate_lcp_kkt` checks the *solution*, not matrix
structure — it detects resulting solution defects, not the failed
structural condition itself).

Concrete failure (from the issue, confirmed against the code): σ = 1%,
h = 0.1, r = 5% gives `a = 5e-5`, `b ≈ 0.05`; discrete diffusion
`a/h² = 0.005` vs drift `b/(2h) = 0.25` → `lower_i < 0`. Cell Péclet
`ρ = |b|·h/(2a) = 50`. This sits inside supported low-vol IV probing.

## Decision: always-on Il'in exponential fitting

Replace the diffusion coefficient per node with a fitted value; keep the
weighted-centered drift stencil:

```
ρ_i(t)  = |b(t)| · h_binding(i) / (2a)
h_binding(i) = dx_right(i) if b(t) > 0, dx_left(i) if b(t) < 0   (either if b = 0)
a_f(i,t) = a · ρ_i · coth(ρ_i)
```

**Sign guarantee (the theorem this issue owns).** `ρ·coth(ρ) ≥ max(1, ρ)`
for all ρ ≥ 0 (requires `a > 0`, which public pricing enforces via positive
volatility; for `b = 0` the helper returns `a` before any division, which
also covers `LaplacianPDE`). Hence `a_f ≥ a` (never less diffusion than
today) and `a_f ≥ a·ρ_i = |b|·h_binding/2`, which is exactly the
binding-side condition, so the binding off-diagonal is ≥ 0 for **all**
σ, h, b (it decays to 0⁺ exponentially as ρ → ∞, and a zero off-diagonal is
admissible for the Z-/M-matrix structure). The non-binding off-diagonal is
`(a_f + |b|·dx/2)/(...) > 0` trivially. Combined with the row-sum condition
`1 + w·r(t) > 0` above, the stage matrix is an M-matrix; the fitting makes
the off-diagonal half unconditional, which is the half the drift can break.

**Accuracy.** `ρ·coth(ρ) = 1 + ρ²/3 − ρ⁴/45 + …`, so the scheme adds
numerical diffusion `a·O(ρ²)` — second-order consistent as `h → 0` for a
refining, suitably regular grid with fixed `a > 0`. On uniform grids this
is the classical Il'in/Allen–Southwell fitted scheme (uniformly convergent
for convection-dominated problems); our per-node binding-side choice is a
monotone **non-uniform extension** of it — we claim local second-order
consistency and the sign guarantee, not the full uniform-convergence
theorem, on arbitrary non-uniform grids. In the convection-dominated limit
`a_f − a → |b|h/2`, i.e. it degrades gracefully to upwind-like first-order
behavior rather than oscillating. On well-resolved grids ρ ~ 1e-2–1e-3, so
coefficients move by ~1e-4–1e-6 relative: inside every existing accuracy
tolerance, but **not bit-identical** to current output (accepted; see
Decisions; one absolute pin must move — see Testing item 3).

**Smoothness.** `ρ·coth(ρ)` is analytic and even in ρ, so a_f is smooth in
σ and h at fixed drift sign. At a `b = 0` crossing on a *non-uniform* cell
the binding side switches between `dx_right` and `dx_left`, so `a_f` is C¹
in b (the correction starts at `b²·h_binding²/(12a)`) but not C² when the
spacings differ. C¹ is sufficient for IV Newton and the FD-vega adaptive
metric; we keep binding-side spacing (a sign-independent
`max(dx_left, dx_right)` would buy analyticity at the cost of extra
diffusion everywhere). No threshold branch exists away from `b = 0`.

**Floating-point contract for the helper** (binding law for the
implementation):

```
z = |b(t)| · h_binding(i) / 2
if z == 0:            a_f = a                     (exact; covers b = 0, LaplacianPDE)
ρ = z / a
if ρ < 1e-4:          a_f = a · (1 + ρ²/3)        (series; rel. error ≤ ρ⁴/45 < 1e-17)
else:                 a_f = z / tanh(ρ)           (tanh saturates to 1; no cosh/sinh
                                                   overflow; a_f → z as ρ → ∞)
a_f = max(a_f, a, z)                              (so a_f − z ≥ 0 and a_f ≥ a hold
                                                   exactly in floating point)
```

The helper exposes **both `a_f` and `z`** (and the drift sign), because the
clamp alone does not make the *assembled* off-diagonal non-negative:
forming it as two independently rounded terms
(`a_f/(dx_l·dx_avg) − b·dx_r/(dx_l·(dx_l+dx_r))`) can round to a tiny
negative even when `a_f == z`. **Assembly is therefore sign-preserving by
construction** (binding law):

```
lower_i = (a_f − b·dx_right/2) / (dx_left · dx_avg)
upper_i = (a_f + b·dx_left/2)  / (dx_right · dx_avg)
diag_i  = c − lower_i − upper_i        (row-sum identity preserved exactly)
```

where the binding-side numerator is computed literally as `a_f − z` (the
same `z` the helper produced; ≥ 0 exactly by the clamp) and the
non-binding numerator as `a_f + |b|·dx_other/2` (a sum of non-negatives).
This replaces the current separate `d2_coeff + d1_coeff` accumulation in
`assemble_jacobian` — algebraically identical (the existing
`d1_coeff_im1 = −b·dx_r/(dx_l·(dx_l+dx_r))` is exactly
`−(b·dx_r/2)/(dx_l·dx_avg)`), but with the sign guarantee carried through
floating point. The high-ρ test asserts the *assembled* binding entry is
≥ 0, including when `tanh(ρ) == 1`.

`a ≤ 0` is outside the contract when `b ≠ 0` (public pricing enforces
σ > 0; the `z == 0` early-out is what keeps `LaplacianPDE` and other
zero-drift operators away from the division).

## Architecture: SpatialOperator-local shared helper

The fitting couples PDE coefficients `(a, b(t))` with local grid spacing —
it is a property of the *discretization*, not the PDE. It lives in
`SpatialOperator` (`src/pde/internal/spatial_operator.hpp`), which already
knows both and already contains the Jacobian's stencil algebra.

Consistency requirement (functional, not stylistic): the projected LCP path
solves `A·u = rhs` where `A` comes from `assemble_jacobian`, while the
TR-BDF2 explicit RHS and the Newton residual come from
`apply_interior`/`apply`. `validate_lcp_kkt` checks the assembled system
against the solution the sweep produced from that RHS. If the two paths
discretize `L` differently, the validator flags our own fix.

Changes, all inside `SpatialOperator`, gated on
`HasJacobianCoefficients<PDE>`:

1. **Private helper** `fitted_second_coeff(t, i)` (name at implementer's
   discretion) computing `a_f(i,t)` from `a`, `b(t)`, and the spacing
   arrays, with the small-ρ series branch.
2. **`assemble_jacobian`** uses `a_f(i,t)` in place of `a` in the three
   second-derivative coefficients. Drift and reaction terms unchanged.
3. **`apply_interior`** gains a coefficient-combine path for
   `HasJacobianCoefficients` PDEs:
   `Lu[i] = a_f(i,t)·d2u[i] + b(t)·du[i] − r(t)·u[i]`,
   reusing the stencil's `d2u`/`du` arrays and bypassing `pde_::operator()`.
   (The stencil's weighted first derivative already matches the Jacobian's
   `d1_coeff` algebra exactly — verified.) PDEs without coefficients keep
   the existing generic `pde_(t, d2u, du, u)` path unchanged.
   **Contract strengthening (documented on the concept):** once a PDE
   satisfies `HasJacobianCoefficients`, its coefficient methods become the
   *authoritative definition* of the interior operator — `operator()` is
   no longer consulted on this path, and fitted use additionally requires
   `a > 0` whenever `b ≠ 0`. True for both current models
   (`BlackScholesPDE::operator()` is exactly `a·d2u + b(t)·du − r(t)·u`;
   `LaplacianPDE` has `b = r = 0`). A test-only PDE whose `operator()` and
   coefficient expansion agree guards this dispatch contract.
4. **Boundary rows stay unfitted.** The ghost-eliminated rows are
   `diag = c − 2a/h²`, `offdiag = +2a/h²` — the off-diagonal already has
   the required Z-matrix sign for any `b` (drift enters only the affine
   term), and the rows participate in strict dominance whenever
   `1 + w·r(t) > 0`. `eval_boundary_row = jacobian·u + affine` holds
   row-wise by construction, so per-row residual/Jacobian consistency is
   unaffected. The asymmetry (interior rows fitted, boundary rows not) is
   a consistent O(ρ²) perturbation and is documented in code and in
   MATHEMATICAL_FOUNDATIONS.

No changes to `BlackScholesPDE`, `LaplacianPDE`, `CenteredDifference`,
`GridSpacing`, the `HasJacobianCoefficients`/`HasBoundaryRows` concepts, or
`PDESolver`.

## Testing

1. **Regression test** (per acceptance criteria, in the #475 regression
   style): full American solve via the public API
   (`AmericanOptionSolver::create` with a custom `PDEGridConfig` uniform
   grid); assert `complementarity_report().violation_count == 0`.
   Canonical fixture (starting point): PUT, S = K = 100, T = 1, r = 5%,
   q = 0, σ = 1%, uniform x-grid over ≈[−2, 2] with spacing h ≈ 0.1
   (n ≈ 41), default time stepping. **Execution task 0 (binding): run this
   fixture on unmodified code and record its nonzero
   `violation_count`/`worst_kind` in the test comment before implementing
   the fix; if it does not violate as-is, adjust within the sweep table
   below until it does and pin the adjusted fixture.** The "nearby sweep"
   is a fixed parameterized table (e.g. σ ∈ {0.5%, 1%, 2%} × h ∈ {0.05,
   0.1} × r ∈ {2%, 5%}), not an informal instruction. A zero-KKT
   assertion can in principle still fail for #473 (active-set) reasons —
   the direct matrix-sign unit tests (item 2) are what attribute the
   improvement specifically to the M-matrix repair.
2. **Helper-invariant unit tests** (internal `SpatialOperator` fixture, in
   the style of `tests/internal/spatial_operator_jacobian_test.cc`):
   - assembled off-diagonal signs on deliberately **asymmetric non-uniform
     cells**, positive and negative drift (binding-side selection);
   - `b = 0` returns exactly `a` (and `LaplacianPDE` Jacobian unchanged);
   - small-ρ (series branch), and overflow-scale ρ (e.g. σ = 1e-4,
     h = 1: a_f finite, and the **assembled** binding off-diagonal ≥ 0
     including when `tanh(ρ) == 1`, i.e. the exactly-zero limit);
   - fitted `apply()` vs `assemble_jacobian()` consistency on a
     non-uniform grid: `L·u` from the matrix equals `apply_interior`
     output with a scale-aware tolerance (absolute floor plus a relative
     term scaled by row-term magnitudes, as in `validate_lcp_kkt` — not a
     bare 1e-14, which is brittle at coarse/high-ρ coefficients);
   - direct matrix-row dominance inspection (row sum `1 + w·r`, positive
     diagonal, non-positive off-diagonals) under r > 0 and a modest r < 0
     satisfying `1 + w·r > 0` — inspecting the matrix, not inferring
     structure from a KKT-clean solve; the condition-violating regime
     (`r ≤ −1/w`) is documented as out of contract, not tested as a solve;
   - a test-only linear PDE whose `operator()` and coefficient expansion
     agree, guarding the coefficient-authority dispatch contract.
   - **Update expected values in `spatial_operator_jacobian_test.cc`**
     (currently computes expectations from the physical `a`; must use the
     fitted coefficient — its FD-consistency test stays and now guards the
     new path).
3. **Known pin churn (explicit):**
   `AmericanOptionTest.NoDivCallPriceUnchangedByEnvelopeBC`
   (tests/american_option_test.cc:527) pins an absolute FDM price at
   1e-12 with nonzero drift; always-on fitting moves it. It gets
   deliberately re-pinned in this PR with a comment citing this spec (this
   is a scheme change, not the toolchain drift its comment warns about).
   Other audited absolute-price goldens (search scope: `grep 1e-12
   tests/*.cc`): envelope closed-forms, stencil unit tests, and analytic
   dimensionless-European comparisons — no FDM solve, unaffected.
   `bspline_bit_identity_test` goldens fit synthetic arrays, no FDM path —
   unaffected. Integration tests with 1e-12 internal-identity assertions
   compare quantities that move together and are expected to hold; any
   that fail get individually justified in the PR. No serialized-format
   change; newly generated price tables simply inherit the scheme change.
4. **Existing suites otherwise unchanged:** QuantLib A/B pins (ATM put
   4.376e-3 bound 5.5e-3), spatial convergence order ≈ 2, mass
   conservation, narrow-grid BC regression. The fitting perturbs
   well-resolved-grid coefficients by ~1e-5 relative — inside all
   tolerances.
5. **Laplacian equality (numerical, not bit-pattern):** the pure-diffusion
   path through the new combine loop (`a·d2u + 0·du − 0·u`) must equal the
   old path numerically (EXPECT_DOUBLE_EQ); bit identity is explicitly NOT
   claimed (signed zeros / FMA contraction may differ).
6. **Drift sign-crossing test:** a time-varying-rate config where `b(t)`
   changes sign during the solve, on a non-uniform grid — solves cleanly,
   zero KKT violations (guards the per-`t` binding-side re-derivation and
   the C¹ crossing).

## Documentation

`docs/MATHEMATICAL_FOUNDATIONS.md`: rewrite the M-matrix/cell-Péclet caveat
(added in #475) — describe the fitted scheme and state the guarantee
**narrowly and causally**: the fitting unconditionally removes
drift-induced positive stage off-diagonals (the cell-Péclet failure mode);
one-pass exactness additionally still requires the row-dominance condition
`1 + w·r(t) > 0` (violated only at absurd negative rates ≲ −2/Δt, kept as
a documented caveat since the API accepts any finite rate) and an interval
active set (#473). Do NOT promise a universally clean KKT report —
`validate_lcp_kkt` may still fire for non-interval active sets or the
negative-rate regime.

## Out of scope

- #473 PSOR fallback (non-interval active sets).
- #474 solver copy/move footgun.
- Any refactor of the apply/assemble duplication beyond the shared helper
  (a coefficient-first "assemble L once, apply as matrix" refactor was
  considered and deferred — see Decisions).

## Risks

- **Hot-path cost of `coth`** (`apply` runs per Newton iteration/stage).
  Mitigation is plan-level: if profiling shows regression, precompute the
  factor array once per stage time into `PDEWorkspace` scratch (b changes
  only with `t`). Correctness does not depend on caching.
- **Time-varying rates:** `b(t)` can change sign across the solve; the
  helper takes `t` and re-derives the binding side, so this is handled by
  construction — but the regression sweep should include one time-varying
  rate config.
- **Pinned-value churn:** bounded by the tight-pin audit above.

## Decisions (brainstorm Q&A, verbatim outcomes)

**Q1 — Which monotone drift discretization?** Options offered:
(A) always-on Il'in fitting (`a_f = a·ρ·coth(ρ)`, smooth, ~2nd order,
perturbs all prices ~1e-6 relative); (B) conditional Il'in (centered below
threshold, fitted above); (C) conditional 1st-order upwind (issue's minimal
candidate). User first asked *"which one is math sound?"* — analysis given:
all three can be made sound with binding-side spacing; conditional Il'in
was **withdrawn** as mischaracterized (at the ρ = 1 switch the fitted
factor is coth(1) ≈ 1.313, a 31% coefficient jump — the least clean); the
continuous minimal alternative `a_f = max(a, |b|·h/2)` (clipped diffusion,
C⁰) was offered instead alongside upwind. Re-asked with corrected options
{always-on Il'in (recommended), clipped diffusion, conditional upwind}.
**Choice: always-on Il'in fitting** — soundest in the strongest sense
(unconditional M-matrix + 2nd-order consistency + smoothness in σ +
classical pedigree), accepting loss of bit-identity with current output
(~1e-6 relative, inside existing tolerances).

**Q2 — Where does the fitting live?** Options offered: (A) SpatialOperator
shared helper (recommended; minimal blast radius, concepts untouched);
(B) coefficient-first refactor (assemble L's tridiagonal once per stage,
residual = matrix apply — consistency by construction but a hot-path
refactor beyond this issue's scope); (C) PDE-level fitted coefficient
(changes the `HasJacobianCoefficients` concept signature, ripples).
**Choice: SpatialOperator shared helper.**

**Folded without a question (presented in the approved design):** boundary
rows unfitted (already M-matrix; row-wise consistency preserved);
per-stage factor caching deferred to plan-level profiling; direct
off-diagonal sign check lives in the regression test rather than a
solver-path assert; docs caveat narrowed to #473. Design approved by user
("ok").

**Design review round 1 (Codex) — resolutions folded above:**
- *Critical:* "unconditional M-matrix" overclaimed — the fitting owns only
  the off-diagonal/Z-sign half; dominance needs `1 + w·r(t) > 0`, and the
  API accepts negative rates. **Resolution: document + test the condition,
  don't enforce it** (violating it needs r ≲ −2/Δt; the KKT validator
  reports it if it ever happens). Docs caveat NOT narrowed to #473 alone.
- *Smoothness:* corrected from "analytic" to C¹ at `b = 0` crossings on
  non-uniform cells; **binding-side spacing kept** over sign-independent
  `max(dx_l, dx_r)` (C¹ suffices for IV Newton / FD-vega; less added
  diffusion).
- *Floating point:* large-ρ overflow-safe formula (`z/tanh(ρ)` + clamp)
  pinned as binding law in the helper contract above.
- *Laplacian:* bit-identity weakened to numerical equality
  (EXPECT_DOUBLE_EQ) — the combine path's extra `0·du − 0·u` ops make bit
  identity a non-contract.
- *Tests:* helper-invariant unit tests added;
  `spatial_operator_jacobian_test.cc` expected-value update called out;
  `NoDivCallPriceUnchangedByEnvelopeBC` named as deliberate re-pin.

**Design review round 2 (Codex) — resolutions folded above:**
- *Critical:* the clamp on `a_f` alone doesn't survive assembly — two
  independently rounded terms can sum to a tiny negative at large ρ.
  **Resolution: sign-preserving reduced assembly is binding law** (helper
  exposes `a_f` and `z`; binding numerator computed literally as
  `a_f − z`; `diag = c − lower − upper`); high-ρ test asserts the
  assembled entry ≥ 0.
- *Dominance wording:* `1 + w·r > 0` stated as sufficient (not iff);
  `validate_lcp_kkt` acknowledged as a solution check, not a structural
  one; direct matrix-row dominance test added.
- *Regression fixture:* pinned concretely (PUT ATM, σ=1%, h≈0.1 uniform
  over [−2,2], r=5%) with a binding execution task 0 to record the
  pre-fix violation count, and a fixed sweep table.
- *Concept contract:* coefficient methods documented as the authoritative
  operator definition under `HasJacobianCoefficients` (+ `a > 0` when
  `b ≠ 0`); test-only PDE dispatch guard added.
- *Minors folded:* non-uniform extension wording for the classical claim;
  scale-aware apply/matrix tolerance; pin-audit wording narrowed to the
  demonstrated scope.
