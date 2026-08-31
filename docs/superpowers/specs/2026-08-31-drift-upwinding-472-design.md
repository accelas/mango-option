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

So the M-matrix condition is **per-side**: for `b > 0` the lower entry binds
(`a ≥ b·dx_right/2`); for `b < 0` the upper entry binds
(`a ≥ |b|·dx_left/2`). Since both derivative stencils annihilate constants,
`diag_L = c − lower_i − upper_i`; once the off-diagonals are ≥ 0 and
`c ≤ 0`, the stage matrix `I − w·L` has positive diagonal, non-positive
off-diagonals, and row sum `1 + w·r` — an M-matrix. Off-diagonal
non-negativity of `L` is therefore the one property to enforce.

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

**M-matrix proof.** `ρ·coth(ρ) ≥ max(1, ρ)` for all ρ ≥ 0. Hence
`a_f ≥ a` (never less diffusion than today) and
`a_f ≥ a·ρ_i = |b|·h_binding/2`, which is exactly the binding-side
condition, so the binding off-diagonal is ≥ 0 for **all** σ, h, b (it decays
to 0⁺ exponentially as ρ → ∞, and a zero off-diagonal is admissible: the
stage matrix stays an M-matrix via its strictly positive diagonal and row
sum `1 + w·r`). The non-binding off-diagonal is `(a_f + |b|·dx/2)/(...) > 0`
trivially. Diagonal and row-sum properties follow as above.

**Accuracy.** `ρ·coth(ρ) = 1 + ρ²/3 − ρ⁴/45 + …`, so the scheme adds
numerical diffusion `a·O(ρ²)` — second-order consistent; this is the
classical uniformly-convergent (Il'in/Allen–Southwell) scheme for
convection-dominated problems. On well-resolved grids ρ ~ 1e-2–1e-3, so
coefficients move by ~1e-4–1e-6 relative: inside every existing accuracy
pin, but **not bit-identical** to current output (accepted; see Decisions).

**Smoothness.** `ρ·coth(ρ)` is analytic and even in ρ. No branch, no
threshold: prices remain smooth in σ and h, which matters for IV Newton
solves and the FD-vega adaptive error metric. Implementations must use the
Taylor series below a small-ρ cutoff so that `b → 0` (and `LaplacianPDE`,
which has `first_derivative_coeff() == 0`) yields exactly `a_f = a` —
bit-identical pure-diffusion behavior and no 0/0.

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
4. **Boundary rows stay unfitted.** The ghost-eliminated rows are
   `diag = c − 2a/h²`, `offdiag = +2a/h²` — already M-matrix for any `b`
   (drift enters only the affine term), and
   `eval_boundary_row = jacobian·u + affine` holds row-wise by
   construction, so per-row residual/Jacobian consistency is unaffected.
   The asymmetry (interior rows fitted, boundary rows not) is a consistent
   O(ρ²) perturbation and is documented in code and in
   MATHEMATICAL_FOUNDATIONS.

No changes to `BlackScholesPDE`, `LaplacianPDE`, `CenteredDifference`,
`GridSpacing`, the `HasJacobianCoefficients`/`HasBoundaryRows` concepts, or
`PDESolver`.

## Testing

1. **Regression test** (per acceptance criteria, in the #475 regression
   style): σ = 1%, h ≈ 0.1 (coarse grid, r = 5%) full American solve;
   assert `complementarity_report().violation_count == 0` (nonzero today)
   and, directly, that the assembled Jacobian's off-diagonals have the
   M-matrix sign at that config. Sweep a few nearby low-vol/coarse-grid
   configs.
2. **Existing suites unchanged:** QuantLib A/B pins (ATM put 4.376e-3
   bound 5.5e-3), spatial convergence order ≈ 2, mass conservation,
   narrow-grid BC regression. The fitting perturbs well-resolved-grid
   coefficients by ~1e-5 relative — inside all tolerances.
3. **Tight-pin audit (explicit task):** any test pinning FDM output at
   ~1e-12 (e.g. the no-dividend call pin from #475, noted
   toolchain-sensitive) must be checked. If such a pin compares two FDM
   solves that are both fitted identically, it holds; if it pins an
   absolute value, it may need re-pinning with justification in the PR.
4. **Laplacian bit-identity:** a test asserting the pure-diffusion path
   (`b = 0`) produces bit-identical output through the new combine path
   (guards the series branch returning exactly `a`).

## Documentation

`docs/MATHEMATICAL_FOUNDATIONS.md`: rewrite the M-matrix/cell-Péclet caveat
(added in #475) — describe the fitted scheme, state the unconditional
M-matrix guarantee, and narrow the remaining one-pass-exactness caveat to
non-interval active sets (#473).

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
