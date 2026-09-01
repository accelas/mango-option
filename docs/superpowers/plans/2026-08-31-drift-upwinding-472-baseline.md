# #472 pre-fix baseline: Jacobian sign flip confirmed, KKT clean

Measurement only — no production code changed. All probes described here
were temporary, run against unmodified code, and reverted; only this doc
is committed.

## Summary

The premise that the *current* centered-difference discretization produces
a nonzero `AmericanOptionSolver::complementarity_report()` on some
low-vol/coarse-grid fixture **does not hold** for every fixture family
tried (round 1 + round 2, 23 + 34 = 57 full-solve cells, 0 violations
anywhere). The underlying defect the M-matrix repair (#472) targets is
real and directly confirmed at the matrix level (Part A): centered
differencing produces a positive Jacobian off-diagonal (`jac.lower() ≈
+0.245`) at cell Péclet ≈ 50, which is the Z-matrix/M-matrix sign
violation the issue describes. But that sign flip does not propagate into
an observable KKT violation for any fixture searched, because of how the
solver's "deep ITM lock" (Dirichlet identity rows) and the projected
Thomas algorithm's self-consistent active-set construction interact with
it (Part B). A **structural** regression test (Part A's direct
matrix-row-sign inspection) is the correct baseline for this issue, not a
full-solve KKT assertion — see Concerns.

## Part A — structural confirmation (Jacobian sign flip)

Temporary test added to `tests/internal/spatial_operator_jacobian_test.cc`
(reverted after measurement; `git diff` before revert showed only this
added `TEST`, nothing else touched):

```cpp
TEST(SpatialOperatorJacobianTest, HighPecletSignFlipProbe472) {
    auto grid_spec = GridSpec<double>::uniform(-2.0, 2.0, 41).value();
    auto grid_buf = grid_spec.generate();
    auto grid_view = grid_buf.view();
    const size_t n = grid_view.size();
    auto pde = BlackScholesPDE<double>(/*sigma=*/0.01, /*rate=*/0.05, /*div=*/0.0);
    auto spacing = std::make_shared<GridSpacing<double>>(grid_view);
    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();
    auto spatial_op = create_spatial_operator(std::move(pde), spacing, workspace);
    auto jac = workspace.jacobian();
    spatial_op.assemble_jacobian(0.0, 1.0, jac);  // J = I - 1.0·L
    // print jac.lower()[i-1], jac.diag()[i], jac.upper()[i] for i in {1,20,39}
}
```

Fixture: uniform grid over [-2, 2], n=41 (h=0.1), σ=1%, r=5%, q=0%,
`coeff_dt = 1.0` so `J = I − L`.

Ran: `TMPDIR=/tmp/codex-skills/b1f0461c-8c94-4e80-ad4d-789f804f4e10 bazel
test //tests/internal:spatial_operator_jacobian_test
--test_filter='*HighPecletSignFlipProbe472*' --test_output=all`

**Measured values (all three interior rows, constant-coefficient grid):**

| i  | `jac.lower()[i-1]` | `jac.diag()[i]` | `jac.upper()[i]` |
|----|---------------------|------------------|---------------------|
| 1  | 0.2447499999999995  | 1.0600000000000012 | -0.25475000000000064 |
| 20 | 0.24475000000000061 | 1.0599999999999989 | -0.25474999999999948 |
| 39 | 0.24474999999999922 | 1.0600000000000023 | -0.25475000000000148 |

`jac.lower() ≈ +0.24475 > 0` at every interior row — the expected sign
flip. (`J = I − L` and the Z-matrix requirement is off-diagonals `≤ 0`, so
a *positive* `jac.lower()` is exactly the violation: the raw operator's
lower off-diagonal `a/h² − b/(2h)` is negative, magnitude ≈ 0.245, matching
hand calculation: `a = ½σ² = 5e-5`, `b = r − q − ½σ² ≈ 0.04995`,
`a/h² − b/(2h) = 0.005 − 0.24975 = −0.24475`, cell Péclet `ρ = bh/2a ≈ 50`.)

This directly confirms the M-matrix defect #472 exists in the current
code, independent of whether it manifests as a solver-level KKT violation
on any particular full-solve fixture.

## Part B — why the projected-Thomas KKT check stays clean despite the flip

Read: `src/math/thomas_solver.hpp` — `solve_thomas_projected2` (the
Brennan-Schwartz-style one-pass LCP solver) and `validate_lcp_kkt`.

**What the validator checks, per row `i`:**
- Primal feasibility: `u[i] >= psi[i] - tol` for every row.
- Complementarity: if `active_mask[i]`, `u[i] == psi[i]` (within `tol`).
- Dual feasibility (active rows only): `(A u)[i] >= rhs[i] - tol`.
- Residual/complementarity (inactive rows): `(A u)[i] == rhs[i]` (within
  `tol`).
- Tolerance: `tol = atol + rtol * scale`, `atol = 1e-12`, `rtol = 1e-10`,
  `scale = |lower[i-1]·u[i-1]| + |diag[i]·u[i]| + |upper[i]·u[i+1]| +
  |rhs[i]|` — a relative floor scaled by the row's own term magnitudes,
  not a bare absolute cutoff. Non-finite inputs are an automatic
  violation (`worst_kind = 2`).

**Why the σ=1% fixture (and everything else tried) stays KKT-clean:**

1. **The active mask is self-consistent by construction.** `active_mask[i]`
   is written by `solve_thomas_projected2` itself, in the same
   back-substitution pass that produces `solution[i]`
   (`active_mask[i-1] = (unconstrained < psi[i-1])`, then
   `solution[i-1] = max(unconstrained, psi[i-1])`). So primal feasibility
   and complementarity for the rows *that pass evaluates* can never fail —
   they are definitionally satisfied by the algorithm that produced them,
   regardless of the matrix's sign structure.
2. **Inactive-row residual correctness comes from LU/Gaussian elimination,
   not from the M-matrix property.** For an inactive row, `solution[i]`
   equals the reduced-row prediction exactly (up to floating point), and
   that equality is what makes `(A u)[i] = rhs[i]` hold — a property of
   correct tridiagonal (LU) elimination on any non-singular matrix, sign
   structure notwithstanding. `worst_kind=2` would only fire near a
   singular/ill-conditioned pivot, which this well-scaled fixture family
   never approaches.
3. **The one real failure mode — dual infeasibility on an active row
   (`worst_kind=1`) — needs the sign flip to actually corrupt which rows
   get clamped, and by how much.** `src/pde/internal/pde_solver.hpp`
   (~line 720, "CRITICAL FIX #2: Lock Deep Exercise Region") pre-empts
   most of the region where this could happen: any node with
   `ψ[i] > 0.95·max(ψ)` that is already on the obstacle and where holding
   is a strict subsolution (`L(ψ)[i] < 0`) gets its Jacobian row replaced
   with an **identity row** (`lower=0, diag=1, upper=0`, `rhs=ψ[i]`)
   *before* the projected Thomas solve runs. This removes the
   drift-affected off-diagonal entirely from the deepest, most
   Péclet-affected part of the active region — the very region most
   likely to show a sign-flip-driven KKT defect never reaches the
   validator with its flipped coefficients intact.
4. **The remaining transition band (near the free boundary, not caught by
   the deep-ITM lock) stays magnitude-diagonally-dominant even though the
   Z-matrix sign condition fails.** At the measured row,
   `|diag| ≈ 1.06` vs. `|lower| + |upper| ≈ 0.245 + 0.255 = 0.50` — still
   comfortably diagonally dominant in absolute value, which is what
   actually governs Gaussian-elimination stability and how far a
   perturbed coefficient can shift the clamp/no-clamp decision. For a
   single, well-separated free boundary (interval active set — no #473
   oscillation), that dominance appears sufficient to keep the one-pass
   sweep's active-set decisions numerically consistent with the true LCP
   solution, even though the sign condition that *guarantees* this in
   general (the M-matrix property) does not hold.
5. Tolerance is not the explanation — `atol=1e-12`/`rtol=1e-10` is tight,
   and the mechanism above (1–4) predicts genuinely-near-zero defects,
   not defects merely below threshold.

In short: the KKT-clean full-solve result is real, not a measurement gap,
but it is not evidence the M-matrix defect is harmless — the deep-ITM
identity-row lock is a second, independent workaround already patching
over exactly the region where the sign flip would otherwise bite, and the
transition-band argument (point 4) is a magnitude-dominance property of
*this specific fixture family*, not a general guarantee (higher Péclet
combined with a wider, less-dominant transition band, or an active set the
deep-ITM lock doesn't reach, could still break it — this is exactly the
#473 concern the design spec already flags).

## Part C — wider KKT search (public API, ATM unless noted)

Temporary combined probe in `tests/american_option_test.cc`
(`WiderKktSearch472PartC`, reverted after measurement). Grid: uniform
[-2, 2], n ∈ {21, 41}, n_time ∈ {20, 50} (family 5 fixes n=21 as
specified). 34 cells total.

Ran: `TMPDIR=/tmp/codex-skills/b1f0461c-8c94-4e80-ad4d-789f804f4e10 bazel
test //tests:american_option_test --test_filter='*WiderKktSearch472PartC*'
--test_output=all`

| Family | Params (spot/strike=100 unless noted) | Cells | Result |
|---|---|---|---|
| 1a | PUT, b<0: r=0.00, q=0.10, σ=1% | n∈{21,41}×n_time∈{20,50} = 4 | 0 violations, all 4 |
| 1b | PUT, b<0: r=0.02, q=0.10, σ=1% | 4 | 0 violations, all 4 |
| 2a | CALL, b>0: r=0.05, q=0.00, σ=1% | 4 | 0 violations, all 4 |
| 2b | CALL, b<0: r=0.00, q=0.10, σ=1% | 4 | 0 violations, all 4 |
| 3a | PUT, off-ATM spot=90, r=0.05, σ=1% | 4 | 0 violations, all 4 |
| 3b | PUT, off-ATM spot=110, r=0.05, σ=1% | 4 | 0 violations, all 4 |
| 4a | PUT, maturity=0.25, r=0.05, σ=1% | 4 | 0 violations, all 4 |
| 4b | PUT, maturity=3.0, r=0.05, σ=1% | 4 | 0 violations, all 4 |
| 5  | PUT, σ=0.5%, r=0.10, n=21 (h=0.2, Péclet≈400) | n_time∈{20,50} = 2 | 0 violations, both |

**34/34 cells: `violation_count = 0`, `max_violation = 0`,
`worst_kind = -1`.** No failing cell was found; the "try 2-3 neighbors"
follow-up step was not needed since the initial search itself never
triggered.

## Round 1 recap (unchanged conclusion, restated for the record)

Round 1 (canonical fixture + the design spec's pinned 12-cell sweep table
σ∈{0.5%,1%,2%}×h∈{0.05,0.1}×r∈{2%,5%}, plus 10 further diagnostic cells —
23 cells total, all via `AmericanOptionSolver::complementarity_report()`)
also found `violation_count = 0` everywhere. Combined with round 2's Part
C, **57 full-solve cells across a wide parameter space (drift sign,
option type, moneyness, maturity, cell Péclet up to ≈400) show zero KKT
violations**, while Part A independently confirms the Jacobian-level
defect the fix targets is present and matches the predicted sign and
magnitude.

## Concerns for whoever picks this up

- **A full-solve KKT regression assertion (`complementarity_report()
  .violation_count == 0`) is not a useful pre/post-fix discriminator for
  this codebase as it stands** — it already reads `0` before the fix, on
  every fixture tried. Task 5 (or whatever task pins the regression test)
  should pin the **Part A structural test** (direct Jacobian off-diagonal
  sign, à la `HighPecletSignFlipProbe472` above) as the primary regression
  guard, since that is the only place a nonzero pre-fix / zero post-fix
  contrast was actually observed. A full-solve KKT assertion can still be
  added as a *non-regressing* sanity check (it should remain 0 after the
  fix too), but it does not by itself demonstrate the fix did anything.
- The deep-ITM identity-row lock (Fix #2 in `pde_solver.hpp`) is doing
  real work masking the sign flip's practical impact for large parts of
  the domain. It is a second, pre-existing correctness patch layered on
  top of the same underlying defect — worth being aware of during the
  #472 implementation in case its `0.95·max(ψ)` threshold interacts with
  the new fitted-coefficient scheme's own boundary-row behavior.
