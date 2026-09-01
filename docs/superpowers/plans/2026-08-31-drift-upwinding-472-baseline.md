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

## Task 3 re-pin: `AmericanOptionTest.NoDivCallPriceUnchangedByEnvelopeBC`

After wiring the Il'in-fitted assembly/apply paths into `SpatialOperator`
(Task 3), this ATM no-dividend call's full FDM solve moved as expected —
the fitting deliberately adds `O(ρ²)` diffusion to every interior row.

- Old pin: `10.447090628631905`
- New value: `10.447225343887069`
- Delta: `+0.00013471525516450811` (≈1.3e-4 on a ~10.45 price, ≈1.3e-5
  relative)

Measured via: `TMPDIR=/tmp/codex-skills/b1f0461c-8c94-4e80-ad4d-789f804f4e10
bazel test //tests:american_option_test --test_output=all
--test_filter='*NoDivCallPriceUnchanged*'`. The test's pin was updated to
the new value in the same commit as the discretization change.

## Task 6 final measurements

**Full suite.** `bazel test //... --test_output=errors`: **150/150 tests
pass** (148-target baseline + 2 new targets added by this branch, per the
Task 6 brief's expectation). No failures, no skips.

**Re-pin delta.** Recorded above under "Task 3 re-pin" —
`NoDivCallPriceUnchangedByEnvelopeBC`: `10.447090628631905 →
10.447225343887069`, Δ = +1.347e-4 (≈1.3e-5 relative), the one deliberate
change to a previously-pinned value on this branch, landed in the same
commit as the discretization change.

**Hot-path A/B (fitted-coefficient caching).** Recorded in
`.superpowers/sdd/2026-08-31-drift-upwinding-472/task-3-report.md`
("IMPORTANT #1: unmeasured std::tanh hot-path cost — measured, then
fixed"): uncached fitting on every apply/assemble call cost 0.794 ms vs.
0.430 ms with fitting off (1.85× regression); caching the fitted
coefficient per sampled drift `(a, b)` in `PDEWorkspace` (invalidated only
when the sampled drift actually changes between stages) brought it back to
0.420 ms — statistically indistinguishable from the fitting-off baseline
(ratio 0.977), i.e. the caching fix fully absorbed the regression.

**ATM-put-vs-QuantLib pin margin.** `tests/quantlib_sweep_regression_test.cc`
(`QuantLibSweepRegression.PricingAccuracyAcrossPutsAndCalls`, "put ATM"
row) prints its measured error. Run:

```
TMPDIR=/tmp/codex-skills/b1f0461c-8c94-4e80-ad4d-789f804f4e10 bazel test \
  //tests:quantlib_sweep_regression_test --test_output=all \
  --test_filter=QuantLibSweepRegression.PricingAccuracyAcrossPutsAndCalls
```

```
scenario             quantlib        mango      abs_err  threshold     margin  result
put ATM              6.090260     6.085964    4.295e-03  5.500e-03  1.205e-03  PASS
```

Measured `abs_err = 4.295e-3` against the pinned `5.5e-3` threshold —
margin `1.205e-3`, assertion unchanged by this branch (Task 6 is
docs-only; the underlying price was already produced by Task 3's
discretization change and Task 5's canonical-fixture pin).

**CI-parity builds.** `bazel build //benchmarks/...` succeeds (all 24
non-`real_market_data` benchmark `cc_binary` targets in
`benchmarks/BUILD.bazel` are tagged `manual`, so the bare wildcard resolves
to the single non-manual target, `real_market_data`, which was already
up-to-date). To verify the full benchmark set still compiles, every
`cc_binary` under `//benchmarks/...` was built explicitly by name
(bypassing the `manual` tag, which only affects wildcard expansion, not
explicit target lists). Two pre-existing build breakages were found and
excluded, both confirmed unchanged from `main` (`git diff eb7436d5 --
benchmarks/ src/math/BUILD.bazel` empty) and therefore out of scope for
this branch:

- `//benchmarks:cubic_spline_template_vs_hardcoded` — depends on
  `//src/math:cubic_spline_nd`, which does not exist.
- `//benchmarks:iv_fdm_sweep` — `iv_fdm_sweep.cc` includes
  `mango/pde/internal/pde_workspace.hpp` without a `deps` entry that
  exposes it (a residual from #419, "Hide PDEWorkspace from public API").

The remaining 22 targets all build cleanly with no warnings from project
code. Neither broken target is in the actual CI benchmark-build step
(`.github/workflows/ci.yml`, "Build benchmarks"), which names 7 specific
targets individually rather than using the `//benchmarks/...` wildcard —
so neither breakage, nor their exclusion here, affects CI's green/red
status.

`bazel build //src/python:mango_option` succeeds, no warnings from project
code.
