# Boundary correctness batch: issues #439 and #455

Status: design for review. Branch `fix/439-455-boundary-correctness`.

## Problem statement

**#439 — American option LCP sweep orientation and dividend-call right BC.**

*Item 1 (corrected — the issue text has the orientation backwards).*
`solve_thomas_projected` (`src/math/thomas_solver.hpp`) runs LU forward
elimination top-down, then projects during right-to-left back-substitution.
Projection-during-substitution is exact only when substitution **starts on
the active side** (Jaillet–Lamberton–Lapeyre orientation). The current sweep
therefore is exact for **right**-interval active sets (dividend-paying
calls) and inexact for **left**-interval ones (puts) — the opposite of the
issue's claim. Verified two ways (2026-08-30 spike, artifacts in session
scratchpad; posted to #439):

- Brute-force LCP enumeration (n=12 M-matrix, coupling w=5): left-active —
  current err 5.9e-2, mirrored exact; right-active — current exact,
  mirrored err 5.9e-2.
- Real pricer vs QuantLib 8000×801: switching puts to the mirrored sweep
  moves the ATM put (S=K=100, r=5%, sigma=20%, T=1) by 2.6e-3 and reduces
  the bias vs QuantLib from −0.0069 to −0.0044; the shift decays to 1.0e-4
  at S=90 and ~0 deep ITM (lock rows pin those). Calls confirm the mirror
  image: the current sweep is the better one for q > r calls everywhere
  tested.

Why puts never failed the suite: time-stepping re-injects obstacle
information through the RHS each step, the deep-ITM lock converts the deep
exercise region to exact Dirichlet pins, and per-step projection keeps
feasibility. The residual defect is a thin band near the free boundary
(~4 bp ATM), below the QuantLib suite's 1–2% tolerances.

History note: the sweep and the deep-ITM lock were co-designed in PR #200
to fix deep-ITM put lift; the monotonicity argument in
`docs/MATHEMATICAL_FOUNDATIONS.md` proves clamp propagation *into* the put
active set but not complementarity on the continuation side, which is
exactly the measured defect. No mirrored variant was ever attempted before
(checked `git log -S` across all branches).

*Item 2.* The call right BC (`src/option/american_option.cpp`,
`RightBCFunction`) is `u = e^x − forward_discount(t)`, blind to continuous
yield q and to discrete dividends. Impact: negligible for direct pricing
(the boundary sits ~5 sigma out), live in the table-building corner shared
with #437 where `ensure_moneyness_coverage` clamps the domain.

Hard constraint (issue #439, memory `no-european-shortcut`): the American
call path stays a genuine PDE/LCP solve for all inputs. No analytic
delegation for dividend-free calls.

**#455 — Neumann Jacobian boundary rows degenerate to identity.**
`apply_spatial_operator` (`src/pde/internal/pde_solver.hpp`) zeroes
`Lu[0]`/`Lu[n−1]`; the FD probes in `build_jacobian_boundaries`
differentiate that zeroed function, so Neumann boundary rows degenerate to
identity and the BC is enforced only post-hoc by the lagged one-sided ghost
formula in `NeumannBC::apply`. This is a regression: the C-era codebase hit
it as issue #6 (2% mass drift, zero-flux diffusion), fixed it with the
ghost-point method (commit ca727578, `L_0 = 2D(u_1−u_0)/dx²`), and lost the
fix in the C→C++ migration. Latent today (no in-repo Neumann user).

Adjacent latent trap found during design: `build_jacobian_boundaries` has
branches only for `dirichlet_tag` and `neumann_tag`; a `RobinBC` would
leave boundary Jacobian rows **uninitialized** (bug class of #433).

## Design

### A. Sweep orientation (#439 item 1)

**A1.** `solve_thomas_projected` gains a compile-time orientation. Public
shape (exact naming finalized in plan):

```cpp
enum class LcpActiveSide { Left, Right };

template<std::floating_point T, LcpActiveSide Side = LcpActiveSide::Right>
ThomasResult<T> solve_thomas_projected(...);
```

- `Right` (default): today's algorithm, unchanged — LU elimination top-down,
  projected substitution right-to-left starting in a right-side active set.
  Existing behavior and call sites are bit-identical by default.
- `Left`: the mirrored algorithm — UL elimination bottom-up, projected
  substitution left-to-right starting in a left-side active set. The spike
  implementation (validated against brute-force LCP references) is the
  reference: `c'[i] = lower[i−1]/denom` couples each row to its LEFT
  neighbor; substitution `x[i] = max(d'[i] − c'[i]·x[i−1], psi[i])`.

**A2.** The CRTP solver learns orientation from the derived class, mirroring
the `HasObstacle` pattern: derived solvers with an obstacle must expose
`static constexpr LcpActiveSide lcp_active_side`. `AmericanPutSolver` →
`Left` (the fix: puts switch to the mirrored sweep). `AmericanCallSolver`
→ `Right` (keeps the current, already-correct sweep).

**A3.** Deep-ITM lock: logic unchanged (orientation-agnostic interior scan,
`L(psi) < 0` subsolution test, identity-row conversion). Its interaction
with both sweeps is covered by dedicated LCP unit tests with identity rows
embedded in the active interval (see Tests). Note the identity-row
factorizations: an identity row has zero off-diagonal couplings in both LU
and UL elimination, so clamped values propagate to the continuation-side
neighbor correctly under the corrected orientation.

**A4.** Single-interval assumption policy (flagged decision, see Decisions):
after the one-pass solve, run an O(n) complementarity check — at each
clamped node (`u_i == psi_i` post-projection) verify `(A·u)_i ≥ rhs_i −
tol`. On violation, fire a USDT trace probe (count + max violation) and
continue; the projection still guarantees `u ≥ psi` and the behavior
matches what puts have silently done since PR #200. Exactness for
non-interval active sets (e.g. r < 0 regimes) is explicitly out of scope;
a follow-up issue will track an iterative (PSOR-style) fallback. No
printf — USDT only, per repo policy.

**A5.** Docs: rewrite the "Why This Works" subsection of
`docs/MATHEMATICAL_FOUNDATIONS.md` with the orientation-correct argument
(substitution must start on the active side; per-option-type mapping), and
note the complementarity check.

### B. Call right BC (#439 item 2)

**B1.** Time/phase conventions (pinned): solver time `t` is backward time =
time-to-expiry tau; calendar time is `T − t`. A dividend at calendar `d_j`
maps to `tau_j = T − d_j`. At solver time `t`, the dividend `j` is in the
option's remaining life iff `tau_j < t` (strict). During the implicit
stages of the step ending exactly at `tau_j` (before the temporal event
fires), the state is the post-dividend calendar side and strict `<`
correctly excludes dividend `j`.

**B2.** The boundary value is the deep-ITM stopping envelope. With
`x = x_max`, continuous yield q, discount factor `DF(t, s)` from solver
time `t` to solver time `s` (both in backward time, `DF` built from the
existing rate-spec machinery), and remaining dividends `tau_j < t` sorted
descending (earliest calendar first):

```
stop now:        e^x − 1
stop at tau_j⁺ (calendar d_j⁻, just before ex-date j):
                 e^x·e^{−q·(t − tau_j)}
                 − Σ_{tau_i > tau_j} (D_i/K)·DF(t, tau_i)·e^{−q·(tau_i − tau_j)}
                 − DF(t, tau_j)
stop at expiry:  e^x·e^{−q·t} − Σ_{all remaining} (D_i/K)·DF(t, tau_i)·e^{−q·tau_i}
                 − DF(t, 0)
u(x_max, t) = max over all candidates
```

The sums run over remaining dividends strictly *earlier in calendar time*
than the stopping date (`tau_i > tau_j`); each carries the discount to its
ex-date and the lost proportional carry `e^{−q·(tau_i − tau_j)}` from
ex-date to the stopping date. With q = 0 and no discrete dividends the
envelope reduces to `max(e^x − 1, e^x − DF(t,0))` = today's formula for
r ≥ 0. The envelope is the exact deep-ITM (linear-regime) value under the
solver's own dividend-jump model; it is evaluated per BC call — O(m²) in
the number of remaining dividends m, negligible (m is small, boundary rows
are two per stage).

**B3.** Event-phase rule: the dividend temporal event's jump interpolation
already produces the pre-dividend boundary value at `tau_j` by construction.
The solver must **not** overwrite it by re-evaluating the analytic BC at
the same numerical time (phase mismatch — strict `<` would give the
post-dividend value). Whether that requires suppressing the post-event BC
re-application or making it phase-aware is an implementation detail pinned
in the plan after reading `process_temporal_events`; the invariant is:
*boundary values at an event time reflect the pre-dividend side once the
event has fired.*

**B4.** `RightBCFunction` is constructed from `params_` (rate spec,
`dividend_yield`, filtered/merged discrete dividends via the existing
`filter_and_merge_dividends`, strike). Dividend-free calls: behavior
unchanged (regression-tested).

### C. Neumann ghost-point restoration (#455)

**C1.** Approach: centered ghost-point elimination — the boundary node
satisfies the PDE with the ghost eliminated via the centered BC. This is
simultaneously the textbook treatment and the restoration of the validated
C-era fix. For linear PDEs `L(u) = a·u_xx + b·u_x + c·u`, left boundary
with ghost spacing `h = dx[0]` and gradient g:

```
u_ghost = u_1 − 2·g·h
L_0     = (2a/h²)·(u_1 − u_0) + c·u_0 + g·(b − 2a/h)
```

Right boundary mirrored (`h = dx[n−2]`):

```
L_{n−1} = (2a/h²)·(u_{n−2} − u_{n−1}) + c·u_{n−1} + g·(b + 2a/h)
```

The drift term's centered difference at the node collapses to exactly g and
contributes only to the affine constant. At `a=D, b=c=0, g=0` this is the
old fix's `2D(u_1−u_0)/dx²`.

Accuracy claim (qualified): the eliminated boundary row has O(h) local
truncation error generically (`(1/3)·u'''·h` term); global second-order
convergence is expected for parabolic problems via boundary-error damping
and is what the convergence test asserts empirically (observed order ≥
~1.8 on a manufactured solution with nonzero third derivative at the
boundary — chosen precisely so the test cannot mask the term).

**C2.** Interface. The boundary rows live on `SpatialOperator`, gated on the
PDE's coefficient support:

```cpp
// on SpatialOperator, requires HasJacobianCoefficients<PDE>:
double eval_boundary_row(double t, BoundarySide side, double g,
                         std::span<const double> u) const;   // L at the node
struct BoundaryRowJacobian { double diag; double offdiag; }; // dL/du at node, neighbor
BoundaryRowJacobian boundary_row_jacobian(double t, BoundarySide side) const;
double boundary_row_affine(double t, BoundarySide side, double g) const;
```

`PDESolver` detects support with a new concept (`HasBoundaryRows<SpatialOp>`
checking those exact signatures) and uses it wherever a Neumann BC is
configured. A Neumann BC with an operator lacking the concept, or any
`RobinBC`, triggers a `static_assert` in the instantiated boundary path
naming the BC type and side (deterministic diagnostic instead of degenerate
or uninitialized rows).

Concept hygiene: `HasJacobianCoefficients` currently accepts nullary
accessors while `assemble_jacobian` calls `first_derivative_coeff(t)` /
`discount_rate(t)`. The concept's `requires` expressions are tightened to
the time-dependent call forms actually used, and `LaplacianPDE` gains
matching accessors (`second_derivative_coeff() = D`,
`first_derivative_coeff(double t = 0) = 0`, `discount_rate(double t = 0) =
0`) so it satisfies the corrected concept.

**C3.** Solver changes (`pde_solver.hpp`), all under `if constexpr` on the
BC tag so Dirichlet-only instantiations are unchanged:

- `apply_spatial_operator`: for a Neumann side, fill the boundary entry via
  `eval_boundary_row` instead of zeroing. The explicit TR-BDF2 RHS
  (`u^n + w·L(u^n)`) then becomes correct automatically.
- `build_jacobian_boundaries`: write the analytic row
  (`diag = 1 − w·jac.diag`, `offdiag = −w·jac.offdiag`); delete the FD
  boundary probes.
- Newton residual: with real boundary `L` entries the generic
  `F = u − rhs − w·L(u)` is already correct at Neumann rows;
  `apply_bc_to_residual` keeps Dirichlet-only overrides.
- Projected path: Neumann rows in the direct solve `A·u = rhs` additionally
  need the affine term folded into the RHS: `rhs_with_bc[row] += w·affine`.
  (No in-repo user combines obstacle + Neumann, but the generic path must
  be correct; covered by a unit test.)
- `apply_boundary_conditions`: becomes tag-aware and applies **Dirichlet
  only**. The lagged Neumann/Robin overwrite is removed from every solver
  call site (initialization, Newton loop, projected path, temporal
  events) — retaining any of them would re-lag the BC and destroy the
  restored accuracy. `NeumannBC::apply()` itself remains as a public
  standalone utility.
- Initial condition under Neumann: `initialize()` no longer forces the
  boundary value; the IC is taken as given (consistent with the BC now
  being enforced by the solve itself).

**C4.** `NeumannBC` API: gradient function only. The stored
`diffusion_coeff_` is vestigial; the two-argument constructor
`NeumannBC(Func, double)` is retained as a `[[deprecated]]` overload that
ignores the coefficient (source compatibility for existing tests/clients);
new single-argument constructor added.

**C5.** Grid preconditions: `n ≥ 3` enforced at solver creation (today
`interior_range` only asserts). Nonuniform grids: ghost spacing equals the
adjacent interior spacing, which keeps the centered BC difference
second-order in the local spacing.

## Tests

**T1 — LCP orientation unit tests** (`tests/thomas_solver_test` extension):
reference by exhaustive active-set enumeration (exact for M-matrices; no
PSOR tolerance ambiguity). Cases per sweep: left- and right-interval active
sets, empty and full active sets, nonconstant obstacle, identity lock rows
embedded inside the active interval, Dirichlet identity rows with RHS
overrides, and a deliberately non-interval active set (asserts the
complementarity check fires / documents the limitation). Full KKT assertions
(primal feasibility, dual feasibility at clamped nodes, complementarity).

**T2 — Put pricing regression** (QuantLib live reference): the spike's put
scenario set pinned as regression tests with tolerances reflecting the
improved accuracy (ATM put bias must be ≤ the mirrored-sweep level, not the
old one). Bit-identity goldens (`bspline_bit_identity_test`) regenerated —
put prices legitimately move by 1e-4…2.6e-3 per $100 strike.

**T3 — Dividend-call pricing** (QuantLib): continuous-dividend calls
q = 8% > r = 5% swept S/K ∈ {1.0, 1.2, 1.3, 1.5, 2.0, 3.0}, tight tolerance
near the free boundary; a discrete-dividend call spanning an ex-div date;
today's suite has exactly one ATM call (#444 gap).

**T4 — Right-BC unit tests** (direct `RightBCFunction` evaluation, no PDE):
tau → 0 limit, both numerical sides of an ex-date (strict-`<` phase rule),
multiple dividends where an intermediate stopping date dominates the
envelope, combined q + discrete dividends, q = 0 no-div reduction to the
current formula, and a table-tail-style regression on a deliberately
narrow grid where the boundary bias reaches sampled nodes (the #437 corner).

**T5 — Neumann restoration** (heat equation via `LaplacianPDE`):
mass conservation on the zero-flux Gaussian (issue #6 setup) to ~1e-10
using trapezoidal quadrature (the quantity the ghost scheme conserves);
manufactured-solution spatial convergence with inhomogeneous, time-varying
Neumann data and nonzero boundary third derivative, asserting observed
order ≥ 1.8; an obstacle+Neumann affine-term unit test at the linear-solve
level; `n = 2` rejection and `n = 3` smoke; Newton-path Neumann solve
converging in few iterations (the row is now genuinely solved).

**T6 — Full gates**: `bazel test //...`, `bazel build //benchmarks/...`,
`bazel build //src/python:mango_option`, no new warnings.

## Delivery

Single PR, commits separated by concern: (1) LCP orientation + trait +
complementarity probe + T1/T2, (2) call right BC + T3/T4, (3) Neumann
restoration + BC-type guard + T5, (4) docs (MATHEMATICAL_FOUNDATIONS
rewrite) + golden regeneration. Update issues #439 (already corrected by
comment) and #455 on merge; file the PSOR-fallback follow-up issue.

## Decisions

- **#439 item 2 in scope here** (user): the BC fix is local, closes #439
  fully, decouples from #437. (Alternatives: defer item 2; continuous-q
  only.)
- **Ghost-point elimination for #455** (user, after asking for the
  mathematically sound option): textbook-correct, second-order, and the
  restoration of the C-era validated fix; the first-order constraint row is
  affirmatively ruled out — it *was* issue #6. (Alternatives: constraint
  row; one-sided FD probing.)
- **Sweep mapping corrected** (evidence-driven, user-approved direction):
  put → mirrored, call → current, per the verified reversal of the issue's
  premise. Batch proceeds as one PR despite put-price movement (measured
  small: ≤ 2.6e-3 per $100 strike).
- **Non-interval active sets: validate + trace, no fallback solver**
  (assistant recommendation, FLAGGED for explicit user sign-off at the plan
  gate): O(n) complementarity check + USDT probe + follow-up issue for
  PSOR. (Alternatives: hard error — breaks currently-working exotic
  inputs; PSOR fallback — new solver component, real scope growth.)
- **Neumann gating via tightened `HasJacobianCoefficients` + new
  `HasBoundaryRows` concept on `SpatialOperator`** (from design review):
  the solver cannot reach the PDE's private coefficients; boundary-row
  methods on the operator are the clean seam.
- **`NeumannBC(Func, double)` retained as deprecated** (from design
  review): removing it is unrelated breakage.
- **Robin BCs: compile-time rejection** with a named diagnostic; Robin row
  assembly is out of scope.
