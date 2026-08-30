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
embedded in the active interval (see Tests). Precision on the mechanism:
the identity row's own off-diagonals are zero, but the *adjacent* rows keep
their coupling to the locked unknown — that retained coupling is what
transfers the pinned value into the neighboring equations, in both
factorization directions.

**A4.** Single-interval assumption policy (flagged decision, see Decisions):
after the one-pass solve, run an O(n) complementarity check at each clamped
node (`u_i == psi_i` post-projection): `(A·u)_i ≥ rhs_i − tol_i` with a
scale-aware tolerance `tol_i = atol + rtol·(|diag_i·u_i| + |rhs_i|)`. The
check lives in a testable free function returning
`{violation_count, max_violation}`; production code emits a USDT probe from
that result (no printf, per repo policy). **On violation beyond tolerance
the solve returns a `SolverError`** — once the solver has proof its result
is not an LCP solution, returning it silently is not acceptable. For all
single-interval regimes (r ≥ 0 puts, dividend calls) the check passes by
construction; only genuinely pathological active-set topologies (e.g.
r < 0 with q > 0) can trip it, and those prices were silently wrong before.
A follow-up issue tracks an iterative (PSOR-style) fallback that would
solve rather than refuse these regimes.

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

**B2.** The boundary value is the deep-ITM (linear-regime) optimal-stopping
value: `u(x_max, t) = max over stopping times s ∈ [0, t]` of

```
f(s) = e^x·e^{−q·(t − s)}
       − Σ_{tau_i > s} (D_i/K)·DF(t, tau_i)·e^{−q·(tau_i − s)}
       − DF(t, s)
```

with `x = x_max`, continuous yield q, `DF(t, s)` the discount factor from
solver time `t` to solver time `s` (backward time; built from the existing
rate-spec machinery), and the sum over remaining dividends strictly earlier
in calendar time than the stopping date; each carries the discount to its
ex-date and the lost proportional carry `e^{−q·(tau_i − s)}` from ex-date
to stopping date.

`f` is piecewise smooth with breakpoints at dividend taus and yield-curve
tenor knots, so the maximum is found exactly by evaluating a finite
candidate set: `s = t` (stop now), `s = 0` (expiry), each `tau_j⁺` (just
before an ex-date), each curve knot inside `(0, t)`, and per flat-forward
segment the analytic interior stationary point of
`A·e^{−q·(t−s)} − K_eff·DF(t, s)` when it falls inside the segment (for a
flat forward rate r_f: `q·A·e^{−q(t−s)} = r_f·DF·(K_eff-term)` has the
closed-form root `s* = t − ln(q·A/(r_f·C))/(q − r_f)`, clamped to the
segment; segments where q or r_f make the derivative one-signed contribute
only endpoints — the plan pins the exact per-segment algebra including the
dividend-sum contribution to the effective constant). Endpoint-only
candidate sets ("now / ex-dates / expiry") are NOT sufficient: between
breakpoints `f` can peak in the interior for supported `RateSpec` inputs,
and on narrow table grids `x_max` is not asymptotically deep, which is
precisely the regime item 2 matters in.

With q = 0 and no discrete dividends the envelope reduces to
`max(e^x − 1, e^x − DF(t,0))` = today's formula for r ≥ 0. Cost: O((m+k)²)
per BC call for m remaining dividends and k curve knots — negligible.

The evaluator is a testable free function in an internal header
(`mango::detail`, e.g. `src/option/detail/call_boundary_envelope.hpp`);
the anonymous-namespace `RightBCFunction` in `american_option.cpp` becomes
a thin wrapper around it (the current nested type is untestable from a
separate translation unit).

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

**B5.** Event/grid alignment precondition (adjacent latent bug, in scope
because B3's phase rule depends on it): `process_temporal_events` applies
an event after evolving to `t_new` even when `event.time < t_new`, so the
jump sees the wrong-time state unless every dividend tau is a time-grid
point. Automatic grids include dividend taus, but a direct
`AmericanOptionSolver` with a custom grid whose `mandatory_times` omit them
does not. Fix: the American solver's grid-resolution path always merges
filtered dividend taus into the mandatory times, including when the caller
supplies their own nonempty list. Regression test: custom grid omitting the
dividend date; assert event alignment, post-event boundary value, and
snapshot values.

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
or uninitialized rows). Structure: each boundary path uses
`if constexpr (HasBoundaryRows<Op>) { ...method calls... } else {
static_assert(false_v<Op>, ...); }` so the diagnostic is the only error —
a bare `static_assert` followed by unconditional method calls would bury it
under substitution failures.

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
ignores the coefficient (source compatibility); new single-argument
constructor added. All in-repo uses migrate to the one-argument form so the
"no new warnings" gate holds; one compatibility compile test keeps the
deprecated overload covered with the warning locally suppressed.

**C5.** Grid preconditions: `n ≥ 3` rejected in `Grid<T>::create` (it
already returns `std::expected`, and every current spatial stencil is
3-point; the plan verifies no legitimate 2-point `Grid` user exists before
pinning the layer — fallback owner if one exists:
`AmericanOptionSolver::create` plus a checked guard on the PDE-solver
path). Nonuniform grids: ghost spacing equals the adjacent interior
spacing, which keeps the centered BC difference second-order in the local
spacing.

## Tests

**T1 — LCP orientation unit tests** (`tests/thomas_solver_test` extension):
reference by exhaustive active-set enumeration (exact for M-matrices; no
PSOR tolerance ambiguity). Cases per sweep: left- and right-interval active
sets, empty and full active sets, nonconstant obstacle, identity lock rows
embedded inside the active interval, Dirichlet identity rows with RHS
overrides, and a deliberately non-interval active set. Full KKT assertions
(primal feasibility, dual feasibility at clamped nodes, complementarity).
The complementarity validator is tested directly through its free-function
API (`{violation_count, max_violation}` on known-bad and known-good
solutions) — observability does not depend on USDT, which compiles to
no-ops without tracing support.

**T2 — Put pricing regression** (QuantLib live reference at 8000 time
steps × 801 grid nodes, `FdBlackScholesVanillaEngine`, Actual365Fixed): the
spike's put scenario set pinned as regressions. Concrete acceptance
thresholds (absolute error vs that reference, from the measured
mirrored-sweep run): ATM S=K=100 r=5% T=1 |err| ≤ 5.5e-3 (measured 4.4e-3;
old sweep 6.9e-3 fails); the remaining scenarios ≤ 1.25× their measured
mirrored-sweep error, exact numbers recorded in the test from the
implementation run. Bit-identity goldens (`bspline_bit_identity_test`)
regenerated — put prices legitimately move by 1e-4…2.6e-3 per $100 strike.

**T3 — Dividend-call pricing** (QuantLib, same engine/conventions):
continuous-dividend calls q = 8% > r = 5% swept
S/K ∈ {1.0, 1.2, 1.3, 1.5, 2.0, 3.0} with absolute tolerance 7.5e-3
(1.5× the worst measured spike error, 4.7e-3 ATM); a discrete-dividend
call spanning an ex-div date, referenced against QuantLib's
`FdBlackScholesVanillaEngine` with a `DividendVanillaOption`-style discrete
schedule (exact QuantLib API and date mapping pinned in the plan; tolerance
set to 1.5× the measured error at implementation, recorded in the test).
Today's suite has exactly one ATM call (#444 gap).

**T4 — Envelope evaluator unit tests** (direct calls to the extracted
`mango::detail` evaluator; no PDE): tau → 0 limit, both numerical sides of
an ex-date (strict-`<` phase rule), multiple dividends where stopping just
before an *intermediate* ex-date dominates, a flat-rate case whose interior
stationary point dominates all endpoint candidates (verified against a
dense brute-force scan of f(s)), an upward/downward `YieldCurve` case with
knots inside the horizon (same brute-force check), combined q + discrete
dividends, q = 0 no-div reduction to the current formula, and a
table-tail-style pricing regression on a deliberately narrow grid where
the boundary value reaches sampled nodes (the #437 corner). Plus the B5
custom-grid regression: dividend date omitted from user `mandatory_times`,
asserting event alignment and post-event boundary value.

**T5 — Neumann restoration**: direct `SpatialOperator` boundary-row algebra
tests with nonzero a, b, c, g on BOTH sides and on a nonuniform grid,
asserting `eval_boundary_row == jacobian·u + affine` and the hand-derived
closed forms (the heat-equation tests alone exercise only `a`); heat
equation via `LaplacianPDE`: mass conservation on the zero-flux Gaussian
(issue #6 setup) to ~1e-10 using trapezoidal quadrature (the quantity the
ghost scheme conserves); manufactured-solution spatial convergence with
inhomogeneous, time-varying Neumann data and nonzero boundary third
derivative, asserting observed order ≥ 1.8; an obstacle+Neumann
affine-term unit test at the linear-solve level; `n = 2` rejection and
`n = 3` smoke; Newton-path Neumann solve converging in few iterations (the
row is now genuinely solved).

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
  mathematically sound option): textbook-correct (globally near-second-order
  for parabolic problems; boundary-row LTE is O(h) generically — see C1's
  qualification), and the restoration of the C-era validated fix; the
  first-order constraint row is affirmatively ruled out — it *was* issue
  #6. (Alternatives: constraint row; one-sided FD probing.)
- **Sweep mapping corrected** (evidence-driven, user-approved direction):
  put → mirrored, call → current, per the verified reversal of the issue's
  premise. Batch proceeds as one PR despite put-price movement (measured
  small: ≤ 2.6e-3 per $100 strike).
- **Non-interval active sets: validate + `SolverError`, no fallback
  solver** (design-review round 2; FLAGGED for explicit user sign-off at
  the plan gate): a result that provably fails complementarity must not be
  returned as a price — refusing honestly beats both silence (those prices
  were silently wrong before) and a PSOR fallback (new solver component,
  real scope growth — filed as a follow-up issue instead). Check passes by
  construction in every single-interval regime.
- **Neumann gating via tightened `HasJacobianCoefficients` + new
  `HasBoundaryRows` concept on `SpatialOperator`** (from design review):
  the solver cannot reach the PDE's private coefficients; boundary-row
  methods on the operator are the clean seam.
- **`NeumannBC(Func, double)` retained as deprecated** (from design
  review): removing it is unrelated breakage.
- **Robin BCs: compile-time rejection** with a named diagnostic; Robin row
  assembly is out of scope.
- **Right BC is the exact piecewise stopping maximum, not an endpoint
  approximation** (design-review round 2): endpoint-only candidate sets
  miss interior stationary points under supported `RateSpec` inputs, and
  narrow table grids — the one regime where item 2 matters — are exactly
  where `x_max` is not asymptotically deep. Candidates = endpoints + curve
  knots + per-segment analytic stationary points.
- **Dividend taus always merged into custom time grids** (design-review
  round 2): `process_temporal_events` fires events at completed steps, so
  B3's phase semantics require every dividend tau to be a grid point; the
  American solver's grid-resolution path enforces it even when the caller
  supplies `mandatory_times`.
