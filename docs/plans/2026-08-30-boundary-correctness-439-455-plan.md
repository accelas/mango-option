# Boundary Correctness Batch (#439, #455) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Brennan-Schwartz sweep orientation (puts currently use the
wrong-side sweep), make the American-call right boundary dividend-aware via
an exact stopping envelope, and restore the ghost-point Neumann boundary
treatment lost in the C→C++ migration.

**Architecture:** Compile-time sweep orientation on the projected Thomas
solver selected by a CRTP trait; a testable stopping-envelope evaluator in
`mango::detail` wrapped by the call solver's right BC with an event-epoch
counter for dividend phase; analytic ghost-eliminated boundary rows on
`SpatialOperator` gated by concepts, replacing degenerate FD probes; a
full-KKT LCP validator reporting (never failing) per solve.

**Tech Stack:** C++23, Bazel/Bzlmod, GoogleTest, QuantLib (test-only,
`linkopts = ["-lQuantLib"]`), USDT via `src/support/ivcalc_trace.h`.

**Spec:** `docs/plans/2026-08-30-boundary-correctness-439-455-design.md`
(read it first — every task argues from it).

## Global Constraints

- SPDX header on every new file (`// SPDX-License-Identifier: MIT`; `#` for BUILD/scripts).
- Library code MUST NOT printf/fprintf — USDT probes only (`src/support/ivcalc_trace.h`).
- No European/analytic delegation on the American call path, ever (spec hard constraint).
- Existing put behavior changes ONLY via the sweep fix; call sweep behavior must stay bit-identical (default template arg = current algorithm).
- No new compiler warnings (`-Wall -Wextra` in test targets).
- Commit messages: imperative mood, ≤50-char subject, body wrapped at 72.
- Regression tests carry the CLAUDE.md comment format (`// Regression:` / `// Bug:`).
- Verification gates before PR: `bazel test //...`, `bazel build //benchmarks/...`, `bazel build //src/python:mango_option`.

## File Structure

- `src/math/thomas_solver.hpp` — mirrored sweep, active-mask overloads, KKT validator (`LcpKktReport`, `validate_lcp_kkt`).
- `src/pde/internal/pde_solver.hpp` — `LcpActiveSide` trait wiring, validator plumbing + probe, Neumann row assembly, tag-aware BC application, event epoch hook, deletion of FD boundary probes.
- `src/pde/internal/spatial_operator.hpp` — tightened `HasJacobianCoefficients`, boundary-row methods, `HasBoundaryRows` support.
- `src/pde/operators/laplacian_pde.hpp` — coefficient accessors.
- `src/pde/core/boundary_conditions.hpp` — `NeumannBC` one-arg ctor, deprecated two-arg.
- `src/pde/core/time_domain.hpp` — exact mandatory-point landing.
- `src/pde/core/grid.hpp` — `n_points < 3` rejection in `GridSpec` factories.
- `src/option/yield_curve.hpp` — knot accessor.
- `src/option/detail/call_boundary_envelope.hpp` (new) — envelope evaluator.
- `src/option/american_option.cpp` — call right BC wrapper + epoch counter, dividend-tau merge in `resolve_grid`.
- `src/option/american_option.hpp` / result plumbing — `complementarity_report()`.
- Tests: `tests/thomas_solver_lcp_test.cc` (new), `tests/call_boundary_envelope_test.cc` (new), `tests/pde_neumann_test.cc` (new), `tests/quantlib_sweep_regression_test.cc` (new), extensions to `tests/american_option_test.cc`; BUILD entries for each.
- Docs: `docs/MATHEMATICAL_FOUNDATIONS.md` §"Projected Thomas".

Task order: 1→2 (sweep core), 3 (validator plumbing), 4 (put/call pricing
regressions + goldens), 5 (envelope evaluator), 6 (BC wiring + grid merge),
7 (operator boundary rows), 8 (Neumann solver wiring), 9 (heat tests),
10 (docs + follow-ups + gates).

---

### Task 1: Mirrored projected Thomas + active mask + KKT validator

**Files:**
- Modify: `src/math/thomas_solver.hpp` (after `solve_thomas_projected`, ~line 422)
- Create: `tests/thomas_solver_lcp_test.cc`
- Modify: `tests/BUILD.bazel` (new `cc_test` target `thomas_solver_lcp_test`)

**Interfaces:**
- Produces:
  - `enum class LcpActiveSide { Left, Right };` (namespace `mango`)
  - `template<std::floating_point T, LcpActiveSide Side = LcpActiveSide::Right> ThomasResult<T> solve_thomas_projected2(std::span<const T> lower, diag, upper, rhs, psi, std::span<T> solution, std::span<T> workspace, std::span<uint8_t> active_mask, const ThomasConfig<T>& config = {})` — `active_mask` size n, written 1 where clamped, 0 otherwise. `Side::Right` MUST reproduce the existing algorithm exactly (same FMA sequence — copy the loop bodies from `solve_thomas_projected`, only adding the mask writes). Keep the existing `solve_thomas_projected` functions untouched (other call sites: cubic splines do NOT use the projected variant — verify with grep; if truly no other callers of the *projected* span overload exist outside pde_solver, you may instead extend it in place — check first with `grep -rn "solve_thomas_projected" src/ tests/ benchmarks/`).
  - `struct LcpKktReport { size_t violation_count; double max_violation; int worst_kind; };` (worst_kind: 0=primal, 1=dual, 2=residual; max_violation = raw KKT defect)
  - `template<std::floating_point T> LcpKktReport validate_lcp_kkt(std::span<const T> lower, diag, upper, rhs, psi, std::span<const T> u, std::span<const uint8_t> active_mask, T atol = 1e-12, T rtol = 1e-10)`
- Consumed by: Task 2 (PDESolver), Task 3 (report plumbing), Task 9 (obstacle+Neumann test).

- [ ] **Step 1: Write failing tests** — create `tests/thomas_solver_lcp_test.cc` with an exhaustive-enumeration LCP reference (dense Gaussian elimination + all 2^n active sets, n ≤ 12; copy the helper from the spike archive if available, else reimplement per the code below) and these cases, each asserting full KKT (primal `u≥psi`, dual `(Au)_i ≥ rhs_i` on active, `|(Au)_i − rhs_i| ≤ tol` on inactive) against the enumerated solution:

```cpp
// SPDX-License-Identifier: MIT
// Regression: Brennan-Schwartz sweep orientation (issue #439, corrected).
// Bug: projection during right-to-left substitution is exact only for
// RIGHT-interval active sets; puts (left interval) got an inexact solve.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <span>
#include "mango/math/thomas_solver.hpp"

namespace {
// Dense LU solve with partial pivoting (reference only).
bool dense_solve(std::vector<std::vector<double>> A, std::vector<double> b,
                 std::vector<double>& x);  // as in the round-1 spike

// Exact LCP by active-set enumeration; unique for M-matrices.
bool lcp_reference(const std::vector<double>& lower, const std::vector<double>& diag,
                   const std::vector<double>& upper, const std::vector<double>& rhs,
                   const std::vector<double>& psi, std::vector<double>& x_out);

struct Sys {
    std::vector<double> lower, diag, upper, rhs, psi;
};
Sys mmatrix(size_t n, double w, std::vector<double> psi, double src = 0.02) {
    Sys s{std::vector<double>(n-1,-w), std::vector<double>(n,1+2*w),
          std::vector<double>(n-1,-w), std::vector<double>(n,src), std::move(psi)};
    return s;
}
std::vector<double> left_obstacle(size_t n)  { std::vector<double> p(n);
    for (size_t i=0;i<n;++i) p[i]=std::max(0.5-double(i)/double(n-1),0.0); return p; }
std::vector<double> right_obstacle(size_t n) { std::vector<double> p(n);
    for (size_t i=0;i<n;++i) p[i]=std::max(double(i)/double(n-1)-0.5,0.0); return p; }
}  // namespace

TEST(LcpSweep, LeftActiveExactWithLeftSweep) { /* mirrored solves left_obstacle exactly */ }
TEST(LcpSweep, RightActiveExactWithRightSweep) { /* current solves right_obstacle exactly */ }
TEST(LcpSweep, EmptyActiveSetBothSweepsMatchThomas) { /* psi = -1 everywhere */ }
TEST(LcpSweep, FullActiveSetBothSweeps) { /* psi = +10 everywhere -> u == psi */ }
TEST(LcpSweep, NonconstantObstacle) { /* sawtooth psi, still interval-active */ }
TEST(LcpSweep, IdentityLockRowsInsideActiveInterval) {
    // Convert two interior rows inside the active interval to identity
    // (lower/upper zeroed, diag=1, rhs=psi) as the deep-ITM lock does;
    // both sweeps must still be exact on their own side.
}
TEST(LcpSweep, DirichletIdentityBoundaryRows) { /* rows 0 and n-1 identity + rhs=g */ }
TEST(LcpKkt, ValidatorFlagsWrongSweepOnLeftActive) {
    // Solve left-active with Side::Right (wrong side); validate_lcp_kkt must
    // return violation_count > 0 with worst_kind == 2 (continuation residual)
    // or 1 (dual) — and the enumerated-reference solution must validate clean.
}
TEST(LcpKkt, NonFiniteInputCountsAsViolation) { /* NaN in u -> violation */ }
```

Each `/* ... */` body is real code the implementer writes following the
first two (fully worked) tests; the reference helpers make every body a
~10-line comparison. Also port the two hand-verified 2-node cases from the
issue #439 comment (ψ=(1,0) and ψ=(0,1) with A=[[2,−1],[−1,2]], rhs=0).

- [ ] **Step 2: Add BUILD target and run to verify failure**

```python
cc_test(
    name = "thomas_solver_lcp_test",
    srcs = ["thomas_solver_lcp_test.cc"],
    deps = ["//src/math:thomas_solver", "@googletest//:gtest_main"],
    copts = ["-Wall", "-Wextra"],
    size = "small",
)
```

Run: `bazel test //tests:thomas_solver_lcp_test --test_output=errors`
Expected: FAIL — `solve_thomas_projected2`, `LcpActiveSide`, `validate_lcp_kkt` undefined.

- [ ] **Step 3: Implement in `thomas_solver.hpp`.** The mirrored (Left)
algorithm — UL elimination bottom-up, projected substitution top-down —
was validated in the spike; implement exactly:

```cpp
enum class LcpActiveSide { Left, Right };

template<std::floating_point T, LcpActiveSide Side = LcpActiveSide::Right>
[[nodiscard]] constexpr ThomasResult<T> solve_thomas_projected2(
    std::span<const T> lower, std::span<const T> diag, std::span<const T> upper,
    std::span<const T> rhs, std::span<const T> psi,
    std::span<T> solution, std::span<T> workspace,
    std::span<uint8_t> active_mask, const ThomasConfig<T>& config = {}) noexcept
{
    // dimension checks identical to solve_thomas_projected, plus
    // active_mask.size() == n
    const size_t n = diag.size();
    // ... (n==0 / n==1 handling as in solve_thomas_projected, mask set)
    std::span<T> c_prime = workspace.subspan(0, n);
    std::span<T> d_prime = workspace.subspan(n, n);
    if constexpr (Side == LcpActiveSide::Right) {
        // EXACT copy of solve_thomas_projected's elimination + substitution,
        // with: active_mask[i] = (unconstrained < psi[i]) ? 1 : 0 at each
        // projection site (and mask for solution[n-1]).
    } else {
        // UL elimination from row n-1 upward; c_prime[i] couples LEFT.
        c_prime[n-1] = lower[n-2] / diag[n-1];
        d_prime[n-1] = rhs[n-1] / diag[n-1];
        for (size_t i = n - 2; i >= 1; --i) {
            const T denom = std::fma(-upper[i], c_prime[i+1], diag[i]);
            if (std::abs(denom) < config.singularity_tol)
                return ThomasResult<T>::error_result("Singular or ill-conditioned matrix");
            const T inv = T(1) / denom;
            c_prime[i] = lower[i-1] * inv;
            d_prime[i] = std::fma(-upper[i], d_prime[i+1], rhs[i]) * inv;
        }
        const T denom0 = std::fma(-upper[0], c_prime[1], diag[0]);
        if (std::abs(denom0) < config.singularity_tol)
            return ThomasResult<T>::error_result("Singular matrix (at first row)");
        d_prime[0] = std::fma(-upper[0], d_prime[1], rhs[0]) / denom0;
        // Projected substitution top-down (starts on the LEFT/active side)
        active_mask[0] = (d_prime[0] < psi[0]) ? 1 : 0;
        solution[0] = std::max(d_prime[0], psi[0]);
        for (size_t i = 1; i < n; ++i) {
            T unconstrained = std::fma(-c_prime[i], solution[i-1], d_prime[i]);
            active_mask[i] = (unconstrained < psi[i]) ? 1 : 0;
            solution[i] = std::max(unconstrained, psi[i]);
        }
    }
    return ThomasResult<T>::ok_result();
}

struct LcpKktReport {
    size_t violation_count = 0;
    double max_violation = 0.0;
    int worst_kind = -1;  // 0=primal u<psi, 1=dual (Au<rhs on active), 2=residual (inactive)
};

template<std::floating_point T>
[[nodiscard]] LcpKktReport validate_lcp_kkt(
    std::span<const T> lower, std::span<const T> diag, std::span<const T> upper,
    std::span<const T> rhs, std::span<const T> psi,
    std::span<const T> u, std::span<const uint8_t> active_mask,
    T atol = T(1e-12), T rtol = T(1e-10))
{
    LcpKktReport rep;
    const size_t n = diag.size();
    auto note = [&](double defect, int kind) {
        rep.violation_count++;
        if (defect > rep.max_violation) { rep.max_violation = defect; rep.worst_kind = kind; }
    };
    for (size_t i = 0; i < n; ++i) {
        const T lo = (i > 0) ? lower[i-1] * u[i-1] : T(0);
        const T hi = (i + 1 < n) ? upper[i] * u[i+1] : T(0);
        const T Au = lo + diag[i] * u[i] + hi;
        const T scale = std::abs(lo) + std::abs(diag[i] * u[i]) + std::abs(hi) + std::abs(rhs[i]);
        const T tol = atol + rtol * scale;
        if (!std::isfinite(Au) || !std::isfinite(u[i]) || !std::isfinite(tol)) {
            note(std::numeric_limits<double>::infinity(), 2);
            continue;
        }
        if (u[i] < psi[i] - tol) note(double(psi[i] - u[i]), 0);
        if (active_mask[i]) { if (Au < rhs[i] - tol) note(double(rhs[i] - Au), 1); }
        else                { if (std::abs(Au - rhs[i]) > tol) note(std::abs(double(Au - rhs[i])), 2); }
    }
    return rep;
}
```

Also add a `TridiagonalMatrixView` convenience overload of
`solve_thomas_projected2` and of `validate_lcp_kkt` mirroring the existing
view overload at the bottom of the file.

- [ ] **Step 4: Run** `bazel test //tests:thomas_solver_lcp_test --test_output=errors` — Expected: PASS.
- [ ] **Step 5: Commit** — `git add src/math/thomas_solver.hpp tests/thomas_solver_lcp_test.cc tests/BUILD.bazel && git commit -m "Add oriented projected Thomas and full-KKT LCP validator"`

---

### Task 2: CRTP orientation trait + PDESolver wiring

**Files:**
- Modify: `src/pde/internal/pde_solver.hpp` (`solve_implicit_stage_projected`, ~line 715 call site; workspace mask storage)
- Modify: `src/pde/internal/pde_workspace.hpp` (active-mask span — check `required_size`; if adding a span is invasive, a `std::array`-free fixed member is wrong — add a `uint8_t` PMR-compatible span the same way existing spans are declared, updating `required_size(n)`)
- Modify: `src/option/american_option.cpp` (traits on both solvers)
- Test: existing suites must stay green; put-price movement is asserted in Task 4

**Interfaces:**
- Consumes: `LcpActiveSide`, `solve_thomas_projected2`, `validate_lcp_kkt` (Task 1).
- Produces:
  - Derived-solver trait: `static constexpr LcpActiveSide lcp_active_side;` — `AmericanPutSolver::lcp_active_side = LcpActiveSide::Left;`, `AmericanCallSolver::lcp_active_side = LcpActiveSide::Right;`
  - Concept in `pde_solver.hpp`: `template<typename D> concept HasLcpActiveSide = requires { { D::lcp_active_side } -> std::convertible_to<LcpActiveSide>; };` with `static_assert(HasLcpActiveSide<Derived>)` inside `solve_implicit_stage_projected` (only obstacle solvers reach it).
  - `PDESolver::lcp_report_` member (`LcpKktReport`, aggregated: counts summed, max kept) + `const LcpKktReport& lcp_report() const` + reset in `solve()` before the time loop.

- [ ] **Step 1:** In `solve_implicit_stage_projected`, replace the `solve_thomas_projected` call with:

```cpp
constexpr LcpActiveSide side = Derived::lcp_active_side;
auto active_mask = workspace_.active_mask();  // new uint8_t span, size n
auto result = solve_thomas_projected2<double, side>(
    workspace_.jacobian(), rhs_with_bc, psi, u,
    workspace_.tridiag_workspace(), active_mask);
```

then AFTER the solve (and after the existing `apply_boundary_conditions(u, t)` line — validation runs on the exact system solved, so run it BEFORE that re-application, immediately after the solve, using `rhs_with_bc` and the jacobian while untouched):

```cpp
auto stage_rep = validate_lcp_kkt<double>(
    workspace_.jacobian().lower(), workspace_.jacobian().diag(),
    workspace_.jacobian().upper(), rhs_with_bc, psi, u, active_mask);
lcp_report_.violation_count += stage_rep.violation_count;
if (stage_rep.max_violation > lcp_report_.max_violation) {
    lcp_report_.max_violation = stage_rep.max_violation;
    lcp_report_.worst_kind = stage_rep.worst_kind;
}
if (stage_rep.violation_count > 0) {
    MANGO_TRACE_ALGO_PROGRESS(MODULE_GRID_PROBE,
        static_cast<int64_t>(stage_rep.violation_count), 0,
        stage_rep.max_violation);
}
```

(Use the existing probe macros from `src/support/ivcalc_trace.h`; if a
better-fitting probe exists — check the header's list — prefer it, else add
`MANGO_TRACE_LCP_KKT(count, max_violation)` alongside the existing macro
definitions in that header.)

- [ ] **Step 2:** Add the `uint8_t` active-mask span to `PDEWorkspace` following the pattern of the existing double spans (adjust `required_size`; `uint8_t` storage may be carved from a double-aligned block — n bytes rounded up — mirror how `tridiag_workspace` is laid out and update `AmericanPDEWorkspace::from_bytes` accounting).
- [ ] **Step 3:** Add the trait constants to both American solvers; reset `lcp_report_ = {}` at the top of `PDESolver::solve()`.
- [ ] **Step 4:** Run `bazel test //tests:american_option_test //tests:pde_solver_test //tests:thomas_solver_lcp_test --test_output=errors`. Expected: PASS, EXCEPT any test pinning exact put prices — record failures; they are addressed in Task 4 (do not loosen anything yet; if a failure is not a put-price pin, debug before proceeding).
- [ ] **Step 5: Commit** — `git commit -am "Select LCP sweep orientation per option type"` (body: explains put→Left fix, validator plumbing, cites #439 and the spike numbers).

---

### Task 3: Public complementarity report on AmericanOptionSolver

**Files:**
- Modify: `src/option/american_option.hpp` (public accessor + member), `src/option/american_option.cpp` (`solve()` copies the report out of the variant before destruction)

**Interfaces:**
- Produces: `const LcpKktReport& AmericanOptionSolver::complementarity_report() const;` — valid after `solve()` (zeroed before; kept from an aborted solve). Include `mango/math/thomas_solver.hpp` for the type (it already reaches the hpp transitively — verify; else forward-declare a small mirror struct in the public header and copy fields).

- [ ] **Step 1:** In `AmericanOptionSolver::solve()`, inside the `std::visit` after the PDE `solve()` call (success or failure), copy `pde_solver.lcp_report()` into `lcp_report_` member.
- [ ] **Step 2:** Unit test (append to `tests/american_option_test.cc`): ATM put solve → `complementarity_report().violation_count == 0` (M-matrix regime must validate clean end-to-end — this is the strongest single assertion in the whole batch that the new put sweep is exact).
- [ ] **Step 3:** Run `bazel test //tests:american_option_test --test_output=errors` — PASS. Commit `"Expose per-solve LCP complementarity report"`.

---

### Task 4: Pricing regressions + golden regeneration (T2/T3 continuous part)

**Files:**
- Create: `tests/quantlib_sweep_regression_test.cc` (+ BUILD target, `linkopts = ["-lQuantLib"]`, `size = "large"`)
- Regenerate: goldens in `tests/bspline_bit_identity_test` (follow the regeneration procedure documented in that test file / its BUILD comments)

**Interfaces:** consumes only public pricing API.

- [ ] **Step 1:** Write the test file: helper `ql_american(spot, strike, maturity, vol, rate, q, is_call)` using `FdBlackScholesVanillaEngine(process, 8000, 801)`, `Actual365Fixed()` — copy verbatim from the spike archive (`sweep_spike_test.cc.keep` in the session scratchpad) or re-derive from `tests/quantlib_validation_framework.hpp`. Scenarios and thresholds (absolute error vs the 8000×801 reference; from the spec):

```cpp
// Regression: put sweep orientation (#439). Old sweep ATM error was 6.9e-3.
struct Row { const char* name; double S,K,T,vol,r,q; bool call; double max_abs_err; };
static const Row kRows[] = {
  {"put ATM",        100,100,1.0, .20,.05,.00,false, 5.5e-3},
  {"put ITM S90",     90,100,1.0, .20,.05,.00,false, 3.0e-3},
  {"put ITM S80",     80,100,1.0, .20,.05,.00,false, 1.0e-3},
  {"put deep S70",    70,100,1.0, .20,.05,.00,false, 5.0e-4},
  {"put nearFB r8",   85,100,1.0, .20,.08,.00,false, 6.5e-3},
  {"put OTM T.25",   110,100,0.25,.30,.05,.00,false, 5.0e-3},
  {"put T2 v25",      90,100,2.0, .25,.05,.00,false, 3.0e-3},
  {"put q2",         100,100,1.0, .20,.05,.02,false, 6.0e-3},
  {"call ATM q8",    100,100,1.0, .20,.05,.08,true,  7.5e-3},
  {"call S120 q8",   120,100,1.0, .20,.05,.08,true,  7.5e-3},
  {"call S130 q8",   130,100,1.0, .20,.05,.08,true,  7.5e-3},
  {"call S150 q8",   150,100,1.0, .20,.05,.08,true,  7.5e-3},
  {"call S200 q8",   200,100,1.0, .20,.05,.08,true,  7.5e-3},
  {"call S300 q8",   300,100,1.0, .20,.05,.08,true,  7.5e-3},
  {"call S120 q4r2", 120,100,2.0, .25,.02,.04,true,  7.5e-3},
};
```

The put thresholds tighten on measured mirrored-sweep errors ×~1.25 (spike
table); if any measured error at implementation exceeds its row, STOP and
investigate before loosening — the spike numbers say they fit.
- [ ] **Step 2:** Run new target — expected PASS (sweep fix landed in Task 2). If ATM put fails at 5.5e-3, the orientation wiring is wrong — debug (check trait constants first), do not loosen.
- [ ] **Step 3:** Run `bazel test //... --test_output=errors 2>&1 | grep -E "FAIL|bit_identity"`. Regenerate the bit-identity goldens per that test's documented procedure; re-run to green. Any OTHER failing test: fix-forward only if it pins old put prices; anything else — investigate.
- [ ] **Step 4: Commit** — `"Pin put/call pricing accuracy after sweep fix"` (body notes golden regeneration and why prices moved).

---

### Task 5: Call boundary envelope evaluator (T4 core)

**Files:**
- Create: `src/option/detail/call_boundary_envelope.hpp` (+ `src/option/BUILD.bazel` header-only lib `call_boundary_envelope`, visibility like the existing `detail` libs — see `price_table_error_mapping` for the pattern)
- Modify: `src/option/yield_curve.hpp` — add `std::span<const TenorPoint> points() const { return curve_; }`
- Create: `tests/call_boundary_envelope_test.cc` (+ BUILD target)

**Interfaces:**
- Produces (namespace `mango::detail`):

```cpp
struct CallBoundaryEnvelope {
    double x_max;             // log-moneyness at the right boundary
    double dividend_yield;    // q
    double maturity;          // T
    RateSpec rate;
    std::vector<Dividend> dividends;  // filter_and_merge_dividends output, calendar-ascending

    // Value at solver (backward) time tau, with n_events_applied dividends
    // already crossed by temporal events (phase source of truth; the last
    // n_events_applied entries of `dividends` — latest calendar first-crossed
    // — are the REMAINING ones). Pass SIZE_MAX to derive remaining purely
    // from time (tau_i < tau strictly) — used by tests.
    double value(double tau, size_t n_events_applied) const;
};
```

- Consumed by: Task 6 (`RightBCFunction`).

- [ ] **Step 1: Failing tests.** In `tests/call_boundary_envelope_test.cc` (deps: the new lib + gtest; no QuantLib):

```cpp
// each test constructs CallBoundaryEnvelope directly
TEST(Envelope, NoDivZeroQReducesToForwardDiscount)  // == max(e^x-1, e^x - e^{-r*tau}); flat r=5%
TEST(Envelope, TauZeroIsIntrinsic)                  // value(0, 0) == e^x - 1 (within 1e-15)
TEST(Envelope, StrictPhasePredicate)                // tau just below/above a tau_j with SIZE_MAX
TEST(Envelope, EpochCounterOverridesTime)           // at tau == tau_j: n=0 excludes, n=1 includes
TEST(Envelope, IntermediateExDateDominates)         // 2 divs; brute-force scan of f(s) agrees
TEST(Envelope, FlatRateInteriorStationaryPoint)     // q=1%, r=8%, x small: interior max; scan agrees
TEST(Envelope, YieldCurveKnotsRespected)            // curve with knots at 0.25/0.5; scan agrees
TEST(Envelope, CombinedContinuousAndDiscrete)       // q=2% + 1 div; scan agrees
```

The brute-force oracle (test-local): dense scan of

```cpp
double f(double s) {  // stopping value at backward time s <= tau
    double sum = 0.0;
    for (remaining div i with tau_i > s) sum += (D_i/K==already normalized amount)
        * DF(tau, tau_i) * std::exp(-q * (tau_i - s));
    return std::exp(x) * std::exp(-q * (tau - s)) - sum - DF(tau, s);
}
// max over s in linspace(0, tau, 20001); envelope must match within 1e-9
```

where `DF(a, b) = D_cal(T-b) / D_cal(T-a)` via `YieldCurve::discount` (flat
curves via `YieldCurve::flat(r)`). NOTE: `Dividend::amount` is a cash
amount; the envelope stores amounts already divided by strike — pin this:
`CallBoundaryEnvelope.dividends[i].amount` is interpreted as D_i/K, and
Task 6 does the division when constructing.

- [ ] **Step 2:** BUILD + run → FAIL (header missing).
- [ ] **Step 3: Implement.** Candidate enumeration inside `value(tau, n_applied)`:

```cpp
double value(double tau, size_t n_applied) const {
    const double A = std::exp(x_max);
    // remaining dividends -> backward times, descending calendar handled here
    std::vector<double> taus;  // tau_i for remaining divs, plus amounts
    // (SIZE_MAX: remaining = {i : T - d_i.calendar_time < tau};
    //  else: remaining = last n_applied entries of `dividends`)
    ...
    auto DF = [&](double a, double b) { ... via rate spec ... };
    // candidates: s = tau (now), s = 0 (expiry), each remaining tau_i
    // (evaluated as the sup over s -> tau_i^+ : include divs with tau_i > s,
    //  i.e. exercise just before the ex-date), each curve knot mapped to
    // backward time s = maturity - tenor when 0 < s < tau, and per smooth
    // segment the closed-form stationary point:
    //   within a segment: f(s) = B * e^{q s} - C * e^{r_f s}
    //   with B = A e^{-q tau} - sum_over_remaining_before_segment(...),
    //        C from DF(tau, s) = C e^{r_f s} on the flat-forward segment;
    //   s* = ln((q*B)/(r_f*C)) / (r_f - q), keep iff inside segment and all
    //   of q, r_f, B, C positive and r_f != q (else endpoints suffice).
    // return the max of f over all kept candidates, floored by intrinsic A-1.
}
```

The segment walk: breakpoints = sorted {0, tau, remaining tau_i, mapped
curve knots}; between consecutive breakpoints the remaining-div set and
forward rate are constant, so B and C are constants computable from the
segment's right end. Constant-rate `RateSpec` is one flat segment.
Implement `f(s)` once and reuse for candidate evaluation — the same code
path the test oracle uses conceptually, but with exact candidates instead
of a scan.

- [ ] **Step 4:** Run → PASS (the two scan-comparison tests are the proof the candidate set is complete; if a scan beats the envelope by >1e-9, a candidate class is missing — check knot mapping `s = T − tenor` first).
- [ ] **Step 5: Commit** — `"Add deep-ITM call boundary stopping envelope"`.

---

### Task 6: Wire call right BC + event phase + dividend-tau grid merge (B3/B4/B5)

**Files:**
- Modify: `src/option/american_option.cpp` — `RightBCFunction`, `init_dividend_events`, `resolve_grid`
- Modify: `src/pde/core/time_domain.hpp` — exact segment landing
- Test: extend `tests/american_option_test.cc`

**Interfaces:**
- Consumes: `CallBoundaryEnvelope` (Task 5).
- Produces: no new public API. `AmericanCallSolver` gains `size_t n_events_applied_ = 0;` and its `RightBCFunction` holds `const CallBoundaryEnvelope* env; const size_t* n_applied;` evaluating `env->value(t, *n_applied)`.

- [ ] **Step 1: Failing regression tests** (append to `tests/american_option_test.cc`):

```cpp
// Regression: custom time grids must include dividend taus (#439 batch, B5)
// Bug: process_temporal_events fires an event only at completed steps, so a
// custom grid omitting the ex-date applied the jump at the wrong state time.
TEST(AmericanOptionTest, CustomGridOmittingDividendDateStillAligns) {
    // PUT, one dividend at calendar 0.5, T=1. Price with (a) auto grid,
    // (b) PDEGridConfig{n_time=100, mandatory_times={}} — same n_time scale.
    // Assert |price_a - price_b| < 5e-3 (was grossly larger with misaligned
    // events on a coarse custom grid — record the observed broken delta in
    // the test comment at implementation time).
}
// Regression: dividend-free call right BC unchanged (#439 item 2 guard)
TEST(AmericanOptionTest, NoDivCallPriceUnchangedByEnvelopeBC) {
    // r=5%, q=0, no divs: price ATM call before/after must match the pinned
    // value from current main to 1e-12 (pin the number at implementation).
}
// Regression: dividend-paying call right BC no longer pinned high (#439 item 2)
TEST(AmericanOptionTest, DiscreteDivCallRightBoundaryEnvelope) {
    // CALL q=0, r=5%, div D=1.5 at 0.25, T=1: envelope at tau just above
    // tau_d subtracts the div PV; direct CallBoundaryEnvelope check plus a
    // full solve that must stay within the Task-4 QuantLib tolerance class.
}
```

- [ ] **Step 2:** Run → FAIL (custom-grid case), FAIL/compile as appropriate.
- [ ] **Step 3: Implement.**
  1. `resolve_grid`: in the `PDEGridConfig` branch, always build the merged list:

```cpp
[&](const PDEGridConfig& eg) {
    std::vector<double> mand = eg.mandatory_times;
    for (const auto& d : filter_and_merge_dividends(params.discrete_dividends,
                                                    params.maturity)) {
        mand.push_back(params.maturity - d.calendar_time);  // tau
    }
    auto td = mand.empty()
        ? TimeDomain::from_n_steps(0.0, params.maturity, eg.n_time)
        : TimeDomain::with_mandatory_points(0.0, params.maturity,
              params.maturity / static_cast<double>(eg.n_time), mand);
    return std::make_pair(eg.grid_spec, td);
}
```

  2. `TimeDomain::with_mandatory_points`: after each segment's subdivision loop, force the final point: `points.back() = seg_end;` (read the loop at `time_domain.hpp:88-100` and set the last emitted sub-point of every segment to the exact boundary rather than `seg_start + j*sub_dt`).
  3. `AmericanCallSolver`: member `CallBoundaryEnvelope envelope_;` built in the constructor (amounts divided by `params.strike`; dividends via `filter_and_merge_dividends`), member `size_t n_events_applied_ = 0;`. `RightBCFunction{&envelope_, &n_events_applied_}` with `double operator()(double t, double) const { return env->value(t, *n_applied); }`. IMPORTANT lifetime: like `dividend_spline_`, the BC captures `this`-relative pointers — wire them in `init_dividends()` (post-variant-placement), not the constructor: keep the constructor storing a null-env `RightBCFunction` and have `init_dividends()` rebuild `right_bc_ = DirichletBC(RightBCFunction{&envelope_, &n_events_applied_});`.
  4. `init_dividend_events` (call path only): wrap each registered callback so it increments the counter AFTER the jump: add an optional `size_t* counter` parameter (puts pass nullptr), and in the lambda `if (counter) ++*counter;` before returning — the post-event `apply_boundary_conditions(u, event.time)` in `process_temporal_events` then evaluates the right BC with the epoch already advanced → pre-dividend side, per spec B3. Reset `n_events_applied_ = 0` wherever `initialize()` is called for a fresh run (the CRTP `initialize` rewinds `next_event_idx_` — mirror that by resetting the counter in `init_dividends()` and having the event wrapper's counter reset handled at `initialize` via a small virtual-free hook: simplest is to reset in `AmericanOptionSolver::solve()` before `initialize`, where the variant is freshly constructed anyway — verify the variant is per-solve (it is: constructed inside `solve()`), so a constructor init to 0 suffices).
- [ ] **Step 4:** Run the three tests + `bazel test //tests:american_option_test //tests:discrete_dividend_event_test //tests:discrete_dividend_accuracy_test --test_output=errors` → PASS.
- [ ] **Step 5: Commit** — `"Make call right boundary dividend-aware via stopping envelope"`.

---

### Task 7: SpatialOperator boundary rows + concept tightening

**Files:**
- Modify: `src/pde/internal/spatial_operator.hpp`, `src/pde/operators/laplacian_pde.hpp`
- Create: `tests/spatial_operator_boundary_test.cc` (+ BUILD)

**Interfaces:**
- Produces (on `SpatialOperator`, each `requires HasJacobianCoefficients<PDE>`):

```cpp
struct BoundaryRowJacobian { double diag; double offdiag; };
double eval_boundary_row(double t, bc::BoundarySide side, double g,
                         std::span<const T> u) const;
BoundaryRowJacobian boundary_row_jacobian(double t, bc::BoundarySide side) const;
double boundary_row_affine(double t, bc::BoundarySide side, double g) const;
```

  and the tightened concept:

```cpp
template<typename PDE>
concept HasJacobianCoefficients = requires(const PDE pde, double t) {
    { pde.second_derivative_coeff() } -> std::convertible_to<double>;
    { pde.first_derivative_coeff(t) } -> std::convertible_to<double>;
    { pde.discount_rate(t) } -> std::convertible_to<double>;
};
```

  plus, in `pde_solver.hpp` (Task 8 consumes): `template<typename Op> concept HasBoundaryRows = requires(const Op op, double t, bc::BoundarySide s, double g, std::span<const double> u) { { op.eval_boundary_row(t, s, g, u) } -> std::convertible_to<double>; { op.boundary_row_jacobian(t, s) }; { op.boundary_row_affine(t, s, g) } -> std::convertible_to<double>; };`
- `LaplacianPDE` gains: `T second_derivative_coeff() const { return D_; }`, `T first_derivative_coeff([[maybe_unused]] double t = 0.0) const { return T(0); }`, `T discount_rate([[maybe_unused]] double t = 0.0) const { return T(0); }`.

- [ ] **Step 1: Failing tests** in `tests/spatial_operator_boundary_test.cc` — direct algebra per spec C1 (nonzero a,b,c,g both sides + nonuniform grid):

```cpp
// Left:  L0 = (2a/h^2)(u1-u0) + c*u0 + g*(b - 2a/h),  h = dx[0]
// Right: Ln = (2a/h^2)(u_{n-2}-u_{n-1}) + c*u_{n-1} + g*(b + 2a/h), h = dx[n-2]
TEST(BoundaryRow, LeftClosedFormBlackScholes)   // sigma=.3, r=.05, q=.02, g=1.7
TEST(BoundaryRow, RightClosedFormBlackScholes)
TEST(BoundaryRow, EvalEqualsJacobianDotUPlusAffine)  // both sides, random u
TEST(BoundaryRow, NonuniformGridUsesAdjacentSpacing) // geometric grid
TEST(BoundaryRow, LaplacianSatisfiesConcept)         // static_assert HasJacobianCoefficients
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement** (implementation is exactly the closed forms; `eval_boundary_row = jac.diag*u[node] + jac.offdiag*u[neighbor] + affine` — implement `eval` in terms of the other two so the identity test is true by construction, and hand-check the closed-form tests against independently coded expectations). Tighten the concept; fix Laplacian. Grep for other `HasJacobianCoefficients` users (`assemble_jacobian` only) — the BS PDE already has defaulted-arg accessors satisfying the tightened form.
- [ ] **Step 4:** Run new + `bazel build //src/...` → green. **Step 5: Commit** `"Add analytic ghost-eliminated boundary rows to SpatialOperator"`.

---

### Task 8: PDESolver Neumann wiring + BC-type guard + NeumannBC API + n>=3

**Files:**
- Modify: `src/pde/internal/pde_solver.hpp` (`apply_spatial_operator`, `build_jacobian_boundaries`, `apply_boundary_conditions`, `solve_implicit_stage_projected` RHS, `initialize`)
- Modify: `src/pde/core/boundary_conditions.hpp` (`NeumannBC` ctors)
- Modify: `src/pde/core/grid.hpp` (`n_points < 3` in the three `GridSpec` factories — FIRST verify `grep -rn "n_points" src/ tests/ | grep -E "\b2\b"` finds no legitimate 2-point PDE user; B-spline/table grids use different types)
- Modify: `tests/boundary_conditions_test.cc` (migrate to 1-arg ctor; keep ONE deprecated-ctor compile check under `#pragma GCC diagnostic ignored "-Wdeprecated-declarations"`)

**Interfaces:**
- Consumes: `HasBoundaryRows`, boundary-row methods (Task 7).
- Produces: Neumann rows genuinely solved on Newton and projected paths; `NeumannBC(Func)` ctor; compile-time rejection for unsupported BC/operator combos.

- [ ] **Step 1:** `apply_spatial_operator` becomes BC-aware:

```cpp
void apply_spatial_operator(double t, std::span<const double> u, std::span<double> Lu) {
    const size_t n = grid_->x().size();
    const auto& spatial_op = derived().spatial_operator();
    spatial_op.apply(t, u, Lu);
    using L = std::remove_cvref_t<decltype(derived().left_boundary())>;
    using R = std::remove_cvref_t<decltype(derived().right_boundary())>;
    if constexpr (std::is_same_v<bc::boundary_tag_t<L>, bc::neumann_tag>) {
        if constexpr (HasBoundaryRows<std::remove_cvref_t<decltype(spatial_op)>>) {
            double g = derived().left_boundary().gradient(t, grid_->x()[0]);
            Lu[0] = spatial_op.eval_boundary_row(t, bc::BoundarySide::Left, g, u);
        } else static_assert(dependent_false_v<L>,
            "Neumann left BC requires a spatial operator with boundary rows "
            "(HasBoundaryRows): the PDE must expose a/b/c coefficients");
    } else if constexpr (std::is_same_v<bc::boundary_tag_t<L>, bc::dirichlet_tag>) {
        Lu[0] = 0.0;
    } else static_assert(dependent_false_v<L>,
        "Unsupported left boundary condition type (Robin rows are not assembled)");
    // ... mirrored for R / Lu[n-1] ...
}
```

with `template<typename> inline constexpr bool dependent_false_v = false;`
near the concepts. Same `if constexpr / else static_assert` structure in
`build_jacobian_boundaries` (analytic rows: `jac.diag()[0] = 1.0 −
coeff_dt*brj.diag; jac.upper()[0] = −coeff_dt*brj.offdiag;` — deleting the
FD probe blocks) and in `apply_bc_to_residual` (Dirichlet branches stay;
Neumann needs NO override once `Lu` is real — leave a comment saying so).
- [ ] **Step 2:** `apply_boundary_conditions` → Dirichlet-only (per side, `if constexpr` on tag; other tags no-op). `initialize()` unchanged in call order (its BC application now touches only Dirichlet sides).
- [ ] **Step 3:** Projected path: after the Dirichlet `rhs_with_bc` overrides, add for each Neumann side `rhs_with_bc[row] += coeff_dt * spatial_op.boundary_row_affine(t, side, g);` — note the sign: the stage equation is `(I − w·∂L/∂u)·u = rhs + w·affine` since `L(u) = jac·u + affine` at the row.
- [ ] **Step 4:** `NeumannBC`: add `explicit NeumannBC(Func f) : func_(std::move(f)) {}`, mark the two-arg ctor `[[deprecated("diffusion coefficient is unused; use NeumannBC(func)")]]`, delete the `diffusion_coeff_` member and `diffusion_coeff()` accessor (the deprecated ctor ignores its second arg). Migrate `tests/boundary_conditions_test.cc` uses. Grep the whole repo for `NeumannBC(` to catch stragglers.
- [ ] **Step 5:** `GridSpec` factories: raise `n_points < 2` → `< 3` with message `"Grid needs >= 3 points (3-point stencil)"` in `uniform`, `log_spaced`, `sinh_spaced` (after the Step-0 grep proves no 2-point user).
- [ ] **Step 6:** Run `bazel test //... --test_output=errors` — everything must stay green (Neumann paths have no in-repo runtime user yet; this step proves no Dirichlet regression). Commit `"Restore ghost-point Neumann boundary rows in PDESolver"`.

---

### Task 9: Neumann validation suite (T5)

**Files:**
- Create: `tests/pde_neumann_test.cc` (+ BUILD; deps: pde core/internal/operators, gtest; no QuantLib)

A minimal CRTP heat solver for tests (in the test file):

```cpp
class HeatNeumannSolver : public mango::PDESolver<HeatNeumannSolver> {
    // LaplacianPDE spatial op via create_spatial_operator, NeumannBC both
    // sides with configurable g_left(t), g_right(t); no obstacle.
    // (Follow AmericanPutSolver's structure minus obstacle/payoff.)
};
```

- [ ] **Step 1: Mass conservation (regression for archived issue #6).**

```cpp
// Regression: Neumann boundary rows must evolve via the PDE (ghost point),
// not algebraic constraints. Bug: identity rows caused ~2% mass drift (C-era
// issue #6); the C++ migration regressed to lagged post-hoc application (#455).
TEST(PDENeumann, ZeroFluxMassConservedToRoundoff) {
    // D=0.1, x in [0,1], n=101, dt=0.01, 100 steps, gaussian exp(-50(x-.5)^2)
    // trapezoidal mass: |M_final/M_0 - 1| < 1e-10
}
```

- [ ] **Step 2: Convergence order with inhomogeneous data.** Manufactured
`u(x,t) = e^{-D k^2 t} sin(k x + 0.3)`, `k = 2.1` on `[0,1]` (nonzero u''' at
both ends), `g_side(t) = du/dx` at the boundary from the formula, initial
condition from `t=0`. Grids n ∈ {41, 81, 161, 321} with `dt ∝ dx` (TR-BDF2
is 2nd order in time); fit `log2(err_n / err_2n)` on the final-time max-norm
against the exact solution; assert fitted order `>= 1.8`. Also run one
geometric (nonuniform) grid refinement pair and assert the error still
shrinks by `>= 3x` per halving.
- [ ] **Step 3: Newton-path health.** Same solver, assert `solve()` succeeds
and the boundary row is genuinely implicit: price two consecutive runs with
`dt` and `dt/2`; the boundary-node error must scale down ~4x (lagged
first-order treatment scales ~2x — this is the discriminator that fails on
the OLD code; note it in the test comment).
- [ ] **Step 4: Obstacle+Neumann affine term** at the linear-solve level:
build a small system through `solve_thomas_projected2` with a Neumann-style
row assembled per Task 8's formulas and nonzero g, compare against the
enumeration reference from Task 1's helpers (include them via a small
shared test header `tests/lcp_test_util.hpp` extracted in this task —
refactor Task 1's test to use it too).
- [ ] **Step 5: Grid floor.** `GridSpec::uniform(0.0, 1.0, 2)` →
`std::unexpected`; `n=3` solve runs.
- [ ] **Step 6:** Run target → PASS; commit `"Validate restored Neumann treatment (mass, order, affine)"`.

---

### Task 10: Docs, follow-up issues, full gates

**Files:**
- Modify: `docs/MATHEMATICAL_FOUNDATIONS.md` §"Projected Thomas Algorithm" + "Why This Works"
- Modify: `CLAUDE.md` only if API guidance changed (none expected)

- [ ] **Step 1:** Rewrite "Why This Works": substitution must START on the
active side; put → left-start (UL), call → right-start (LU); the old
argument proved clamp propagation into the active set but not
continuation-side complementarity (cite #439 comment + spike numbers);
document the full-KKT validator and the M-matrix caveat (high cell Péclet)
with pointers to the follow-up issues.
- [ ] **Step 2:** File follow-up issues with `gh issue create`:
  1. "Drift upwinding: enforce monotone (M-matrix) discretization" — cite
     the σ=1%, h=0.1 counterexample and the validator hook.
  2. "Iterative LCP fallback (PSOR) for non-interval active sets" — cite
     r<0 regimes and the validator report.
- [ ] **Step 3: Full gates** (must all pass before the PR):

```bash
bazel test //... --test_output=errors
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

- [ ] **Step 4:** Commit docs (`"Correct Brennan-Schwartz orientation docs"`),
then push and open the PR per CLAUDE.md's workflow (summary must cover: put
sweep fix + measured price movement + golden regeneration; call BC
envelope; Neumann restoration; new diagnostics; issues #439/#455 linked,
follow-ups referenced). Pre-merge review happens on the open PR (spec-driven
-delivery gate 2).

---

## Self-review notes (already applied)

- Spec coverage: A1/A2 → Tasks 1–2; A3 → Task 1 lock-row test; A4 → Tasks
  1–3; A5 → Task 10; B1–B4 → Tasks 5–6; B5 → Task 6; C1–C5 → Tasks 7–8;
  T1–T5 → Tasks 1, 4, 5, 6, 9; T6 → Task 10.
- Type consistency: `LcpActiveSide`, `solve_thomas_projected2`,
  `LcpKktReport`, `validate_lcp_kkt`, `CallBoundaryEnvelope::value(tau,
  n_events_applied)`, `BoundaryRowJacobian`, `eval_boundary_row`,
  `boundary_row_jacobian`, `boundary_row_affine`, `HasBoundaryRows`,
  `lcp_active_side` — single spelling throughout; keep them verbatim.
- Known judgment points left to executors ON PURPOSE, with guardrails:
  workspace layout for the active mask (Task 2 step 2 names the pattern),
  probe macro choice (Task 2 step 1 names the fallback), exact pinned
  prices in Task 6's no-div guard (measured at implementation, then frozen).
