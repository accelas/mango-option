# Latent Trap Fixes (#441) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix ten latent traps from issue #441 (items 1–3, 5–11) as independent, regression-tested changes on one branch.

**Architecture:** Each task is a self-contained fix + regression test in the CLAUDE.md `// Regression:` / `// Bug:` format. No new subsystems; one new internal header/target for the factory error mapper (task 10). Behavior changes are confined to unused/latent paths.

**Tech Stack:** C++23, Bazel/Bzlmod, GoogleTest, LAPACKE.

**Spec:** `docs/plans/2026-08-29-latent-traps-441-design.md` — read it first; it carries the rationale, the decisions (including two recorded reviewer disagreements), and the exact contracts.

## Global Constraints

- Every new file starts with `// SPDX-License-Identifier: MIT` (or `#` form for scripts/BUILD).
- The repo compiles with `-Wall -Wextra -Werror`; an unhandled enum value in an exhaustive switch is a build failure — that is intentional in task 10.
- Regression tests follow CLAUDE.md format: `// Regression: <what went wrong>` + `// Bug: <root cause>` above the TEST.
- Commit messages: imperative mood, ≤50-char subject, body wrapped at 72.
- Line numbers below are from main @ `b87a6b65`; verify against the actual file before editing — drift is expected, constructs are authoritative.
- Do not touch item 4 (Neumann Jacobian boundary rows) — it is split into a separate issue.

---

### Task 1: TimeDomain dt-constructor overshoots t_end

**Files:**
- Modify: `src/pde/core/time_domain.hpp:17-27`
- Test: `tests/time_domain_test.cc`

**Interfaces:**
- Produces: `TimeDomain(t_start, t_end, dt)` where stored `dt() <= requested dt` and `time_points().back() ≈ t_end`.

- [ ] **Step 1: Write the failing tests** (append to `tests/time_domain_test.cc`)

```cpp
// ===========================================================================
// Regression tests for bugs found during code review (issue #441)
// ===========================================================================

// Regression: TimeDomain(t_start, t_end, dt) must not overshoot t_end
// Bug: n_steps was ceil'd but dt kept at the requested value, so
//      time_points() ended at t_start + n_steps*dt > t_end
//      (e.g. (0, 1, 0.3) -> last point 1.2, integrating past maturity).
TEST(TimeDomainTest, DtCtorLandsOnTEnd) {
    mango::TimeDomain td(0.0, 1.0, 0.3);
    EXPECT_EQ(td.n_steps(), 4u);          // ceil(1.0 / 0.3)
    EXPECT_DOUBLE_EQ(td.dt(), 0.25);      // shrunk from 0.3
    auto pts = td.time_points();
    ASSERT_EQ(pts.size(), 5u);
    EXPECT_NEAR(pts.back(), 1.0, 1e-12);  // tolerance, not bit-exact
}

// Regression: dt larger than the whole span must still produce one step
// Bug: same root cause; ceil(span/dt) < 1 would make an empty time grid.
TEST(TimeDomainTest, DtCtorDtLargerThanSpan) {
    mango::TimeDomain td(0.0, 0.1, 1.0);
    EXPECT_EQ(td.n_steps(), 1u);
    EXPECT_DOUBLE_EQ(td.dt(), 0.1);
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `bazel test //tests:time_domain_test --test_output=all`
Expected: `DtCtorLandsOnTEnd` FAILS (dt stays 0.3, last point 1.2).

- [ ] **Step 3: Implement** — replace the constructor in `src/pde/core/time_domain.hpp`:

```cpp
    /// Construct time domain from a maximum time step size
    ///
    /// n_steps = ceil((t_end - t_start) / dt) and dt is then shrunk to
    /// (t_end - t_start) / n_steps, so the grid lands on t_end (within
    /// floating-point rounding) and the actual step never exceeds the
    /// requested dt.  Same semantics as from_n_steps.
    ///
    /// Preconditions (documented, not validated): finite inputs,
    /// t_end > t_start, dt > 0.
    ///
    /// @param t_start Initial time
    /// @param t_end Final time
    /// @param dt Maximum time step size
    TimeDomain(double t_start, double t_end, double dt)
        : t_start_(t_start)
        , t_end_(t_end)
    {
        n_steps_ = std::max<size_t>(
            1, static_cast<size_t>(std::ceil((t_end - t_start) / dt)));
        dt_ = (t_end - t_start) / static_cast<double>(n_steps_);
    }
```

(`<algorithm>` and `<cmath>` are already included.)

- [ ] **Step 4: Run tests to verify pass**

Run: `bazel test //tests:time_domain_test --test_output=all`
Expected: all PASS (the existing `BasicConfiguration`/`TimePointGeneration` tests use exact multiples and still pass).

- [ ] **Step 5: Commit**

```bash
git add src/pde/core/time_domain.hpp tests/time_domain_test.cc
git commit -m "Shrink dt so TimeDomain dt-ctor lands on t_end"
```

---

### Task 2: Thomas solvers reject valid n==0 and accept nothing malformed

**Files:**
- Modify: `src/math/thomas_solver.hpp` — both `solve_thomas` (span overload, validation ~:100) and `solve_thomas_projected` (validation ~:336). Only the overloads that take raw spans and validate sizes; leave any matrix-view overloads alone.
- Test: `tests/tridiagonal_solver_test.cc`

**Interfaces:**
- Produces: `solve_thomas`/`solve_thomas_projected` return `ok_result()` for all-empty systems; still reject size mismatches, including nonempty companions with empty `diag`.

- [ ] **Step 1: Write the failing tests** (append to `tests/tridiagonal_solver_test.cc`; match the file's existing call style for both functions — read its existing tests first and reuse their span setup)

```cpp
// ===========================================================================
// Regression tests for bugs found during code review (issue #441)
// ===========================================================================

// Regression: n==0 must be trivial success, not a size error
// Bug: `lower.size() != n - 1` ran before the n==0 check; with n==0 the
//      subtraction wrapped to SIZE_MAX, so empty systems returned
//      "Lower diagonal size must be n-1" and the success path was dead code.
TEST(ThomasSolverTest, EmptySystemSucceeds) {
    std::span<const double> empty_c;
    std::span<double> empty_m;
    auto result = mango::solve_thomas<double>(
        empty_c, empty_c, empty_c, empty_c, empty_m, empty_m);
    EXPECT_TRUE(result.ok());
}

// Regression: empty diag with nonempty companion spans must still error
// Bug: naive hoisting of the n==0 check above validation would have
//      accepted malformed calls (empty diag, nonempty rhs) as success.
TEST(ThomasSolverTest, EmptyDiagNonemptyRhsRejected) {
    std::vector<double> rhs = {1.0};
    std::vector<double> sol(1);
    std::vector<double> ws(2);
    std::span<const double> empty_c;
    auto result = mango::solve_thomas<double>(
        empty_c, empty_c, empty_c,
        std::span<const double>{rhs}, std::span<double>{sol},
        std::span<double>{ws});
    EXPECT_FALSE(result.ok());
}

// Regression: same n==0 ordering bug in the projected (obstacle) variant
// Bug: duplicate of the wrap-around in solve_thomas_projected.
TEST(ThomasSolverTest, ProjectedEmptySystemSucceeds) {
    std::span<const double> empty_c;
    std::span<double> empty_m;
    auto result = mango::solve_thomas_projected<double>(
        empty_c, empty_c, empty_c, empty_c, empty_c, empty_m, empty_m);
    EXPECT_TRUE(result.ok());
}
```

(Adjust argument lists to the real signatures — `solve_thomas_projected` takes `psi` and the config defaults; read the declarations first.)

- [ ] **Step 2: Run to verify failure**

Run: `bazel test //tests:tridiagonal_solver_test --test_output=all`
Expected: the two `Empty*Succeeds` tests FAIL with the misleading size error.

- [ ] **Step 3: Implement** — in both functions, replace the validation prologue:

```cpp
    const size_t n = diag.size();
    const size_t offdiag_size = (n == 0) ? 0 : n - 1;

    // Validate dimensions (offdiag_size avoids n-1 wrap-around at n==0)
    if (lower.size() != offdiag_size) {
        return Result::error_result("Lower diagonal size must be n-1");
    }
    if (upper.size() != offdiag_size) {
        return Result::error_result("Upper diagonal size must be n-1");
    }
    // ... keep the remaining size checks (rhs, solution, workspace, psi) as-is ...
```

The existing `if (n == 0) return Result::ok_result();` in "Handle trivial cases" stays where it is (now reachable). Note: `workspace.size() < 2 * n` passes trivially for n==0 — that is fine.

- [ ] **Step 4: Run to verify pass**

Run: `bazel test //tests:tridiagonal_solver_test --test_output=all`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/math/thomas_solver.hpp tests/tridiagonal_solver_test.cc
git commit -m "Make Thomas n==0 success path reachable"
```

---

### Task 3: PDESolver drops temporal events on reuse

**Files:**
- Modify: `src/pde/internal/pde_solver.hpp` — `initialize()` (~:97) and lifecycle doc comments on `initialize()`/`solve()`.
- Test: `tests/internal/temporal_event_test.cc`

**Interfaces:**
- Produces: reuse contract `initialize(ic); solve();` per run; events replay on every initialized run.

- [ ] **Step 1: Write the failing test** (append to `tests/internal/temporal_event_test.cc`; reuse the file's existing `TestPDESolver`/`make_test_solver` helpers, grid/workspace setup, and event-registration mechanism — read an existing test in the file and mirror its scaffolding exactly)

```cpp
// Regression: a reused solver must replay temporal events
// Bug: next_event_idx_ was never reset, so a second initialize()+solve()
//      on the same instance silently skipped all events (dividends dropped).
TEST(TemporalEventTest, ReusedSolverReplaysEvents) {
    // ... same grid / time / workspace scaffolding as EventAppliedAfterStep ...
    int event_calls = 0;
    // register one mid-interval event that bumps the solution and counts calls
    // (same registration call the existing tests use), callback increments
    // event_calls and adds 1.0 to every u[i]

    auto ic = [](std::span<const double> /*x*/, std::span<double> u) {
        std::fill(u.begin(), u.end(), 0.0);
    };

    solver.initialize(ic);
    ASSERT_TRUE(solver.solve().has_value());
    std::vector<double> first(solver_solution.begin(), solver_solution.end());
    EXPECT_EQ(event_calls, 1);

    solver.initialize(ic);
    ASSERT_TRUE(solver.solve().has_value());
    std::vector<double> second(solver_solution.begin(), solver_solution.end());

    EXPECT_EQ(event_calls, 2);  // fired on BOTH runs
    for (size_t i = 0; i < first.size(); ++i) {
        EXPECT_DOUBLE_EQ(second[i], first[i]);
    }
}
```

(`solver_solution` = however the fixture reads the solution back, e.g. `grid->solution()`; use Dirichlet-0 BCs and the Laplacian operator like the neighboring tests.)

- [ ] **Step 2: Run to verify failure**

Run: `bazel test //tests/internal:temporal_event_test --test_output=all` (check the actual target name in `tests/internal/BUILD.bazel` or `tests/BUILD.bazel` first)
Expected: FAIL — `event_calls == 1` after the second run.

- [ ] **Step 3: Implement** — in `initialize()` add the reset, and document the lifecycle:

```cpp
    /// Initialize with initial condition
    ///
    /// Starts a new run: sets the initial condition and rewinds the
    /// temporal-event cursor.  The reuse contract is one
    /// `initialize(ic); solve();` pair per run — calling solve() again
    /// without re-initializing is unsupported (it would evolve the
    /// already-final state).
    ///
    /// @param ic Initial condition function: ic(x, u)
    template<typename IC>
    void initialize(IC&& ic) {
        next_event_idx_ = 0;  // rewind event replay for the new run
        ...existing body unchanged...
    }
```

Add one line to `solve()`'s doc comment: `/// Requires a preceding initialize(ic) for each run; see initialize().`

- [ ] **Step 4: Run to verify pass**

Run: `bazel test //tests/internal:temporal_event_test --test_output=all`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pde/internal/pde_solver.hpp tests/internal/temporal_event_test.cc
git commit -m "Rewind temporal-event cursor in PDESolver initialize"
```

---

### Task 4: Newton ConvergenceFailure reports residual 0

**Files:**
- Modify: `src/pde/internal/pde_solver.hpp` — Newton iteration (~:747-810).
- Modify: `src/support/error_types.hpp` — `SolverError::residual` comment (~:24).
- Test: `tests/internal/pde_solver_test.cc`

**Interfaces:**
- Produces: `SolverError.residual` on `ConvergenceFailure` = last computed normalized RMS step delta (finite, > 0 after ≥1 iteration with a nonzero update).

- [ ] **Step 1: Write the failing test** (append to `tests/internal/pde_solver_test.cc`, reusing its `TestPDESolver` scaffolding — mirror `HeatEquationDirichletBC`'s setup)

```cpp
// Regression: ConvergenceFailure must report the actual last step delta
// Bug: the loop's final statement copied u into newton_u_old, then the
//      failure path computed the delta of u against that same buffer —
//      comparing u with itself, so every failure reported residual == 0.
TEST(PDESolverTest, ConvergenceFailureReportsNonzeroResidual) {
    // Heat-equation fixture as in HeatEquationDirichletBC, but with a
    // config forcing failure: max_iter = 1, tolerance = 1e-300 so one
    // iteration with a genuinely nonzero first update cannot converge.
    // (Set config via the same mechanism the fixture uses — a
    //  SolverConfig/config_ parameter on construction; read the fixture.)
    auto result = solver.solve();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::SolverErrorCode::ConvergenceFailure);
    EXPECT_TRUE(std::isfinite(result.error().residual));
    EXPECT_GT(result.error().residual, 0.0);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `bazel test //tests/internal:pde_solver_test --test_output=all`
Expected: FAIL — `residual` equals 0.

- [ ] **Step 3: Implement** — in the Newton loop:

```cpp
        double last_error = std::numeric_limits<double>::infinity();
        for (size_t iter = 0; iter < config_.max_iter; ++iter) {
            ...
            double error = compute_step_delta_error(u, workspace_.newton_u_old());
            last_error = error;
            if (error < config_.tolerance) {
                return {};
            }
            std::copy(u.begin(), u.end(), workspace_.newton_u_old().begin());
        }

        return std::unexpected(SolverError{
            .code = SolverErrorCode::ConvergenceFailure,
            .iterations = config_.max_iter,
            .residual = last_error
        });
```

In `error_types.hpp`, change the `residual` comment:

```cpp
    double residual{0.0};  // Code-dependent diagnostic: for ConvergenceFailure,
                           // the solution-norm-normalized RMS Newton step delta
                           // (not a PDE residual); infinity for
                           // LinearSolveFailure; other codes may carry other
                           // values (e.g. grid-validation data).
```

- [ ] **Step 4: Run to verify pass**

Run: `bazel test //tests/internal:pde_solver_test --test_output=all`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pde/internal/pde_solver.hpp src/support/error_types.hpp tests/internal/pde_solver_test.cc
git commit -m "Report real step delta on Newton convergence failure"
```

---

### Task 5: NonUniformSpacing copies alias the source buffer

**Files:**
- Modify: `src/pde/core/grid.hpp` — `NonUniformSpacing` struct (~:534-613).
- Test: `tests/grid_spacing_test.cc`

**Interfaces:**
- Produces: copy ctor/assignment re-point `sections_view_` into the copy's own `precomputed`; explicitly defaulted moves; moved-from objects support only destruction/reassignment.

- [ ] **Step 1: Write the failing test** (append to `tests/grid_spacing_test.cc`; `NonUniformSpacing` is used directly — its members `precomputed` and `sections_view_` are public)

```cpp
// ===========================================================================
// Regression tests for bugs found during code review (issue #441)
// ===========================================================================

// Regression: copying NonUniformSpacing must not alias the source buffer
// Bug: the implicitly-generated copy copied sections_view_ verbatim, so the
//      copy's mdspan pointed into the SOURCE's precomputed vector —
//      use-after-free once the source was destroyed.
TEST(GridSpacingTest, NonUniformSpacingCopyOwnsItsView) {
    std::vector<double> x = {0.0, 0.1, 0.3, 0.7, 1.0};

    auto make_copy = [&]() {
        mango::NonUniformSpacing<double> source(std::span<const double>{x});
        mango::NonUniformSpacing<double> copy = source;  // copy-construct
        return copy;  // source destroyed on return
    };
    auto copy = make_copy();
    EXPECT_EQ(copy.sections_view_.data_handle(), copy.precomputed.data());
    // Values still correct: dx_left_inv[0] = 1/(x[1]-x[0]) = 10.0
    EXPECT_DOUBLE_EQ(copy.dx_left_inv()[0], 10.0);
}

// Regression: copy-assignment has the same aliasing bug
// Bug: same root cause via the implicit copy-assignment operator.
TEST(GridSpacingTest, NonUniformSpacingCopyAssignOwnsItsView) {
    std::vector<double> x = {0.0, 0.1, 0.3, 0.7, 1.0};
    std::vector<double> y = {0.0, 0.2, 0.5, 0.9, 2.0};
    mango::NonUniformSpacing<double> target(std::span<const double>{y});
    {
        mango::NonUniformSpacing<double> source(std::span<const double>{x});
        target = source;
    }  // source destroyed
    EXPECT_EQ(target.sections_view_.data_handle(), target.precomputed.data());
    EXPECT_DOUBLE_EQ(target.dx_left_inv()[0], 10.0);
}

// Regression: moves must keep the view valid (vector move preserves data())
TEST(GridSpacingTest, NonUniformSpacingMoveKeepsViewValid) {
    std::vector<double> x = {0.0, 0.1, 0.3, 0.7, 1.0};
    auto make_moved = [&]() {
        mango::NonUniformSpacing<double> source(std::span<const double>{x});
        return mango::NonUniformSpacing<double>(std::move(source));
    };
    auto moved = make_moved();
    EXPECT_EQ(moved.sections_view_.data_handle(), moved.precomputed.data());
    EXPECT_DOUBLE_EQ(moved.dx_left_inv()[0], 10.0);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `bazel test //tests:grid_spacing_test --test_output=all`
Expected: the two copy tests FAIL on the `data_handle()` assertion (may also crash under ASan — either counts).

- [ ] **Step 3: Implement** — add to `NonUniformSpacing` after the existing constructor:

```cpp
    /// Copy ops re-point sections_view_ into this object's own buffer;
    /// the defaulted copies would alias the source's vector (dangling
    /// once the source dies).  Moves are safe defaulted: vector move
    /// preserves data(), so the transferred view stays valid.  A
    /// moved-from object supports only destruction and reassignment —
    /// never evaluation.
    NonUniformSpacing(const NonUniformSpacing& other)
        : n(other.n)
        , precomputed(other.precomputed)
        , sections_view_(precomputed.data(), 5, n - 2)
    {}

    NonUniformSpacing& operator=(const NonUniformSpacing& other) {
        if (this != &other) {
            NonUniformSpacing tmp(other);      // strong exception safety
            *this = std::move(tmp);
        }
        return *this;
    }

    NonUniformSpacing(NonUniformSpacing&&) noexcept = default;
    NonUniformSpacing& operator=(NonUniformSpacing&&) noexcept = default;
    ~NonUniformSpacing() = default;
```

(Members are declared in order `n`, `precomputed`, `sections_view_`, so the init list order above is correct.)

- [ ] **Step 4: Run to verify pass**

Run: `bazel test //tests:grid_spacing_test --test_output=all` and `bazel test //tests:grid_test //tests:grid_spacing_mdspan_test //tests:grid_spacing_data_test`
Expected: all PASS (Grid's variant copy now deep-copies correctly).

- [ ] **Step 5: Commit**

```bash
git add src/pde/core/grid.hpp tests/grid_spacing_test.cc
git commit -m "Re-point NonUniformSpacing view on copy"
```

---

### Task 6: BSplineND copies alias the source coefficient buffer

**Files:**
- Modify: `src/math/bspline/bspline_nd.hpp` — class `BSplineND` (private ctor ~:275; members ~:268-272).
- Test: `tests/bspline_nd_test.cc`

**Interfaces:**
- Produces: `BSplineND` copy ctor/assignment re-point `coeffs_view_` via `create_coeffs_view(coeffs_.data(), dims_)`; explicitly defaulted moves; value semantics preserved (it is returned via `std::expected<BSplineND, …>`).

- [ ] **Step 1: Write the failing test** (append to `tests/bspline_nd_test.cc`; mirror an existing test's `create(grids, knots, coeffs)` setup — the file already builds small splines, reuse one of its fixtures/helpers for a valid 1D or 2D spline and a known-value query point)

```cpp
// ===========================================================================
// Regression tests for bugs found during code review (issue #441)
// ===========================================================================

// Regression: copying BSplineND must not alias the source's coefficients
// Bug: the implicitly-generated copy copied coeffs_view_ verbatim, so the
//      copy evaluated through the SOURCE's coeffs_ vector — use-after-free
//      once the source was destroyed.
TEST(BSplineNDTest, CopyOwnsItsCoefficientView) {
    // build a small valid spline exactly like the existing tests do
    auto make_copy = [&]() {
        auto source = mango::BSplineND<double, 1>::create(grids, knots, coeffs).value();
        double expected = source.eval(query);   // use the file's eval call style
        auto copy = source;                     // copy-construct
        return std::pair{copy, expected};
    };
    auto [copy, expected] = make_copy();        // source destroyed
    EXPECT_DOUBLE_EQ(copy.eval(query), expected);
}

// Regression: copy-assignment has the same aliasing bug
TEST(BSplineNDTest, CopyAssignOwnsItsCoefficientView) {
    auto target = mango::BSplineND<double, 1>::create(grids2, knots2, coeffs2).value();
    double expected = 0.0;
    {
        auto source = mango::BSplineND<double, 1>::create(grids, knots, coeffs).value();
        expected = source.eval(query);
        target = source;
    }  // source destroyed
    EXPECT_DOUBLE_EQ(target.eval(query), expected);
}

// Regression: moved-to object must evaluate correctly after source death
TEST(BSplineNDTest, MoveKeepsCoefficientViewValid) {
    auto make_moved = [&]() {
        auto source = mango::BSplineND<double, 1>::create(grids, knots, coeffs).value();
        double expected = source.eval(query);
        return std::pair{std::move(source), expected};
    };
    auto [moved, expected] = make_moved();
    EXPECT_DOUBLE_EQ(moved.eval(query), expected);
}
```

(Replace `grids/knots/coeffs/query` and the `eval` spelling with the file's actual fixture data and API — read two existing tests first. The dangling read may return the right value by luck without ASan; the copy tests are still the specified regression, and the `data`-identity check is done structurally in Task 5's style only where members are public — here behavior + ASan in CI is the guard.)

- [ ] **Step 2: Run to verify current behavior**

Run: `bazel test //tests:bspline_nd_test --test_output=all`
Expected: copy tests FAIL (wrong values or ASan violation) — if they happen to pass due to the freed buffer still holding data, run once with `--config=asan` if available, else `bazel test //tests:bspline_nd_test --test_env=ASAN_OPTIONS= --copt=-fsanitize=address --linkopt=-fsanitize=address`; document in the commit message that the test guards via sanitizer.

- [ ] **Step 3: Implement** — add to `BSplineND`'s public section:

```cpp
    /// Copy ops re-point coeffs_view_ into this object's own coeffs_;
    /// the defaulted copies would alias the source's buffer.  Moves are
    /// safe defaulted (vector move preserves data()).  A moved-from
    /// object supports only destruction and reassignment.
    BSplineND(const BSplineND& other)
        : grids_(other.grids_)
        , knots_(other.knots_)
        , coeffs_(other.coeffs_)
        , coeffs_view_(nullptr, CoeffExtents{})
        , dims_(other.dims_)
    {
        coeffs_view_ = create_coeffs_view(coeffs_.data(), dims_);
    }

    BSplineND& operator=(const BSplineND& other) {
        if (this != &other) {
            BSplineND tmp(other);   // strong exception safety
            *this = std::move(tmp);
        }
        return *this;
    }

    BSplineND(BSplineND&&) noexcept = default;
    BSplineND& operator=(BSplineND&&) noexcept = default;
    ~BSplineND() = default;
```

(`create_coeffs_view` is a private static at ~:349 — public members may call it; keep the copy ctor's dummy-then-assign pattern the private ctor already uses.)

- [ ] **Step 4: Run to verify pass**

Run: `bazel test //tests:bspline_nd_test //tests:bspline_nd_mdspan_test --test_output=all`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/math/bspline/bspline_nd.hpp tests/bspline_nd_test.cc
git commit -m "Re-point BSplineND coefficient view on copy"
```

---

### Task 7: CubicSpline2D::eval ignores its documented extrapolation contract

**Files:**
- Modify: `src/math/cubic_spline_solver.hpp` — `CubicSpline2D::eval` (~:445-476).
- Test: `tests/cubic_spline_2d_test.cc` — new regressions + flip any existing extrapolation tests/comments that pin the old cubic behavior.

**Interfaces:**
- Produces: `eval(x, y)` clamps both coordinates to the grid domain (nearest-boundary extrapolation, as documented). 1D `CubicSpline::eval` is UNCHANGED.

- [ ] **Step 1: Write the failing tests** (append; read the file's existing build/eval setup first)

```cpp
// ===========================================================================
// Regression tests for bugs found during code review (issue #441)
// ===========================================================================

// Regression: out-of-domain eval must clamp, per the documented contract
// Bug: docs promised "extrapolation uses nearest boundary value" but the
//      code evaluated the boundary cubic at the out-of-range offset,
//      diverging cubically off-grid.
TEST(CubicSpline2DTest, ExtrapolationClampsToBoundary) {
    // grid with nonzero boundary slope so cubic extrapolation would diverge
    std::vector<double> xs = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> ys = {0.0, 1.0, 2.0, 3.0};
    // z = x^2 + y^2 sampled on the grid (build exactly as existing tests do)
    ...
    // outside on each side clamps to the boundary evaluation:
    EXPECT_DOUBLE_EQ(spline.eval(5.0, 1.5), spline.eval(3.0, 1.5));
    EXPECT_DOUBLE_EQ(spline.eval(-2.0, 1.5), spline.eval(0.0, 1.5));
    EXPECT_DOUBLE_EQ(spline.eval(1.5, 5.0), spline.eval(1.5, 3.0));
    EXPECT_DOUBLE_EQ(spline.eval(1.5, -2.0), spline.eval(1.5, 0.0));
    // corner: both coordinates outside
    EXPECT_DOUBLE_EQ(spline.eval(5.0, -2.0), spline.eval(3.0, 0.0));
}
```

Also: grep the file for existing extrapolation tests mentioning "natural spline extrapolation" — update their expectations and comments to the clamped contract (they currently pin the bug).

- [ ] **Step 2: Run to verify failure**

Run: `bazel test //tests:cubic_spline_2d_test --test_output=all`
Expected: new test FAILS (diverged cubic values).

- [ ] **Step 3: Implement** — at the top of `CubicSpline2D::eval`, after the `is_built()` check:

```cpp
        // Documented contract: extrapolation uses the nearest boundary
        // value — clamp both coordinates to the grid domain.
        x_eval = std::clamp(x_eval, x_.front(), x_.back());
        y_eval = std::clamp(y_eval, y_.front(), y_.back());
```

(Ensure `<algorithm>` is included; `x_`/`y_` are the stored grid vectors — verify the member names in the class.)

- [ ] **Step 4: Run to verify pass**

Run: `bazel test //tests:cubic_spline_2d_test --test_output=all`
Expected: all PASS (including the flipped legacy tests).

- [ ] **Step 5: Commit**

```bash
git add src/math/cubic_spline_solver.hpp tests/cubic_spline_2d_test.cc
git commit -m "Clamp CubicSpline2D eval to documented boundary contract"
```

---

### Task 8: Bandwidth template parameter is a stack-overflow trap

**Files:**
- Modify: `src/math/bspline/bspline_collocation.hpp` (~:74-78) and `src/math/bspline/bspline_collocation_workspace.hpp` (~:35-40).

**Interfaces:**
- Produces: `static_assert(Bandwidth == 4)` in `BSplineCollocation1D`; `static_assert(Bandwidth > 0)` in `BSplineCollocationWorkspace`.

- [ ] **Step 1: Implement** (compile-time guard — the static_asserts ARE the regression tests; no runtime test)

In `BSplineCollocation1D`'s class body, right after `static constexpr size_t BANDWIDTH = Bandwidth;`:

```cpp
    // Reserved parameter: only cubic (Bandwidth == 4) is supported —
    // cubic_basis_nonuniform unconditionally writes 4 entries, so any
    // smaller Bandwidth is a stack buffer overflow (issue #441 item 9).
    static_assert(Bandwidth == 4,
                  "BSplineCollocation1D supports only Bandwidth == 4 (cubic)");
```

Update the class's `@tparam Bandwidth` doc from "degree + 1" generality to "reserved; must be 4 (cubic) until basis evaluation is generalized".

In `BSplineCollocationWorkspace`, next to the `KL`/`KU` constants:

```cpp
    static_assert(Bandwidth > 0,
                  "Bandwidth == 0 underflows the KL/KU band constants");
```

- [ ] **Step 2: Verify the guarded builds still compile**

Run: `bazel test //tests:bspline_collocation_1d_test //tests:bspline_collocation_workspace_test //tests:bspline_fitter_4d_separable_test`
Expected: all PASS (only `Bandwidth == 4` is instantiated in-repo).

- [ ] **Step 3: Commit**

```bash
git add src/math/bspline/bspline_collocation.hpp src/math/bspline/bspline_collocation_workspace.hpp
git commit -m "Guard B-spline collocation Bandwidth at compile time"
```

---

### Task 9: Workspace pivots typed int, not lapack_int

**Files:**
- Modify: `src/math/bspline/bspline_collocation_workspace.hpp` — every pivot-related `int` (~:30, :54, :68-70, :111-114, :127, :133, :165).
- Modify: `tests/bspline_collocation_workspace_test.cc` — `sizeof(int)` arithmetic and the 400-byte comments (~:14, :19, :78).

**Interfaces:**
- Produces: `pivots()` returns `std::span<lapack_int>`; layout math uses `sizeof(lapack_int)`/`alignof(lapack_int)`.

- [ ] **Step 1: Add the type assertion + update test arithmetic** in `tests/bspline_collocation_workspace_test.cc`:

```cpp
// Regression: pivot storage must use lapack_int, not int
// Bug: the workspace sized and sliced pivots as int but handed them to
//      LAPACKE_dgbtrf/dgbtrs (lapack_int*) — a compile break under ILP64.
//      NOTE: under this repo's LP64 CI lapack_int == int, so this assert
//      cannot catch a regression here; the real guard would be an ILP64
//      build (out of scope, see design doc).
using Pivot = typename decltype(std::declval<mango::BSplineCollocationWorkspace<double, 4>&>().pivots())::element_type;
static_assert(std::same_as<Pivot, lapack_int>);
```

Replace every `sizeof(int)` in the file's pivot pointer arithmetic with `sizeof(lapack_int)`, and reword the `100*4 = 400 bytes` comments to `100*sizeof(lapack_int)`. Add `#include <lapacke.h>` if not present.

- [ ] **Step 2: Implement** — in the workspace header, replace `int` with `lapack_int` at: the doc comment (:30), `block_alignment_int` (`alignof(int)` → `alignof(lapack_int)`; consider renaming to `block_alignment_pivot`), the sizing (`n * sizeof(int)` → `n * sizeof(lapack_int)`), the slicing (`std::span<int>` / `start_array_lifetime<int>` → `lapack_int`), the accessors, and the member. Add `#include <lapacke.h>` (the include `banded_matrix_solver.hpp` uses).

- [ ] **Step 3: Run to verify**

Run: `bazel test //tests:bspline_collocation_workspace_test //tests:bspline_collocation_1d_test //tests:bspline_banded_solver_test --test_output=all`
Expected: all PASS (LP64 behavior unchanged).

- [ ] **Step 4: Commit**

```bash
git add src/math/bspline/bspline_collocation_workspace.hpp tests/bspline_collocation_workspace_test.cc
git commit -m "Type workspace pivots as lapack_int"
```

---

### Task 10: Factory collapses distinct build failures into InvalidGridSize

**Files:**
- Modify: `src/support/error_types.hpp` — append `PriceTableBuildFailed` to `ValidationErrorCode` (~:29-46); add cases to `convert_to_iv_error(const ValidationError&)` (~:262) and `convert_to_price_table_error(const ValidationError&)` (~:348).
- Modify: `src/option/iv_result.hpp` — explicit case in `validation_error_to_iv_error` (~:55).
- Create: `src/option/detail/price_table_error_mapping.hpp` + `src/option/detail/price_table_error_mapping.cpp`.
- Modify: `src/option/price_table_factory.cpp` — remove the local `to_validation_error`, include the detail header, rewire the five hardcoded sites (~:297, :332, :535, :541, :558).
- Modify: `src/option/BUILD.bazel` — new restricted-visibility target.
- Modify: `tests/BUILD.bazel` — new test target.
- Create: `tests/price_table_error_mapping_test.cc`.
- Test also: `tests/error_types_test.cc` (conversion coverage).

**Interfaces:**
- Produces: `mango::detail::to_validation_error(const PriceTableError&) -> ValidationError` (external linkage, internal header); `ValidationErrorCode::PriceTableBuildFailed` mapping to `IVErrorCode::InvalidGridConfig` and `PriceTableErrorCode::SurfaceBuildFailed`.

- [ ] **Step 1: Create the detail header** `src/option/detail/price_table_error_mapping.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

// INTERNAL, UNSTABLE — implementation detail of price_table_factory,
// exposed only so the PriceTableError -> ValidationError mapping is
// directly unit-testable.  Not part of the public API.

#include "mango/support/error_types.hpp"

namespace mango::detail {

/// Map a price-table build failure to the factory's public ValidationError.
/// Grid-shaped failures keep their specific codes; everything else becomes
/// the generic PriceTableBuildFailed (issue #441 item 7).
[[nodiscard]] ValidationError to_validation_error(const PriceTableError& error);

}  // namespace mango::detail
```

- [ ] **Step 2: Create the implementation** `src/option/detail/price_table_error_mapping.cpp` — move the function from `price_table_factory.cpp`, replacing the `default:` arm with explicit grouped cases (no `default:`, so a future enum value trips `-Werror=switch`):

```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/detail/price_table_error_mapping.hpp"

namespace mango::detail {

ValidationError to_validation_error(const PriceTableError& error) {
    switch (error.code) {
        case PriceTableErrorCode::NonPositiveValue:
            return ValidationError{ValidationErrorCode::InvalidBounds, 0.0,
                                   error.axis_index};
        case PriceTableErrorCode::InsufficientGridPoints:
        case PriceTableErrorCode::GridNotSorted:
            return ValidationError{ValidationErrorCode::InvalidGridSize,
                                   static_cast<double>(error.count),
                                   error.axis_index};
        case PriceTableErrorCode::InvalidConfig:
        case PriceTableErrorCode::EmptyBatch:
        case PriceTableErrorCode::ExtractionFailed:
        case PriceTableErrorCode::RepairFailed:
        case PriceTableErrorCode::FittingFailed:
        case PriceTableErrorCode::SurfaceBuildFailed:
        case PriceTableErrorCode::SerializationFailed:
        case PriceTableErrorCode::ArenaAllocationFailed:
        case PriceTableErrorCode::TensorCreationFailed:
            return ValidationError{ValidationErrorCode::PriceTableBuildFailed,
                                   static_cast<double>(error.count),
                                   error.axis_index};
    }
    // Unreachable: the switch is exhaustive (-Werror=switch enforces it).
    return ValidationError{ValidationErrorCode::PriceTableBuildFailed, 0.0, 0};
}

}  // namespace mango::detail
```

(Copy the exact field names — `axis_index`, `count` — from the current function; if PR #454 merged meanwhile and added arms like `NoViableSurface`, keep those arms verbatim above the grouped block.)

- [ ] **Step 3: Enum + conversion switches** in `src/support/error_types.hpp`:

Append to `ValidationErrorCode` (LAST, preserving ordinals):

```cpp
    DiscreteDividendMismatch,
    PriceTableBuildFailed      ///< Price table construction failed (non-grid cause)
```

In `convert_to_iv_error(const ValidationError&)` add to the `InvalidGridConfig` group:

```cpp
        case ValidationErrorCode::PriceTableBuildFailed:
            code = IVErrorCode::InvalidGridConfig;
            break;
```

In `convert_to_price_table_error(const ValidationError&)`:

```cpp
        case ValidationErrorCode::PriceTableBuildFailed:
            code = PriceTableErrorCode::SurfaceBuildFailed;
            break;
```

In `src/option/iv_result.hpp` `validation_error_to_iv_error`, before the `default:`:

```cpp
        case ValidationErrorCode::PriceTableBuildFailed:
            // A build failure is a grid/table problem, not an arbitrage signal.
            code = IVErrorCode::InvalidGridConfig;
            break;
```

- [ ] **Step 4: Rewire the factory.** In `price_table_factory.cpp`: delete the local `to_validation_error`, add `#include "mango/option/detail/price_table_error_mapping.hpp"`, qualify existing calls as `detail::to_validation_error(...)`. Replace each of the five hardcoded sites; where the failure in hand is an `InterpolationError` (the three spline/surface-construction sites around :535-:558), preserve its payload:

```cpp
        return std::unexpected(detail::to_validation_error(
            convert_to_price_table_error(interp_result.error())));
```

and where the failure is a surface/wrapper construction with no payload (the sites around :297/:332), use:

```cpp
        return std::unexpected(detail::to_validation_error(
            PriceTableError{PriceTableErrorCode::SurfaceBuildFailed, 0, 0}));
```

(Check `PriceTableError`'s actual field list/order at `error_types.hpp:168` before writing the braces; use designated initializers if that's the surrounding style.)

- [ ] **Step 5: Bazel wiring.** In `src/option/BUILD.bazel`:

```python
cc_library(
    name = "price_table_error_mapping",
    srcs = ["detail/price_table_error_mapping.cpp"],
    hdrs = ["detail/price_table_error_mapping.hpp"],
    deps = ["//src/support:error_types"],
    copts = ["-Wall", "-Wextra", "-Werror", "-O3"],
    visibility = ["//src/option:__pkg__", "//tests:__pkg__"],
    strip_include_prefix = "/src/option",
    include_prefix = "mango/option",
)
```

Add `":price_table_error_mapping"` to `price_table_factory`'s `deps`. (Match the neighboring targets' copts style exactly.)

- [ ] **Step 6: Write the tests.** Create `tests/price_table_error_mapping_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/detail/price_table_error_mapping.hpp"
#include <gtest/gtest.h>

using mango::PriceTableError;
using mango::PriceTableErrorCode;
using mango::ValidationErrorCode;

// Regression: distinct build failures must not surface as InvalidGridSize
// Bug: to_validation_error's default: arm mapped fitting, repair,
//      extraction, serialization, allocation, and config failures all to
//      InvalidGridSize — a grid-size lie that destroyed diagnostics.
TEST(PriceTableErrorMappingTest, BuildFailuresMapToPriceTableBuildFailed) {
    for (auto code : {PriceTableErrorCode::InvalidConfig,
                      PriceTableErrorCode::EmptyBatch,
                      PriceTableErrorCode::ExtractionFailed,
                      PriceTableErrorCode::RepairFailed,
                      PriceTableErrorCode::FittingFailed,
                      PriceTableErrorCode::SurfaceBuildFailed,
                      PriceTableErrorCode::SerializationFailed,
                      PriceTableErrorCode::ArenaAllocationFailed,
                      PriceTableErrorCode::TensorCreationFailed}) {
        auto ve = mango::detail::to_validation_error(
            PriceTableError{code, 0, 0});
        EXPECT_EQ(ve.code, ValidationErrorCode::PriceTableBuildFailed)
            << "code " << static_cast<int>(code);
    }
}

TEST(PriceTableErrorMappingTest, SpecificArmsUnchanged) {
    EXPECT_EQ(mango::detail::to_validation_error(
                  PriceTableError{PriceTableErrorCode::NonPositiveValue, 0, 0}).code,
              ValidationErrorCode::InvalidBounds);
    EXPECT_EQ(mango::detail::to_validation_error(
                  PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 0, 3}).code,
              ValidationErrorCode::InvalidGridSize);
    EXPECT_EQ(mango::detail::to_validation_error(
                  PriceTableError{PriceTableErrorCode::GridNotSorted, 0, 0}).code,
              ValidationErrorCode::InvalidGridSize);
}
```

(Fix the `PriceTableError` construction to its real fields — read the struct first.) Append to `tests/error_types_test.cc`:

```cpp
// Regression: PriceTableBuildFailed must convert without UB
// Bug: convert_to_iv_error / convert_to_price_table_error switch over
//      ValidationErrorCode with an uninitialized local and no default; a
//      new enum value without a case reads uninitialized memory.
TEST(ErrorTypesTest, PriceTableBuildFailedConversions) {
    mango::ValidationError ve{mango::ValidationErrorCode::PriceTableBuildFailed};
    EXPECT_EQ(mango::convert_to_iv_error(ve).code,
              mango::IVErrorCode::InvalidGridConfig);
    EXPECT_EQ(mango::convert_to_price_table_error(ve).code,
              mango::PriceTableErrorCode::SurfaceBuildFailed);
}
```

and a matching `validation_error_to_iv_error` check in `tests/iv_error_types_test.cc`:

```cpp
// Regression: a build failure must not masquerade as an arbitrage violation
// Bug: validation_error_to_iv_error's default: arm mapped every unlisted
//      code to ArbitrageViolation.
TEST(IVErrorTypesTest, PriceTableBuildFailedMapsToInvalidGridConfig) {
    mango::ValidationError ve{mango::ValidationErrorCode::PriceTableBuildFailed};
    EXPECT_EQ(mango::validation_error_to_iv_error(ve).code,
              mango::IVErrorCode::InvalidGridConfig);
}
```

Add the new test target to `tests/BUILD.bazel` (copy the shape of `error_types_test`'s target, deps `//src/option:price_table_error_mapping` + gtest).

- [ ] **Step 7: Run**

Run: `bazel test //tests:price_table_error_mapping_test //tests:error_types_test //tests:iv_error_types_test //tests:price_table_factory_test --test_output=all`
Expected: all PASS. Also `bazel build //src/python:mango_option` (Python formatting switch has a safe default — must still compile).

- [ ] **Step 8: Commit**

```bash
git add src/support/error_types.hpp src/option/iv_result.hpp \
        src/option/detail/ src/option/price_table_factory.cpp \
        src/option/BUILD.bazel tests/BUILD.bazel \
        tests/price_table_error_mapping_test.cc tests/error_types_test.cc \
        tests/iv_error_types_test.cc
git commit -m "Stop collapsing price table build failures"
```

---

### Task 11: Root-finding tolerance units — docs fix + optional brent_tol_x

**Files:**
- Modify: `src/math/root_finding.hpp` — `RootFindingConfig` (~:20-33) and `find_root_brent` (~:175-250).
- Test: `tests/root_finding_test.cc`.

**Interfaces:**
- Produces: `RootFindingConfig::brent_tol_x` (`std::optional<double>`, declared AFTER `brent_tol_abs` so every existing designated initializer stays valid); unset ⇒ bit-for-bit current behavior.

- [ ] **Step 1: Write the tests** (append to `tests/root_finding_test.cc`)

```cpp
// ===========================================================================
// Regression tests for bugs found during code review (issue #441)
// ===========================================================================

// Regression: unset brent_tol_x must reproduce current behavior exactly
// Bug: brent_tol_abs served both |f| and x-distance comparisons; the new
//      optional x-tolerance must not change any existing caller's result.
TEST(RootFindingTest, BrentTolXUnsetMatchesLegacyBehavior) {
    auto f = [](double x) { return x * x - 2.0; };
    mango::RootFindingConfig legacy{.max_iter = 100, .brent_tol_abs = 1e-6};
    mango::RootFindingConfig with_field{.max_iter = 100, .brent_tol_abs = 1e-6};
    // brent_tol_x left unset in both
    auto r1 = mango::find_root_brent(f, 0.0, 2.0, legacy);
    auto r2 = mango::find_root_brent(f, 0.0, 2.0, with_field);
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());
    EXPECT_DOUBLE_EQ(r1->root, r2->root);
    EXPECT_EQ(r1->iterations, r2->iterations);
}

// Regression: one knob for |f| and x-distance is scale-dependent
// Bug: with a steep objective (f-values huge relative to x), a loose
//      brent_tol_abs stops on the |b-a| test at poor x-accuracy; there was
//      no way to tighten x-accuracy without also tightening the |f| test.
TEST(RootFindingTest, BrentTolXControlsXAccuracyIndependently) {
    // Steep function: root at sqrt(2), |f'| ~ 2e6 near the root
    auto f = [](double x) { return 1e6 * (x * x - 2.0); };
    const double root = std::sqrt(2.0);

    mango::RootFindingConfig loose{.max_iter = 200, .brent_tol_abs = 1e-2};
    auto r_loose = mango::find_root_brent(f, 0.0, 2.0, loose);
    ASSERT_TRUE(r_loose.has_value());

    mango::RootFindingConfig tight_x{.max_iter = 200, .brent_tol_abs = 1e-2};
    tight_x.brent_tol_x = 1e-12;
    auto r_tight = mango::find_root_brent(f, 0.0, 2.0, tight_x);
    ASSERT_TRUE(r_tight.has_value());

    // With tight x-tolerance the root is at least as accurate, and the
    // stopping cannot have come from the (now-tight) bracket-width test
    // at a worse x-error than the loose run allowed.
    EXPECT_LE(std::abs(r_tight->root - root), std::abs(r_loose->root - root) + 1e-12);
    EXPECT_LT(std::abs(r_tight->root - root), 1e-6);
}
```

(`find_root_brent`'s exact name/signature: check the header — adjust the calls to the real API, including how results expose `root`/`iterations`.)

- [ ] **Step 2: Run to verify state**

Run: `bazel test //tests:root_finding_test --test_output=all`
Expected: first test fails to COMPILE (`brent_tol_x` doesn't exist) — that is the red step.

- [ ] **Step 3: Implement.** In `RootFindingConfig`:

```cpp
    /// Absolute residual convergence tolerance: Newton stops when
    /// |f(x)| < tolerance.  NOT relative — callers must scale it to
    /// their objective's units.
    double tolerance = 1e-6;
```

```cpp
    /// Absolute tolerance on f-values for Brent's method: the endpoint
    /// root checks and the |f(b)| stopping test.  Also the fallback for
    /// x-distance comparisons when brent_tol_x is unset (historical
    /// behavior: one knob served both units).
    double brent_tol_abs = 1e-6;

    /// Optional absolute tolerance on x-distances for Brent's method:
    /// the |b - a| bracket-width stopping test and the
    /// interpolation-vs-bisection safeguard comparisons (|b - c|, |c - d|).
    /// Unset: falls back to brent_tol_abs, preserving legacy behavior.
    /// MUST be declared after brent_tol_abs (designated-initializer order).
    std::optional<double> brent_tol_x = std::nullopt;
```

(Add `#include <optional>` if missing.) In `find_root_brent`, near the top:

```cpp
    const double tol_x = config.brent_tol_x.value_or(config.brent_tol_abs);
```

then replace `config.brent_tol_abs` with `tol_x` in exactly three places: the `std::abs(b - a) < …` stopping test, condition4 (`std::abs(b - c) < …`), and condition5 (`std::abs(c - d) < …`). The endpoint checks and `std::abs(fb) < …` tests keep `config.brent_tol_abs`.

- [ ] **Step 4: Run to verify pass**

Run: `bazel test //tests:root_finding_test //tests:brent_cpp_test //tests:iv_solver_test --test_output=all`
Expected: all PASS (unset field ⇒ unchanged behavior everywhere).

- [ ] **Step 5: Commit**

```bash
git add src/math/root_finding.hpp tests/root_finding_test.cc
git commit -m "Split Brent x-tolerance from residual tolerance"
```

---

### Task 12: Full pre-PR gate

- [ ] **Step 1:** `bazel test //...` — expected: all tests pass (capture the count; compare to a baseline run if one was recorded).
- [ ] **Step 2:** `bazel build //benchmarks/...` — expected: builds clean.
- [ ] **Step 3:** `bazel build //src/python:mango_option` — expected: builds clean.
- [ ] **Step 4:** `bazel build //crates/mango-option:mango_option 2>/dev/null || true` — if the Rust target exists, it must build too.
- [ ] **Step 5:** No commit; this gate feeds the holistic review + PR step of the delivery workflow.
