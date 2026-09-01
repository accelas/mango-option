# Il'in-Fitted Drift Discretization (#472) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee the stage Jacobian's off-diagonal (Z-matrix) signs for all volatility/grid-spacing combinations by replacing the raw diffusion coefficient with an Il'in exponentially-fitted one, closing the M-matrix hole in one-pass Brennan-Schwartz exactness.

**Architecture:** A free helper `mango::operators::detail::fitted_diffusion(a, b, dx_left, dx_right) -> {a_f, z}` implements the fitting; `SpatialOperator::assemble_jacobian` uses sign-preserving reduced assembly from `{a_f, z}`, and `SpatialOperator::apply_interior` gains a coefficient-combine path for `HasJacobianCoefficients` PDEs so residual/RHS and Jacobian discretize the identical operator. `PDESolver::solve()` additionally validates `γ ∈ (0,1)`.

**Tech Stack:** C++23, Bazel, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-08-31-drift-upwinding-472-design.md` — READ IT FIRST; it contains binding law (floating-point contract, sampling discipline, sign-preserving assembly) that this plan implements.

## Global Constraints

- Every new file starts with `// SPDX-License-Identifier: MIT` (first line).
- Project builds with `-Werror` for `//src/...`, `//tests/...`; no warnings.
- Library code MUST NOT printf/fprintf (USDT probes only — no new logging needed here).
- Commit messages: imperative mood, ≤50-char subject, body wrapped at 72.
- Run tests with `TMPDIR` pointed at the session scratch dir (see execution skill).
- Binding law from the spec (verbatim constraints):
  - Helper clamp: `a_f = max(a_f, a, z)` so `a_f − z ≥ 0` and `a_f ≥ a` hold exactly in floating point.
  - Binding off-diagonal numerator computed literally as `a_f − z`.
  - Diagonal: `diag = c − lower − upper` (row-sum identity by construction).
  - Coefficients `a`, `b(t)`, `r(t)` sampled once per `apply_interior`/`assemble_jacobian` invocation, never per node.
  - `a < 0` out of contract (debug assert); `a == 0, b ≠ 0` returns `a_f = z`; `a == 0, b == 0` via the `z == 0` branch.

---

### Task 1: Baseline KKT measurement (no production change)

**Files:**
- Temporarily modify (then revert): `tests/american_option_test.cc`
- Create: `docs/superpowers/plans/2026-08-31-drift-upwinding-472-baseline.md`

**Interfaces:**
- Produces: recorded pre-fix `violation_count` and `worst_kind` for the canonical fixture, consumed by Task 5's test comment.

- [ ] **Step 1: Temporarily add the probe test** at the end of `tests/american_option_test.cc` (before the closing of the file's namespace, alongside the other `TEST(AmericanOptionTest, ...)` cases):

```cpp
// TEMPORARY probe for #472 baseline — reverted after measurement.
TEST(AmericanOptionTest, HighPecletKktProbe472) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.0,
                   .option_type = OptionType::PUT},
        0.01);  // sigma = 1%

    PDEGridConfig grid_config{
        .grid_spec = GridSpec<double>::uniform(-2.0, 2.0, 41).value(),  // h = 0.1
        .n_time = 50};
    auto solver = AmericanOptionSolver::create(params, grid_config);
    ASSERT_TRUE(solver.has_value());
    auto result = solver->solve();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(solver->complementarity_report().violation_count, 0u)
        << "PRE-FIX BASELINE: max_violation="
        << solver->complementarity_report().max_violation
        << " worst_kind=" << solver->complementarity_report().worst_kind
        << " violation_count="
        << solver->complementarity_report().violation_count;
}
```

(If `PDEGridConfig`/`GridSpec` need includes, `tests/american_option_test.cc` already includes `mango/option/american_option.hpp`, which pulls `grid_spec_types.hpp`.)

- [ ] **Step 2: Run it and capture the failure output**

Run: `bazel test //tests:american_option_test --test_filter='*HighPecletKktProbe472*' --test_output=all 2>&1 | grep -A3 PRE-FIX`
Expected: FAIL with nonzero `violation_count` (that's the measurement). If it unexpectedly PASSES, vary within the spec's sweep table (σ ∈ {0.005, 0.01, 0.02} × n ∈ {81, 41} (h ∈ {0.05, 0.1}) × r ∈ {0.02, 0.05}) until a failing combination is found, and note which one becomes canonical.

- [ ] **Step 3: Record the measurement.** Create `docs/superpowers/plans/2026-08-31-drift-upwinding-472-baseline.md` containing: the exact fixture (all params, grid, n_time), the observed `violation_count`, `max_violation`, `worst_kind`, and the bazel command used.

- [ ] **Step 4: Revert the probe.** FIRST verify the probe is the file's only modification (`git status --porcelain tests/american_option_test.cc` must show exactly ` M tests/american_option_test.cc`, and `git diff tests/american_option_test.cc` must show only the probe hunk — this task added nothing else and Task 1 runs on a clean branch). Then:

Run: `git checkout -- tests/american_option_test.cc`

If the diff shows anything besides the probe hunk, STOP — remove only the probe by editing the file instead of checking out.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-08-31-drift-upwinding-472-baseline.md
git commit -m "Record pre-fix KKT baseline for #472 fixture"
```

---

### Task 2: `fitted_diffusion` helper

**Files:**
- Create: `src/pde/internal/fitted_diffusion.hpp`
- Modify: `src/pde/internal/BUILD.bazel` (add header to the `workspace` target's `hdrs`)
- Create: `tests/internal/fitted_diffusion_test.cc`
- Modify: `tests/internal/BUILD.bazel`

**Interfaces:**
- Produces: `mango::operators::detail::FittedDiffusion { double a_f; double z; }` and
  `inline FittedDiffusion fitted_diffusion(double a, double b, double dx_left, double dx_right)`.
  Task 3 consumes both. (Free `detail::` function — not a private member —
  per the spec's plan-review-round-1 amendment: same discretization layer,
  directly unit-testable. Binding side is implied by `sign(b)`, not a
  returned field. `double`-only is deliberate: production instantiations
  are `double`; templatize only if a non-double `SpatialOperator` ever
  materializes.)

- [ ] **Step 1: Write the failing tests** — create `tests/internal/fitted_diffusion_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
//
// Unit tests for the Il'in exponentially-fitted diffusion helper (#472).
// The floating-point contract lives in
// docs/superpowers/specs/2026-08-31-drift-upwinding-472-design.md.

#include "mango/pde/internal/fitted_diffusion.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango::operators::detail {
namespace {

TEST(FittedDiffusionTest, ZeroDriftReturnsAExactly) {
    auto fd = fitted_diffusion(0.02, 0.0, 0.1, 0.2);
    EXPECT_EQ(fd.a_f, 0.02);  // bit-exact: LaplacianPDE path must not move
    EXPECT_EQ(fd.z, 0.0);
}

// Regression: sigma > 0 passes public validation but 0.5*sigma*sigma can
// underflow to 0. Bug guard: a == 0, b != 0 must hit the convection-limit
// branch (a_f = z), not divide by zero.
TEST(FittedDiffusionTest, ZeroDiffusionConvectionLimit) {
    auto fd = fitted_diffusion(0.0, 0.05, 0.1, 0.1);
    EXPECT_EQ(fd.z, 0.5 * 0.05 * 0.1);
    EXPECT_EQ(fd.a_f, fd.z);
}

TEST(FittedDiffusionTest, SeriesBranchNearIdentityForTinyDrift) {
    const double a = 0.02;
    auto fd = fitted_diffusion(a, 1e-8, 0.01, 0.01);
    EXPECT_GE(fd.a_f, a);
    EXPECT_NEAR(fd.a_f, a, a * 1e-12);
}

TEST(FittedDiffusionTest, SeriesAndDirectAgreeAtCutoff) {
    // rho = z/a straddling the 1e-4 series cutoff: branch must be seamless.
    const double a = 1.0;
    for (double rho : {0.99e-4, 1.01e-4}) {
        const double b = 2.0 * rho * a;  // h = 1 => z = b/2 = rho*a
        auto fd = fitted_diffusion(a, b, 1.0, 1.0);
        const double exact = fd.z / std::tanh(rho);
        EXPECT_NEAR(fd.a_f, exact, a * 1e-14) << "rho=" << rho;
    }
}

TEST(FittedDiffusionTest, LargeRhoApproachesZExactly) {
    // sigma = 1e-4 -> a = 5e-9; rho = z/a huge; tanh saturates to 1.
    const double a = 5e-9;
    const double b = 0.05;
    auto fd = fitted_diffusion(a, b, 1.0, 1.0);
    EXPECT_EQ(fd.z, 0.025);
    EXPECT_TRUE(std::isfinite(fd.a_f));
    EXPECT_GE(fd.a_f - fd.z, 0.0);   // binding numerator, exact in FP
    EXPECT_EQ(fd.a_f, fd.z);         // tanh(rho) == 1.0 here
}

TEST(FittedDiffusionTest, ClampInvariantsHoldAcrossSweep) {
    for (double a : {0.0, 5e-9, 5e-5, 0.005, 0.02, 0.08}) {
        for (double b : {-0.25, -0.03, -1e-6, 0.0, 1e-6, 0.03, 0.25}) {
            for (double dxl : {0.01, 0.1, 0.5}) {
                for (double dxr : {0.01, 0.1, 0.5}) {
                    auto fd = fitted_diffusion(a, b, dxl, dxr);
                    EXPECT_GE(fd.a_f, a);
                    EXPECT_GE(fd.a_f - fd.z, 0.0)
                        << "a=" << a << " b=" << b
                        << " dxl=" << dxl << " dxr=" << dxr;
                    EXPECT_TRUE(std::isfinite(fd.a_f));
                }
            }
        }
    }
}

TEST(FittedDiffusionTest, BindingSideFollowsDriftSign) {
    // b > 0 binds dx_right; b < 0 binds dx_left. Mirror symmetry:
    const double a = 0.005;
    auto pos = fitted_diffusion(a, 0.25, 0.05, 0.2);
    auto neg = fitted_diffusion(a, -0.25, 0.2, 0.05);
    EXPECT_EQ(pos.a_f, neg.a_f);
    EXPECT_EQ(pos.z, neg.z);
    EXPECT_EQ(pos.z, 0.5 * 0.25 * 0.2);  // dx_right for b > 0
}

// Guards the spec's C1-at-crossing claim on an asymmetric cell:
// continuity at b = 0 and the quadratic correction z^2/(3a) for small b.
TEST(FittedDiffusionTest, NearZeroDriftContinuityAndQuadraticCorrection) {
    const double a = 0.02, dxl = 0.05, dxr = 0.2;
    const double eps = 1e-12;
    auto minus = fitted_diffusion(a, -eps, dxl, dxr);
    auto zero  = fitted_diffusion(a, 0.0, dxl, dxr);
    auto plus  = fitted_diffusion(a, eps, dxl, dxr);
    EXPECT_NEAR(minus.a_f, zero.a_f, a * 1e-15);
    EXPECT_NEAR(plus.a_f, zero.a_f, a * 1e-15);

    const double b = 1e-3;  // small but in the quadratic regime
    auto fd = fitted_diffusion(a, b, dxl, dxr);
    const double z = 0.5 * b * dxr;
    EXPECT_NEAR(fd.a_f - a, z * z / (3.0 * a), 0.01 * z * z / (3.0 * a));
}

}  // namespace
}  // namespace mango::operators::detail
```

- [ ] **Step 2: Add the BUILD target** in `tests/internal/BUILD.bazel` (next to `spatial_operator_jacobian_test`):

```python
cc_test(
    name = "fitted_diffusion_test",
    size = "small",
    srcs = ["fitted_diffusion_test.cc"],
    deps = [
        "//src/pde/internal:workspace",
        "@googletest//:gtest_main",
    ],
)
```

- [ ] **Step 3: Run to verify failure**

Run: `bazel test //tests/internal:fitted_diffusion_test --test_output=errors`
Expected: FAIL to build — `mango/pde/internal/fitted_diffusion.hpp` not found.

- [ ] **Step 4: Implement** — create `src/pde/internal/fitted_diffusion.hpp`:

```cpp
// SPDX-License-Identifier: MIT
/**
 * @file fitted_diffusion.hpp
 * @brief Il'in exponentially-fitted diffusion coefficient (issue #472)
 *
 * Replaces the raw diffusion coefficient a = sigma^2/2 with
 * a_f = a * rho * coth(rho), rho = |b| * h_binding / (2a), where
 * h_binding is the neighbor spacing whose off-diagonal the drift can
 * flip (dx_right for b > 0, dx_left for b < 0). Guarantees
 * a_f >= max(a, |b| * h_binding / 2) exactly in floating point, which
 * makes both off-diagonals of the discrete spatial operator
 * non-negative for every sigma/h/b combination — the Z-matrix half of
 * the Brennan-Schwartz one-pass exactness requirement.
 *
 * Contract (binding; see the #472 design spec):
 *  - a < 0 is outside the contract (debug assert).
 *  - a == 0, b != 0 returns the representable convection limit a_f = z.
 *  - b == 0 returns exactly a (bit-exact pure-diffusion behavior).
 *  - Callers sample a and b once per assembly/apply invocation and pass
 *    them in; this function is pure.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>

namespace mango::operators::detail {

/// Fitted diffusion for one interior cell.
struct FittedDiffusion {
    double a_f;  ///< fitted diffusion coefficient; >= max(a, z) exactly
    double z;    ///< binding half-cell drift mass |b| * h_binding / 2
};

inline FittedDiffusion fitted_diffusion(double a, double b,
                                        double dx_left, double dx_right) {
    assert(a >= 0.0 && "negative diffusion is outside the fitting contract");
    const double h_binding = (b > 0.0) ? dx_right : dx_left;
    const double z = 0.5 * std::abs(b) * h_binding;
    if (z == 0.0) {
        return {a, 0.0};  // b == 0: pure diffusion, bit-exact passthrough
    }
    if (a == 0.0) {
        return {z, z};  // sigma^2/2 underflowed: exact convection limit
    }
    const double rho = z / a;
    double a_f;
    if (rho < 1e-4) {
        // rho*coth(rho) = 1 + rho^2/3 - rho^4/45 + ...; truncation error
        // <= rho^4/45 < 1e-17 relative at the cutoff.
        a_f = a * (1.0 + rho * rho / 3.0);
    } else {
        // a * rho * coth(rho) == z / tanh(rho); tanh saturates to 1 for
        // large rho, so a_f -> z with no overflow.
        a_f = z / std::tanh(rho);
    }
    // Exact-in-FP sign invariant: a_f - z >= 0 and a_f >= a.
    a_f = std::max({a_f, a, z});
    return {a_f, z};
}

}  // namespace mango::operators::detail
```

- [ ] **Step 5: Register the header** — in `src/pde/internal/BUILD.bazel`, add `"fitted_diffusion.hpp"` to the `hdrs` list of the `workspace` cc_library (keep alphabetical order if the list has one).

- [ ] **Step 6: Run to verify pass**

Run: `bazel test //tests/internal:fitted_diffusion_test --test_output=errors`
Expected: PASS (8 tests).

- [ ] **Step 7: Commit**

```bash
git add src/pde/internal/fitted_diffusion.hpp src/pde/internal/BUILD.bazel \
        tests/internal/fitted_diffusion_test.cc tests/internal/BUILD.bazel
git commit -m "Add Il'in fitted-diffusion helper for #472"
```

---

### Task 3: Fitted assembly + coefficient-combine path in SpatialOperator

**Files:**
- Modify: `src/pde/internal/spatial_operator.hpp` (`assemble_jacobian` ~lines 133–174, `apply_interior` ~lines 84–108, `HasJacobianCoefficients` doc comment ~lines 32–44)
- Modify: `tests/internal/spatial_operator_jacobian_test.cc` (expected values in `NonUniformFirstDerivativeConsistency`)
- Create: `tests/internal/spatial_operator_fitted_test.cc`
- Modify: `tests/internal/BUILD.bazel`

**Interfaces:**
- Consumes: `detail::fitted_diffusion(a, b, dx_left, dx_right) -> {a_f, z}` from Task 2.
- Produces: fitted `assemble_jacobian` and `apply_interior` (signatures unchanged); every later task relies on residual == Jacobian operator.

- [ ] **Step 1: Write the failing tests** — create `tests/internal/spatial_operator_fitted_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
//
// Tests for the Il'in-fitted drift discretization in SpatialOperator (#472):
// Z-matrix off-diagonal signs, sign-preserving assembly, coefficient-combine
// dispatch, and apply/Jacobian consistency.

#include "mango/pde/internal/spatial_operator.hpp"
#include "mango/pde/internal/fitted_diffusion.hpp"
#include "mango/pde/operators/black_scholes_pde.hpp"
#include "mango/pde/operators/laplacian_pde.hpp"
#include "mango/pde/operators/centered_difference.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/pde/internal/pde_workspace.hpp"
#include "mango/pde/internal/operator_factory.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace mango::operators {
namespace {

// Every local type here is `auto`-deduced, so this compiles for ANY PDE
// type (BlackScholesPDE, LaplacianPDE, CountingPDE) — a struct fixture
// would have to name create_spatial_operator's PDE-dependent return type.
template <typename PDE, typename Fn>
void with_operator(GridSpec<double> spec, PDE pde, Fn&& fn) {
    auto grid_buf = spec.generate();
    auto grid_view = grid_buf.view();
    auto spacing = std::make_shared<GridSpacing<double>>(grid_view);
    std::vector<double> buffer(PDEWorkspace::required_size(grid_view.size()));
    auto workspace = PDEWorkspace::from_buffer(buffer, grid_view.size()).value();
    auto op = create_spatial_operator(std::move(pde), spacing, workspace);
    fn(op, workspace, grid_view, spacing);
}

// Regression: high cell-Peclet drift flipped an off-diagonal sign (#472).
// Bug: centered drift outweighs diffusion when |b|*h/2 > a; the fitted
// coefficient must keep both off-diagonals of L non-negative.
TEST(SpatialOperatorFittedTest, OffDiagonalSignsHighPecletBothDriftSigns) {
    // sigma=1%, r=5%: b = 0.05 - 0 - 5e-5 > 0. Mirror with a dividend
    // yield of 10% to flip b negative. Sinh grid gives asymmetric cells.
    for (double div_yield : {0.0, 0.10}) {
        with_operator(
            GridSpec<double>::sinh_spaced(-2.0, 2.0, 21, 3.0).value(),
            BlackScholesPDE<double>(0.01, 0.05, div_yield),
            [&](auto& op, auto& workspace, auto& grid_view, auto&) {
                auto jac = workspace.jacobian();
                op.assemble_jacobian(0.0, 1.0, jac);  // J = I - L
                const size_t n = grid_view.size();
                for (size_t i = 1; i < n - 1; ++i) {
                    // Off-diagonals of L >= 0, i.e. jac.lower/upper <= 0.
                    EXPECT_LE(jac.lower()[i - 1], 0.0)
                        << "i=" << i << " q=" << div_yield;
                    EXPECT_LE(jac.upper()[i], 0.0)
                        << "i=" << i << " q=" << div_yield;
                }
            });
    }
}

// Regression: at overflow-scale Peclet, tanh(rho) == 1 makes a_f == z and
// the binding numerator must be assembled as literally a_f - z (== 0),
// not as two independently rounded terms that can go tiny-negative.
TEST(SpatialOperatorFittedTest, AssembledBindingEntryNonNegativeAtExtremeRho) {
    with_operator(
        GridSpec<double>::uniform(-5.0, 5.0, 11).value(),      // h = 1
        BlackScholesPDE<double>(1e-4, 0.05, 0.0),              // a = 5e-9
        [&](auto& op, auto& workspace, auto& grid_view, auto&) {
            auto jac = workspace.jacobian();
            op.assemble_jacobian(0.0, 1.0, jac);
            const size_t n = grid_view.size();
            for (size_t i = 1; i < n - 1; ++i) {
                EXPECT_TRUE(std::isfinite(jac.lower()[i - 1]));
                EXPECT_TRUE(std::isfinite(jac.diag()[i]));
                EXPECT_TRUE(std::isfinite(jac.upper()[i]));
                // b > 0 binds the lower entry; <= 0 exactly (>= 0 in L).
                EXPECT_LE(jac.lower()[i - 1], 0.0) << "i=" << i;
                EXPECT_LE(jac.upper()[i], 0.0) << "i=" << i;
            }
        });
}

// Direct matrix-row dominance inspection (spec: sufficient condition
// 1 + w*r > 0). Inspects the matrix, not a KKT-clean solve.
TEST(SpatialOperatorFittedTest, RowDominanceForPositiveAndModestNegativeRate) {
    for (double rate : {0.05, -0.01}) {
        with_operator(
            GridSpec<double>::sinh_spaced(-2.0, 2.0, 31, 3.0).value(),
            BlackScholesPDE<double>(0.20, rate, 0.02),
            [&](auto& op, auto& workspace, auto& grid_view, auto&) {
                auto jac = workspace.jacobian();
                const double w = 0.5;  // stage weight; 1 + w*r > 0 for both
                op.assemble_jacobian(0.0, w, jac);
                const size_t n = grid_view.size();
                for (size_t i = 1; i < n - 1; ++i) {
                    // Full M-matrix row structure: positive diagonal,
                    // non-positive off-diagonals, row sum 1 + w*r.
                    EXPECT_GT(jac.diag()[i], 0.0);
                    EXPECT_LE(jac.lower()[i - 1], 0.0)
                        << "rate=" << rate << " i=" << i;
                    EXPECT_LE(jac.upper()[i], 0.0)
                        << "rate=" << rate << " i=" << i;
                    const double row_sum =
                        jac.lower()[i - 1] + jac.diag()[i] + jac.upper()[i];
                    const double scale = std::abs(jac.lower()[i - 1]) +
                                         std::abs(jac.diag()[i]) +
                                         std::abs(jac.upper()[i]);
                    EXPECT_NEAR(row_sum, 1.0 + w * rate, scale * 1e-14 + 1e-15)
                        << "rate=" << rate << " i=" << i;
                }
            });
    }
}

// Spec test: assembly continuity across the drift-sign crossing on an
// asymmetric (sinh) grid — coefficients at b = ±eps must straddle b = 0
// continuously (guards the C1 claim at the operator level, not just in
// the helper). b = r − q − σ²/2 with σ=0.20, q=0.02 crosses 0 at r=0.04.
TEST(SpatialOperatorFittedTest, NearZeroDriftAssemblyContinuity) {
    const double eps = 1e-9;
    std::vector<std::vector<double>> lowers;
    for (double rate : {0.04 - eps, 0.04, 0.04 + eps}) {
        with_operator(
            GridSpec<double>::sinh_spaced(-2.0, 2.0, 15, 3.0).value(),
            BlackScholesPDE<double>(0.20, rate, 0.02),
            [&](auto& op, auto& workspace, auto& grid_view, auto&) {
                auto jac = workspace.jacobian();
                op.assemble_jacobian(0.0, 1.0, jac);
                std::vector<double> row;
                for (size_t i = 1; i < grid_view.size() - 1; ++i) {
                    row.push_back(jac.lower()[i - 1]);
                    row.push_back(jac.diag()[i]);
                    row.push_back(jac.upper()[i]);
                }
                lowers.push_back(std::move(row));
            });
    }
    ASSERT_EQ(lowers.size(), 3u);
    for (size_t k = 0; k < lowers[0].size(); ++k) {
        // eps-sized drift change moves coefficients by O(eps/h^2) at most;
        // 1e-4 absolute is orders looser than that but far tighter than
        // any discontinuity a binding-side switch bug would produce.
        EXPECT_NEAR(lowers[0][k], lowers[1][k], 1e-4) << "k=" << k;
        EXPECT_NEAR(lowers[2][k], lowers[1][k], 1e-4) << "k=" << k;
    }
}

// Spec test: LaplacianPDE's Jacobian must be unchanged by the fitting
// (b = 0 => a_f == a exactly). Off-diagonals assemble through the exact
// same expression as before (a/(dx*dx_avg)); the diagonal is rebuilt as
// c − lower − upper, numerically identical within rounding.
TEST(SpatialOperatorFittedTest, LaplacianJacobianUnchanged) {
    const double D = 0.1;
    with_operator(
        GridSpec<double>::sinh_spaced(0.0, 1.0, 17, 2.0).value(),
        LaplacianPDE<double>(D),
        [&](auto& op, auto& workspace, auto& grid_view, auto&) {
            auto jac = workspace.jacobian();
            op.assemble_jacobian(0.0, 1.0, jac);
            const auto& x = grid_view.span();
            for (size_t i = 1; i < grid_view.size() - 1; ++i) {
                const double dxl = x[i] - x[i - 1];
                const double dxr = x[i + 1] - x[i];
                const double dxa = (dxl + dxr) / 2.0;
                const double lower = D / (dxl * dxa);
                const double upper = D / (dxr * dxa);
                EXPECT_DOUBLE_EQ(jac.lower()[i - 1], -lower) << "i=" << i;
                EXPECT_DOUBLE_EQ(jac.upper()[i], -upper) << "i=" << i;
                EXPECT_NEAR(jac.diag()[i], 1.0 + lower + upper,
                            (1.0 + lower + upper) * 1e-15)
                    << "i=" << i;
            }
        });
}

// The functional consistency the projected LCP path needs: L*u assembled
// from the Jacobian must equal apply()'s Lu (scale-aware tolerance).
TEST(SpatialOperatorFittedTest, ApplyMatchesAssembledMatrix) {
    with_operator(
        GridSpec<double>::sinh_spaced(-2.0, 2.0, 25, 3.0).value(),
        BlackScholesPDE<double>(0.01, 0.05, 0.0),  // high Peclet
        [&](auto& op, auto& workspace, auto& grid_view, auto&) {
            auto jac = workspace.jacobian();
            op.assemble_jacobian(0.0, 1.0, jac);  // J = I - L

            const size_t n = grid_view.size();
            const auto& x = grid_view.span();
            std::vector<double> u(n), Lu(n, 0.0);
            for (size_t i = 0; i < n; ++i)
                u[i] = std::max(1.0 - std::exp(x[i]), 0.0);

            op.apply(0.0, u, Lu);

            double max_u = 0.0;
            for (double v : u) max_u = std::max(max_u, std::abs(v));
            for (size_t i = 1; i < n - 1; ++i) {
                const double Ju = jac.lower()[i - 1] * u[i - 1] +
                                  jac.diag()[i] * u[i] +
                                  jac.upper()[i] * u[i + 1];
                const double L_from_matrix = u[i] - Ju;  // L = I - J at w = 1
                const double row_mag = (std::abs(jac.lower()[i - 1]) +
                                        std::abs(jac.diag()[i]) +
                                        std::abs(jac.upper()[i])) * max_u;
                EXPECT_NEAR(Lu[i], L_from_matrix, row_mag * 1e-13 + 1e-14)
                    << "i=" << i;
            }
        });
}

// Companion on a UNIFORM grid: the uniform stencil uses the canonical
// stored spacing (GridSpacing::spacing()), so both assembly and apply
// must fit with that same h — per-cell coordinate diffs differ in the
// last ulp and would break this identity (plan-review round 2 finding).
TEST(SpatialOperatorFittedTest, ApplyMatchesAssembledMatrixUniformGrid) {
    with_operator(
        GridSpec<double>::uniform(-2.0, 2.0, 41).value(),
        BlackScholesPDE<double>(0.01, 0.05, 0.0),  // high Peclet
        [&](auto& op, auto& workspace, auto& grid_view, auto&) {
            auto jac = workspace.jacobian();
            op.assemble_jacobian(0.0, 1.0, jac);  // J = I - L

            const size_t n = grid_view.size();
            const auto& x = grid_view.span();
            std::vector<double> u(n), Lu(n, 0.0);
            for (size_t i = 0; i < n; ++i)
                u[i] = std::max(1.0 - std::exp(x[i]), 0.0);

            op.apply(0.0, u, Lu);

            double max_u = 0.0;
            for (double v : u) max_u = std::max(max_u, std::abs(v));
            for (size_t i = 1; i < n - 1; ++i) {
                const double Ju = jac.lower()[i - 1] * u[i - 1] +
                                  jac.diag()[i] * u[i] +
                                  jac.upper()[i] * u[i + 1];
                const double L_from_matrix = u[i] - Ju;
                const double row_mag = (std::abs(jac.lower()[i - 1]) +
                                        std::abs(jac.diag()[i]) +
                                        std::abs(jac.upper()[i])) * max_u;
                EXPECT_NEAR(Lu[i], L_from_matrix, row_mag * 1e-13 + 1e-14)
                    << "i=" << i;
            }
        });
}

// Dispatch guard: for HasJacobianCoefficients PDEs the coefficient methods
// are the authoritative operator definition; operator() must NOT be called
// on the apply path. Numerical agreement alone cannot prove the dispatch,
// so operator() is observable via a counter.
struct CountingPDE {
    static inline int op_calls = 0;
    double operator()(double, double d2u, double du, double u) const {
        ++op_calls;
        return 0.02 * d2u + 0.03 * du - 0.05 * u;
    }
    double operator()(double d2u, double du, double u) const {
        return (*this)(0.0, d2u, du, u);
    }
    double second_derivative_coeff() const { return 0.02; }
    double first_derivative_coeff(double = 0.0) const { return 0.03; }
    double discount_rate(double = 0.0) const { return 0.05; }
};

TEST(SpatialOperatorFittedTest, CoefficientPathDoesNotCallOperator) {
    static_assert(HasJacobianCoefficients<CountingPDE>);
    with_operator(
        GridSpec<double>::uniform(-1.0, 1.0, 11).value(), CountingPDE{},
        [&](auto& op, auto&, auto& grid_view, auto&) {
            const size_t n = grid_view.size();
            std::vector<double> u(n, 1.0), Lu(n, 0.0);
            CountingPDE::op_calls = 0;
            op.apply(0.0, u, Lu);
            EXPECT_EQ(CountingPDE::op_calls, 0);
        });
}

// Laplacian equality is numerical, not bit-pattern (spec): the combine
// path a*d2u + 0*du - 0*u must equal D*d2u.
TEST(SpatialOperatorFittedTest, LaplacianCombinePathNumericallyIdentical) {
    const double D = 0.1;
    with_operator(
        GridSpec<double>::sinh_spaced(0.0, 1.0, 17, 2.0).value(),
        LaplacianPDE<double>(D),
        [&](auto& op, auto&, auto& grid_view, auto& spacing) {
            const size_t n = grid_view.size();
            const auto& x = grid_view.span();
            std::vector<double> u(n), Lu(n, 0.0), d2u(n, 0.0);
            for (size_t i = 0; i < n; ++i) u[i] = std::sin(3.0 * x[i]);

            op.apply(0.0, u, Lu);

            CenteredDifference<double> stencil(*spacing);
            stencil.compute_second_derivative(u, d2u, 1, n - 1);
            for (size_t i = 1; i < n - 1; ++i) {
                EXPECT_DOUBLE_EQ(Lu[i], D * d2u[i]) << "i=" << i;
            }
        });
}

}  // namespace
}  // namespace mango::operators
```

Note: `with_operator` uses only `auto` locals, so no grid/operator type names are spelled out; if any construction detail differs from the real API, copy the exact setup lines from `spatial_operator_jacobian_test.cc:28-50` into the helper — the assertions are the deliverable, not the fixture shape.

- [ ] **Step 2: Add BUILD target** in `tests/internal/BUILD.bazel`:

```python
cc_test(
    name = "spatial_operator_fitted_test",
    size = "small",
    srcs = ["spatial_operator_fitted_test.cc"],
    deps = [
        "//src/pde/internal:workspace",
        "//src/pde/operators:black_scholes_pde",
        "//src/pde/operators:centered_difference",
        "//src/pde/operators:laplacian_pde",
        "//src/pde/core:grid",
        "@googletest//:gtest_main",
    ],
)
```

(`//src/pde/internal:workspace` lists `centered_difference` under `deps`,
not `exports`, so the direct dep is required for the Laplacian test's
include — src/pde/operators/BUILD.bazel:28 has the target.)

- [ ] **Step 3: Run to verify failure**

Run: `bazel test //tests/internal:spatial_operator_fitted_test --test_output=errors`
Expected: FAIL — `OffDiagonalSignsHighPecletBothDriftSigns` and `AssembledBindingEntryNonNegativeAtExtremeRho` fail (positive off-diagonals in L today); `CoefficientPathDoesNotCallOperator` fails (operator() is called today).

- [ ] **Step 4: Implement in `src/pde/internal/spatial_operator.hpp`.**

4a. Add `#include "mango/pde/internal/fitted_diffusion.hpp"` to the includes.

4b. Extend the `HasJacobianCoefficients` doc comment (keep the concept body unchanged) — append:

```cpp
/// CONTRACT (#472): for any PDE satisfying this concept, the coefficient
/// methods are the AUTHORITATIVE definition of the interior operator:
/// SpatialOperator evaluates L(u) = a_f·u'' + b(t)·u' − r(t)·u from them
/// (with Il'in-fitted a_f; see fitted_diffusion.hpp) and never calls
/// operator() on interior nodes. Accessors must be pure (deterministic,
/// side-effect free) at fixed t, and a = second_derivative_coeff() must
/// be >= 0 unconditionally (a < 0 is outside the fitting contract; a == 0
/// is supported as the convection limit — see fitted_diffusion.hpp).
```

4c. Replace the body of the interior loop in `assemble_jacobian` (keep signature and the boundary-row note):

```cpp
        // Sample coefficients ONCE per invocation (concept contract: pure
        // at fixed t). The per-node quantity is the fitted diffusion only.
        const T a = pde_.second_derivative_coeff();   // σ²/2
        const T b = pde_.first_derivative_coeff(t);   // r(t) - d - σ²/2
        const T c = -pde_.discount_rate(t);           // -r(t)

        const size_t n = jac.size();
        const auto& grid = spacing_->grid();
        // Uniform grids: use the canonical stored spacing, NOT per-cell
        // coordinate differences — CenteredDifference's uniform stencil
        // uses spacing_->spacing() for every derivative, and per-cell
        // diffs differ from it in the last ulp, which would break the
        // exact apply/Jacobian identity the fitted scheme requires.
        const bool uniform = spacing_->is_uniform();
        const T h_uniform = uniform ? spacing_->spacing() : T(0);

        for (size_t i = 1; i < n - 1; ++i) {
            const T dx_left = uniform ? h_uniform : grid[i] - grid[i-1];
            const T dx_right = uniform ? h_uniform : grid[i+1] - grid[i];
            const T dx_avg = (dx_left + dx_right) / 2.0;

            const auto fd = detail::fitted_diffusion(a, b, dx_left, dx_right);

            // Sign-preserving reduced assembly (#472): the binding-side
            // numerator is computed literally as a_f − z, which the
            // helper's clamp keeps >= 0 exactly in floating point. The
            // non-binding numerator is a sum of non-negatives. This
            // replaces the separate d2+d1 coefficient accumulation, which
            // could round the binding entry to a tiny negative at high
            // cell Péclet.
            T num_lower, num_upper;
            if (b >= 0.0) {
                num_lower = fd.a_f - fd.z;               // binding (z = b·dx_r/2)
                num_upper = fd.a_f + T(0.5) * b * dx_left;
            } else {
                num_lower = fd.a_f - T(0.5) * b * dx_right;  // adds |b|·dx_r/2
                num_upper = fd.a_f - fd.z;               // binding (z = |b|·dx_l/2)
            }
            const T lower = num_lower / (dx_left * dx_avg);
            const T upper = num_upper / (dx_right * dx_avg);
            const T diag = c - lower - upper;  // row-sum identity by construction

            jac.lower()[i - 1] = -coeff_dt * lower;
            jac.diag()[i] = 1.0 - coeff_dt * diag;
            jac.upper()[i] = -coeff_dt * upper;
        }
```

4d. In `apply_interior`, replace the final combine loop with a
coefficient-dispatch version (the stencil computation of `d2u`/`du` stays):

```cpp
        if constexpr (HasJacobianCoefficients<PDE>) {
            // Coefficient-combine path (#472): coefficients are the
            // authoritative operator definition (concept contract), and the
            // fitted diffusion keeps this residual/RHS evaluation the SAME
            // operator the assembled Jacobian represents — the projected
            // LCP stage solves A·u = rhs with A from assemble_jacobian, so
            // the two paths must not diverge.
            const T a = pde_.second_derivative_coeff();
            const T b = pde_.first_derivative_coeff(t);
            const T r = pde_.discount_rate(t);
            const auto& grid = spacing_->grid();
            if (spacing_->is_uniform()) {
                // Canonical stored spacing — must match assemble_jacobian's
                // uniform branch and the stencil (NOT grid[1] - grid[0],
                // which differs in the last ulp).
                const T h = spacing_->spacing();
                const T a_f = detail::fitted_diffusion(a, b, h, h).a_f;
                for (size_t i = start; i < end; ++i) {
                    Lu[i] = a_f * d2u[i] + b * du[i] - r * u[i];
                }
            } else {
                for (size_t i = start; i < end; ++i) {
                    const T dx_left = grid[i] - grid[i-1];
                    const T dx_right = grid[i+1] - grid[i];
                    const T a_f =
                        detail::fitted_diffusion(a, b, dx_left, dx_right).a_f;
                    Lu[i] = a_f * d2u[i] + b * du[i] - r * u[i];
                }
            }
        } else {
            for (size_t i = start; i < end; ++i) {
                if constexpr (TimeDependentPDE<PDE>) {
                    Lu[i] = pde_(t, d2u[i], du[i], u[i]);
                } else {
                    Lu[i] = pde_(d2u[i], du[i], u[i]);
                }
            }
        }
```

- [ ] **Step 5: Update `tests/internal/spatial_operator_jacobian_test.cc`** — in `NonUniformFirstDerivativeConsistency` (lines ~70–88), the expected coefficients must use the fitted diffusion and reduced assembly. Add `#include "mango/pde/internal/fitted_diffusion.hpp"` and replace the expected-coefficient computation:

```cpp
        // #472: expected coefficients use the Il'in-fitted diffusion and the
        // sign-preserving reduced assembly (see fitted_diffusion.hpp).
        auto fd = detail::fitted_diffusion(a, b, dx_left, dx_right);
        double num_lower, num_upper;
        if (b >= 0.0) {
            num_lower = fd.a_f - fd.z;
            num_upper = fd.a_f + 0.5 * b * dx_left;
        } else {
            num_lower = fd.a_f - 0.5 * b * dx_right;
            num_upper = fd.a_f - fd.z;
        }
        double expected_lower = num_lower / (dx_left * dx_avg);
        double expected_upper = num_upper / (dx_right * dx_avg);
        double expected_diag  = c - expected_lower - expected_upper;
```

(Delete the now-unused `d2_*`/`d1_*` locals and `d1_denom`; keep the EXPECT_NEAR comparisons and messages. Tolerances may need loosening from `1e-14` to relative form `std::abs(expected_*) * 1e-14 + 1e-14` because expected and actual now compute through the identical algebra — if they match exactly, keep 1e-14.)

- [ ] **Step 6: Run the operator tests**

Run: `bazel test //tests/internal:spatial_operator_fitted_test //tests/internal:spatial_operator_jacobian_test //tests/internal:fitted_diffusion_test --test_output=errors`
Expected: all PASS (the existing `JacobianMatchesApplyFiniteDifference` now exercises the fitted path on both sides).

- [ ] **Step 7: Document the unfitted boundary rows in code.** In `spatial_operator.hpp`, extend the doc comment above `boundary_row_jacobian` with:

```cpp
    /// NOTE (#472): boundary rows deliberately use the raw (unfitted)
    /// diffusion coefficient a. Their ghost-eliminated off-diagonal
    /// +2a/h² already has the Z-matrix sign for any drift b (drift enters
    /// only the affine term), and each row's eval == jacobian·u + affine
    /// identity is internal to the row, so residual/Jacobian consistency
    /// is unaffected by the interior fitting.
```

- [ ] **Step 8: Run the FULL suite** (the FDM path feeds QuantLib A/B, convergence, IV, Greeks, and price-table tests — a subset run is not enough to bound this change)

Run: `bazel test //... --test_output=errors`
Expected: everything passes EXCEPT `AmericanOptionTest.NoDivCallPriceUnchangedByEnvelopeBC` (deliberate re-pin, next step). Any OTHER failure is unplanned churn — STOP and investigate; do not loosen tolerances without written justification.

- [ ] **Step 9: Re-pin `NoDivCallPriceUnchangedByEnvelopeBC`** (must land in the same commit as the discretization change so no commit leaves the suite red).

Run: `bazel test //tests:american_option_test --test_output=all --test_filter='*NoDivCallPriceUnchanged*' 2>&1 | grep -E "Actual|Which is|difference"`

Take the new actual value from the failure output; in `tests/american_option_test.cc` (~line 527) set `kPinnedPrice` to it and replace the pin's comment paragraph about toolchain sensitivity with:

```cpp
    // Re-pinned 2026-08-31 for #472: the Il'in-fitted drift discretization
    // deliberately perturbs every FDM solve by O(ρ²) added diffusion (see
    // docs/superpowers/specs/2026-08-31-drift-upwinding-472-design.md).
    // Previous pin: 10.447090628631905. The 1e-12 tolerance remains
    // toolchain-sensitive (FP reassociation) — same precedent as PR #468's
    // x86-64-v3 pin. On an unexplained failure, verify the toolchain (not
    // the discretization) changed before re-pinning.
```

Record the old→new delta in `docs/superpowers/plans/2026-08-31-drift-upwinding-472-baseline.md` (spec-required measurement).

- [ ] **Step 10: Re-run the full suite — must be green**

Run: `bazel test //... --test_output=errors`
Expected: ALL tests pass.

- [ ] **Step 11: Commit**

```bash
git add src/pde/internal/spatial_operator.hpp \
        tests/internal/spatial_operator_jacobian_test.cc \
        tests/internal/spatial_operator_fitted_test.cc tests/internal/BUILD.bazel \
        tests/american_option_test.cc \
        docs/superpowers/plans/2026-08-31-drift-upwinding-472-baseline.md
git commit -m "Apply Il'in fitting in SpatialOperator assembly and apply"
```

---

### Task 4: γ validation in PDESolver::solve()

**Files:**
- Modify: `src/pde/internal/pde_solver.hpp` (top of `solve()`, ~line 141)
- Modify: `tests/internal/pde_solver_test.cc`

**Interfaces:**
- Consumes: existing `SolverErrorCode::InvalidConfiguration` (src/support/error_types.hpp:17).
- Produces: `solve()` rejects `γ ∉ (0,1)` with `SolverError{InvalidConfiguration, .iterations = 0, .residual = <supplied γ>}`.

- [ ] **Step 1: Write the failing test** — append to `tests/internal/pde_solver_test.cc` (reuse the file's `TestPDESolver` helper and the HeatEquationDirichletBC setup pattern at lines 43–97):

```cpp
// Regression: TR-BDF2 stage weights w1 = γ·dt/2 and w2 = (1−γ)·dt/(2−γ)
// are positive only for γ in (0,1); outside it the stage matrix loses the
// Z-matrix premise of #472's M-matrix guarantee.
// Bug guard: set_config accepted any TRBDF2Config silently.
TEST(PDESolverTest, InvalidGammaRejectedAtSolve) {
    const double D = 0.1;
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time = mango::TimeDomain::from_n_steps(0.0, 0.1, 10);
    auto grid = mango::Grid<double>::create(grid_spec, time).value();

    std::pmr::monotonic_buffer_resource pool;
    size_t buffer_size = mango::PDEWorkspace::required_size(grid->n_space());
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, &pool);
    auto workspace = mango::PDEWorkspace::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()},
        grid->x(), grid->n_space()).value();

    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto pde = mango::operators::LaplacianPDE<double>(D);
    auto spacing = std::make_shared<mango::GridSpacing<double>>(grid->spacing());
    auto op = mango::operators::create_spatial_operator(std::move(pde), spacing,
                                                        workspace);

    for (double gamma : {0.0, 1.0, 1.5, -0.5,
                         std::numeric_limits<double>::quiet_NaN()}) {
        auto solver = TestPDESolver(grid, workspace, left_bc, right_bc, op);
        mango::TRBDF2Config config;
        config.gamma = gamma;
        solver.set_config(config);
        solver.initialize([](std::span<const double>, std::span<double> u) {
            std::fill(u.begin(), u.end(), 0.0);
        });
        auto status = solver.solve();
        ASSERT_FALSE(status.has_value()) << "gamma=" << gamma;
        EXPECT_EQ(status.error().code,
                  mango::SolverErrorCode::InvalidConfiguration)
            << "gamma=" << gamma;
        EXPECT_EQ(status.error().iterations, 0u);
        if (std::isnan(gamma)) {
            EXPECT_TRUE(std::isnan(status.error().residual));
        } else {
            EXPECT_EQ(status.error().residual, gamma);
        }
    }
}
```

(Add `#include <limits>` if missing. If `TestPDESolver` is in an anonymous namespace above, place this test in the same file scope as the existing TESTs.)

- [ ] **Step 2: Run to verify failure**

Run: `bazel test //tests/internal:pde_solver_test --test_output=errors --test_filter='*InvalidGamma*'`
Expected: FAIL — solve() currently succeeds (or diverges) instead of returning InvalidConfiguration.

- [ ] **Step 3: Implement** — in `PDESolver::solve()`, immediately AFTER `lcp_report_ = LcpKktReport{};` (the report reset must run even on the rejection path, or a reused solver would return a stale report from its previous solve) and before any stepping/state mutation:

```cpp
        // #472: TR-BDF2 stage weights w1 = γ·dt/2, w2 = (1−γ)·dt/(2−γ) are
        // positive — the Z-matrix premise of the fitted discretization's
        // M-matrix guarantee — only for γ in (0,1). Reject anything else.
        if (!(std::isfinite(config_.gamma) &&
              config_.gamma > 0.0 && config_.gamma < 1.0)) {
            return std::unexpected(SolverError{
                .code = SolverErrorCode::InvalidConfiguration,
                .iterations = 0,
                .residual = config_.gamma});
        }
```

(`<cmath>` is already included; verify, add if not.)

- [ ] **Step 4: Run to verify pass**

Run: `bazel test //tests/internal:pde_solver_test --test_output=errors`
Expected: PASS (all tests in the file, including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/pde/internal/pde_solver.hpp tests/internal/pde_solver_test.cc
git commit -m "Validate TR-BDF2 gamma at solve() entry"
```

---

### Task 5: Public regression tests (canonical fixture, sweep, sign crossing)

**Files:**
- Modify: `tests/american_option_test.cc`

**Interfaces:**
- Consumes: fitted solver from Tasks 3–4; baseline numbers from Task 1's `docs/superpowers/plans/2026-08-31-drift-upwinding-472-baseline.md`.

- [ ] **Step 1: Add the canonical regression + sweep** (new section at the end of `tests/american_option_test.cc`, after the #439/#455 regression block). Fill `<COUNT>`, `<KIND>` from Task 1's baseline doc:

```cpp
// ===========================================================================
// Regression tests for #472: Il'in-fitted drift discretization (M-matrix)
// ===========================================================================

// Regression: high cell-Péclet drift broke the M-matrix property (#472).
// Bug: centered drift at σ=1%, h=0.1 flips a stage off-diagonal sign, so
// the one-pass projected Thomas sweep is inexact and validate_lcp_kkt
// reports violations. Pre-fix baseline (recorded 2026-08-31, task 1 of
// docs/superpowers/plans/2026-08-31-drift-upwinding-472.md):
// violation_count = <COUNT>, worst_kind = <KIND>.
TEST(AmericanOptionTest, HighPecletComplementarityCleanAfterFitting) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.0,
                   .option_type = OptionType::PUT},
        0.01);

    PDEGridConfig grid_config{
        .grid_spec = GridSpec<double>::uniform(-2.0, 2.0, 41).value(),
        .n_time = 50};
    auto solver = AmericanOptionSolver::create(params, grid_config);
    ASSERT_TRUE(solver.has_value());
    auto result = solver->solve();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(solver->complementarity_report().violation_count, 0u)
        << "max_violation=" << solver->complementarity_report().max_violation
        << " worst_kind=" << solver->complementarity_report().worst_kind;
}

// Fixed sweep table around the canonical fixture (spec Testing item 1).
TEST(AmericanOptionTest, LowVolCoarseGridComplementaritySweep) {
    for (double sigma : {0.005, 0.01, 0.02}) {
        for (size_t n_space : {41u, 81u}) {   // h = 0.1, 0.05
            for (double rate : {0.02, 0.05}) {
                PricingParams params(
                    OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                               .rate = rate, .dividend_yield = 0.0,
                               .option_type = OptionType::PUT},
                    sigma);
                PDEGridConfig grid_config{
                    .grid_spec =
                        GridSpec<double>::uniform(-2.0, 2.0, n_space).value(),
                    .n_time = 50};
                auto solver = AmericanOptionSolver::create(params, grid_config);
                ASSERT_TRUE(solver.has_value());
                auto result = solver->solve();
                ASSERT_TRUE(result.has_value())
                    << "sigma=" << sigma << " n=" << n_space << " r=" << rate;
                EXPECT_EQ(solver->complementarity_report().violation_count, 0u)
                    << "sigma=" << sigma << " n=" << n_space << " r=" << rate
                    << " worst_kind="
                    << solver->complementarity_report().worst_kind;
            }
        }
    }
}

// Guards the per-t binding-side re-derivation: b(t) = r(t) − q − σ²/2
// crosses zero during the solve (σ=10% ⇒ σ²/2 = 0.005; q = 2%; curve
// rates straddle 2.5%).
TEST(AmericanOptionTest, DriftSignCrossingSolvesCleanly) {
    // TenorPoint stores ln(D(t)) = -∫r ds, NOT a rate (yield_curve.hpp).
    // Forward rates: 1% on [0, 0.25] (log-discount -0.0025), 4% on
    // [0.25, 2.0] (-0.0025 - 0.04*1.75 = -0.0725). Same construction
    // pattern as AmericanOptionPricingTest.PricingWithYieldCurve
    // (tests/american_option_test.cc:~185).
    std::vector<TenorPoint> points = {
        {0.0, 0.0}, {0.25, -0.0025}, {2.0, -0.0725}};
    auto curve = YieldCurve::from_points(points).value();
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = curve, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT},
        0.10);

    PDEGridConfig grid_config{
        .grid_spec = GridSpec<double>::sinh_spaced(-2.0, 2.0, 101, 3.0).value(),
        .n_time = 200};
    auto solver = AmericanOptionSolver::create(params, grid_config);
    ASSERT_TRUE(solver.has_value());
    auto result = solver->solve();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(solver->complementarity_report().violation_count, 0u)
        << "worst_kind=" << solver->complementarity_report().worst_kind;
    EXPECT_TRUE(std::isfinite(result->value_at(100.0)));
}
```

(If `41u` literals fight the `size_t` loop type, use `size_t n_space : {size_t{41}, size_t{81}}`.)

- [ ] **Step 2: Run the new tests**

Run: `bazel test //tests:american_option_test --test_output=errors --test_filter='*HighPeclet*:*LowVolCoarse*:*DriftSignCrossing*'`
Expected: PASS. **The sweep table is immutable — never delete or narrow a cell.** If a cell fails with nonzero count and an active-set (non-interval) `worst_kind` WHILE the Task 3 matrix-sign tests pass at the same parameters, that is a genuine #473 finding: STOP, record the cell and report, and surface it to the user before proceeding. Any other failure is our bug — investigate.

(The `NoDivCallPriceUnchangedByEnvelopeBC` re-pin already happened in Task 3 — same commit as the discretization change, so no commit boundary is red.)

- [ ] **Step 3: Run the full american_option_test**

Run: `bazel test //tests:american_option_test --test_output=errors`
Expected: PASS (all).

- [ ] **Step 4: Commit**

```bash
git add tests/american_option_test.cc
git commit -m "Add #472 KKT regression and sweep tests"
```

---

### Task 6: Docs, full verification, measurements

**Files:**
- Modify: `docs/MATHEMATICAL_FOUNDATIONS.md` (the M-matrix/cell-Péclet caveat added by #475 — find it: `grep -n "Péclet\|M-matrix" docs/MATHEMATICAL_FOUNDATIONS.md`)
- Modify: `docs/superpowers/plans/2026-08-31-drift-upwinding-472-baseline.md` (final measurements)

- [ ] **Step 1: Rewrite the caveat.** Replace the existing cell-Péclet caveat text with (adapt surrounding prose to fit the section's voice; keep LaTeX/math style consistent with the file):

```markdown
**Drift discretization (issue #472).** The drift term uses an Il'in
exponentially-fitted diffusion coefficient: per interior cell,
a_f = a·ρ·coth(ρ) with ρ = |b|·h_binding/(2a), where h_binding is the
neighbor spacing on the side the drift can flip (dx_right for b > 0,
dx_left for b < 0). Since ρ·coth(ρ) ≥ max(1, ρ), the fitted off-diagonals
of L are non-negative for every σ/h/b combination — the cell-Péclet
failure mode of the former centered drift stencil is removed
unconditionally, at the cost of O(ρ²) added numerical diffusion (the
scheme stays second-order consistent on refining grids and degrades
gracefully to upwind-like behavior in the convection-dominated limit).

One-pass Brennan-Schwartz exactness still requires, beyond these
off-diagonal signs: (1) strict row dominance 1 + w·r(t) > 0 — automatic
for r ≥ 0 and violated only at absurd negative rates r ≤ −1/w ≈ −2/Δt,
which the API accepts and therefore remains a documented caveat (the
solver does validate the TR-BDF2 γ ∈ (0,1), so stage weights are always
positive); and (2) an interval active set touching the sweep's starting
side (issue #473). `validate_lcp_kkt` checks the solved system's KKT
conditions — it detects resulting solution defects, not matrix structure,
so a clean report is expected but not a structural guarantee.

The ghost-eliminated boundary rows deliberately retain the raw
(unfitted) diffusion coefficient a: their off-diagonal +2a/h² already
has the required sign for any drift (drift enters those rows only
through the affine term), so fitting them would add diffusion without
buying any structural property.
```

- [ ] **Step 2: Full test suite**

Run: `bazel test //... --test_output=errors`
Expected: **all tests pass** (148 targets at baseline; now +2 new targets). Any failure not named in this plan is a defect — fix before proceeding, do not loosen tolerances without written justification in the commit.

- [ ] **Step 3: CI-parity builds**

Run: `bazel build //benchmarks/...` then `bazel build //src/python:mango_option`
Expected: both succeed, no warnings from project code.

- [ ] **Step 4: Record measurements.** Append to the baseline doc: the re-pin delta from Task 5, and the ATM-put-vs-QuantLib pin's current margin (from the accuracy test's output if it prints one, else note "within pinned bound, unchanged assertion"). These are the spec's "estimates to be measured" numbers.

- [ ] **Step 5: Commit**

```bash
git add docs/MATHEMATICAL_FOUNDATIONS.md \
        docs/superpowers/plans/2026-08-31-drift-upwinding-472-baseline.md
git commit -m "Document Il'in-fitted drift scheme and narrowed caveats"
```

---

## Self-review notes (already applied)

- Spec coverage: helper contract → Task 2; sign-preserving assembly + combine + dispatch/concept docs + boundary-row code comment + deliberate re-pin (same commit as the change, so every commit boundary is green) → Task 3; γ validation → Task 4; canonical fixture + immutable sweep + sign-crossing → Task 5 (fixture measurement → Task 1); docs caveat (incl. unfitted-boundary-rows sentence) + measurements → Task 6.
- Plan-review round 1 folds: `with_operator` auto-deduced fixture (the struct fixture couldn't name PDE-dependent operator types); yield-curve test uses `YieldCurve::from_points` with log-discounts (`TenorPoint.log_discount`, mandatory `{0,0}` point); γ check inserted AFTER the `lcp_report_` reset; full-suite run before the Task 3 commit; direct `centered_difference` BUILD dep; Task 1 probe-revert guard; concept doc requires `a ≥ 0` unconditionally.
- Plan-review round 2 fold: on uniform grids BOTH `assemble_jacobian` and `apply_interior` fit with the canonical `spacing_->spacing()` (the h `CenteredDifference`'s uniform stencil uses), never per-cell coordinate diffs, which differ in the last ulp; `ApplyMatchesAssembledMatrixUniformGrid` guards it.
- Perf risk (spec "Risks"): the uniform-grid fast path in Task 3 computes the fitted factor once per invocation; per-stage caching stays deferred unless the full suite's performance tests regress (none pin apply-path latency; if `//tests:american_option_performance_test` exists and fails, that triggers the spec's caching mitigation as a follow-up commit).
- Type consistency: `FittedDiffusion{a_f, z}` and `fitted_diffusion(a, b, dx_left, dx_right)` used identically in Tasks 2, 3; γ error payload `{InvalidConfiguration, 0, gamma}` consistent between Task 4 test and implementation.
