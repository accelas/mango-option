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

// PRIMARY #472 regression: the issue's exact fixture (PUT ATM, r=5%, q=0,
// sigma=1%, uniform h=0.1 on [-2,2]). Pre-fix baseline measured 2026-09-01
// (docs/superpowers/plans/2026-08-31-drift-upwinding-472-baseline.md):
// jac.lower = +0.24475 on every interior row (L's lower off-diagonal
// NEGATIVE — the M-matrix break), diag ≈ 1.06, upper ≈ -0.25475. The
// full-solve KKT report is clean pre-fix (deep-ITM identity-row lock +
// magnitude dominance mask it), so THIS structural assertion is what
// attributes the repair to the discretization.
TEST(SpatialOperatorFittedTest, IssueFixtureUniformGridOffDiagonalSigns) {
    with_operator(
        GridSpec<double>::uniform(-2.0, 2.0, 41).value(),
        BlackScholesPDE<double>(0.01, 0.05, 0.0),
        [&](auto& op, auto& workspace, auto& grid_view, auto&) {
            auto jac = workspace.jacobian();
            op.assemble_jacobian(0.0, 1.0, jac);  // J = I - L
            const size_t n = grid_view.size();
            for (size_t i = 1; i < n - 1; ++i) {
                EXPECT_LE(jac.lower()[i - 1], 0.0) << "i=" << i;   // was +0.24475
                EXPECT_LE(jac.upper()[i], 0.0) << "i=" << i;
                EXPECT_GT(jac.diag()[i], 0.0) << "i=" << i;
            }
        });
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
    std::vector<std::vector<double>> rows;
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
                rows.push_back(std::move(row));
            });
    }
    ASSERT_EQ(rows.size(), 3u);
    for (size_t k = 0; k < rows[0].size(); ++k) {
        // This is a continuity check across b=0 AT THE OPERATOR LEVEL only
        // (mirroring fitted_diffusion's own continuity contract): an
        // eps-sized drift change should move assembled coefficients by
        // O(eps/h^2) at most, and 1e-4 absolute is orders looser than
        // that. It is not a sensitive-enough tolerance to catch every
        // binding-side selection bug (e.g. an inverted b>=0 branch would
        // still pass at eps=1e-9) -- that is covered separately by the
        // sign-preserving assembly tests above.
        EXPECT_NEAR(rows[0][k], rows[1][k], 1e-4) << "k=" << k;
        EXPECT_NEAR(rows[2][k], rows[1][k], 1e-4) << "k=" << k;
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

// Shared body for the apply-vs-assembled-matrix consistency check: L*u
// assembled from the Jacobian must equal apply()'s Lu (scale-aware
// tolerance). Factored out since the sinh-grid and uniform-grid variants
// below are identical apart from the GridSpec.
void check_apply_matches_assembled_matrix(GridSpec<double> spec) {
    with_operator(
        spec,
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

// The functional consistency the projected LCP path needs: L*u assembled
// from the Jacobian must equal apply()'s Lu (scale-aware tolerance).
TEST(SpatialOperatorFittedTest, ApplyMatchesAssembledMatrix) {
    check_apply_matches_assembled_matrix(
        GridSpec<double>::sinh_spaced(-2.0, 2.0, 25, 3.0).value());
}

// Companion on a UNIFORM grid: exercises the canonical-stored-spacing
// (GridSpacing::spacing()) branch that both assembly and apply take on a
// uniform grid, instead of the sinh grid's per-cell coordinate diffs
// (plan-review round 2 finding). Note this tolerance (row_mag * 1e-13)
// would NOT by itself catch a last-ulp h mismatch between the two
// paths — that perturbs Lu by ~1e-16, well under the bound — so this
// test documents which branch is exercised rather than asserting the
// last-ulp identity.
TEST(SpatialOperatorFittedTest, ApplyMatchesAssembledMatrixUniformGrid) {
    check_apply_matches_assembled_matrix(
        GridSpec<double>::uniform(-2.0, 2.0, 41).value());
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

// Regression (#472 fix round 1): the per-node fitted-diffusion cache
// (ensure_fitted_cache/a_f_cache) must invalidate when the sampled drift
// b changes (callable-rate PDE) and must reuse cached values -- giving
// bit-identical output -- when (a, b) are unchanged.
TEST(SpatialOperatorFittedTest, FittedCacheInvalidatesOnRateChange) {
    auto rate_fn = [](double t) { return t < 0.5 ? 0.05 : 0.09; };
    with_operator(
        GridSpec<double>::sinh_spaced(-2.0, 2.0, 21, 3.0).value(),
        BlackScholesPDE(0.01, rate_fn, 0.0),
        [&](auto& op, auto& workspace, auto& grid_view, auto&) {
            const size_t n = grid_view.size();
            auto jac = workspace.jacobian();

            op.assemble_jacobian(0.0, 1.0, jac);
            std::vector<double> lower_t0(jac.lower().begin(), jac.lower().end());
            std::vector<double> upper_t0(jac.upper().begin(), jac.upper().end());
            std::vector<double> diag_t0(jac.diag().begin(), jac.diag().end());

            // Re-assemble at the SAME t (cache hit): bit-identical output.
            op.assemble_jacobian(0.0, 1.0, jac);
            for (size_t i = 1; i < n - 1; ++i) {
                EXPECT_EQ(jac.lower()[i - 1], lower_t0[i - 1]) << "i=" << i;
                EXPECT_EQ(jac.upper()[i], upper_t0[i]) << "i=" << i;
                EXPECT_EQ(jac.diag()[i], diag_t0[i]) << "i=" << i;
            }

            // Assemble at a DIFFERENT t (rate steps 0.05 -> 0.09 at t=0.5):
            // the cache must invalidate, so at least one interior
            // off-diagonal must differ from the t=0 assembly.
            op.assemble_jacobian(1.0, 1.0, jac);
            bool any_differs = false;
            for (size_t i = 1; i < n - 1; ++i) {
                if (jac.lower()[i - 1] != lower_t0[i - 1] ||
                    jac.upper()[i] != upper_t0[i]) {
                    any_differs = true;
                }
            }
            EXPECT_TRUE(any_differs)
                << "cache did not invalidate on rate change";
        });
}

}  // namespace
}  // namespace mango::operators
