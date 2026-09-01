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
// not as two independently rounded terms that can go tiny-negative. This
// must run on an ASYMMETRIC grid: the canonical two-term form (a_f -
// z as a single subtraction) is bit-identical to the old two-term
// accumulation on ANY grid, uniform or not, so what actually needs the
// asymmetric cells here is exercising a binding entry on unequal-length
// neighbor cells at the extreme-rho regime where tanh(rho) == 1 and
// a_f == z exactly — the assertion below is on the assembled entry's
// sign, not on distinguishing the two arithmetic forms.
TEST(SpatialOperatorFittedTest, AssembledBindingEntryNonNegativeAtExtremeRho) {
    with_operator(
        GridSpec<double>::sinh_spaced(-5.0, 5.0, 11, 3.0).value(),  // asymmetric cells
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
//
// The same-t re-assembly half below (a cache hit reproducing t=0 output
// bit-for-bit) is a real check, but the divergence half that FOLLOWS it
// ("assemble at a different t, expect *some* entry to differ from t=0")
// cannot actually fail on a stale cache: even if ensure_fitted_cache()
// wrongly reused the t=0 a_f values at t=1.0, the drift term z (and hence
// the assembled off-diagonals, via num_lower/num_upper) is recomputed
// inline from the freshly-sampled b every call -- a stale a_f still
// produces a numerically different row once b has changed. So "any_differs
// == true" would pass whether or not the cache actually invalidated.
//
// The discriminating check is therefore an EXACT cross-check against a
// second, independently-assembled operator that never went through a
// cache transition at all: a constant-rate PDE fixed at r=0.09 (same
// sigma, dividend, and grid), assembled fresh at t=0. If FittedCache's
// t=1.0 assembly used a stale (r=0.05) a_f, its bands would numerically
// differ from this reference's bands (both still finite, so "any_differs"
// above would not have caught it either); a correct cache produces
// bit-identical bands, because both paths sample the same (a=0.01,
// b(0.09), grid) and ensure_fitted_cache is otherwise deterministic.
TEST(SpatialOperatorFittedTest, FittedCacheInvalidatesOnRateChange) {
    auto rate_fn = [](double t) { return t < 0.5 ? 0.05 : 0.09; };
    std::vector<double> lower_t1, upper_t1, diag_t1;
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
            // off-diagonal must differ from the t=0 assembly. (Weak check
            // -- see the discriminating cross-check below.)
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

            lower_t1.assign(jac.lower().begin(), jac.lower().end());
            upper_t1.assign(jac.upper().begin(), jac.upper().end());
            diag_t1.assign(jac.diag().begin(), jac.diag().end());
        });

    // Discriminating cross-check: a fresh operator/workspace, constant
    // rate fixed at the SAME value (0.09) the callable-rate PDE steps to
    // at t=1.0, same sigma/dividend/grid. A stale a_f_cache in the first
    // operator would make its t=1.0 bands numerically diverge from this
    // one; a correctly-invalidated cache makes them EXACTLY equal, since
    // both assemblies sample the identical (a, b, grid) triple through
    // the same deterministic fitted_diffusion() computation.
    with_operator(
        GridSpec<double>::sinh_spaced(-2.0, 2.0, 21, 3.0).value(),
        BlackScholesPDE<double>(0.01, 0.09, 0.0),
        [&](auto& op, auto& workspace, auto& grid_view, auto&) {
            const size_t n = grid_view.size();
            auto jac = workspace.jacobian();
            op.assemble_jacobian(0.0, 1.0, jac);
            ASSERT_EQ(lower_t1.size(), n - 1);
            for (size_t i = 1; i < n - 1; ++i) {
                EXPECT_EQ(jac.lower()[i - 1], lower_t1[i - 1]) << "i=" << i;
                EXPECT_EQ(jac.upper()[i], upper_t1[i]) << "i=" << i;
                EXPECT_EQ(jac.diag()[i], diag_t1[i]) << "i=" << i;
            }
        });
}

// Regression (#472 gate-2 fix): the fitted-cache validity metadata used to
// live on SpatialOperator itself (cached_a_/cached_b_/cached_grid_), so
// when two SpatialOperator instances shared one PDEWorkspace, each kept an
// independent validity key while both wrote the SAME a_f_cache array. After
// op B overwrote the cache with its own coefficients, op A's next call
// would see its OWN unchanged (a, b, grid) sample still match its own
// local key and return early -- silently reading back op B's fitted
// values instead of its own. Co-locating the key with the data in the
// workspace buffer (PDEWorkspace::fitted_cache_meta()) fixes this: the key
// that says the cache is valid travels with the array it describes, so a
// mismatched writer always misses and recomputes, regardless of which
// operator makes the call.
TEST(SpatialOperatorFittedTest, TwoOperatorsSharingWorkspaceStayCorrect) {
    auto spec = GridSpec<double>::sinh_spaced(-2.0, 2.0, 21, 3.0).value();
    auto grid_buf = spec.generate();
    auto grid_view = grid_buf.view();
    auto spacing = std::make_shared<GridSpacing<double>>(grid_view);
    const size_t n = grid_view.size();

    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();

    // Two operators, same workspace, same grid (same spacing_ pointer),
    // different sampled coefficients (rate 0.05 vs 0.09).
    auto op_a = create_spatial_operator(
        BlackScholesPDE<double>(0.01, 0.05, 0.0), spacing, workspace);
    auto op_b = create_spatial_operator(
        BlackScholesPDE<double>(0.01, 0.09, 0.0), spacing, workspace);

    auto jac = workspace.jacobian();

    op_a.assemble_jacobian(0.0, 1.0, jac);
    std::vector<double> lower_a1(jac.lower().begin(), jac.lower().end());
    std::vector<double> diag_a1(jac.diag().begin(), jac.diag().end());
    std::vector<double> upper_a1(jac.upper().begin(), jac.upper().end());

    // op B writes the SAME shared a_f_cache with its own coefficients.
    op_b.assemble_jacobian(0.0, 1.0, jac);

    // op A re-assembles at the SAME (a, b) it started with. Its own local
    // validity key (pre-fix) would still match, so it must not silently
    // reuse op B's fitted values.
    op_a.assemble_jacobian(0.0, 1.0, jac);
    std::vector<double> lower_a2(jac.lower().begin(), jac.lower().end());
    std::vector<double> diag_a2(jac.diag().begin(), jac.diag().end());
    std::vector<double> upper_a2(jac.upper().begin(), jac.upper().end());

    for (size_t i = 0; i < lower_a1.size(); ++i) {
        EXPECT_EQ(lower_a2[i], lower_a1[i]) << "i=" << i;
    }
    for (size_t i = 0; i < diag_a1.size(); ++i) {
        EXPECT_EQ(diag_a2[i], diag_a1[i]) << "i=" << i;
    }
    for (size_t i = 0; i < upper_a1.size(); ++i) {
        EXPECT_EQ(upper_a2[i], upper_a1[i]) << "i=" << i;
    }

    // Cross-check against a THIRD operator, A's PDE, over its OWN fresh
    // workspace -- never went through a cache-sharing transition at all.
    std::vector<double> buffer_c(PDEWorkspace::required_size(n));
    auto workspace_c = PDEWorkspace::from_buffer(buffer_c, n).value();
    auto op_c = create_spatial_operator(
        BlackScholesPDE<double>(0.01, 0.05, 0.0), spacing, workspace_c);
    auto jac_c = workspace_c.jacobian();
    op_c.assemble_jacobian(0.0, 1.0, jac_c);

    for (size_t i = 0; i < lower_a1.size(); ++i) {
        EXPECT_EQ(jac_c.lower()[i], lower_a1[i]) << "i=" << i;
    }
    for (size_t i = 0; i < diag_a1.size(); ++i) {
        EXPECT_EQ(jac_c.diag()[i], diag_a1[i]) << "i=" << i;
    }
    for (size_t i = 0; i < upper_a1.size(); ++i) {
        EXPECT_EQ(jac_c.upper()[i], upper_a1[i]) << "i=" << i;
    }
}

// ===========================================================================
// Regression tests for #472 gate-2 pre-merge review: deep-ITM exercise lock
// criterion must use the RAW (unfitted) operator, not the Il'in-fitted one.
// ===========================================================================

// apply_unfitted() must combine the RAW diffusion coefficient a with the
// same stencil derivatives apply() uses -- Lu[i] = a*d2u[i] + b*du[i] -
// r*u[i], with NO Il'in fitting. Cross-checked against an independent
// CenteredDifference computation over the same spacing, not against
// apply()'s own internals.
TEST(SpatialOperatorFittedTest, ApplyUnfittedMatchesRawCombine) {
    with_operator(
        GridSpec<double>::sinh_spaced(-4.0, 0.0, 21, 3.0).value(),
        BlackScholesPDE<double>(0.20, 0.0, 0.0),  // sigma=20%, r=0, q=0
        [&](auto& op, auto&, auto& grid_view, auto& spacing) {
            const size_t n = grid_view.size();
            const auto& x = grid_view.span();
            std::vector<double> u(n);
            for (size_t i = 0; i < n; ++i) {
                u[i] = std::max(1.0 - std::exp(x[i]), 0.0);  // put payoff
            }

            std::vector<double> Lu(n, 0.0);
            op.apply_unfitted(0.0, u, Lu);

            // Independent reference: raw a/b/r combine over the SAME
            // stencil (CenteredDifference), computed directly here rather
            // than reusing apply_interior_impl.
            CenteredDifference<double> stencil(*spacing);
            std::vector<double> d2u(n, 0.0), du(n, 0.0);
            stencil.compute_second_derivative(u, d2u, 1, n - 1);
            stencil.compute_first_derivative(u, du, 1, n - 1);

            const double a = 0.02;           // sigma^2/2
            const double b = 0.0 - 0.0 - 0.02;  // r - q - sigma^2/2
            const double r = 0.0;
            for (size_t i = 1; i < n - 1; ++i) {
                const double expected = a * d2u[i] + b * du[i] - r * u[i];
                // Tolerance, not bit-exact equality: apply_unfitted()'s
                // internal combine loop and this test's independent one
                // are identical source but distinct call sites, and
                // MANGO_TARGET_CLONES-vectorized stencil kernels feeding
                // them can pick FMA contraction differently at the
                // scalar/vector boundary of each loop -- observed diffs
                // are ~1e-14 relative, far below anything a fitted-vs-raw
                // contamination bug (order-1 relative) would produce.
                EXPECT_NEAR(Lu[i], expected, std::abs(expected) * 1e-12 + 1e-18)
                    << "i=" << i;
            }
        });
}

// The Il'in-fitted operator's extra diffusion must never make L(psi) LARGER
// than the raw operator's at deep-ITM put nodes: psi'' = -e^x < 0 there, so
// the fitted a_f >= a bias is strictly downward (fitted <= raw). On an
// asymmetric (sinh) grid at least one deep node must show a STRICT
// inequality -- the bias is real, not just non-positive by construction.
TEST(SpatialOperatorFittedTest, FittedLPsiNeverAboveRawForDeepPut) {
    with_operator(
        GridSpec<double>::sinh_spaced(-4.0, 0.0, 21, 3.0).value(),
        BlackScholesPDE<double>(0.20, 0.0, 0.0),
        [&](auto& op, auto&, auto& grid_view, auto&) {
            const size_t n = grid_view.size();
            const auto& x = grid_view.span();
            std::vector<double> psi(n);
            for (size_t i = 0; i < n; ++i) {
                psi[i] = std::max(1.0 - std::exp(x[i]), 0.0);
            }
            const double psi_max = *std::max_element(psi.begin(), psi.end());

            std::vector<double> fitted(n, 0.0), raw(n, 0.0);
            op.apply(0.0, psi, fitted);
            op.apply_unfitted(0.0, psi, raw);

            bool any_strict = false;
            for (size_t i = 1; i < n - 1; ++i) {
                if (psi[i] > 0.95 * psi_max) {
                    EXPECT_LE(fitted[i], raw[i] + 1e-15 * std::abs(raw[i]))
                        << "i=" << i << " fitted=" << fitted[i]
                        << " raw=" << raw[i];
                    if (fitted[i] < raw[i]) any_strict = true;
                }
            }
            EXPECT_TRUE(any_strict)
                << "expected at least one deep node with a strict fitted < "
                   "raw bias on this asymmetric grid";
        });
}

}  // namespace
}  // namespace mango::operators
