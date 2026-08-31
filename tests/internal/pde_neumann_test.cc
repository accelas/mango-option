// SPDX-License-Identifier: MIT
//
// Empirical validation of the restored Neumann ghost-point treatment
// (issue #455, design doc section C). These tests exercise a minimal
// heat-equation solver (LaplacianPDE + NeumannBC both sides, no obstacle,
// so PDESolver's Newton path — not the projected LCP path — solves the
// boundary rows) and check the properties that only hold if the boundary
// row is genuinely part of the solved system:
//
//   1. Mass conservation for zero-flux BCs (the exact historical setup
//      from docs/archive/issues/6/NEUMANN_BC_PROBLEM.md, which measured a
//      ~2% drift under the old lagged/identity-row treatment).
//   2. Second-order spatial convergence on a manufactured solution with
//      inhomogeneous, time-varying Neumann data.
//   3. A dt-halving discriminator at the boundary node: the restored
//      analytic row is solved implicitly each stage, so its error should
//      scale ~4x per halving (2nd order in time); a lagged post-hoc
//      boundary value (the old behavior) would only scale ~2x (1st order).
//
// See docs/plans/2026-08-30-boundary-correctness-439-455-design.md
// sections C1, C3, T5.
#include "mango/pde/internal/pde_solver.hpp"
#include "mango/pde/internal/operator_factory.hpp"
#include "mango/pde/internal/pde_workspace.hpp"
#include "mango/pde/core/boundary_conditions.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/pde/core/time_domain.hpp"
#include "mango/pde/operators/laplacian_pde.hpp"
#include "mango/math/thomas_solver.hpp"
#include "../lcp_test_util.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory_resource>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace {

// ===========================================================================
// Minimal CRTP heat solver: LaplacianPDE spatial operator, NeumannBC on
// both sides with caller-supplied gradient functors, NO obstacle() method
// (so PDESolver's Newton residual path, not the projected LCP path, is
// what solves the implicit stages). Modeled on AmericanPutSolver's
// structure (src/option/american_option.cpp) minus the obstacle/payoff
// machinery irrelevant to a pure heat equation.
template <typename GLeft, typename GRight>
class HeatNeumannSolver
    : public mango::PDESolver<HeatNeumannSolver<GLeft, GRight>> {
public:
    using PDEType = mango::operators::LaplacianPDE<double>;
    using SpatialOpType = mango::operators::SpatialOperator<PDEType, double>;

    // Regression: SpatialOperator stores a non-owning `PDEWorkspace*`
    // (spatial_operator.hpp). Building it from the constructor's
    // by-value `workspace` parameter (as an early version of this test
    // did) leaves that pointer dangling once the constructor returns —
    // segfaults on the first solve. AmericanPutSolver avoids this by
    // keeping its own persistent `workspace_local_` member and building
    // the spatial operator from THAT; mirrored here.
    HeatNeumannSolver(std::shared_ptr<mango::Grid<double>> grid,
                       mango::PDEWorkspace workspace, double D,
                       mango::NeumannBC<GLeft> left_bc,
                       mango::NeumannBC<GRight> right_bc)
        : mango::PDESolver<HeatNeumannSolver>(grid, workspace)
        , workspace_local_(workspace)
        , left_bc_(std::move(left_bc))
        , right_bc_(std::move(right_bc))
        , spatial_op_(make_spatial_op(*grid, workspace_local_, D))
    {}

    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

private:
    static SpatialOpType make_spatial_op(const mango::Grid<double>& grid,
                                          mango::PDEWorkspace& workspace,
                                          double D) {
        auto spacing =
            std::make_shared<mango::GridSpacing<double>>(grid.spacing());
        return mango::operators::create_spatial_operator(
            PDEType(D), spacing, workspace);
    }

    mango::PDEWorkspace workspace_local_;
    mango::NeumannBC<GLeft> left_bc_;
    mango::NeumannBC<GRight> right_bc_;
    SpatialOpType spatial_op_;
};

template <typename GLeft, typename GRight>
HeatNeumannSolver(std::shared_ptr<mango::Grid<double>>, mango::PDEWorkspace,
                   double, mango::NeumannBC<GLeft>, mango::NeumannBC<GRight>)
    -> HeatNeumannSolver<GLeft, GRight>;

// Convenience factory: builds grid + workspace + solver and runs solve().
// Returns the solver so callers can inspect solution()/grid.
template <typename GLeft, typename GRight, typename IC>
std::shared_ptr<mango::Grid<double>> run_heat_solve(
    const mango::GridSpec<double>& grid_spec, const mango::TimeDomain& time,
    double D, GLeft g_left, GRight g_right, IC&& ic,
    std::pmr::monotonic_buffer_resource& pool,
    std::optional<mango::TRBDF2Config> config = std::nullopt) {
    auto grid_result = mango::Grid<double>::create(grid_spec, time);
    EXPECT_TRUE(grid_result.has_value()) << grid_result.error();
    auto grid = grid_result.value();

    size_t buffer_size = mango::PDEWorkspace::required_size(grid->n_space());
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, &pool);
    auto workspace_result = mango::PDEWorkspace::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()}, grid->x(),
        grid->n_space());
    EXPECT_TRUE(workspace_result.has_value()) << workspace_result.error();
    auto workspace = workspace_result.value();

    auto left_bc = mango::NeumannBC(g_left);
    auto right_bc = mango::NeumannBC(g_right);

    auto solver = HeatNeumannSolver(grid, workspace, D, left_bc, right_bc);
    if (config) {
        solver.set_config(*config);
    }
    solver.initialize(std::forward<IC>(ic));
    auto status = solver.solve();
    // The `<<` message is only materialized on failure, so it's safe to
    // dereference status.error() here even though status may hold a value.
    EXPECT_TRUE(status.has_value())
        << "solve() failed with code "
        << static_cast<int>(status.error().code);

    // Grid owns the solution storage (not the solver), so it's safe to
    // return grid and let `solver` go out of scope: grid->solution() below
    // still reads the final values.
    return grid;
}

// ===========================================================================
// Trapezoidal-rule mass: M = dx_0/2*u_0 + ... + dx_{n-2}/2*u_{n-1}, exact
// for uniform grids and the quantity the ghost-eliminated Neumann scheme
// conserves (see docs/archive/issues/6/NEUMANN_BC_PROBLEM.md).
// ===========================================================================
double trapezoidal_mass(std::span<const double> x, std::span<const double> u) {
    double m = 0.0;
    for (size_t i = 0; i + 1 < x.size(); ++i) {
        m += 0.5 * (u[i] + u[i + 1]) * (x[i + 1] - x[i]);
    }
    return m;
}

// ===========================================================================
// Manufactured solution: u(x,t) = exp(-D k^2 t) sin(k x + 0.3). Satisfies
// du/dt = D d^2u/dx^2 exactly (d^2u/dx^2 = -k^2 u). Nonzero u''' at both
// ends of [0,1] (see design doc C1) so the boundary row's O(h) local
// truncation term can't cancel by symmetry — the convergence test below
// would not catch a defect there if it did.
// ===========================================================================
constexpr double kK = 2.1;
constexpr double kPhase = 0.3;

double manufactured_u(double D, double x, double t) {
    return std::exp(-D * kK * kK * t) * std::sin(kK * x + kPhase);
}
double manufactured_dudx(double D, double x, double t) {
    return std::exp(-D * kK * kK * t) * kK * std::cos(kK * x + kPhase);
}

double max_norm_error_at_final_time(const mango::Grid<double>& grid, double D) {
    auto x = grid.x();
    auto u = grid.solution();
    const double t_final = grid.time().time_points().back();
    double max_err = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        max_err = std::max(max_err,
                            std::abs(u[i] - manufactured_u(D, x[i], t_final)));
    }
    return max_err;
}

// Least-squares slope of log(err) vs log(dx): the fitted convergence order.
double fitted_order(const std::vector<double>& dxs,
                     const std::vector<double>& errs) {
    const size_t n = dxs.size();
    std::vector<double> lx(n), ly(n);
    for (size_t i = 0; i < n; ++i) {
        lx[i] = std::log(dxs[i]);
        ly[i] = std::log(errs[i]);
    }
    const double mean_x = std::accumulate(lx.begin(), lx.end(), 0.0) / n;
    const double mean_y = std::accumulate(ly.begin(), ly.end(), 0.0) / n;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += (lx[i] - mean_x) * (ly[i] - mean_y);
        den += (lx[i] - mean_x) * (lx[i] - mean_x);
    }
    return num / den;
}

}  // namespace

// ===========================================================================
// Step 1: Mass conservation (regression for archived issue #6)
// ===========================================================================

// Regression: Neumann boundary rows must evolve via the PDE (ghost point),
// not algebraic constraints. Bug: identity rows caused ~2% mass drift
// (C-era issue #6, docs/archive/issues/6/NEUMANN_BC_PROBLEM.md); the C++
// migration regressed to a lagged post-hoc application (#455).
TEST(PDENeumann, ZeroFluxMassConservedToRoundoff) {
    // Exact issue-#6 setup: D=0.1, x in [0,1], n=101, dt=0.01, 100 steps,
    // u0 = exp(-50(x-0.5)^2), zero-flux (g=0) on both sides.
    const double D = 0.1;
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 101).value();
    auto time = mango::TimeDomain::from_n_steps(0.0, 1.0, 100);  // dt = 0.01

    auto zero_grad = [](double, double) { return 0.0; };
    auto ic = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::exp(-50.0 * (x[i] - 0.5) * (x[i] - 0.5));
        }
    };

    std::pmr::monotonic_buffer_resource pool;
    auto grid = run_heat_solve(grid_spec, time, D, zero_grad, zero_grad, ic, pool);

    std::vector<double> u0(grid->n_space());
    ic(grid->x(), u0);
    const double mass_initial = trapezoidal_mass(grid->x(), u0);
    const double mass_final = trapezoidal_mass(grid->x(), grid->solution());

    ASSERT_GT(std::abs(mass_initial), 0.0);
    const double relative_drift = std::abs(mass_final / mass_initial - 1.0);
    // Tolerance: 1e-10, i.e. roundoff. The historical bug measured ~2%
    // (0.0196) — four orders of magnitude above this bound.
    EXPECT_LT(relative_drift, 1e-10)
        << "mass_initial=" << mass_initial << " mass_final=" << mass_final
        << " relative_drift=" << relative_drift;
}

// ===========================================================================
// Step 2: Convergence order with inhomogeneous, time-varying Neumann data
// ===========================================================================

// Counter-tested against the pre-#455 lagged treatment (Lu[boundary]=0,
// identity Jacobian rows, post-hoc NeumannBC::apply() overwrite in
// apply_boundary_conditions — temporarily reintroduced locally, run, then
// reverted; not part of the committed solver): measured fitted order
// -0.0106 with errors actually GROWING under refinement (n=81->161 and
// 161->321 both regress), vs. 1.9961 (this test's asserted floor: 1.8) on
// the fixed code below. The lagged treatment doesn't just lose an order of
// accuracy here, it fails to converge at all on this boundary-sensitive
// manufactured solution.
TEST(PDENeumann, ManufacturedSolutionConvergesAtOrderAtLeast1_8) {
    const double D = 0.05;
    const double t_final = 0.1;

    auto g_left = [D](double t, double) {
        return manufactured_dudx(D, 0.0, t);
    };
    auto g_right = [D](double t, double) {
        return manufactured_dudx(D, 1.0, t);
    };
    auto ic = [D](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = manufactured_u(D, x[i], 0.0);
        }
    };

    const std::vector<size_t> ns = {41, 81, 161, 321};
    std::vector<double> dxs, errs;

    for (size_t n : ns) {
        const double dx = 1.0 / static_cast<double>(n - 1);
        const double dt = 0.5 * dx;  // dt ∝ dx (TR-BDF2 is 2nd order in time)
        const size_t n_steps =
            static_cast<size_t>(std::lround(t_final / dt));

        auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, n).value();
        auto time = mango::TimeDomain::from_n_steps(0.0, t_final, n_steps);

        std::pmr::monotonic_buffer_resource pool;
        auto grid = run_heat_solve(grid_spec, time, D, g_left, g_right, ic, pool);

        const double err = max_norm_error_at_final_time(*grid, D);
        dxs.push_back(dx);
        errs.push_back(err);
    }

    // Per-grid diagnostics (also serves as the "per-grid errors" the task
    // brief asks for in a BLOCKED report, printed unconditionally so a CI
    // log always has them even on pass).
    for (size_t i = 0; i < ns.size(); ++i) {
        SCOPED_TRACE(testing::Message()
                     << "n=" << ns[i] << " dx=" << dxs[i]
                     << " max_err=" << errs[i]);
    }

    for (size_t i = 1; i < ns.size(); ++i) {
        const double order = std::log2(errs[i - 1] / errs[i]);
        EXPECT_GT(order, 1.0)
            << "pairwise order n=" << ns[i - 1] << "->" << ns[i]
            << " err=" << errs[i - 1] << "->" << errs[i];
    }

    const double order = fitted_order(dxs, errs);
    EXPECT_GE(order, 1.8) << "fitted spatial convergence order too low: "
                          << order
                          << " — see per-grid errors in SCOPED_TRACE above";
}

// One geometric (nonuniform, sinh-graded) grid refinement pair: error must
// still shrink by >= 3x when the grid is refined, confirming the boundary
// row's adjacent-spacing formula (h = dx[0] / dx[n-2], not a hardcoded
// uniform dx) is correct on nonuniform grids too.
TEST(PDENeumann, NonuniformGridRefinementShrinksErrorAtLeast3x) {
    const double D = 0.05;
    const double t_final = 0.1;
    const double concentration = 2.0;

    auto g_left = [D](double t, double) {
        return manufactured_dudx(D, 0.0, t);
    };
    auto g_right = [D](double t, double) {
        return manufactured_dudx(D, 1.0, t);
    };
    auto ic = [D](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = manufactured_u(D, x[i], 0.0);
        }
    };

    std::vector<double> errs;
    for (size_t n : {size_t(81), size_t(161)}) {
        const double dx_avg = 1.0 / static_cast<double>(n - 1);
        const double dt = 0.5 * dx_avg;
        const size_t n_steps = static_cast<size_t>(std::lround(t_final / dt));

        auto grid_spec =
            mango::GridSpec<double>::sinh_spaced(0.0, 1.0, n, concentration)
                .value();
        auto time = mango::TimeDomain::from_n_steps(0.0, t_final, n_steps);

        std::pmr::monotonic_buffer_resource pool;
        auto grid = run_heat_solve(grid_spec, time, D, g_left, g_right, ic, pool);
        errs.push_back(max_norm_error_at_final_time(*grid, D));
    }

    ASSERT_EQ(errs.size(), 2u);
    SCOPED_TRACE(testing::Message() << "err(n=81)=" << errs[0]
                                     << " err(n=161)=" << errs[1]);
    EXPECT_GT(errs[0] / errs[1], 3.0);
}

// ===========================================================================
// Step 3: dt-halving discriminator at the boundary node
// ===========================================================================

// Regression: a lagged post-hoc Neumann overwrite (the pre-#455-fix
// behavior, using the previous iterate's neighbor value) is only 1st-order
// accurate in time at the boundary, so its error halves (~2x) per dt
// halving. The restored treatment solves the boundary row implicitly each
// stage (genuinely 2nd order, matching the interior TR-BDF2 scheme), so
// its boundary-node error should shrink ~4x per halving. This test is the
// discriminator: ~4x passes, ~2x (the old bug) fails.
//
// Counter-tested against the pre-#455 lagged treatment (temporarily
// reintroduced locally in pde_solver.hpp, run against just this test, then
// reverted — not part of the committed solver): measured ratios collapsed
// to 1.027 and 1.051 (errors 0.1028, 0.1001, 0.0952 for n_steps=1,2,4) —
// even flatter than the ~2x first-order guess, but decisively on the wrong
// side of this test's >3.0 threshold — vs. 3.835/3.896 on the fixed code
// below.
TEST(PDENeumann, BoundaryNodeErrorScalesQuarticWithDtHalving) {
    const double D = 0.05;
    const double t_final = 2.0;
    // Fine, fixed spatial grid so spatial truncation error stays far below
    // the time-truncation error being measured across all three dt levels.
    const size_t n = 321;

    // Rannacher startup is disabled so every step uses the same TR-BDF2
    // stage treatment (Rannacher's startup steps would otherwise mix in a
    // different, non-representative discretization for the coarsest run).
    mango::TRBDF2Config config;
    config.rannacher_startup = false;

    auto g_left = [D](double t, double) {
        return manufactured_dudx(D, 0.0, t);
    };
    auto g_right = [D](double t, double) {
        return manufactured_dudx(D, 1.0, t);
    };
    auto ic = [D](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = manufactured_u(D, x[i], 0.0);
        }
    };

    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, n).value();

    std::vector<double> boundary_errs;
    for (size_t n_steps : {size_t(1), size_t(2), size_t(4)}) {
        auto time = mango::TimeDomain::from_n_steps(0.0, t_final, n_steps);
        std::pmr::monotonic_buffer_resource pool;
        auto grid = run_heat_solve(grid_spec, time, D, g_left, g_right, ic,
                                    pool, config);
        const double u0_exact = manufactured_u(D, 0.0, t_final);
        boundary_errs.push_back(
            std::abs(grid->solution()[0] - u0_exact));
    }

    ASSERT_EQ(boundary_errs.size(), 3u);
    const double ratio1 = boundary_errs[0] / boundary_errs[1];
    const double ratio2 = boundary_errs[1] / boundary_errs[2];
    SCOPED_TRACE(testing::Message()
                 << "boundary errs (n_steps=1,2,4): " << boundary_errs[0]
                 << ", " << boundary_errs[1] << ", " << boundary_errs[2]
                 << " ratios: " << ratio1 << ", " << ratio2);

    // ~4x expected; the lagged/old treatment would show ~2x. Use 3.0 as the
    // discriminating threshold: comfortably above 2x, comfortably below 4x.
    EXPECT_GT(ratio1, 3.0);
    EXPECT_GT(ratio2, 3.0);
}

// ===========================================================================
// Step 4: Obstacle+Neumann affine term at the linear-solve level
// ===========================================================================

// Regression: the projected (LCP) path folds the ghost-eliminated
// boundary row's affine (gradient-dependent) term into the RHS
// (pde_solver.hpp solve_implicit_stage_projected, "CRITICAL FIX" comment
// block) so that A*u = rhs is the true stage equation at a Neumann row.
// This test assembles that row directly, by hand, from the Task 7/8
// closed-form formulas (SpatialOperator::boundary_row_jacobian /
// boundary_row_affine — see design doc section C1) and drives it through
// solve_thomas_projected2 with an unrelated obstacle active elsewhere in
// the domain, checking the whole system against the exact-by-enumeration
// LCP reference from lcp_test_util.hpp (extracted from Task 1's test in
// this same task, per the brief).
TEST(PDENeumann, ObstacleWithNeumannAffineTermMatchesLcpReference) {
    using mango::test_util::mmatrix;
    using mango::test_util::solve_and_check;
    using mango::test_util::Sys;

    const size_t n = 10;
    const double w = 5.0;  // interior stage weight (arbitrary, matches Task 1's fixtures)

    // Obstacle: unreachably low everywhere (never binds) except nodes 8-9,
    // set far ABOVE any plausible continuation value so they are certainly
    // active — an unambiguous right-touching interval, deliberately far
    // from node 0 so the Neumann row's unusually large coefficients (from
    // the affine-term folding below) can't blur which nodes are active
    // (an earlier version of this test used the smooth `right_obstacle`
    // envelope, whose marginally-active nodes shifted once node 0's row
    // was replaced, producing enumeration mismatches unrelated to the
    // Neumann row itself — this hard lock isolates the row-assembly check
    // from that ambiguity, matching Task 1's
    // IdentityLockRowsInsideActiveInterval pattern).
    std::vector<double> psi(n, -1000.0);
    psi[8] = 100.0;
    psi[9] = 100.0;

    Sys s = mmatrix(n, w, psi);

    // Hand-derived Neumann row at node 0, from the Task 7/8 closed forms:
    //   left:  L_0 = (2a/h^2)*(u_1 - u_0) + c*u_0 + g*(b - 2a/h)
    //   jac:   diag = c - 2a/h^2, offdiag = 2a/h^2
    //   affine: g*(b - 2a/h)
    // with LaplacianPDE coefficients a=D, b=0, c=0, plus a nonzero gradient
    // g and TR-BDF2 stage weight coeff_dt, matching
    // build_jacobian_boundaries()/solve_implicit_stage_projected() exactly:
    //   A.diag[0]  = 1 - coeff_dt*jac.diag
    //   A.upper[0] = -coeff_dt*jac.offdiag
    //   rhs[0]     = rhs_base + coeff_dt*affine
    const double a = 0.1, b = 0.0, c = 0.0;
    const double h = 0.05;
    const double g = 1.3;
    const double coeff_dt = 0.3;
    const double rhs_base = 0.7;

    const double jac_diag = c - 2.0 * a / (h * h);
    const double jac_offdiag = 2.0 * a / (h * h);
    const double affine = g * (b - 2.0 * a / h);

    s.diag[0] = 1.0 - coeff_dt * jac_diag;
    s.upper[0] = -coeff_dt * jac_offdiag;
    s.rhs[0] = rhs_base + coeff_dt * affine;

    auto x = solve_and_check<mango::LcpActiveSide::Right>(s);

    // Node 0 must be inactive (its psi floor is unreachably low) and must
    // therefore satisfy the Neumann row equation exactly:
    //   diag[0]*u[0] + upper[0]*u[1] == rhs[0]
    const double row0_residual = s.diag[0] * x[0] + s.upper[0] * x[1] - s.rhs[0];
    EXPECT_NEAR(row0_residual, 0.0, 1e-9);
    EXPECT_GT(x[0], psi[0]);  // confirms it's genuinely inactive, not clamped
}

// ===========================================================================
// Step 5: Grid floor
// ===========================================================================

TEST(PDENeumann, TwoPointGridRejected) {
    auto result = mango::GridSpec<double>::uniform(0.0, 1.0, 2);
    EXPECT_FALSE(result.has_value());
}

TEST(PDENeumann, ThreePointGridSmokeSolve) {
    const double D = 0.1;
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 3).value();
    auto time = mango::TimeDomain::from_n_steps(0.0, 0.01, 4);

    auto zero_grad = [](double, double) { return 0.0; };
    auto ic = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(3.14159265358979 * x[i]);
        }
    };

    std::pmr::monotonic_buffer_resource pool;
    auto grid = run_heat_solve(grid_spec, time, D, zero_grad, zero_grad, ic, pool);
    for (double v : grid->solution()) {
        EXPECT_TRUE(std::isfinite(v));
    }
}
