// SPDX-License-Identifier: MIT
// Regression: Brennan-Schwartz sweep orientation (issue #439, corrected).
// Bug: projection during right-to-left substitution is exact only for
// RIGHT-interval active sets; puts (left interval) got an inexact solve.
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "mango/math/thomas_solver.hpp"
#include "lcp_test_util.hpp"

// Enumeration/dense-solve reference helpers (dense_solve, lcp_reference,
// Sys, mmatrix, {left,right,sawtooth}_obstacle, solve_and_check) now live in
// lcp_test_util.hpp (extracted in Task 9 so pde_neumann_test.cc can reuse
// them); pulled in unqualified here to keep every test body below unchanged.
using namespace mango::test_util;

TEST(LcpSweep, LeftActiveExactWithLeftSweep) {
    Sys s = mmatrix(12, 5.0, left_obstacle(12));
    solve_and_check<mango::LcpActiveSide::Left>(s);
}

TEST(LcpSweep, RightActiveExactWithRightSweep) {
    Sys s = mmatrix(12, 5.0, right_obstacle(12));
    solve_and_check<mango::LcpActiveSide::Right>(s);
}

TEST(LcpSweep, EmptyActiveSetBothSweepsMatchThomas) {
    const size_t n = 12;
    Sys s = mmatrix(n, 5.0, std::vector<double>(n, -1.0));

    std::vector<double> x_plain(n), ws_plain(2 * n);
    ASSERT_TRUE(mango::solve_thomas<double>(
                    std::span<const double>(s.lower), std::span<const double>(s.diag),
                    std::span<const double>(s.upper), std::span<const double>(s.rhs),
                    std::span<double>(x_plain), std::span<double>(ws_plain))
                    .ok());

    auto x_left = solve_and_check<mango::LcpActiveSide::Left>(s);
    auto x_right = solve_and_check<mango::LcpActiveSide::Right>(s);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_left[i], x_plain[i], 1e-9);
        EXPECT_NEAR(x_right[i], x_plain[i], 1e-9);
    }
}

TEST(LcpSweep, FullActiveSetBothSweeps) {
    const size_t n = 12;
    Sys s = mmatrix(n, 5.0, std::vector<double>(n, 10.0));

    auto x_left = solve_and_check<mango::LcpActiveSide::Left>(s);
    auto x_right = solve_and_check<mango::LcpActiveSide::Right>(s);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(x_left[i], s.psi[i]);
        EXPECT_DOUBLE_EQ(x_right[i], s.psi[i]);
    }
}

TEST(LcpSweep, NonconstantObstacle) {
    Sys s = mmatrix(12, 5.0, sawtooth_obstacle(12));
    solve_and_check<mango::LcpActiveSide::Left>(s);
}

TEST(LcpSweep, IdentityLockRowsInsideActiveInterval) {
    // Convert two interior rows inside the active interval to identity
    // (lower/upper zeroed, diag=1, rhs=psi) as the deep-ITM lock does;
    // both sweeps must still be exact on their own side.
    Sys s_left = mmatrix(12, 5.0, left_obstacle(12));
    for (size_t i : {size_t(2), size_t(3)}) {
        s_left.diag[i] = 1.0;
        s_left.rhs[i] = s_left.psi[i];
        if (i > 0) s_left.lower[i - 1] = 0.0;
        if (i + 1 < s_left.diag.size()) s_left.upper[i] = 0.0;
    }
    solve_and_check<mango::LcpActiveSide::Left>(s_left);

    // Mirror on the right-active side: lock rows well inside the interval
    // touching the right boundary (n=12, right_obstacle active near i>=6).
    Sys s_right = mmatrix(12, 5.0, right_obstacle(12));
    for (size_t i : {size_t(8), size_t(9)}) {
        s_right.diag[i] = 1.0;
        s_right.rhs[i] = s_right.psi[i];
        if (i > 0) s_right.lower[i - 1] = 0.0;
        if (i + 1 < s_right.diag.size()) s_right.upper[i] = 0.0;
    }
    solve_and_check<mango::LcpActiveSide::Right>(s_right);
}

TEST(LcpSweep, DirichletIdentityBoundaryRows) {
    const size_t n = 12;
    Sys s = mmatrix(n, 5.0, std::vector<double>(n, -100.0));  // never binds
    s.diag[0] = 1.0;
    s.upper[0] = 0.0;
    s.rhs[0] = 2.0;
    s.diag[n - 1] = 1.0;
    s.lower[n - 2] = 0.0;
    s.rhs[n - 1] = 1.0;

    auto x_left = solve_and_check<mango::LcpActiveSide::Left>(s);
    auto x_right = solve_and_check<mango::LcpActiveSide::Right>(s);

    EXPECT_NEAR(x_left[0], 2.0, 1e-9);
    EXPECT_NEAR(x_left[n - 1], 1.0, 1e-9);
    EXPECT_NEAR(x_right[0], 2.0, 1e-9);
    EXPECT_NEAR(x_right[n - 1], 1.0, 1e-9);
}

TEST(LcpKkt, ValidatorFlagsWrongSweepOnLeftActive) {
    // Solve left-active with Side::Right (wrong side); validate_lcp_kkt must
    // return violation_count > 0 with worst_kind == 2 (continuation residual)
    // or 1 (dual) — and the enumerated-reference solution must validate clean.
    Sys s = mmatrix(12, 5.0, left_obstacle(12));
    const size_t n = s.diag.size();

    std::vector<double> ref;
    ASSERT_TRUE(lcp_reference(s.lower, s.diag, s.upper, s.rhs, s.psi, ref));

    std::vector<uint8_t> mask_ref(n);
    for (size_t i = 0; i < n; ++i)
        mask_ref[i] = (ref[i] <= s.psi[i] + 1e-9) ? uint8_t{1} : uint8_t{0};
    auto rep_ref = mango::validate_lcp_kkt<double>(
        std::span<const double>(s.lower), std::span<const double>(s.diag),
        std::span<const double>(s.upper), std::span<const double>(s.rhs),
        std::span<const double>(s.psi), std::span<const double>(ref),
        std::span<const uint8_t>(mask_ref));
    EXPECT_EQ(rep_ref.violation_count, 0u);

    std::vector<double> x_wrong(n), ws(2 * n);
    std::vector<uint8_t> mask_wrong(n);
    auto r_wrong = mango::solve_thomas_projected2<double, mango::LcpActiveSide::Right>(
        std::span<const double>(s.lower), std::span<const double>(s.diag),
        std::span<const double>(s.upper), std::span<const double>(s.rhs),
        std::span<const double>(s.psi), std::span<double>(x_wrong),
        std::span<double>(ws), std::span<uint8_t>(mask_wrong));
    ASSERT_TRUE(r_wrong.ok());

    auto rep_wrong = mango::validate_lcp_kkt<double>(
        std::span<const double>(s.lower), std::span<const double>(s.diag),
        std::span<const double>(s.upper), std::span<const double>(s.rhs),
        std::span<const double>(s.psi), std::span<const double>(x_wrong),
        std::span<const uint8_t>(mask_wrong));
    EXPECT_GT(rep_wrong.violation_count, 0u);
    EXPECT_TRUE(rep_wrong.worst_kind == 1 || rep_wrong.worst_kind == 2)
        << "worst_kind=" << rep_wrong.worst_kind;
}

TEST(LcpKkt, NonFiniteInputCountsAsViolation) {
    std::vector<double> lower{-1.0}, diag{2.0, 2.0}, upper{-1.0}, rhs{0.0, 0.0},
        psi{0.0, 0.0};
    std::vector<double> u{1.0, std::numeric_limits<double>::quiet_NaN()};
    std::vector<uint8_t> mask{0, 0};
    auto rep = mango::validate_lcp_kkt<double>(
        std::span<const double>(lower), std::span<const double>(diag),
        std::span<const double>(upper), std::span<const double>(rhs),
        std::span<const double>(psi), std::span<const double>(u),
        std::span<const uint8_t>(mask));
    EXPECT_GT(rep.violation_count, 0u);
    EXPECT_EQ(rep.worst_kind, 2);
}

// Regression: a NaN comparison is always false, so `u[i] < psi[i] - tol`
// silently passes when psi[i] itself is NaN. Node 1 here is otherwise a
// textbook-clean active node (Au == rhs exactly, dual condition holds at
// equality) — without a psi finiteness check, this candidate validates
// clean despite carrying a non-finite obstacle value.
TEST(LcpKkt, NonFinitePsiCountsAsViolation) {
    std::vector<double> lower{-1.0}, diag{2.0, 2.0}, upper{-1.0}, rhs{0.0, 0.0};
    std::vector<double> psi{-100.0, std::numeric_limits<double>::quiet_NaN()};
    std::vector<double> u{0.0, 0.0};
    std::vector<uint8_t> mask{0, 1};  // node 1 marked active
    auto rep = mango::validate_lcp_kkt<double>(
        std::span<const double>(lower), std::span<const double>(diag),
        std::span<const double>(upper), std::span<const double>(rhs),
        std::span<const double>(psi), std::span<const double>(u),
        std::span<const uint8_t>(mask));
    EXPECT_GT(rep.violation_count, 0u);
    EXPECT_EQ(rep.worst_kind, 2);
}

// Regression: the Left branch's starting division c_prime[n-1] =
// lower[n-2]/diag[n-1] had no singularity guard, unlike every other
// division in both branches. A degenerate diag[n-1] must be rejected with
// an error result, not silently produce NaN through ok_result().
TEST(LcpSweep, LeftBranchRejectsSingularLastDiagonal) {
    const size_t n = 6;
    Sys s = mmatrix(n, 5.0, left_obstacle(n));
    s.diag[n - 1] = 1e-20;  // degenerate: below default singularity_tol

    std::vector<double> x(n), ws(2 * n);
    std::vector<uint8_t> mask(n);
    auto r = mango::solve_thomas_projected2<double, mango::LcpActiveSide::Left>(
        std::span<const double>(s.lower), std::span<const double>(s.diag),
        std::span<const double>(s.upper), std::span<const double>(s.rhs),
        std::span<const double>(s.psi), std::span<double>(x),
        std::span<double>(ws), std::span<uint8_t>(mask));
    EXPECT_FALSE(r.ok());
}

// Regression: pre-merge review of #439/#455 found that validate_lcp_kkt
// checked active nodes only for dual feasibility ((Au)_i >= rhs_i), never
// for obstacle equality (u[i] == psi[i]). A lying/buggy active_mask that
// claims a node is active while u sits strictly above psi therefore passed
// silently: identity system (rhs=psi=0), u=1, mask=1 validated clean before
// this fix, even though complementarity requires u[i] == psi[i] on active
// nodes.
TEST(LcpKkt, ActiveNodeAbovePsiCountsAsViolation) {
    std::vector<double> lower{}, diag{1.0}, upper{}, rhs{0.0}, psi{0.0};
    std::vector<double> u{1.0};
    std::vector<uint8_t> mask{1};  // claims active, but u != psi
    auto rep = mango::validate_lcp_kkt<double>(
        std::span<const double>(lower), std::span<const double>(diag),
        std::span<const double>(upper), std::span<const double>(rhs),
        std::span<const double>(psi), std::span<const double>(u),
        std::span<const uint8_t>(mask));
    EXPECT_GT(rep.violation_count, 0u);
    EXPECT_EQ(rep.worst_kind, 0);
}

// Hand-verified 2-node cases from the issue #439 comment:
// A = [[2,-1],[-1,2]], rhs = 0.
TEST(LcpSweep, TwoNodeLeftActivePsiOneZero) {
    Sys s{std::vector<double>{-1.0}, std::vector<double>{2.0, 2.0},
          std::vector<double>{-1.0}, std::vector<double>{0.0, 0.0},
          std::vector<double>{1.0, 0.0}};
    auto x = solve_and_check<mango::LcpActiveSide::Left>(s);
    EXPECT_NEAR(x[0], 1.0, 1e-9);
    EXPECT_NEAR(x[1], 0.5, 1e-9);
}

TEST(LcpSweep, TwoNodeRightActivePsiZeroOne) {
    Sys s{std::vector<double>{-1.0}, std::vector<double>{2.0, 2.0},
          std::vector<double>{-1.0}, std::vector<double>{0.0, 0.0},
          std::vector<double>{0.0, 1.0}};
    auto x = solve_and_check<mango::LcpActiveSide::Right>(s);
    EXPECT_NEAR(x[0], 0.5, 1e-9);
    EXPECT_NEAR(x[1], 1.0, 1e-9);
}

// Regression: validate_lcp_kkt indexed lower/upper/rhs/psi/u/active_mask by
// diag.size() without validating span sizes, risking out-of-bounds reads on
// malformed input.
// Bug: Unlike adjacent Thomas solvers which validate dimensions first, this
// function had no dimension checks, allowing it to dereference mismatched
// spans.
TEST(LcpKkt, MismatchedSpansYieldSentinelViolation) {
    std::vector<double> lower{-1.0}, diag{2.0, 2.0}, upper{-1.0}, rhs{0.0, 0.0},
        psi{0.0, 0.0};
    std::vector<double> u{1.0};  // Too short: should have 2 elements
    std::vector<uint8_t> mask{0, 0};
    auto rep = mango::validate_lcp_kkt<double>(
        std::span<const double>(lower), std::span<const double>(diag),
        std::span<const double>(upper), std::span<const double>(rhs),
        std::span<const double>(psi), std::span<const double>(u),
        std::span<const uint8_t>(mask));
    EXPECT_GE(rep.violation_count, 1u);
    EXPECT_EQ(rep.max_violation, std::numeric_limits<double>::infinity());
}
