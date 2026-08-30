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

namespace {

// Solve dense linear system via Gaussian elimination with partial pivoting.
// (Reference only — adapted from the round-1 sweep spike.)
bool dense_solve(std::vector<std::vector<double>> A, std::vector<double> b,
                  std::vector<double>& x) {
    const size_t n = b.size();
    for (size_t col = 0; col < n; ++col) {
        size_t piv = col;
        for (size_t r = col + 1; r < n; ++r)
            if (std::abs(A[r][col]) > std::abs(A[piv][col])) piv = r;
        if (std::abs(A[piv][col]) < 1e-14) return false;
        std::swap(A[piv], A[col]);
        std::swap(b[piv], b[col]);
        for (size_t r = col + 1; r < n; ++r) {
            double f = A[r][col] / A[col][col];
            for (size_t c = col; c < n; ++c) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    x.assign(n, 0.0);
    for (size_t i = n; i-- > 0;) {
        double s = b[i];
        for (size_t c = i + 1; c < n; ++c) s -= A[i][c] * x[c];
        x[i] = s / A[i][i];
    }
    return true;
}

// Exact LCP solution: A x >= d, x >= psi, complementary; found by
// enumerating all 2^n active sets (n small). Unique for M-matrices.
bool lcp_reference(const std::vector<double>& lower,
                    const std::vector<double>& diag,
                    const std::vector<double>& upper,
                    const std::vector<double>& rhs,
                    const std::vector<double>& psi,
                    std::vector<double>& x_out) {
    const size_t n = diag.size();
    auto Arow = [&](size_t i, size_t j) -> double {
        if (i == j) return diag[i];
        if (j + 1 == i) return lower[j];
        if (j == i + 1) return upper[i];
        return 0.0;
    };
    for (unsigned mask = 0; mask < (1u << n); ++mask) {
        // Active set S: x_i = psi_i for i in S; solve A x = rhs on complement.
        std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
        std::vector<double> b(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            if (mask & (1u << i)) {
                A[i][i] = 1.0;
                b[i] = psi[i];
            } else {
                for (size_t j = 0; j < n; ++j) A[i][j] = Arow(i, j);
                b[i] = rhs[i];
            }
        }
        std::vector<double> x;
        if (!dense_solve(A, b, x)) continue;
        bool ok = true;
        for (size_t i = 0; i < n && ok; ++i) {
            double Ax = 0.0;
            for (size_t j = 0; j < n; ++j) Ax += Arow(i, j) * x[j];
            if (mask & (1u << i)) {
                if (Ax < rhs[i] - 1e-9) ok = false;  // dual feasibility
            } else {
                if (x[i] < psi[i] - 1e-9) ok = false;  // primal feasibility
            }
        }
        if (ok) {
            x_out = x;
            return true;
        }
    }
    return false;
}

struct Sys {
    std::vector<double> lower, diag, upper, rhs, psi;
};
Sys mmatrix(size_t n, double w, std::vector<double> psi, double src = 0.02) {
    Sys s{std::vector<double>(n - 1, -w), std::vector<double>(n, 1 + 2 * w),
          std::vector<double>(n - 1, -w), std::vector<double>(n, src),
          std::move(psi)};
    return s;
}
std::vector<double> left_obstacle(size_t n) {
    std::vector<double> p(n);
    for (size_t i = 0; i < n; ++i)
        p[i] = std::max(0.5 - double(i) / double(n - 1), 0.0);
    return p;
}
std::vector<double> right_obstacle(size_t n) {
    std::vector<double> p(n);
    for (size_t i = 0; i < n; ++i)
        p[i] = std::max(double(i) / double(n - 1) - 0.5, 0.0);
    return p;
}
// Nonconstant (sawtooth) ripple on top of the left-decreasing envelope;
// small enough that the true active set stays a left-touching interval.
std::vector<double> sawtooth_obstacle(size_t n) {
    std::vector<double> p(n);
    for (size_t i = 0; i < n; ++i) {
        const double envelope = std::max(0.5 - double(i) / double(n - 1), 0.0);
        const double ripple = 0.0005 * ((i % 2 == 0) ? 1.0 : -1.0);
        p[i] = envelope + ripple;
    }
    return p;
}

// Solves the LCP with the given sweep orientation, checks the result
// against the enumerated dense reference (tight tolerance) and against the
// full KKT conditions (validate_lcp_kkt reports zero violations). Returns
// the computed solution for callers that want additional assertions.
template <mango::LcpActiveSide Side>
std::vector<double> solve_and_check(const Sys& s, double tol = 1e-9) {
    const size_t n = s.diag.size();
    std::vector<double> ref;
    // A failed enumeration leaves `ref` empty; the EXPECT_NEAR loop below
    // would then index it out of bounds, so fail and return early instead
    // of continuing into UB (ASSERT_TRUE itself can't be used here since
    // this helper returns non-void).
    if (!lcp_reference(s.lower, s.diag, s.upper, s.rhs, s.psi, ref)) {
        ADD_FAILURE() << "lcp_reference enumeration found no feasible solution";
        return std::vector<double>(n, std::numeric_limits<double>::quiet_NaN());
    }

    std::vector<double> x(n), ws(2 * n);
    std::vector<uint8_t> mask(n);
    auto r = mango::solve_thomas_projected2<double, Side>(
        std::span<const double>(s.lower), std::span<const double>(s.diag),
        std::span<const double>(s.upper), std::span<const double>(s.rhs),
        std::span<const double>(s.psi), std::span<double>(x),
        std::span<double>(ws), std::span<uint8_t>(mask));
    EXPECT_TRUE(r.ok()) << r.message();

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], ref[i], tol) << "i=" << i;
    }

    auto rep = mango::validate_lcp_kkt<double>(
        std::span<const double>(s.lower), std::span<const double>(s.diag),
        std::span<const double>(s.upper), std::span<const double>(s.rhs),
        std::span<const double>(s.psi), std::span<const double>(x),
        std::span<const uint8_t>(mask));
    EXPECT_EQ(rep.violation_count, 0u)
        << "worst_kind=" << rep.worst_kind
        << " max_violation=" << rep.max_violation;

    return x;
}

}  // namespace

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
