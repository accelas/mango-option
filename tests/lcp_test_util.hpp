// SPDX-License-Identifier: MIT
//
// Shared LCP enumeration/reference-solve test helpers.
//
// Extracted from thomas_solver_lcp_test.cc (Task 1, issue #439) so the
// obstacle+Neumann affine-term unit test in pde_neumann_test.cc (Task 9,
// issue #455) can reuse the exact-by-enumeration reference solver instead
// of duplicating it. Behavior is unchanged from the original file; only the
// location moved.
//
// Anonymous namespace: this header is included into exactly one .cc each
// (no shared library target), so each translation unit gets its own private
// copy of these symbols — no ODR concerns, matching the style of the
// original test file before extraction.
#pragma once

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "mango/math/thomas_solver.hpp"

namespace mango::test_util {
namespace {

// Solve dense linear system via Gaussian elimination with partial pivoting.
// (Reference only — adapted from the round-1 sweep spike.)
//
// [[maybe_unused]] throughout this header: different consumers (the
// original LCP orientation tests, the Neumann affine-term test) exercise
// different subsets of these helpers, and each is a private per-TU copy
// (anonymous namespace) under -Werror=unused-function.
[[maybe_unused]] bool dense_solve(std::vector<std::vector<double>> A, std::vector<double> b,
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
[[maybe_unused]] bool lcp_reference(const std::vector<double>& lower,
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
[[maybe_unused]] Sys mmatrix(size_t n, double w, std::vector<double> psi, double src = 0.02) {
    Sys s{std::vector<double>(n - 1, -w), std::vector<double>(n, 1 + 2 * w),
          std::vector<double>(n - 1, -w), std::vector<double>(n, src),
          std::move(psi)};
    return s;
}
[[maybe_unused]] std::vector<double> left_obstacle(size_t n) {
    std::vector<double> p(n);
    for (size_t i = 0; i < n; ++i)
        p[i] = std::max(0.5 - double(i) / double(n - 1), 0.0);
    return p;
}
[[maybe_unused]] std::vector<double> right_obstacle(size_t n) {
    std::vector<double> p(n);
    for (size_t i = 0; i < n; ++i)
        p[i] = std::max(double(i) / double(n - 1) - 0.5, 0.0);
    return p;
}
// Nonconstant (sawtooth) ripple on top of the left-decreasing envelope;
// small enough that the true active set stays a left-touching interval.
[[maybe_unused]] std::vector<double> sawtooth_obstacle(size_t n) {
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
}  // namespace mango::test_util
