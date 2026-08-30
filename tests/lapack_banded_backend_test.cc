// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include "mango/math/lapack_banded_backend.hpp"

// Regression: LAPACK calls and banded-layout knowledge leaked above the
// solver layer (issue #464)
// Bug: bspline_collocation.hpp held six direct LAPACKE_* call sites plus
// layout math; the backend concept makes the library swappable.
static_assert(mango::BandedSolverBackend<mango::LapackBandedBackend, double>);
static_assert(!mango::BandedSolverBackend<mango::LapackBandedBackend, float>);
static_assert(!mango::BandedSolverBackend<int, double>);

TEST(LapackBandedBackendTest, PackMatchesLapackBandedLayout) {
    // 5-point cubic band (bandwidth 4): row i covers cols [col_start[i], ...)
    constexpr std::size_t n = 5, bw = 4;
    std::vector<double> band_rows(n * bw);
    std::iota(band_rows.begin(), band_rows.end(), 1.0);   // distinct values
    std::vector<int> col_start{0, 0, 0, 1, 1};
    std::vector<double> factors(
        mango::LapackBandedBackend::factor_storage_size(n, bw), -7.0);
    mango::LapackBandedBackend::pack(band_rows, col_start, n, bw, factors);
    // Expected offsets from the documented LAPACK formula
    // (kl+ku+i-j) + j*ldab with kl=ku=3, ldab=10:
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j_end = std::min(static_cast<std::size_t>(col_start[i]) + bw, n);
        for (std::size_t j = col_start[i]; j < j_end; ++j) {
            const std::size_t off = (6 + i - j) + j * 10;
            EXPECT_EQ(factors[off], band_rows[i * bw + (j - col_start[i])])
                << "i=" << i << " j=" << j;
        }
    }
    // Everything outside the band was zeroed, not left at the sentinel
    EXPECT_EQ(std::count(factors.begin(), factors.end(), -7.0), 0);
}

TEST(LapackBandedBackendTest, FactorizeSolveRoundTrip) {
    // Identity-band system: diagonal-only band solves x = b exactly
    constexpr std::size_t n = 6, bw = 4;
    std::vector<double> band_rows(n * bw, 0.0);
    std::vector<int> col_start(n);
    for (std::size_t i = 0; i < n; ++i) {
        col_start[i] = static_cast<int>(i > 2 ? i - 2 : 0);
        band_rows[i * bw + (i - col_start[i])] = 2.0;   // diag = 2
    }
    std::vector<double> factors(mango::LapackBandedBackend::factor_storage_size(n, bw));
    std::vector<lapack_int> pivots(mango::LapackBandedBackend::pivot_storage_size(n, bw));
    mango::LapackBandedBackend::pack(band_rows, col_start, n, bw, factors);
    ASSERT_TRUE(mango::LapackBandedBackend::factorize(factors, pivots, n, bw).ok());
    std::vector<double> x{2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
    ASSERT_TRUE(mango::LapackBandedBackend::solve(factors, pivots, x, n, bw).ok());
    for (std::size_t i = 0; i < n; ++i) EXPECT_DOUBLE_EQ(x[i], (i + 1) * 1.0);
    EXPECT_GT(mango::LapackBandedBackend::condition(factors, pivots, 2.0, n, bw), 0.0);
}
