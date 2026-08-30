// SPDX-License-Identifier: MIT
/**
 * @file lapack_banded_backend.hpp
 * @brief LAPACK implementation of the `BandedSolverBackend` concept
 *
 * `mango::LapackBandedBackend` is a stateless policy type: it owns the
 * LAPACK banded storage layout, the pivot representation, and every call
 * into LAPACKE (dgbtrf/dgbtrs/dgbcon). Consumers (e.g. the B-spline
 * collocation solver) depend only on `BandedSolverBackend`, not on LAPACK
 * directly.
 */

#pragma once

#include "mango/math/banded_solver_backend.hpp"
#include "mango/math/lapack_banded_layout.hpp"
#include <experimental/mdspan>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <lapacke.h>

namespace mango {

/// LAPACK-backed banded solver policy (double precision only)
struct LapackBandedBackend {
    using pivot_type = lapack_int;

    /// Storage required for the packed banded factors: ldab * n, where
    /// ldab = 2*kl + ku + 1 with kl = ku = bandwidth - 1.
    static constexpr std::size_t factor_storage_size(std::size_t n, std::size_t bandwidth) {
        return (3 * (bandwidth - 1) + 1) * n;
    }

    /// Storage required for pivots: one per row.
    static constexpr std::size_t pivot_storage_size(std::size_t n, std::size_t) {
        return n;
    }

    /// Pack the neutral band-row representation into LAPACK banded storage.
    ///
    /// `band_rows` is n*bandwidth row-major values (row i occupies
    /// `band_rows[i*bandwidth .. i*bandwidth+bandwidth)`), and `col_start[i]`
    /// is the first matrix column that row i's band covers. This mirrors
    /// `fill_lapack_band` from bspline_collocation.hpp.
    static void pack(std::span<const double> band_rows, std::span<const int> col_start,
                      std::size_t n, std::size_t bandwidth, std::span<double> factors) {
        using extents_type = std::experimental::dextents<std::size_t, 2>;
        using mapping_type = lapack_banded_layout::mapping<extents_type>;
        using band_view_type = std::experimental::mdspan<double, extents_type, lapack_banded_layout>;

        const auto kl = static_cast<typename extents_type::index_type>(bandwidth - 1);
        const auto ku = kl;

        // Zero the destination band storage
        std::fill(factors.begin(), factors.end(), 0.0);

        band_view_type out_band(factors.data(), mapping_type(extents_type{n, n}, kl, ku));

        for (std::size_t i = 0; i < n; ++i) {
            const int row_col_start = col_start[i];
            const int col_end = std::min(row_col_start + static_cast<int>(bandwidth), static_cast<int>(n));

            for (int j = row_col_start; j < col_end; ++j) {
                const std::size_t band_idx = static_cast<std::size_t>(j - row_col_start);
                const double value = band_rows[i * bandwidth + band_idx];
                out_band[i, static_cast<std::size_t>(j)] = value;
            }
        }
    }

    /// LU-factorize the packed band in place via LAPACKE_dgbtrf.
    static BandedResult<double> factorize(std::span<double> factors, std::span<lapack_int> pivots,
                                           std::size_t n, std::size_t bandwidth) {
        using Result = BandedResult<double>;

        const auto lapack_n = static_cast<lapack_int>(n);
        const auto kl = static_cast<lapack_int>(bandwidth - 1);
        const auto ku = kl;
        const auto ldab = static_cast<lapack_int>(3 * (bandwidth - 1) + 1);

        const lapack_int info = LAPACKE_dgbtrf(
            LAPACK_COL_MAJOR,
            lapack_n, lapack_n, kl, ku,
            factors.data(),
            ldab,
            pivots.data()
        );

        if (info < 0) {
            return Result::error_result("LAPACKE_dgbtrf: invalid argument");
        }
        if (info > 0) {
            return Result::error_result("Matrix is singular");
        }

        return Result::ok_result();
    }

    /// Solve LU·x = b in place via LAPACKE_dgbtrs, using factors from
    /// `factorize()`.
    static BandedResult<double> solve(std::span<const double> factors, std::span<const lapack_int> pivots,
                                       std::span<double> x, std::size_t n, std::size_t bandwidth) {
        using Result = BandedResult<double>;

        const auto lapack_n = static_cast<lapack_int>(n);
        const auto kl = static_cast<lapack_int>(bandwidth - 1);
        const auto ku = kl;
        const auto ldab = static_cast<lapack_int>(3 * (bandwidth - 1) + 1);
        const lapack_int nrhs = 1;

        const lapack_int info = LAPACKE_dgbtrs(
            LAPACK_COL_MAJOR,
            'N',
            lapack_n, kl, ku, nrhs,
            factors.data(),
            ldab,
            pivots.data(),
            x.data(),
            lapack_n
        );

        if (info < 0) {
            return Result::error_result("LAPACKE_dgbtrs: invalid argument");
        }
        if (info > 0) {
            return Result::error_result("LAPACKE_dgbtrs: zero pivot");
        }

        return Result::ok_result();
    }

    /// Estimate the condition number from LU factors via LAPACKE_dgbcon.
    static double condition(std::span<const double> factors, std::span<const lapack_int> pivots,
                             double norm, std::size_t n, std::size_t bandwidth) {
        if (norm == 0.0) {
            return std::numeric_limits<double>::infinity();
        }

        const auto lapack_n = static_cast<lapack_int>(n);
        const auto kl = static_cast<lapack_int>(bandwidth - 1);
        const auto ku = kl;
        const auto ldab = static_cast<lapack_int>(3 * (bandwidth - 1) + 1);

        double rcond = 0.0;

        const lapack_int info = LAPACKE_dgbcon(
            LAPACK_COL_MAJOR,
            '1',
            lapack_n, kl, ku,
            factors.data(),
            ldab,
            pivots.data(),
            norm,
            &rcond
        );

        if (info != 0 || rcond == 0.0) {
            return std::numeric_limits<double>::infinity();
        }

        return 1.0 / rcond;
    }
};

}  // namespace mango
