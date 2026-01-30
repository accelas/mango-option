// SPDX-License-Identifier: MIT
#pragma once

#include <experimental/mdspan>
#include <cstddef>

namespace mango {

/// Custom mdspan layout matching LAPACK banded storage
///
/// Maps logical matrix index (i,j) to LAPACK banded storage offset.
/// Formula: AB(kl + ku + i - j, j) where AB is column-major
///
/// LAPACK banded storage stores an n×n matrix A with kl sub-diagonals
/// and ku super-diagonals in a 2D array AB of dimension (ldab, n) where
/// ldab = 2*kl + ku + 1. The j-th column of A is stored in the j-th
/// column of AB.
///
/// Reference: http://www.netlib.org/lapack/explore-html/d3/d49/dgbtrf_8f.html
struct lapack_banded_layout {
    template<class Extents>
    struct mapping {
        using extents_type = Extents;
        using index_type = typename Extents::index_type;
        using size_type = typename Extents::size_type;
        using rank_type = typename Extents::rank_type;
        using layout_type = lapack_banded_layout;

        static_assert(Extents::rank() == 2,
                     "LAPACK banded layout requires rank-2 extents");

    private:
        extents_type extents_;
        index_type kl_;      ///< Number of sub-diagonals
        index_type ku_;      ///< Number of super-diagonals
        index_type ldab_;    ///< Leading dimension (= 2*kl + ku + 1)

    public:
        /// Construct mapping for n×n matrix with kl sub-diagonals and ku super-diagonals
        constexpr mapping(extents_type ext, index_type kl, index_type ku) noexcept
            : extents_(ext)
            , kl_(kl)
            , ku_(ku)
            , ldab_(2 * kl + ku + 1)
        {}

        /// Map (i, j) to flat offset
        ///
        /// Returns offset for LAPACK banded storage: AB(kl + ku + i - j, j)
        /// in column-major layout.
        constexpr index_type operator()(index_type i, index_type j) const noexcept {
            // LAPACK formula: AB(kl + ku + i - j, j)
            const index_type row_offset = kl_ + ku_ + i - j;

            // Column-major: offset = row + col * ldab
            return row_offset + j * ldab_;
        }

        constexpr const extents_type& extents() const noexcept { return extents_; }

        static constexpr bool is_always_unique() noexcept { return true; }
        static constexpr bool is_always_exhaustive() noexcept { return false; }
        static constexpr bool is_always_strided() noexcept { return true; }

        constexpr bool is_unique() const noexcept { return true; }
        constexpr bool is_exhaustive() const noexcept { return false; }
        constexpr bool is_strided() const noexcept { return true; }

        constexpr index_type required_span_size() const noexcept {
            return ldab_ * extents_.extent(1);  // ldab * n
        }

        constexpr index_type stride(rank_type r) const noexcept {
            if (r == 0) return 1;          // Row stride (column-major)
            if (r == 1) return ldab_;      // Column stride
            return 0;
        }
    };
};

}  // namespace mango
