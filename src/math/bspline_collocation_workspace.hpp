// src/math/bspline_collocation_workspace.hpp

#pragma once

#include "src/support/lifetime.hpp"
#include <span>
#include <expected>
#include <string>
#include <cstddef>
#include <algorithm>    // for std::max
#include <type_traits>

namespace mango {

/// Workspace for B-spline collocation solver
///
/// Slices external BYTE buffer into typed spans with proper alignment.
/// Uses start_array_lifetime to start object lifetimes, avoiding strict-aliasing UB.
///
/// Required arrays for bandwidth=4 (cubic B-splines):
/// - band_storage: 10n doubles (LAPACK banded format: ldab=10)
/// - lapack_storage: 10n doubles (LU factorization copy)
/// - pivots: n integers (pivot indices) - separate int storage
/// - coeffs: n doubles (result buffer)
///
/// All storage regions are aligned to 64-byte boundaries for SIMD.
///
template<typename T>
struct BSplineCollocationWorkspace {
    static constexpr size_t ALIGNMENT = 64;  // Cache line / AVX-512
    static constexpr size_t BANDWIDTH = 4;
    static constexpr size_t LDAB = 10;  // 2*kl + ku + 1 for bandwidth=4

    static_assert(std::is_trivially_destructible_v<T>,
        "BSplineCollocationWorkspace<T> requires trivially destructible T "
        "because no destructor is called when the workspace goes out of scope");

    /// Effective alignment for each block (max of SIMD alignment and type alignment)
    static constexpr size_t block_alignment_T = std::max(ALIGNMENT, alignof(T));
    static constexpr size_t block_alignment_int = std::max(ALIGNMENT, alignof(int));

    /// Calculate required buffer size in BYTES
    static size_t required_bytes(size_t n) {
        size_t offset = 0;

        // band_storage: 10n × sizeof(T), 64-byte aligned
        offset = align_up(offset, block_alignment_T);
        offset += LDAB * n * sizeof(T);

        // lapack_storage: 10n × sizeof(T), 64-byte aligned
        offset = align_up(offset, block_alignment_T);
        offset += LDAB * n * sizeof(T);

        // pivots: n × sizeof(int), 64-byte aligned
        offset = align_up(offset, block_alignment_int);
        offset += n * sizeof(int);

        // coeffs: n × sizeof(T), 64-byte aligned
        offset = align_up(offset, block_alignment_T);
        offset += n * sizeof(T);

        // Final alignment
        return align_up(offset, ALIGNMENT);
    }

    /// Create workspace from external BYTE buffer
    ///
    /// Uses start_array_lifetime to properly start the lifetime of objects
    /// in the buffer. This is NOT equivalent to std::launder, which only
    /// provides pointer provenance but does NOT create objects.
    static std::expected<BSplineCollocationWorkspace, std::string>
    from_bytes(std::span<std::byte> buffer, size_t n) {
        size_t required = required_bytes(n);
        if (buffer.size() < required) {
            return std::unexpected("Buffer too small for BSplineCollocationWorkspace");
        }

        BSplineCollocationWorkspace ws;
        ws.n_ = n;

        std::byte* ptr = buffer.data();
        size_t offset = 0;

        // band_storage - start lifetime of T[LDAB*n]
        offset = align_up(offset, block_alignment_T);
        ws.band_storage_ = std::span<T>(
            start_array_lifetime<T>(ptr + offset, LDAB * n), LDAB * n);
        offset += LDAB * n * sizeof(T);

        // lapack_storage - start lifetime of T[LDAB*n]
        offset = align_up(offset, block_alignment_T);
        ws.lapack_storage_ = std::span<T>(
            start_array_lifetime<T>(ptr + offset, LDAB * n), LDAB * n);
        offset += LDAB * n * sizeof(T);

        // pivots - start lifetime of int[n]
        offset = align_up(offset, block_alignment_int);
        ws.pivots_ = std::span<int>(
            start_array_lifetime<int>(ptr + offset, n), n);
        offset += n * sizeof(int);

        // coeffs - start lifetime of T[n]
        offset = align_up(offset, block_alignment_T);
        ws.coeffs_ = std::span<T>(
            start_array_lifetime<T>(ptr + offset, n), n);

        return ws;
    }

    // Accessors - return typed spans (lifetime started by from_bytes)
    std::span<T> band_storage() { return band_storage_; }
    std::span<T> lapack_storage() { return lapack_storage_; }
    std::span<int> pivots() { return pivots_; }  // Properly typed int span
    std::span<T> coeffs() { return coeffs_; }

    // Const accessors
    std::span<const T> band_storage() const { return band_storage_; }
    std::span<const T> lapack_storage() const { return lapack_storage_; }
    std::span<const int> pivots() const { return pivots_; }
    std::span<const T> coeffs() const { return coeffs_; }

    size_t size() const { return n_; }

private:
    size_t n_ = 0;
    std::span<T> band_storage_;
    std::span<T> lapack_storage_;
    std::span<int> pivots_;  // int, not T
    std::span<T> coeffs_;
};

} // namespace mango
