// SPDX-License-Identifier: MIT
#pragma once

#include "mango/support/aligned_allocator.hpp"
#include "mango/math/safe_math.hpp"
#include <experimental/mdspan>
#include <memory>
#include <expected>
#include <string>
#include <array>
#include <vector>

namespace mango {

/// N-dimensional tensor with aligned storage and mdspan view
///
/// Uses AlignedVector (64-byte aligned) for SIMD-friendly memory layout.
/// The tensor owns its storage via shared_ptr for safe sharing.
///
/// @tparam N Number of dimensions
template <size_t N>
struct PriceTensor {
    std::shared_ptr<AlignedVector<double>> storage;  ///< Owns the memory
    std::experimental::mdspan<double, std::experimental::dextents<size_t, N>> view;  ///< Type-safe view

    /// Create tensor with given shape
    ///
    /// Allocated memory is uninitialized. Caller must initialize all elements
    /// before use. The tensor owns a shared reference to the storage, keeping
    /// memory alive until all references are destroyed.
    ///
    /// @param shape Number of elements per dimension
    /// @return PriceTensor on success, or error message on failure
    ///         Error conditions:
    ///         - Shape overflow (product of dimensions exceeds SIZE_MAX)
    [[nodiscard]] static std::expected<PriceTensor, std::string>
    create(std::array<size_t, N> shape) {
        // Calculate total elements with overflow check
        auto total_result = safe_product(shape);
        if (!total_result.has_value()) {
            return std::unexpected("Tensor shape overflow: product of dimensions exceeds SIZE_MAX");
        }
        size_t total = total_result.value();

        // Allocate aligned storage
        auto storage_ptr = std::make_shared<AlignedVector<double>>(total);

        // Create mdspan view
        PriceTensor tensor;
        tensor.storage = storage_ptr;

        // Construct mdspan with dextents
        std::experimental::dextents<size_t, N> extents;
        if constexpr (N == 1) {
            extents = std::experimental::dextents<size_t, 1>(shape[0]);
        } else if constexpr (N == 2) {
            extents = std::experimental::dextents<size_t, 2>(shape[0], shape[1]);
        } else if constexpr (N == 3) {
            extents = std::experimental::dextents<size_t, 3>(shape[0], shape[1], shape[2]);
        } else if constexpr (N == 4) {
            extents = std::experimental::dextents<size_t, 4>(shape[0], shape[1], shape[2], shape[3]);
        } else if constexpr (N == 5) {
            extents = std::experimental::dextents<size_t, 5>(shape[0], shape[1], shape[2], shape[3], shape[4]);
        } else {
            static_assert(N <= 5, "PriceTensor supports up to 5 dimensions");
        }

        tensor.view = std::experimental::mdspan<double, std::experimental::dextents<size_t, N>>(
            storage_ptr->data(), extents);

        return tensor;
    }
};

} // namespace mango
