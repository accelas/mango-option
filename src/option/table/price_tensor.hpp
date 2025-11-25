#pragma once

#include "src/support/memory/aligned_arena.hpp"
#include "src/math/safe_math.hpp"
#include <experimental/mdspan>
#include <memory>
#include <expected>
#include <string>
#include <array>

namespace mango {

/// N-dimensional tensor with arena ownership and mdspan view
///
/// Wraps aligned memory allocation with type-safe mdspan access.
/// The arena keeps memory alive via shared_ptr ownership.
///
/// @tparam N Number of dimensions
template <size_t N>
struct PriceTensor {
    std::shared_ptr<memory::AlignedArena> arena;  ///< Owns the memory
    std::experimental::mdspan<double, std::experimental::dextents<size_t, N>> view;  ///< Type-safe view

    /// Create tensor with given shape, allocating from arena
    ///
    /// Allocated memory is uninitialized. Caller must initialize all elements
    /// before use. The tensor owns a shared reference to the arena, keeping
    /// memory alive until all references are destroyed.
    ///
    /// @param shape Number of elements per dimension
    /// @param arena_ptr Shared pointer to memory arena
    /// @return PriceTensor on success, or error message on failure
    ///         Error conditions:
    ///         - Insufficient arena capacity for requested size
    ///         - Arena allocation failure (returns nullptr)
    [[nodiscard]] static std::expected<PriceTensor, std::string>
    create(std::array<size_t, N> shape, std::shared_ptr<memory::AlignedArena> arena_ptr) {
        // Calculate total elements with overflow check
        auto total_result = safe_product(shape);
        if (!total_result.has_value()) {
            return std::unexpected("Tensor shape overflow: product of dimensions exceeds SIZE_MAX");
        }
        size_t total = total_result.value();

        // Allocate from arena
        double* data = arena_ptr->allocate(total);
        if (!data) {
            return std::unexpected("Insufficient arena memory for tensor of size " +
                                 std::to_string(total * sizeof(double)) + " bytes");
        }

        // Create mdspan view
        PriceTensor tensor;
        tensor.arena = arena_ptr;

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

        tensor.view = std::experimental::mdspan<double, std::experimental::dextents<size_t, N>>(data, extents);

        return tensor;
    }
};

} // namespace mango
