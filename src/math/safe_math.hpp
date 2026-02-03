// SPDX-License-Identifier: MIT
#pragma once

#include "mango/support/error_types.hpp"
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>

namespace mango {

/// Safely multiply two size_t values, detecting overflow via __int128
///
/// Uses 128-bit arithmetic to detect when the product exceeds SIZE_MAX.
/// This is more efficient than division-based overflow checks.
///
/// @param a First operand
/// @param b Second operand
/// @return Product if no overflow, OverflowError otherwise
[[nodiscard]] inline std::expected<size_t, OverflowError>
safe_multiply(size_t a, size_t b) noexcept {
    // Use unsigned __int128 for overflow detection
    using uint128_t = unsigned __int128;

    uint128_t product = static_cast<uint128_t>(a) * static_cast<uint128_t>(b);

    if (product > std::numeric_limits<size_t>::max()) {
        return std::unexpected(OverflowError{a, b});
    }

    return static_cast<size_t>(product);
}

/// Safely compute product of multiple size_t values
///
/// Multiplies values sequentially, checking for overflow at each step.
/// Uses C++23 monadic and_then for clean error propagation.
///
/// @tparam Container Range type with size_t-convertible elements
/// @param values Container of values to multiply
/// @return Product if no overflow, OverflowError otherwise
template <typename Container>
[[nodiscard]] std::expected<size_t, OverflowError>
safe_product(const Container& values) noexcept {
    std::expected<size_t, OverflowError> result{1};

    for (const auto& v : values) {
        result = result.and_then([v](size_t acc) {
            return safe_multiply(acc, static_cast<size_t>(v));
        });
        if (!result) return result;
    }

    return result;
}

/// Safely compute product of array dimensions (grid sizes)
///
/// Convenience function for computing tensor sizes from shape arrays.
/// Uses C++23 monadic and_then for clean error propagation.
///
/// @tparam N Number of dimensions
/// @param sizes Array of dimension sizes
/// @return Total size if no overflow, OverflowError otherwise
template <size_t N>
[[nodiscard]] std::expected<size_t, OverflowError>
safe_product(const std::array<size_t, N>& sizes) noexcept {
    std::expected<size_t, OverflowError> result{1};

    for (size_t i = 0; i < N; ++i) {
        result = result.and_then([&sizes, i](size_t acc) {
            return safe_multiply(acc, sizes[i]);
        });
        if (!result) return result;
    }

    return result;
}

/// Safely compute product of span of sizes (compile-time extent optimization)
///
/// When extent is known at compile time, the loop can be unrolled.
/// Uses C++23 monadic and_then for clean error propagation.
///
/// @tparam N Span extent (can be std::dynamic_extent)
/// @param sizes Span of dimension sizes
/// @return Total size if no overflow, OverflowError otherwise
template <size_t N = std::dynamic_extent>
[[nodiscard]] std::expected<size_t, OverflowError>
safe_product(std::span<const size_t, N> sizes) noexcept {
    std::expected<size_t, OverflowError> result{1};

    for (const auto& v : sizes) {
        result = result.and_then([v](size_t acc) {
            return safe_multiply(acc, v);
        });
        if (!result) return result;
    }

    return result;
}

} // namespace mango
