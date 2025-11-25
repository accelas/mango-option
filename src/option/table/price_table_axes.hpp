#pragma once

#include <array>
#include <vector>
#include <string>
#include <expected>
#include <algorithm>
#include "src/support/error_types.hpp"
#include "src/math/safe_math.hpp"

namespace mango {

/// Metadata for N-dimensional price table axes
///
/// Stores grid points and optional axis names for each dimension.
/// All grids must be strictly monotonic increasing.
///
/// @tparam N Number of dimensions (axes)
template <size_t N>
struct PriceTableAxes {
    std::array<std::vector<double>, N> grids;  ///< Grid points per axis
    std::array<std::string, N> names;          ///< Optional names (e.g., "moneyness", "maturity")

    /// Calculate total number of grid points (product of all axis sizes)
    ///
    /// Uses safe multiplication with overflow detection via __int128.
    /// Returns 0 on overflow (callers should validate grids first).
    [[nodiscard]] size_t total_points() const noexcept {
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            auto result = safe_multiply(total, grids[i].size());
            if (!result.has_value()) {
                return 0;  // Overflow - return 0 to signal error
            }
            total = result.value();
        }
        return total;
    }

    /// Calculate total number of grid points with overflow checking
    ///
    /// @return Total points or OverflowError if product exceeds SIZE_MAX
    [[nodiscard]] std::expected<size_t, OverflowError> total_points_checked() const noexcept {
        const auto s = shape();
        return safe_product(std::span<const size_t, N>(s));
    }

    /// Validate all grids are non-empty and strictly monotonic
    ///
    /// @return Empty expected on success, ValidationError on failure
    [[nodiscard]] std::expected<void, ValidationError> validate() const {
        for (size_t i = 0; i < N; ++i) {
            if (grids[i].empty()) {
                return std::unexpected(ValidationError(
                    ValidationErrorCode::InvalidGridSize,
                    0.0,
                    i));
            }

            // Check strict monotonicity
            for (size_t j = 1; j < grids[i].size(); ++j) {
                if (grids[i][j] <= grids[i][j-1]) {
                    return std::unexpected(ValidationError(
                        ValidationErrorCode::UnsortedGrid,
                        grids[i][j],
                        i));
                }
            }
        }
        return {};
    }

    /// Get shape (number of points per axis)
    [[nodiscard]] std::array<size_t, N> shape() const noexcept {
        std::array<size_t, N> s;
        for (size_t i = 0; i < N; ++i) {
            s[i] = grids[i].size();
        }
        return s;
    }
};

} // namespace mango
