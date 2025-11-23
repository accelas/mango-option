#pragma once

#include <array>
#include <vector>
#include <string>
#include <expected>
#include <algorithm>

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
    [[nodiscard]] size_t total_points() const noexcept {
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            total *= grids[i].size();
        }
        return total;
    }

    /// Validate all grids are non-empty and strictly monotonic
    ///
    /// @return Empty expected on success, error message on failure
    [[nodiscard]] std::expected<void, std::string> validate() const {
        for (size_t i = 0; i < N; ++i) {
            if (grids[i].empty()) {
                return std::unexpected("Axis " + std::to_string(i) + " is empty");
            }

            // Check strict monotonicity
            for (size_t j = 1; j < grids[i].size(); ++j) {
                if (grids[i][j] <= grids[i][j-1]) {
                    return std::unexpected(
                        "Axis " + std::to_string(i) + " is not strictly monotonic at index " +
                        std::to_string(j));
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
