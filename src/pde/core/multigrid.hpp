#pragma once

#include "src/pde/core/grid.hpp"
#include "src/support/expected.hpp"
#include <unordered_map>
#include <stdexcept>
#include <numeric>

namespace mango {

/// Grid axes for multi-dimensional grids
enum class GridAxis {
    Space,      // PDE solver spatial dimension
    Time,       // PDE solver time dimension
    Moneyness,  // Price table: S/K ratio
    Maturity,   // Price table: time to maturity
    Volatility, // Price table: implied volatility
    Rate,       // Price table: risk-free rate
    Dividend    // Price table: dividend yield
};

/// Multi-dimensional grid container
///
/// Manages multiple GridBuffer objects indexed by GridAxis.
/// Each axis is independent and can have different spacing (uniform, log, sinh).
/// Total grid points = product of all axis sizes.
///
/// Example: 4D price table (moneyness × maturity × volatility × rate)
/// - 50 moneyness points × 30 maturity × 20 volatility × 10 rate = 300,000 points
class MultiGridBuffer {
public:
    /// Add an axis to the multi-dimensional grid
    ///
    /// @param axis Axis identifier (e.g., GridAxis::Moneyness)
    /// @param spec Grid specification for this axis
    /// @return Success indicator or error message
    expected<void, std::string> add_axis(GridAxis axis, const GridSpec<>& spec) {
        if (buffers_.contains(axis)) {
            return unexpected("Axis already exists in MultiGridBuffer");
        }
        buffers_.emplace(axis, spec.generate());
        return {};
    }

    /// Check if an axis exists
    bool has_axis(GridAxis axis) const {
        return buffers_.contains(axis);
    }

    /// Get size of a specific axis
    ///
    /// @param axis Axis identifier
    /// @return Number of points along this axis or error message
    expected<size_t, std::string> axis_size(GridAxis axis) const {
        auto it = buffers_.find(axis);
        if (it == buffers_.end()) {
            return unexpected("Axis not found in MultiGridBuffer");
        }
        return it->second.size();
    }

    /// Get total number of grid points (product of all axis sizes)
    size_t total_points() const {
        if (buffers_.empty()) return 0;

        return std::accumulate(
            buffers_.begin(), buffers_.end(), size_t{1},
            [](size_t product, const auto& pair) {
                return product * pair.second.size();
            }
        );
    }

    /// Get view of axis data
    ///
    /// @param axis Axis identifier
    /// @return Span view of grid points for this axis or error message
    expected<std::span<const double>, std::string> axis_view(GridAxis axis) const {
        auto it = buffers_.find(axis);
        if (it == buffers_.end()) {
            return unexpected("Axis not found in MultiGridBuffer");
        }
        return it->second.span();
    }

    /// Get number of axes
    size_t n_axes() const {
        return buffers_.size();
    }

private:
    std::unordered_map<GridAxis, GridBuffer<>> buffers_;
};

}  // namespace mango
