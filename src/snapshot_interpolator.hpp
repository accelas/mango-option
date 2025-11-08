#pragma once

#include "cubic_spline_solver.hpp"
#include <span>
#include <vector>
#include <optional>

namespace mango {

/// Cubic spline interpolator for snapshot data
///
/// Modern C++ implementation using CubicSpline<double>.
/// Supports interpolation from pre-computed derivative arrays.
class SnapshotInterpolator {
public:
    SnapshotInterpolator() = default;

    // Rule of five: default move, delete copy
    SnapshotInterpolator(const SnapshotInterpolator&) = delete;
    SnapshotInterpolator& operator=(const SnapshotInterpolator&) = delete;
    SnapshotInterpolator(SnapshotInterpolator&&) = default;
    SnapshotInterpolator& operator=(SnapshotInterpolator&&) = default;
    ~SnapshotInterpolator() = default;

    /// Build spline from snapshot data
    ///
    /// @param x X-coordinates (must be strictly increasing)
    /// @param y Y-coordinates
    /// @return Optional error message (nullopt on success)
    ///
    /// @note On failure, the interpolator is reset to unbuilt state.
    ///       Previous state is NOT preserved across failed rebuilds.
    [[nodiscard]] std::optional<std::string_view> build(
        std::span<const double> x,
        std::span<const double> y)
    {
        // Reset to unbuilt state BEFORE attempting build
        // This prevents state corruption if build fails
        built_ = false;

        // Build cubic spline first (validates input)
        auto error = spline_.build(x, y);
        if (error.has_value()) {
            // Leave in unbuilt state on failure
            x_.clear();
            y_.clear();
            return error;
        }

        // Only update grid storage after successful build
        x_.assign(x.begin(), x.end());
        y_.assign(y.begin(), y.end());
        built_ = true;
        return std::nullopt;
    }

    /// Rebuild spline with new y-values on the same x-grid
    ///
    /// PERFORMANCE: Much faster than build() when grid unchanged.
    /// Reuses cached interval widths and grid structure.
    ///
    /// @param y New Y-coordinates (must match existing grid)
    /// @return Optional error message (nullopt on success)
    ///
    /// @pre build() must have been called successfully at least once
    [[nodiscard]] std::optional<std::string_view> rebuild_same_grid(
        std::span<const double> y)
    {
        if (!built_) {
            return "Must call build() before rebuild_same_grid()";
        }

        auto error = spline_.rebuild_same_grid(y);
        if (error.has_value()) {
            built_ = false;
            return error;
        }

        // Update stored y-values
        y_.assign(y.begin(), y.end());
        return std::nullopt;
    }

    /// Evaluate interpolant
    ///
    /// @param x_eval Evaluation point
    /// @return Interpolated value (or 0 if not built)
    [[nodiscard]] double eval(double x_eval) const noexcept {
        if (!built_) return 0.0;
        return spline_.eval(x_eval);
    }

    /// Interpolate from pre-computed data array
    ///
    /// Uses the same grid as build() but evaluates with different data.
    /// Useful for evaluating derivatives without re-building the spline.
    ///
    /// Uses cubic spline interpolation for smoothness.
    ///
    /// @param x_eval Evaluation point
    /// @param data Pre-computed values at grid points (same grid as build())
    /// @return Interpolated value
    [[nodiscard]] double eval_from_data(double x_eval, std::span<const double> data) const noexcept {
        if (x_.empty() || data.size() != x_.size()) {
            return 0.0;
        }

        // Build a temporary cubic spline for the derivative data
        // PERFORMANCE: Uses cached grid structure via rebuild_same_grid in derived_spline_
        // This is much faster than building from scratch

        // Lazy initialization of derived spline
        if (!derived_spline_built_ || derived_data_.size() != data.size()) {
            // First time or size changed: build from scratch
            auto error = derived_spline_.build(std::span{x_}, data);
            if (error.has_value()) {
                // Fallback to linear interpolation on error
                return eval_from_data_linear(x_eval, data);
            }
            derived_spline_built_ = true;
            derived_data_.assign(data.begin(), data.end());
        } else {
            // Grid same, just rebuild with new data (fast path)
            auto error = derived_spline_.rebuild_same_grid(data);
            if (error.has_value()) {
                // Fallback to linear interpolation on error
                return eval_from_data_linear(x_eval, data);
            }
            derived_data_.assign(data.begin(), data.end());
        }

        return derived_spline_.eval(x_eval);
    }

    /// Check if spline has been built
    [[nodiscard]] bool is_built() const noexcept {
        return built_;
    }

    /// Get the underlying spline (for advanced use)
    [[nodiscard]] const CubicSpline<double>& get_spline() const noexcept {
        return spline_;
    }

private:
    /// Linear interpolation fallback
    [[nodiscard]] double eval_from_data_linear(double x_eval, std::span<const double> data) const noexcept {
        // Find bracketing interval using binary search
        if (x_eval <= x_.front()) {
            return data.front();
        }
        if (x_eval >= x_.back()) {
            return data.back();
        }

        // Binary search for interval
        auto it = std::lower_bound(x_.begin(), x_.end(), x_eval);
        size_t i = std::distance(x_.begin(), it);

        // Adjust to get the interval [x[i], x[i+1]]
        if (i > 0 && (i == x_.size() || x_[i] > x_eval)) {
            --i;
        }
        i = std::min(i, x_.size() - 2);

        // Linear interpolation
        const double t = (x_eval - x_[i]) / (x_[i+1] - x_[i]);
        return (1.0 - t) * data[i] + t * data[i+1];
    }

    CubicSpline<double> spline_;
    std::vector<double> x_;  // Grid points (for eval_from_data)
    std::vector<double> y_;  // Values (for eval_from_data)
    bool built_ = false;

    // Cached spline for derivative data (mutable for lazy initialization in const methods)
    mutable CubicSpline<double> derived_spline_;
    mutable std::vector<double> derived_data_;
    mutable bool derived_spline_built_ = false;
};

}  // namespace mango
