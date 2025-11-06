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
    /// Currently uses linear interpolation (TODO: cubic basis functions).
    ///
    /// @param x_eval Evaluation point
    /// @param data Pre-computed values at grid points (same grid as build())
    /// @return Interpolated value
    [[nodiscard]] double eval_from_data(double x_eval, std::span<const double> data) const noexcept {
        if (x_.empty() || data.size() != x_.size()) {
            return 0.0;
        }

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

    /// Check if spline has been built
    [[nodiscard]] bool is_built() const noexcept {
        return built_;
    }

    /// Get the underlying spline (for advanced use)
    [[nodiscard]] const CubicSpline<double>& get_spline() const noexcept {
        return spline_;
    }

private:
    CubicSpline<double> spline_;
    std::vector<double> x_;  // Grid points (for eval_from_data)
    std::vector<double> y_;  // Values (for eval_from_data)
    bool built_ = false;
};

}  // namespace mango
