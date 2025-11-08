#pragma once

#include "cubic_spline_solver.hpp"
#include <cstdint>
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

        // CRITICAL: Invalidate derived spline cache when grid changes
        // The derived spline caches interval widths (h_) from the grid,
        // so it must be rebuilt from scratch when x_ changes
        invalidate_derived_spline();

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

        // Bump epoch to invalidate stale derivative caches
        ++data_epoch_;

        return std::nullopt;
    }

    /// Get current data epoch
    ///
    /// The epoch increments whenever rebuild_same_grid() succeeds.
    /// Callers can use this to track data freshness for derivative caches.
    ///
    /// @return Current epoch value
    [[nodiscard]] uint64_t current_epoch() const noexcept {
        return data_epoch_;
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
    /// PERFORMANCE: Uses epoch-based caching to avoid thrashing when
    /// alternating between different arrays (e.g., first and second derivatives).
    /// Maintains a 2-slot LRU cache. Cache entries are keyed by (pointer, epoch)
    /// for O(1) freshness checks without O(n) content comparison.
    ///
    /// @param x_eval Evaluation point
    /// @param data Pre-computed values at grid points (same grid as build())
    /// @param epoch Data epoch from current_epoch() after rebuild_same_grid()
    /// @return Interpolated value
    [[nodiscard]] double eval_from_data(double x_eval, std::span<const double> data, uint64_t epoch) const {
        if (x_.empty() || data.size() != x_.size()) {
            return 0.0;
        }

        // PERFORMANCE: Epoch-based cache using pointer + epoch comparison (O(1))
        // Handles both alternating arrays AND reused buffers with changed values
        const double* data_ptr = data.data();

        // Check cache[0]: pointer match + epoch match
        if (cache_[0].data_ptr == data_ptr && cache_[0].built && cache_[0].epoch == epoch) {
            // Cache hit: same pointer, same epoch
            return cache_[0].spline.eval(x_eval);
        }

        // Check cache[1]: pointer match + epoch match
        if (cache_[1].data_ptr == data_ptr && cache_[1].built && cache_[1].epoch == epoch) {
            // Cache hit in slot 1: promote to slot 0 (LRU)
            std::swap(cache_[0], cache_[1]);
            return cache_[0].spline.eval(x_eval);
        }

        // Cache miss (pointer or epoch mismatch): rebuild
        // Evict LRU (slot 1) and build in slot 0
        cache_[1] = std::move(cache_[0]);  // Demote slot 0 to slot 1

        auto error = cache_[0].spline.build(std::span{x_}, data);
        if (error.has_value()) {
            // Fallback to linear interpolation on error
            cache_[0].built = false;
            cache_[0].data_ptr = nullptr;
            cache_[0].epoch = 0;
            return eval_from_data_linear(x_eval, data);
        }

        cache_[0].built = true;
        cache_[0].data_ptr = data_ptr;
        cache_[0].epoch = epoch;
        return cache_[0].spline.eval(x_eval);
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
    /// Cache entry for derived splines
    struct DerivedSplineCache {
        CubicSpline<double> spline;
        const double* data_ptr = nullptr;     ///< Pointer for fast lookup
        uint64_t epoch = 0;                   ///< Data epoch for O(1) freshness check
        bool built = false;
    };

    /// Invalidate derived spline cache
    ///
    /// CRITICAL: Must be called whenever the grid (x_) changes.
    /// The derived spline caches interval widths from x_, so a grid
    /// change makes its cached state invalid.
    void invalidate_derived_spline() noexcept {
        cache_[0] = DerivedSplineCache{};
        cache_[1] = DerivedSplineCache{};
    }

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

    // Epoch counter for tracking data freshness (incremented on rebuild_same_grid)
    uint64_t data_epoch_ = 0;

    // LRU cache for derived splines (size 2: handles first + second derivative)
    // Mutable for lazy initialization in const eval_from_data
    mutable DerivedSplineCache cache_[2];
};

}  // namespace mango
