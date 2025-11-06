#pragma once

#include <span>
#include <memory>
#include <vector>

extern "C" {
#include "src/cubic_spline.h"
}

namespace mango {

/// Cubic spline interpolator for snapshot data
///
/// Supports interpolation from pre-computed derivative arrays
/// (avoids re-differentiation of spline).
class SnapshotInterpolator {
public:
    SnapshotInterpolator() = default;
    ~SnapshotInterpolator() {
        if (spline_) {
            pde_spline_destroy(spline_);
        }
    }

    SnapshotInterpolator(const SnapshotInterpolator&) = delete;
    SnapshotInterpolator& operator=(const SnapshotInterpolator&) = delete;

    /// Build spline from snapshot data
    void build(std::span<const double> x, std::span<const double> y) {
        if (spline_) {
            pde_spline_destroy(spline_);
            spline_ = nullptr;
        }

        x_.assign(x.begin(), x.end());
        y_.assign(y.begin(), y.end());

        spline_ = pde_spline_create(x_.data(), y_.data(), x_.size());
    }

    /// Evaluate interpolant
    double eval(double x_eval) const {
        return pde_spline_eval(spline_, x_eval);
    }

    /// Interpolate from pre-computed data array
    ///
    /// Uses cubic spline for C2-continuous interpolation.
    /// Builds a temporary spline from the external data.
    ///
    /// @param x_eval Evaluation point
    /// @param data Pre-computed values at grid points (same grid as build())
    /// @return Interpolated value
    double eval_from_data(double x_eval, std::span<const double> data) const {
        // Build cubic spline from external data
        // Note: This requires one spline construction per call, but provides
        // C2 continuity essential for Newton convergence
        CubicSpline* temp_spline = pde_spline_create(x_.data(), data.data(), x_.size());
        if (!temp_spline) {
            throw std::runtime_error("Failed to create temporary spline");
        }

        double result = pde_spline_eval(temp_spline, x_eval);
        pde_spline_destroy(temp_spline);
        return result;
    }

private:
    CubicSpline* spline_ = nullptr;
    std::vector<double> x_;
    std::vector<double> y_;
};

}  // namespace mango
