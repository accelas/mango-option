#pragma once

#include <span>
#include <memory>
#include <vector>

extern "C" {
#include "common/cubic_spline.h"
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
    /// Uses spline basis functions but evaluates with external data.
    /// Avoids re-differentiating the spline.
    ///
    /// @param x_eval Evaluation point
    /// @param data Pre-computed values at grid points (same grid as build())
    /// @return Interpolated value
    double eval_from_data(double x_eval, std::span<const double> data) const {
        // Simple linear interpolation for now (TODO: use spline basis)
        // Find bracketing interval
        size_t i = 0;
        while (i < x_.size() - 1 && x_[i+1] < x_eval) {
            ++i;
        }

        if (i >= x_.size() - 1) {
            return data.back();
        }

        // Linear interpolation
        double t = (x_eval - x_[i]) / (x_[i+1] - x_[i]);
        return (1.0 - t) * data[i] + t * data[i+1];
    }

private:
    CubicSpline* spline_ = nullptr;
    std::vector<double> x_;
    std::vector<double> y_;
};

}  // namespace mango
