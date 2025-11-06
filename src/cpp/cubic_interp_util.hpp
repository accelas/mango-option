#pragma once

#include <span>
#include <vector>
#include <memory>

extern "C" {
#include "src/cubic_spline.h"
}

namespace mango {

/**
 * @brief RAII wrapper for cubic spline interpolation
 *
 * Provides C2-continuous interpolation for smooth derivatives,
 * essential for Newton-based convergence in IV solvers.
 *
 * Usage:
 *   CubicInterpolator interp(x_data, y_data);
 *   double value = interp.eval(x_query);
 */
class CubicInterpolator {
public:
    /// Construct spline from data points
    CubicInterpolator(std::span<const double> x, std::span<const double> y)
        : x_(x.begin(), x.end())
        , y_(y.begin(), y.end())
    {
        if (x.size() != y.size()) {
            throw std::invalid_argument("CubicInterpolator: x and y must have same size");
        }
        if (x.size() < 2) {
            throw std::invalid_argument("CubicInterpolator: need at least 2 points");
        }

        spline_ = pde_spline_create(x_.data(), y_.data(), x_.size());
        if (!spline_) {
            throw std::runtime_error("Failed to create cubic spline");
        }
    }

    /// Destructor
    ~CubicInterpolator() {
        if (spline_) {
            pde_spline_destroy(spline_);
        }
    }

    // Delete copy operations (spline_ is unique)
    CubicInterpolator(const CubicInterpolator&) = delete;
    CubicInterpolator& operator=(const CubicInterpolator&) = delete;

    // Allow move operations
    CubicInterpolator(CubicInterpolator&& other) noexcept
        : spline_(other.spline_)
        , x_(std::move(other.x_))
        , y_(std::move(other.y_))
    {
        other.spline_ = nullptr;
    }

    CubicInterpolator& operator=(CubicInterpolator&& other) noexcept {
        if (this != &other) {
            if (spline_) {
                pde_spline_destroy(spline_);
            }
            spline_ = other.spline_;
            x_ = std::move(other.x_);
            y_ = std::move(other.y_);
            other.spline_ = nullptr;
        }
        return *this;
    }

    /// Evaluate spline at point x_eval
    double eval(double x_eval) const {
        return pde_spline_eval(spline_, x_eval);
    }

    /// Evaluate spline derivative at point x_eval
    double eval_derivative(double x_eval) const {
        return pde_spline_eval_derivative(spline_, x_eval);
    }

private:
    CubicSpline* spline_ = nullptr;
    std::vector<double> x_;
    std::vector<double> y_;
};

/**
 * @brief One-shot cubic spline interpolation (convenience function)
 *
 * For cases where you only need to evaluate the spline once or a few times.
 * If evaluating many times, create a CubicInterpolator and reuse it.
 *
 * @param x Grid points (must be sorted)
 * @param y Function values at grid points
 * @param x_eval Evaluation point
 * @return Interpolated value
 */
inline double cubic_interpolate(std::span<const double> x,
                                  std::span<const double> y,
                                  double x_eval) {
    CubicInterpolator interp(x, y);
    return interp.eval(x_eval);
}

}  // namespace mango
