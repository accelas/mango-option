/**
 * @file yield_curve.hpp
 * @brief Yield curve with log-linear discount interpolation
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace mango {

/// Point on a yield curve: tenor and log-discount factor
struct TenorPoint {
    double tenor;        // Time in years (0.0, 0.25, 0.5, 1.0, ...)
    double log_discount; // ln(D(t)) where D(t) = exp(-integral_0^t r(s)ds)
};

/// Yield curve with log-linear discount interpolation
///
/// Stores discrete tenor points and interpolates ln(D(t)) linearly.
/// This implies piecewise-constant forward rates between tenors,
/// which is arbitrage-free and industry-standard.
class YieldCurve {
    std::vector<TenorPoint> curve_;  // Sorted by tenor, curve_[0].tenor == 0

public:
    /// Default constructor (empty curve)
    YieldCurve() = default;

    /// Construct flat curve (constant rate)
    static YieldCurve flat(double rate) {
        YieldCurve curve;
        // Two points: t=0 and t=100 (far future)
        // ln(D(t)) = -r*t for flat curve
        curve.curve_.push_back({0.0, 0.0});
        curve.curve_.push_back({100.0, -rate * 100.0});
        return curve;
    }

    /// Instantaneous forward rate at time t
    double rate(double t) const {
        if (curve_.size() < 2) return 0.0;
        if (t <= 0.0) return rate_between(0);

        // Binary search for bracketing interval
        auto it = std::upper_bound(curve_.begin(), curve_.end(), t,
            [](double t, const TenorPoint& p) { return t < p.tenor; });

        if (it == curve_.begin()) return rate_between(0);
        if (it == curve_.end()) return rate_between(curve_.size() - 2);

        size_t idx = static_cast<size_t>(std::distance(curve_.begin(), it)) - 1;
        return rate_between(idx);
    }

    /// Discount factor D(t) = exp(ln_D(t))
    double discount(double t) const {
        return std::exp(log_discount(t));
    }

    /// Log discount factor ln(D(t)) via linear interpolation
    double log_discount(double t) const {
        if (curve_.size() < 2) return 0.0;
        if (t <= 0.0) return 0.0;

        // Binary search for bracketing interval
        auto it = std::upper_bound(curve_.begin(), curve_.end(), t,
            [](double t, const TenorPoint& p) { return t < p.tenor; });

        if (it == curve_.begin()) return 0.0;
        if (it == curve_.end()) {
            // Extrapolate flat beyond last tenor
            const auto& last = curve_.back();
            const auto& prev = curve_[curve_.size() - 2];
            double rate = -(last.log_discount - prev.log_discount) /
                          (last.tenor - prev.tenor);
            return last.log_discount - rate * (t - last.tenor);
        }

        // Linear interpolation
        const auto& right = *it;
        const auto& left = *std::prev(it);
        double alpha = (t - left.tenor) / (right.tenor - left.tenor);
        return left.log_discount + alpha * (right.log_discount - left.log_discount);
    }

private:
    /// Forward rate between curve_[idx] and curve_[idx+1]
    double rate_between(size_t idx) const {
        if (idx + 1 >= curve_.size()) return 0.0;
        const auto& left = curve_[idx];
        const auto& right = curve_[idx + 1];
        double dt = right.tenor - left.tenor;
        if (dt <= 0.0) return 0.0;
        return -(right.log_discount - left.log_discount) / dt;
    }
};

}  // namespace mango
