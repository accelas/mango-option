// SPDX-License-Identifier: MIT
#include "src/option/table/segmented_multi_kref_surface.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace mango {

std::expected<SegmentedMultiKRefSurface, ValidationError>
SegmentedMultiKRefSurface::create(std::vector<Entry> entries) {
    if (entries.empty()) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidGridSize, 0.0, 0));
    }

    // Validate K_ref consistency: Entry.K_ref must match its surface's K_ref
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].K_ref <= 0.0 ||
            std::abs(entries[i].K_ref - entries[i].surface.K_ref()) > 1e-10) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidStrike, entries[i].K_ref, i));
        }
    }

    // Sort entries by K_ref
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.K_ref < b.K_ref; });

    // Remove duplicate K_refs (would cause division by zero in interpolation)
    entries.erase(
        std::unique(entries.begin(), entries.end(),
                    [](const Entry& a, const Entry& b) {
                        return std::abs(a.K_ref - b.K_ref) < 1e-12;
                    }),
        entries.end());

    if (entries.empty()) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidGridSize, 0.0, 0));
    }

    // Verify all entries share option_type and dividend_yield
    auto expected_type = entries.front().surface.option_type();
    auto expected_yield = entries.front().surface.dividend_yield();
    for (size_t i = 1; i < entries.size(); ++i) {
        if (entries[i].surface.option_type() != expected_type) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::OptionTypeMismatch, static_cast<double>(i), i));
        }
        if (std::abs(entries[i].surface.dividend_yield() - expected_yield) > 1e-10) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::DividendYieldMismatch, entries[i].surface.dividend_yield(), i));
        }
    }

    // Compute bounds as intersection across all entries
    double m_min = -std::numeric_limits<double>::infinity();
    double m_max = std::numeric_limits<double>::infinity();
    double tau_min = -std::numeric_limits<double>::infinity();
    double tau_max = std::numeric_limits<double>::infinity();
    double sigma_min = -std::numeric_limits<double>::infinity();
    double sigma_max = std::numeric_limits<double>::infinity();
    double rate_min = -std::numeric_limits<double>::infinity();
    double rate_max = std::numeric_limits<double>::infinity();

    for (const auto& entry : entries) {
        m_min = std::max(m_min, entry.surface.m_min());
        m_max = std::min(m_max, entry.surface.m_max());
        tau_min = std::max(tau_min, entry.surface.tau_min());
        tau_max = std::min(tau_max, entry.surface.tau_max());
        sigma_min = std::max(sigma_min, entry.surface.sigma_min());
        sigma_max = std::min(sigma_max, entry.surface.sigma_max());
        rate_min = std::max(rate_min, entry.surface.rate_min());
        rate_max = std::min(rate_max, entry.surface.rate_max());
    }

    // Validate intersection is non-degenerate
    if (m_min >= m_max || tau_min >= tau_max ||
        sigma_min >= sigma_max || rate_min >= rate_max) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidBounds, 0.0, 0));
    }

    SegmentedMultiKRefSurface result;
    result.entries_ = std::move(entries);
    result.m_min_ = m_min;
    result.m_max_ = m_max;
    result.tau_min_ = tau_min;
    result.tau_max_ = tau_max;
    result.sigma_min_ = sigma_min;
    result.sigma_max_ = sigma_max;
    result.rate_min_ = rate_min;
    result.rate_max_ = rate_max;
    result.option_type_ = result.entries_.front().surface.option_type();
    result.dividend_yield_ = result.entries_.front().surface.dividend_yield();

    return result;
}

// Catmull-Rom cubic interpolation: given 4 values y[0..3] at positions x[0..3],
// evaluate at position t (where x[1] <= t <= x[2]).
static double catmull_rom(const std::array<double, 4>& x,
                          const std::array<double, 4>& y,
                          double t) {
    // Normalize to [0, 1] on the central interval [x1, x2]
    double h = x[2] - x[1];
    double u = (t - x[1]) / h;
    double u2 = u * u;
    double u3 = u2 * u;

    // Slopes at x[1] and x[2] using Catmull-Rom formula
    double m1 = (y[2] - y[0]) / (x[2] - x[0]);
    double m2 = (y[3] - y[1]) / (x[3] - x[1]);

    // Scale slopes to the interval width
    m1 *= h;
    m2 *= h;

    // Hermite basis
    return (2.0 * u3 - 3.0 * u2 + 1.0) * y[1]
         + (u3 - 2.0 * u2 + u)          * m1
         + (-2.0 * u3 + 3.0 * u2)       * y[2]
         + (u3 - u2)                     * m2;
}

// C1 Hermite interpolation (Fritsch-Carlson) for n=2 or n=3 points.
// x[] are strictly increasing positions in log(K_ref) space.
// y[] are normalized values (value/K_ref).
// t can be inside or outside [x[0], x[n-1]] (extrapolation supported).
static double hermite_interp(const double* x, const double* y, size_t n,
                             double t) {
    if (n == 2) {
        double u = (t - x[0]) / (x[1] - x[0]);
        return (1.0 - u) * y[0] + u * y[1];
    }

    // n == 3: Full Fritsch-Carlson for non-uniform spacing
    const double h0 = x[1] - x[0];
    const double h1 = x[2] - x[1];
    const double d0 = (y[1] - y[0]) / h0;
    const double d1 = (y[2] - y[1]) / h1;

    // Weighted endpoint slopes (correct for non-uniform spacing)
    double m0 = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    double m2 = ((2.0 * h1 + h0) * d1 - h1 * d0) / (h0 + h1);

    // Interior slope
    double m1;
    if (d0 * d1 > 0.0) {
        // Data is monotone over both intervals: weighted harmonic mean
        const double w1 = 2.0 * h1 + h0;
        const double w2 = 2.0 * h0 + h1;
        m1 = (w1 + w2) / (w1 / d0 + w2 / d1);
    } else {
        // Non-monotone data: C1 central difference (no monotone force)
        m1 = (h0 * d1 + h1 * d0) / (h0 + h1);
    }

    // Monotonicity limiter on ALL slopes (only when data is monotone)
    if (d0 * d1 > 0.0) {
        auto limit = [](double& m, double d_near) {
            if (m * d_near <= 0.0) { m = 0.0; return; }
            double lim = 3.0 * std::abs(d_near);
            if (std::abs(m) > lim) m = std::copysign(lim, m);
        };
        limit(m0, d0);
        limit(m1, d0);
        limit(m1, d1);
        limit(m2, d1);
    }

    // Choose interval (supports extrapolation: t can be outside [x[0], x[2]])
    size_t i = (t <= x[1]) ? 0 : 1;

    const double hi = x[i + 1] - x[i];
    const double u = (t - x[i]) / hi;
    const double u2 = u * u;
    const double u3 = u2 * u;

    const double mi  = (i == 0) ? m0 : m1;
    const double mi1 = (i == 0) ? m1 : m2;

    // Hermite basis (slopes scaled by interval width)
    return (2.0 * u3 - 3.0 * u2 + 1.0) * y[i]
         + (u3 - 2.0 * u2 + u)          * (mi * hi)
         + (-2.0 * u3 + 3.0 * u2)       * y[i + 1]
         + (u3 - u2)                     * (mi1 * hi);
}

// Find the index of the lower bracketing entry for a given strike.
// Returns the index i such that entries_[i].K_ref <= strike < entries_[i+1].K_ref.
size_t SegmentedMultiKRefSurface::find_bracket(double strike) const {
    auto it = std::upper_bound(
        entries_.begin(), entries_.end(), strike,
        [](double s, const Entry& e) { return s < e.K_ref; });
    // it points to first entry with K_ref > strike; prev is the lower bracket
    return static_cast<size_t>(std::prev(it) - entries_.begin());
}

// Interpolate a per-entry quantity across K_refs using Catmull-Rom in log(K_ref)
// on normalized values (value/K_ref), then scale by strike.
// eval_fn(entry_index) should return the raw value (price or vega) for that entry.
template <typename EvalFn>
static double interp_across_krefs(
    const std::vector<SegmentedMultiKRefSurface::Entry>& entries,
    double strike, size_t lo_idx, EvalFn eval_fn) {
    const size_t n = entries.size();
    const double log_strike = std::log(strike);

    // C1 Hermite for 2-3 points (Fritsch-Carlson for n=3, linear for n=2)
    if (n < 4) {
        std::array<double, 3> xs, ys;
        for (size_t i = 0; i < n; ++i) {
            xs[i] = std::log(entries[i].K_ref);
            ys[i] = eval_fn(i) / entries[i].K_ref;
        }
        double result = hermite_interp(xs.data(), ys.data(), n, log_strike);
        return std::max(result, 0.0) * strike;  // clamp non-negative
    }

    // Select 4 entries centered on the bracket [lo_idx, lo_idx+1]
    size_t i1 = lo_idx;
    size_t i2 = lo_idx + 1;

    // Use virtual points at edges (linear extrapolation) instead of clamping
    std::array<double, 4> x, y;
    x[1] = std::log(entries[i1].K_ref);
    x[2] = std::log(entries[i2].K_ref);
    y[1] = eval_fn(i1) / entries[i1].K_ref;
    y[2] = eval_fn(i2) / entries[i2].K_ref;

    if (i1 > 0) {
        size_t i0 = i1 - 1;
        x[0] = std::log(entries[i0].K_ref);
        y[0] = eval_fn(i0) / entries[i0].K_ref;
    } else {
        // Virtual left point: linear extrapolation from [i1, i2]
        x[0] = 2.0 * x[1] - x[2];
        y[0] = 2.0 * y[1] - y[2];
    }

    if (i2 + 1 < n) {
        size_t i3 = i2 + 1;
        x[3] = std::log(entries[i3].K_ref);
        y[3] = eval_fn(i3) / entries[i3].K_ref;
    } else {
        // Virtual right point: linear extrapolation from [i1, i2]
        x[3] = 2.0 * x[2] - x[1];
        y[3] = 2.0 * y[2] - y[1];
    }

    // Clamp result to non-negative (Catmull-Rom can overshoot at edges)
    double result = catmull_rom(x, y, log_strike);
    return std::max(result, 0.0) * strike;
}

double SegmentedMultiKRefSurface::price(double spot, double strike,
                                         double tau, double sigma,
                                         double rate) const {
    const size_t n = entries_.size();

    // Single entry or strike outside K_ref range: use nearest entry
    if (n == 1 || strike <= entries_.front().K_ref) {
        return entries_.front().surface.price(spot, strike, tau, sigma, rate);
    }
    if (strike >= entries_.back().K_ref) {
        return entries_.back().surface.price(spot, strike, tau, sigma, rate);
    }

    size_t lo_idx = find_bracket(strike);

    // Exact match
    if (strike == entries_[lo_idx].K_ref) {
        return entries_[lo_idx].surface.price(spot, strike, tau, sigma, rate);
    }

    return interp_across_krefs(entries_, strike, lo_idx, [&](size_t i) {
        return entries_[i].surface.price(spot, strike, tau, sigma, rate);
    });
}

double SegmentedMultiKRefSurface::vega(double spot, double strike,
                                        double tau, double sigma,
                                        double rate) const {
    const size_t n = entries_.size();

    if (n == 1 || strike <= entries_.front().K_ref) {
        return entries_.front().surface.vega(spot, strike, tau, sigma, rate);
    }
    if (strike >= entries_.back().K_ref) {
        return entries_.back().surface.vega(spot, strike, tau, sigma, rate);
    }

    size_t lo_idx = find_bracket(strike);

    if (strike == entries_[lo_idx].K_ref) {
        return entries_[lo_idx].surface.vega(spot, strike, tau, sigma, rate);
    }

    return interp_across_krefs(entries_, strike, lo_idx, [&](size_t i) {
        return entries_[i].surface.vega(spot, strike, tau, sigma, rate);
    });
}

double SegmentedMultiKRefSurface::m_min() const noexcept { return m_min_; }
double SegmentedMultiKRefSurface::m_max() const noexcept { return m_max_; }
double SegmentedMultiKRefSurface::tau_min() const noexcept { return tau_min_; }
double SegmentedMultiKRefSurface::tau_max() const noexcept { return tau_max_; }
double SegmentedMultiKRefSurface::sigma_min() const noexcept { return sigma_min_; }
double SegmentedMultiKRefSurface::sigma_max() const noexcept { return sigma_max_; }
double SegmentedMultiKRefSurface::rate_min() const noexcept { return rate_min_; }
double SegmentedMultiKRefSurface::rate_max() const noexcept { return rate_max_; }


OptionType SegmentedMultiKRefSurface::option_type() const noexcept {
    return option_type_;
}

double SegmentedMultiKRefSurface::dividend_yield() const noexcept {
    return dividend_yield_;
}
}  // namespace mango
