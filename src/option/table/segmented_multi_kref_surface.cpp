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

    // Need at least 4 points for Catmull-Rom; fall back to linear otherwise
    if (n < 4) {
        size_t hi_idx = lo_idx + 1;
        double w = (strike - entries[lo_idx].K_ref) /
                   (entries[hi_idx].K_ref - entries[lo_idx].K_ref);
        double v_lo = eval_fn(lo_idx) / entries[lo_idx].K_ref;
        double v_hi = eval_fn(hi_idx) / entries[hi_idx].K_ref;
        return ((1.0 - w) * v_lo + w * v_hi) * strike;
    }

    // Select 4 entries centered on the bracket [lo_idx, lo_idx+1]
    // Clamp so indices stay in [0, n-1]
    size_t i1 = lo_idx;
    size_t i0 = (i1 > 0) ? i1 - 1 : 0;
    size_t i2 = lo_idx + 1;
    size_t i3 = (i2 + 1 < n) ? i2 + 1 : n - 1;

    // If clamped, duplicate the edge point (degrades to quadratic at boundaries)
    std::array<double, 4> x = {
        std::log(entries[i0].K_ref),
        std::log(entries[i1].K_ref),
        std::log(entries[i2].K_ref),
        std::log(entries[i3].K_ref),
    };
    std::array<double, 4> y = {
        eval_fn(i0) / entries[i0].K_ref,
        eval_fn(i1) / entries[i1].K_ref,
        eval_fn(i2) / entries[i2].K_ref,
        eval_fn(i3) / entries[i3].K_ref,
    };

    return catmull_rom(x, y, log_strike) * strike;
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
