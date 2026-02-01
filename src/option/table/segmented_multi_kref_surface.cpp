// SPDX-License-Identifier: MIT
#include "src/option/table/segmented_multi_kref_surface.hpp"

#include <algorithm>
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

    return result;
}

double SegmentedMultiKRefSurface::price(double spot, double strike,
                                         double tau, double sigma,
                                         double rate) const {
    const size_t n = entries_.size();

    // Single entry or strike at/below lowest K_ref: use first entry
    if (n == 1 || strike <= entries_.front().K_ref) {
        return entries_.front().surface.price(spot, strike,
                                              tau, sigma, rate);
    }

    // Strike at/above highest K_ref: use last entry
    if (strike >= entries_.back().K_ref) {
        return entries_.back().surface.price(spot, strike,
                                             tau, sigma, rate);
    }

    // Find bracketing entries via upper_bound
    auto it = std::upper_bound(
        entries_.begin(), entries_.end(), strike,
        [](double s, const Entry& e) { return s < e.K_ref; });

    // it points to first entry with K_ref > strike
    const auto& hi = *it;
    const auto& lo = *std::prev(it);

    // Exact match on lower bound
    if (strike == lo.K_ref) {
        return lo.surface.price(spot, strike, tau, sigma, rate);
    }

    // Each surface evaluates at the actual strike (EEP segments handle this
    // via strike homogeneity; RawPrice segments use K_ref internally).
    // Interpolate across K_ref surfaces for additional accuracy.
    double w = (strike - lo.K_ref) / (hi.K_ref - lo.K_ref);
    double p_lo = lo.surface.price(spot, strike, tau, sigma, rate);
    double p_hi = hi.surface.price(spot, strike, tau, sigma, rate);

    return (1.0 - w) * p_lo + w * p_hi;
}

double SegmentedMultiKRefSurface::vega(double spot, double strike,
                                        double tau, double sigma,
                                        double rate) const {
    const size_t n = entries_.size();

    if (n == 1 || strike <= entries_.front().K_ref) {
        return entries_.front().surface.vega(spot, strike,
                                             tau, sigma, rate);
    }

    if (strike >= entries_.back().K_ref) {
        return entries_.back().surface.vega(spot, strike,
                                            tau, sigma, rate);
    }

    auto it = std::upper_bound(
        entries_.begin(), entries_.end(), strike,
        [](double s, const Entry& e) { return s < e.K_ref; });

    const auto& hi = *it;
    const auto& lo = *std::prev(it);

    if (strike == lo.K_ref) {
        return lo.surface.vega(spot, strike, tau, sigma, rate);
    }

    double w = (strike - lo.K_ref) / (hi.K_ref - lo.K_ref);
    double v_lo = lo.surface.vega(spot, strike, tau, sigma, rate);
    double v_hi = hi.surface.vega(spot, strike, tau, sigma, rate);

    return (1.0 - w) * v_lo + w * v_hi;
}

double SegmentedMultiKRefSurface::m_min() const noexcept { return m_min_; }
double SegmentedMultiKRefSurface::m_max() const noexcept { return m_max_; }
double SegmentedMultiKRefSurface::tau_min() const noexcept { return tau_min_; }
double SegmentedMultiKRefSurface::tau_max() const noexcept { return tau_max_; }
double SegmentedMultiKRefSurface::sigma_min() const noexcept { return sigma_min_; }
double SegmentedMultiKRefSurface::sigma_max() const noexcept { return sigma_max_; }
double SegmentedMultiKRefSurface::rate_min() const noexcept { return rate_min_; }
double SegmentedMultiKRefSurface::rate_max() const noexcept { return rate_max_; }

}  // namespace mango
