// SPDX-License-Identifier: MIT
#include "src/option/table/segmented_price_surface.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace mango {

std::expected<SegmentedPriceSurface, ValidationError>
SegmentedPriceSurface::create(Config config) {
    if (config.segments.empty()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 0});
    }

    if (config.K_ref <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidStrike, config.K_ref, 0});
    }

    if (config.T <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidMaturity, config.T, 0});
    }

    // Verify segments are ordered by tau_start ascending
    for (size_t i = 1; i < config.segments.size(); ++i) {
        if (config.segments[i].tau_start <= config.segments[i - 1].tau_start) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::UnsortedGrid, config.segments[i].tau_start, i});
        }
    }

    SegmentedPriceSurface result;
    result.K_ref_ = config.K_ref;
    result.T_ = config.T;
    result.segments_ = std::move(config.segments);

    result.dividends_.reserve(config.dividends.size());
    for (auto& [t, amount] : config.dividends) {
        result.dividends_.push_back(DividendEntry{t, amount});
    }

    // Sort dividends by calendar time
    std::sort(result.dividends_.begin(), result.dividends_.end(),
              [](const DividendEntry& a, const DividendEntry& b) {
                  return a.calendar_time < b.calendar_time;
              });

    return result;
}

const SegmentedPriceSurface::Segment&
SegmentedPriceSurface::find_segment(double tau) const {
    assert(!segments_.empty());

    // Segments are ordered by tau_start ascending.
    // Segment 0: [tau_start, tau_end] (inclusive on both ends for the first)
    // Segment i>0: (tau_start, tau_end] (exclusive start, inclusive end)
    //
    // Search from back (highest tau) to front.
    for (size_t i = segments_.size(); i > 0; --i) {
        const auto& seg = segments_[i - 1];
        if (i - 1 == 0) {
            // First segment: inclusive on both ends
            if (tau >= seg.tau_start && tau <= seg.tau_end) {
                return seg;
            }
        } else {
            // Later segments: exclusive start, inclusive end
            if (tau > seg.tau_start && tau <= seg.tau_end) {
                return seg;
            }
        }
    }

    // Fallback: clamp to nearest segment
    if (tau <= segments_.front().tau_start) {
        return segments_.front();
    }
    return segments_.back();
}

double SegmentedPriceSurface::compute_spot_adjustment(
    double spot, double t_query, double t_boundary) const {
    double adjustment = 0.0;
    for (const auto& div : dividends_) {
        // Subtract dividends where t_query < t_div <= t_boundary
        if (div.calendar_time > t_query && div.calendar_time <= t_boundary) {
            adjustment += div.amount;
        }
    }
    return spot - adjustment;
}

double SegmentedPriceSurface::price(double spot, double strike,
                                     double tau, double sigma, double rate) const {
    // 1. Convert to calendar time
    double t_query = T_ - tau;

    // 2. Find segment
    const auto& seg = find_segment(tau);

    // 3. Compute spot adjustment
    double t_boundary = T_ - seg.tau_start;
    double S_adj = compute_spot_adjustment(spot, t_query, t_boundary);

    // 4. Clamp S_adj
    if (S_adj <= 0.0) {
        S_adj = 1e-8;
    }

    // 5. Convert to local segment time and clamp to segment grid bounds
    double tau_local = std::clamp(tau - seg.tau_start,
                                  seg.surface.tau_min(),
                                  seg.surface.tau_max());

    // 6. Delegate to segment surface (always use K_ref_ as strike;
    //    all segments are built for this K_ref, and RawPrice segments
    //    require strike == K_ref)
    return seg.surface.price(S_adj, K_ref_, tau_local, sigma, rate);
}

double SegmentedPriceSurface::vega(double spot, double strike,
                                    double tau, double sigma, double rate) const {
    const auto& seg = find_segment(tau);

    // EEP segments have analytic vega; RawPrice segments need FD
    if (seg.surface.metadata().content == SurfaceContent::EarlyExercisePremium) {
        // Compute spot adjustment for analytic vega delegation
        double t_query = T_ - tau;
        double t_boundary = T_ - seg.tau_start;
        double S_adj = compute_spot_adjustment(spot, t_query, t_boundary);
        if (S_adj <= 0.0) {
            S_adj = 1e-8;
        }
        double tau_local = std::clamp(tau - seg.tau_start,
                                      seg.surface.tau_min(),
                                      seg.surface.tau_max());
        return seg.surface.vega(S_adj, K_ref_, tau_local, sigma, rate);
    }

    // RawPrice: finite difference vega using this->price() (includes spot adjustment)
    double eps = std::max(1e-4, 1e-4 * sigma);
    double p_up = price(spot, strike, tau, sigma + eps, rate);
    double p_dn = price(spot, strike, tau, sigma - eps, rate);
    return (p_up - p_dn) / (2.0 * eps);
}

double SegmentedPriceSurface::m_min() const noexcept {
    return segments_.front().surface.m_min();
}

double SegmentedPriceSurface::m_max() const noexcept {
    return segments_.front().surface.m_max();
}

double SegmentedPriceSurface::tau_min() const noexcept {
    return segments_.front().tau_start + segments_.front().surface.tau_min();
}

double SegmentedPriceSurface::tau_max() const noexcept {
    return segments_.back().tau_end;
}

double SegmentedPriceSurface::sigma_min() const noexcept {
    return segments_.front().surface.sigma_min();
}

double SegmentedPriceSurface::sigma_max() const noexcept {
    return segments_.front().surface.sigma_max();
}

double SegmentedPriceSurface::rate_min() const noexcept {
    return segments_.front().surface.rate_min();
}

double SegmentedPriceSurface::rate_max() const noexcept {
    return segments_.front().surface.rate_max();
}

}  // namespace mango
