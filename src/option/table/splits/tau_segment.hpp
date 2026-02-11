// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/split_surface.hpp"
#include <algorithm>
#include <tuple>
#include <vector>

namespace mango {

/// Split policy for tau-segmented surfaces (discrete dividends).
/// Merges SegmentLookup + SegmentedTransform from old architecture.
class TauSegmentSplit {
public:
    TauSegmentSplit(std::vector<double> tau_start, std::vector<double> tau_end,
                    std::vector<double> tau_min, std::vector<double> tau_max,
                    double K_ref)
        : tau_start_(std::move(tau_start))
        , tau_end_(std::move(tau_end))
        , tau_min_(std::move(tau_min))
        , tau_max_(std::move(tau_max))
        , K_ref_(K_ref)
    {}

    [[nodiscard]] BracketResult bracket(
        double /*spot*/, double /*strike*/, double tau,
        double /*sigma*/, double /*rate*/) const noexcept {
        BracketResult br;
        const size_t n = tau_start_.size();
        if (n == 0) return br;

        size_t idx = 0;
        for (size_t i = n; i > 0; --i) {
            const size_t j = i - 1;
            if (j == 0) {
                if (tau >= tau_start_[j] && tau <= tau_end_[j]) { idx = j; break; }
            } else {
                if (tau > tau_start_[j] && tau <= tau_end_[j]) { idx = j; break; }
            }
        }
        if (tau <= tau_start_.front()) idx = 0;
        else if (tau >= tau_end_.back()) idx = n - 1;

        br.entries[0] = {idx, 1.0};
        br.count = 1;
        return br;
    }

    [[nodiscard]] std::tuple<double, double, double, double, double>
    to_local(size_t i, double spot, double /*strike*/,
             double tau, double sigma, double rate) const noexcept {
        double local_tau = std::clamp(tau - tau_start_[i], tau_min_[i], tau_max_[i]);
        double local_spot = spot > 0.0 ? spot : 1e-8;
        return {local_spot, K_ref_, local_tau, sigma, rate};
    }

    [[nodiscard]] double normalize(size_t /*i*/, double /*strike*/,
                                    double raw) const noexcept {
        return raw * K_ref_;
    }

    [[nodiscard]] double denormalize(double combined, double /*spot*/, double /*strike*/,
                                      double /*tau*/, double /*sigma*/,
                                      double /*rate*/) const noexcept {
        return combined;
    }

private:
    std::vector<double> tau_start_, tau_end_, tau_min_, tau_max_;
    double K_ref_;
};

}  // namespace mango
