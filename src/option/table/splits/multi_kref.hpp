// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/split_surface.hpp"
#include <tuple>
#include <vector>

namespace mango {

/// Split policy for multi-K_ref surfaces (discrete dividends).
/// Merges KRefBracket + KRefTransform from old architecture.
class MultiKRefSplit {
public:
    explicit MultiKRefSplit(std::vector<double> k_refs)
        : k_refs_(std::move(k_refs)) {}

    [[nodiscard]] BracketResult bracket(
        double /*spot*/, double strike, double /*tau*/,
        double /*sigma*/, double /*rate*/) const noexcept {
        BracketResult br;
        const size_t n = k_refs_.size();
        if (n == 0) return br;
        if (n == 1 || strike <= k_refs_.front()) {
            br.entries[0] = {0, 1.0};
            br.count = 1;
            return br;
        }
        if (strike >= k_refs_.back()) {
            br.entries[0] = {n - 1, 1.0};
            br.count = 1;
            return br;
        }
        size_t hi = 1;
        while (hi < n && k_refs_[hi] < strike) ++hi;
        size_t lo = hi - 1;
        double t = (strike - k_refs_[lo]) / (k_refs_[hi] - k_refs_[lo]);
        br.entries[0] = {lo, 1.0 - t};
        br.entries[1] = {hi, t};
        br.count = 2;
        return br;
    }

    [[nodiscard]] std::tuple<double, double, double, double, double>
    to_local(size_t i, double spot, double /*strike*/,
             double tau, double sigma, double rate) const noexcept {
        return {spot, k_refs_[i], tau, sigma, rate};
    }

    [[nodiscard]] double normalize(size_t i, double /*strike*/,
                                    double raw) const noexcept {
        return raw / k_refs_[i];
    }

    [[nodiscard]] double denormalize(double combined, double /*spot*/, double strike,
                                      double /*tau*/, double /*sigma*/,
                                      double /*rate*/) const noexcept {
        return combined * strike;
    }

    [[nodiscard]] const std::vector<double>& k_refs() const noexcept { return k_refs_; }

private:
    std::vector<double> k_refs_;
};

}  // namespace mango
