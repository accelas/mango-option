// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "mango/option/option_spec.hpp"
#include "mango/option/table/price_table_metadata.hpp"

namespace mango {

struct PriceQuery {
    double spot;
    double strike;
    double tau;
    double sigma;
    double rate;
};

struct SliceWeight {
    size_t index;
    double weight;
};

struct Bracket {
    std::array<SliceWeight, 4> items{};
    size_t size = 0;

    [[nodiscard]] constexpr std::span<const SliceWeight> span() const noexcept {
        return {items.data(), size};
    }
};

struct Sample {
    double value;
    double weight;
};

template <typename S>
concept PriceSurface =
    requires(const S& s, const PriceQuery& q) {
        { s.price(q) } -> std::same_as<double>;
        { s.vega(q) } -> std::same_as<double>;
    } ||
    requires(const S& s, double spot, double strike, double tau, double sigma, double rate) {
        { s.price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
        { s.vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    };

template <typename Split>
concept SplitStrategy = requires(const Split& s, const PriceQuery& q) {
    { s.key(q) } -> std::convertible_to<double>;
    { s.bracket(s.key(q)) } -> std::same_as<Bracket>;
    { s.num_slices() } -> std::same_as<size_t>;
};

template <typename Xform>
concept SliceTransform = requires(const Xform& x, size_t i, const PriceQuery& q, double raw) {
    { x.to_local(i, q) } -> std::same_as<PriceQuery>;
    { x.normalize_value(i, q, raw) } -> std::same_as<double>;
};

template <typename Combiner>
concept CombineStrategy = requires(const Combiner& c, std::span<const Sample> samples,
                                   const PriceQuery& q) {
    { c.combine(samples, q) } -> std::same_as<double>;
};

namespace detail {

template <typename S>
[[nodiscard]] inline double call_price(const S& s, const PriceQuery& q) {
    if constexpr (requires { s.price(q); }) {
        return s.price(q);
    } else {
        return s.price(q.spot, q.strike, q.tau, q.sigma, q.rate);
    }
}

template <typename S>
[[nodiscard]] inline double call_vega(const S& s, const PriceQuery& q) {
    if constexpr (requires { s.vega(q); }) {
        return s.vega(q);
    } else {
        return s.vega(q.spot, q.strike, q.tau, q.sigma, q.rate);
    }
}

}  // namespace detail

template <PriceSurface Inner,
          SplitStrategy Split,
          SliceTransform Xform,
          CombineStrategy Combiner>
class SplicedSurface {
public:
    using Slice = Inner;

    SplicedSurface(std::vector<Slice> slices, Split split, Xform xform, Combiner comb)
        : slices_(std::move(slices))
        , split_(std::move(split))
        , xform_(std::move(xform))
        , comb_(std::move(comb))
    {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        Bracket br = split_.bracket(split_.key(q));
        std::array<Sample, 4> samples{};
        for (size_t i = 0; i < br.size; ++i) {
            const auto& sw = br.items[i];
            const auto& slice = slices_[sw.index];
            PriceQuery local = xform_.to_local(sw.index, q);
            double raw = detail::call_price(slice, local);
            double norm = xform_.normalize_value(sw.index, q, raw);
            samples[i] = Sample{norm, sw.weight};
        }
        double combined = comb_.combine({samples.data(), br.size}, q);
        // Call denormalize if the transform supports it (e.g., KRefTransform).
        if constexpr (requires { xform_.denormalize(combined, q); }) {
            return xform_.denormalize(combined, q);
        }
        return combined;
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        Bracket br = split_.bracket(split_.key(q));
        std::array<Sample, 4> samples{};
        for (size_t i = 0; i < br.size; ++i) {
            const auto& sw = br.items[i];
            const auto& slice = slices_[sw.index];
            PriceQuery local = xform_.to_local(sw.index, q);
            double raw = detail::call_vega(slice, local);
            double norm = xform_.normalize_value(sw.index, q, raw);
            samples[i] = Sample{norm, sw.weight};
        }
        double combined = comb_.combine({samples.data(), br.size}, q);
        // Call denormalize if the transform supports it (e.g., KRefTransform).
        if constexpr (requires { xform_.denormalize(combined, q); }) {
            return xform_.denormalize(combined, q);
        }
        return combined;
    }

    [[nodiscard]] size_t num_slices() const noexcept { return slices_.size(); }

private:
    std::vector<Slice> slices_;
    Split split_;
    Xform xform_;
    Combiner comb_;
};

class SegmentLookup {
public:
    SegmentLookup(std::vector<double> tau_start, std::vector<double> tau_end)
        : tau_start_(std::move(tau_start))
        , tau_end_(std::move(tau_end))
    {}

    [[nodiscard]] double key(const PriceQuery& q) const noexcept { return q.tau; }
    [[nodiscard]] size_t num_slices() const noexcept { return tau_start_.size(); }

    [[nodiscard]] Bracket bracket(double tau) const noexcept {
        Bracket br;
        const size_t n = tau_start_.size();
        if (n == 0) {
            return br;  // Empty: return bracket with size=0
        }

        size_t idx = 0;

        for (size_t i = n; i > 0; --i) {
            const size_t j = i - 1;
            if (j == 0) {
                if (tau >= tau_start_[j] && tau <= tau_end_[j]) {
                    idx = j;
                    break;
                }
            } else {
                if (tau > tau_start_[j] && tau <= tau_end_[j]) {
                    idx = j;
                    break;
                }
            }
        }

        if (tau <= tau_start_.front()) {
            idx = 0;
        } else if (tau >= tau_end_.back()) {
            idx = n - 1;
        }

        br.items[0] = SliceWeight{idx, 1.0};
        br.size = 1;
        return br;
    }

private:
    std::vector<double> tau_start_;
    std::vector<double> tau_end_;
};

class LinearBracket {
public:
    explicit LinearBracket(std::vector<double> grid)
        : grid_(std::move(grid))
    {}

    [[nodiscard]] double key(const PriceQuery& q) const noexcept { return q.tau; }
    [[nodiscard]] size_t num_slices() const noexcept { return grid_.size(); }

    [[nodiscard]] Bracket bracket(double tau) const noexcept {
        Bracket br;
        const size_t n = grid_.size();
        if (n == 0) {
            return br;
        }
        if (tau <= grid_.front()) {
            br.items[0] = SliceWeight{0, 1.0};
            br.size = 1;
            return br;
        }
        if (tau >= grid_.back()) {
            br.items[0] = SliceWeight{n - 1, 1.0};
            br.size = 1;
            return br;
        }

        size_t hi = 1;
        while (hi < n && grid_[hi] < tau) {
            ++hi;
        }
        size_t lo = hi - 1;

        double tau_lo = grid_[lo];
        double tau_hi = grid_[hi];
        double t = (tau - tau_lo) / (tau_hi - tau_lo);

        br.items[0] = SliceWeight{lo, 1.0 - t};
        br.items[1] = SliceWeight{hi, t};
        br.size = 2;
        return br;
    }

private:
    std::vector<double> grid_;
};

/// Split strategy for K_ref bracket interpolation.
/// Finds two K_refs bracketing the query strike and computes linear weights.
class KRefBracket {
public:
    explicit KRefBracket(std::vector<double> k_refs)
        : k_refs_(std::move(k_refs))
    {}

    [[nodiscard]] double key(const PriceQuery& q) const noexcept { return q.strike; }
    [[nodiscard]] size_t num_slices() const noexcept { return k_refs_.size(); }

    [[nodiscard]] Bracket bracket(double strike) const noexcept {
        Bracket br;
        const size_t n = k_refs_.size();
        if (n == 0) {
            return br;
        }
        if (n == 1 || strike <= k_refs_.front()) {
            br.items[0] = SliceWeight{0, 1.0};
            br.size = 1;
            return br;
        }
        if (strike >= k_refs_.back()) {
            br.items[0] = SliceWeight{n - 1, 1.0};
            br.size = 1;
            return br;
        }

        size_t hi = 1;
        while (hi < n && k_refs_[hi] < strike) {
            ++hi;
        }
        size_t lo = hi - 1;

        double k_lo = k_refs_[lo];
        double k_hi = k_refs_[hi];
        double t = (strike - k_lo) / (k_hi - k_lo);

        br.items[0] = SliceWeight{lo, 1.0 - t};
        br.items[1] = SliceWeight{hi, t};
        br.size = 2;
        return br;
    }

    [[nodiscard]] const std::vector<double>& k_refs() const noexcept { return k_refs_; }

private:
    std::vector<double> k_refs_;
};

struct WeightedSum {
    [[nodiscard]] double combine(std::span<const Sample> samples,
                                 const PriceQuery&) const noexcept {
        double v = 0.0;
        for (const auto& s : samples) {
            v += s.weight * s.value;
        }
        return v;
    }
};

struct IdentityTransform {
    [[nodiscard]] PriceQuery to_local(size_t, const PriceQuery& q) const noexcept { return q; }
    [[nodiscard]] double normalize_value(size_t, const PriceQuery&, double raw) const noexcept {
        return raw;
    }
};

struct SegmentedTransform {
    std::vector<double> tau_start;
    std::vector<double> tau_min;
    std::vector<double> tau_max;
    std::vector<SurfaceContent> content;
    std::vector<Dividend> dividends;
    double K_ref = 0.0;
    double T = 0.0;

    [[nodiscard]] PriceQuery to_local(size_t i, const PriceQuery& q) const {
        PriceQuery out = q;

        // Convert to local segment time and clamp.
        out.tau = std::clamp(q.tau - tau_start[i], tau_min[i], tau_max[i]);

        // Spot adjustment for EEP segments.
        if (content[i] == SurfaceContent::EarlyExercisePremium) {
            double t_query = T - q.tau;
            double t_boundary = T - tau_start[i];
            out.spot = compute_spot_adjustment(q.spot, t_query, t_boundary);
        }

        // RawPrice segments are only valid at K_ref.
        if (content[i] == SurfaceContent::RawPrice) {
            out.strike = K_ref;
        }

        if (out.spot <= 0.0) {
            out.spot = 1e-8;
        }

        return out;
    }

    [[nodiscard]] double normalize_value(size_t i, const PriceQuery&, double raw) const noexcept {
        if (content[i] == SurfaceContent::RawPrice) {
            return raw * K_ref;
        }
        return raw;
    }

private:
    [[nodiscard]] double compute_spot_adjustment(double spot, double t_query,
                                                 double t_boundary) const {
        double adjustment = 0.0;
        for (const auto& div : dividends) {
            if (div.calendar_time > t_query && div.calendar_time <= t_boundary) {
                adjustment += div.amount;
            }
        }
        return spot - adjustment;
    }
};

struct KRefTransform {
    std::vector<double> k_refs;

    [[nodiscard]] PriceQuery to_local(size_t, const PriceQuery& q) const noexcept { return q; }

    [[nodiscard]] double normalize_value(size_t i, const PriceQuery&, double raw) const noexcept {
        return raw / k_refs[i];
    }

    [[nodiscard]] double denormalize(double combined, const PriceQuery& q) const noexcept {
        return combined * q.strike;
    }
};

}  // namespace mango
