// SPDX-License-Identifier: MIT
//
// Piecewise Chebyshev 4D evaluator with C∞ overlap blending in (x, τ).
#pragma once

#include "piecewise_element_builder.hpp"
#include "bump_blend.hpp"
#include "mango/option/table/price_query.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace mango {

class PiecewiseBlendedEvaluator {
public:
    PiecewiseBlendedEvaluator(PiecewiseElementSet elements,
                               OptionType type, double K_ref,
                               double dividend_yield)
        : elems_(std::move(elements)), type_(type),
          K_ref_(K_ref), dividend_yield_(dividend_yield) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        double eep = eval_blended(x, q.tau, q.sigma, q.rate);

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return eep * (q.strike / K_ref_) + eu.value();
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        double eep_vega = vega_blended(x, q.tau, q.sigma, q.rate);

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return (q.strike / K_ref_) * eep_vega + eu.vega();
    }

    const PiecewiseElementSet& element_set() const { return elems_; }

private:
    /// Evaluate single element.
    [[nodiscard]] double eval_element(size_t idx, double x, double tau,
                                       double sigma, double rate) const {
        return elems_.elements[idx].eval({x, tau, sigma, rate});
    }

    /// Evaluate partial derivative w.r.t. sigma (axis 2) for single element.
    [[nodiscard]] double partial_sigma_element(
        size_t idx, double x, double tau, double sigma, double rate) const {
        return elems_.elements[idx].partial(2, {x, tau, sigma, rate});
    }

    /// Find elements for a given (x, tau_band).
    /// Returns {elem_idx, weight} pairs. 1 or 2 entries.
    struct WeightedElement { size_t idx; double weight; };

    [[nodiscard]] std::vector<WeightedElement>
    find_elements(double x, size_t band) const {
        // Check overlap zones for this band
        for (const auto& oz : elems_.x_overlaps) {
            if (elems_.specs[oz.left_idx].tau_band != band) continue;
            if (x >= oz.x_lo && x <= oz.x_hi) {
                double w_right = overlap_weight_right(x, oz.x_lo, oz.x_hi);
                return {{oz.left_idx, 1.0 - w_right},
                        {oz.right_idx, w_right}};
            }
        }

        // Not in any overlap — find the single element
        size_t base = band * 3;
        for (size_t i = 0; i < 3; ++i) {
            size_t idx = base + i;
            const auto& spec = elems_.specs[idx];
            if (x >= spec.x_lo && x <= spec.x_hi) {
                return {{idx, 1.0}};
            }
        }

        // Fallback: closest element
        size_t closest = base;
        double best_dist = 1e99;
        for (size_t i = 0; i < 3; ++i) {
            size_t idx = base + i;
            const auto& spec = elems_.specs[idx];
            double mid = (spec.x_lo + spec.x_hi) / 2.0;
            double dist = std::abs(x - mid);
            if (dist < best_dist) { best_dist = dist; closest = idx; }
        }
        return {{closest, 1.0}};
    }

    /// Evaluate EEP with full (x, τ) blending.
    [[nodiscard]] double eval_blended(double x, double tau,
                                       double sigma, double rate) const {
        auto eval_band = [&](size_t band) -> double {
            auto elems = find_elements(x, band);
            double result = 0.0;
            for (const auto& [idx, w] : elems) {
                result += w * eval_element(idx, x, tau, sigma, rate);
            }
            return result;
        };

        // τ-band routing
        if (tau < elems_.tau_blend_lo) {
            return eval_band(0);  // short only
        }
        if (tau > elems_.tau_blend_hi) {
            return eval_band(1);  // long only
        }

        // Blend between τ-bands
        double w_long = overlap_weight_right(
            tau, elems_.tau_blend_lo, elems_.tau_blend_hi);
        return (1.0 - w_long) * eval_band(0) + w_long * eval_band(1);
    }

    /// Evaluate EEP vega with full (x, τ) blending.
    [[nodiscard]] double vega_blended(double x, double tau,
                                       double sigma, double rate) const {
        auto vega_band = [&](size_t band) -> double {
            auto elems = find_elements(x, band);
            double result = 0.0;
            for (const auto& [idx, w] : elems) {
                result += w * partial_sigma_element(idx, x, tau, sigma, rate);
            }
            return result;
        };

        if (tau < elems_.tau_blend_lo) return vega_band(0);
        if (tau > elems_.tau_blend_hi) return vega_band(1);

        double w_long = overlap_weight_right(
            tau, elems_.tau_blend_lo, elems_.tau_blend_hi);
        return (1.0 - w_long) * vega_band(0) + w_long * vega_band(1);
    }

    PiecewiseElementSet elems_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

}  // namespace mango
