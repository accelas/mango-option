// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <concepts>
#include <expected>
#include <limits>
#include <tuple>
#include <vector>

#include "mango/option/table/greek_types.hpp"
#include "mango/option/option_spec.hpp"

namespace mango {

struct BracketResult {
    struct Entry { size_t index; double weight; };
    std::array<Entry, 2> entries{};
    size_t count = 0;
};

template <typename S>
concept SplitPolicy = requires(const S& s, double spot, double strike,
                                double tau, double sigma, double rate) {
    { s.bracket(spot, strike, tau, sigma, rate) } -> std::same_as<BracketResult>;
    { s.to_local(size_t{}, spot, strike, tau, sigma, rate) }
        -> std::same_as<std::tuple<double, double, double, double, double>>;
    { s.normalize(size_t{}, strike, double{}) } -> std::same_as<double>;
    { s.denormalize(double{}, spot, strike, tau, sigma, rate) } -> std::same_as<double>;
};

/// Composable surface split. Routes queries to pieces via SplitPolicy,
/// with per-slice remapping and value normalization.
/// A split with a nonidentity linear spot map supplies spot_scale(index, K)
/// so first/second spot derivatives include the corresponding chain factors.
template <typename Inner, SplitPolicy Split>
class SplitSurface {
public:
    static constexpr bool requires_strike_bounds = [] {
        if constexpr (requires { Split::requires_strike_bounds; }) {
            return Split::requires_strike_bounds;
        } else if constexpr (requires { Inner::requires_strike_bounds; }) {
            return Inner::requires_strike_bounds;
        }
        return false;
    }();
    SplitSurface(std::vector<Inner> pieces, Split split)
        : pieces_(std::move(pieces)), split_(std::move(split)) {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        if constexpr (requires { split_.contains_maturity(tau); }) {
            if (!split_.contains_maturity(tau)) return std::numeric_limits<double>::quiet_NaN();
        }
        auto br = split_.bracket(spot, strike, tau, sigma, rate);
        double result = 0.0;
        for (size_t i = 0; i < br.count; ++i) {
            auto [ls, lk, lt, lv, lr] = split_.to_local(
                br.entries[i].index, spot, strike, tau, sigma, rate);
            double raw = pieces_[br.entries[i].index].price(ls, lk, lt, lv, lr);
            double norm = split_.normalize(br.entries[i].index, strike, raw);
            result += br.entries[i].weight * norm;
        }
        return split_.denormalize(result, spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        if constexpr (requires { split_.contains_maturity(tau); }) {
            if (!split_.contains_maturity(tau)) return std::numeric_limits<double>::quiet_NaN();
        }
        auto br = split_.bracket(spot, strike, tau, sigma, rate);
        double result = 0.0;
        for (size_t i = 0; i < br.count; ++i) {
            auto [ls, lk, lt, lv, lr] = split_.to_local(
                br.entries[i].index, spot, strike, tau, sigma, rate);
            double raw = pieces_[br.entries[i].index].vega(ls, lk, lt, lv, lr);
            double norm = split_.normalize(br.entries[i].index, strike, raw);
            result += br.entries[i].weight * norm;
        }
        return split_.denormalize(result, spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] std::expected<double, GreekError>
    greek(Greek g, const PricingParams& params) const {
        if constexpr (requires { split_.contains_maturity(params.maturity); }) {
            if (!split_.contains_maturity(params.maturity)) return std::unexpected(GreekError::OutOfDomain);
        }
        double spot = params.spot, strike = params.strike;
        double tau = params.maturity, sigma = params.volatility;
        double rate = get_zero_rate(params.rate, params.maturity);

        auto br = split_.bracket(spot, strike, tau, sigma, rate);
        double result = 0.0;
        for (size_t i = 0; i < br.count; ++i) {
            auto [ls, lk, lt, lv, lr] = split_.to_local(
                br.entries[i].index, spot, strike, tau, sigma, rate);
            PricingParams local_params(
                OptionSpec{.spot = ls, .strike = lk, .maturity = lt,
                    .rate = lr, .dividend_yield = params.dividend_yield,
                    .option_type = params.option_type},
                lv);
            auto piece_greek = pieces_[br.entries[i].index].greek(g, local_params);
            if (!piece_greek.has_value()) return std::unexpected(piece_greek.error());
            const double chain = g == Greek::Delta
                ? local_spot_scale(br.entries[i].index, strike) : 1.0;
            double norm = split_.normalize(br.entries[i].index, strike, *piece_greek * chain);
            result += br.entries[i].weight * norm;
        }
        return split_.denormalize(result, spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] std::expected<double, GreekError>
    gamma(const PricingParams& params) const {
        if constexpr (requires { split_.contains_maturity(params.maturity); }) {
            if (!split_.contains_maturity(params.maturity)) return std::unexpected(GreekError::OutOfDomain);
        }
        double spot = params.spot, strike = params.strike;
        double tau = params.maturity, sigma = params.volatility;
        double rate = get_zero_rate(params.rate, params.maturity);

        auto br = split_.bracket(spot, strike, tau, sigma, rate);
        double result = 0.0;
        for (size_t i = 0; i < br.count; ++i) {
            auto [ls, lk, lt, lv, lr] = split_.to_local(
                br.entries[i].index, spot, strike, tau, sigma, rate);
            PricingParams local_params(
                OptionSpec{.spot = ls, .strike = lk, .maturity = lt,
                    .rate = lr, .dividend_yield = params.dividend_yield,
                    .option_type = params.option_type},
                lv);
            auto piece_gamma = pieces_[br.entries[i].index].gamma(local_params);
            if (!piece_gamma.has_value()) return std::unexpected(piece_gamma.error());
            const double chain = local_spot_scale(br.entries[i].index, strike);
            double norm = split_.normalize(br.entries[i].index, strike, *piece_gamma * chain * chain);
            result += br.entries[i].weight * norm;
        }
        return split_.denormalize(result, spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] size_t num_pieces() const noexcept { return pieces_.size(); }
    [[nodiscard]] bool contains_maturity(double tau) const noexcept {
        if constexpr (requires { split_.contains_maturity(tau); }) {
            return split_.contains_maturity(tau);
        } else if constexpr (requires { pieces_.front().contains_maturity(tau); }) {
            for (const auto& piece : pieces_) {
                if (!piece.contains_maturity(tau)) return false;
            }
        }
        return true;
    }
    [[nodiscard]] bool contains_strike(double strike) const noexcept {
        if constexpr (requires { split_.contains_strike(strike); }) {
            return split_.contains_strike(strike);
        }
        return true;
    }
    [[nodiscard]] const std::vector<Inner>& pieces() const noexcept { return pieces_; }
    [[nodiscard]] const Split& split() const noexcept { return split_; }

private:
    [[nodiscard]] double local_spot_scale(size_t index, double strike) const noexcept {
        if constexpr (requires { split_.spot_scale(index, strike); }) {
            return split_.spot_scale(index, strike);
        }
        return 1.0;
    }

    std::vector<Inner> pieces_;
    Split split_;
};

}  // namespace mango
