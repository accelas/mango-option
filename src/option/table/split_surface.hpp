// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <concepts>
#include <expected>
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
template <typename Inner, SplitPolicy Split>
class SplitSurface {
public:
    SplitSurface(std::vector<Inner> pieces, Split split)
        : pieces_(std::move(pieces)), split_(std::move(split)) {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
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
            double norm = split_.normalize(br.entries[i].index, strike, *piece_greek);
            result += br.entries[i].weight * norm;
        }
        return split_.denormalize(result, spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] std::expected<double, GreekError>
    gamma(const PricingParams& params) const {
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
            double norm = split_.normalize(br.entries[i].index, strike, *piece_gamma);
            result += br.entries[i].weight * norm;
        }
        return split_.denormalize(result, spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] size_t num_pieces() const noexcept { return pieces_.size(); }
    [[nodiscard]] const std::vector<Inner>& pieces() const noexcept { return pieces_; }
    [[nodiscard]] const Split& split() const noexcept { return split_; }

private:
    std::vector<Inner> pieces_;
    Split split_;
};

}  // namespace mango
