// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <concepts>
#include <tuple>
#include <vector>

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

    [[nodiscard]] size_t num_pieces() const noexcept { return pieces_.size(); }

private:
    std::vector<Inner> pieces_;
    Split split_;
};

}  // namespace mango
