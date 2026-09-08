// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

namespace mango {
namespace {

struct PriceProbe {
    double spot;
    double tau;
    double sigma;
};

// Declared before changing defaults: tails, exercise transitions, maturity
// strata, both sides of the backward event at .15, and off-node volatility.
std::vector<PriceProbe> required_probes() {
    std::vector<PriceProbe> probes;
    for (double spot : {50.0, 80.0, 95.0, 100.0, 105.0, 120.0, 200.0})
        for (double tau : {0.01, 0.075, 0.149, 0.151, 0.25})
            for (double sigma : {0.05, 0.10, 0.15})
                probes.push_back({spot, tau, sigma});
    for (double spot : {99.0, 99.5, 100.5, 101.0})
        for (double tau : {0.01, 0.151})
            for (double sigma : {0.05, 0.073, 0.127})
                probes.push_back({spot, tau, sigma});
    return probes;
}

std::optional<double> converged_reference(
    const PriceProbe& query, double rate, OptionType type) {
    PricingParams params(
        OptionSpec{.spot = query.spot, .strike = 100.0,
                   .maturity = query.tau, .rate = rate, .option_type = type},
        query.sigma);
    // Independent fixed-expiry contract: T0=.25 and d=.10 imply event_tau=.15.
    // A query after that calendar event has no future dividend.
    if (query.tau > 0.15) params.discrete_dividends = {{query.tau - 0.15, 1.0}};

    std::array<double, 2> prices{};
    const std::array profiles{GridAccuracyProfile::High, GridAccuracyProfile::Ultra};
    for (size_t i = 0; i < profiles.size(); ++i) {
        auto solver = AmericanOptionSolver::create(
            params, PDEGridSpec{make_grid_accuracy(profiles[i])});
        if (!solver.has_value()) {
            ADD_FAILURE() << "direct reference creation failed";
            return std::nullopt;
        }
        auto result = solver->solve();
        if (!result.has_value()) {
            ADD_FAILURE() << "direct reference solve failed";
            return std::nullopt;
        }
        prices[i] = result->value();
    }
    EXPECT_NEAR(prices[0], prices[1], 0.001)
        << "reference must converge below the one-cent fit budget";
    return prices[1];
}

class ManualChebyshevAccuracy : public testing::TestWithParam<OptionType> {};

// Regression #486: the old 33-moneyness-node default and broad parameter
// headroom missed the 0.01 quote budget even after raw time labels were fixed.
TEST_P(ManualChebyshevAccuracy, DefaultsMeetDeclaredPriceCohort) {
    const OptionType type = GetParam();
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = type,
        .discrete_dividends = {{0.1, 1.0}}, .maturity = 0.25,
        .kref_config = {.K_refs = {100.0}},
        .strike_bounds = StrikeBounds{100.0, 100.0}};
    IVGrid domain{
        .moneyness = {std::log(0.5), 0.0, std::log(2.0)},
        .vol = {0.05, 0.15}, .rate = {0.03, 0.05}};
    auto table = build_chebyshev_segmented_manual(config, domain);
    ASSERT_TRUE(table.has_value());

    const auto probes = required_probes();
    double max_error = 0.0;
    double squared_error = 0.0;
    for (const auto& query : probes) {
        SCOPED_TRACE(testing::Message() << "S=" << query.spot
            << " tau=" << query.tau << " sigma=" << query.sigma);
        ASSERT_TRUE(table->contains_maturity(query.tau));
        const auto reference = converged_reference(query, 0.05, type);
        ASSERT_TRUE(reference.has_value());
        const double price = table->price(
            query.spot, 100.0, query.tau, query.sigma, 0.05);
        const double error = std::abs(price - *reference);
        EXPECT_NEAR(price, *reference, 0.01);
        max_error = std::max(max_error, error);
        squared_error += error * error;
    }
    std::cout << "manual Chebyshev type=" << static_cast<int>(type)
              << " queries=" << probes.size() << " max=" << max_error
              << " rms=" << std::sqrt(squared_error / probes.size()) << '\n';

    // Cardinal evaluations expose the actual PDE rows, independently of
    // off-node fitting. Use K=K_ref so strike blending cannot hide raw errors.
    const auto& segmented = table->inner().pieces().front();
    for (size_t j = 0; j < segmented.num_pieces(); ++j) {
        const auto& interp = segmented.pieces()[j].interpolant();
        const auto bounds = interp.domain();
        const auto shape = interp.num_pts();
        const auto m = chebyshev_nodes(shape[0], bounds.lo[0], bounds.hi[0]);
        const auto tau = chebyshev_nodes(shape[1], bounds.lo[1], bounds.hi[1]);
        const auto sigma = chebyshev_nodes(shape[2], bounds.lo[2], bounds.hi[2]);
        const auto rate = chebyshev_nodes(shape[3], bounds.lo[3], bounds.hi[3]);
        for (size_t mi : {shape[0] / 2 - 1, shape[0] / 2, shape[0] / 2 + 1}) {
            for (size_t ti : {size_t{0}, shape[1] / 2, shape[1] - 1}) {
                const PriceProbe query{
                    100.0 * std::exp(m[mi]),
                    tau[ti] + segmented.split().tau_start()[j],
                    sigma[shape[2] / 2]};
                const double r = rate[shape[3] / 2];
                SCOPED_TRACE(testing::Message() << "cardinal segment=" << j
                    << " S=" << query.spot << " tau=" << query.tau);
                if (query.tau == 0.0) {
                    const double payoff = type == OptionType::PUT
                        ? std::max(100.0 - query.spot, 0.0)
                        : std::max(query.spot - 100.0, 0.0);
                    EXPECT_NEAR(segmented.price(query.spot, 100.0, 0.0, query.sigma, r),
                                payoff, 1e-10);
                    continue;
                }
                const auto reference = converged_reference(query, r, type);
                ASSERT_TRUE(reference.has_value());
                EXPECT_NEAR(table->price(query.spot, 100.0, query.tau, query.sigma, r),
                            *reference, 0.001);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(BothOptionTypes, ManualChebyshevAccuracy,
    testing::Values(OptionType::PUT, OptionType::CALL));

}  // namespace
}  // namespace mango
