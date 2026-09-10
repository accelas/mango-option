// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"

using namespace mango;

// Regression: call tail coverage must be checked through the fitted surface.
// Bug: Put-only boundary measurements left the call sampling path untested.
TEST(GridCoverageAccuracyTest, RawChebyshevCallTailsMatchEuropeanReference) {
    const auto m = chebyshev_nodes(9, -2.0, 2.0);
    const auto tau = chebyshev_nodes(3, 0.1, 0.5);
    const auto sigma = chebyshev_nodes(3, 0.1, 0.4);
    const auto rates = chebyshev_nodes(2, 0.03, 0.05);
    auto pieces = build_chebyshev_segmented_pieces(
        100.0, OptionType::CALL, 0.0, {}, {0.1, 0.5}, {false},
        m, tau, sigma, rates);
    ASSERT_TRUE(pieces) << pieces.error();
    ASSERT_EQ(pieces->leaves.size(), 1u);
    for (double x : {m.front(), 0.0, m.back()}) {
        for (double vol : sigma) {
            SCOPED_TRACE(testing::Message() << "x=" << x << " sigma=" << vol);
            PricingParams p(OptionSpec{
                .spot = 100.0 * std::exp(x), .strike = 100.0, .maturity = 0.5,
                .rate = 0.05, .option_type = OptionType::CALL}, vol);
            // Cash-free, zero-yield calls at positive rates have an independent
            // European price. The raw leaf has no EEP add-back/floor to mask error.
            const double expected = EuropeanOptionResult(p).value();
            const double actual = 100.0 * pieces->leaves.front().price(
                p.spot, p.strike, p.maturity - 0.1, vol, 0.05);
            EXPECT_NEAR(actual, expected, 0.003);
        }
    }
}
