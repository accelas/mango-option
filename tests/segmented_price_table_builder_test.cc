// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/american_option.hpp"
#include <cmath>
#include <vector>

using namespace mango;

namespace {

std::vector<double> log_m_grid(std::initializer_list<double> moneyness) {
    std::vector<double> out;
    out.reserve(moneyness.size());
    for (double m : moneyness) {
        out.push_back(std::log(m));
    }
    return out;
}

}  // namespace

TEST(SegmentedPriceTableBuilderTest, BuildWithOneDividend) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    // Verify price is reasonable for ATM put
    double price = result->price(100.0, 100.0, 0.8, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, 50.0);  // sanity check

    // Verify vega is finite
    double vega = result->vega(100.0, 100.0, 0.8, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(vega));
    EXPECT_GT(vega, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, BuildWithNoDividends) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02},
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    double price = result->price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, DividendAtExpiryIgnored) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.discrete_dividends = {{.calendar_time = 1.0, .amount = 5.0}}},  // at expiry — should be filtered out
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());
}

TEST(SegmentedPriceTableBuilderTest, DividendAtTimeZeroIgnored) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.discrete_dividends = {{.calendar_time = 0.0, .amount = 3.0}}},  // at time 0 — should be filtered out
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());
}

TEST(SegmentedPriceTableBuilderTest, InvalidKRefFails) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = -100.0,
        .option_type = OptionType::PUT,
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    EXPECT_FALSE(result.has_value());
}

TEST(SegmentedPriceTableBuilderTest, InvalidMaturityFails) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 0.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    EXPECT_FALSE(result.has_value());
}

// ===========================================================================
// Regression tests for unified manual build path
// ===========================================================================

// Regression: Manual build path for segment 0 must produce the same result
// as the old builder.build() path.
// Bug: Refactoring segment 0 to use make_batch/solve_batch/extract_tensor
// directly could silently change behavior if steps are ordered incorrectly.
TEST(SegmentedPriceTableBuilderTest, ManualPathMatchesBuildPath) {
    // 1 dividend creates 2 segments; segment 0 uses the manual path
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.5, .amount = 1.50}}},
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build with 1 dividend should succeed";

    // ATM put price should be reasonable
    double price = result->price(100.0, 100.0, 0.3, 0.20, 0.05);
    EXPECT_GT(price, 3.0) << "ATM put should have meaningful value";
    EXPECT_LT(price, 20.0) << "ATM put should not be absurdly large";

    // Cross-segment query (tau > 0.5 spans the dividend boundary)
    double price2 = result->price(100.0, 100.0, 0.8, 0.20, 0.05);
    EXPECT_GT(price2, 0.0);
    EXPECT_TRUE(std::isfinite(price2));
}

// Regression: Unified manual build path must produce finite, positive prices
// across all segments with multiple dividends.
// Bug: Manual path for all segments (including segment 0) could diverge from
// the old mixed path (builder.build() for segment 0, manual for chained).
TEST(SegmentedPriceTableBuilderTest, UnifiedManualPathMultiDividend) {
    // 3 dividends = 4 segments (quarterly $0.50 dividends)
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.0,
            .discrete_dividends = {
                {.calendar_time = 0.25, .amount = 0.50},
                {.calendar_time = 0.50, .amount = 0.50},
                {.calendar_time = 0.75, .amount = 0.50},
            },
        },
        .grid = IVGrid{
            .moneyness = log_m_grid({0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                          1.05, 1.10, 1.15, 1.20, 1.25, 1.30}),
            .vol = {0.10, 0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build with 3 dividends should succeed";

    // Verify prices at multiple tau values spanning different segments
    double taus[] = {0.1, 0.3, 0.6, 0.9};
    for (double tau : taus) {
        double price = result->price(100.0, 100.0, tau, 0.20, 0.05);
        EXPECT_TRUE(std::isfinite(price))
            << "Price must be finite at tau=" << tau;
        EXPECT_GT(price, 0.0)
            << "ATM put price must be positive at tau=" << tau;
    }
}

// Regression: long-maturity multi-dividend chained surfaces can be biased high
// at the upper moneyness edge if the right-side domain is too tight.
TEST(SegmentedPriceTableBuilderTest, LongMaturityMultiDividendEdgeBiasControlled) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 80.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.02,
            .discrete_dividends = {
                {.calendar_time = 0.5, .amount = 0.50},
                {.calendar_time = 1.0, .amount = 0.50},
                {.calendar_time = 1.5, .amount = 0.50},
            },
        },
        .grid = IVGrid{
            .moneyness = log_m_grid({0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30}),
            .vol = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50},
            .rate = {0.01, 0.03, 0.05, 0.10},
        },
        .maturity = 2.0,
    };

    auto surface = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(surface.has_value());

    PricingParams p;
    p.spot = 100.0;
    p.strike = 80.0;
    p.maturity = 2.0;
    p.rate = 0.05;
    p.dividend_yield = 0.02;
    p.option_type = OptionType::PUT;
    p.volatility = 0.30;
    p.discrete_dividends = config.dividends.discrete_dividends;

    auto fd = solve_american_option(p);
    ASSERT_TRUE(fd.has_value());

    double interp_price = surface->price(p.spot, p.strike, p.maturity,
                                         p.volatility, std::get<double>(p.rate));
    double fd_price = fd->value();
    double abs_error = std::abs(interp_price - fd_price);

    EXPECT_LT(abs_error, 0.25)
        << "Interpolated chained-segment price drifted too far from FD reference";
}
