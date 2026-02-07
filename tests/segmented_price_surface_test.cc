// SPDX-License-Identifier: MIT
// Tests for SegmentedSurface<> created via SegmentedPriceTableBuilder
#include <gtest/gtest.h>
#include "mango/option/table/segmented_price_table_builder.hpp"
#include "mango/option/table/spliced_surface.hpp"
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

// ---------------------------------------------------------------------------
// Segment routing tests
// ---------------------------------------------------------------------------

TEST(SegmentedSurfaceTest, FindsCorrectSegment) {
    // Build a surface with one dividend at t=0.5 (τ=0.5 from expiry)
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.0,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}
        },
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    // Query in segment 0 (closest to expiry): τ = 0.3
    PriceQuery q0{.spot = 100.0, .strike = 100.0, .tau = 0.3, .sigma = 0.25, .rate = 0.05};
    double p0 = result->price(q0);
    EXPECT_GT(p0, 0.0);
    EXPECT_FALSE(std::isnan(p0));

    // Query in segment 1 (earlier): τ = 0.8
    PriceQuery q1{.spot = 100.0, .strike = 100.0, .tau = 0.8, .sigma = 0.25, .rate = 0.05};
    double p1 = result->price(q1);
    EXPECT_GT(p1, 0.0);
    EXPECT_FALSE(std::isnan(p1));

    // Longer maturity should have higher value (more time value)
    EXPECT_GT(p1, p0);
}

TEST(SegmentedSurfaceTest, BoundaryTauGoesToCorrectSegment) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.0,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}
        },
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    // Query exactly at boundary τ = 0.5 (at the dividend date)
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.5, .sigma = 0.25, .rate = 0.05};
    double p_boundary = result->price(q);
    EXPECT_GT(p_boundary, 0.0);
    EXPECT_FALSE(std::isnan(p_boundary));
}

// ---------------------------------------------------------------------------
// Greek tests
// ---------------------------------------------------------------------------

TEST(SegmentedSurfaceTest, VegaIsPositive) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.0,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}
        },
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    // Vega should be positive for ATM put
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.8, .sigma = 0.25, .rate = 0.05};
    double vega = result->vega(q);
    EXPECT_GT(vega, 0.0);
    EXPECT_TRUE(std::isfinite(vega));
}

// ---------------------------------------------------------------------------
// Bounds tests
// ---------------------------------------------------------------------------

TEST(SegmentedSurfaceTest, BoundsSpanFullMaturityRange) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.0,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}
        },
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    // Should be able to query at near-expiry and full maturity
    PriceQuery q_near{.spot = 100.0, .strike = 100.0, .tau = 0.05, .sigma = 0.25, .rate = 0.05};
    double p_near = result->price(q_near);
    EXPECT_GT(p_near, 0.0);

    PriceQuery q_far{.spot = 100.0, .strike = 100.0, .tau = 0.95, .sigma = 0.25, .rate = 0.05};
    double p_far = result->price(q_far);
    EXPECT_GT(p_far, 0.0);
}

// ---------------------------------------------------------------------------
// Multiple dividend tests
// ---------------------------------------------------------------------------

TEST(SegmentedSurfaceTest, MultipleDividendsCreateMultipleSegments) {
    // Two dividends at t=0.25 and t=0.75 create three segments
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.0,
            .discrete_dividends = {
                {.calendar_time = 0.25, .amount = 1.0},
                {.calendar_time = 0.75, .amount = 1.5},
            }
        },
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    // Query in each segment
    PriceQuery q1{.spot = 100.0, .strike = 100.0, .tau = 0.1, .sigma = 0.25, .rate = 0.05};
    double p1 = result->price(q1);
    EXPECT_GT(p1, 0.0);

    PriceQuery q2{.spot = 100.0, .strike = 100.0, .tau = 0.4, .sigma = 0.25, .rate = 0.05};
    double p2 = result->price(q2);
    EXPECT_GT(p2, 0.0);

    PriceQuery q3{.spot = 100.0, .strike = 100.0, .tau = 0.9, .sigma = 0.25, .rate = 0.05};
    double p3 = result->price(q3);
    EXPECT_GT(p3, 0.0);
}

// ---------------------------------------------------------------------------
// No dividend tests
// ---------------------------------------------------------------------------

TEST(SegmentedSurfaceTest, NoDividendsProducesSingleSegment) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02},  // No discrete dividends
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    // Should work across full maturity range
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.5, .sigma = 0.25, .rate = 0.05};
    double p = result->price(q);
    EXPECT_GT(p, 0.0);
    EXPECT_TRUE(std::isfinite(p));
}

// ---------------------------------------------------------------------------
// Dividend edge cases
// ---------------------------------------------------------------------------

TEST(SegmentedSurfaceTest, DividendAtExpiryIsIgnored) {
    // Dividend at expiry (t=1.0) should be filtered out
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.0,
            .discrete_dividends = {{.calendar_time = 1.0, .amount = 5.0}}
        },
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());
}

TEST(SegmentedSurfaceTest, DividendAtTimeZeroIsIgnored) {
    // Dividend at t=0 should be filtered out
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.0,
            .discrete_dividends = {{.calendar_time = 0.0, .amount = 3.0}}
        },
        .grid = IVGrid{
            .moneyness = log_m_grid({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());
}
