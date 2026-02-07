// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/segmented_price_table_builder.hpp"
#include <cmath>

using namespace mango;

TEST(SegmentedPriceTableBuilderTest, BuildWithOneDividend) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},
        .grid = IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    // Verify price is reasonable for ATM put
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.8, .sigma = 0.20, .rate = 0.05};
    double price = result->price(q);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, 50.0);  // sanity check

    // Verify vega is finite
    double vega = result->vega(q);
    EXPECT_TRUE(std::isfinite(vega));
    EXPECT_GT(vega, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, BuildWithNoDividends) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02},
        .grid = IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.5, .sigma = 0.20, .rate = 0.05};
    double price = result->price(q);
    EXPECT_GT(price, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, DividendAtExpiryIgnored) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.discrete_dividends = {{.calendar_time = 1.0, .amount = 5.0}}},  // at expiry — should be filtered out
        .grid = IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
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
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
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
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
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
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
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
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build with 1 dividend should succeed";

    // ATM put price should be reasonable
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.3, .sigma = 0.20, .rate = 0.05};
    double price = result->price(q);
    EXPECT_GT(price, 3.0) << "ATM put should have meaningful value";
    EXPECT_LT(price, 20.0) << "ATM put should not be absurdly large";

    // Cross-segment query (tau > 0.5 spans the dividend boundary)
    PriceQuery q2{.spot = 100.0, .strike = 100.0, .tau = 0.8, .sigma = 0.20, .rate = 0.05};
    double price2 = result->price(q2);
    EXPECT_GT(price2, 0.0);
    EXPECT_TRUE(std::isfinite(price2));
}

// ===========================================================================
// Numerical EEP tests
// ===========================================================================

TEST(SegmentedPriceTableBuilderTest, NumericalEEPWithOneDividend) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},
        .grid = IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
        .use_numerical_eep = true,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "NumericalEEP build should succeed";

    // Query in the chained segment time range (tau > T - t_div = 0.5)
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.8, .sigma = 0.20, .rate = 0.05};
    double price = result->price(q);
    EXPECT_GT(price, 0.0) << "ATM put price must be positive in chained segment";
    EXPECT_LT(price, 50.0) << "ATM put price should be reasonable";
    EXPECT_TRUE(std::isfinite(price));

    // Verify vega is finite and positive
    double vega = result->vega(q);
    EXPECT_TRUE(std::isfinite(vega)) << "Vega must be finite";
    EXPECT_GT(vega, 0.0) << "Vega should be positive for ATM put";
}

TEST(SegmentedPriceTableBuilderTest, NumericalEEPWithMultipleDividends) {
    // 2 dividends = 3 segments; chained segments 1 and 2 use numerical EEP
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.0,
            .discrete_dividends = {
                {.calendar_time = 0.25, .amount = 1.50},
                {.calendar_time = 0.75, .amount = 1.50},
            },
        },
        .grid = IVGrid{
            .moneyness = {0.80, 0.90, 1.00, 1.10, 1.20},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
        .use_numerical_eep = true,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "NumericalEEP build with 2 dividends should succeed";

    // Verify prices at multiple tau values spanning different segments
    double taus[] = {0.1, 0.4, 0.9};
    for (double tau : taus) {
        PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = tau, .sigma = 0.20, .rate = 0.05};
        double price = result->price(q);
        EXPECT_TRUE(std::isfinite(price))
            << "Price must be finite at tau=" << tau;
        EXPECT_GT(price, 0.0)
            << "ATM put price must be positive at tau=" << tau;
    }
}

TEST(SegmentedPriceTableBuilderTest, NumericalEEPNoDividendsFallsBack) {
    // No dividends: use_numerical_eep flag should have no effect (single segment)
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02},
        .grid = IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
        .use_numerical_eep = true,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.5, .sigma = 0.20, .rate = 0.05};
    double price = result->price(q);
    EXPECT_GT(price, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, NumericalEEPGreeksConsistency) {
    // Build a surface with numerical EEP and one dividend
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},
        .grid = IVGrid{
            .moneyness = {0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
        .use_numerical_eep = true,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "NumericalEEP build should succeed";

    // Query point in the chained segment time range (tau > 0.5)
    const double spot = 100.0;
    const double strike = 100.0;
    const double tau = 0.8;
    const double sigma = 0.20;
    const double rate = 0.05;

    PriceQuery q{.spot = spot, .strike = strike, .tau = tau, .sigma = sigma, .rate = rate};
    double price = result->price(q);
    ASSERT_TRUE(std::isfinite(price)) << "Base price must be finite";
    ASSERT_GT(price, 0.0) << "ATM put price must be positive";

    // --- FD Delta: (price(spot+h) - price(spot-h)) / (2h) ---
    const double h_spot = 0.5;
    PriceQuery q_up = q;
    q_up.spot = spot + h_spot;
    PriceQuery q_dn = q;
    q_dn.spot = spot - h_spot;
    double price_up = result->price(q_up);
    double price_dn = result->price(q_dn);
    double fd_delta = (price_up - price_dn) / (2.0 * h_spot);

    // For a put, delta should be negative (price decreases as spot increases)
    EXPECT_TRUE(std::isfinite(fd_delta)) << "FD delta must be finite";
    EXPECT_LT(fd_delta, 0.0) << "Put delta should be negative";
    EXPECT_GT(fd_delta, -1.0) << "Put delta should be > -1";

    // --- FD Vega: (price(sigma+h) - price(sigma-h)) / (2h) ---
    const double h_sigma = 1e-4;
    PriceQuery q_vup = q;
    q_vup.sigma = sigma + h_sigma;
    PriceQuery q_vdn = q;
    q_vdn.sigma = sigma - h_sigma;
    double price_vup = result->price(q_vup);
    double price_vdn = result->price(q_vdn);
    double fd_vega = (price_vup - price_vdn) / (2.0 * h_sigma);

    // Vega from the surface's analytic method
    double surface_vega = result->vega(q);

    EXPECT_TRUE(std::isfinite(fd_vega)) << "FD vega must be finite";
    EXPECT_TRUE(std::isfinite(surface_vega)) << "Surface vega must be finite";
    EXPECT_GT(fd_vega, 0.0) << "Vega should be positive for ATM put";
    EXPECT_GT(surface_vega, 0.0) << "Surface vega should be positive for ATM put";

    // FD vega should be consistent with surface vega within tolerance
    EXPECT_NEAR(fd_vega, surface_vega, 0.1)
        << "FD vega (" << fd_vega << ") should match surface vega ("
        << surface_vega << ") within tolerance";

    // --- Also check delta in the base segment (tau < 0.5) ---
    PriceQuery q_base{.spot = spot, .strike = strike, .tau = 0.3, .sigma = sigma, .rate = rate};
    PriceQuery q_base_up = q_base;
    q_base_up.spot = spot + h_spot;
    PriceQuery q_base_dn = q_base;
    q_base_dn.spot = spot - h_spot;
    double fd_delta_base = (result->price(q_base_up) - result->price(q_base_dn)) / (2.0 * h_spot);

    EXPECT_TRUE(std::isfinite(fd_delta_base)) << "FD delta (base segment) must be finite";
    EXPECT_LT(fd_delta_base, 0.0) << "Put delta (base segment) should be negative";
    EXPECT_GT(fd_delta_base, -1.0) << "Put delta (base segment) should be > -1";

    // FD vega in base segment should match surface vega
    PriceQuery q_base_vup = q_base;
    q_base_vup.sigma = sigma + h_sigma;
    PriceQuery q_base_vdn = q_base;
    q_base_vdn.sigma = sigma - h_sigma;
    double fd_vega_base = (result->price(q_base_vup) - result->price(q_base_vdn)) / (2.0 * h_sigma);
    double surface_vega_base = result->vega(q_base);
    EXPECT_NEAR(fd_vega_base, surface_vega_base, 0.1)
        << "FD vega base (" << fd_vega_base << ") should match surface vega base ("
        << surface_vega_base << ")";
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
            .moneyness = {0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                          1.05, 1.10, 1.15, 1.20, 1.25, 1.30},
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
        PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = tau, .sigma = 0.20, .rate = 0.05};
        double price = result->price(q);
        EXPECT_TRUE(std::isfinite(price))
            << "Price must be finite at tau=" << tau;
        EXPECT_GT(price, 0.0)
            << "ATM put price must be positive at tau=" << tau;
    }
}
