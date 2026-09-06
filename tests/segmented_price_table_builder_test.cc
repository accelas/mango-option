// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/american_option.hpp"
#include <cmath>
#include <vector>

using namespace mango;

namespace {

// #458: use #488's raw end-to-end snapshots while increasing the real
// moneyness data sites through the former collocation failure region.
TEST(SegmentedPriceTableBuilderTest, RawDividendSamplesCrossFormerFittingCliff) {
    // Existing dividend/support expansion produces 118,160,188,281 actual
    // sites from these explicit requested grids; no data sites are moved.
    for (size_t n : {50u, 68u, 80u, 120u}) {
        SCOPED_TRACE(n);
        std::vector<double> log_m(n);
        for (size_t i = 0; i < n; ++i) {
            log_m[i] = std::log(0.92) + (std::log(1.08)-std::log(0.92))*i/(n-1);
        }
        SegmentedPriceTableBuilder::Config config{
            .K_ref = 100.0, .option_type = OptionType::PUT,
            .dividends = {.dividend_yield = 0.01,
                          .discrete_dividends = {{0.25, 1.5}, {0.5, 1.5}}},
            .grid = {.moneyness = log_m, .vol = {0.1, 0.15, 0.2, 0.3},
                     .rate = {0.02, 0.03, 0.05, 0.07}},
            .maturity = 1.0,
            .tau_points_per_segment = 8,
            .pde_accuracy = make_grid_accuracy(GridAccuracyProfile::High),
        };
        auto surface = SegmentedPriceTableBuilder::build(config);
        ASSERT_TRUE(surface.has_value()) << surface.error();
        PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0,
            .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.01,
            .option_type = OptionType::PUT}, 0.1);
        p.discrete_dividends = config.dividends.discrete_dividends;
        auto solver = AmericanOptionSolver::create(
            p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::Ultra)});
        ASSERT_TRUE(solver.has_value());
        auto reference = solver->solve();
        ASSERT_TRUE(reference.has_value());
        EXPECT_NEAR(surface->price(100.0, 100.0, 1.0, 0.1, 0.05),
                    reference->value(), 0.003);
    }
}

std::vector<double> log_m_grid(std::initializer_list<double> moneyness) {
    std::vector<double> out;
    out.reserve(moneyness.size());
    for (double m : moneyness) {
        out.push_back(std::log(m));
    }
    return out;
}

}  // namespace

// Regression #488: a post-dividend backward-time segment must contain raw
// end-to-end PDE samples, rather than evolution of a fitted initial state.
TEST(SegmentedPriceTableBuilderTest, LowVolatilityFixedExpiryMatchesDirectSolve) {
    std::vector<double> x;
    for (int i = -10; i <= 10; ++i) x.push_back(0.05 * i);
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.discrete_dividends = {{0.5, 3.0}}},
        .grid = {.moneyness = x, .vol = {0.1, 0.15, 0.2, 0.3},
                 .rate = {0.02, 0.03, 0.05, 0.07}},
        .maturity = 1.0,
        .tau_points_per_segment = 5,
        .pde_accuracy = make_grid_accuracy(GridAccuracyProfile::Ultra),
    };
    auto surface = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(surface.has_value()) << surface.error();

    // At remaining maturity .75, elapsed time is .25, so the originally
    // anchored .5 dividend is now .25 away. K equals K_ref: no strike blend.
    PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0,
        .maturity = 0.75, .rate = 0.05, .option_type = OptionType::PUT}, 0.1);
    p.discrete_dividends = {{0.25, 3.0}};
    auto high = AmericanOptionSolver::create(
        p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
    auto ultra = AmericanOptionSolver::create(
        p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::Ultra)});
    ASSERT_TRUE(high.has_value());
    ASSERT_TRUE(ultra.has_value());
    auto reference = high->solve();
    auto converged = ultra->solve();
    ASSERT_TRUE(reference.has_value());
    ASSERT_TRUE(converged.has_value());
    ASSERT_NEAR(reference->value(), converged->value(), 0.001);
    const double actual = surface->price(100.0, 100.0, 0.75, 0.1, 0.05);
    std::cout << "FIXED_EXPIRY price=" << actual << " direct=" << converged->value()
              << " difference=" << actual - converged->value() << '\n';
    EXPECT_NEAR(actual, converged->value(), 0.003);
}

TEST(SegmentedPriceTableBuilderTest, DiagnosticsCountRawRowsAndSingleExpirySolves) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0, .option_type = OptionType::PUT,
        .dividends = {.discrete_dividends = {{0.25, 1.0}, {0.5, 1.0}}},
        .grid = {.moneyness = {-0.2, -0.1, 0.0, 0.1, 0.2},
                 .vol = {0.1, 0.15, 0.2, 0.3},
                 .rate = {0.02, 0.03, 0.05, 0.07}},
        .maturity = 1.0,
        .tau_target_dt = 0.01, .tau_points_min = 4, .tau_points_max = 4,
    };
    auto result = SegmentedPriceTableBuilder::build_with_diagnostics(config);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_EQ(result->pde_solves, 16u);  // independent of the three segments
    EXPECT_EQ(result->sample_rows, 3u * 4u * 16u);
    EXPECT_EQ(result->tau_point_cap_hits, 3u);
    EXPECT_EQ(result->surface.num_pieces(), 3u);
    EXPECT_FALSE(result->surface.contains_maturity(0.5));
    EXPECT_FALSE(result->surface.contains_maturity(0.75));
    EXPECT_TRUE(result->surface.contains_maturity(0.8));
}

TEST(SegmentedPriceTableBuilderTest, RejectsUnsortedSampleAxesBeforeSolving) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0, .option_type = OptionType::PUT,
        .grid = {.moneyness = {-0.2, -0.1, 0.0, 0.1, 0.2},
                 .vol = {0.15, 0.1, 0.2, 0.3},
                 .rate = {0.02, 0.03, 0.05, 0.07}},
        .maturity = 1.0,
    };
    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::GridNotSorted);
    EXPECT_EQ(result.error().axis_index, 2u);
}

TEST(SegmentedPriceTableBuilderTest, RawCallAndPutPricesAndGreeksShareTheSameSurface) {
    std::vector<double> x;
    for (int i = -10; i <= 10; ++i) x.push_back(0.05 * i);
    for (auto type : {OptionType::PUT, OptionType::CALL}) {
        SCOPED_TRACE(static_cast<int>(type));
        SegmentedPriceTableBuilder::Config config{
            .K_ref = 100.0, .option_type = type,
            .dividends = {.discrete_dividends = {{0.5, 3.0}}},
            .grid = {.moneyness = x, .vol = {0.1, 0.15, 0.2, 0.3},
                     .rate = {0.02, 0.03, 0.05, 0.07}},
            .maturity = 1.0,
            .pde_accuracy = make_grid_accuracy(GridAccuracyProfile::High),
        };
        auto surface = SegmentedPriceTableBuilder::build(config);
        ASSERT_TRUE(surface.has_value()) << surface.error();
        for (double tau : {0.25, 0.75}) {
            PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0,
                .maturity = tau, .rate = 0.05, .option_type = type}, 0.2);
            if (tau > 0.5) p.discrete_dividends = {{tau - 0.5, 3.0}};
            auto fine = AmericanOptionSolver::create(
                p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
            auto finer = AmericanOptionSolver::create(
                p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::Ultra)});
            ASSERT_TRUE(fine.has_value());
            ASSERT_TRUE(finer.has_value());
            auto reference = fine->solve();
            auto converged = finer->solve();
            ASSERT_TRUE(reference.has_value());
            ASSERT_TRUE(converged.has_value());
            ASSERT_NEAR(reference->value(), converged->value(), 0.001);
            const auto price = [&](double spot, double time, double sigma, double rate) {
                return surface->price(spot, 100.0, time, sigma, rate);
            };
            EXPECT_NEAR(price(100, tau, 0.2, 0.05), converged->value(), 0.003);

            // Public Greeks must differentiate the delivered raw-sample
            // surface. Finite differences use interior points in each axis.
            auto delta = surface->greek(Greek::Delta, p);
            auto gamma = surface->gamma(p);
            auto theta = surface->greek(Greek::Theta, p);
            auto rho = surface->greek(Greek::Rho, p);
            ASSERT_TRUE(delta.has_value());
            ASSERT_TRUE(gamma.has_value());
            ASSERT_TRUE(theta.has_value());
            ASSERT_TRUE(rho.has_value());
            constexpr double hs = 0.005, h = 1e-5;
            const double mid = price(100, tau, 0.2, 0.05);
            const double up = price(100 + hs, tau, 0.2, 0.05);
            const double down = price(100 - hs, tau, 0.2, 0.05);
            EXPECT_NEAR(*delta, (up - down) / (2 * hs), 1e-7);
            EXPECT_NEAR(*gamma, (up - 2 * mid + down) / (hs * hs), 1e-7);
            EXPECT_NEAR(surface->vega(100, 100, tau, 0.2, 0.05),
                (price(100, tau, 0.2 + h, 0.05) - price(100, tau, 0.2 - h, 0.05)) / (2 * h), 1e-6);
            EXPECT_NEAR(*theta,
                (price(100, tau - h, 0.2, 0.05) - price(100, tau + h, 0.2, 0.05)) / (2 * h), 1e-6);
            EXPECT_NEAR(*rho,
                (price(100, tau, 0.2, 0.05 + h) - price(100, tau, 0.2, 0.05 - h)) / (2 * h), 1e-6);
        }
    }
}

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
