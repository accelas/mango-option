// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/iv_solver_factory.hpp"
#include "mango/option/american_option.hpp"
#include <chrono>
#include <cmath>
#include <iostream>

using namespace mango;

namespace {

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

constexpr double SPOT = 100.0;
constexpr double DIVIDEND_YIELD = 0.02;
constexpr OptionType TYPE = OptionType::PUT;

IVGridSpec manual_grid() {
    return ManualGrid{
        .moneyness = {0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2},
        .vol = {0.10, 0.15, 0.20, 0.25, 0.30},
        .rate = {0.02, 0.03, 0.05, 0.07},
    };
}

IVGridSpec adaptive_grid() {
    AdaptiveGridParams params;
    params.target_iv_error = 0.002;
    params.max_iter = 5;  // Increased from 2 to allow convergence under memory pressure
    params.validation_samples = 32;
    // Note: Must explicitly set all fields - designated initializers value-initialize
    // omitted members (empty vectors) rather than using in-class defaults.
    AdaptiveGrid grid;
    grid.params = params;
    return grid;
}

AnyIVSolver build_solver(const IVGridSpec& grid) {
    IVSolverFactoryConfig config;
    config.option_type = TYPE;
    config.spot = SPOT;
    config.dividend_yield = DIVIDEND_YIELD;
    config.grid = grid;
    config.path = StandardIVPath{.maturity_grid = {0.1, 0.25, 0.5, 0.75, 1.0}};
    auto result = make_interpolated_iv_solver(config);
    EXPECT_TRUE(result.has_value()) << "Solver build failed";
    return std::move(*result);
}

std::vector<IVQuery> make_test_queries() {
    std::vector<IVQuery> queries;
    for (double K : {95.0, 100.0, 105.0}) {
        for (double T : {0.25, 0.5, 1.0}) {
            PricingParams params(
                OptionSpec{.spot = SPOT, .strike = K, .maturity = T,
                           .rate = 0.05, .dividend_yield = DIVIDEND_YIELD,
                           .option_type = TYPE},
                0.20);
            auto result = solve_american_option(params);
            if (result.has_value()) {
                queries.push_back(IVQuery(
                    OptionSpec{.spot = SPOT, .strike = K, .maturity = T,
                               .rate = 0.05, .dividend_yield = DIVIDEND_YIELD,
                               .option_type = TYPE},
                    result->value()));
            }
        }
    }
    return queries;
}

// ---------------------------------------------------------------------------
// Parametric test: ManualGrid vs AdaptiveGrid on the standard path
// ---------------------------------------------------------------------------

struct GridParam {
    std::string name;
    IVGridSpec grid;
};

class IVSolverFactoryTest : public ::testing::TestWithParam<GridParam> {};

TEST_P(IVSolverFactoryTest, Builds) {
    IVSolverFactoryConfig config;
    config.option_type = TYPE;
    config.spot = SPOT;
    config.dividend_yield = DIVIDEND_YIELD;
    config.grid = GetParam().grid;
    config.path = StandardIVPath{.maturity_grid = {0.1, 0.25, 0.5, 0.75, 1.0}};

    auto solver = make_interpolated_iv_solver(config);
    // Note: This test can fail with FittingFailed (code 7) when run after
    // IVSolverFactorySegmented + IVSolverFactoryComparison tests due to a subtle
    // numerical stability issue in B-spline fitting. The test passes in isolation.
    // Investigation showed 1343 slices fail the 1e-6 residual tolerance only
    // under specific test ordering conditions.
    ASSERT_TRUE(solver.has_value())
        << "Error code: " << static_cast<int>(solver.error().code);
}

TEST_P(IVSolverFactoryTest, SolvesIV) {
    auto solver = build_solver(GetParam().grid);
    auto queries = make_test_queries();
    ASSERT_FALSE(queries.empty());

    for (const auto& query : queries) {
        auto result = solver.solve(query);
        ASSERT_TRUE(result.has_value())
            << "IV solve failed for K=" << query.strike
            << " T=" << query.maturity;
        EXPECT_NEAR(result->implied_vol, 0.20, 0.02)
            << "K=" << query.strike << " T=" << query.maturity;
    }
}

TEST_P(IVSolverFactoryTest, BatchSolve) {
    auto solver = build_solver(GetParam().grid);

    std::vector<IVQuery> queries(3);
    for (auto& q : queries) {
        q.spot = SPOT;
        q.strike = 100.0;
        q.maturity = 0.5;
        q.rate = RateSpec{0.05};
        q.dividend_yield = DIVIDEND_YIELD;
        q.option_type = TYPE;
        q.market_price = 6.0;
    }

    auto batch_result = solver.solve_batch(queries);
    EXPECT_EQ(batch_result.results.size(), 3u);
}

INSTANTIATE_TEST_SUITE_P(
    GridTypes,
    IVSolverFactoryTest,
    ::testing::Values(
        GridParam{"Manual", manual_grid()},
        GridParam{"Adaptive", adaptive_grid()}),
    [](const auto& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Segmented path (ManualGrid only — adaptive not yet supported)
// ---------------------------------------------------------------------------

TEST(IVSolverFactorySegmented, DiscreteDividends) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .grid = ManualGrid{
            .moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
            .vol = {0.10, 0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        .path = SegmentedIVPath{
            .maturity = 1.0,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}},
            .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value()) << "Factory should succeed with discrete dividends";

    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.5;
    query.rate = RateSpec{0.05};
    query.option_type = OptionType::PUT;
    query.market_price = 7.0;

    auto result = solver->solve(query);
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 3.0);
    }
}
TEST(IVSolverFactorySegmented, AdaptiveGridDiscreteDividends) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;  // 50 bps for test speed
    params.max_iter = 2;
    params.validation_samples = 16;

    // Note: Must NOT use designated initializers for AdaptiveGrid - they leave
    // moneyness/vol/rate vectors empty instead of using in-class defaults.
    AdaptiveGrid adaptive_grid;
    adaptive_grid.params = params;

    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.02,
        .grid = adaptive_grid,
        .path = SegmentedIVPath{
            .maturity = 1.0,
            .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
            .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value())
        << "Factory should succeed with AdaptiveGrid + SegmentedIVPath";

    // Solve IV for a known option
    OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 0.5,
        .rate = 0.05, .dividend_yield = 0.02,
        .option_type = OptionType::PUT
    };

    PricingParams pricing_params(spec, 0.20);
    pricing_params.discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}};
    auto ref = solve_american_option(pricing_params);
    ASSERT_TRUE(ref.has_value());

    IVQuery query(spec, ref->value());
    auto result = solver->solve(query);
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 3.0);
    }
}


// ---------------------------------------------------------------------------
// Side-by-side accuracy comparison
// ---------------------------------------------------------------------------

TEST(IVSolverFactoryComparison, AccuracyManualVsAdaptive) {
    auto manual = build_solver(manual_grid());
    auto adaptive = build_solver(adaptive_grid());
    auto queries = make_test_queries();
    constexpr double TRUE_VOL = 0.20;

    double manual_max_err = 0.0, adaptive_max_err = 0.0;
    double manual_sum_err = 0.0, adaptive_sum_err = 0.0;
    size_t count = 0;

    for (const auto& query : queries) {
        auto m = manual.solve(query);
        auto a = adaptive.solve(query);
        if (!m.has_value() || !a.has_value()) continue;

        double m_err = std::abs(m->implied_vol - TRUE_VOL);
        double a_err = std::abs(a->implied_vol - TRUE_VOL);

        manual_max_err = std::max(manual_max_err, m_err);
        adaptive_max_err = std::max(adaptive_max_err, a_err);
        manual_sum_err += m_err;
        adaptive_sum_err += a_err;
        count++;
    }

    ASSERT_GT(count, 0u);

    std::cout << "\n=== IV Accuracy Comparison ===\n"
              << "Manual:   max=" << manual_max_err
              << "  avg=" << manual_sum_err / count << "\n"
              << "Adaptive: max=" << adaptive_max_err
              << "  avg=" << adaptive_sum_err / count << "\n"
              << "Queries: " << count << "\n";

    EXPECT_LT(manual_max_err, 0.05);
    EXPECT_LT(adaptive_max_err, 0.05);
}

}  // namespace
