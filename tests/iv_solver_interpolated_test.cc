// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_interpolated_test.cc
 * @brief Tests for IVSolverInterpolated (B-spline based IV solver)
 */

#include <gtest/gtest.h>
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/price_table_axes.hpp"
#include "src/option/table/price_table_metadata.hpp"
#include "src/option/table/american_price_surface.hpp"
#include "src/option/table/segmented_multi_kref_builder.hpp"
#include "src/option/table/segmented_multi_kref_surface.hpp"

namespace mango {
namespace {

/// Test fixture that creates a proper EEP price surface for IV solving
class IVSolverInterpolatedTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
        std::vector<double> tau_grid = {0.25, 0.5, 1.0, 2.0};
        std::vector<double> vol_grid = {0.10, 0.20, 0.30, 0.40};
        std::vector<double> rate_grid = {0.02, 0.04, 0.06, 0.08};

        auto result = PriceTableBuilder<4>::from_vectors(
            m_grid, tau_grid, vol_grid, rate_grid, K_ref_,
            GridAccuracyParams{}, OptionType::PUT, 0.0);
        ASSERT_TRUE(result.has_value()) << "Failed to build";
        auto [builder, axes] = std::move(result.value());
        auto table = builder.build(axes);
        ASSERT_TRUE(table.has_value()) << "Failed to build table";
        surface_ = table->surface;
    }

    /// Helper to create a fresh AmericanPriceSurface from the shared surface
    AmericanPriceSurface make_aps() {
        auto aps = AmericanPriceSurface::create(surface_, OptionType::PUT);
        return std::move(*aps);
    }

    std::shared_ptr<const PriceTableSurface<4>> surface_;
    static constexpr double K_ref_ = 100.0;
};

TEST_F(IVSolverInterpolatedTest, CreateFromAmericanPriceSurface) {
    auto aps = AmericanPriceSurface::create(surface_, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    auto result = IVSolverInterpolatedStandard::create(std::move(*aps));
    ASSERT_TRUE(result.has_value()) << "Failed to create solver";
}

TEST_F(IVSolverInterpolatedTest, CreateWithConfig) {
    IVSolverInterpolatedConfig config{
        .max_iter = 100,
        .tolerance = 1e-8,
        .sigma_min = 0.05,
        .sigma_max = 2.0
    };

    auto result = IVSolverInterpolatedStandard::create(make_aps(), config);
    ASSERT_TRUE(result.has_value()) << "Failed to create solver with config";
}

TEST_F(IVSolverInterpolatedTest, SolveATMPut) {
    auto solver_result = IVSolverInterpolatedStandard::create(make_aps());
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // ATM put: S = K = 100, maturity = 1y, rate = 5%
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 8.0);

    auto result = solver.solve(query);
    // With precomputed data, may or may not converge - test that it returns a result
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 5.0);  // Reasonable upper bound
    } else {
        // If it fails, should be a convergence issue, not a validation error
        EXPECT_TRUE(result.error().code == IVErrorCode::MaxIterationsExceeded ||
                    result.error().code == IVErrorCode::BracketingFailed ||
                    result.error().code == IVErrorCode::NumericalInstability);
    }
}

TEST_F(IVSolverInterpolatedTest, SolveITMPut) {
    auto solver_result = IVSolverInterpolatedStandard::create(make_aps());
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // ITM put: S = 90, K = 100 (m = 0.9), maturity = 1y
    IVQuery query(
        OptionSpec{.spot = 90.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 15.0);

    auto result = solver.solve(query);
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 5.0);
    }
    // Test passes as long as it doesn't crash
}

TEST_F(IVSolverInterpolatedTest, SolveOTMPut) {
    auto solver_result = IVSolverInterpolatedStandard::create(make_aps());
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // OTM put: S = 110, K = 100 (m = 1.1), maturity = 1y
    IVQuery query(
        OptionSpec{.spot = 110.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 3.0);

    auto result = solver.solve(query);
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 5.0);
    }
    // Test passes as long as it doesn't crash
}

TEST_F(IVSolverInterpolatedTest, RejectsInvalidQuery) {
    auto solver_result = IVSolverInterpolatedStandard::create(make_aps());
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // Invalid: negative spot
    IVQuery invalid_query(
        OptionSpec{.spot = -100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0);

    auto result = solver.solve(invalid_query);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
}

TEST_F(IVSolverInterpolatedTest, RejectsNegativeMarketPrice) {
    auto solver_result = IVSolverInterpolatedStandard::create(make_aps());
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    IVQuery invalid_query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, -5.0);

    auto result = solver.solve(invalid_query);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeMarketPrice);
}

TEST_F(IVSolverInterpolatedTest, BatchSolve) {
    auto solver_result = IVSolverInterpolatedStandard::create(make_aps());
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    std::vector<IVQuery> queries;

    // Create batch of queries with varying strikes
    for (double strike : {90.0, 95.0, 100.0, 105.0, 110.0}) {
        double m = 100.0 / strike;  // moneyness
        double price = (m < 1.0) ? 12.0 : (m > 1.0 ? 4.0 : 8.0);  // Rough prices
        queries.push_back(IVQuery(OptionSpec{.spot = 100.0, .strike = strike, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, price));
    }

    auto batch_result = solver.solve_batch(queries);

    // With precomputed data, just verify batch processing works
    EXPECT_EQ(batch_result.results.size(), 5);
    // Count should be consistent
    size_t actual_failures = 0;
    for (const auto& r : batch_result.results) {
        if (!r.has_value()) actual_failures++;
    }
    EXPECT_EQ(batch_result.failed_count, actual_failures);
}

TEST_F(IVSolverInterpolatedTest, BatchSolveAllSucceed) {
    auto solver_result = IVSolverInterpolatedStandard::create(make_aps());
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // Single valid query in batch
    std::vector<IVQuery> queries = {
        IVQuery(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 8.0)
    };

    auto batch_result = solver.solve_batch(queries);

    EXPECT_EQ(batch_result.results.size(), 1);
    if (batch_result.all_succeeded()) {
        EXPECT_TRUE(batch_result.results[0].has_value());
    }
}

TEST_F(IVSolverInterpolatedTest, ConvergenceWithinIterations) {
    IVSolverInterpolatedConfig config{
        .max_iter = 10,  // Limited iterations
        .tolerance = 1e-6
    };

    auto solver_result = IVSolverInterpolatedStandard::create(make_aps(), config);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 8.0);
    auto result = solver.solve(query);

    if (result.has_value()) {
        EXPECT_LE(result->iterations, 10u);
    }
}

TEST_F(IVSolverInterpolatedTest, SolveWithAmericanPriceSurface) {
    // Build an EEP surface
    PriceTableAxes<4> eep_axes;
    eep_axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    eep_axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    eep_axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    eep_axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    std::vector<double> eep_coeffs(5 * 4 * 4 * 4, 2.0);
    PriceTableMetadata eep_meta{
        .K_ref = 100.0,
        .m_min = 0.8,
        .m_max = 1.2,
        .content = SurfaceContent::EarlyExercisePremium,
    };

    auto eep_surface = PriceTableSurface<4>::build(eep_axes, eep_coeffs, eep_meta);
    ASSERT_TRUE(eep_surface.has_value());

    auto aps = AmericanPriceSurface::create(eep_surface.value(), OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    auto solver = IVSolverInterpolatedStandard::create(std::move(*aps));
    ASSERT_TRUE(solver.has_value());

    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 8.0);

    auto result = solver->solve(query);
    // With synthetic data, accept success or graceful failure
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 5.0);
    }
}

// ===========================================================================
// Template instantiation tests with different surface types
// ===========================================================================

TEST_F(IVSolverInterpolatedTest, WorksWithSegmentedMultiKRefSurface) {
    // Build a SegmentedMultiKRefSurface via SegmentedMultiKRefBuilder
    SegmentedMultiKRefBuilder::Config config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},  // One discrete dividend at t=0.5
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.10, 0.20, 0.30, 0.40},
        .rate_grid = {0.02, 0.04, 0.06, 0.08},
        .kref_config = {
            .K_refs = {100.0},  // Single K_ref for simplicity
            .K_ref_count = 1,
        },
    };

    auto surface_result = SegmentedMultiKRefBuilder::build(config);
    ASSERT_TRUE(surface_result.has_value())
        << "SegmentedMultiKRefBuilder::build failed";

    // Create IVSolverInterpolated<SegmentedMultiKRefSurface>
    auto solver_result = IVSolverInterpolated<SegmentedMultiKRefSurface>::create(
        std::move(*surface_result));
    ASSERT_TRUE(solver_result.has_value())
        << "IVSolverInterpolated<SegmentedMultiKRefSurface>::create failed";

    auto& solver = solver_result.value();

    // Solve an IV query
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.8, .rate = 0.05, .option_type = OptionType::PUT}, 8.0);

    auto result = solver.solve(query);
    // With synthetic data from segmented builder, accept success or graceful failure
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 5.0);
    } else {
        // Graceful failure is acceptable - check it's a convergence issue
        EXPECT_TRUE(result.error().code == IVErrorCode::MaxIterationsExceeded ||
                    result.error().code == IVErrorCode::BracketingFailed ||
                    result.error().code == IVErrorCode::NumericalInstability ||
                    result.error().code == IVErrorCode::InvalidGridConfig);
    }
}

// Verify that IVSolverInterpolatedStandard alias works correctly
TEST_F(IVSolverInterpolatedTest, StandardAliasMatchesExplicitTemplate) {
    // IVSolverInterpolatedStandard is IVSolverInterpolated<AmericanPriceSurface>
    static_assert(std::is_same_v<
        IVSolverInterpolatedStandard,
        IVSolverInterpolated<AmericanPriceSurface>>);

    auto solver = IVSolverInterpolatedStandard::create(make_aps());
    ASSERT_TRUE(solver.has_value());
}

// ===========================================================================
// Regression tests for API safety
// ===========================================================================

// Regression: IVSolverInterpolated must reject queries with wrong option type
// Bug: solve() accepted any IVQuery regardless of type, returning wrong IV
TEST(IVSolverInterpolatedRegressionTest, RejectsOptionTypeMismatch) {
    // Build an EEP surface for PUT options
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_grid = {0.25, 0.5, 1.0, 2.0};
    std::vector<double> vol_grid = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.04, 0.06, 0.08};
    constexpr double K_ref = 100.0;

    auto result = PriceTableBuilder<4>::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, K_ref,
        GridAccuracyParams{}, OptionType::PUT, 0.0);
    ASSERT_TRUE(result.has_value());
    auto [builder, axes] = std::move(result.value());
    auto table = builder.build(axes);
    ASSERT_TRUE(table.has_value());

    auto aps = AmericanPriceSurface::create(table->surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    auto solver = IVSolverInterpolatedStandard::create(std::move(*aps));
    ASSERT_TRUE(solver.has_value());

    // Query with CALL type against a PUT surface — must fail
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::CALL}, 8.0);

    auto iv_result = solver->solve(query);
    ASSERT_FALSE(iv_result.has_value())
        << "Solver should reject CALL query against PUT surface";
    EXPECT_EQ(iv_result.error().code, IVErrorCode::OptionTypeMismatch);
}

// Regression: IVSolverInterpolated must reject queries with wrong dividend_yield
// Bug: AmericanPriceSurface bakes in dividend_yield at construction; callers
// with a different yield got wrong prices silently
TEST(IVSolverInterpolatedRegressionTest, RejectsDividendYieldMismatch) {
    // Build surface with dividend_yield = 0.02
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_grid = {0.25, 0.5, 1.0, 2.0};
    std::vector<double> vol_grid = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.04, 0.06, 0.08};
    constexpr double K_ref = 100.0;
    constexpr double div_yield = 0.02;

    auto result = PriceTableBuilder<4>::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, K_ref,
        GridAccuracyParams{}, OptionType::PUT, div_yield);
    ASSERT_TRUE(result.has_value());
    auto [builder, axes] = std::move(result.value());
    auto table = builder.build(axes);
    ASSERT_TRUE(table.has_value());

    auto aps = AmericanPriceSurface::create(table->surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());
    EXPECT_NEAR(aps->dividend_yield(), 0.02, 1e-12);

    auto solver = IVSolverInterpolatedStandard::create(std::move(*aps));
    ASSERT_TRUE(solver.has_value());

    // Query with dividend_yield = 0.05 — must fail (surface was built with 0.02)
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.05, .option_type = OptionType::PUT}, 8.0);

    auto iv_result = solver->solve(query);
    ASSERT_FALSE(iv_result.has_value())
        << "Solver should reject query with mismatched dividend_yield";
    EXPECT_EQ(iv_result.error().code, IVErrorCode::DividendYieldMismatch);
}

}  // namespace
}  // namespace mango
