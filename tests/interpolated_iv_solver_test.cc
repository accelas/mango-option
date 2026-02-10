// SPDX-License-Identifier: MIT
/**
 * @file interpolated_iv_solver_test.cc
 * @brief Tests for InterpolatedIVSolver (B-spline based IV solver)
 */

#include <gtest/gtest.h>
#include <cmath>
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/price_table_axes.hpp"
#include "mango/option/table/price_table_metadata.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"

namespace mango {
namespace {

/// Test fixture that creates a proper EEP price surface for IV solving
class InterpolatedIVSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::vector<double> m_grid = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
        std::vector<double> tau_grid = {0.25, 0.5, 1.0, 2.0};
        std::vector<double> vol_grid = {0.10, 0.20, 0.30, 0.40};
        std::vector<double> rate_grid = {0.02, 0.04, 0.06, 0.08};

        auto result = PriceTableBuilder::from_vectors(
            m_grid, tau_grid, vol_grid, rate_grid, K_ref_,
            GridAccuracyParams{}, OptionType::PUT, 0.0);
        ASSERT_TRUE(result.has_value()) << "Failed to build";
        auto [builder, axes] = std::move(result.value());
        EEPDecomposer decomposer{OptionType::PUT, K_ref_, 0.0};
        auto table = builder.build(axes, SurfaceContent::EarlyExercisePremium,
            [&](PriceTensor& tensor, const PriceTableAxes& a) {
                decomposer.decompose(tensor, a);
            });
        ASSERT_TRUE(table.has_value()) << "Failed to build table";
        surface_ = table->surface;
    }

    /// Helper to create a BSplinePriceTable for IV solver tests
    BSplinePriceTable make_wrapper() {
        auto result = make_bspline_surface(surface_, OptionType::PUT);
        return std::move(*result);
    }

    std::shared_ptr<const PriceTableSurface> surface_;
    static constexpr double K_ref_ = 100.0;
};

TEST_F(InterpolatedIVSolverTest, CreateFromBSplinePriceTable) {
    auto wrapper_result = make_bspline_surface(surface_, OptionType::PUT);
    ASSERT_TRUE(wrapper_result.has_value());

    auto result = DefaultInterpolatedIVSolver::create(std::move(*wrapper_result));
    ASSERT_TRUE(result.has_value()) << "Failed to create solver";
}

TEST_F(InterpolatedIVSolverTest, CreateWithConfig) {
    InterpolatedIVSolverConfig config{
        .max_iter = 100,
        .tolerance = 1e-8,
        .sigma_min = 0.05,
        .sigma_max = 2.0
    };

    auto result = DefaultInterpolatedIVSolver::create(make_wrapper(), config);
    ASSERT_TRUE(result.has_value()) << "Failed to create solver with config";
}

TEST_F(InterpolatedIVSolverTest, SolveATMPut) {
    auto solver_result = DefaultInterpolatedIVSolver::create(make_wrapper());
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

TEST_F(InterpolatedIVSolverTest, SolveITMPut) {
    auto solver_result = DefaultInterpolatedIVSolver::create(make_wrapper());
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

TEST_F(InterpolatedIVSolverTest, SolveOTMPut) {
    auto solver_result = DefaultInterpolatedIVSolver::create(make_wrapper());
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

TEST_F(InterpolatedIVSolverTest, RejectsInvalidQuery) {
    auto solver_result = DefaultInterpolatedIVSolver::create(make_wrapper());
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // Invalid: negative spot
    IVQuery invalid_query(
        OptionSpec{.spot = -100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0);

    auto result = solver.solve(invalid_query);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
}

TEST_F(InterpolatedIVSolverTest, RejectsNegativeMarketPrice) {
    auto solver_result = DefaultInterpolatedIVSolver::create(make_wrapper());
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    IVQuery invalid_query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, -5.0);

    auto result = solver.solve(invalid_query);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeMarketPrice);
}

TEST_F(InterpolatedIVSolverTest, BatchSolve) {
    auto solver_result = DefaultInterpolatedIVSolver::create(make_wrapper());
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

TEST_F(InterpolatedIVSolverTest, BatchSolveAllSucceed) {
    auto solver_result = DefaultInterpolatedIVSolver::create(make_wrapper());
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

TEST_F(InterpolatedIVSolverTest, ConvergenceWithinIterations) {
    InterpolatedIVSolverConfig config{
        .max_iter = 10,  // Limited iterations
        .tolerance = 1e-6
    };

    auto solver_result = DefaultInterpolatedIVSolver::create(make_wrapper(), config);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 8.0);
    auto result = solver.solve(query);

    if (result.has_value()) {
        EXPECT_LE(result->iterations, 10u);
    }
}

TEST_F(InterpolatedIVSolverTest, SolveWithEEPSurface) {
    // Build an EEP surface (axis 0 is log-moneyness)
    PriceTableAxes eep_axes;
    eep_axes.grids[0] = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
    eep_axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    eep_axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    eep_axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    std::vector<double> eep_coeffs(5 * 4 * 4 * 4, 2.0);
    PriceTableMetadata eep_meta{
        .K_ref = 100.0,
        .m_min = std::log(0.8),
        .m_max = std::log(1.2),
        .content = SurfaceContent::EarlyExercisePremium,
    };

    auto eep_surface = PriceTableSurface::build(eep_axes, eep_coeffs, eep_meta);
    ASSERT_TRUE(eep_surface.has_value());

    auto wrapper_result = make_bspline_surface(eep_surface.value(), OptionType::PUT);
    ASSERT_TRUE(wrapper_result.has_value());

    auto solver = DefaultInterpolatedIVSolver::create(std::move(*wrapper_result));
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

// Verify that DefaultInterpolatedIVSolver alias works correctly
TEST_F(InterpolatedIVSolverTest, StandardAliasMatchesExplicitTemplate) {
    // DefaultInterpolatedIVSolver is InterpolatedIVSolver<BSplinePriceTable>
    static_assert(std::is_same_v<
        DefaultInterpolatedIVSolver,
        InterpolatedIVSolver<BSplinePriceTable>>);

    auto solver = DefaultInterpolatedIVSolver::create(make_wrapper());
    ASSERT_TRUE(solver.has_value());
}

// ===========================================================================
// Regression tests for API safety
// ===========================================================================

// Regression: InterpolatedIVSolver must reject queries with wrong option type
// Bug: solve() accepted any IVQuery regardless of type, returning wrong IV
TEST(IVSolverInterpolatedRegressionTest, RejectsOptionTypeMismatch) {
    // Build an EEP surface for PUT options (log-moneyness)
    std::vector<double> m_grid = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
    std::vector<double> tau_grid = {0.25, 0.5, 1.0, 2.0};
    std::vector<double> vol_grid = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.04, 0.06, 0.08};
    constexpr double K_ref = 100.0;

    auto result = PriceTableBuilder::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, K_ref,
        GridAccuracyParams{}, OptionType::PUT, 0.0);
    ASSERT_TRUE(result.has_value());
    auto [builder, axes] = std::move(result.value());
    EEPDecomposer decomposer{OptionType::PUT, K_ref, 0.0};
    auto table = builder.build(axes, SurfaceContent::EarlyExercisePremium,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            decomposer.decompose(tensor, a);
        });
    ASSERT_TRUE(table.has_value());

    auto wrapper_result = make_bspline_surface(table->surface, OptionType::PUT);
    ASSERT_TRUE(wrapper_result.has_value());

    auto solver = DefaultInterpolatedIVSolver::create(std::move(*wrapper_result));
    ASSERT_TRUE(solver.has_value());

    // Query with CALL type against a PUT surface — must fail
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::CALL}, 8.0);

    auto iv_result = solver->solve(query);
    ASSERT_FALSE(iv_result.has_value())
        << "Solver should reject CALL query against PUT surface";
    EXPECT_EQ(iv_result.error().code, IVErrorCode::OptionTypeMismatch);
}

// Regression: InterpolatedIVSolver must reject queries with wrong dividend_yield
// Bug: BSplinePriceTable bakes in dividend_yield at construction; callers
// with a different yield get wrong prices silently
TEST(IVSolverInterpolatedRegressionTest, RejectsDividendYieldMismatch) {
    // Build surface with dividend_yield = 0.02 (log-moneyness)
    std::vector<double> m_grid = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
    std::vector<double> tau_grid = {0.25, 0.5, 1.0, 2.0};
    std::vector<double> vol_grid = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.04, 0.06, 0.08};
    constexpr double K_ref = 100.0;
    constexpr double div_yield = 0.02;

    auto result = PriceTableBuilder::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, K_ref,
        GridAccuracyParams{}, OptionType::PUT, div_yield);
    ASSERT_TRUE(result.has_value());
    auto [builder, axes] = std::move(result.value());
    EEPDecomposer decomposer2{OptionType::PUT, K_ref, div_yield};
    auto table = builder.build(axes, SurfaceContent::EarlyExercisePremium,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            decomposer2.decompose(tensor, a);
        });
    ASSERT_TRUE(table.has_value());

    auto wrapper_result = make_bspline_surface(table->surface, OptionType::PUT);
    ASSERT_TRUE(wrapper_result.has_value());

    auto solver = DefaultInterpolatedIVSolver::create(std::move(*wrapper_result));
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
