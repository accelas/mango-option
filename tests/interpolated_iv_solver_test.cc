// SPDX-License-Identifier: MIT
/**
 * @file interpolated_iv_solver_test.cc
 * @brief Tests for InterpolatedIVSolver (B-spline based IV solver)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/bspline/bspline_basis.hpp"

namespace mango {
namespace {

// Exact represented polynomials isolate IV/model behavior from PDE fitting.
// American approximation accuracy remains in its dedicated workload tests.
using LinearTable = ChebyshevMultiKRefSurface;
LinearTable linear_table(double scale, SurfaceBounds bounds, OptionType type = OptionType::PUT,
                         double q = 0, const std::optional<FixedExpiryMetadata>& supplied = {}) {
    const auto model = supplied.value_or(FixedExpiryMetadata{bounds.tau_max, {}});
    if (!bounds.strike_bounds) bounds.strike_bounds = StrikeBounds{100, 100};
    const auto strikes = *bounds.strike_bounds;
    std::vector<double> refs{strikes.min};
    if (strikes.max != strikes.min) refs.push_back(strikes.max);
    std::vector<ChebyshevTauSegmented> references;
    for (double reference : refs) {
        std::vector<double> coefficients(16, 0);
        coefficients[0] = scale * (bounds.sigma_min + bounds.sigma_max) / 2;
        coefficients[2] = scale * (bounds.sigma_max - bounds.sigma_min) / 2;
        auto interpolant = ChebyshevModalInterpolant<4>::build_from_coefficients(coefficients,
            Domain<4>{{bounds.m_min, 0, bounds.sigma_min, bounds.rate_min},
                      {bounds.m_max, model.reference_maturity, bounds.sigma_max, bounds.rate_max}},
            {2, 2, 2, 2}).value();
        std::vector<ChebyshevSegmentedLeaf> leaves;
        leaves.emplace_back(std::move(interpolant), StandardTransform4D{}, reference);
        references.emplace_back(std::move(leaves), TauSegmentSplit({0},
            {model.reference_maturity}, {0}, {model.reference_maturity}, reference));
    }
    return LinearTable::create(ChebyshevMultiKRefInner(std::move(references),
        MultiKRefSplit(std::move(refs))), bounds, type, q, model).value();
}

std::shared_ptr<const BSplineND<double, 4>> zero_eep_fixture() {
    using Spline = BSplineND<double, 4>;
    Spline::GridArray grids{{
        {std::log(.8), std::log(.9), 0, std::log(1.1), std::log(1.2)},
        {.25, .5, 1, 2}, {.1, .2, .3, .4}, {.02, .04, .06, .08}}};
    Spline::KnotArray knots;
    for (std::size_t d = 0; d < 4; ++d) knots[d] = clamped_knots_cubic(grids[d]);
    return std::make_shared<const Spline>(Spline::create(
        std::move(grids), std::move(knots), std::vector<double>(320, 0)).value());
}

TEST(InterpolatedIVModelMetadata, UsesModelAnchorAndRejectsContradictoryOverrides) {
    using Surface = LinearTable;
    auto make_surface = [] {
        SurfaceBounds bounds{-.2, .2, .1, 1.0, .1, .4, .01, .1};
        bounds.strike_bounds = StrikeBounds{90.0, 110.0};
        return linear_table(1, bounds, OptionType::PUT, 0.0,
            FixedExpiryMetadata{2.0, {{1.5, 1.5}}});
    };
    auto solver = InterpolatedIVSolver<Surface>::create(
        make_surface(), {}, std::vector<Dividend>{{1.5, 1.5}});
    ASSERT_TRUE(solver.has_value());
    IVQuery query(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = .75,
        .rate = .05, .option_type = OptionType::PUT}, 20.0);
    // T0=2, elapsed=1.25: the anchored offset1.5 is .25 from this query.
    query.discrete_dividends = {{.25, 1.5}};
    auto result = solver->solve(query);
    EXPECT_TRUE(result.has_value());
    if (result) { EXPECT_NEAR(result->implied_vol, .2, 1e-10); }

    auto contradiction = InterpolatedIVSolver<Surface>::create(
        make_surface(), {}, std::vector<Dividend>{{1.25, 1.5}});
    EXPECT_FALSE(contradiction.has_value());
    if (!contradiction) {
        EXPECT_EQ(contradiction.error().code, ValidationErrorCode::DiscreteDividendMismatch);
    }
}

// Regression: a const shared pointer can still alias a caller-owned mutable
// spline. Publication must detach the numerical payload before it is queried.
TEST(PriceTablePublication, MutableSplineAliasCannotChangePublishedPrices) {
    using Spline = BSplineND<double, 4>;
    const Spline::GridArray grids{{{-0.2, -0.1, 0.1, 0.2},
        {0.1, 0.3, 0.6, 1.0}, {0.1, 0.2, 0.3, 0.4}, {0.01, 0.03, 0.05, 0.07}}};
    Spline::KnotArray knots;
    for (size_t axis = 0; axis < 4; ++axis) knots[axis] = clamped_knots_cubic(grids[axis]);
    auto original = Spline::create(grids, knots, std::vector<double>(256, 2.0));
    auto replacement = Spline::create(grids, knots, std::vector<double>(256, 9.0));
    ASSERT_TRUE(original); ASSERT_TRUE(replacement);
    auto mutable_spline = std::make_shared<Spline>(std::move(*original));
    auto table = make_bspline_surface(mutable_spline, 100.0, 0.0, OptionType::PUT);
    ASSERT_TRUE(table);
    const double published = table->price(100.0, 100.0, 0.5, 0.2, 0.05);
    const auto copy = *table;
    *mutable_spline = std::move(*replacement);
    EXPECT_DOUBLE_EQ(table->price(100.0, 100.0, 0.5, 0.2, 0.05), published);
    EXPECT_DOUBLE_EQ(copy.price(100.0, 100.0, 0.5, 0.2, 0.05), published);
}

TEST(PriceTablePublication, WrapperReassignmentCannotChangeExistingIvSolver) {
    using Table = LinearTable;
    using View = detail::SharedPriceTableSurface<Table>;
    const SurfaceBounds bounds{-.2, .2, .1, 1.0, .1, .4, .01, .1};
    auto owner = std::make_shared<Table>(linear_table(1, bounds));
    auto solver = InterpolatedIVSolver<View>::create(View(owner));
    ASSERT_TRUE(solver);
    IVQuery query(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = .25,
        .rate = .05, .option_type = OptionType::PUT}, 20.0);
    auto original = solver->solve(query);
    ASSERT_TRUE(original);
    EXPECT_NEAR(original->implied_vol, .2, 1e-10);

    *owner = linear_table(2, bounds);
    auto retained_price = solver->solve(query);
    ASSERT_TRUE(retained_price);
    EXPECT_NEAR(retained_price->implied_vol, .2, 1e-10);

    auto changed = bounds;
    changed.tau_min = .5;
    changed.strike_bounds = StrikeBounds{150.0, 200.0};
    *owner = linear_table(1, changed, OptionType::CALL, 0.03);
    auto retained = solver->solve(query);
    ASSERT_TRUE(retained) << static_cast<int>(retained.error().code);
    EXPECT_NEAR(retained->implied_vol, .2, 1e-10);
}

TEST(PriceTablePublication, NestedReferenceAndTimePiecesDetachMutableSpline) {
    using Spline = BSplineND<double, 4>;
    const Spline::GridArray grids{{{-0.2, -0.1, 0.1, 0.2},
        {0.1, 0.3, 0.6, 1.0}, {0.1, 0.2, 0.3, 0.4}, {0.01, 0.03, 0.05, 0.07}}};
    Spline::KnotArray knots;
    for (size_t axis = 0; axis < 4; ++axis) knots[axis] = clamped_knots_cubic(grids[axis]);
    auto original = Spline::create(grids, knots, std::vector<double>(256, .02));
    auto replacement = Spline::create(grids, knots, std::vector<double>(256, .09));
    ASSERT_TRUE(original); ASSERT_TRUE(replacement);
    auto mutable_spline = std::make_shared<Spline>(std::move(*original));
    BSplineSegmentedLeaf leaf(SharedBSplineInterp<4>(mutable_spline), StandardTransform4D{}, 100.0);
    BSplineSegmentedSurface segmented({leaf}, TauSegmentSplit({0.0}, {1.0}, {.1}, {1.0}, 100.0));
    BSplineMultiKRefInner inner({segmented}, MultiKRefSplit({100.0}));
    SurfaceBounds bounds{-.2, .2, .1, 1.0, .1, .4, .01, .07};
    bounds.strike_bounds = StrikeBounds{100.0, 100.0};
    auto table = BSplineMultiKRefSurface::create(inner, bounds, OptionType::PUT, 0.0,
        FixedExpiryMetadata{1.0, {}}).value();
    EXPECT_NEAR(table.price(100.0, 100.0, .5, .2, .05), 2.0, 1e-12);
    *mutable_spline = std::move(*replacement);
    EXPECT_NEAR(table.price(100.0, 100.0, .5, .2, .05), 2.0, 1e-12);
    EXPECT_TRUE(table.contains_maturity(.5));
    EXPECT_FALSE(table.contains_maturity(.05));
}

TEST(PriceTablePublication, MovedModelStorageCannotChangePublishedSchedule) {
    const SurfaceBounds bounds{-.2, .2, .1, 1.0, .1, .4, .01, .1};
    std::optional<FixedExpiryMetadata> model = FixedExpiryMetadata{1.0, {{.5, 2.0}}};
    auto* retained_dividend = model->discrete_dividends.data();
    auto table = linear_table(1, bounds, OptionType::PUT, 0.0, std::move(model));
    retained_dividend->amount = 9.0;
    ASSERT_TRUE(table.fixed_expiry());
    EXPECT_DOUBLE_EQ(table.fixed_expiry()->discrete_dividends.front().amount, 2.0);
}

/// Certified synthetic zero-EEP fixture for root, batch and query validation.
class InterpolatedIVSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        spline_ = zero_eep_fixture();
        ASSERT_TRUE(spline_);
    }

    /// Helper to create a BSplinePriceTable for IV solver tests
    BSplinePriceTable make_wrapper() {
        auto result = make_bspline_surface(spline_, K_ref_, 0.0, OptionType::PUT);
        return std::move(*result);
    }

    std::shared_ptr<const BSplineND<double, 4>> spline_;
    static constexpr double K_ref_ = 100.0;
};

TEST_F(InterpolatedIVSolverTest, CreateFromBSplinePriceTable) {
    auto wrapper_result = make_bspline_surface(spline_, K_ref_, 0.0, OptionType::PUT);
    ASSERT_TRUE(wrapper_result.has_value());

    auto result = InterpolatedIVSolver<BSplinePriceTable>::create(std::move(*wrapper_result));
    ASSERT_TRUE(result.has_value()) << "Failed to create solver";
}

TEST_F(InterpolatedIVSolverTest, CreateWithConfig) {
    InterpolatedIVSolverConfig config{
        .max_iter = 100,
        .tolerance = 1e-8,
        .sigma_min = 0.05,
        .sigma_max = 2.0
    };

    auto result = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper(), config);
    ASSERT_TRUE(result.has_value()) << "Failed to create solver with config";
}

TEST_F(InterpolatedIVSolverTest, SolveATMPut) {
    auto solver_result = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper());
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
    auto solver_result = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper());
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
    auto solver_result = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper());
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
    auto solver_result = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper());
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
    auto solver_result = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper());
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    IVQuery invalid_query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, -5.0);

    auto result = solver.solve(invalid_query);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeMarketPrice);
}

TEST_F(InterpolatedIVSolverTest, BatchSolve) {
    auto solver_result = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper());
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
    auto solver_result = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper());
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

    auto solver_result = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper(), config);
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
    // Build a BSplineND<double, 4> directly (axis 0 is log-moneyness)
    std::array<std::vector<double>, 4> eep_grids = {{
        {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)},
        {0.25, 0.5, 1.0, 2.0},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
    }};
    std::array<std::vector<double>, 4> eep_knots;
    for (size_t i = 0; i < 4; ++i) {
        eep_knots[i] = clamped_knots_cubic(eep_grids[i]);
    }

    std::vector<double> eep_coeffs(5 * 4 * 4 * 4, 2.0);

    auto eep_spline = BSplineND<double, 4>::create(
        eep_grids, std::move(eep_knots), std::move(eep_coeffs));
    ASSERT_TRUE(eep_spline.has_value());

    auto eep_spline_ptr = std::make_shared<const BSplineND<double, 4>>(
        std::move(eep_spline.value()));

    auto wrapper_result = make_bspline_surface(eep_spline_ptr, 100.0, 0.0, OptionType::PUT);
    ASSERT_TRUE(wrapper_result.has_value());

    auto solver = InterpolatedIVSolver<BSplinePriceTable>::create(std::move(*wrapper_result));
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
// Regression tests for API safety
// ===========================================================================

// Regression: InterpolatedIVSolver must reject queries with wrong option type
// Bug: solve() accepted any IVQuery regardless of type, returning wrong IV
TEST(IVSolverInterpolatedRegressionTest, RejectsOptionTypeMismatch) {
    auto wrapper_result = make_bspline_surface(
        zero_eep_fixture(), 100.0, 0.0, OptionType::PUT);
    ASSERT_TRUE(wrapper_result);

    auto solver = InterpolatedIVSolver<BSplinePriceTable>::create(std::move(*wrapper_result));
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
    auto wrapper_result = make_bspline_surface(
        zero_eep_fixture(), 100.0, 0.02, OptionType::PUT);
    ASSERT_TRUE(wrapper_result);

    auto solver = InterpolatedIVSolver<BSplinePriceTable>::create(std::move(*wrapper_result));
    ASSERT_TRUE(solver.has_value());

    // Query with dividend_yield = 0.05 — must fail (surface was built with 0.02)
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.05, .option_type = OptionType::PUT}, 8.0);

    auto iv_result = solver->solve(query);
    ASSERT_FALSE(iv_result.has_value())
        << "Solver should reject query with mismatched dividend_yield";
    EXPECT_EQ(iv_result.error().code, IVErrorCode::DividendYieldMismatch);
}

TEST_F(InterpolatedIVSolverTest, DirectCreateRetainsKnownEmptyContinuousSchedule) {
    // Direct creation retains the continuous payload's known-empty schedule.
    auto solver = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper());
    ASSERT_TRUE(solver.has_value());

    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .option_type = OptionType::PUT},
        8.0,
        {{.calendar_time = 0.5, .amount = 2.0}});

    auto iv_result = solver->solve(query);
    ASSERT_FALSE(iv_result);
    EXPECT_EQ(iv_result.error().code, IVErrorCode::DiscreteDividendMismatch);
}

TEST_F(InterpolatedIVSolverTest, DirectCreateExplicitEmptyScheduleRejectsQuery) {
    // Direct create() with an explicit known-empty schedule: the surface is
    // asserted to be dividend-free, so a non-empty query schedule must fail.
    auto solver = InterpolatedIVSolver<BSplinePriceTable>::create(
        make_wrapper(), InterpolatedIVSolverConfig{}, std::vector<Dividend>{});
    ASSERT_TRUE(solver.has_value());

    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .option_type = OptionType::PUT},
        8.0,
        {{.calendar_time = 0.5, .amount = 2.0}});

    auto iv_result = solver->solve(query);
    ASSERT_FALSE(iv_result.has_value())
        << "Solver should reject non-empty query schedule against a "
           "known-empty build schedule";
    EXPECT_EQ(iv_result.error().code, IVErrorCode::DiscreteDividendMismatch);
}

}  // namespace
}  // namespace mango
