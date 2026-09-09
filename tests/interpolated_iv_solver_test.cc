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

// Exact affine-volatility prices isolate model provenance from fitting.
struct LinearVolatilityPrice {
    double scale = 1.0;
    double price(double, double strike, double, double sigma, double) const { return scale * strike * sigma; }
    double vega(double, double strike, double, double, double) const { return scale * strike; }
};

TEST(InterpolatedIVModelMetadata, UsesModelAnchorAndRejectsContradictoryOverrides) {
    using Surface = PriceTable<LinearVolatilityPrice>;
    auto make_surface = [] {
        SurfaceBounds bounds{-.2, .2, .1, 1.0, .1, .4, .01, .1};
        bounds.strike_bounds = StrikeBounds{90.0, 110.0};
        return Surface({}, bounds, OptionType::PUT, 0.0,
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
    using Table = PriceTable<LinearVolatilityPrice>;
    using View = detail::SharedPriceTableSurface<Table>;
    const SurfaceBounds bounds{-.2, .2, .1, 1.0, .1, .4, .01, .1};
    auto owner = std::make_shared<Table>(LinearVolatilityPrice{}, bounds, OptionType::PUT, 0.0);
    auto solver = InterpolatedIVSolver<View>::create(View(owner));
    ASSERT_TRUE(solver);
    IVQuery query(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = .25,
        .rate = .05, .option_type = OptionType::PUT}, 20.0);
    auto original = solver->solve(query);
    ASSERT_TRUE(original);
    EXPECT_NEAR(original->implied_vol, .2, 1e-10);

    *owner = Table(LinearVolatilityPrice{2.0}, bounds, OptionType::PUT, 0.0);
    auto retained_price = solver->solve(query);
    ASSERT_TRUE(retained_price);
    EXPECT_NEAR(retained_price->implied_vol, .2, 1e-10);

    auto changed = bounds;
    changed.tau_min = .5;
    changed.strike_bounds = StrikeBounds{150.0, 200.0};
    *owner = Table({}, changed, OptionType::CALL, 0.03);
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
    BSplineMultiKRefSurface table(std::move(inner), bounds, OptionType::PUT, 0.0,
        FixedExpiryMetadata{1.0, {}});
    EXPECT_NEAR(table.price(100.0, 100.0, .5, .2, .05), 2.0, 1e-12);
    *mutable_spline = std::move(*replacement);
    EXPECT_NEAR(table.price(100.0, 100.0, .5, .2, .05), 2.0, 1e-12);
    EXPECT_TRUE(table.contains_maturity(.5));
    EXPECT_FALSE(table.contains_maturity(.05));
}

TEST(PriceTablePublication, MovedModelStorageCannotChangePublishedSchedule) {
    using Table = PriceTable<LinearVolatilityPrice>;
    const SurfaceBounds bounds{-.2, .2, .1, 1.0, .1, .4, .01, .1};
    std::optional<FixedExpiryMetadata> model = FixedExpiryMetadata{1.0, {{.5, 2.0}}};
    auto* retained_dividend = model->discrete_dividends.data();
    Table table({}, bounds, OptionType::PUT, 0.0, std::move(model));
    retained_dividend->amount = 9.0;
    ASSERT_TRUE(table.fixed_expiry());
    EXPECT_DOUBLE_EQ(table.fixed_expiry()->discrete_dividends.front().amount, 2.0);
}

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
        auto table = builder.build(axes,
            [&](PriceTensor& tensor, const PriceTableAxes& a) {
                BSplineTensorAccessor accessor(tensor, a, K_ref_);
                eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));
            });
        ASSERT_TRUE(table.has_value()) << "Failed to build table";
        spline_ = table->spline;
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
    auto table = builder.build(axes,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            BSplineTensorAccessor accessor(tensor, a, K_ref);
            eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));
        });
    ASSERT_TRUE(table.has_value());

    auto wrapper_result = make_bspline_surface(table->spline, K_ref, 0.0, OptionType::PUT);
    ASSERT_TRUE(wrapper_result.has_value());

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
    auto table = builder.build(axes,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            BSplineTensorAccessor accessor(tensor, a, K_ref);
            eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, div_yield));
        });
    ASSERT_TRUE(table.has_value());

    auto wrapper_result = make_bspline_surface(table->spline, K_ref, div_yield, OptionType::PUT);
    ASSERT_TRUE(wrapper_result.has_value());

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

// ===========================================================================
// D8: multiple-root screen and signed vega pre-check
//
// The screen is exercised against a stub surface rather than a fitted
// B-spline table: the builder's EEP decomposition pushes a fitted sigma
// profile back toward monotonicity, so an engineered fold does not survive
// a real fit — and a screen must be tested on shapes it is meant to catch.
// The stub implements exactly the duck-typed interface that
// InterpolatedIVSolver::solve uses (price, vega, bounds, option_type,
// dividend_yield) and records every evaluation so scan cost is assertable.
// ===========================================================================

class ScreenStubSurface {
public:
    using Fn = std::function<double(double)>;

    ScreenStubSurface(Fn price_of_sigma, Fn vega_of_sigma,
                      double sigma_lo = 0.10, double sigma_hi = 0.50)
        : price_(std::move(price_of_sigma))
        , vega_(std::move(vega_of_sigma))
        , sigma_lo_(sigma_lo)
        , sigma_hi_(sigma_hi)
        , price_sigmas_(std::make_shared<std::vector<double>>())
        , vega_sigmas_(std::make_shared<std::vector<double>>()) {}

    /// Price depends on sigma only; the other axes are flat by construction.
    [[nodiscard]] double price(double, double, double, double sigma, double) const {
        price_sigmas_->push_back(sigma);
        return price_(sigma);
    }

    [[nodiscard]] double vega(double, double, double, double sigma, double) const {
        vega_sigmas_->push_back(sigma);
        return vega_(sigma);
    }

    [[nodiscard]] double m_min() const noexcept { return std::log(0.5); }
    [[nodiscard]] double m_max() const noexcept { return std::log(2.0); }
    [[nodiscard]] double tau_min() const noexcept { return 0.05; }
    [[nodiscard]] double tau_max() const noexcept { return 3.0; }
    [[nodiscard]] double sigma_min() const noexcept { return sigma_lo_; }
    [[nodiscard]] double sigma_max() const noexcept { return sigma_hi_; }
    [[nodiscard]] double rate_min() const noexcept { return 0.0; }
    [[nodiscard]] double rate_max() const noexcept { return 0.20; }
    [[nodiscard]] OptionType option_type() const noexcept { return OptionType::PUT; }
    [[nodiscard]] double dividend_yield() const noexcept { return 0.0; }

    [[nodiscard]] size_t price_calls() const { return price_sigmas_->size(); }
    [[nodiscard]] const std::vector<double>& price_sigmas() const { return *price_sigmas_; }
    [[nodiscard]] const std::vector<double>& vega_sigmas() const { return *vega_sigmas_; }

private:
    Fn price_;
    Fn vega_;
    double sigma_lo_;
    double sigma_hi_;
    // shared_ptr so the counters survive the copy into the solver.
    std::shared_ptr<std::vector<double>> price_sigmas_;
    std::shared_ptr<std::vector<double>> vega_sigmas_;
};

constexpr double kMarketPrice = 8.0;

/// ATM PUT query.  Time value == market price, so adaptive_bounds leaves the
/// bracket at the surface's own sigma range.
IVQuery screen_query() {
    return IVQuery(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                              .rate = 0.05, .option_type = OptionType::PUT},
                   kMarketPrice);
}

TEST(IVRootSensitivityTest, HealthyBracketCannotAdmitFlatBoundaryRoot) {
    auto price = [](double s) {
        return kMarketPrice + 100.0 * std::pow(s - 0.10, 3);
    };
    auto vega = [](double s) { return 300.0 * std::pow(s - 0.10, 2); };
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        ScreenStubSurface(price, vega));
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
    ASSERT_TRUE(result.error().last_vol.has_value());
    EXPECT_DOUBLE_EQ(*result.error().last_vol, 0.10);
}

TEST(IVRootSensitivityTest, PositiveVegaCannotIdentifyRoundedPricePlateau) {
    // The real-valued line has positive slope, but every representable price
    // in this bracket is 8.0. Disabling the optional precheck cannot make its
    // lower endpoint an identifiable IV.
    auto price = [](double s) { return kMarketPrice + 1e-16 * (s - 0.10); };
    InterpolatedIVSolverConfig config;
    config.vega_threshold = 0.0;
    config.detect_multiple_roots = false;
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        ScreenStubSurface(price, [](double) { return 1e-16; }), config);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
}

TEST(IVRootSensitivityTest, ZeroThresholdCannotDisableFlatRootAdmission) {
    InterpolatedIVSolverConfig config;
    config.vega_threshold = 0.0;
    config.detect_multiple_roots = false;
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        ScreenStubSurface([](double) { return kMarketPrice; },
                          [](double) { return 0.0; }), config);
    ASSERT_TRUE(solver.has_value());
    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
}

TEST(IVRootSensitivityTest, InteriorRootRequiresSensitivityAfterBrent) {
    auto price = [](double s) {
        return kMarketPrice + 100.0 * std::pow(s - 0.30, 3);
    };
    auto vega = [](double s) { return 300.0 * std::pow(s - 0.30, 2); };
    InterpolatedIVSolverConfig config;
    config.detect_multiple_roots = false;
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        ScreenStubSurface(price, vega), config);
    ASSERT_TRUE(solver.has_value());
    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
    ASSERT_TRUE(result.error().last_vol.has_value());
    EXPECT_NEAR(*result.error().last_vol, 0.30, 1e-5);
}

TEST(IVRootSensitivityTest, CallerCanRaiseFinalRootSensitivityFloor) {
    auto price = [](double s) {
        return kMarketPrice + 0.01 * (s - 0.10) + std::pow(s - 0.10, 2);
    };
    auto vega = [](double s) { return 0.01 + 2.0 * (s - 0.10); };
    InterpolatedIVSolverConfig config;
    config.vega_threshold = 0.02;
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        ScreenStubSurface(price, vega), config);
    ASSERT_TRUE(solver.has_value());
    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
    EXPECT_DOUBLE_EQ(result.error().final_error, 0.01);
}

TEST(IVRootSensitivityTest, NonfiniteRootVegaRefusesDespiteHealthyProbes) {
    for (double bad_vega : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
        auto price = [](double s) { return kMarketPrice + 10.0 * (s - 0.10); };
        auto vega = [bad_vega](double s) { return s == 0.10 ? bad_vega : 10.0; };
        auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
            ScreenStubSurface(price, vega));
        ASSERT_TRUE(solver.has_value());
        auto result = solver->solve(screen_query());
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, IVErrorCode::NumericalInstability);
        ASSERT_TRUE(result.error().last_vol.has_value());
        EXPECT_DOUBLE_EQ(*result.error().last_vol, 0.10);
    }
}

TEST(IVRootSensitivityTest, ReturnedVegaBelongsToTheRoot) {
    auto price = [](double s) { return kMarketPrice + s * s - 0.27 * 0.27; };
    auto vega = [](double s) { return 2.0 * s; };
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        ScreenStubSurface(price, vega));
    ASSERT_TRUE(solver.has_value());
    auto result = solver->solve(screen_query());
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->vega.has_value());
    EXPECT_NEAR(result->implied_vol, 0.27, 1e-5);
    EXPECT_NEAR(*result->vega, 0.54, 2e-5);
}

TEST(IVRootSensitivityTest, NumericalAdmissionHasNoAbsoluteCurrencyFloor) {
    constexpr double quote = 8e-200;
    auto price = [](double s) { return quote + 1e-198 * (s - 0.10); };
    InterpolatedIVSolverConfig config;
    config.vega_threshold = 0.0;
    config.detect_multiple_roots = false;
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        ScreenStubSurface(price, [](double) { return 1e-198; }), config);
    ASSERT_TRUE(solver.has_value());
    auto query = screen_query();
    query.market_price = quote;
    auto result = solver->solve(query);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->vega.has_value());
    EXPECT_DOUBLE_EQ(*result->vega, 1e-198);
}

TEST(IVRootSensitivityTest, RejectsInvalidSensitivityConfiguration) {
    auto price = [](double s) { return kMarketPrice + 10.0 * (s - 0.10); };
    const auto surface = ScreenStubSurface(price, [](double) { return 10.0; });
    for (double tolerance : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
        InterpolatedIVSolverConfig config;
        config.tolerance = tolerance;
        EXPECT_FALSE(InterpolatedIVSolver<ScreenStubSurface>::create(surface, config));
    }
    for (double threshold : {-1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
        InterpolatedIVSolverConfig config;
        config.vega_threshold = threshold;
        EXPECT_FALSE(InterpolatedIVSolver<ScreenStubSurface>::create(surface, config));
    }
}

/// Central-difference vega of a price profile (the stub's default).
ScreenStubSurface::Fn fd_vega(ScreenStubSurface::Fn price) {
    return [price](double sigma) {
        constexpr double h = 1e-4;
        return (price(sigma + h) - price(sigma - h)) / (2.0 * h);
    };
}

/// Objective with three roots at 0.21 / 0.36 / 0.46, none on a scan node
/// (the 17-point scan of [0.10, 0.50] has nodes every 0.025).
ScreenStubSurface::Fn three_root_price() {
    return [](double s) {
        return kMarketPrice + 100.0 * (s - 0.21) * (s - 0.36) * (s - 0.46);
    };
}

ScreenStubSurface make_stub(ScreenStubSurface::Fn price,
                            double sigma_lo = 0.10, double sigma_hi = 0.50) {
    return ScreenStubSurface(price, fd_vega(price), sigma_lo, sigma_hi);
}

// Regression: a surface with three roots in the bracket must be reported as
// ambiguous instead of silently returning whichever root Brent lands on.
TEST(IVScreenTest, ThreeCrossingsReportMultipleRoots) {
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        make_stub(three_root_price()));
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::MultipleRoots);
    EXPECT_DOUBLE_EQ(result.error().final_error, 3.0);
    ASSERT_TRUE(result.error().last_vol.has_value());
    // Low-sigma scan endpoint of the lowest transition interval [0.200, 0.225]
    // — an interval bound, not a root.
    EXPECT_NEAR(*result.error().last_vol, 0.200, 1e-12);
}

// Defended failure: with the screen off, the same query returns a single
// implied vol with no indication that two other vols price identically.
// This documents the bug class the screen exists to catch.
TEST(IVScreenTest, ScreenDisabledSilentlyReturnsOneOfThreeRoots) {
    InterpolatedIVSolverConfig config{};
    config.detect_multiple_roots = false;

    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        make_stub(three_root_price()), config);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_TRUE(result.has_value()) << "unscreened Brent converges silently";

    const double iv = result->implied_vol;
    const bool is_a_root = std::abs(iv - 0.21) < 1e-4 ||
                           std::abs(iv - 0.36) < 1e-4 ||
                           std::abs(iv - 0.46) < 1e-4;
    EXPECT_TRUE(is_a_root) << "returned " << iv;
    // Brent's bracket walk lands on the *highest* root (0.46); the lowest
    // (0.21) is the conventional answer.  Nothing in the result says the
    // caller got one of three — that silence is the defended failure.
    EXPECT_NEAR(iv, 0.46, 1e-4);
    EXPECT_GT(std::abs(iv - 0.21), 1e-3)
        << "unscreened solve happened to return the lowest root";
}

// A fold narrower than the old 9-point scan spacing (0.05) but at least one
// 17-point cell wide (0.025) must be caught.
TEST(IVScreenTest, FoldInsideFormerNinePointIntervalIsCaught) {
    // Base objective rises through 0.21; a triangular notch centred on the
    // scan node 0.325 dives negative and returns to zero at 0.30 and 0.35 —
    // exactly the nodes a 9-point scan would have used.
    auto price = [](double s) {
        const double notch = std::max(0.0, 1.0 - std::abs(s - 0.325) / 0.025);
        return kMarketPrice + (s - 0.21) - 0.3 * notch;
    };

    // A 9-point scan sees only these two, both positive: the fold is invisible.
    EXPECT_GT(price(0.30) - kMarketPrice, 0.0);
    EXPECT_GT(price(0.35) - kMarketPrice, 0.0);
    EXPECT_LT(price(0.325) - kMarketPrice, 0.0);

    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(make_stub(price));
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::MultipleRoots);
    EXPECT_DOUBLE_EQ(result.error().final_error, 3.0);
}

// Tangency: the objective touches zero at a scan node with the same sign on
// both sides.  Root selection is ambiguous, so the screen refuses.
TEST(IVScreenTest, TangencyReportsMultipleRoots) {
    auto price = [](double s) { return kMarketPrice + 4.0 * (s - 0.30) * (s - 0.30); };

    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(make_stub(price));
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::MultipleRoots);
    // An even-multiplicity contact is at least a double root.
    EXPECT_DOUBLE_EQ(result.error().final_error, 2.0);
    ASSERT_TRUE(result.error().last_vol.has_value());
    EXPECT_NEAR(*result.error().last_vol, 0.30, 1e-12);
}

// Boundary root at sigma_min that also satisfies the configured tolerance.
TEST(IVScreenTest, EndpointRootWithinToleranceIsReturned) {
    auto price = [](double s) { return kMarketPrice + 10.0 * (s - 0.10); };

    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(make_stub(price));
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_TRUE(result.has_value()) << "endpoint root must be honored";
    EXPECT_NEAR(result->implied_vol, 0.10, 1e-12);
    EXPECT_LE(result->final_error, 1e-6);
}

// zero_tol (1e-9 * spot = 1e-7 here) must never loosen a tighter configured
// tolerance: the endpoint residual is a "zero" for the scan but not a
// converged root for the caller.
TEST(IVScreenTest, EndpointRootOutsideConfiguredToleranceFailsBracketing) {
    auto price = [](double s) { return kMarketPrice + 5e-8 + 10.0 * (s - 0.10); };

    InterpolatedIVSolverConfig config{};
    config.tolerance = 1e-12;

    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        make_stub(price), config);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::BracketingFailed);
    EXPECT_NEAR(result.error().final_error, 5e-8, 1e-12);
}

// Endpoint root plus an interior crossing: two roots, so MultipleRoots.
TEST(IVScreenTest, EndpointRootPlusInteriorTransitionReportsMultipleRoots) {
    auto price = [](double s) {
        return kMarketPrice + 40.0 * (s - 0.10) * (0.4125 - s);
    };

    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(make_stub(price));
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::MultipleRoots);
    EXPECT_DOUBLE_EQ(result.error().final_error, 2.0);
    ASSERT_TRUE(result.error().last_vol.has_value());
    EXPECT_NEAR(*result.error().last_vol, 0.10, 1e-12);
}

// Every sample a zero: an unresolved continuum of roots.
TEST(IVScreenTest, AllZeroScanReportsMultipleRootsWithZeroError) {
    auto price = [](double) { return kMarketPrice; };

    InterpolatedIVSolverConfig config{};
    config.vega_threshold = 0.0;  // a flat surface has zero vega by design

    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        ScreenStubSurface(price, [](double) { return 0.0; }), config);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::MultipleRoots);
    EXPECT_DOUBLE_EQ(result.error().final_error, 0.0);
    ASSERT_TRUE(result.error().last_vol.has_value());
    EXPECT_NEAR(*result.error().last_vol, 0.10, 1e-12);
}

// A monotone surface must invert to the same root with and without the
// screen, and the screen must cost exactly its 17 scan evaluations.
TEST(IVScreenTest, MonotoneSurfaceUnchangedAndCostsSeventeenEvals) {
    auto price = [](double s) { return kMarketPrice + 20.0 * (s - 0.2637); };

    InterpolatedIVSolverConfig off{};
    off.detect_multiple_roots = false;
    auto stub_off = make_stub(price);
    auto solver_off = InterpolatedIVSolver<ScreenStubSurface>::create(stub_off, off);
    ASSERT_TRUE(solver_off.has_value());
    auto result_off = solver_off->solve(screen_query());
    ASSERT_TRUE(result_off.has_value());

    auto stub_on = make_stub(price);
    auto solver_on = InterpolatedIVSolver<ScreenStubSurface>::create(stub_on);
    ASSERT_TRUE(solver_on.has_value());
    auto result_on = solver_on->solve(screen_query());
    ASSERT_TRUE(result_on.has_value());

    EXPECT_NEAR(result_on->implied_vol, 0.2637, 1e-6);
    EXPECT_NEAR(result_on->implied_vol, result_off->implied_vol, 1e-6);
    EXPECT_EQ(stub_on.price_calls(), stub_off.price_calls() + 17)
        << "screen must cost exactly its 17 scan evaluations";

    // Those 17 evaluations are the uniform scan grid, endpoints included.
    ASSERT_GE(stub_on.price_sigmas().size(), 17u);
    for (size_t i = 0; i < 17; ++i) {
        const double expected = 0.10 + (0.50 - 0.10) * static_cast<double>(i) / 16.0;
        EXPECT_NEAR(stub_on.price_sigmas()[i], expected, 1e-12) << "scan point " << i;
    }
    EXPECT_DOUBLE_EQ(stub_on.price_sigmas()[16], 0.50) << "last sample lands on the bound";
}

// A converged root whose narrowed interval has a falling objective means the
// surface is non-monotone there; the root is not trustworthy.
TEST(IVScreenTest, NegativeNarrowedSlopeReportsMultipleRoots) {
    auto price = [](double s) { return kMarketPrice + 20.0 * (0.31 - s); };

    InterpolatedIVSolverConfig config{};
    config.vega_threshold = 0.0;  // the pre-check would reject this first

    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        make_stub(price), config);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::MultipleRoots);
    ASSERT_TRUE(result.error().last_vol.has_value());
    EXPECT_NEAR(*result.error().last_vol, 0.30, 1e-12);
}

// A non-finite objective anywhere in the scan is a broken surface, not an
// ambiguous one.
TEST(IVScreenTest, NonFiniteScanSampleIsNumericalInstability) {
    auto price = [](double s) {
        return (std::abs(s - 0.325) < 1e-12)
            ? std::numeric_limits<double>::quiet_NaN()
            : kMarketPrice + 20.0 * (s - 0.30);
    };
    // Explicit finite vega so the pre-check cannot claim the failure first.
    auto stub = ScreenStubSurface(price, [](double) { return 20.0; });

    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(stub);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NumericalInstability);
    ASSERT_TRUE(result.error().last_vol.has_value());
    EXPECT_NEAR(*result.error().last_vol, 0.325, 1e-12);
    EXPECT_EQ(stub.price_calls(), 10u) << "must stop at the offending sample";
}

// Zero features: the screen must fall through to Brent on the full bracket
// and report exactly what the unscreened path reports.
TEST(IVScreenTest, NoTransitionFallsThroughToBracketingFailed) {
    // Strictly positive across [0.10, 0.50]: the market price is below the
    // surface everywhere, so there is no root to find.
    auto price = [](double s) { return kMarketPrice + 1.0 + 20.0 * (s - 0.10); };

    auto solver_on = InterpolatedIVSolver<ScreenStubSurface>::create(make_stub(price));
    ASSERT_TRUE(solver_on.has_value());
    auto result_on = solver_on->solve(screen_query());
    ASSERT_FALSE(result_on.has_value());
    EXPECT_EQ(result_on.error().code, IVErrorCode::BracketingFailed);

    InterpolatedIVSolverConfig off{};
    off.detect_multiple_roots = false;
    auto solver_off = InterpolatedIVSolver<ScreenStubSurface>::create(make_stub(price), off);
    ASSERT_TRUE(solver_off.has_value());
    auto result_off = solver_off->solve(screen_query());
    ASSERT_FALSE(result_off.has_value());
    EXPECT_EQ(result_off.error().code, result_on.error().code)
        << "screened and unscreened paths must agree when there is no root";
}

// ---------------------------------------------------------------------------
// Signed vega pre-check (D8.1)
// ---------------------------------------------------------------------------

// Regression: the pre-check used std::abs(vega), so a strongly negative vega
// counted as healthy sensitivity.
TEST(IVScreenTest, AllNegativeProbeVegasRejected) {
    auto price = [](double s) { return kMarketPrice + 20.0 * (s - 0.30); };
    auto vega = [](double s) { return s < 0.25 ? -5.0 : (s < 0.35 ? -3.0 : -1.0); };

    InterpolatedIVSolverConfig config{};
    config.vega_threshold = 0.5;

    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(
        ScreenStubSurface(price, vega), config);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
    EXPECT_DOUBLE_EQ(result.error().final_error, -1.0) << "signed maximum";
}

TEST(IVScreenTest, MixedSignProbeVegasPassToScreen) {
    auto price = [](double s) { return kMarketPrice + 20.0 * (s - 0.30); };
    auto vega = [](double s) { return s < 0.25 ? -5.0 : (s < 0.35 ? 2.0 : 1.0); };

    InterpolatedIVSolverConfig config{};
    config.vega_threshold = 0.5;

    auto stub = ScreenStubSurface(price, vega);
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(stub, config);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->implied_vol, 0.30, 1e-6);
    EXPECT_GE(stub.price_calls(), 17u) << "the screen must have run";
}

// Regression: probes were fixed at {0.10, 0.25, 0.50} and could fall entirely
// outside a narrower bracket.  They must be the bracket's quartiles.
TEST(IVScreenTest, ProbeVegasEvaluatedAtBracketQuartiles) {
    auto price = [](double s) { return kMarketPrice + 20.0 * (s - 0.70); };

    auto stub = make_stub(price, 0.60, 0.80);
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(stub);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_TRUE(result.has_value());

    ASSERT_EQ(stub.vega_sigmas().size(), 4u);
    EXPECT_NEAR(stub.vega_sigmas()[0], 0.65, 1e-12);
    EXPECT_NEAR(stub.vega_sigmas()[1], 0.70, 1e-12);
    EXPECT_NEAR(stub.vega_sigmas()[2], 0.75, 1e-12);
    EXPECT_DOUBLE_EQ(stub.vega_sigmas()[3], result->implied_vol);
}

TEST(IVScreenTest, NonFiniteProbeVegaIsNumericalInstability) {
    auto price = [](double s) { return kMarketPrice + 20.0 * (s - 0.30); };
    auto vega = [](double s) {
        return (s > 0.25 && s < 0.35) ? std::numeric_limits<double>::quiet_NaN() : 1.0;
    };

    auto stub = ScreenStubSurface(price, vega);
    auto solver = InterpolatedIVSolver<ScreenStubSurface>::create(stub);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve(screen_query());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NumericalInstability);
    EXPECT_EQ(stub.price_calls(), 0u) << "must reject before any price eval";
}

// Regression: create()'s build_dividends default was known-empty, so direct
// construction over a dividend-aware surface falsely rejected correct
// non-empty query schedules (PR #449 pre-merge review, P1)
// Bug: the generic template cannot know surface provenance; the default is
//      now nullopt (unknown -> validation skipped), and validation is
//      opt-in via an explicit schedule.
TEST_F(InterpolatedIVSolverTest, DirectCreateDefaultsToUnknownSchedule) {
    // Direct create() with no build_dividends argument: unknown provenance.
    auto solver = InterpolatedIVSolver<BSplinePriceTable>::create(make_wrapper());
    ASSERT_TRUE(solver.has_value());

    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .option_type = OptionType::PUT},
        8.0,
        {{.calendar_time = 0.5, .amount = 2.0}});

    auto iv_result = solver->solve(query);
    // Unknown provenance skips schedule validation; the query may still
    // fail for other numerical reasons, but never for a schedule mismatch.
    if (!iv_result.has_value()) {
        EXPECT_NE(iv_result.error().code, IVErrorCode::DiscreteDividendMismatch);
    }
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
