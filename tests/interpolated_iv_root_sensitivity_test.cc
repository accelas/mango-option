// SPDX-License-Identifier: MIT
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/serialization/from_data.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <limits>

namespace {
using namespace mango;

// Actual stored modal prices, constant in the other physical coordinates.
// x=4*sigma-1.5 on [.125,.625]; coefficients are quote units at K=128.
auto certified_curve(const std::array<double, 4>& coefficients,
                     double strike = 128.0, double sigma_hi = .625) {
    PriceTableData record;
    record.surface_type = surface_types::kChebyshev4DSegmented;
    record.option_type = OptionType::PUT;
    record.ratio_bounds = MoneynessBounds{.99, 1.01};
    record.strike_bounds = StrikeBounds{strike, strike};
    record.fixed_expiry = FixedExpiryMetadata{1.0, {}};
    record.bounds_m_min = std::log(.99);
    record.bounds_m_max = std::log(1.01);
    record.bounds_tau_min = .25;
    record.bounds_tau_max = record.maturity = .75;
    record.bounds_sigma_min = .125;
    record.bounds_sigma_max = sigma_hi;
    record.bounds_rate_min = .03125;
    record.bounds_rate_max = .09375;
    PriceTableData::Segment segment;
    segment.K_ref = strike;
    segment.tau_start = segment.tau_min = 0.0;
    segment.tau_end = segment.tau_max = 1.0;
    segment.ndim = 4;
    segment.interp_type = "chebyshev_modal";
    segment.domain_lo = {-.125, 0.0, .125, .03125};
    segment.domain_hi = {.125, 1.0, sigma_hi, .09375};
    segment.num_pts = {2, 2, 4, 2};
    segment.values.assign(32, 0.0);
    for (size_t i = 0; i < coefficients.size(); ++i) {
        segment.values[2 * i] = coefficients[i] / 128.0;
    }
    record.segments.push_back(std::move(segment));
    return from_data<ChebyshevMultiKRefInner>(record);
}

IVQuery root_query(double quote = 8.0, double strike = 128.0) {
    return IVQuery(OptionSpec{.spot = strike, .strike = strike, .maturity = .5,
        .rate = .0625, .option_type = OptionType::PUT}, quote);
}

TEST(IVRootSensitivityTest, HealthyBracketCannotAdmitFlatBoundaryRoot) {
    // P=8+(x+1)^3, with exactly zero derivative at the lower endpoint.
    auto table = certified_curve({10.5, 3.75, 1.5, .25});
    ASSERT_TRUE(table);
    ASSERT_EQ(table->proof_status(), PriceProofStatus::Certified);
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table);
    ASSERT_TRUE(solver);
    auto result = solver->solve(root_query());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
    ASSERT_TRUE(result.error().last_vol);
    EXPECT_DOUBLE_EQ(*result.error().last_vol, .125);
}

TEST(IVRootSensitivityTest, PositiveVegaCannotIdentifyRoundedPricePlateau) {
    auto table = certified_curve({8.0, 1e-16, 0.0, 0.0});
    ASSERT_TRUE(table);
    ASSERT_GT(table->vega(128, 128, .5, .125, .0625), 0.0);
    InterpolatedIVSolverConfig config;
    config.vega_threshold = 0.0;
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config);
    ASSERT_TRUE(solver);
    auto result = solver->solve(root_query());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
}

TEST(IVRootSensitivityTest, ZeroThresholdCannotDisableFlatRootAdmission) {
    auto table = certified_curve({8.0, 0.0, 0.0, 0.0});
    ASSERT_TRUE(table);
    InterpolatedIVSolverConfig config;
    config.vega_threshold = 0.0;
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config);
    ASSERT_TRUE(solver);
    auto result = solver->solve(root_query());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
}

TEST(IVRootSensitivityTest, InteriorRootRequiresSensitivityAfterBrent) {
    // P=8+x^3 is nondecreasing, with zero derivative at the interior root.
    auto table = certified_curve({8.0, .75, 0.0, .25});
    ASSERT_TRUE(table);
    InterpolatedIVSolverConfig config;
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config);
    ASSERT_TRUE(solver);
    auto result = solver->solve(root_query());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
}

TEST(IVRootSensitivityTest, CallerCanRaiseFinalRootSensitivityFloor) {
    // P=8+.01*(sigma-.125)+(sigma-.125)^2.
    auto table = certified_curve({8.09625, .1275, .03125, 0.0});
    ASSERT_TRUE(table);
    InterpolatedIVSolverConfig config;
    config.vega_threshold = .02;
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config);
    ASSERT_TRUE(solver);
    auto result = solver->solve(root_query());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, IVErrorCode::VegaTooSmall);
    EXPECT_NEAR(result.error().final_error, .01, 1e-14);
}

TEST(IVRootSensitivityTest, ReturnedVegaBelongsToTheRoot) {
    // P=8+sigma^2-.375^2; root .375, vega .75.
    auto table = certified_curve({8.03125, .1875, .03125, 0.0});
    ASSERT_TRUE(table);
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table);
    ASSERT_TRUE(solver);
    auto result = solver->solve(root_query());
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->vega);
    EXPECT_NEAR(result->implied_vol, .375, 1e-5);
    EXPECT_NEAR(*result->vega, .75, 2e-5);
}

TEST(IVRootSensitivityTest, FiniteCertifiedPriceCanHaveUnrepresentableRootVega) {
    // Price/K ranges from 1/16 to 3/16, so prices remain representable.
    // The narrow sigma domain makes dPrice/dSigma=16*K overflow at this scale.
    const double strike = std::numeric_limits<double>::max() / 4.0;
    auto table = certified_curve({16.0, 8.0, 0.0, 0.0}, strike, .1328125);
    ASSERT_TRUE(table) << static_cast<int>(table.error().code);
    ASSERT_EQ(table->proof_status(), PriceProofStatus::Certified);
    ASSERT_TRUE(std::isfinite(table->price(strike, strike, .5, .125, .0625)));
    ASSERT_FALSE(std::isfinite(table->vega(strike, strike, .5, .125, .0625)));
    InterpolatedIVSolverConfig config;
    config.vega_threshold = 0.0;
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config);
    ASSERT_TRUE(solver);
    auto result = solver->solve(root_query(strike / 16.0, strike));
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, IVErrorCode::NumericalInstability);
    ASSERT_TRUE(result.error().last_vol);
    EXPECT_DOUBLE_EQ(*result.error().last_vol, .125);
}

TEST(IVRootSensitivityTest, NumericalAdmissionHasNoAbsoluteCurrencyFloor) {
    constexpr double quote = 8e-200;
    auto table = certified_curve({quote + 2.5e-199, 2.5e-199, 0.0, 0.0});
    ASSERT_TRUE(table);
    InterpolatedIVSolverConfig config;
    config.vega_threshold = 0.0;
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config);
    ASSERT_TRUE(solver);
    auto result = solver->solve(root_query(quote));
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->vega);
    EXPECT_NEAR(*result->vega, 1e-198, 1e-210);
}

TEST(IVRootSensitivityTest, RejectsInvalidSensitivityConfiguration) {
    auto table = certified_curve({9.0, 1.0, 0.0, 0.0});
    ASSERT_TRUE(table);
    for (double tolerance : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
        InterpolatedIVSolverConfig config;
        config.tolerance = tolerance;
        EXPECT_FALSE(InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config));
    }
    for (double threshold : {-1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
        InterpolatedIVSolverConfig config;
        config.vega_threshold = threshold;
        EXPECT_FALSE(InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config));
    }
}
TEST(IVBracketTest, ExplicitHighVolatilityBracketHasNoHiddenPriceBasedCap) {
    auto table = certified_curve({8, 1, 0, 0}, 128, 8.125);
    ASSERT_TRUE(table);
    InterpolatedIVSolverConfig config;
    config.sigma_max = 8.125;
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config);
    ASSERT_TRUE(solver);
    auto result = solver->solve(root_query());
    ASSERT_TRUE(result);
    EXPECT_NEAR(result->implied_vol, 4.125, 1e-6);

    auto defaults = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table);
    ASSERT_TRUE(defaults);
    auto outside_default = defaults->solve(root_query());
    ASSERT_FALSE(outside_default);
    EXPECT_EQ(outside_default.error().code, IVErrorCode::BracketingFailed);
}

TEST(IVBracketTest, DisjointCallerAndTableBoundsRefuseInsteadOfWidening) {
    auto table = certified_curve({8, 1, 0, 0});
    ASSERT_TRUE(table);
    InterpolatedIVSolverConfig config;
    config.sigma_min = 1;
    config.sigma_max = 2;
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config);
    ASSERT_FALSE(solver);
    EXPECT_EQ(solver.error().code, ValidationErrorCode::InvalidBounds);
}

TEST(IVBracketTest, CallerBoundsRestrictTheCertifiedDomain) {
    auto table = certified_curve({8, 1, 0, 0});
    ASSERT_TRUE(table);
    InterpolatedIVSolverConfig config;
    config.sigma_min = .2;
    config.sigma_max = .3;
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config);
    ASSERT_TRUE(solver);
    auto inside = solver->solve(root_query(7.5));
    ASSERT_TRUE(inside);
    EXPECT_NEAR(inside->implied_vol, .25, 1e-6);
    auto outside = solver->solve(root_query(8));
    ASSERT_FALSE(outside);
    EXPECT_EQ(outside.error().code, IVErrorCode::BracketingFailed);
}

TEST(IVBracketTest, ConfiguredVolatilityBoundsMustBeFinitePositiveAndOrdered) {
    auto table = certified_curve({8, 1, 0, 0});
    ASSERT_TRUE(table);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (auto [lo, hi] : std::array<std::pair<double, double>, 7>{
             {{0, .5}, {-.1, .5}, {.5, .5}, {.6, .5}, {nan, .5}, {.1, nan}, {.1, inf}}}) {
        InterpolatedIVSolverConfig config;
        config.sigma_min = lo;
        config.sigma_max = hi;
        auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(*table, config);
        ASSERT_FALSE(solver);
        EXPECT_EQ(solver.error().code, ValidationErrorCode::InvalidBounds);
    }
}

}  // namespace
