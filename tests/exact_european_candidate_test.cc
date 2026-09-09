// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/exact_european_candidate.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace mango;

struct Inputs {
    BSplineND<double, 4>::GridArray grids{{
        {-.2, -.05, .05, .2}, {.1, .5, 1, 2},
        {.1, .2, .3, .5}, {0, .025, .05, .1}}};
    BSplineND<double, 4>::KnotArray knots;
    SurfaceBounds bounds{-.2, .2, .1, 2, .1, .5, 0, .1};
    Inputs() {
        for (size_t d = 0; d < grids.size(); ++d) {
            knots[d] = {grids[d].front(), grids[d].front(), grids[d].front(), grids[d].front(),
                        grids[d].back(), grids[d].back(), grids[d].back(), grids[d].back()};
        }
    }
};

TEST(ExactEuropeanCandidateTest, PublishesKnownCallAndRetainsExplicitCoordinates) {
    Inputs in;
    auto candidate = detail::make_exact_european_bspline_candidate(
        in.grids, in.knots, in.bounds, 100.0, OptionType::CALL, 0.0);
    ASSERT_TRUE(candidate);
    ASSERT_TRUE(*candidate);
    auto table = BSplinePriceTable::create(**candidate, in.bounds, OptionType::CALL, 0.0);
    ASSERT_TRUE(table);
    EXPECT_EQ(table->proof_status(), PriceProofStatus::Certified);
    EXPECT_NEAR(table->price(100, 100, 1, .2, .05), 10.450583572185565, 1e-12);
    EXPECT_NEAR(table->vega(100, 100, 1, .2, .05), 37.52403469169379, 1e-11);
    PricingParams params{OptionSpec{.spot=100, .strike=100, .maturity=1,
        .rate=.05, .option_type=OptionType::CALL}, .2};
    const auto delta = table->delta(params);
    const auto gamma = table->gamma(params);
    ASSERT_TRUE(delta);
    ASSERT_TRUE(gamma);
    EXPECT_NEAR(*delta, .6368306511756191, 1e-12);
    EXPECT_NEAR(*gamma, .018762017345846895, 1e-13);
    const auto& spline = (**candidate).leaf().interpolant().get();
    for (size_t d = 0; d < 4; ++d) {
        EXPECT_EQ(spline.grid(d), in.grids[d]);
        EXPECT_EQ(spline.knots(d), in.knots[d]);
    }
    auto solver = InterpolatedIVSolver<BSplinePriceTable>::create(*table);
    ASSERT_TRUE(solver);
    IVQuery query(OptionSpec{.spot=100, .strike=100, .maturity=1,
        .rate=.05, .option_type=OptionType::CALL}, 10.450583572185565);
    auto iv = solver->solve(query);
    ASSERT_TRUE(iv);
    EXPECT_NEAR(iv->implied_vol, .2, 1e-6);
}

TEST(ExactEuropeanCandidateTest, NeverAppliesToOtherExerciseModels) {
    Inputs in;
    auto ineligible = [&](OptionType type, double q,
                          std::optional<FixedExpiryMetadata> model = std::nullopt) {
        auto candidate = detail::make_exact_european_bspline_candidate(
            in.grids, in.knots, in.bounds, 100.0, type, q, model);
        ASSERT_TRUE(candidate);
        EXPECT_FALSE(*candidate);
    };
    ineligible(OptionType::PUT, 0.0);
    ineligible(OptionType::CALL, std::numeric_limits<double>::denorm_min());
    ineligible(OptionType::CALL, 0.0, FixedExpiryMetadata{2.0, {{1.0, 1.0}}});
    in.bounds.rate_min = -std::numeric_limits<double>::denorm_min();
    ineligible(OptionType::CALL, 0.0);
}

TEST(ExactEuropeanCandidateTest, RequestedDomainControlsEligibilityNotSupportPadding) {
    Inputs in;
    in.grids[3] = {-.05, 0.0, .05, .1};
    in.knots[3] = {-.05, -.05, -.05, -.05, .1, .1, .1, .1};
    auto candidate = detail::make_exact_european_bspline_candidate(
        in.grids, in.knots, in.bounds, 100.0, OptionType::CALL, 0.0);
    ASSERT_TRUE(candidate);
    ASSERT_TRUE(*candidate);
    auto table = BSplinePriceTable::create(**candidate, in.bounds, OptionType::CALL, 0.0);
    ASSERT_TRUE(table);
    EXPECT_DOUBLE_EQ(table->rate_min(), 0.0);
    EXPECT_EQ((**candidate).leaf().interpolant().get().knots(3), in.knots[3]);
}

TEST(ExactEuropeanCandidateTest, InvalidEligibleInputsReturnErrors) {
    Inputs in;
    auto bad_reference = detail::make_exact_european_bspline_candidate(
        in.grids, in.knots, in.bounds, 0.0, OptionType::CALL, 0.0);
    ASSERT_FALSE(bad_reference);
    EXPECT_EQ(bad_reference.error().code, PriceTableErrorCode::InvalidConfig);
    auto bad_model = detail::make_exact_european_bspline_candidate(
        in.grids, in.knots, in.bounds, 100.0, OptionType::CALL, 0.0,
        FixedExpiryMetadata{.5, {}});
    EXPECT_FALSE(bad_model);
    in.knots[0].pop_back();
    auto bad_knots = detail::make_exact_european_bspline_candidate(
        in.grids, in.knots, in.bounds, 100.0, OptionType::CALL, 0.0);
    EXPECT_FALSE(bad_knots);
}
}  // namespace
