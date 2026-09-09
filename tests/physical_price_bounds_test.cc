// SPDX-License-Identifier: MIT
#include "mango/option/table/certification/physical_bounds.hpp"
#include <array>
#include <gtest/gtest.h>
#include <limits>
namespace {
using namespace mango;
using namespace mango::detail::certification;
TEST(PhysicalPriceBoundsTest, DecreasingPremiumCanHavePositivePhysicalVega) {
    const QuoteBox point{Interval(1), Interval(1), Interval(.2), Interval(0)};
    const PriceBounds raw{Interval(1), Interval(-10)};
    auto total = continuous_eep(raw, point, 100, OptionType::PUT, 0);
    ASSERT_TRUE(total.finite());
    EXPECT_GT(total.sigma_partial.lower_bound(), .29);
    EXPECT_LT(total.sigma_partial.upper_bound(), .30);
}
} // namespace

TEST(PhysicalPriceBoundsTest, FloorsAndMixturesDoNotInventLeafViolationWitnesses) {
    using namespace mango;
    using namespace mango::detail::certification;
    auto masked = zero_floor({Interval::hull(-2, -1), Interval::hull(-100, -50)});
    EXPECT_TRUE(masked.value.exact_zero());
    EXPECT_TRUE(masked.sigma_partial.exact_zero());
    const std::array<PriceBounds, 2> pieces{
        {{Interval(.1), Interval(-.1)}, {Interval(.2), Interval(.4)}}};
    const std::array<Interval, 2> weights{Interval(.25), Interval(.75)};
    auto combined = weighted_sum(pieces, weights);
    EXPECT_TRUE(combined.sigma_partial.strictly_positive());
    auto floor = intrinsic_floor({Interval(.1), Interval(-2)}, Interval(.5), OptionType::PUT);
    EXPECT_TRUE(floor.value.contains(.5));
    EXPECT_TRUE(floor.sigma_partial.exact_zero());
    auto crossing =
        intrinsic_floor({Interval::hull(.49, .51), Interval(-2)}, Interval(.5), OptionType::PUT);
    EXPECT_FALSE(crossing.sigma_partial.strictly_negative());
    EXPECT_TRUE(crossing.sigma_partial.contains(0));
}

TEST(PhysicalPriceBoundsTest, DimensionlessSigmaDerivativeIncludesBothCoordinates) {
    using namespace mango;
    using namespace mango::detail::certification;
    const QuoteBox point{Interval(1), Interval(1), Interval(1), Interval(.5)};
    // f(u,z)=u-z+.5 at u=.5,z=0 has f_sigma=1+2=3, not f_z=-1.
    auto total =
        dimensionless_eep(Interval(1), Interval(1), Interval(-1), point, 100, OptionType::PUT);
    ASSERT_TRUE(total.finite());
    EXPECT_GT(total.sigma_partial.lower_bound(), .271);
    EXPECT_LT(total.sigma_partial.upper_bound(), .273);
    // A large positive f_z makes the full physical expression decrease.
    auto negative =
        dimensionless_eep(Interval(1), Interval(0), Interval(20), point, 100, OptionType::PUT);
    EXPECT_TRUE(negative.sigma_partial.strictly_negative());
}

TEST(PhysicalPriceBoundsTest, EuropeanBoundsIncludeExpiryLimitAndNegativeRates) {
    using namespace mango;
    using namespace mango::detail::certification;
    auto expiry =
        european({Interval(.5), Interval(0), Interval(.2), Interval(-.05)}, OptionType::PUT, 0);
    EXPECT_TRUE(expiry.value.contains(.5));
    EXPECT_TRUE(expiry.sigma_partial.exact_zero());
    auto short_life = european({Interval(1), Interval::hull(0, .1), Interval(.2), Interval(.05)},
                               OptionType::CALL, 0);
    EXPECT_TRUE(short_life.finite());
    EXPECT_TRUE(short_life.value.nonnegative());
    EXPECT_TRUE(short_life.sigma_partial.nonnegative());
    EXPECT_TRUE(short_life.sigma_partial.contains(0));
    auto negative_rate =
        european({Interval(1), Interval(1), Interval(.2), Interval(-.05)}, OptionType::PUT, 0);
    EXPECT_GT(negative_rate.sigma_partial.lower_bound(), .394);
    EXPECT_LT(negative_rate.sigma_partial.upper_bound(), .395);
}

TEST(PhysicalPriceBoundsTest, OnlyInactiveBranchesMayIgnoreUnknownDerivativeBounds) {
    using namespace mango;
    using namespace mango::detail::certification;
    const Interval nan(std::numeric_limits<double>::quiet_NaN());
    EXPECT_FALSE(zero_floor({nan, Interval(0)}).finite());
    EXPECT_TRUE(zero_floor({Interval(-1), nan}).sigma_partial.exact_zero());
    const std::array<PriceBounds, 2> pieces{{{nan, nan}, {Interval(.1), Interval(.2)}}};
    EXPECT_TRUE(weighted_sum(pieces, std::array<Interval, 2>{Interval(0), Interval(1)}).finite());
    EXPECT_FALSE(
        weighted_sum(pieces, std::array<Interval, 2>{Interval(.5), Interval(.5)}).finite());
    EXPECT_FALSE(weighted_sum(pieces, std::array<Interval, 2>{Interval(-1), Interval(2)}).finite());
}

TEST(PhysicalPriceBoundsTest, OuterIntrinsicFloorNeedsMoreThanStrikeEndpointSigns) {
    using namespace mango;
    using namespace mango::detail::certification;
    const Interval ratio(.5); // put intrinsic/K=.5
    const std::array<PriceBounds,2> ends{{{Interval(1),Interval(0)},
                                        {Interval(.4),Interval(-1)}}};
    EXPECT_TRUE(intrinsic_floor(ends[0],ratio,OptionType::PUT).sigma_partial.nonnegative());
    EXPECT_TRUE(intrinsic_floor(ends[1],ratio,OptionType::PUT).sigma_partial.nonnegative());
    const std::array<Interval,2> weights{Interval(.5),Interval(.5)};
    // The interior blend lies above intrinsic and decreases with sigma,
    // although each separately projected endpoint is sigma-flat.
    auto interior=intrinsic_floor(weighted_sum(ends,weights),ratio,OptionType::PUT);
    EXPECT_TRUE(interior.sigma_partial.strictly_negative());
}
