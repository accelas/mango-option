// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"

using namespace mango;

// Mock inner that returns spot / strike + offset (easy to verify routing)
struct MockInner {
    double offset = 0.0;
    double price(double spot, double strike, double /*tau*/,
                 double /*sigma*/, double /*rate*/) const {
        return spot / strike + offset;
    }
    double vega(double /*spot*/, double /*strike*/, double /*tau*/,
                double /*sigma*/, double /*rate*/) const {
        return 0.1 + offset;
    }
};

// --- TauSegmentSplit tests ---

TEST(TauSegmentSplitTest, RoutesToCorrectSegment) {
    TauSegmentSplit split({0.0, 0.25, 0.50}, {0.25, 0.50, 1.0},
                          {0.0, 0.0, 0.0}, {0.25, 0.25, 0.50}, 100.0);

    auto br = split.bracket(100, 100, 0.10, 0.20, 0.05);  // tau=0.10 -> segment 0
    EXPECT_EQ(br.count, 1u);
    EXPECT_EQ(br.entries[0].index, 0u);
    EXPECT_NEAR(br.entries[0].weight, 1.0, 1e-12);

    auto br2 = split.bracket(100, 100, 0.30, 0.20, 0.05);  // tau=0.30 -> segment 1
    EXPECT_EQ(br2.entries[0].index, 1u);

    auto br3 = split.bracket(100, 100, 0.75, 0.20, 0.05);  // tau=0.75 -> segment 2
    EXPECT_EQ(br3.entries[0].index, 2u);
}

TEST(TauSegmentSplitTest, ToLocalRemapsTauAndStrike) {
    TauSegmentSplit split({0.0, 0.25}, {0.25, 0.50},
                          {0.0, 0.0}, {0.24, 0.24}, 100.0);

    auto [ls, lk, lt, lv, lr] = split.to_local(1, 110.0, 95.0, 0.30, 0.20, 0.05);
    EXPECT_NEAR(lt, 0.05, 1e-12);       // 0.30 - 0.25 = 0.05
    EXPECT_NEAR(lk, 100.0, 1e-12);      // strike -> K_ref
    EXPECT_NEAR(ls, 110.0, 1e-12);      // spot unchanged
    EXPECT_NEAR(lv, 0.20, 1e-12);       // sigma unchanged
    EXPECT_NEAR(lr, 0.05, 1e-12);       // rate unchanged
}

TEST(TauSegmentSplitTest, NormalizeMultipliesByKRef) {
    TauSegmentSplit split({0.0}, {1.0}, {0.0}, {1.0}, 100.0);
    EXPECT_NEAR(split.normalize(0, 95.0, 0.5), 50.0, 1e-12);  // 0.5 * 100
}

TEST(TauSegmentSplitTest, DenormalizeIsIdentity) {
    TauSegmentSplit split({0.0}, {1.0}, {0.0}, {1.0}, 100.0);
    EXPECT_NEAR(split.denormalize(42.0, 100, 100, 0.5, 0.2, 0.05), 42.0, 1e-12);
}

// --- MultiKRefSplit tests ---

TEST(MultiKRefSplitTest, BracketsCorrectly) {
    MultiKRefSplit split({80.0, 100.0, 120.0});

    // Below first -> clamp to index 0
    auto br = split.bracket(100, 70.0, 0.5, 0.20, 0.05);
    EXPECT_EQ(br.count, 1u);
    EXPECT_EQ(br.entries[0].index, 0u);

    // Between 80 and 100 -> interpolate
    auto br2 = split.bracket(100, 90.0, 0.5, 0.20, 0.05);
    EXPECT_EQ(br2.count, 2u);
    EXPECT_EQ(br2.entries[0].index, 0u);  // K_ref=80
    EXPECT_EQ(br2.entries[1].index, 1u);  // K_ref=100
    EXPECT_NEAR(br2.entries[0].weight, 0.5, 1e-12);
    EXPECT_NEAR(br2.entries[1].weight, 0.5, 1e-12);

    // Above last -> clamp to last
    auto br3 = split.bracket(100, 130.0, 0.5, 0.20, 0.05);
    EXPECT_EQ(br3.count, 1u);
    EXPECT_EQ(br3.entries[0].index, 2u);
}

TEST(MultiKRefSplitTest, ToLocalPreservesMoneyness) {
    MultiKRefSplit split({80.0, 100.0, 120.0});
    auto [ls, lk, lt, lv, lr] = split.to_local(1, 110.0, 95.0, 0.5, 0.20, 0.05);
    EXPECT_NEAR(lk, 100.0, 1e-12);      // strike -> K_ref[1]
    EXPECT_NEAR(ls / lk, 110.0 / 95.0, 1e-12);  // S/K unchanged
}

TEST(MultiKRefSplitTest, NormalizeDividesByKRef) {
    MultiKRefSplit split({80.0, 100.0, 120.0});
    EXPECT_NEAR(split.normalize(1, 95.0, 50.0), 0.5, 1e-12);  // 50.0 / 100.0
}

TEST(MultiKRefSplitTest, DenormalizeMultipliesByStrike) {
    MultiKRefSplit split({80.0, 100.0, 120.0});
    EXPECT_NEAR(split.denormalize(0.5, 100, 95.0, 0.5, 0.2, 0.05), 47.5, 1e-12);
}

// --- SplitSurface integration ---

TEST(SplitSurfaceTest, SingleSegmentPassesThrough) {
    TauSegmentSplit split({0.0}, {1.0}, {0.0}, {1.0}, 100.0);
    MockInner inner{.offset = 0.0};
    SplitSurface<MockInner, TauSegmentSplit> surface({std::move(inner)}, std::move(split));

    // MockInner returns spot/strike, to_local sets strike=K_ref=100
    // MockInner sees spot=110, strike=100 -> 110/100 = 1.1
    // normalize: 1.1 * 100 = 110
    // denormalize: identity -> 110
    double p = surface.price(110.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_NEAR(p, 110.0, 1e-12);
}

// An exact homogeneous price function isolates reference-coordinate mapping
// from PDE samples and table fitting. Every reference represents the same
// function of the original physical query after normalization.
struct SmoothHomogeneousPrice {
    double price(double spot, double strike, double tau, double sigma, double rate) const {
        return spot * spot / strike + strike * (2.0 * sigma + 3.0 * rate + 4.0 * tau);
    }
    double vega(double, double strike, double, double, double) const { return 2.0 * strike; }
    std::expected<double, GreekError> greek(Greek g, const PricingParams& p) const {
        switch (g) {
            case Greek::Delta: return 2.0 * p.spot / p.strike;
            case Greek::Vega: return 2.0 * p.strike;
            case Greek::Theta: return -4.0 * p.strike;
            case Greek::Rho: return 3.0 * p.strike;
        }
        return std::unexpected(GreekError::NumericalFailure);
    }
    std::expected<double, GreekError> gamma(const PricingParams& p) const {
        return 2.0 / p.strike;
    }
};

TEST(SplitSurfaceTest, MultiKRefPreservesMoneynessWhileSpotAndStrikeVary) {
    SplitSurface<SmoothHomogeneousPrice, MultiKRefSplit> surface(
        std::vector<SmoothHomogeneousPrice>(3), MultiKRefSplit({80.0, 100.0, 120.0}));
    for (double strike : {80.0, 90.0, 100.0, 110.0, 120.0}) {
        for (double m : {0.8, 1.0, 1.2}) {
            const double expected = strike * (m * m + 2.0 * 0.2 + 3.0 * 0.05 + 4.0 * 0.5);
            EXPECT_NEAR(surface.price(m * strike, strike, 0.5, 0.2, 0.05), expected, 2e-12)
                << "strike=" << strike << " S/K=" << m;
        }
    }
}

TEST(SplitSurfaceTest, MultiKRefGreeksFollowTheSpotMapChainRule) {
    SplitSurface<SmoothHomogeneousPrice, MultiKRefSplit> surface(
        std::vector<SmoothHomogeneousPrice>(3), MultiKRefSplit({80.0, 100.0, 120.0}));
    for (double strike : {80.0, 90.0, 100.0, 110.0, 120.0}) {
        for (double m : {0.8, 1.0, 1.2}) {
            PricingParams p(OptionSpec{.spot = m * strike, .strike = strike,
                .maturity = 0.5, .rate = 0.05, .option_type = OptionType::CALL}, 0.2);
            auto delta = surface.greek(Greek::Delta, p);
            auto gamma = surface.gamma(p);
            ASSERT_TRUE(delta.has_value());
            ASSERT_TRUE(gamma.has_value());
            EXPECT_NEAR(*delta, 2.0 * m, 1e-12);
            EXPECT_NEAR(*gamma, 2.0 / strike, 1e-14);
            const double h = 0.1;
            const double mid = surface.price(p.spot, strike, 0.5, 0.2, 0.05);
            const double up = surface.price(p.spot + h, strike, 0.5, 0.2, 0.05);
            const double dn = surface.price(p.spot - h, strike, 0.5, 0.2, 0.05);
            EXPECT_NEAR(*delta, (up - dn) / (2 * h), 1e-10);
            EXPECT_NEAR(*gamma, (up - 2 * mid + dn) / (h * h), 1e-10);
            EXPECT_NEAR(surface.vega(p.spot, strike, 0.5, 0.2, 0.05), 2.0 * strike, 1e-12);
            EXPECT_NEAR(*surface.greek(Greek::Vega, p), 2.0 * strike, 1e-12);
            EXPECT_NEAR(*surface.greek(Greek::Theta, p), -4.0 * strike, 1e-12);
            EXPECT_NEAR(*surface.greek(Greek::Rho, p), 3.0 * strike, 1e-12);
        }
    }
}

TEST(SplitSurfaceTest, ExactReferenceDoesNotReadAnInactiveNeighbor) {
    SplitSurface<MockInner, MultiKRefSplit> surface(
        {MockInner{.offset = std::numeric_limits<double>::quiet_NaN()}, MockInner{}, MockInner{}},
        MultiKRefSplit({80.0, 100.0, 120.0}));
    // At K=100 only that exact reference contributes. An invalid value at
    // the zero-weight K=80 neighbor must not contaminate the result.
    EXPECT_DOUBLE_EQ(surface.price(100.0, 100.0, 0.5, 0.2, 0.05), 1.0);
}

TEST(SplitSurfaceTest, SegmentedCheckedQueriesRequireStrikeMetadata) {
    using Multi = SplitSurface<SmoothHomogeneousPrice, MultiKRefSplit>;
    SurfaceBounds bounds{-.3, .3, .1, 1.0, .1, .4, .01, .1};
    PriceTable<Multi> missing(Multi(std::vector<SmoothHomogeneousPrice>(3),
        MultiKRefSplit({80.0, 100.0, 120.0})), bounds, OptionType::CALL, 0.0);
    EXPECT_FALSE(missing.contains_strike(100.0));
    bounds.strike_bounds = StrikeBounds{90.0, 110.0};
    PriceTable<Multi> published(Multi(std::vector<SmoothHomogeneousPrice>(3),
        MultiKRefSplit({80.0, 100.0, 120.0})), bounds, OptionType::CALL, 0.0);
    EXPECT_TRUE(published.contains_strike(90.0));
    EXPECT_TRUE(published.contains_strike(110.0));
    EXPECT_FALSE(published.contains_strike(80.0));  // extra support is not publication
    EXPECT_FALSE(published.contains_strike(120.0));
    bounds.strike_bounds.reset();
    PriceTable<SmoothHomogeneousPrice> continuous({}, bounds, OptionType::CALL, 0.0);
    EXPECT_TRUE(continuous.contains_strike(1000.0));
}

TEST(SplitSurfaceTest, RequestedRatiosAndPositiveMaturityDefineAdmission) {
    SurfaceBounds bounds{std::log(.1), std::log(.13), 0, 1, .1, .4, .01, .1};
    bounds.ratio_bounds = MoneynessBounds{.1, .13};
    PriceTable<SmoothHomogeneousPrice> table({}, bounds, OptionType::CALL, 0.0);
    EXPECT_TRUE(table.contains_moneyness(10, 100));
    EXPECT_EQ(table.ratio_bounds().min, .1);
    EXPECT_FALSE(table.contains_maturity(0));
    EXPECT_TRUE(table.contains_maturity(std::nextafter(0., 1.)));
}
