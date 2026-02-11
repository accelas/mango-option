// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/split_surface.hpp"
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

TEST(MultiKRefSplitTest, ToLocalSetsStrikeToKRef) {
    MultiKRefSplit split({80.0, 100.0, 120.0});
    auto [ls, lk, lt, lv, lr] = split.to_local(1, 110.0, 95.0, 0.5, 0.20, 0.05);
    EXPECT_NEAR(lk, 100.0, 1e-12);      // strike -> K_ref[1]
    EXPECT_NEAR(ls, 110.0, 1e-12);      // spot unchanged
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
