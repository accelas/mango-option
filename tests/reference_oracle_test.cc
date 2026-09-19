// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_metrics.hpp"
#include "mango/option/grid_spec_types.hpp"
#include <cmath>

using namespace mango;

static PricingParams put_1y_with_divs() {
    PricingParams p;
    p.spot = 100.0; p.strike = 100.0; p.maturity = 1.0; p.rate = 0.05;
    p.dividend_yield = 0.02; p.option_type = OptionType::PUT; p.volatility = 0.2;
    p.discrete_dividends = {{0.25, 0.5}, {0.5, 0.5}, {0.75, 0.5}};
    return p;
}

// Spec D1: every level is odd, nested (every 2^k-th node), shares the middle
// index, and the fine grid does not depend on how many levels were asked for.
TEST(ReferenceGridFamily, LevelsAreNestedOddAndShareMiddleNode) {
    const auto acc = make_grid_accuracy(kReferenceAccuracy);
    auto fam3 = make_reference_grid_family(put_1y_with_divs(), acc, 3);
    ASSERT_TRUE(fam3.has_value());
    ASSERT_EQ(fam3->levels.size(), 4u);
    EXPECT_EQ(fam3->point_counts[0] % 16, 1u);
    EXPECT_FALSE(fam3->rounded_down);
    auto fine = fam3->levels[0].grid_spec.generate();
    auto fine_pts = fine.view().span();
    for (size_t k = 1; k <= 3; ++k) {
        // Keep the GridBuffer alive in a named variable: generate() returns
        // a temporary, and chaining .view().span() directly on it would
        // leave `pts` dangling once the temporary is destroyed.
        auto buf = fam3->levels[k].grid_spec.generate();
        auto pts = buf.view().span();
        EXPECT_EQ(pts.size() % 2, 1u) << "level " << k;
        ASSERT_EQ((fine_pts.size() - 1) >> k, pts.size() - 1) << "level " << k;
        for (size_t j = 0; j < pts.size(); ++j) {
            EXPECT_NEAR(pts[j], fine_pts[j << k], 1e-12 * (1.0 + std::abs(fine_pts[j << k])))
                << "level " << k << " node " << j;
        }
        EXPECT_DOUBLE_EQ(pts[(pts.size() - 1) / 2], fine_pts[(fine_pts.size() - 1) / 2]);
        EXPECT_EQ(fam3->levels[k].n_time, (fam3->levels[0].n_time + (1u << k) - 1) >> k);
        EXPECT_TRUE(fam3->levels[k].mandatory_times.empty());
    }
    auto fam1 = make_reference_grid_family(put_1y_with_divs(), acc, 1);
    ASSERT_TRUE(fam1.has_value());
    EXPECT_EQ(fam1->point_counts[0], fam3->point_counts[0]);
    EXPECT_EQ(fam1->levels[0].n_time, fam3->levels[0].n_time);
}

// The fine count never exceeds the profile's strict cap; below the cap it is
// the smallest n = 1 (mod 16) at or above the estimate.
TEST(ReferenceGridFamily, RoundsUpWithinCapElseDownAndFlags) {
    auto acc = make_grid_accuracy(kReferenceAccuracy);
    auto p = put_1y_with_divs();
    auto est = estimate_pde_grid(p, acc);
    ASSERT_TRUE(est.has_value());
    const size_t n0 = est->first.n_points();
    auto fam = make_reference_grid_family(p, acc, 1);
    ASSERT_TRUE(fam.has_value());
    EXPECT_GE(fam->point_counts[0], n0);
    EXPECT_LT(fam->point_counts[0], n0 + 16);
    // Force the cap right at the estimate: rounding up is impossible.
    acc.max_spatial_points = n0;
    acc.min_spatial_points = std::min(acc.min_spatial_points, n0);
    auto capped = make_reference_grid_family(p, acc, 1);
    ASSERT_TRUE(capped.has_value());
    EXPECT_LE(capped->point_counts[0], n0);
    EXPECT_EQ(capped->point_counts[0] % 16, 1u);
    EXPECT_TRUE(capped->rounded_down);
}

// The oracle rolls dividends onto a fixed-expiry contract exactly as
// make_validate_fn does, and solves on the grid it is handed.
TEST(ReferenceOracle, SolvesOnGivenGridAndMatchesValidateFn) {
    ReferenceOracle oracle{.dividend_yield = 0.02, .option_type = OptionType::PUT,
                           .discrete_dividends = {{0.25, 0.5}, {0.5, 0.5}, {0.75, 0.5}},
                           .reference_maturity = 1.0,
                           .accuracy = make_grid_accuracy(kReferenceAccuracy)};
    auto p = oracle.contract(100.0, 110.0, 0.4, 0.2, 0.05);
    auto fam = make_reference_grid_family(p, oracle.accuracy, 1);
    ASSERT_TRUE(fam.has_value());
    auto v = oracle.solve(p, fam->levels[0]);
    ASSERT_TRUE(v.has_value());
    auto validate = make_validate_fn(0.02, OptionType::PUT,
                                     {{0.25, 0.5}, {0.5, 0.5}, {0.75, 0.5}}, 1.0);
    auto ref = validate(100.0, 110.0, 0.4, 0.2, 0.05);
    ASSERT_TRUE(ref.has_value());
    // Same profile; the family adds at most 15 spatial points, so the two agree
    // far inside the High profile's own two-grid difference.
    EXPECT_NEAR(*v, *ref, 1e-4);
    auto half = oracle.solve(p, fam->levels[1]);
    ASSERT_TRUE(half.has_value());
    EXPECT_NE(*half, *v);
}
