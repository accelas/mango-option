// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_metrics.hpp"
#include "mango/option/grid_spec_types.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

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

// Spec D8 (rev 6): `refine_grid_config` runs the family's nesting argument
// upward.  The refined grid has factor*(n-1)+1 points and factor*n_time
// steps, and the input grid is exactly its every-factor-th-node
// subsequence -- the mirror of LevelsAreNestedOddAndShareMiddleNode.  The
// profile's point cap does not apply: 2G and 4G above a High fine grid
// exceed max_spatial_points by design.
TEST(RefineGridConfig, RefinementsContainTheInputAsEveryFactorthNode) {
    const auto acc = make_grid_accuracy(kReferenceAccuracy);
    auto fam = make_reference_grid_family(put_1y_with_divs(), acc, 1);
    ASSERT_TRUE(fam.has_value());
    const PDEGridConfig& g = fam->levels[0];
    auto g_buf = g.grid_spec.generate();
    auto g_pts = g_buf.view().span();

    for (size_t factor : {2u, 4u}) {
        auto refined = refine_grid_config(g, factor);
        ASSERT_TRUE(refined.has_value()) << "factor " << factor;
        // Keep the GridBuffer alive in a named variable: generate() returns
        // a temporary, and chaining .view().span() on it would dangle.
        auto buf = refined->grid_spec.generate();
        auto pts = buf.view().span();
        ASSERT_EQ(pts.size(), factor * (g_pts.size() - 1) + 1) << "factor " << factor;
        EXPECT_EQ(refined->n_time, g.n_time * factor) << "factor " << factor;
        EXPECT_EQ(refined->mandatory_times, g.mandatory_times) << "factor " << factor;
        for (size_t j = 0; j < g_pts.size(); ++j) {
            EXPECT_NEAR(pts[j * factor], g_pts[j], 1e-12 * (1.0 + std::abs(g_pts[j])))
                << "factor " << factor << " node " << j;
        }
        EXPECT_GT(pts.size(), acc.max_spatial_points) << "factor " << factor;
    }
    EXPECT_FALSE(refine_grid_config(g, 0).has_value());
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

// ===========================================================================
// Stencil references and resolution (spec D1 / D2)
// ===========================================================================

// A fake oracle: price(sigma) = base + slope*(sigma-0.2) on the fine grid,
// plus `coarse_bias` on any grid whose point count is below the fine count.
struct FakeStencil {
    double slope = 40.0, coarse_bias = 1e-4;
    std::vector<std::pair<size_t, size_t>> calls;  // (n_points, n_time)
    StencilSolveFn fn() {
        return [this](const PricingParams& p, const PDEGridConfig& g) -> std::expected<double, SolverError> {
            calls.emplace_back(g.grid_spec.n_points(), g.n_time);
            const bool coarse = g.grid_spec.n_points() < calls.front().first;
            return 10.0 + slope * (p.volatility - 0.2) + (coarse ? coarse_bias : 0.0);
        };
    }
};

static ReferenceOracle plain_oracle() {
    return ReferenceOracle{.dividend_yield = 0.0, .option_type = OptionType::PUT,
                           .discrete_dividends = {}, .reference_maturity = std::nullopt,
                           .accuracy = make_grid_accuracy(kReferenceAccuracy)};
}

// Spec D1/L2: six solves, three identical fine configs, three identical coarse
// configs, coarse = every-other-node of fine.
TEST(StencilRefs, SixSolvesOnOneNestedGridPair) {
    FakeStencil fake;
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto counter = std::make_shared<ReferenceSolveCounter>();
    auto prep = make_stencil_refs_fn(params, plain_oracle(), counter, fake.fn());
    auto refs = prep(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    ASSERT_EQ(fake.calls.size(), 6u);
    EXPECT_EQ(fake.calls[0], fake.calls[2]); EXPECT_EQ(fake.calls[0], fake.calls[4]);
    EXPECT_EQ(fake.calls[1], fake.calls[3]); EXPECT_EQ(fake.calls[1], fake.calls[5]);
    EXPECT_EQ(fake.calls[1].first, (fake.calls[0].first - 1) / 2 + 1);
    EXPECT_EQ(counter->fine_attempts.load(), 3u);
    EXPECT_EQ(counter->coarse_attempts.load(), 3u);
    EXPECT_TRUE(refs->resolved);
    EXPECT_DOUBLE_EQ(refs->sigma_lo, 0.2 - 5e-4);
    EXPECT_DOUBLE_EQ(refs->sigma_hi, 0.2 + 5e-4);
    // delta = max(F_s * |bias| / (2^p - 1), kReferenceUncertaintyFloor * K)
    // -- here the two-grid term (3e-4) dominates the floor (1e-5).
    EXPECT_NEAR(refs->delta,
                std::max(kRichardsonSafetyFactor * 1e-4
                             / (std::pow(2.0, kReferenceConvergenceOrder) - 1.0),
                         kReferenceUncertaintyFloor * 100.0),
                1e-15);
    // Achieved time steps are recorded for both levels (record only).
    EXPECT_EQ(refs->fine_steps, static_cast<uint32_t>(fake.calls[0].second));
    EXPECT_EQ(refs->coarse_steps, static_cast<uint32_t>(fake.calls[1].second));
}

// Spec D1, rev 5: where both discretizations agree exactly -- both on the
// obstacle at a near-intrinsic point -- the two-grid difference sees nothing,
// and the estimate falls back to the oracle's calibrated accuracy scale.
TEST(StencilRefs, ZeroCoarseBiasFloorsTheUncertaintyEstimate) {
    FakeStencil fake{.slope = 40.0, .coarse_bias = 0.0};
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto prep = make_stencil_refs_fn(params, plain_oracle(),
                                     std::make_shared<ReferenceSolveCounter>(), fake.fn());
    auto refs = prep(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    // Two-grid term is exactly zero, so each estimate is the floor times the
    // strike of the contract solved.
    EXPECT_DOUBLE_EQ(refs->delta, kReferenceUncertaintyFloor * 100.0);
    EXPECT_DOUBLE_EQ(refs->delta_lo, kReferenceUncertaintyFloor * 100.0);
    EXPECT_DOUBLE_EQ(refs->delta_hi, kReferenceUncertaintyFloor * 100.0);
    // A slope of 40 separates the bracket by 0.02 per side, far above the
    // 2e-5 the floored estimates now require.
    EXPECT_TRUE(refs->resolved);
}

// Rev 5: the floor is what stops a bracket separated by microdollars from
// "resolving".  Slope 0.02 moves the price by 1e-5 per side over
// sigma0 +- 5e-4, which is below delta + delta_lo = 2e-5.
TEST(StencilRefs, SeparationBelowTwiceTheFloorIsUnresolved) {
    FakeStencil fake{.slope = 0.02, .coarse_bias = 0.0};
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto prep = make_stencil_refs_fn(params, plain_oracle(),
                                     std::make_shared<ReferenceSolveCounter>(), fake.fn());
    auto refs = prep(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    EXPECT_DOUBLE_EQ(refs->delta, kReferenceUncertaintyFloor * 100.0);
    EXPECT_NEAR(refs->ref_price - refs->bracket_lo_price, 1e-5, 1e-15);
    EXPECT_LT(refs->ref_price - refs->bracket_lo_price,
              refs->delta + refs->delta_lo);
    EXPECT_FALSE(refs->resolved);
    // The base price is still present: preparation succeeded, resolution did not.
    EXPECT_TRUE(std::isfinite(refs->ref_price));
}

// Spec D2, reviewer example: y=10, lo=9.85, hi=10.15, delta=0.10 at every
// point -> intervals overlap -> unresolved.
TEST(StencilRefs, OverlappingEndpointIntervalsAreUnresolved) {
    ErrorRefs r{.ref_price = 10.0, .bracket_lo_price = 9.85, .bracket_hi_price = 10.15,
                .sigma_lo = 0.1, .sigma_hi = 0.3, .delta = 0.10, .delta_lo = 0.10, .delta_hi = 0.10};
    EXPECT_FALSE(stencil_resolved(r));
    r.delta = r.delta_lo = r.delta_hi = 0.07;   // 10-0.07 > 9.85+0.07 and 10.15-0.07 > 10+0.07
    EXPECT_TRUE(stencil_resolved(r));
    r.bracket_lo_price = 10.02;                  // reversed ordering
    EXPECT_FALSE(stencil_resolved(r));
}

// Flat reference (exercise region): lo == y == hi -> unresolved even with zero delta.
TEST(StencilRefs, FlatReferenceIsUnresolved) {
    FakeStencil fake; fake.slope = 0.0; fake.coarse_bias = 0.0;
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto prep = make_stencil_refs_fn(params, plain_oracle(), std::make_shared<ReferenceSolveCounter>(), fake.fn());
    auto refs = prep(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    EXPECT_FALSE(refs->resolved);
    EXPECT_DOUBLE_EQ(refs->ref_price, 10.0);   // base price still present (partial stencil contract)
}

// sigma0 - tau <= 0 -> unresolved with the base price present, one fine solve only.
TEST(StencilRefs, SigmaBelowToleranceIsUnresolvedWithBase) {
    FakeStencil fake;
    AdaptiveGridParams params; params.target_iv_error = 0.5;
    auto counter = std::make_shared<ReferenceSolveCounter>();
    auto prep = make_stencil_refs_fn(params, plain_oracle(), counter, fake.fn());
    auto refs = prep(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    EXPECT_FALSE(refs->resolved);
    EXPECT_TRUE(std::isnan(refs->bracket_lo_price));
    EXPECT_EQ(counter->fine_attempts.load(), 1u);
}

// A failed bracket solve -> unresolved (base present); a failed base solve -> unexpected.
TEST(StencilRefs, PartialAndBaseFailures) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    size_t n = 0;
    StencilSolveFn fail_third = [&](const PricingParams&, const PDEGridConfig&) -> std::expected<double, SolverError> {
        if (++n == 3) return std::unexpected(SolverError{});
        return 10.0;
    };
    auto counter = std::make_shared<ReferenceSolveCounter>();
    auto prep = make_stencil_refs_fn(params, plain_oracle(), counter, fail_third);
    auto refs = prep(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value()); EXPECT_FALSE(refs->resolved);
    EXPECT_EQ(counter->fine_failures.load() + counter->coarse_failures.load(), 1u);
    // The stencil stops at the first failed solve: the point is unresolved
    // whatever the remaining three would have returned, so they are not run.
    EXPECT_EQ(counter->fine_attempts.load() + counter->coarse_attempts.load(), 3u);
    // The base failure propagates the solver's own code, not a default one.
    StencilSolveFn fail_first = [](const PricingParams&, const PDEGridConfig&) -> std::expected<double, SolverError> {
        return std::unexpected(SolverError{SolverErrorCode::ConvergenceFailure}); };
    auto counter2 = std::make_shared<ReferenceSolveCounter>();
    auto prep2 = make_stencil_refs_fn(params, plain_oracle(), counter2, fail_first);
    auto base_failed = prep2(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_FALSE(base_failed.has_value());
    EXPECT_EQ(base_failed.error().code, SolverErrorCode::ConvergenceFailure);
    // The base solve is the only one attempted: nothing follows an invalid point.
    EXPECT_EQ(counter2->fine_attempts.load(), 1u);
    EXPECT_EQ(counter2->fine_failures.load(), 1u);
    EXPECT_EQ(counter2->coarse_attempts.load(), 0u);

    // A non-finite base price is a failure of its own kind.
    StencilSolveFn nan_first = [](const PricingParams&, const PDEGridConfig&) -> std::expected<double, SolverError> {
        return std::numeric_limits<double>::quiet_NaN(); };
    auto prep3 = make_stencil_refs_fn(params, plain_oracle(),
                                      std::make_shared<ReferenceSolveCounter>(), nan_first);
    auto nan_base = prep3(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_FALSE(nan_base.has_value());
    EXPECT_EQ(nan_base.error().code, SolverErrorCode::NonFiniteSolution);
}

// A flat near-cap stencil is unresolved: the fake returns the same fine price
// at all three sigmas, so lo == y == hi and the separation inequalities reject
// the point before target validity is ever consulted.  The upper-limit rule
// itself is pinned by SeparatedStencilStillFailsOnUpperBoundTarget below.
TEST(StencilRefs, FlatNearCapStencilIsUnresolved) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    size_t n = 0;
    // Fine solves return 99.99 for a call on S = 100, coarse solves 99.5.
    StencilSolveFn near_cap = [&](const PricingParams& p, const PDEGridConfig& g) -> std::expected<double, SolverError> {
        (void)p; (void)g; return (++n % 2 == 1) ? 99.99 : 99.5; };
    auto oracle = plain_oracle(); oracle.option_type = OptionType::CALL;
    auto prep = make_stencil_refs_fn(params, oracle, std::make_shared<ReferenceSolveCounter>(), near_cap);
    auto refs = prep(100.0, 90.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    EXPECT_FALSE(refs->resolved);
}

// Same rule, isolated: the D2 inequalities pass and the ONLY thing that fails
// is the upper no-arbitrage bound on the target y + delta.  Fine prices are
// lo = 99.00, y = 99.99, hi = 100.50 on a call with S = 100; every coarse
// solve sits `kDiff` below its fine partner so all three estimates are 0.02.
// Then y - d = 99.97 > lo + d_lo = 99.02 and hi - d_hi = 100.48 > y + d =
// 100.01, so the point is separated -- but y + d = 100.01 exceeds the call's
// upper bound (spot), so the target set is not priceable and the point is
// unresolved.  (hi itself is never a validated target, only y and y +- d.)
TEST(StencilRefs, SeparatedStencilStillFailsOnUpperBoundTarget) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    const double kDelta = 0.02;
    const double kDiff = kDelta * (std::pow(2.0, kReferenceConvergenceOrder) - 1.0)
                       / kRichardsonSafetyFactor;
    const double fine[3] = {99.99, 99.00, 100.50};  // y, lo, hi in solve order
    size_t n = 0;
    StencilSolveFn stencil = [&](const PricingParams&, const PDEGridConfig&)
        -> std::expected<double, SolverError> {
        const size_t i = n++;
        const double base = fine[i / 2];
        return (i % 2 == 0) ? base : base - kDiff;  // fine, then its coarse partner
    };
    auto oracle = plain_oracle(); oracle.option_type = OptionType::CALL;
    auto prep = make_stencil_refs_fn(params, oracle, std::make_shared<ReferenceSolveCounter>(), stencil);
    auto refs = prep(100.0, 90.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    EXPECT_EQ(n, 6u);
    EXPECT_NEAR(refs->delta, kDelta, 1e-12);
    // The separation inequalities alone admit this stencil ...
    EXPECT_TRUE(stencil_resolved(*refs));
    // ... but the upper-bound check on y + delta does not.
    EXPECT_FALSE(refs->resolved);
    // The complement: the same stencil with a tenth of the two-grid
    // difference keeps the separation and puts y + delta back under spot.
    const double kSmallDiff = kDiff / 10.0;
    n = 0;
    StencilSolveFn tight = [&](const PricingParams&, const PDEGridConfig&)
        -> std::expected<double, SolverError> {
        const size_t i = n++;
        const double base = fine[i / 2];
        return (i % 2 == 0) ? base : base - kSmallDiff;
    };
    auto prep_tight = make_stencil_refs_fn(params, oracle, std::make_shared<ReferenceSolveCounter>(), tight);
    auto tight_refs = prep_tight(100.0, 90.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(tight_refs.has_value());
    EXPECT_NEAR(tight_refs->delta, kDelta / 10.0, 1e-12);
    EXPECT_TRUE(tight_refs->resolved);
}

// Spec D9: the put's upper limit is `K` times the largest discount factor
// over [0, T], which exceeds `K` once the rate is negative.  A deep-ITM put
// on S = 1, K = 100 at r = -2 % for one year may be worth up to 100 *
// e^{0.02} = 102.0201.  Fine prices lo = 101.00, y = 102.01, hi = 102.50 with
// every coarse solve `kDiff` below its fine partner give three 0.02
// estimates: the stencil separates (y - d = 101.99 > lo + d_lo = 101.02, and
// hi - d_hi = 102.48 > y + d = 102.03), but y + d = 102.03 is above the
// limit, so the target set is not priceable and the point is unresolved.
//
// The companion above (SeparatedStencilStillFailsOnUpperBoundTarget) pins the
// same rule for a call at its own limit, the spot.
TEST(StencilRefs, NegativeRatePutUpperBoundRejectsUpperTarget) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    const double kDelta = 0.02;
    const double kDiff = kDelta * (std::pow(2.0, kReferenceConvergenceOrder) - 1.0)
                       / kRichardsonSafetyFactor;
    const double fine[3] = {102.01, 101.00, 102.50};  // y, lo, hi in solve order
    size_t n = 0;
    StencilSolveFn stencil = [&](const PricingParams&, const PDEGridConfig&)
        -> std::expected<double, SolverError> {
        const size_t i = n++;
        const double base = fine[i / 2];
        return (i % 2 == 0) ? base : base - kDiff;  // fine, then its coarse partner
    };
    auto prep = make_stencil_refs_fn(
        params, plain_oracle(), std::make_shared<ReferenceSolveCounter>(), stencil);
    auto refs = prep(1.0, 100.0, 1.0, 0.2, -0.02);
    ASSERT_TRUE(refs.has_value());
    EXPECT_NEAR(refs->delta, kDelta, 1e-12);
    // The separation inequalities alone admit this stencil ...
    EXPECT_TRUE(stencil_resolved(*refs));
    // ... but y + delta is 102.03 against a limit of 102.0201.
    EXPECT_FALSE(refs->resolved);

    // The complement: a tenth of the two-grid difference keeps the
    // separation and puts y + delta back under the limit.
    const double kSmallDiff = kDiff / 10.0;
    n = 0;
    StencilSolveFn tight = [&](const PricingParams&, const PDEGridConfig&)
        -> std::expected<double, SolverError> {
        const size_t i = n++;
        const double base = fine[i / 2];
        return (i % 2 == 0) ? base : base - kSmallDiff;
    };
    auto prep_tight = make_stencil_refs_fn(
        params, plain_oracle(), std::make_shared<ReferenceSolveCounter>(), tight);
    auto tight_refs = prep_tight(1.0, 100.0, 1.0, 0.2, -0.02);
    ASSERT_TRUE(tight_refs.has_value());
    EXPECT_NEAR(tight_refs->delta, kDelta / 10.0, 1e-12);
    EXPECT_TRUE(tight_refs->resolved);
}

// ===========================================================================
// Round-trip scorer (spec D3)
// ===========================================================================

static RefinementContext score_ctx() {
    return RefinementContext{
        .spot = 100.0, .dividend_yield = 0.0, .option_type = OptionType::PUT,
        .bounds = {.m_min = -0.5, .m_max = 0.5, .tau_min = 0.05, .tau_max = 2.0,
                   .sigma_min = 0.05, .sigma_max = 0.6, .rate_min = 0.0, .rate_max = 0.1},
        .sample_bounds = {.m_min = -0.3, .m_max = 0.3, .tau_min = 0.1, .tau_max = 1.0,
                          .sigma_min = 0.1, .sigma_max = 0.5, .rate_min = 0.01, .rate_max = 0.09}};
}

static ErrorRefs resolved_refs(double y) {
    return ErrorRefs{.ref_price = y, .bracket_lo_price = y - 0.02, .bracket_hi_price = y + 0.02,
                     .sigma_lo = 0.2995, .sigma_hi = 0.3005,
                     .delta = 1e-4, .delta_lo = 1e-4, .delta_hi = 1e-4, .resolved = true};
}

// Surface: price = 10 + 40*(sigma-0.3) + bias; vega = 40.
static SurfaceHandle linear_handle(double bias) {
    return SurfaceHandle{
        .price = [bias](double, double, double, double s, double) { return 10.0 + 40.0 * (s - 0.3) + bias; },
        .vega = [](double, double, double, double, double) { return 40.0; }};
}

TEST(RoundTripScore, MeasuresBiasAsSigmaDistanceOverThreeTargets) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    // bias 0.04 -> root at 0.3 - 0.001; targets y +- 1e-4 add +- 2.5e-6.
    auto s = score(linear_handle(0.04), resolved_refs(10.0), 100.0, 100.0, 0.5, 0.3, 0.05);
    EXPECT_EQ(s.status, PointStatus::Measured);
    EXPECT_NEAR(s.iv_error, 0.001 + 2.5e-6, 1e-7);
    EXPECT_NEAR(s.price_residual, 0.04 / 100.0, 1e-12);
    EXPECT_FALSE(s.edge_band_rescue);
}

TEST(RoundTripScore, UnresolvedReferenceStillRecordsResidual) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    auto refs = resolved_refs(10.0); refs.resolved = false;
    auto s = score(linear_handle(0.04), refs, 100.0, 100.0, 0.5, 0.3, 0.05);
    EXPECT_EQ(s.status, PointStatus::ReferenceUnresolved);
    EXPECT_NEAR(s.price_residual, 4e-4, 1e-12);
    EXPECT_TRUE(std::isnan(s.iv_error));
}

// Spec D3, rev 5: acceptance runs on the published range widened by tau_iv.
// A root within the user's own tolerance beyond a published edge is a
// measurement; the exact product bracket's refusal becomes the diagnostic.
TEST(RoundTripScore, EdgeMissWithinToleranceMeasuresAndFlagsExactBracket) {
    AdaptiveGridParams params; params.target_iv_error = 5e-3;   // band 50 bps
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    // Surface overprices by 0.04 at sigma=0.1 -> root at 0.099, i.e. 10 bps
    // below the published edge and inside the 50 bps band.
    auto refs = resolved_refs(10.0 + 40.0 * (0.1 - 0.3));
    auto s = score(linear_handle(0.04), refs, 100.0, 100.0, 0.5, 0.1, 0.05);
    EXPECT_EQ(s.status, PointStatus::Measured);
    EXPECT_NEAR(s.iv_error, 1e-3 + 2.5e-6, 1e-7);
    EXPECT_TRUE(s.edge_band_rescue)
        << "the recovered sigma lies outside the un-widened product bracket";
}

// Beyond the tolerance band the outcome is what the shipped inversion
// reports: no root.  The diagnostic is only ever set on a Measured point.
TEST(RoundTripScore, EdgeMissBeyondToleranceIsNoRootWithoutFlag) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;   // band 5 bps
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    auto refs = resolved_refs(10.0 + 40.0 * (0.1 - 0.3));
    auto s = score(linear_handle(0.04), refs, 100.0, 100.0, 0.5, 0.1, 0.05);
    EXPECT_EQ(s.status, PointStatus::SurfaceNoRoot);
    EXPECT_FALSE(s.edge_band_rescue);
    EXPECT_TRUE(std::isnan(s.iv_error));
}

// Spec D3, rev 5: the acceptance band behaves the same on every backend.
// A B-spline fits exactly the published sigma range, so `fit.sigma ==
// sample.sigma`; without the edge extension the band would be clipped back
// to the published range and never reach a root just outside it.
static RefinementContext bspline_like_ctx() {
    RefinementContext c = score_ctx();
    c.bounds.sigma_min = c.sample_bounds.sigma_min;   // no sigma headroom
    c.bounds.sigma_max = c.sample_bounds.sigma_max;
    return c;
}

TEST(RoundTripScore, EdgeExtensionMeasuresWithoutFitDomainHeadroom) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;  // band 5 bps
    auto score = make_round_trip_score_fn(params, bspline_like_ctx(), OptionType::PUT);
    // Surface: 10 + 40*(sigma - 0.3) + bias.  sample.sigma_min = 0.1, so a
    // bias of 40 * 3e-4 = 0.012 puts the root at 0.0997, 3 bps below the
    // published edge and inside the 5 bps band.
    auto refs = resolved_refs(10.0 + 40.0 * (0.1 - 0.3));
    auto s = score(linear_handle(0.012), refs, 100.0, 100.0, 0.5, 0.1, 0.05);
    EXPECT_EQ(s.status, PointStatus::Measured);
    // sigma0 is 0.1; the three targets sit at y +- 1e-4, i.e. +- 2.5e-6.
    EXPECT_NEAR(s.iv_error, 3e-4 + 2.5e-6, 1e-7);
    EXPECT_TRUE(s.edge_band_rescue)
        << "the root is outside the un-widened product bracket";
}

TEST(RoundTripScore, EdgeExtensionStopsAtTheBandWithoutHeadroom) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;  // band 5 bps
    auto score = make_round_trip_score_fn(params, bspline_like_ctx(), OptionType::PUT);
    // bias 40 * 8e-4 = 0.032 -> root at 0.0992, 8 bps below the edge and
    // outside the band: the outcome is the shipped inversion's.
    auto refs = resolved_refs(10.0 + 40.0 * (0.1 - 0.3));
    auto s = score(linear_handle(0.032), refs, 100.0, 100.0, 0.5, 0.1, 0.05);
    EXPECT_EQ(s.status, PointStatus::SurfaceNoRoot);
    EXPECT_FALSE(s.edge_band_rescue);
}

// A globally linear surface is scored identically whether the fit domain has
// sigma headroom or not.  This pins the *band*, not the extension: on a linear
// handle the first-order extension reproduces the surface exactly, so the two
// contexts must agree to the last bit.  What discriminates the extension is
// EdgeExtensionUsesEdgeTangentNotRawEvaluation below.
TEST(RoundTripScore, UnclippedBandScoresLinearSurfaceAlikeWithAndWithoutHeadroom) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto refs = resolved_refs(10.0 + 40.0 * (0.1 - 0.3));
    auto with_headroom = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT)(
        linear_handle(0.012), refs, 100.0, 100.0, 0.5, 0.1, 0.05);
    auto extended = make_round_trip_score_fn(params, bspline_like_ctx(), OptionType::PUT)(
        linear_handle(0.012), refs, 100.0, 100.0, 0.5, 0.1, 0.05);
    EXPECT_EQ(with_headroom.status, PointStatus::Measured);
    EXPECT_EQ(extended.status, with_headroom.status);
    EXPECT_NEAR(extended.iv_error, with_headroom.iv_error, 1e-12);
    EXPECT_EQ(extended.edge_band_rescue, with_headroom.edge_band_rescue);
}

// Spec D3, rev 5: what the extension actually is.  This handle is defined
// only on the fit domain -- it returns NaN outside it -- and is *curved*
// inside, so the two things the extension claims are both observable:
//
//   1. the scorer never evaluates the surface outside its fit domain.  A raw
//      evaluation at the band floor would return NaN and the point would
//      score SurfaceNonFinite, so this test fails outright if the clamp is
//      removed;
//   2. what it extrapolates is the edge *tangent*, price(sigma) = S(e) +
//      V(e)*(sigma - e), not the surface's own continuation.  The expected
//      root below is computed from S(e) and V(e) alone and pinned to 1e-9,
//      which the quadratic's own root misses by ~2.2e-7 -- the
//      O(vomma * tau_iv^2) the spec allows, here made visible.
TEST(RoundTripScore, EdgeExtensionUsesEdgeTangentNotRawEvaluation) {
    const auto ctx = bspline_like_ctx();          // fit.sigma == sample.sigma
    const double edge = ctx.bounds.sigma_min;     // 0.1
    // Inside [0.1, 0.5]: 10 + 40*(s-0.3) + 50*(s-0.3)^2, vega 40 + 100*(s-0.3).
    // Rising over the whole band (vega 20 at the low edge, 60 at the high
    // one), so the screen sees exactly one crossing.  Outside: NaN.
    const auto curved = [](double s) {
        return 10.0 + 40.0 * (s - 0.3) + 50.0 * (s - 0.3) * (s - 0.3);
    };
    const auto curved_vega = [](double s) { return 40.0 + 100.0 * (s - 0.3); };
    const SurfaceHandle handle{
        .price = [&](double, double, double, double s, double) {
            return (s < 0.1 || s > 0.5) ? std::numeric_limits<double>::quiet_NaN()
                                        : curved(s);
        },
        .vega = [&](double, double, double, double s, double) {
            return (s < 0.1 || s > 0.5) ? std::numeric_limits<double>::quiet_NaN()
                                        : curved_vega(s);
        }};

    const double S_e = curved(edge);         // 4.0
    const double V_e = curved_vega(edge);    // 20.0
    ASSERT_GT(V_e, 0.0);

    AdaptiveGridParams params; params.target_iv_error = 5e-4;   // band 5 bps
    // Put the tangent's root 3 bps below the edge, inside the 5 bps band.
    const double y = S_e + V_e * (-3e-4);
    auto refs = resolved_refs(y);

    auto s = make_round_trip_score_fn(params, ctx, OptionType::PUT)(
        handle, refs, 100.0, 100.0, 0.5, edge, 0.05);

    ASSERT_EQ(s.status, PointStatus::Measured)
        << "status " << static_cast<int>(s.status)
        << " -- SurfaceNonFinite here means the band was evaluated on the raw "
           "handle instead of the edge tangent";
    // sigma_hat_k = edge + (target_k - S_e) / V_e for each of y +- delta and y;
    // the worst distance from sigma0 = edge is the lowest target's.
    const double worst_target = y - refs.delta;
    const double tangent_expected = std::abs((worst_target - S_e) / V_e);
    // 2e-8, not tighter: the shipped inversion stops on a 1e-6 price residual,
    // which is 5e-8 of sigma at this slope, so Brent itself lands ~5e-9 out.
    EXPECT_NEAR(s.iv_error, tangent_expected, 2e-8)
        << "the extension must follow the edge tangent, not the surface's own "
           "curvature";
    // And it is not the quadratic's own root: continuing 50*(s-0.3)^2 past the
    // edge puts the crossing at 0.09969977 rather than the tangent's 0.0997,
    // a gap of 2.3e-7 -- eleven times the tolerance above, and the
    // O(vomma * tau_iv^2) the spec allows, made visible.
    const double u = (-40.0 + std::sqrt(1600.0 - 200.0 * (10.0 - worst_target)))
                   / 100.0;
    const double quadratic_expected = std::abs((0.3 + u) - edge);
    EXPECT_GT(std::abs(s.iv_error - quadratic_expected), 1e-7)
        << "scored the surface's own continuation, not its edge tangent";
    EXPECT_TRUE(s.edge_band_rescue)
        << "the root is outside the un-widened product bracket";
}

TEST(RoundTripScore, MapsEveryFailureKind) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    const auto refs = resolved_refs(10.0);
    SurfaceHandle flat{.price = [](double, double, double, double, double) { return 10.0; },
                       .vega = [](double, double, double, double, double) { return 0.0; }};
    EXPECT_EQ(score(flat, refs, 100, 100, 0.5, 0.3, 0.05).status, PointStatus::SurfaceVegaTooSmall);
    SurfaceHandle nan_mid{.price = [](double, double, double, double s, double) { return s > 0.35 ? std::nan("") : 10.0 + 40.0 * (s - 0.3); },
                          .vega = [](double, double, double, double, double) { return 40.0; }};
    EXPECT_EQ(score(nan_mid, refs, 100, 100, 0.5, 0.3, 0.05).status, PointStatus::SurfaceNonFinite);
    // Decreasing crossing: MultipleRoots via the post-Brent slope check.
    SurfaceHandle falling{.price = [](double, double, double, double s, double) { return 10.0 - 40.0 * (s - 0.3); },
                          .vega = [](double, double, double, double, double) { return 40.0; }};
    auto f = score(falling, refs, 100, 100, 0.5, 0.3, 0.05);
    EXPECT_EQ(f.status, PointStatus::SurfaceAmbiguous);
    // y+delta straddles the top of the surface's range -> the worst of the three targets wins.
    SurfaceHandle capped{.price = [](double, double, double, double s, double) { return std::min(10.0 + 40.0 * (s - 0.3), 10.00005); },
                         .vega = [](double, double, double, double, double) { return 40.0; }};
    EXPECT_EQ(score(capped, refs, 100, 100, 0.5, 0.3, 0.05).status, PointStatus::SurfaceNoRoot);
}

// The sixth status, which `MapsEveryFailureKind` cannot reach: on any
// fixture the shipped 50-iteration budget converges, so the only way to see
// the exhausted budget is to shrink it -- which is what the scorer's
// `base_policy` parameter exists for.  Brent tests convergence at the top of
// each pass, so one pass cannot report a root however good its step was.
//
// Regression: `PointStatus::SurfaceNonConvergent` had no fixture at all, so
// nothing pinned that `MaxIterationsExceeded` reaches the scorer as itself
// rather than collapsing into another status.
// Bug: the scorer always built the product policy internally, leaving the
// iteration budget with no injection point.
TEST(RoundTripScore, ExhaustedIterationBudgetScoresNonConvergent) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    SurfaceInversionPolicy one_pass;   // the product policy, one Brent pass
    one_pass.max_iter = 1;
    auto s = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT,
                                      one_pass)(
        linear_handle(0.0), resolved_refs(10.0), 100.0, 100.0, 0.5, 0.3, 0.05);
    EXPECT_EQ(s.status, PointStatus::SurfaceNonConvergent);
    EXPECT_TRUE(std::isnan(s.iv_error));
    EXPECT_FALSE(s.edge_band_rescue);

    // The default is the shipped policy, so the same point measures: the
    // budget is what this test varied and nothing else.
    auto shipped = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT)(
        linear_handle(0.0), resolved_refs(10.0), 100.0, 100.0, 0.5, 0.3, 0.05);
    EXPECT_EQ(shipped.status, PointStatus::Measured);
}

// Spec D3: across the three targets the point is recorded under the most
// severe outcome -- NonFinite > NonConvergent > Ambiguous > NoRoot >
// VegaTooSmall -- whichever target produced it.
//
// Only pairs that can differ *between targets* are exercisable here, and
// NonFinite and VegaTooSmall are not among them: a NaN in the objective is a
// NaN at every target price, and the vega pre-check reads the same bracket
// for all three whenever the published range sits inside the cap ladder's
// first rung, as it does on every fixture in this file.
TEST(RoundTripScore, MostSevereTargetOutcomeWins) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    SurfaceInversionPolicy one_pass;
    one_pass.max_iter = 1;
    auto one_pass_score =
        make_round_trip_score_fn(params, score_ctx(), OptionType::PUT, one_pass);
    auto shipped = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    const auto refs = resolved_refs(10.0);   // targets 9.9999, 10.0, 10.0001

    // Ceiling at 10.00005: the highest target has no root, the other two
    // exhaust the one-pass budget.  The severe outcome comes first in target
    // order, so this pins that a later milder one does not overwrite it.
    SurfaceHandle capped{
        .price = [](double, double, double, double s, double) {
            return std::min(10.0 + 40.0 * (s - 0.3), 10.00005); },
        .vega = [](double, double, double, double, double) { return 40.0; }};
    EXPECT_EQ(one_pass_score(capped, refs, 100, 100, 0.5, 0.3, 0.05).status,
              PointStatus::SurfaceNonConvergent);

    // Floor at 9.99995: now the lowest target has no root, so the severe
    // outcome arrives second and must still win.
    SurfaceHandle floored{
        .price = [](double, double, double, double s, double) {
            return std::max(10.0 + 40.0 * (s - 0.3), 9.99995); },
        .vega = [](double, double, double, double, double) { return 40.0; }};
    EXPECT_EQ(one_pass_score(floored, refs, 100, 100, 0.5, 0.3, 0.05).status,
              PointStatus::SurfaceNonConvergent);

    // With the shipped budget the two other targets measure, so what remains
    // is the single failing target's own outcome: the ordering above is what
    // the one-pass policy changed, not the fixtures.
    EXPECT_EQ(shipped(capped, refs, 100, 100, 0.5, 0.3, 0.05).status,
              PointStatus::SurfaceNoRoot);
    EXPECT_EQ(shipped(floored, refs, 100, 100, 0.5, 0.3, 0.05).status,
              PointStatus::SurfaceNoRoot);

    // Ambiguous over NoRoot, on the shipped policy: a falling surface capped
    // above 10.00005 leaves the highest target rootless and makes the other
    // two fail the post-Brent slope check.
    SurfaceHandle falling_capped{
        .price = [](double, double, double, double s, double) {
            return std::min(10.0 - 40.0 * (s - 0.3), 10.00005); },
        .vega = [](double, double, double, double, double) { return 40.0; }};
    EXPECT_EQ(shipped(falling_capped, refs, 100, 100, 0.5, 0.3, 0.05).status,
              PointStatus::SurfaceAmbiguous);
}

// A handle without `vega` cannot be round-tripped: calling an empty
// std::function would throw, and library code here does not throw.  The
// residual needs only `price`, so it is still recorded.
TEST(RoundTripScore, HandleWithoutVegaScoresNonFiniteAndKeepsResidual) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    SurfaceHandle price_only{
        .price = [](double, double, double, double s, double) { return 10.0 + 40.0 * (s - 0.3) + 0.04; }};
    auto s = score(price_only, resolved_refs(10.0), 100.0, 100.0, 0.5, 0.3, 0.05);
    EXPECT_EQ(s.status, PointStatus::SurfaceNonFinite);
    EXPECT_TRUE(std::isfinite(s.price_residual));
    EXPECT_NEAR(s.price_residual, 4e-4, 1e-12);
    EXPECT_TRUE(std::isnan(s.iv_error));
}

// ===========================================================================
// Probe adapter (spec D1/L6)
// ===========================================================================

namespace {

// Records the coordinates the adapter forwarded and hands back a stencil with
// distinct values in every field, so a scaling applied to the wrong one shows.
struct RecordingBase {
    std::shared_ptr<std::pair<double, double>> seen =
        std::make_shared<std::pair<double, double>>(0.0, 0.0);

    PrepareRefsFn fn() const {
        auto seen_ = seen;
        return [seen_](double spot, double strike, double /*tau*/,
                       double sigma, double /*rate*/)
            -> std::expected<ErrorRefs, SolverError> {
            *seen_ = {spot, strike};
            ErrorRefs r;
            r.ref_price = 5.0;
            r.bracket_lo_price = 4.0;
            r.bracket_hi_price = 6.0;
            r.sigma_lo = sigma - 1e-3;
            r.sigma_hi = sigma + 1e-3;
            r.delta = 0.25;
            r.delta_lo = 0.125;
            r.delta_hi = 0.5;
            r.resolved = true;
            r.fine_steps = 512;
            r.coarse_steps = 256;
            return r;
        };
    }
};

}  // namespace

// The probe's reference must be solved on the probe's own contract, at
// (spot/a, K_ref), with every monetary field scaled by a = strike/K_ref and
// nothing else touched.
TEST(ProbeScaledRefs, SolvesProbeContractAndScalesMonetaryFields) {
    RecordingBase base;
    const double K_ref = 100.0;
    const double strike = 125.0;
    const double a = strike / K_ref;  // 1.25
    auto adapted = make_probe_scaled_refs_fn(base.fn(), K_ref);

    auto refs = adapted(100.0, strike, 0.5, 0.2, 0.04);
    ASSERT_TRUE(refs.has_value());

    // The base saw the probe's coordinates, not the query's.
    EXPECT_DOUBLE_EQ(base.seen->first, 100.0 / a);
    EXPECT_DOUBLE_EQ(base.seen->second, K_ref);

    EXPECT_DOUBLE_EQ(refs->ref_price, a * 5.0);
    EXPECT_DOUBLE_EQ(refs->bracket_lo_price, a * 4.0);
    EXPECT_DOUBLE_EQ(refs->bracket_hi_price, a * 6.0);
    EXPECT_DOUBLE_EQ(refs->delta, a * 0.25);
    EXPECT_DOUBLE_EQ(refs->delta_lo, a * 0.125);
    EXPECT_DOUBLE_EQ(refs->delta_hi, a * 0.5);

    // Volatility is not a price: the stencil's sigma coordinates, its
    // resolution and its step counts are scale-invariant.
    EXPECT_DOUBLE_EQ(refs->sigma_lo, 0.2 - 1e-3);
    EXPECT_DOUBLE_EQ(refs->sigma_hi, 0.2 + 1e-3);
    EXPECT_TRUE(refs->resolved);
    EXPECT_EQ(refs->fine_steps, 512u);
    EXPECT_EQ(refs->coarse_steps, 256u);
}

// At the reference strike the probe *is* the query, so the adapter is the
// identity and forwards the query untouched.
TEST(ProbeScaledRefs, IsIdentityAtTheReferenceStrike) {
    RecordingBase base;
    const double K_ref = 100.0;
    auto adapted = make_probe_scaled_refs_fn(base.fn(), K_ref);

    auto refs = adapted(97.0, K_ref, 0.5, 0.2, 0.04);
    ASSERT_TRUE(refs.has_value());
    EXPECT_DOUBLE_EQ(base.seen->first, 97.0);
    EXPECT_DOUBLE_EQ(base.seen->second, K_ref);
    EXPECT_DOUBLE_EQ(refs->ref_price, 5.0);
    EXPECT_DOUBLE_EQ(refs->bracket_lo_price, 4.0);
    EXPECT_DOUBLE_EQ(refs->bracket_hi_price, 6.0);
    EXPECT_DOUBLE_EQ(refs->delta, 0.25);
    EXPECT_DOUBLE_EQ(refs->delta_lo, 0.125);
    EXPECT_DOUBLE_EQ(refs->delta_hi, 0.5);
}
