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
    // delta = F_s * |bias| / (2^p - 1)
    EXPECT_NEAR(refs->delta, kRichardsonSafetyFactor * 1e-4 / (std::pow(2.0, kReferenceConvergenceOrder) - 1.0), 1e-15);
    // Achieved time steps are recorded for both levels (record only).
    EXPECT_EQ(refs->fine_steps, static_cast<uint32_t>(fake.calls[0].second));
    EXPECT_EQ(refs->coarse_steps, static_cast<uint32_t>(fake.calls[1].second));
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

// Root beyond the published edge: NoRoot under the exact product bracket;
// the tau_iv edge band rescues it only as a diagnostic flag.
TEST(RoundTripScore, EdgeMissIsNoRootWithRescueFlag) {
    AdaptiveGridParams params; params.target_iv_error = 5e-3;   // band 50 bps
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    // Surface overprices by 0.04 at sigma=0.1 -> root at 0.099 (1e-3 below the edge, inside the band)
    auto refs = resolved_refs(10.0 + 40.0 * (0.1 - 0.3));
    auto s = score(linear_handle(0.04), refs, 100.0, 100.0, 0.5, 0.1, 0.05);
    EXPECT_EQ(s.status, PointStatus::SurfaceNoRoot);
    EXPECT_TRUE(s.edge_band_rescue);
    params.target_iv_error = 5e-4;   // band 5 bps: root is 10 bps outside -> no rescue
    auto score2 = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    auto s2 = score2(linear_handle(0.04), refs, 100.0, 100.0, 0.5, 0.1, 0.05);
    EXPECT_EQ(s2.status, PointStatus::SurfaceNoRoot);
    EXPECT_FALSE(s2.edge_band_rescue);
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
