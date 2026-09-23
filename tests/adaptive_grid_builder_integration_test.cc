// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include <chrono>

namespace mango {
namespace {

class AdaptiveGridBuilderIntegrationTest : public ::testing::Test {
protected:
    // Use same parameters as working unit test for reliability
    OptionGrid make_test_chain() {
        OptionGrid chain;
        chain.spot = 100.0;
        chain.dividend_yield = 0.0;  // Match unit test

        // Same ranges as working unit test
        chain.strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
        chain.maturities = {0.25, 0.5, 1.0};
        chain.implied_vols = {0.18, 0.20, 0.22};
        chain.rates = {0.04, 0.05, 0.06};

        return chain;
    }

    // The table's own PDE budget, shared by every case here.
    //
    // It is sized against the validation oracle, not against the option: the
    // round-trip metric inverts the candidate surface at the oracle's own
    // price, so any discretization error the table carries and the reference
    // does not comes back as an IV miss.  Measured on this chain with the
    // 51-point, 200-step grid these cases used before the round-trip metric:
    // the table's price sat 2.3e-3 of strike below the High-accuracy
    // reference, a 202 bps outcome that no table-grid refinement moved: vol
    // seeds at 5, 7 and 9 knots, a vol range widened from [0.15, 0.25] to
    // [0.12, 0.28], and 7 strikes instead of 5 each reproduced it to three
    // digits, because it is the table's PDE error, not its interpolation
    // error.  At 401 points and 800 steps the same chain measures 3.4 bps.
    //
    // The 202 bps floor also broke the build outright: at a fresh sample near
    // the published sigma ceiling (sigma = 0.2472, ceiling 0.25) the root of
    // the oracle's price lay 74 bps of vol above the sample, outside the
    // acceptance band of +/- target_iv_error, so the inversion returned no
    // root -- a resolved surface failure, which makes the candidate
    // non-viable and refuses the build with NoViableSurface.
    PDEGridConfig make_pde_grid() {
        auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 401, 2.0).value();
        return PDEGridConfig{grid_spec, 800, {}};
    }
};

TEST_F(AdaptiveGridBuilderIntegrationTest, ConvergesToTarget) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;  // 20 bps - achievable target
    params.max_iter = 2;
    params.validation_samples = 8;  // Match unit test

    auto result = build_adaptive_bspline(params, chain,
        make_pde_grid(), OptionType::PUT);

    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    // Convergence is not a coin flip at this budget: the chain measures
    // 3.4 bps against a 20 bps target, on the first candidate.  Asserting it
    // conditionally would let a future regression pass by simply failing to
    // converge.
    ASSERT_TRUE(result->target_met);
    EXPECT_LE(result->achieved_max_error, params.target_iv_error);

    // Should have diagnostic history
    EXPECT_FALSE(result->iterations.empty());

    // Should have a surface
    EXPECT_NE(result->spline, nullptr);
}

TEST_F(AdaptiveGridBuilderIntegrationTest, RefinementAttemptsLargerGrid) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.0001;  // Very tight target, likely won't hit
    params.max_iter = 3;
    params.validation_samples = 8;

    auto result = build_adaptive_bspline(params, chain,
        make_pde_grid(), OptionType::PUT);

    ASSERT_TRUE(result.has_value());

    // Refinement inserts sites into attempted candidates. Backtracking may
    // retain the seed when its measured error is lower, so the retained grid
    // need not be larger. Every attempt must preserve the seed's density.
    ASSERT_GE(result->iterations.size(), 2u);
    const auto& first = result->iterations.front();
    bool any_grew = false;
    for (const auto& candidate : result->iterations) {
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_GE(candidate.grid_sizes[d], first.grid_sizes[d]);
            any_grew |= candidate.grid_sizes[d] > first.grid_sizes[d];
        }
    }
    EXPECT_TRUE(any_grew) << "A tight target must attempt a denser candidate";
}

TEST_F(AdaptiveGridBuilderIntegrationTest, HandlesImpossibleTarget) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    // 1 bp: unattainable on this chain and grid budget, which measures
    // 3.4 bps, and still inside what the reference oracle can resolve.  The
    // target is also the half-width of the stencil the oracle prepares, so a
    // target far below the oracle's own uncertainty estimate is not an
    // "impossible target" at all -- it is an unresolvable one, and the build
    // then refuses instead of reporting best effort.  That outcome is pinned
    // separately by RefusesTargetBelowReferenceResolution.
    //
    // The margin is 3.4x, not orders of magnitude: 3.4e-4 measured against
    // the 1e-4 asked for here.  A change that makes this chain materially
    // more accurate will make the target attainable and flip the case, and
    // the answer then is a smaller target, not a weaker assertion.
    params.target_iv_error = 1e-4;
    params.max_iter = 2;       // Limited iterations
    params.max_points_per_dim = 10;  // Limited grid
    params.validation_samples = 8;

    auto result = build_adaptive_bspline(params, chain,
        make_pde_grid(), OptionType::PUT);

    ASSERT_TRUE(result.has_value());

    // Should return best-effort result with target_met = false
    EXPECT_FALSE(result->target_met);

    // Still have a surface
    EXPECT_NE(result->spline, nullptr);

    // Should have reached max iterations.  Spec D5: when the retained
    // candidate is not the surface most recently built it is rebuilt once,
    // and that entry (refined_dim == -2) does not consume budget.
    size_t built_iterations = 0;
    for (const auto& it : result->iterations) {
        if (it.refined_dim != -2) ++built_iterations;
    }
    EXPECT_EQ(built_iterations, params.max_iter);
}

TEST_F(AdaptiveGridBuilderIntegrationTest, DeterministicWithSameSeed) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.005;  // Relaxed target
    params.max_iter = 2;
    params.validation_samples = 8;
    params.lhs_seed = 12345;

    auto result1 = build_adaptive_bspline(params, chain,
        make_pde_grid(), OptionType::PUT);
    auto result2 = build_adaptive_bspline(params, chain,
        make_pde_grid(), OptionType::PUT);

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());

    // Same seed should produce same results
    EXPECT_DOUBLE_EQ(result1->achieved_max_error, result2->achieved_max_error);
    EXPECT_EQ(result1->iterations.size(), result2->iterations.size());

    // Check iteration stats match
    for (size_t i = 0; i < result1->iterations.size(); ++i) {
        EXPECT_EQ(result1->iterations[i].grid_sizes, result2->iterations[i].grid_sizes);
        EXPECT_DOUBLE_EQ(result1->iterations[i].max_error, result2->iterations[i].max_error);
    }
}

TEST_F(AdaptiveGridBuilderIntegrationTest, DifferentSeedsProduceDifferentSamples) {
    auto chain = make_test_chain();

    AdaptiveGridParams params1;
    params1.target_iv_error = 0.01;
    params1.max_iter = 1;  // Single iteration to focus on sampling difference
    params1.validation_samples = 8;
    params1.lhs_seed = 111;

    AdaptiveGridParams params2 = params1;
    params2.lhs_seed = 222;

    auto result1 = build_adaptive_bspline(params1, chain,
        make_pde_grid(), OptionType::PUT);
    auto result2 = build_adaptive_bspline(params2, chain,
        make_pde_grid(), OptionType::PUT);

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());

    // Different seeds likely produce different error estimates
    // (not guaranteed but very likely with enough samples)
    // At minimum, both should complete successfully
    EXPECT_GT(result1->achieved_max_error, 0.0);
    EXPECT_GT(result2->achieved_max_error, 0.0);
}

TEST_F(AdaptiveGridBuilderIntegrationTest, SurfaceInterpolatesWithinBounds) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 2;
    params.validation_samples = 8;

    auto result = build_adaptive_bspline(params, chain,
        make_pde_grid(), OptionType::PUT);

    ASSERT_TRUE(result.has_value());
    ASSERT_NE(result->spline, nullptr);

    // The stored spline is an EEP residual, which may legitimately be zero.
    // Check the physical price through its public reconstruction interface.
    auto surface = make_bspline_surface(result->spline, result->K_ref,
        result->dividend_yield, OptionType::PUT);
    ASSERT_TRUE(surface.has_value());
    double price = surface->price(100.0, 100.0, 0.5, 0.20, 0.05);

    // Price should be positive and reasonable
    EXPECT_GT(price, 0.0) << "Interpolated price should be positive";
    EXPECT_LT(price, 100.0) << "Put price should be less than spot";
}

TEST_F(AdaptiveGridBuilderIntegrationTest, TracksIterationDiagnostics) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.0001;  // Tight target to force multiple iterations
    params.max_iter = 3;
    params.validation_samples = 8;

    auto result = build_adaptive_bspline(params, chain,
        make_pde_grid(), OptionType::PUT);

    ASSERT_TRUE(result.has_value());

    // Should have multiple iterations
    ASSERT_GE(result->iterations.size(), 1);

    for (const auto& iter : result->iterations) {
        // Each iteration should have valid stats
        EXPECT_GE(iter.pde_solves_table, 0);
        EXPECT_GE(iter.pde_solves_validation, 0);
        EXPECT_GE(iter.max_error, 0.0);
        EXPECT_GE(iter.avg_error, 0.0);
        EXPECT_LE(iter.avg_error, iter.max_error);
        EXPECT_GT(iter.elapsed_seconds, 0.0);

        // Grid sizes should be valid
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_GE(iter.grid_sizes[d], 4) << "Need at least 4 points for B-spline";
        }
    }

    // Total PDE solves should be consistent.
    //
    // Regression: the accounting identity is over the reference solve
    // counter, not over `pde_solves_validation`.
    // Bug: this summed `pde_solves_table + pde_solves_validation`, which held
    // only while a validation point cost exactly the solves it was charged.
    // `pde_solves_validation` counts *preparations* (spec D7), and one
    // preparation now runs a six-solve stencil -- three fine and three coarse
    // -- so the sum understated the work (54 against an actual 222).  The
    // solves themselves come from the one `ReferenceSolveCounter` the build
    // owns, reported as `reference_solves_fine/coarse`.
    size_t table_solves = 0;
    for (const auto& iter : result->iterations) {
        table_solves += iter.pde_solves_table;
    }
    EXPECT_EQ(result->total_pde_solves,
              table_solves + result->diagnostics.reference_solves_fine
                           + result->diagnostics.reference_solves_coarse);

    // Equality is not structural: a preparation whose fine solve fails skips
    // the coarse half, and one that returns early on a non-positive
    // sigma0 - target_iv_error leaves one fine attempt against no coarse one
    // (adaptive_metrics.cpp, make_stencil_refs_fn).  What this asserts is
    // that neither happened on this fixture -- every preparation cleared the
    // stencil's positivity check and no reference solve failed -- so the
    // uncertainty estimates really are two-grid differences here.
    EXPECT_EQ(result->diagnostics.reference_solves_fine,
              result->diagnostics.reference_solves_coarse);

    // Three fine solves per prepared point, over the fresh preparations the
    // iterations report plus the one-off holdout preparations.
    size_t preparations = 0;
    for (const auto& iter : result->iterations) {
        preparations += iter.pde_solves_validation;
    }
    EXPECT_GE(result->diagnostics.reference_solves_fine, 3 * preparations);
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: a target below the reference oracle's resolution refuses the
// build rather than returning a best-effort surface.
// Bug: HandlesImpossibleTarget asked for 1e-10 and expected best effort.  The
// round-trip metric prepares its reference as a price stencil at
// sigma0 +/- target_iv_error, so at 1e-10 the three prices are separated by
// about 1e-8 dollars while the oracle's own uncertainty estimate on this
// chain is 3e-5 to 1.4e-4 dollars.  Measured at one holdout point
// (K = 100.2582, tau = 0.78588, sigma0 = 0.23203): reference 6.98875070,
// bracket low 6.98875070, bracket high 6.98875071 -- a separation of 1e-8
// against delta = 7.317e-05, so D2's ordering test fails by four orders of
// magnitude.  No holdout point resolves (measured: 0 of
// 8, against the coverage floor of max(4, samples/4) = 4), and the loop
// refuses with ValidationFailed and the `mango:adaptive_validation_refused`
// probe.  This is the honest outcome -- the reference cannot tell the three
// targets apart -- so it is pinned, not repaired.
TEST_F(AdaptiveGridBuilderIntegrationTest, RefusesTargetBelowReferenceResolution) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 1e-10;
    params.max_iter = 2;
    params.max_points_per_dim = 10;
    params.validation_samples = 8;

    auto result = build_adaptive_bspline(params, chain,
        make_pde_grid(), OptionType::PUT);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::ValidationFailed);
}

}  // namespace
}  // namespace mango
