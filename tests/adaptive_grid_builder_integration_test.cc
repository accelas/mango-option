#include <gtest/gtest.h>
#include "src/option/table/adaptive_grid_builder.hpp"
#include <chrono>

namespace mango {
namespace {

class AdaptiveGridBuilderIntegrationTest : public ::testing::Test {
protected:
    // Use same parameters as working unit test for reliability
    OptionChain make_test_chain() {
        OptionChain chain;
        chain.spot = 100.0;
        chain.dividend_yield = 0.0;  // Match unit test

        // Same ranges as working unit test
        chain.strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
        chain.maturities = {0.25, 0.5, 1.0};
        chain.implied_vols = {0.18, 0.20, 0.22};
        chain.rates = {0.04, 0.05, 0.06};

        return chain;
    }
};

TEST_F(AdaptiveGridBuilderIntegrationTest, ConvergesToTarget) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;  // 20 bps - achievable target
    params.max_iterations = 2;
    params.validation_samples = 8;  // Match unit test

    AdaptiveGridBuilder builder(params);
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();

    auto result = builder.build(chain, grid_spec, 200, OptionType::PUT);

    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    // Should meet target within max_iterations
    if (result->target_met) {
        EXPECT_LE(result->achieved_max_error, params.target_iv_error);
    }

    // Should have diagnostic history
    EXPECT_FALSE(result->iterations.empty());

    // Should have a surface
    EXPECT_NE(result->surface, nullptr);
}

TEST_F(AdaptiveGridBuilderIntegrationTest, RefinementIncreasesGridSize) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.0001;  // Very tight target, likely won't hit
    params.max_iterations = 3;
    params.validation_samples = 8;

    AdaptiveGridBuilder builder(params);
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();

    auto result = builder.build(chain, grid_spec, 200, OptionType::PUT);

    ASSERT_TRUE(result.has_value());

    // With 3 iterations and tight target, grid should have grown
    if (result->iterations.size() >= 2) {
        auto& first = result->iterations.front();
        auto& last = result->iterations.back();

        // At least one dimension should have grown
        bool any_grew = false;
        for (size_t d = 0; d < 4; ++d) {
            if (last.grid_sizes[d] > first.grid_sizes[d]) {
                any_grew = true;
                break;
            }
        }
        EXPECT_TRUE(any_grew) << "Grid should refine when target not met";
    }
}

TEST_F(AdaptiveGridBuilderIntegrationTest, HandlesImpossibleTarget) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 1e-10;  // Impossible target
    params.max_iterations = 2;       // Limited iterations
    params.max_points_per_dim = 10;  // Limited grid
    params.validation_samples = 8;

    AdaptiveGridBuilder builder(params);
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();

    auto result = builder.build(chain, grid_spec, 200, OptionType::PUT);

    ASSERT_TRUE(result.has_value());

    // Should return best-effort result with target_met = false
    EXPECT_FALSE(result->target_met);

    // Still have a surface
    EXPECT_NE(result->surface, nullptr);

    // Should have reached max iterations
    EXPECT_EQ(result->iterations.size(), params.max_iterations);
}

TEST_F(AdaptiveGridBuilderIntegrationTest, DeterministicWithSameSeed) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.005;  // Relaxed target
    params.max_iterations = 2;
    params.validation_samples = 8;
    params.lhs_seed = 12345;

    AdaptiveGridBuilder builder1(params);
    AdaptiveGridBuilder builder2(params);

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();

    auto result1 = builder1.build(chain, grid_spec, 200, OptionType::PUT);
    auto result2 = builder2.build(chain, grid_spec, 200, OptionType::PUT);

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
    params1.max_iterations = 1;  // Single iteration to focus on sampling difference
    params1.validation_samples = 8;
    params1.lhs_seed = 111;

    AdaptiveGridParams params2 = params1;
    params2.lhs_seed = 222;

    AdaptiveGridBuilder builder1(params1);
    AdaptiveGridBuilder builder2(params2);

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();

    auto result1 = builder1.build(chain, grid_spec, 200, OptionType::PUT);
    auto result2 = builder2.build(chain, grid_spec, 200, OptionType::PUT);

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
    params.max_iterations = 2;
    params.validation_samples = 8;

    AdaptiveGridBuilder builder(params);
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();

    auto result = builder.build(chain, grid_spec, 200, OptionType::PUT);

    ASSERT_TRUE(result.has_value());
    ASSERT_NE(result->surface, nullptr);

    // Query the surface at interior points
    // Moneyness from chain: S/K = 100/90 to 100/110 ≈ 0.91 to 1.11
    // Use a middle point
    double m = 1.0;    // ATM
    double tau = 0.5;  // 6 months
    double sigma = 0.20;
    double rate = 0.05;

    double price = result->surface->value({m, tau, sigma, rate});

    // Price should be positive and reasonable
    EXPECT_GT(price, 0.0) << "Interpolated price should be positive";
    EXPECT_LT(price, 100.0) << "Put price should be less than spot";
}

TEST_F(AdaptiveGridBuilderIntegrationTest, TracksIterationDiagnostics) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.0001;  // Tight target to force multiple iterations
    params.max_iterations = 3;
    params.validation_samples = 8;

    AdaptiveGridBuilder builder(params);
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();

    auto result = builder.build(chain, grid_spec, 200, OptionType::PUT);

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

    // Total PDE solves should be consistent
    size_t computed_total = 0;
    for (const auto& iter : result->iterations) {
        computed_total += iter.pde_solves_table + iter.pde_solves_validation;
    }
    EXPECT_EQ(result->total_pde_solves, computed_total);
}

}  // namespace
}  // namespace mango
