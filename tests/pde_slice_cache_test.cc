// SPDX-License-Identifier: MIT
#include "pde_slice_cache.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

TEST(PDESliceCacheTest, EmptyCacheHasNoSlices) {
    PDESliceCache cache;
    EXPECT_EQ(cache.num_cached_pairs(), 0u);
    EXPECT_FALSE(cache.has_slice(0, 0));
}

TEST(PDESliceCacheTest, StoreAndRetrieveSlice) {
    PDESliceCache cache;
    std::vector<double> x_grid = {-0.5, -0.25, 0.0, 0.25, 0.5};
    std::vector<double> values = {1.0, 2.0, 3.0, 2.0, 1.0};
    cache.store_slice(0, 0, 0, x_grid, values);
    EXPECT_TRUE(cache.has_slice(0, 0));
    EXPECT_EQ(cache.num_cached_pairs(), 1u);

    auto* spline = cache.get_slice(0, 0, 0);
    ASSERT_NE(spline, nullptr);
    EXPECT_NEAR(spline->eval(0.0), 3.0, 1e-10);
}

TEST(PDESliceCacheTest, MultipleTauSnapshots) {
    PDESliceCache cache;
    std::vector<double> x_grid = {0.0, 0.5, 1.0};
    std::vector<double> v0 = {1.0, 2.0, 3.0};
    std::vector<double> v1 = {4.0, 5.0, 6.0};
    cache.store_slice(0, 0, 0, x_grid, v0);
    cache.store_slice(0, 0, 1, x_grid, v1);
    EXPECT_EQ(cache.num_tau_slices(0, 0), 2u);

    auto* s0 = cache.get_slice(0, 0, 0);
    auto* s1 = cache.get_slice(0, 0, 1);
    ASSERT_NE(s0, nullptr);
    ASSERT_NE(s1, nullptr);
    EXPECT_NEAR(s0->eval(0.5), 2.0, 1e-10);
    EXPECT_NEAR(s1->eval(0.5), 5.0, 1e-10);
}

TEST(PDESliceCacheTest, DifferentSigmaRatePairs) {
    PDESliceCache cache;
    std::vector<double> x_grid = {0.0, 1.0};
    std::vector<double> va = {1.0, 2.0};
    std::vector<double> vb = {3.0, 4.0};
    std::vector<double> vc = {5.0, 6.0};
    cache.store_slice(0, 0, 0, x_grid, va);
    cache.store_slice(1, 0, 0, x_grid, vb);
    cache.store_slice(0, 1, 0, x_grid, vc);
    EXPECT_EQ(cache.num_cached_pairs(), 3u);
    EXPECT_TRUE(cache.has_slice(0, 0));
    EXPECT_TRUE(cache.has_slice(1, 0));
    EXPECT_TRUE(cache.has_slice(0, 1));
    EXPECT_FALSE(cache.has_slice(1, 1));
}

TEST(PDESliceCacheTest, NewSigmaNodesCount) {
    PDESliceCache cache;
    std::vector<double> x_grid = {0.0, 1.0};
    std::vector<double> vals = {1.0, 2.0};
    for (size_t s = 0; s < 3; ++s) {
        cache.store_slice(s, 0, 0, x_grid, vals);
    }
    std::vector<size_t> wanted = {0, 1, 2, 3, 4};
    std::vector<size_t> rates = {0};
    auto missing = cache.missing_pairs(wanted, rates);
    EXPECT_EQ(missing.size(), 2u);
}

TEST(PDESliceCacheTest, PdeSolveCount) {
    PDESliceCache cache;
    EXPECT_EQ(cache.total_pde_solves(), 0u);
    cache.record_pde_solves(15);
    EXPECT_EQ(cache.total_pde_solves(), 15u);
    cache.record_pde_solves(10);
    EXPECT_EQ(cache.total_pde_solves(), 25u);
}

}  // namespace
}  // namespace mango
