// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/chebyshev/chebyshev_pde_cache.hpp"

using namespace mango;

TEST(ChebyshevPDECacheTest, MissingPairsReturnsAllInitially) {
    ChebyshevPDECache cache;
    std::vector<double> sigmas = {0.10, 0.20, 0.30};
    std::vector<double> rates = {0.03, 0.05};
    auto missing = cache.missing_pairs(sigmas, rates);
    EXPECT_EQ(missing.size(), 6u);  // 3 x 2
}

TEST(ChebyshevPDECacheTest, StoreAndRetrieveSlice) {
    ChebyshevPDECache cache;
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> v = {1.0, 1.5, 2.0};
    cache.store_slice(0.20, 0.05, /*tau_idx=*/0, x, v);

    auto* spline = cache.get_slice(0.20, 0.05, 0);
    ASSERT_NE(spline, nullptr);
    EXPECT_NEAR(spline->eval(0.25), 1.25, 0.1);
}

TEST(ChebyshevPDECacheTest, MissingPairsExcludesCached) {
    ChebyshevPDECache cache;
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> v = {1.0, 1.5, 2.0};
    cache.store_slice(0.20, 0.05, 0, x, v);

    std::vector<double> sigmas = {0.10, 0.20, 0.30};
    std::vector<double> rates = {0.03, 0.05};
    auto missing = cache.missing_pairs(sigmas, rates);
    // (0.20, 0.05) is cached, so 5 remain
    EXPECT_EQ(missing.size(), 5u);
}

TEST(ChebyshevPDECacheTest, QuantizationMatchesCrossLevel) {
    ChebyshevPDECache cache;
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> v = {1.0, 1.5, 2.0};
    // Store at a value computed one way
    double sigma = 0.05 + (0.50 - 0.05) * 0.5;  // 0.275
    cache.store_slice(sigma, 0.05, 0, x, v);

    // Query with a value computed a different way (same physical value)
    double sigma2 = 0.275;
    auto* spline = cache.get_slice(sigma2, 0.05, 0);
    ASSERT_NE(spline, nullptr);
}
