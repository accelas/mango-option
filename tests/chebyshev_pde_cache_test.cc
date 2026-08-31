// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <cmath>
#include "mango/option/table/chebyshev/chebyshev_pde_cache.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"

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

// ===========================================================================
// Regression tests for the #419 incident closure (D6)
// ===========================================================================

// Regression: a NaN PDE slice was stored as invalid, then extraction did
// `if (!spline) continue` over a zero-initialized tensor — the surface built
// "successfully" out of silent zeros
// Bug: ChebyshevPDECache::store_slice discarded the CubicSpline build error
// and build_segment_leaves treated missing slices as skippable
TEST(ChebyshevPDECacheTest, InvalidSliceFailsSegmentExtraction) {
    mango::ChebyshevPDECache cache;
    std::vector<double> x = {-0.5, 0.0, 0.5, 1.0};
    std::vector<double> bad = {0.1, std::nan(""), 0.2, 0.3};
    cache.store_slice(0.2, 0.05, 0, x, bad);
    ASSERT_EQ(cache.get_slice(0.2, 0.05, 0), nullptr);  // marked invalid

    std::vector<double> seg_bounds = {0.0, 1.0};
    std::vector<bool> seg_is_gap = {false};
    std::vector<double> m = {-0.5, 0.0, 0.5};
    std::vector<double> tau = {0.5};
    std::vector<double> sigma = {0.2};
    std::vector<double> rate = {0.05};

    auto leaves = mango::detail::build_segment_leaves(
        cache, /*K_ref=*/100.0, seg_bounds, seg_is_gap, /*include_gaps=*/false,
        m, tau, sigma, rate);
    ASSERT_FALSE(leaves.has_value());
    EXPECT_EQ(leaves.error().code, mango::PriceTableErrorCode::ExtractionFailed);
}

// Regression: a non-gap segment containing no tau nodes silently became a
// zeros-placeholder leaf, pricing the whole real segment as 0
// Bug: the Nt_seg == 0 placeholder branch did not check seg_is_gap
TEST(ChebyshevPDECacheTest, EmptyRealSegmentFailsExtraction) {
    mango::ChebyshevPDECache cache;
    std::vector<double> x = {-0.5, 0.0, 0.5, 1.0};
    std::vector<double> good = {0.1, 0.15, 0.2, 0.3};
    // Only one tau node is supplied below (tau = {0.75}), so it maps to
    // tau_idx 0 within build_segment_leaves.
    cache.store_slice(0.2, 0.05, 0, x, good);
    ASSERT_NE(cache.get_slice(0.2, 0.05, 0), nullptr);  // valid slice

    // Two real segments: [0.0, 0.5) and [0.5, 1.0]. The single tau node
    // falls entirely in the second segment, so the first segment has zero
    // tau nodes despite being a real (non-gap) segment.
    std::vector<double> seg_bounds = {0.0, 0.5, 1.0};
    std::vector<bool> seg_is_gap = {false, false};
    std::vector<double> m = {-0.5, 0.0, 0.5};
    std::vector<double> tau = {0.75};
    std::vector<double> sigma = {0.2};
    std::vector<double> rate = {0.05};

    auto leaves = mango::detail::build_segment_leaves(
        cache, /*K_ref=*/100.0, seg_bounds, seg_is_gap, /*include_gaps=*/true,
        m, tau, sigma, rate);
    ASSERT_FALSE(leaves.has_value());
    EXPECT_EQ(leaves.error().code, mango::PriceTableErrorCode::ExtractionFailed);
}
