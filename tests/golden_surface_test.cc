// SPDX-License-Identifier: MIT
/**
 * @file golden_surface_test.cc
 * @brief Tests for pre-computed golden dimensionless surface
 */

#include "mango/option/table/golden_surface.hpp"

#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

TEST(GoldenSurfaceTest, ReconstructsSuccessfully) {
    auto surface = golden_dimensionless_surface();
    ASSERT_NE(surface, nullptr);
    EXPECT_EQ(surface->num_segments(), 3u);
}

TEST(GoldenSurfaceTest, AccuracySpotCheck) {
    auto surface = golden_dimensionless_surface();
    ASSERT_NE(surface, nullptr);

    // ATM put, moderate tau', moderate kappa
    double eep = surface->value({0.0, 0.04, 0.0});
    EXPECT_GE(eep, 0.0) << "EEP should be non-negative";
    EXPECT_LT(eep, 0.2) << "EEP should be bounded for normalized price";

    // Deep ITM put (S/K = 0.7, high kappa)
    double eep_itm = surface->value({std::log(0.7), 0.10, std::log(5.0)});
    EXPECT_GE(eep_itm, 0.0);

    // Deep OTM put (S/K = 1.4, low kappa)
    double eep_otm = surface->value({std::log(1.4), 0.10, std::log(0.05)});
    EXPECT_GE(eep_otm, 0.0);
    EXPECT_LT(eep_otm, 0.01) << "Deep OTM put EEP should be near zero";
}

TEST(GoldenSurfaceTest, SegmentCoverage) {
    auto surface = golden_dimensionless_surface();
    ASSERT_NE(surface, nullptr);

    // Check that segments cover the physical ln kappa domain
    // Physical domain: ln(2*0.005/0.64) to ln(2*0.10/0.01)
    double lk_min_phys = std::log(2.0 * 0.005 / (0.80 * 0.80));
    double lk_max_phys = std::log(2.0 * 0.10 / (0.10 * 0.10));

    EXPECT_LE(surface->segments().front().lk_min, lk_min_phys + 1e-10);
    EXPECT_GE(surface->segments().back().lk_max, lk_max_phys - 1e-10);
}

TEST(GoldenSurfaceTest, CachesAcrossCalls) {
    auto s1 = golden_dimensionless_surface();
    auto s2 = golden_dimensionless_surface();
    EXPECT_EQ(s1.get(), s2.get()) << "Should return same cached instance";
}

}  // namespace
}  // namespace mango
