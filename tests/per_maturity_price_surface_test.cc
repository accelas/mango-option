// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <cmath>
#include "mango/option/table/per_maturity_price_surface.hpp"
#include "mango/option/table/price_table_surface.hpp"

namespace mango {
namespace {

// Helper to create a simple 3D surface with predictable values
// Surface returns f(m, sigma, rate) = m + sigma + rate
std::shared_ptr<const PriceTableSurface<3>> make_test_surface_3d(double offset = 0.0) {
    PriceTableAxes<3> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};  // moneyness
    axes.grids[1] = {0.10, 0.20, 0.30, 0.40};   // sigma
    axes.grids[2] = {0.02, 0.04, 0.06, 0.08};   // rate

    // Simple coefficients that produce predictable interpolation
    size_t n0 = axes.grids[0].size();
    size_t n1 = axes.grids[1].size();
    size_t n2 = axes.grids[2].size();
    std::vector<double> coeffs(n0 * n1 * n2);

    // Fill with simple values: coeff = m + sigma + rate + offset
    for (size_t i2 = 0; i2 < n2; ++i2) {
        for (size_t i1 = 0; i1 < n1; ++i1) {
            for (size_t i0 = 0; i0 < n0; ++i0) {
                size_t idx = i0 + n0 * (i1 + n1 * i2);
                coeffs[idx] = axes.grids[0][i0] + axes.grids[1][i1] +
                              axes.grids[2][i2] + offset;
            }
        }
    }

    PriceTableMetadata meta{.K_ref = 100.0};
    auto result = PriceTableSurface<3>::build(std::move(axes), std::move(coeffs), meta);
    return result.value();
}

// ===========================================================================
// Build validation tests
// ===========================================================================

TEST(PerMaturityPriceSurfaceTest, BuildSucceeds) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(0.1));
    surfaces.push_back(make_test_surface_3d(0.2));
    surfaces.push_back(make_test_surface_3d(0.3));

    std::vector<double> tau_grid = {0.25, 0.5, 0.75, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto result = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta));

    ASSERT_TRUE(result.has_value()) << "Error code: "
        << static_cast<int>(result.error().code);

    auto surface = result.value();
    EXPECT_EQ(surface->num_maturities(), 4);
    EXPECT_EQ(surface->tau_grid().size(), 4);
    EXPECT_DOUBLE_EQ(surface->metadata().K_ref, 100.0);
}

TEST(PerMaturityPriceSurfaceTest, RejectEmptySurfaces) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    std::vector<double> tau_grid = {0.25, 0.5};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto result = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta));

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

TEST(PerMaturityPriceSurfaceTest, RejectMismatchedSizes) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(0.1));

    // 3 tau points but only 2 surfaces
    std::vector<double> tau_grid = {0.25, 0.5, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto result = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta));

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

TEST(PerMaturityPriceSurfaceTest, RejectSingleMaturity) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));

    // Only 1 maturity point
    std::vector<double> tau_grid = {0.5};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto result = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta));

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

TEST(PerMaturityPriceSurfaceTest, RejectUnsortedTauGrid) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(0.1));
    surfaces.push_back(make_test_surface_3d(0.2));

    // Tau grid not sorted
    std::vector<double> tau_grid = {0.25, 1.0, 0.5};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto result = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta));

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

TEST(PerMaturityPriceSurfaceTest, RejectNullSurface) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(nullptr);  // null surface
    surfaces.push_back(make_test_surface_3d(0.2));

    std::vector<double> tau_grid = {0.25, 0.5, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto result = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta));

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

// ===========================================================================
// Value interpolation tests
// ===========================================================================

TEST(PerMaturityPriceSurfaceTest, ValueAtGridPoint) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(1.0));

    std::vector<double> tau_grid = {0.5, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto surface = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta)).value();

    // At tau=0.5 (first grid point), should get first surface value
    double v1 = surface->value(1.0, 0.5, 0.20, 0.04);
    // At tau=1.0 (second grid point), should get second surface value
    double v2 = surface->value(1.0, 1.0, 0.20, 0.04);

    // Second surface has offset +1.0
    EXPECT_NEAR(v2 - v1, 1.0, 0.1);
}

TEST(PerMaturityPriceSurfaceTest, ValueInterpolatesMidpoint) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(2.0));

    std::vector<double> tau_grid = {0.0, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto surface = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta)).value();

    double v0 = surface->value(1.0, 0.0, 0.20, 0.04);
    double v1 = surface->value(1.0, 1.0, 0.20, 0.04);
    double v_mid = surface->value(1.0, 0.5, 0.20, 0.04);

    // Linear interpolation: midpoint should be average
    EXPECT_NEAR(v_mid, (v0 + v1) / 2.0, 0.1);
}

TEST(PerMaturityPriceSurfaceTest, ValueClampsBelowMinTau) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(1.0));

    std::vector<double> tau_grid = {0.5, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto surface = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta)).value();

    // Query at tau < min_tau should clamp to first surface
    double v_below = surface->value(1.0, 0.1, 0.20, 0.04);
    double v_first = surface->value(1.0, 0.5, 0.20, 0.04);

    EXPECT_DOUBLE_EQ(v_below, v_first);
}

TEST(PerMaturityPriceSurfaceTest, ValueClampsAboveMaxTau) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(1.0));

    std::vector<double> tau_grid = {0.5, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto surface = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta)).value();

    // Query at tau > max_tau should clamp to last surface
    double v_above = surface->value(1.0, 2.0, 0.20, 0.04);
    double v_last = surface->value(1.0, 1.0, 0.20, 0.04);

    EXPECT_DOUBLE_EQ(v_above, v_last);
}

TEST(PerMaturityPriceSurfaceTest, ValueWith4DCoords) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(1.0));

    std::vector<double> tau_grid = {0.5, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto surface = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta)).value();

    // Test 4D array interface
    std::array<double, 4> coords = {1.0, 0.75, 0.20, 0.04};
    double v1 = surface->value(coords);
    double v2 = surface->value(1.0, 0.75, 0.20, 0.04);

    EXPECT_DOUBLE_EQ(v1, v2);
}

// ===========================================================================
// Partial derivative tests
// ===========================================================================

TEST(PerMaturityPriceSurfaceTest, PartialWrtTau) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(1.0));

    std::vector<double> tau_grid = {0.0, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto surface = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta)).value();

    // axis=1 is tau. Derivative of linear interpolation is (v1-v0)/(tau1-tau0)
    double partial_tau = surface->partial(1, 1.0, 0.5, 0.20, 0.04);

    // Offset changes by 1.0 over tau span of 1.0, so derivative should be ~1.0
    EXPECT_NEAR(partial_tau, 1.0, 0.2);
}

TEST(PerMaturityPriceSurfaceTest, PartialWrtMoneyness) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(0.0));

    std::vector<double> tau_grid = {0.5, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto surface = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta)).value();

    // axis=0 is moneyness. Since coeffs = m + sigma + rate, d/dm should be ~1
    double partial_m = surface->partial(0, 1.0, 0.75, 0.20, 0.04);

    // The B-spline derivative should be close to 1
    EXPECT_NEAR(partial_m, 1.0, 0.3);
}

TEST(PerMaturityPriceSurfaceTest, PartialInvalidAxisReturnsNaN) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(0.0));

    std::vector<double> tau_grid = {0.5, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto surface = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta)).value();

    // axis=4 is invalid (only 0-3 are valid)
    double partial_invalid = surface->partial(4, 1.0, 0.75, 0.20, 0.04);

    EXPECT_TRUE(std::isnan(partial_invalid));
}

TEST(PerMaturityPriceSurfaceTest, PartialWith4DCoords) {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    surfaces.push_back(make_test_surface_3d(0.0));
    surfaces.push_back(make_test_surface_3d(1.0));

    std::vector<double> tau_grid = {0.5, 1.0};
    PriceTableMetadata meta{.K_ref = 100.0};

    auto surface = PerMaturityPriceSurface::build(
        std::move(surfaces), std::move(tau_grid), std::move(meta)).value();

    // Test 4D array interface
    std::array<double, 4> coords = {1.0, 0.75, 0.20, 0.04};
    double p1 = surface->partial(1, coords);
    double p2 = surface->partial(1, 1.0, 0.75, 0.20, 0.04);

    EXPECT_DOUBLE_EQ(p1, p2);
}

} // namespace
} // namespace mango
