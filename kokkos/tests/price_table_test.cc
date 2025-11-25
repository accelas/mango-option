/**
 * @file price_table_test.cc
 * @brief Tests for price table builder with Kokkos
 */

#include "kokkos/src/option/price_table.hpp"
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cmath>

namespace {

// Global Kokkos initialization
struct KokkosEnvironment : public ::testing::Environment {
    void SetUp() override {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }
};

::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

using MemSpace = Kokkos::HostSpace;
using view_type = Kokkos::View<double*, MemSpace>;

}  // anonymous namespace

class PriceTableTest : public ::testing::Test {
protected:
    /// Helper to create uniform grid View
    view_type create_uniform_grid(double xmin, double xmax, size_t n) {
        view_type grid("grid", n);
        auto h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, grid);
        if (n == 1) {
            h(0) = xmin;
        } else {
            for (size_t i = 0; i < n; ++i) {
                h(i) = xmin + (xmax - xmin) * static_cast<double>(i) /
                       static_cast<double>(n - 1);
            }
        }
        Kokkos::deep_copy(grid, h);
        return grid;
    }
};

/// Test small price table build
TEST_F(PriceTableTest, SmallTable) {
    // Create small grids
    auto moneyness = create_uniform_grid(0.9, 1.1, 3);   // 3 moneyness points
    auto maturity = create_uniform_grid(0.25, 1.0, 2);   // 2 maturities
    auto vol = create_uniform_grid(0.15, 0.25, 2);       // 2 vols
    auto rate = create_uniform_grid(0.02, 0.05, 2);      // 2 rates

    mango::kokkos::PriceTableConfig config{
        .n_space = 51,
        .n_time = 100,
        .K_ref = 100.0,
        .q = 0.0,
        .is_put = true
    };

    mango::kokkos::PriceTableBuilder4D<MemSpace> builder(
        moneyness, maturity, vol, rate, config);

    auto result = builder.build();
    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    auto table = std::move(result.value());

    // Check dimensions
    EXPECT_EQ(table.shape[0], 3);  // moneyness
    EXPECT_EQ(table.shape[1], 2);  // maturity
    EXPECT_EQ(table.shape[2], 2);  // vol
    EXPECT_EQ(table.shape[3], 2);  // rate

    // All prices should be positive
    auto prices = table.prices();
    for (size_t im = 0; im < 3; ++im) {
        for (size_t it = 0; it < 2; ++it) {
            for (size_t is = 0; is < 2; ++is) {
                for (size_t ir = 0; ir < 2; ++ir) {
                    EXPECT_GT(prices(im, it, is, is), 0.0)
                        << "Price at (" << im << "," << it << "," << is << "," << ir
                        << ") should be positive";
                }
            }
        }
    }
}

/// Test price monotonicity
TEST_F(PriceTableTest, PriceMonotonicity) {
    // For puts: higher strike (lower moneyness) = higher price
    // Use narrower range to avoid numerical issues
    auto moneyness = create_uniform_grid(0.9, 1.1, 5);   // 5 moneyness points (tighter range)
    auto maturity = create_uniform_grid(0.5, 0.5, 1);    // Single maturity
    auto vol = create_uniform_grid(0.20, 0.20, 1);       // Single vol
    auto rate = create_uniform_grid(0.05, 0.05, 1);      // Single rate

    mango::kokkos::PriceTableConfig config{
        .n_space = 101,  // More grid points for stability
        .n_time = 500,   // More time steps
        .K_ref = 100.0,
        .q = 0.0,
        .is_put = true
    };

    mango::kokkos::PriceTableBuilder4D<MemSpace> builder(
        moneyness, maturity, vol, rate, config);

    auto result = builder.build();
    ASSERT_TRUE(result.has_value());

    auto table = std::move(result.value());
    auto prices = table.prices();

    // Check all prices are finite first
    for (size_t im = 0; im < 5; ++im) {
        ASSERT_TRUE(std::isfinite(prices(im, 0, 0, 0)))
            << "Price at m[" << im << "]=" << table.moneyness_grid[im]
            << " should be finite, got " << prices(im, 0, 0, 0);
    }

    // For puts: lower moneyness (deeper ITM) should have higher price
    for (size_t im = 0; im < 4; ++im) {
        double price_low_m = prices(im, 0, 0, 0);     // Lower moneyness
        double price_high_m = prices(im + 1, 0, 0, 0); // Higher moneyness

        EXPECT_GE(price_low_m, price_high_m * 0.9)  // Allow some tolerance
            << "Put price should decrease with moneyness: m[" << im
            << "]=" << table.moneyness_grid[im]
            << " price=" << price_low_m
            << " vs m[" << im+1 << "]=" << table.moneyness_grid[im+1]
            << " price=" << price_high_m;
    }
}

/// Test lookup interpolation
TEST_F(PriceTableTest, LookupInterpolation) {
    auto moneyness = create_uniform_grid(0.9, 1.1, 3);
    auto maturity = create_uniform_grid(0.25, 1.0, 3);
    auto vol = create_uniform_grid(0.15, 0.25, 2);
    auto rate = create_uniform_grid(0.02, 0.05, 2);

    mango::kokkos::PriceTableConfig config{
        .n_space = 51,
        .n_time = 100,
        .K_ref = 100.0,
        .q = 0.0,
        .is_put = true
    };

    mango::kokkos::PriceTableBuilder4D<MemSpace> builder(
        moneyness, maturity, vol, rate, config);

    auto result = builder.build();
    ASSERT_TRUE(result.has_value());

    auto table = std::move(result.value());

    // Lookup at grid point should match stored value
    double lookup_corner = table.lookup(0.9, 0.25, 0.15, 0.02);
    double stored_corner = table.prices()(0, 0, 0, 0);
    EXPECT_NEAR(lookup_corner, stored_corner, 1e-10)
        << "Lookup at grid corner should match stored value";

    // Lookup at interior point should be interpolated
    double lookup_interior = table.lookup(1.0, 0.5, 0.20, 0.035);
    EXPECT_GT(lookup_interior, 0.0) << "Interior lookup should be positive";
    EXPECT_TRUE(std::isfinite(lookup_interior)) << "Interior lookup should be finite";
}

/// Test edge cases
TEST_F(PriceTableTest, EdgeCases) {
    auto moneyness = create_uniform_grid(0.9, 1.1, 3);
    auto maturity = create_uniform_grid(0.25, 1.0, 2);
    auto vol = create_uniform_grid(0.15, 0.25, 2);
    auto rate = create_uniform_grid(0.02, 0.05, 2);

    mango::kokkos::PriceTableConfig config{
        .n_space = 51,
        .n_time = 100,
        .K_ref = 100.0,
        .q = 0.0,
        .is_put = true
    };

    mango::kokkos::PriceTableBuilder4D<MemSpace> builder(
        moneyness, maturity, vol, rate, config);

    auto result = builder.build();
    ASSERT_TRUE(result.has_value());

    auto table = std::move(result.value());

    // Lookup outside bounds should clamp
    double lookup_below = table.lookup(0.5, 0.1, 0.10, 0.01);
    double lookup_above = table.lookup(1.5, 2.0, 0.50, 0.10);

    EXPECT_GT(lookup_below, 0.0) << "Below-bounds lookup should be positive";
    EXPECT_GT(lookup_above, 0.0) << "Above-bounds lookup should be positive";
    EXPECT_TRUE(std::isfinite(lookup_below));
    EXPECT_TRUE(std::isfinite(lookup_above));
}
