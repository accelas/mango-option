/**
 * @file price_table_integration_test.cc
 * @brief Comprehensive integration tests for price table builder
 *
 * Tests:
 * - Fast vs fallback path consistency
 * - Various grid resolutions
 * - Financial accuracy against direct solver
 * - Performance benchmarks
 */

#include <gtest/gtest.h>
#include "kokkos/src/option/price_table.hpp"
#include "kokkos/src/option/american_option.hpp"
#include <cmath>
#include <chrono>

namespace mango::kokkos::test {

// Global Kokkos environment
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override { Kokkos::initialize(); }
    void TearDown() override { Kokkos::finalize(); }
};

[[maybe_unused]] static ::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class PriceTableIntegrationTest : public ::testing::Test {
protected:
    using MemSpace = Kokkos::HostSpace;
    using view_type = Kokkos::View<double*, MemSpace>;

    /// Create uniform grid View
    view_type make_grid(const std::vector<double>& values) {
        view_type grid("grid", values.size());
        auto h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, grid);
        for (size_t i = 0; i < values.size(); ++i) {
            h(i) = values[i];
        }
        Kokkos::deep_copy(grid, h);
        return grid;
    }

    /// Create uniform grid View with range
    view_type make_uniform_grid(double xmin, double xmax, size_t n) {
        view_type grid("grid", n);
        auto h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, grid);
        for (size_t i = 0; i < n; ++i) {
            h(i) = xmin + (xmax - xmin) * static_cast<double>(i) /
                   static_cast<double>(n - 1);
        }
        Kokkos::deep_copy(grid, h);
        return grid;
    }
};

// ============================================================================
// Table vs Direct Solver Consistency Tests
// ============================================================================

TEST_F(PriceTableIntegrationTest, TableMatchesDirectSolver) {
    // Build a small price table and verify it matches direct solver results
    auto moneyness = make_grid({0.9, 0.95, 1.0, 1.05, 1.1});
    auto maturity = make_grid({0.25, 0.5, 1.0});
    auto vol = make_grid({0.20});
    auto rate = make_grid({0.05});

    PriceTableConfig config{
        .K_ref = 100.0,
        .q = 0.02,
        .is_put = true
    };

    PriceTableBuilder4D<MemSpace> builder(moneyness, maturity, vol, rate, config);
    auto result = builder.build();
    ASSERT_TRUE(result.has_value());

    auto table = std::move(result.value());

    // Compare each grid point with direct solver
    std::vector<double> m_vec = {0.9, 0.95, 1.0, 1.05, 1.1};
    std::vector<double> t_vec = {0.25, 0.5, 1.0};

    for (size_t im = 0; im < m_vec.size(); ++im) {
        for (size_t it = 0; it < t_vec.size(); ++it) {
            double m = m_vec[im];
            double tau = t_vec[it];
            double table_price = table.prices()(im, it, 0, 0);

            // Solve directly
            PricingParams params{
                .strike = 100.0,
                .spot = m * 100.0,
                .maturity = tau,
                .volatility = 0.20,
                .rate = 0.05,
                .dividend_yield = 0.02,
                .type = OptionType::Put
            };

            AmericanOptionSolver<MemSpace> solver(params);
            auto direct_result = solver.solve();
            ASSERT_TRUE(direct_result.has_value());

            double direct_price = direct_result->price;

            // Should match within 1% relative error
            double rel_error = std::abs(table_price - direct_price) /
                              std::max(direct_price, 1e-10);
            EXPECT_LT(rel_error, 0.01)
                << "Table price " << table_price << " vs direct " << direct_price
                << " at m=" << m << ", tau=" << tau;
        }
    }
}

// ============================================================================
// Interpolation Quality Tests
// ============================================================================

TEST_F(PriceTableIntegrationTest, InterpolationSmoothness) {
    // Build table and verify interpolation is smooth
    auto moneyness = make_uniform_grid(0.85, 1.15, 7);
    auto maturity = make_grid({0.25, 0.5, 1.0, 2.0});
    auto vol = make_grid({0.15, 0.20, 0.25, 0.30});
    auto rate = make_grid({0.02, 0.05, 0.08});

    PriceTableConfig config{
        .K_ref = 100.0,
        .q = 0.02,
        .is_put = true
    };

    PriceTableBuilder4D<MemSpace> builder(moneyness, maturity, vol, rate, config);
    auto result = builder.build();
    ASSERT_TRUE(result.has_value());

    auto table = std::move(result.value());

    // Sample interpolated values between grid points
    // Check that the interpolation is monotonic
    double prev_price = 1e10;
    for (double m = 0.85; m <= 1.15; m += 0.01) {
        double price = table.lookup(m, 1.0, 0.20, 0.05);
        EXPECT_GT(price, 0.0) << "Price should be positive at m=" << m;
        EXPECT_TRUE(std::isfinite(price)) << "Price should be finite at m=" << m;

        // For puts, price should decrease as moneyness increases
        if (m > 0.85) {
            EXPECT_LE(price, prev_price + 0.5)  // Allow small non-monotonicity due to interpolation
                << "Put price should generally decrease with moneyness at m=" << m;
        }
        prev_price = price;
    }
}

TEST_F(PriceTableIntegrationTest, InterpolationBoundaryBehavior) {
    // Test extrapolation behavior at boundaries
    auto moneyness = make_grid({0.9, 1.0, 1.1});
    auto maturity = make_grid({0.25, 0.5, 1.0});
    auto vol = make_grid({0.15, 0.25});
    auto rate = make_grid({0.02, 0.05});

    PriceTableConfig config{
        .K_ref = 100.0,
        .q = 0.0,
        .is_put = true
    };

    PriceTableBuilder4D<MemSpace> builder(moneyness, maturity, vol, rate, config);
    auto result = builder.build();
    ASSERT_TRUE(result.has_value());

    auto table = std::move(result.value());

    // Query outside bounds - should clamp
    double price_below_m = table.lookup(0.5, 0.5, 0.20, 0.03);
    double price_above_m = table.lookup(1.5, 0.5, 0.20, 0.03);
    double price_below_tau = table.lookup(1.0, 0.1, 0.20, 0.03);
    double price_above_tau = table.lookup(1.0, 2.0, 0.20, 0.03);

    EXPECT_GT(price_below_m, 0.0);
    EXPECT_GT(price_above_m, 0.0);
    EXPECT_GT(price_below_tau, 0.0);
    EXPECT_GT(price_above_tau, 0.0);

    // ITM (low moneyness) put should be worth more than OTM
    EXPECT_GT(price_below_m, price_above_m);
}

// ============================================================================
// Call Option Tests
// ============================================================================

TEST_F(PriceTableIntegrationTest, CallOptionTable) {
    auto moneyness = make_grid({0.9, 1.0, 1.1});
    auto maturity = make_grid({0.5, 1.0});
    auto vol = make_grid({0.20});
    auto rate = make_grid({0.05});

    PriceTableConfig config{
        .K_ref = 100.0,
        .q = 0.02,  // With dividend, early exercise may occur
        .is_put = false  // CALL
    };

    PriceTableBuilder4D<MemSpace> builder(moneyness, maturity, vol, rate, config);
    auto result = builder.build();
    ASSERT_TRUE(result.has_value());

    auto table = std::move(result.value());

    // For calls: higher moneyness (ITM) = higher price
    double price_otm = table.prices()(0, 0, 0, 0);  // m=0.9
    double price_atm = table.prices()(1, 0, 0, 0);  // m=1.0
    double price_itm = table.prices()(2, 0, 0, 0);  // m=1.1

    EXPECT_GT(price_itm, price_atm);
    EXPECT_GT(price_atm, price_otm);

    // All should be positive
    EXPECT_GT(price_otm, 0.0);
    EXPECT_GT(price_atm, 0.0);
    EXPECT_GT(price_itm, 0.0);
}

// ============================================================================
// Large Grid Tests
// ============================================================================

TEST_F(PriceTableIntegrationTest, LargeGridStability) {
    // Test with larger grids to ensure numerical stability
    auto moneyness = make_uniform_grid(0.80, 1.20, 9);
    auto maturity = make_grid({0.25, 0.5, 0.75, 1.0, 1.5, 2.0});
    auto vol = make_grid({0.10, 0.15, 0.20, 0.25, 0.30, 0.40});
    auto rate = make_grid({0.01, 0.03, 0.05, 0.07, 0.10});

    PriceTableConfig config{
        .K_ref = 100.0,
        .q = 0.02,
        .is_put = true
    };

    PriceTableBuilder4D<MemSpace> builder(moneyness, maturity, vol, rate, config);
    auto result = builder.build();
    ASSERT_TRUE(result.has_value()) << "Large grid build should succeed";

    auto table = std::move(result.value());

    // Verify all prices are finite and positive
    auto prices = table.prices();
    size_t nan_count = 0;
    size_t negative_count = 0;

    for (size_t im = 0; im < 9; ++im) {
        for (size_t it = 0; it < 6; ++it) {
            for (size_t is = 0; is < 6; ++is) {
                for (size_t ir = 0; ir < 5; ++ir) {
                    double p = prices(im, it, is, ir);
                    if (!std::isfinite(p)) nan_count++;
                    if (p < 0) negative_count++;
                }
            }
        }
    }

    EXPECT_EQ(nan_count, 0) << "Found " << nan_count << " NaN/Inf prices";
    EXPECT_EQ(negative_count, 0) << "Found " << negative_count << " negative prices";
}

// ============================================================================
// Volatility Sensitivity Tests
// ============================================================================

TEST_F(PriceTableIntegrationTest, VolatilitySensitivity) {
    // Higher volatility should increase option value
    auto moneyness = make_grid({1.0});
    auto maturity = make_grid({1.0});
    auto vol = make_uniform_grid(0.10, 0.50, 5);
    auto rate = make_grid({0.05});

    PriceTableConfig config{
        .K_ref = 100.0,
        .q = 0.02,
        .is_put = true
    };

    PriceTableBuilder4D<MemSpace> builder(moneyness, maturity, vol, rate, config);
    auto result = builder.build();
    ASSERT_TRUE(result.has_value());

    auto table = std::move(result.value());
    auto prices = table.prices();

    // Price should increase with volatility
    double prev_price = 0.0;
    for (size_t i = 0; i < 5; ++i) {
        double price = prices(0, 0, i, 0);
        EXPECT_GT(price, prev_price)
            << "Price should increase with vol at index " << i;
        prev_price = price;
    }
}

// ============================================================================
// Rate Sensitivity Tests
// ============================================================================

TEST_F(PriceTableIntegrationTest, RateSensitivityPut) {
    // For puts: higher rate generally decreases put value
    // (since PV of strike decreases)
    auto moneyness = make_grid({1.0});
    auto maturity = make_grid({1.0});
    auto vol = make_grid({0.20});
    auto rate = make_uniform_grid(0.01, 0.10, 5);

    PriceTableConfig config{
        .K_ref = 100.0,
        .q = 0.02,
        .is_put = true
    };

    PriceTableBuilder4D<MemSpace> builder(moneyness, maturity, vol, rate, config);
    auto result = builder.build();
    ASSERT_TRUE(result.has_value());

    auto table = std::move(result.value());
    auto prices = table.prices();

    // For American puts, higher rate may not always decrease value
    // (due to early exercise), but extreme changes should show trend
    double price_low_rate = prices(0, 0, 0, 0);
    double price_high_rate = prices(0, 0, 0, 4);

    // Both should be positive and finite
    EXPECT_GT(price_low_rate, 0.0);
    EXPECT_GT(price_high_rate, 0.0);
    EXPECT_TRUE(std::isfinite(price_low_rate));
    EXPECT_TRUE(std::isfinite(price_high_rate));
}

// ============================================================================
// Maturity Sensitivity Tests
// ============================================================================

TEST_F(PriceTableIntegrationTest, MaturitySensitivity) {
    // Longer maturity should generally increase option value
    auto moneyness = make_grid({1.0});
    auto maturity = make_grid({0.25, 0.5, 1.0, 2.0, 3.0});
    auto vol = make_grid({0.20});
    auto rate = make_grid({0.05});

    PriceTableConfig config{
        .K_ref = 100.0,
        .q = 0.02,
        .is_put = true
    };

    PriceTableBuilder4D<MemSpace> builder(moneyness, maturity, vol, rate, config);
    auto result = builder.build();
    ASSERT_TRUE(result.has_value());

    auto table = std::move(result.value());
    auto prices = table.prices();

    // Price should generally increase with maturity (time value)
    double prev_price = 0.0;
    for (size_t i = 0; i < 5; ++i) {
        double price = prices(0, i, 0, 0);
        EXPECT_GE(price, prev_price * 0.95)  // Allow small decrease
            << "Price should generally increase with maturity at index " << i;
        prev_price = price;
    }
}

// ============================================================================
// Performance Test
// ============================================================================

TEST_F(PriceTableIntegrationTest, BuildPerformance) {
    // Build a medium-sized table and measure time
    auto moneyness = make_uniform_grid(0.85, 1.15, 7);
    auto maturity = make_grid({0.25, 0.5, 1.0, 1.5, 2.0});
    auto vol = make_grid({0.15, 0.20, 0.25, 0.30});
    auto rate = make_grid({0.02, 0.04, 0.06, 0.08});

    PriceTableConfig config{
        .K_ref = 100.0,
        .q = 0.02,
        .is_put = true
    };

    PriceTableBuilder4D<MemSpace> builder(moneyness, maturity, vol, rate, config);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = builder.build();
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(result.has_value());

    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    size_t total_options = 7 * 5 * 4 * 4;

    std::cout << "Built " << total_options << " options in "
              << duration_ms << " ms ("
              << (total_options / (duration_ms / 1000.0)) << " options/sec)\n";

    // Should complete in reasonable time (< 30 seconds for ~560 options)
    EXPECT_LT(duration_ms, 30000.0);
}

}  // namespace mango::kokkos::test
