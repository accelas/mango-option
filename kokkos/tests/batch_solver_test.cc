/**
 * @file batch_solver_test.cc
 * @brief Tests for batched American option solver with Kokkos
 */

#include "kokkos/src/option/batch_solver.hpp"
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

class BatchSolverTest : public ::testing::Test {
protected:
    // Reference prices from Black-Scholes (European) for validation
    // American puts should be >= European puts
    double european_put_price(double S, double K, double T, double sigma, double r, double q) {
        double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);

        // Normal CDF approximation
        auto norm_cdf = [](double x) {
            return 0.5 * std::erfc(-x / std::sqrt(2.0));
        };

        double Kexp = K * std::exp(-r * T);
        double Sexp = S * std::exp(-q * T);
        return Kexp * norm_cdf(-d2) - Sexp * norm_cdf(-d1);
    }
};

/// Test single option in batch (degenerate case)
TEST_F(BatchSolverTest, SingleOption) {
    mango::kokkos::BatchPricingParams params{
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .is_put = true
    };

    view_type strikes("strikes", 1);
    view_type spots("spots", 1);
    Kokkos::deep_copy(strikes, 100.0);
    Kokkos::deep_copy(spots, 100.0);

    mango::kokkos::BatchAmericanSolver<MemSpace> solver(params, strikes, spots, 101, 500);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << "Solve should succeed";

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.value());

    // ATM put should have positive value
    EXPECT_GT(results_h(0).price, 0.0) << "ATM put should have positive price";

    // American put >= European put
    double european = european_put_price(100.0, 100.0, 1.0, 0.20, 0.05, 0.02);
    EXPECT_GE(results_h(0).price, european * 0.99) << "American >= European";

    // Reasonable price range (ATM put with these params should be ~6-8)
    EXPECT_GT(results_h(0).price, 4.0);
    EXPECT_LT(results_h(0).price, 15.0);

    // Delta should be negative for puts
    EXPECT_LT(results_h(0).delta, 0.0) << "Put delta should be negative";
    EXPECT_GT(results_h(0).delta, -1.0) << "Delta should be > -1";
}

/// Test batch of options with varying strikes
TEST_F(BatchSolverTest, MultipleStrikes) {
    mango::kokkos::BatchPricingParams params{
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .is_put = true
    };

    // Create batch with strikes from ITM to OTM
    const size_t n_options = 5;
    view_type strikes("strikes", n_options);
    view_type spots("spots", n_options);

    auto strikes_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, strikes);
    auto spots_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, spots);

    // Strikes: 90 (OTM), 95, 100 (ATM), 105, 110 (ITM) for puts
    strikes_h(0) = 90.0;
    strikes_h(1) = 95.0;
    strikes_h(2) = 100.0;
    strikes_h(3) = 105.0;
    strikes_h(4) = 110.0;

    // All spots at 100
    for (size_t i = 0; i < n_options; ++i) {
        spots_h(i) = 100.0;
    }

    Kokkos::deep_copy(strikes, strikes_h);
    Kokkos::deep_copy(spots, spots_h);

    mango::kokkos::BatchAmericanSolver<MemSpace> solver(params, strikes, spots, 101, 500);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << "Solve should succeed";

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.value());

    // ITM puts (higher K) should have higher prices
    for (size_t i = 0; i < n_options - 1; ++i) {
        EXPECT_LT(results_h(i).price, results_h(i + 1).price)
            << "Higher strike put should be worth more: K=" << strikes_h(i)
            << " vs K=" << strikes_h(i + 1);
    }

    // OTM put (K=90) should be small but positive
    EXPECT_GT(results_h(0).price, 0.0);
    EXPECT_LT(results_h(0).price, 5.0);

    // Deep ITM put (K=110) should be close to intrinsic
    double intrinsic_110 = 110.0 - 100.0;  // = 10
    EXPECT_GT(results_h(4).price, intrinsic_110 * 0.95);
}

/// Test call options
TEST_F(BatchSolverTest, CallOptions) {
    mango::kokkos::BatchPricingParams params{
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .is_put = false  // Call
    };

    view_type strikes("strikes", 3);
    view_type spots("spots", 3);

    auto strikes_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, strikes);
    auto spots_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, spots);

    strikes_h(0) = 90.0;   // ITM call
    strikes_h(1) = 100.0;  // ATM call
    strikes_h(2) = 110.0;  // OTM call

    for (size_t i = 0; i < 3; ++i) {
        spots_h(i) = 100.0;
    }

    Kokkos::deep_copy(strikes, strikes_h);
    Kokkos::deep_copy(spots, spots_h);

    mango::kokkos::BatchAmericanSolver<MemSpace> solver(params, strikes, spots, 101, 500);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.value());

    // ITM calls (lower K) should have higher prices
    EXPECT_GT(results_h(0).price, results_h(1).price) << "ITM call > ATM call";
    EXPECT_GT(results_h(1).price, results_h(2).price) << "ATM call > OTM call";

    // Deltas should be positive for calls
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_GT(results_h(i).delta, 0.0) << "Call delta should be positive";
        EXPECT_LT(results_h(i).delta, 1.0) << "Delta should be < 1";
    }

    // ITM call delta > ATM > OTM
    EXPECT_GT(results_h(0).delta, results_h(1).delta);
    EXPECT_GT(results_h(1).delta, results_h(2).delta);
}

/// Test batch size and dimensions
TEST_F(BatchSolverTest, BatchDimensions) {
    mango::kokkos::BatchPricingParams params{
        .maturity = 0.5,
        .volatility = 0.25,
        .rate = 0.03,
        .dividend_yield = 0.01,
        .is_put = true
    };

    const size_t n_options = 100;
    view_type strikes("strikes", n_options);
    view_type spots("spots", n_options);

    // Fill with varying strikes
    Kokkos::parallel_for("fill_strikes", n_options,
        KOKKOS_LAMBDA(const size_t i) {
            strikes(i) = 80.0 + 0.4 * static_cast<double>(i);  // 80 to 120
            spots(i) = 100.0;
        });
    Kokkos::fence();

    mango::kokkos::BatchAmericanSolver<MemSpace> solver(params, strikes, spots, 51, 250);

    EXPECT_EQ(solver.batch_size(), n_options);
    EXPECT_EQ(solver.n_space(), 51);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Verify all results are reasonable
    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.value());
    for (size_t i = 0; i < n_options; ++i) {
        EXPECT_GT(results_h(i).price, 0.0) << "Option " << i << " should have positive price";
        EXPECT_TRUE(std::isfinite(results_h(i).price)) << "Option " << i << " price should be finite";
        EXPECT_TRUE(std::isfinite(results_h(i).delta)) << "Option " << i << " delta should be finite";
    }
}

/// Test different spot prices per option
TEST_F(BatchSolverTest, VaryingSpots) {
    mango::kokkos::BatchPricingParams params{
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.0,
        .is_put = true
    };

    view_type strikes("strikes", 3);
    view_type spots("spots", 3);

    auto strikes_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, strikes);
    auto spots_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, spots);

    // All same strike, varying spots
    strikes_h(0) = 100.0;
    strikes_h(1) = 100.0;
    strikes_h(2) = 100.0;

    spots_h(0) = 90.0;   // ITM put (S < K)
    spots_h(1) = 100.0;  // ATM put
    spots_h(2) = 110.0;  // OTM put (S > K)

    Kokkos::deep_copy(strikes, strikes_h);
    Kokkos::deep_copy(spots, spots_h);

    mango::kokkos::BatchAmericanSolver<MemSpace> solver(params, strikes, spots, 101, 500);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.value());

    // ITM put (S=90, K=100) should be most valuable
    EXPECT_GT(results_h(0).price, results_h(1).price) << "ITM put > ATM put";
    EXPECT_GT(results_h(1).price, results_h(2).price) << "ATM put > OTM put";

    // ITM put should be worth at least intrinsic
    double intrinsic = 100.0 - 90.0;  // = 10
    EXPECT_GT(results_h(0).price, intrinsic * 0.95);
}
