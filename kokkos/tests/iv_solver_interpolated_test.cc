/**
 * @file iv_solver_interpolated_test.cc
 * @brief Tests for GPU-accelerated IV solver with price table interpolation
 */

#include "kokkos/src/option/iv_solver_interpolated.hpp"
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

class IVSolverInterpolatedTest : public ::testing::Test {
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

    /// Build a small price table for testing
    mango::kokkos::PriceTable4D build_test_price_table() {
        // Create coarse grids for fast testing
        auto moneyness = create_uniform_grid(0.8, 1.2, 5);   // 5 moneyness points
        auto maturity = create_uniform_grid(0.25, 1.0, 3);   // 3 maturities
        auto vol = create_uniform_grid(0.10, 0.40, 5);       // 5 vols
        auto rate = create_uniform_grid(0.02, 0.08, 3);      // 3 rates

        mango::kokkos::PriceTableConfig config{
            .n_space = 51,
            .n_time = 200,
            .K_ref = 100.0,
            .q = 0.0,
            .is_put = true
        };

        mango::kokkos::PriceTableBuilder4D<MemSpace> builder(
            moneyness, maturity, vol, rate, config);

        auto result = builder.build();
        EXPECT_TRUE(result.has_value()) << "Price table build should succeed";

        return std::move(result.value());
    }
};

/// Test solver creation
TEST_F(IVSolverInterpolatedTest, SolverCreation) {
    auto table = build_test_price_table();

    mango::kokkos::IVSolverConfig config{
        .max_iterations = 50,
        .tolerance = 1e-6,
        .sigma_min = 0.05,
        .sigma_max = 2.0
    };

    auto solver = mango::kokkos::IVSolverInterpolated<MemSpace>::create(table, config);
    ASSERT_TRUE(solver.has_value()) << "Solver creation should succeed";
}

/// Test single IV solve at grid point
TEST_F(IVSolverInterpolatedTest, SingleIVAtGridPoint) {
    auto table = build_test_price_table();

    mango::kokkos::IVSolverConfig config{
        .max_iterations = 50,
        .tolerance = 1e-6,
        .sigma_min = 0.05,
        .sigma_max = 2.0
    };

    auto solver_result = mango::kokkos::IVSolverInterpolated<MemSpace>::create(table, config);
    ASSERT_TRUE(solver_result.has_value());
    auto solver = std::move(solver_result.value());

    // Create query at grid point
    // moneyness = 1.0, maturity = 0.5, vol = 0.20, rate = 0.05
    const double K = 100.0;
    const double S = 100.0;  // m = 1.0
    const double tau = 0.5;
    const double sigma_true = 0.20;
    const double r = 0.05;

    // Get market price from table
    const double market_price = table.lookup(1.0, tau, sigma_true, r) * (K / 100.0);

    // Create query
    Kokkos::View<mango::kokkos::IVQuery*, MemSpace> queries("queries", 1);
    auto q_h = Kokkos::create_mirror_view(queries);
    q_h(0) = mango::kokkos::IVQuery{
        .strike = K,
        .spot = S,
        .maturity = tau,
        .rate = r,
        .dividend_yield = 0.0,
        .type = mango::kokkos::OptionType::Put,
        .market_price = market_price
    };
    Kokkos::deep_copy(queries, q_h);

    // Solve
    auto results = solver.solve_batch(queries);
    ASSERT_TRUE(results.has_value()) << "Solve should succeed";

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results.value());

    // Verify convergence
    EXPECT_TRUE(results_h(0).converged) << "Should converge";
    EXPECT_NEAR(results_h(0).implied_vol, sigma_true, 1e-4)
        << "Should recover true volatility";
    EXPECT_LT(results_h(0).iterations, 20) << "Should converge quickly";
}

/// Test single IV solve at interior point
TEST_F(IVSolverInterpolatedTest, SingleIVAtInteriorPoint) {
    auto table = build_test_price_table();

    mango::kokkos::IVSolverConfig config;
    auto solver_result = mango::kokkos::IVSolverInterpolated<MemSpace>::create(table, config);
    ASSERT_TRUE(solver_result.has_value());
    auto solver = std::move(solver_result.value());

    // Create query at interior point (not on grid)
    const double K = 100.0;
    const double S = 95.0;  // m = 0.95
    const double tau = 0.6;
    const double sigma_true = 0.23;
    const double r = 0.045;

    // Get market price from table
    const double m = S / K;
    const double market_price = table.lookup(m, tau, sigma_true, r) * (K / 100.0);

    // Create query
    Kokkos::View<mango::kokkos::IVQuery*, MemSpace> queries("queries", 1);
    auto q_h = Kokkos::create_mirror_view(queries);
    q_h(0) = mango::kokkos::IVQuery{
        .strike = K,
        .spot = S,
        .maturity = tau,
        .rate = r,
        .dividend_yield = 0.0,
        .type = mango::kokkos::OptionType::Put,
        .market_price = market_price
    };
    Kokkos::deep_copy(queries, q_h);

    // Solve
    auto results = solver.solve_batch(queries);
    ASSERT_TRUE(results.has_value());

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results.value());

    // Verify convergence (may be less accurate due to interpolation)
    EXPECT_TRUE(results_h(0).converged) << "Should converge";
    EXPECT_NEAR(results_h(0).implied_vol, sigma_true, 1e-3)
        << "Should approximately recover volatility";
}

/// Test batch IV solve
TEST_F(IVSolverInterpolatedTest, BatchIVSolve) {
    auto table = build_test_price_table();

    mango::kokkos::IVSolverConfig config;
    auto solver_result = mango::kokkos::IVSolverInterpolated<MemSpace>::create(table, config);
    ASSERT_TRUE(solver_result.has_value());
    auto solver = std::move(solver_result.value());

    // Create batch of queries
    const size_t n_queries = 10;
    Kokkos::View<mango::kokkos::IVQuery*, MemSpace> queries("queries", n_queries);
    auto q_h = Kokkos::create_mirror_view(queries);

    std::vector<double> true_vols;

    for (size_t i = 0; i < n_queries; ++i) {
        const double K = 100.0;
        const double S = 90.0 + 2.0 * i;  // Vary spot
        const double tau = 0.5;
        const double sigma = 0.15 + 0.02 * i;  // Vary volatility
        const double r = 0.05;

        true_vols.push_back(sigma);

        const double m = S / K;
        const double market_price = table.lookup(m, tau, sigma, r) * (K / 100.0);

        q_h(i) = mango::kokkos::IVQuery{
            .strike = K,
            .spot = S,
            .maturity = tau,
            .rate = r,
            .dividend_yield = 0.0,
            .type = mango::kokkos::OptionType::Put,
            .market_price = market_price
        };
    }
    Kokkos::deep_copy(queries, q_h);

    // Solve batch
    auto results = solver.solve_batch(queries);
    ASSERT_TRUE(results.has_value());

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results.value());

    // Verify all converged
    for (size_t i = 0; i < n_queries; ++i) {
        EXPECT_TRUE(results_h(i).converged)
            << "Query " << i << " should converge";
        EXPECT_NEAR(results_h(i).implied_vol, true_vols[i], 1e-3)
            << "Query " << i << " should recover volatility";
    }
}

/// Test ITM option
TEST_F(IVSolverInterpolatedTest, ITMOption) {
    auto table = build_test_price_table();

    mango::kokkos::IVSolverConfig config;
    auto solver_result = mango::kokkos::IVSolverInterpolated<MemSpace>::create(table, config);
    ASSERT_TRUE(solver_result.has_value());
    auto solver = std::move(solver_result.value());

    // Deep ITM put: spot well below strike
    const double K = 100.0;
    const double S = 85.0;  // m = 0.85
    const double tau = 0.5;
    const double sigma_true = 0.20;
    const double r = 0.05;

    const double m = S / K;
    const double market_price = table.lookup(m, tau, sigma_true, r) * (K / 100.0);

    Kokkos::View<mango::kokkos::IVQuery*, MemSpace> queries("queries", 1);
    auto q_h = Kokkos::create_mirror_view(queries);
    q_h(0) = mango::kokkos::IVQuery{
        .strike = K,
        .spot = S,
        .maturity = tau,
        .rate = r,
        .dividend_yield = 0.0,
        .type = mango::kokkos::OptionType::Put,
        .market_price = market_price
    };
    Kokkos::deep_copy(queries, q_h);

    auto results = solver.solve_batch(queries);
    ASSERT_TRUE(results.has_value());

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results.value());

    EXPECT_TRUE(results_h(0).converged) << "ITM option should converge";
    EXPECT_NEAR(results_h(0).implied_vol, sigma_true, 2e-3)
        << "Should recover volatility for ITM option";
}

/// Test OTM option
TEST_F(IVSolverInterpolatedTest, OTMOption) {
    auto table = build_test_price_table();

    mango::kokkos::IVSolverConfig config;
    auto solver_result = mango::kokkos::IVSolverInterpolated<MemSpace>::create(table, config);
    ASSERT_TRUE(solver_result.has_value());
    auto solver = std::move(solver_result.value());

    // Deep OTM put: spot well above strike
    const double K = 100.0;
    const double S = 115.0;  // m = 1.15
    const double tau = 0.5;
    const double sigma_true = 0.20;
    const double r = 0.05;

    const double m = S / K;
    const double market_price = table.lookup(m, tau, sigma_true, r) * (K / 100.0);

    Kokkos::View<mango::kokkos::IVQuery*, MemSpace> queries("queries", 1);
    auto q_h = Kokkos::create_mirror_view(queries);
    q_h(0) = mango::kokkos::IVQuery{
        .strike = K,
        .spot = S,
        .maturity = tau,
        .rate = r,
        .dividend_yield = 0.0,
        .type = mango::kokkos::OptionType::Put,
        .market_price = market_price
    };
    Kokkos::deep_copy(queries, q_h);

    auto results = solver.solve_batch(queries);
    ASSERT_TRUE(results.has_value());

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results.value());

    EXPECT_TRUE(results_h(0).converged) << "OTM option should converge";
    EXPECT_NEAR(results_h(0).implied_vol, sigma_true, 2e-3)
        << "Should recover volatility for OTM option";
}

/// Test low volatility
TEST_F(IVSolverInterpolatedTest, LowVolatility) {
    auto table = build_test_price_table();

    mango::kokkos::IVSolverConfig config;
    auto solver_result = mango::kokkos::IVSolverInterpolated<MemSpace>::create(table, config);
    ASSERT_TRUE(solver_result.has_value());
    auto solver = std::move(solver_result.value());

    // Low volatility
    const double K = 100.0;
    const double S = 100.0;
    const double tau = 0.5;
    const double sigma_true = 0.12;
    const double r = 0.05;

    const double market_price = table.lookup(1.0, tau, sigma_true, r) * (K / 100.0);

    Kokkos::View<mango::kokkos::IVQuery*, MemSpace> queries("queries", 1);
    auto q_h = Kokkos::create_mirror_view(queries);
    q_h(0) = mango::kokkos::IVQuery{
        .strike = K,
        .spot = S,
        .maturity = tau,
        .rate = r,
        .dividend_yield = 0.0,
        .type = mango::kokkos::OptionType::Put,
        .market_price = market_price
    };
    Kokkos::deep_copy(queries, q_h);

    auto results = solver.solve_batch(queries);
    ASSERT_TRUE(results.has_value());

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results.value());

    EXPECT_TRUE(results_h(0).converged) << "Low vol should converge";
    EXPECT_NEAR(results_h(0).implied_vol, sigma_true, 1e-3);
}

/// Test high volatility
TEST_F(IVSolverInterpolatedTest, HighVolatility) {
    auto table = build_test_price_table();

    mango::kokkos::IVSolverConfig config;
    auto solver_result = mango::kokkos::IVSolverInterpolated<MemSpace>::create(table, config);
    ASSERT_TRUE(solver_result.has_value());
    auto solver = std::move(solver_result.value());

    // High volatility
    const double K = 100.0;
    const double S = 100.0;
    const double tau = 0.5;
    const double sigma_true = 0.35;
    const double r = 0.05;

    const double market_price = table.lookup(1.0, tau, sigma_true, r) * (K / 100.0);

    Kokkos::View<mango::kokkos::IVQuery*, MemSpace> queries("queries", 1);
    auto q_h = Kokkos::create_mirror_view(queries);
    q_h(0) = mango::kokkos::IVQuery{
        .strike = K,
        .spot = S,
        .maturity = tau,
        .rate = r,
        .dividend_yield = 0.0,
        .type = mango::kokkos::OptionType::Put,
        .market_price = market_price
    };
    Kokkos::deep_copy(queries, q_h);

    auto results = solver.solve_batch(queries);
    ASSERT_TRUE(results.has_value());

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results.value());

    EXPECT_TRUE(results_h(0).converged) << "High vol should converge";
    EXPECT_NEAR(results_h(0).implied_vol, sigma_true, 1e-3);
}

/// Test convergence verification: result IV should reproduce market price
TEST_F(IVSolverInterpolatedTest, ConvergenceVerification) {
    auto table = build_test_price_table();

    mango::kokkos::IVSolverConfig config;
    auto solver_result = mango::kokkos::IVSolverInterpolated<MemSpace>::create(table, config);
    ASSERT_TRUE(solver_result.has_value());
    auto solver = std::move(solver_result.value());

    const double K = 100.0;
    const double S = 98.0;
    const double tau = 0.5;
    const double sigma_true = 0.22;
    const double r = 0.05;

    const double m = S / K;
    const double market_price = table.lookup(m, tau, sigma_true, r) * (K / 100.0);

    Kokkos::View<mango::kokkos::IVQuery*, MemSpace> queries("queries", 1);
    auto q_h = Kokkos::create_mirror_view(queries);
    q_h(0) = mango::kokkos::IVQuery{
        .strike = K,
        .spot = S,
        .maturity = tau,
        .rate = r,
        .dividend_yield = 0.0,
        .type = mango::kokkos::OptionType::Put,
        .market_price = market_price
    };
    Kokkos::deep_copy(queries, q_h);

    auto results = solver.solve_batch(queries);
    ASSERT_TRUE(results.has_value());

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results.value());

    EXPECT_TRUE(results_h(0).converged);

    // Verify that using the result IV gives back the market price
    const double implied_vol = results_h(0).implied_vol;
    const double reconstructed_price = table.lookup(m, tau, implied_vol, r) * (K / 100.0);

    EXPECT_NEAR(reconstructed_price, market_price, 1e-5)
        << "Implied vol should reproduce market price";
}
