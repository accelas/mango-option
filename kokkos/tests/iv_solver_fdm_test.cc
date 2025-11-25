#include <gtest/gtest.h>
#include "kokkos/src/option/iv_solver_fdm.hpp"
#include "kokkos/src/option/american_option.hpp"
#include <cmath>

namespace mango::kokkos::test {

// Global setup/teardown for Kokkos - once per test program
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        Kokkos::initialize();
    }
    void TearDown() override {
        Kokkos::finalize();
    }
};

// Register the global environment
[[maybe_unused]] static ::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class IVSolverFDMTest : public ::testing::Test {
    // No per-test setup/teardown needed for Kokkos
};

TEST_F(IVSolverFDMTest, ATMPutIVSolve) {
    // ATM American put: solve for IV given a market price
    // First, price at known vol to get "market price"
    PricingParams pricing_params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,  // True vol
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put
    };

    // Get "market price" from PDE solver
    AmericanOptionSolver<HostMemSpace> pricer(pricing_params);
    auto price_result = pricer.solve();
    ASSERT_TRUE(price_result.has_value());
    double market_price = price_result->price;

    // Now solve for IV
    IVQuery query{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put,
        .market_price = market_price
    };

    IVSolverFDMConfig config{
        .max_iterations = 100,
        .tolerance = 1e-6,
        .sigma_min = 0.01,
        .sigma_max = 3.0,
        .n_space = 101,
        .n_time = 500
    };

    IVSolverFDM<HostMemSpace> solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_EQ(result->code, IVResultCode::Success);

    // Should recover true volatility within tolerance
    EXPECT_NEAR(result->implied_vol, 0.20, 0.005);  // 0.5% absolute error
    EXPECT_LT(result->iterations, 50);  // Should converge quickly
    EXPECT_LT(result->final_error, 1e-4);  // Price error should be small
}

TEST_F(IVSolverFDMTest, ITMPutIVSolve) {
    // ITM American put: solve for IV
    PricingParams pricing_params{
        .strike = 100.0,
        .spot = 90.0,  // ITM put
        .maturity = 1.0,
        .volatility = 0.25,  // True vol
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put
    };

    AmericanOptionSolver<HostMemSpace> pricer(pricing_params);
    auto price_result = pricer.solve();
    ASSERT_TRUE(price_result.has_value());
    double market_price = price_result->price;

    IVQuery query{
        .strike = 100.0,
        .spot = 90.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put,
        .market_price = market_price
    };

    IVSolverFDM<HostMemSpace> solver;
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_NEAR(result->implied_vol, 0.25, 0.005);
}

TEST_F(IVSolverFDMTest, OTMCallIVSolve) {
    // OTM American call: solve for IV
    PricingParams pricing_params{
        .strike = 110.0,
        .spot = 100.0,  // OTM call
        .maturity = 0.5,
        .volatility = 0.30,  // True vol
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Call
    };

    AmericanOptionSolver<HostMemSpace> pricer(pricing_params);
    auto price_result = pricer.solve();
    ASSERT_TRUE(price_result.has_value());
    double market_price = price_result->price;

    IVQuery query{
        .strike = 110.0,
        .spot = 100.0,
        .maturity = 0.5,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Call,
        .market_price = market_price
    };

    IVSolverFDM<HostMemSpace> solver;
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_NEAR(result->implied_vol, 0.30, 0.005);
}

TEST_F(IVSolverFDMTest, ConvergenceVerification) {
    // Verify that solved IV reproduces market price via PDE solver
    IVQuery query{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put,
        .market_price = 7.0  // Arbitrary market price
    };

    IVSolverFDM<HostMemSpace> solver;
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);

    // Verify: price the option at solved IV
    PricingParams verify_params{
        .strike = query.strike,
        .spot = query.spot,
        .maturity = query.maturity,
        .volatility = result->implied_vol,
        .rate = query.rate,
        .dividend_yield = query.dividend_yield,
        .type = query.type
    };

    AmericanOptionSolver<HostMemSpace> verify_pricer(verify_params);
    auto verify_price = verify_pricer.solve();
    ASSERT_TRUE(verify_price.has_value());

    // Reproduced price should match market price within tolerance
    // Note: finite discretization error means we can't expect perfect match
    EXPECT_NEAR(verify_price->price, query.market_price, 0.02);  // 2 cents tolerance
}

TEST_F(IVSolverFDMTest, InvalidNegativeSpot) {
    IVQuery query{
        .strike = 100.0,
        .spot = -10.0,  // Invalid
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put,
        .market_price = 7.0
    };

    IVSolverFDM<HostMemSpace> solver;
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), IVResultCode::InvalidParams);
}

TEST_F(IVSolverFDMTest, InvalidNegativeMarketPrice) {
    IVQuery query{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put,
        .market_price = -5.0  // Invalid
    };

    IVSolverFDM<HostMemSpace> solver;
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), IVResultCode::InvalidParams);
}

TEST_F(IVSolverFDMTest, ArbitrageViolation) {
    // Market price below intrinsic value (arbitrage)
    IVQuery query{
        .strike = 100.0,
        .spot = 80.0,  // Intrinsic = 20
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put,
        .market_price = 15.0  // Below intrinsic (20)
    };

    IVSolverFDM<HostMemSpace> solver;
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), IVResultCode::ArbitrageViolation);
}

TEST_F(IVSolverFDMTest, BatchSolve) {
    // Create batch of 3 queries
    Kokkos::View<IVQuery*, HostMemSpace> queries("queries", 3);
    auto queries_h = Kokkos::create_mirror_view(queries);

    // Query 1: ATM put, vol=0.20
    {
        PricingParams params{
            .strike = 100.0, .spot = 100.0, .maturity = 1.0,
            .volatility = 0.20, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Put
        };
        AmericanOptionSolver<HostMemSpace> pricer(params);
        auto price_result = pricer.solve();
        ASSERT_TRUE(price_result.has_value());

        queries_h(0) = IVQuery{
            .strike = 100.0, .spot = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Put,
            .market_price = price_result->price
        };
    }

    // Query 2: ITM put, vol=0.25
    {
        PricingParams params{
            .strike = 100.0, .spot = 90.0, .maturity = 1.0,
            .volatility = 0.25, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Put
        };
        AmericanOptionSolver<HostMemSpace> pricer(params);
        auto price_result = pricer.solve();
        ASSERT_TRUE(price_result.has_value());

        queries_h(1) = IVQuery{
            .strike = 100.0, .spot = 90.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Put,
            .market_price = price_result->price
        };
    }

    // Query 3: OTM call, vol=0.30
    {
        PricingParams params{
            .strike = 110.0, .spot = 100.0, .maturity = 0.5,
            .volatility = 0.30, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Call
        };
        AmericanOptionSolver<HostMemSpace> pricer(params);
        auto price_result = pricer.solve();
        ASSERT_TRUE(price_result.has_value());

        queries_h(2) = IVQuery{
            .strike = 110.0, .spot = 100.0, .maturity = 0.5,
            .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Call,
            .market_price = price_result->price
        };
    }

    Kokkos::deep_copy(queries, queries_h);

    // Solve batch
    IVSolverFDM<HostMemSpace> solver;
    auto results = solver.solve_batch(queries);

    // Verify all converged
    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);

    EXPECT_TRUE(results_h(0).converged);
    EXPECT_NEAR(results_h(0).implied_vol, 0.20, 0.005);

    EXPECT_TRUE(results_h(1).converged);
    EXPECT_NEAR(results_h(1).implied_vol, 0.25, 0.005);

    EXPECT_TRUE(results_h(2).converged);
    EXPECT_NEAR(results_h(2).implied_vol, 0.30, 0.005);
}

TEST_F(IVSolverFDMTest, DifferentGridResolutions) {
    // Test that IV solver works with different grid sizes
    PricingParams pricing_params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put
    };

    AmericanOptionSolver<HostMemSpace> pricer(pricing_params);
    auto price_result = pricer.solve();
    ASSERT_TRUE(price_result.has_value());
    double market_price = price_result->price;

    IVQuery query{
        .strike = 100.0, .spot = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.02,
        .type = OptionType::Put,
        .market_price = market_price
    };

    // Coarse grid
    {
        IVSolverFDMConfig config{
            .n_space = 51,
            .n_time = 250
        };
        IVSolverFDM<HostMemSpace> solver(config);
        auto result = solver.solve(query);

        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(result->converged);
        // Coarser grid means larger discretization error
        EXPECT_NEAR(result->implied_vol, 0.20, 0.01);
    }

    // Fine grid
    {
        IVSolverFDMConfig config{
            .n_space = 201,
            .n_time = 1000
        };
        IVSolverFDM<HostMemSpace> solver(config);
        auto result = solver.solve(query);

        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(result->converged);
        // Finer grid should give better accuracy
        EXPECT_NEAR(result->implied_vol, 0.20, 0.005);
    }
}

TEST_F(IVSolverFDMTest, ShortMaturityIV) {
    // Test with very short maturity (potential numerical challenges)
    PricingParams pricing_params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 0.1,  // 1.2 months
        .volatility = 0.25,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put
    };

    AmericanOptionSolver<HostMemSpace> pricer(pricing_params);
    auto price_result = pricer.solve();
    ASSERT_TRUE(price_result.has_value());
    double market_price = price_result->price;

    IVQuery query{
        .strike = 100.0, .spot = 100.0, .maturity = 0.1,
        .rate = 0.05, .dividend_yield = 0.02,
        .type = OptionType::Put,
        .market_price = market_price
    };

    IVSolverFDM<HostMemSpace> solver;
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_NEAR(result->implied_vol, 0.25, 0.01);  // Larger tolerance for short maturity
}

TEST_F(IVSolverFDMTest, HighVolatilityIV) {
    // Test with high volatility
    PricingParams pricing_params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.80,  // 80% vol
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put
    };

    AmericanOptionSolver<HostMemSpace> pricer(pricing_params);
    auto price_result = pricer.solve();
    ASSERT_TRUE(price_result.has_value());
    double market_price = price_result->price;

    IVQuery query{
        .strike = 100.0, .spot = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.02,
        .type = OptionType::Put,
        .market_price = market_price
    };

    IVSolverFDM<HostMemSpace> solver;
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_NEAR(result->implied_vol, 0.80, 0.01);
}

}  // namespace mango::kokkos::test
