#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
#include "../src/american_option.h"
#include "../src/price_table.h"
}

// Test fixture for unified grid tests
class UnifiedGridTest : public ::testing::Test {
protected:
    AmericanOptionGrid default_grid = {
        .x_min = -0.7,      // ln(0.5) ≈ -0.7 (50% of strike)
        .x_max = 0.7,       // ln(2.0) ≈ 0.7 (200% of strike)
        .n_points = 101,    // Number of spatial grid points
        .dt = 0.001,
        .n_steps = 500
    };
};

// Test basic solve correctness on provided moneyness grid
TEST_F(UnifiedGridTest, BasicSolveCorrectness) {
    double m_grid[] = {0.8, 0.9, 1.0, 1.1, 1.2};
    const size_t n_m = 5;

    OptionData option = {
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.25,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_solve(
        &option, m_grid, n_m, 0.001, 250
    );

    EXPECT_EQ(result.status, 0);
    ASSERT_NE(result.solver, nullptr);

    const double *solution = pde_solver_get_solution(result.solver);
    ASSERT_NE(solution, nullptr);

    // ATM (m=1.0) should have positive value
    EXPECT_GT(solution[2], 0.0);

    // Monotonicity: put prices decrease with increasing moneyness
    for (size_t i = 1; i < n_m; i++) {
        EXPECT_LT(solution[i], solution[i-1])
            << "Put price should decrease with moneyness at indices "
            << i-1 << " and " << i;
    }

    // OTM puts (m > 1.0) should be worth less than ITM puts (m < 1.0)
    EXPECT_LT(solution[3], solution[1]);  // m=1.1 < m=0.9
    EXPECT_LT(solution[4], solution[0]);  // m=1.2 < m=0.8

    american_option_free_result(&result);
}

// Test that unified grid produces same results as legacy API
TEST_F(UnifiedGridTest, EquivalenceWithLegacyAPI) {
    double m_grid[] = {0.85, 0.95, 1.0, 1.05, 1.15};
    const size_t n_m = 5;

    OptionData option = {
        .strike = 100.0,
        .volatility = 0.25,
        .risk_free_rate = 0.03,
        .time_to_maturity = 0.5,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    // Solve with unified grid
    AmericanOptionResult unified_result = american_option_solve(
        &option, m_grid, n_m, 0.001, 500
    );
    EXPECT_EQ(unified_result.status, 0);
    const double *unified_solution = pde_solver_get_solution(unified_result.solver);

    // Solve with legacy API (automatic grid)
    AmericanOptionResult legacy_result = american_option_price(&option, &default_grid);
    EXPECT_EQ(legacy_result.status, 0);

    // Compare ATM value (should be similar)
    double unified_atm = unified_solution[2];  // m=1.0 is at index 2
    double legacy_atm = american_option_get_value_at_spot(
        legacy_result.solver, 100.0, option.strike
    );

    // Allow 5% tolerance due to different grid spacing
    double rel_diff = std::abs(unified_atm - legacy_atm) / legacy_atm;
    EXPECT_LT(rel_diff, 0.05)
        << "Unified ATM: " << unified_atm << ", Legacy ATM: " << legacy_atm;

    american_option_free_result(&unified_result);
    american_option_free_result(&legacy_result);
}

// Test zero-copy property: solution lives on exact grid provided
TEST_F(UnifiedGridTest, ZeroCopyProperty) {
    double m_grid[] = {0.7, 0.85, 1.0, 1.15, 1.3};
    const size_t n_m = 5;

    OptionData option = {
        .strike = 100.0,
        .volatility = 0.30,
        .risk_free_rate = 0.04,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_solve(
        &option, m_grid, n_m, 0.001, 1000
    );
    EXPECT_EQ(result.status, 0);

    const double *solution = pde_solver_get_solution(result.solver);
    const double *grid = pde_solver_get_grid(result.solver);

    // Verify zero-copy property: grid used by solver matches input grid
    // This confirms no interpolation overhead or grid reallocation
    EXPECT_EQ(grid, m_grid) << "Solver should use exact grid provided (zero-copy)";

    // Verify solution makes sense at boundaries
    // Deep ITM put (m=0.7): should be close to intrinsic value
    double spot_itm = m_grid[0] * option.strike;
    double intrinsic_itm = option.strike - spot_itm;
    EXPECT_GT(solution[0], intrinsic_itm * 0.9);  // At least 90% of intrinsic

    // Deep OTM put (m=1.3): should be close to zero
    EXPECT_LT(solution[4], 3.0);  // Very small value

    american_option_free_result(&result);
}

// Test grid with non-uniform spacing (log-spaced)
TEST_F(UnifiedGridTest, NonUniformGrid) {
    // Log-spaced grid (more points near ATM)
    std::vector<double> m_grid;
    const double m_min = 0.7;
    const double m_max = 1.3;
    const size_t n_m = 11;

    double log_min = std::log(m_min);
    double log_max = std::log(m_max);
    for (size_t i = 0; i < n_m; i++) {
        double log_m = log_min + i * (log_max - log_min) / (n_m - 1);
        m_grid.push_back(std::exp(log_m));
    }

    OptionData option = {
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.5,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_solve(
        &option, m_grid.data(), n_m, 0.001, 500
    );

    EXPECT_EQ(result.status, 0);
    const double *solution = pde_solver_get_solution(result.solver);

    // Verify monotonicity still holds
    for (size_t i = 1; i < n_m; i++) {
        EXPECT_LT(solution[i], solution[i-1]);
    }

    // Verify ATM region has reasonable values
    size_t atm_idx = n_m / 2;  // Middle of grid should be near ATM
    EXPECT_GT(solution[atm_idx], 1.0);
    EXPECT_LT(solution[atm_idx], 15.0);

    american_option_free_result(&result);
}

// Test fine grid (many points)
TEST_F(UnifiedGridTest, FineGrid) {
    const size_t n_m = 51;
    std::vector<double> m_grid(n_m);

    // Linear spacing
    for (size_t i = 0; i < n_m; i++) {
        m_grid[i] = 0.7 + i * (1.3 - 0.7) / (n_m - 1);
    }

    OptionData option = {
        .strike = 100.0,
        .volatility = 0.25,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.25,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_solve(
        &option, m_grid.data(), n_m, 0.001, 250
    );

    EXPECT_EQ(result.status, 0);
    const double *solution = pde_solver_get_solution(result.solver);

    // Fine grid should produce smooth solution
    // Check that changes are gradual (no large jumps)
    for (size_t i = 1; i < n_m - 1; i++) {
        double change1 = std::abs(solution[i] - solution[i-1]);
        double change2 = std::abs(solution[i+1] - solution[i]);

        // Changes should be relatively similar (smoothness)
        if (change1 > 0.1 && change2 > 0.1) {
            double ratio = std::max(change1, change2) / std::min(change1, change2);
            EXPECT_LT(ratio, 3.0) << "Large discontinuity at index " << i;
        }
    }

    american_option_free_result(&result);
}

// Test error handling: unsorted grid
TEST_F(UnifiedGridTest, UnsortedGridError) {
    double m_grid[] = {0.8, 1.0, 0.9, 1.1, 1.2};  // Unsorted!
    const size_t n_m = 5;

    OptionData option = {
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.25,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_solve(
        &option, m_grid, n_m, 0.001, 250
    );

    // Should fail with unsorted grid
    EXPECT_NE(result.status, 0);

    american_option_free_result(&result);
}

// Test minimum grid size
TEST_F(UnifiedGridTest, MinimumGridSize) {
    double m_grid[] = {0.9, 1.0, 1.1};  // Only 3 points
    const size_t n_m = 3;

    OptionData option = {
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.25,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_solve(
        &option, m_grid, n_m, 0.001, 250
    );

    // Should succeed (minimum valid grid)
    EXPECT_EQ(result.status, 0);
    const double *solution = pde_solver_get_solution(result.solver);

    // Basic sanity checks
    EXPECT_GT(solution[0], solution[1]);  // m=0.9 > m=1.0
    EXPECT_GT(solution[1], solution[2]);  // m=1.0 > m=1.1

    american_option_free_result(&result);
}
