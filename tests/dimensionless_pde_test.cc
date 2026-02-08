// SPDX-License-Identifier: MIT
/**
 * @file dimensionless_pde_test.cc
 * @brief Validate PDE equivalence between physical and dimensionless parameters
 *
 * Mathematical claim: The Black-Scholes PDE with physical parameters (sigma, r, q=0)
 * produces the same normalized price V/K as solving with dimensionless mapping:
 *
 *   sigma_eff = sqrt(2)
 *   r_eff     = kappa = 2*r / sigma^2
 *   q_eff     = 0
 *   T_eff     = tau'  = sigma^2 * T / 2
 *
 * Both parameterizations use the same spatial grid (same log-moneyness domain
 * and point count) to eliminate discretization differences and isolate the
 * mathematical equivalence being tested.
 */

#include "mango/option/american_option.hpp"
#include "mango/option/table/price_table_surface.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace mango {
namespace {

// ---------------------------------------------------------------------------
// Build a common grid spec for a given domain.  Both the physical and
// dimensionless solves share the same spatial grid (log-moneyness domain)
// so that discretization error does not mask the mathematical identity.
// ---------------------------------------------------------------------------
GridSpec<double> make_common_grid(double x_min, double x_max, size_t n_points) {
    constexpr double alpha = 3.95;  // optimal_sinh_alpha(5.0)
    auto spec = GridSpec<double>::multi_sinh_spaced(x_min, x_max, n_points, {
        {.center_x = 0.0, .alpha = alpha, .weight = 1.0},
    });
    return spec.value();
}

// ---------------------------------------------------------------------------
// Solve with explicit grid config and return value_at(S) / K
// ---------------------------------------------------------------------------
double solve_with_grid(const PricingParams& params,
                       const GridSpec<double>& grid_spec,
                       size_t n_time,
                       double spot) {
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n);
    EXPECT_TRUE(workspace.has_value());

    auto solver = AmericanOptionSolver::create(
        params, workspace.value(),
        PDEGridConfig{.grid_spec = grid_spec, .n_time = n_time});
    EXPECT_TRUE(solver.has_value()) << "Solver creation failed";

    auto result = solver->solve();
    EXPECT_TRUE(result.has_value()) << "Solve failed";
    return result->value_at(spot) / params.strike;
}

// ---------------------------------------------------------------------------
// Solve physical and dimensionless with matched grids
// ---------------------------------------------------------------------------
struct EquivResult {
    double physical;
    double dimensionless;
};

EquivResult solve_both(double spot, double strike, double maturity,
                       double sigma, double rate, OptionType type) {
    // Derived dimensionless parameters
    const double kappa = 2.0 * rate / (sigma * sigma);
    const double tau_prime = sigma * sigma * maturity / 2.0;
    const double sigma_eff = std::sqrt(2.0);

    // Common spatial grid: +-5 sigma_sqrt_T around x=0, 401 points
    const double sigma_sqrt_T = sigma * std::sqrt(maturity);
    const double x_min = -5.0 * sigma_sqrt_T;
    const double x_max = 5.0 * sigma_sqrt_T;
    constexpr size_t n_spatial = 401;
    auto grid_spec = make_common_grid(x_min, x_max, n_spatial);

    // Time steps: use the finer of the two maturity-based counts
    // to ensure both solves have adequate temporal resolution.
    // At least 2000 steps for the physical solve (longer maturity).
    const size_t n_time_phys = 2000;
    // Scale dimensionless time steps by tau'/T ratio (same dt effectively)
    const size_t n_time_dim = std::max<size_t>(
        200, static_cast<size_t>(std::ceil(n_time_phys * tau_prime / maturity)));

    PricingParams phys_params(
        OptionSpec{.spot = spot, .strike = strike, .maturity = maturity,
                   .rate = rate, .dividend_yield = 0.0, .option_type = type},
        sigma);

    PricingParams dim_params(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau_prime,
                   .rate = kappa, .dividend_yield = 0.0, .option_type = type},
        sigma_eff);

    return {
        .physical = solve_with_grid(phys_params, grid_spec, n_time_phys, spot),
        .dimensionless = solve_with_grid(dim_params, grid_spec, n_time_dim, spot),
    };
}

// ===========================================================================
// Test 1: Put equivalence at multiple moneyness levels
// sigma=0.30, r=0.06, K=100, T=1.0 -> kappa=1.333, tau'=0.045
// ===========================================================================
TEST(DimensionlessPDETest, PutEquivalenceAtMultipleMoneyness) {
    const double sigma = 0.30;
    const double rate = 0.06;
    const double strike = 100.0;
    const double maturity = 1.0;
    const double kappa = 2.0 * rate / (sigma * sigma);
    const double tau_prime = sigma * sigma * maturity / 2.0;

    // Verify derived parameters
    EXPECT_NEAR(kappa, 1.333, 0.001);
    EXPECT_NEAR(tau_prime, 0.045, 0.001);

    const std::vector<double> moneyness_levels = {0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20};
    const double tol = 5e-4;

    for (double m : moneyness_levels) {
        const double spot = m * strike;
        auto [v_physical, v_dimensionless] = solve_both(
            spot, strike, maturity, sigma, rate, OptionType::PUT);

        EXPECT_NEAR(v_physical, v_dimensionless, tol)
            << "Put equivalence failed at moneyness=" << m
            << " (S=" << spot << ", K=" << strike << ")"
            << "\n  physical V/K = " << v_physical
            << "\n  dimensionless V/K = " << v_dimensionless
            << "\n  kappa = " << kappa << ", tau' = " << tau_prime;
    }
}

// ===========================================================================
// Test 2: Call equivalence at multiple moneyness levels
// sigma=0.25, r=0.04, K=100, T=0.5 -> kappa=1.28, tau'=0.015625
// ===========================================================================
TEST(DimensionlessPDETest, CallEquivalenceAtMultipleMoneyness) {
    const double sigma = 0.25;
    const double rate = 0.04;
    const double strike = 100.0;
    const double maturity = 0.5;
    const double kappa = 2.0 * rate / (sigma * sigma);
    const double tau_prime = sigma * sigma * maturity / 2.0;

    // Verify derived parameters
    EXPECT_NEAR(kappa, 1.28, 0.001);
    EXPECT_NEAR(tau_prime, 0.015625, 0.0001);

    const std::vector<double> moneyness_levels = {0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20};
    const double tol = 5e-4;

    for (double m : moneyness_levels) {
        const double spot = m * strike;
        auto [v_physical, v_dimensionless] = solve_both(
            spot, strike, maturity, sigma, rate, OptionType::CALL);

        EXPECT_NEAR(v_physical, v_dimensionless, tol)
            << "Call equivalence failed at moneyness=" << m
            << " (S=" << spot << ", K=" << strike << ")"
            << "\n  physical V/K = " << v_physical
            << "\n  dimensionless V/K = " << v_dimensionless
            << "\n  kappa = " << kappa << ", tau' = " << tau_prime;
    }
}

// ===========================================================================
// Test 3: Equivalence across multiple kappa values (ATM only)
// Tests 5 different (sigma, r) pairs to sweep kappa space
// ===========================================================================
TEST(DimensionlessPDETest, EquivalenceAcrossMultipleKappaValues) {
    struct TestCase {
        double sigma;
        double rate;
    };

    const std::vector<TestCase> cases = {
        {0.10, 0.02},  // kappa = 4.0
        {0.20, 0.05},  // kappa = 2.5
        {0.30, 0.08},  // kappa = 1.778
        {0.40, 0.03},  // kappa = 0.375
        {0.50, 0.10},  // kappa = 0.8
    };

    const double strike = 100.0;
    const double spot = 100.0;  // ATM
    const double maturity = 1.0;
    const double tol = 1e-3;

    for (const auto& tc : cases) {
        const double kappa = 2.0 * tc.rate / (tc.sigma * tc.sigma);
        const double tau_prime = tc.sigma * tc.sigma * maturity / 2.0;

        // Test put
        auto [v_phys_put, v_dim_put] = solve_both(
            spot, strike, maturity, tc.sigma, tc.rate, OptionType::PUT);

        EXPECT_NEAR(v_phys_put, v_dim_put, tol)
            << "ATM put equivalence failed for sigma=" << tc.sigma << " r=" << tc.rate
            << "\n  kappa = " << kappa << ", tau' = " << tau_prime
            << "\n  physical V/K = " << v_phys_put
            << "\n  dimensionless V/K = " << v_dim_put;

        // Test call
        auto [v_phys_call, v_dim_call] = solve_both(
            spot, strike, maturity, tc.sigma, tc.rate, OptionType::CALL);

        EXPECT_NEAR(v_phys_call, v_dim_call, tol)
            << "ATM call equivalence failed for sigma=" << tc.sigma << " r=" << tc.rate
            << "\n  kappa = " << kappa << ", tau' = " << tau_prime
            << "\n  physical V/K = " << v_phys_call
            << "\n  dimensionless V/K = " << v_dim_call;
    }
}

// ===========================================================================
// Compile test: PriceTableSurfaceND<3> instantiation via alias
// ===========================================================================
TEST(DimensionlessSurface, SurfaceND3Compiles) {
    // Verify PriceTableSurfaceND<3> can be instantiated via axes
    PriceTableAxesND<3> axes;
    axes.grids[0] = {-1.0, -0.5, 0.0, 0.5, 1.0};       // log-moneyness
    axes.grids[1] = {0.0, 0.01, 0.02, 0.04, 0.08};      // tau_prime
    axes.grids[2] = {-2.0, -1.0, 0.0, 1.0, 2.0};        // ln_kappa
    axes.names = {"log_moneyness", "tau_prime", "ln_kappa"};

    auto validate_result = axes.validate();
    EXPECT_TRUE(validate_result.has_value());

    auto shape = axes.shape();
    EXPECT_EQ(shape[0], 5u);
    EXPECT_EQ(shape[1], 5u);
    EXPECT_EQ(shape[2], 5u);
}

}  // namespace
}  // namespace mango
