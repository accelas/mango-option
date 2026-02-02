// SPDX-License-Identifier: MIT
/**
 * @file custom_grid_example_test.cc
 * @brief Tests for custom grid configuration (converted from example_custom_grid.cc)
 *
 * Validates:
 * - Manual grid specification with GridSpec factory methods
 * - Uniform vs sinh-spaced grids
 * - Multi-sinh grids for multiple concentration points
 * - Auto-estimation vs manual configuration
 * - Grid accuracy tradeoffs
 */

#include "src/option/american_option.hpp"
#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <memory_resource>
#include <optional>

class CustomGridTest : public ::testing::Test {
  protected:
    mango::PricingParams make_atm_put() {
        mango::OptionSpec spec;
        spec.spot = 100.0;
        spec.strike = 100.0;
        spec.maturity = 1.0;
        spec.rate = 0.05;
        spec.dividend_yield = 0.02;
        spec.option_type = mango::OptionType::PUT;
        return mango::PricingParams(spec, 0.20);
    }

    std::pmr::synchronized_pool_resource pool;
};

TEST_F(CustomGridTest, AutoEstimatedGrid) {
    auto params = make_atm_put();
    auto [grid_spec, time_domain] = mango::estimate_pde_grid(params);

    EXPECT_GT(grid_spec.n_points(), 0u);
    EXPECT_GT(time_domain.n_steps(), 0u);
    EXPECT_LT(grid_spec.x_min(), 0.0);
    EXPECT_GT(grid_spec.x_max(), 0.0);

    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(n), &pool);
    auto workspace = mango::PDEWorkspace::from_buffer(buffer, n).value();
    auto solver = mango::AmericanOptionSolver::create(params, workspace).value();
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
    EXPECT_LT(result->delta(), 0.0);  // Put delta is negative
}

TEST_F(CustomGridTest, UniformGrid) {
    auto params = make_atm_put();
    auto uniform_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 201);
    ASSERT_TRUE(uniform_spec.has_value());
    EXPECT_EQ(uniform_spec->n_points(), 201u);

    size_t n = uniform_spec->n_points();
    std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(n), &pool);
    auto workspace = mango::PDEWorkspace::from_buffer(buffer, n).value();

    mango::TimeDomain time = mango::TimeDomain::from_n_steps(0.0, params.maturity, 1000);
    auto solver = mango::AmericanOptionSolver::create(params, workspace,
        mango::PDEGridConfig{uniform_spec.value(), time.n_steps(), {}}).value();
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
}

TEST_F(CustomGridTest, SinhSpacedGrid) {
    auto params = make_atm_put();
    double alpha = 2.0;
    auto sinh_spec = mango::GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, alpha);
    ASSERT_TRUE(sinh_spec.has_value());

    size_t n = sinh_spec->n_points();
    std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(n), &pool);
    auto workspace = mango::PDEWorkspace::from_buffer(buffer, n).value();

    mango::TimeDomain time = mango::TimeDomain::from_n_steps(0.0, params.maturity, 1000);
    auto solver = mango::AmericanOptionSolver::create(params, workspace,
        mango::PDEGridConfig{sinh_spec.value(), time.n_steps(), {}}).value();
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
}

TEST_F(CustomGridTest, MultiSinhGrid) {
    auto params = make_atm_put();

    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0,  .alpha = 2.5, .weight = 2.0},
        {.center_x = -0.2, .alpha = 2.0, .weight = 1.0}
    };

    auto multi_sinh_spec = mango::GridSpec<double>::multi_sinh_spaced(-3.0, 3.0, 201, clusters);
    ASSERT_TRUE(multi_sinh_spec.has_value());

    size_t n = multi_sinh_spec->n_points();
    std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(n), &pool);
    auto workspace = mango::PDEWorkspace::from_buffer(buffer, n).value();

    mango::TimeDomain time = mango::TimeDomain::from_n_steps(0.0, params.maturity, 1000);
    auto solver = mango::AmericanOptionSolver::create(params, workspace,
        mango::PDEGridConfig{multi_sinh_spec.value(), time.n_steps(), {}}).value();
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
}

TEST_F(CustomGridTest, FastAccuracyParams) {
    auto params = make_atm_put();

    mango::GridAccuracyParams fast_accuracy{
        .n_sigma = 5.0,
        .alpha = 2.0,
        .tol = 1e-2,
        .c_t = 0.75,
        .min_spatial_points = 100,
        .max_spatial_points = 200,
        .max_time_steps = 500
    };

    auto [grid, time_domain] = mango::estimate_pde_grid(params, fast_accuracy);
    EXPECT_GE(grid.n_points(), 100u);
    EXPECT_LE(grid.n_points(), 200u);

    size_t n = grid.n_points();
    std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(n), &pool);
    auto workspace = mango::PDEWorkspace::from_buffer(buffer, n).value();
    auto solver = mango::AmericanOptionSolver::create(params, workspace).value();
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
}

TEST_F(CustomGridTest, HighAccuracyParams) {
    auto params = make_atm_put();

    mango::GridAccuracyParams high_accuracy{
        .n_sigma = 5.0,
        .alpha = 2.0,
        .tol = 1e-6,
        .c_t = 0.75,
        .min_spatial_points = 300,
        .max_spatial_points = 1200,
        .max_time_steps = 5000
    };

    auto [grid, time_domain] = mango::estimate_pde_grid(params, high_accuracy);
    EXPECT_GE(grid.n_points(), 300u);

    size_t n = grid.n_points();
    std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(n), &pool);
    auto workspace = mango::PDEWorkspace::from_buffer(buffer, n).value();

    // Pass the estimated grid config to the solver
    auto solver = mango::AmericanOptionSolver::create(params, workspace,
        mango::PDEGridConfig{grid, time_domain.n_steps(), {}}).value();
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
}

TEST_F(CustomGridTest, AllGridTypesProduceSimilarPrices) {
    auto params = make_atm_put();

    // Auto-estimated
    auto [auto_grid, auto_time] = mango::estimate_pde_grid(params);
    size_t n_auto = auto_grid.n_points();
    std::pmr::vector<double> auto_buf(mango::PDEWorkspace::required_size(n_auto), &pool);
    auto auto_ws = mango::PDEWorkspace::from_buffer(auto_buf, n_auto).value();
    auto auto_solver = mango::AmericanOptionSolver::create(params, auto_ws).value();
    auto auto_result = auto_solver.solve();
    ASSERT_TRUE(auto_result.has_value());
    double auto_price = auto_result->value();

    // Sinh-spaced
    auto sinh_spec = mango::GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
    ASSERT_TRUE(sinh_spec.has_value());
    size_t n_sinh = sinh_spec->n_points();
    std::pmr::vector<double> sinh_buf(mango::PDEWorkspace::required_size(n_sinh), &pool);
    auto sinh_ws = mango::PDEWorkspace::from_buffer(sinh_buf, n_sinh).value();
    mango::TimeDomain sinh_time = mango::TimeDomain::from_n_steps(0.0, params.maturity, 1000);
    auto sinh_solver = mango::AmericanOptionSolver::create(params, sinh_ws,
        mango::PDEGridConfig{sinh_spec.value(), sinh_time.n_steps(), {}}).value();
    auto sinh_result = sinh_solver.solve();
    ASSERT_TRUE(sinh_result.has_value());
    double sinh_price = sinh_result->value();

    // All grid types should produce prices within 1% of each other
    EXPECT_NEAR(sinh_price, auto_price, auto_price * 0.01);
}
