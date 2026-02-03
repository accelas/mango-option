// SPDX-License-Identifier: MIT
/**
 * @file american_option_new_api_test.cc
 * @brief Tests for new AmericanOptionSolver API with PDEWorkspace
 */

#include "mango/option/american_option.hpp"
#include "mango/option/american_option_result.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include "mango/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <memory_resource>

namespace mango {
namespace {

class AmericanOptionNewAPITest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_unique<std::pmr::synchronized_pool_resource>();
    }

    std::unique_ptr<std::pmr::synchronized_pool_resource> pool_;
};

TEST_F(AmericanOptionNewAPITest, SolveWithPDEWorkspace) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    // Create workspace matching grid size
    size_t n_space = 101;
    size_t workspace_size = PDEWorkspace::required_size(n_space);
    std::pmr::vector<double> buffer(workspace_size, pool_.get());

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n_space);
    ASSERT_TRUE(workspace_result.has_value()) << workspace_result.error();

    // NEW API: Pass PDEWorkspace directly
    auto solver = AmericanOptionSolver::create(params, workspace_result.value()).value();
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value()) << "Solver failed: "
        << static_cast<int>(result.error().code);

    // Verify result is the NEW wrapper type
    EXPECT_GT(result->value(), 0.0);
    EXPECT_LT(result->value(), params.strike);

    // Verify pricing params are accessible
    EXPECT_DOUBLE_EQ(result->spot(), 100.0);
    EXPECT_DOUBLE_EQ(result->strike(), 100.0);
    EXPECT_EQ(result->option_type(), OptionType::PUT);
}

TEST_F(AmericanOptionNewAPITest, SolveWithSnapshots) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    size_t n_space = 101;
    size_t workspace_size = PDEWorkspace::required_size(n_space);
    std::pmr::vector<double> buffer(workspace_size, pool_.get());

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n_space);
    ASSERT_TRUE(workspace_result.has_value());

    // NEW API: Pass snapshot times
    std::vector<double> snapshot_times = {0.0, 0.5, 1.0};
    auto solver = AmericanOptionSolver::create(params, workspace_result.value(), std::nullopt, std::span<const double>(snapshot_times)).value();

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Verify snapshots were recorded
    EXPECT_TRUE(result->has_snapshots());
    EXPECT_EQ(result->num_snapshots(), 3);

    auto times = result->snapshot_times();
    EXPECT_EQ(times.size(), 3);
}

TEST_F(AmericanOptionNewAPITest, CallOptionWithNewAPI) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::CALL}, 0.20);

    size_t n_space = 101;
    size_t workspace_size = PDEWorkspace::required_size(n_space);
    std::pmr::vector<double> buffer(workspace_size, pool_.get());

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n_space);
    ASSERT_TRUE(workspace_result.has_value());

    auto solver = AmericanOptionSolver::create(params, workspace_result.value()).value();
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
    EXPECT_EQ(result->option_type(), OptionType::CALL);
}

TEST_F(AmericanOptionNewAPITest, ValueAtInterpolation) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    size_t n_space = 101;
    size_t workspace_size = PDEWorkspace::required_size(n_space);
    std::pmr::vector<double> buffer(workspace_size, pool_.get());

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n_space);
    ASSERT_TRUE(workspace_result.has_value());

    auto solver = AmericanOptionSolver::create(params, workspace_result.value()).value();
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());

    // Test interpolation at different spot prices
    double value_atm = result->value_at(100.0);
    double value_itm = result->value_at(90.0);
    double value_otm = result->value_at(110.0);

    // Put option: ITM > ATM > OTM
    EXPECT_GT(value_itm, value_atm);
    EXPECT_GT(value_atm, value_otm);
}

TEST_F(AmericanOptionNewAPITest, GreeksComputation) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    size_t n_space = 101;
    size_t workspace_size = PDEWorkspace::required_size(n_space);
    std::pmr::vector<double> buffer(workspace_size, pool_.get());

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n_space);
    ASSERT_TRUE(workspace_result.has_value());

    auto solver = AmericanOptionSolver::create(params, workspace_result.value()).value();
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());

    // Greeks should be computable
    double delta = result->delta();
    double gamma = result->gamma();

    // Put delta: negative, in range [-1, 0]
    EXPECT_LE(delta, 0.0);
    EXPECT_GE(delta, -1.0);

    // Gamma: positive
    EXPECT_GE(gamma, 0.0);
}

} // namespace
} // namespace mango
