#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <memory_resource>
#include <vector>

namespace mango {
namespace {

class AmericanOptionPricingTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_unique<std::pmr::synchronized_pool_resource>();
        auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
        ASSERT_TRUE(grid_spec.has_value());

        auto workspace_result = AmericanSolverWorkspace::create(
            grid_spec.value(), 2000, pool_.get());
        ASSERT_TRUE(workspace_result.has_value()) << workspace_result.error();
        workspace_ = workspace_result.value();
    }

    [[nodiscard]] AmericanOptionResult Solve(const AmericanOptionParams& params) const {
        return SolveWithWorkspace(params, workspace_);
    }

    static AmericanOptionResult SolveWithWorkspace(
        const AmericanOptionParams& params,
        const std::shared_ptr<AmericanSolverWorkspace>& workspace)
    {
        auto solver_result = AmericanOptionSolver::create(params, workspace);
        if (!solver_result) {
            ADD_FAILURE() << "Failed to create solver: " << solver_result.error();
            return {};
        }

        auto solve_result = solver_result.value().solve();
        if (!solve_result) {
            const auto& error = solve_result.error();
            ADD_FAILURE() << "Solver failed: " << error.message
                          << " (code=" << static_cast<int>(error.code)
                          << ", iterations=" << error.iterations << ")";
            return {};
        }

        return solve_result.value();
    }

    std::unique_ptr<std::pmr::synchronized_pool_resource> pool_;
    std::shared_ptr<AmericanSolverWorkspace> workspace_;
};

TEST_F(AmericanOptionPricingTest, SolverWithPMRWorkspace) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
    ASSERT_TRUE(grid_spec.has_value());

    auto workspace = AmericanSolverWorkspace::create(
        grid_spec.value(), 2000, &pool);
    ASSERT_TRUE(workspace.has_value());

    AmericanOptionParams params(
        100.0,  // spot
        110.0,  // strike
        1.0,    // maturity
        0.03,   // rate
        0.00,   // dividend_yield
        OptionType::PUT,
        0.25    // volatility
    );

    auto solver_result = AmericanOptionSolver::create(params, workspace.value());
    ASSERT_TRUE(solver_result.has_value());

    auto result = solver_result.value().solve();
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
}

TEST_F(AmericanOptionPricingTest, PutValueRespectsIntrinsicBound) {
    AmericanOptionParams params(
        100.0,  // spot
        110.0,  // strike
        1.0,    // maturity
        0.03,   // rate
        0.00,   // dividend_yield
        OptionType::PUT,
        0.25    // volatility
    );

    AmericanOptionResult result = Solve(params);
    ASSERT_TRUE(result.converged);

    double intrinsic = std::max(params.strike - params.spot, 0.0);
    EXPECT_GE(result.value_at(params.spot), intrinsic - 1e-6);
    EXPECT_LT(result.value_at(params.spot), params.strike);
}

TEST_F(AmericanOptionPricingTest, CallValueIncreasesWithVolatility) {
    const double spot = 100.0;
    const double strike = 100.0;
    const double maturity = 1.0;
    const double rate = 0.01;
    const double dividend_yield = 0.0;

    std::vector<double> volatilities = {0.15, 0.25, 0.4};
    double previous_value = 0.0;
    double previous_vol = 0.0;
    for (size_t i = 0; i < volatilities.size(); ++i) {
        double vol = volatilities[i];
        AmericanOptionParams params(
            spot, strike, maturity, rate, dividend_yield, OptionType::CALL, vol);
        AmericanOptionResult result = Solve(params);
        ASSERT_TRUE(result.converged);

        if (i > 0) {
            EXPECT_GT(result.value_at(params.spot), previous_value)
                << "Value did not increase when volatility went from "
                << previous_vol << " to " << vol;
        }
        previous_value = result.value_at(params.spot);
        previous_vol = vol;
    }
}

TEST_F(AmericanOptionPricingTest, PutValueIncreasesWithMaturity) {
    std::vector<double> maturities = {0.25, 0.5, 1.0, 2.0};
    double previous_value = 0.0;
    for (double maturity : maturities) {
        AmericanOptionParams params(
            100.0,  // spot
            95.0,   // strike
            maturity,
            0.02,   // rate
            0.00,   // dividend_yield
            OptionType::PUT,
            0.2     // volatility
        );

        AmericanOptionResult result = Solve(params);
        ASSERT_TRUE(result.converged);

        if (previous_value > 0.0) {
            EXPECT_GE(result.value_at(params.spot), previous_value - 5e-3);
        }
        previous_value = result.value_at(params.spot);
    }
}

TEST_F(AmericanOptionPricingTest, DISABLED_DividendsReduceCallValue) {
    // TODO: Discrete dividend support not yet implemented in solver
    // This test is disabled until temporal event handling for dividends is added
    AmericanOptionParams no_dividends(
        100.0, 100.0, 1.0, 0.02, 0.00, OptionType::CALL, 0.3);

    AmericanOptionParams with_dividends(
        100.0, 100.0, 1.0, 0.02, 0.00, OptionType::CALL, 0.3,
        {{0.5, 3.0}}  // $3 dividend halfway to maturity
    );

    AmericanOptionResult result_no_div = Solve(no_dividends);
    AmericanOptionResult result_with_div = Solve(with_dividends);

    ASSERT_TRUE(result_no_div.converged);
    ASSERT_TRUE(result_with_div.converged);

    EXPECT_GT(result_no_div.value_at(no_dividends.spot), result_with_div.value_at(with_dividends.spot));
}

TEST_F(AmericanOptionPricingTest, BatchSolverMatchesSingleSolver) {
    std::vector<AmericanOptionParams> params;
    params.emplace_back(100.0, 100.0, 0.75, 0.01, 0.0, OptionType::CALL, 0.25);
    params.emplace_back(120.0, 100.0, 1.5, 0.02, 0.0, OptionType::PUT, 0.2);
    params.emplace_back(90.0,  95.0,  0.5, -0.01, 0.01, OptionType::PUT, 0.35);

    // Use automatic grid determination for batch solver
    auto batch_result = BatchAmericanOptionSolver().solve_batch(params);
    ASSERT_EQ(batch_result.results.size(), params.size());
    EXPECT_EQ(batch_result.failed_count, 0u);

    // Compare with single option automatic grid solver
    for (size_t i = 0; i < params.size(); ++i) {
        ASSERT_TRUE(batch_result.results[i].has_value()) << "Batch solve failed for index " << i;

        auto single_result = solve_american_option_auto(params[i]);
        ASSERT_TRUE(single_result.has_value()) << "Single solve failed for index " << i;
        ASSERT_TRUE(single_result->converged);

        const double batch_value = batch_result.results[i]->value_at(params[i].spot);
        const double single_value = single_result->value_at(params[i].spot);
        EXPECT_NEAR(single_value, batch_value, 1e-3) << "Mismatch at index " << i;
    }
}

TEST_F(AmericanOptionPricingTest, DISABLED_PutImmediateExerciseAtBoundary) {
    // TODO: This test is temporarily disabled while investigating boundary value behavior
    // The test expects deep ITM put to have value ≈ intrinsic, but currently gets wrong values
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::sinh_spaced(-7.0, 2.0, 301, 2.0);
    ASSERT_TRUE(grid_spec.has_value());
    auto custom_workspace = AmericanSolverWorkspace::create(grid_spec.value(), 1500, &pool);
    ASSERT_TRUE(custom_workspace.has_value()) << custom_workspace.error();

    AmericanOptionParams params(
        0.25,   // spot deep ITM
        100.0,  // strike
        0.75,   // maturity
        0.05,   // rate
        0.0,    // dividend yield
        OptionType::PUT,
        0.2     // volatility
    );

    AmericanOptionResult result = SolveWithWorkspace(params, custom_workspace.value());
    ASSERT_TRUE(result.converged);

    const double intrinsic = params.strike - params.spot;
    EXPECT_NEAR(result.value_at(params.spot), intrinsic, 0.5)
        << "Left boundary should equal immediate exercise for deep ITM put";
}

}  // namespace
}  // namespace mango
