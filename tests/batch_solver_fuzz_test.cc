/**
 * @file batch_solver_fuzz_test.cc
 * @brief Fuzz tests for batch solver using FuzzTest
 *
 * These tests use property-based fuzzing to discover edge cases that
 * hand-written tests miss. FuzzTest generates random valid inputs and
 * verifies that invariants hold.
 *
 * Requires Earthly container with Clang + libc++ due to C++23 ABI issues.
 *
 * Run via:
 *   earthly +fuzz-test
 *
 * Or manually in a libc++ environment:
 *   bazel test --config=fuzz //tests:batch_solver_fuzz_test
 */

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"
#include "src/option/american_option_batch.hpp"
#include "src/pde/core/grid.hpp"

using namespace mango;

// ============================================================================
// Property: Batch solver should never crash for valid inputs
// ============================================================================

void BatchSolverNeverCrashes(
    size_t n_points,
    size_t batch_size,
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    bool use_shared_grid)
{
    // Skip invalid configurations that are expected to fail
    if (n_points < 11 || n_points > 1001) return;  // Valid grid range
    if (batch_size < 1 || batch_size > 100) return;
    if (spot <= 0 || strike <= 0 || maturity <= 0 || volatility <= 0) return;

    // Create grid spec
    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, n_points, 2.0);
    if (!grid_spec_result.has_value()) return;

    // Create batch of options with varying strikes
    std::vector<AmericanOptionParams> params;
    double strike_step = 0.5;
    for (size_t i = 0; i < batch_size; ++i) {
        double K = strike + i * strike_step;
        if (K <= 0) continue;
        params.push_back(PricingParams(
            spot, K, maturity, rate, 0.0,
            OptionType::PUT, volatility, {}));
    }

    if (params.empty()) return;

    // Create custom grid config
    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, maturity, 500);
    auto custom_grid = std::make_pair(grid_spec_result.value(), time_domain);

    // This should never crash
    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, use_shared_grid, nullptr, custom_grid);

    // Basic invariants
    EXPECT_EQ(results.results.size(), params.size());
}

FUZZ_TEST(BatchSolverFuzz, BatchSolverNeverCrashes)
    .WithDomains(
        fuzztest::InRange<size_t>(11, 501),     // n_points
        fuzztest::InRange<size_t>(1, 50),       // batch_size
        fuzztest::InRange(50.0, 200.0),         // spot
        fuzztest::InRange(50.0, 200.0),         // strike
        fuzztest::InRange(0.01, 3.0),           // maturity
        fuzztest::InRange(0.05, 1.0),           // volatility
        fuzztest::InRange(-0.05, 0.20),         // rate
        fuzztest::Arbitrary<bool>()             // use_shared_grid
    );

// ============================================================================
// Property: Option prices should be non-negative
// ============================================================================

void OptionPricesNonNegative(
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    double dividend_yield,
    bool is_call)
{
    if (spot <= 0 || strike <= 0 || maturity <= 0 || volatility <= 0) return;

    std::vector<AmericanOptionParams> params;
    params.push_back(PricingParams(
        spot, strike, maturity, rate, dividend_yield,
        is_call ? OptionType::CALL : OptionType::PUT,
        volatility, {}));

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, false);

    ASSERT_EQ(results.results.size(), 1);
    if (results.results[0].has_value()) {
        EXPECT_GE(results.results[0]->value(), 0.0)
            << "Option price must be non-negative";
    }
}

FUZZ_TEST(BatchSolverFuzz, OptionPricesNonNegative)
    .WithDomains(
        fuzztest::InRange(10.0, 500.0),         // spot
        fuzztest::InRange(10.0, 500.0),         // strike
        fuzztest::InRange(0.01, 5.0),           // maturity
        fuzztest::InRange(0.01, 2.0),           // volatility
        fuzztest::InRange(-0.10, 0.30),         // rate
        fuzztest::InRange(0.0, 0.10),           // dividend_yield
        fuzztest::Arbitrary<bool>()             // is_call
    );

// ============================================================================
// Property: Put-call parity bounds (American options)
// ============================================================================
// For American options:
//   Call >= max(0, S - K * exp(-rT))
//   Put >= max(0, K * exp(-rT) - S)

void AmericanOptionLowerBounds(
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    bool is_call)
{
    if (spot <= 0 || strike <= 0 || maturity <= 0.01 || volatility < 0.05) return;

    std::vector<AmericanOptionParams> params;
    params.push_back(PricingParams(
        spot, strike, maturity, rate, 0.0,
        is_call ? OptionType::CALL : OptionType::PUT,
        volatility, {}));

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, false);

    ASSERT_EQ(results.results.size(), 1);
    if (!results.results[0].has_value()) return;

    double price = results.results[0]->value();
    double discount = std::exp(-rate * maturity);

    if (is_call) {
        double lower_bound = std::max(0.0, spot - strike * discount);
        EXPECT_GE(price, lower_bound - 0.01)
            << "Call price violates lower bound";
    } else {
        double lower_bound = std::max(0.0, strike * discount - spot);
        EXPECT_GE(price, lower_bound - 0.01)
            << "Put price violates lower bound";
    }
}

FUZZ_TEST(BatchSolverFuzz, AmericanOptionLowerBounds)
    .WithDomains(
        fuzztest::InRange(50.0, 150.0),         // spot
        fuzztest::InRange(50.0, 150.0),         // strike
        fuzztest::InRange(0.1, 2.0),            // maturity
        fuzztest::InRange(0.10, 0.60),          // volatility
        fuzztest::InRange(-0.02, 0.15),         // rate
        fuzztest::Arbitrary<bool>()             // is_call
    );

// ============================================================================
// Property: Grid size consistency (critical for issue #272)
// ============================================================================

void GridSizeConsistency(size_t n_points, size_t batch_size, bool use_shared_grid)
{
    if (n_points < 21 || n_points > 301) return;
    if (batch_size < 1 || batch_size > 20) return;

    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, n_points, 2.0);
    if (!grid_spec_result.has_value()) return;

    std::vector<AmericanOptionParams> params;
    for (size_t i = 0; i < batch_size; ++i) {
        params.push_back(PricingParams(
            100.0, 90.0 + i * 2.0, 1.0, 0.05, 0.02,
            OptionType::PUT, 0.20, {}));
    }

    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, 1.0, 500);
    auto custom_grid = std::make_pair(grid_spec_result.value(), time_domain);

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, use_shared_grid, nullptr, custom_grid);

    // The key invariant: no InvalidConfiguration errors from grid mismatch
    // This was the bug in issue #272
    EXPECT_EQ(results.failed_count, 0)
        << "Grid size " << n_points << " with batch " << batch_size
        << " (shared=" << use_shared_grid << ") caused failures";

    for (size_t i = 0; i < results.results.size(); ++i) {
        if (!results.results[i].has_value()) {
            auto err = results.results[i].error();
            EXPECT_NE(err.code, SolverErrorCode::InvalidConfiguration)
                << "InvalidConfiguration at index " << i
                << " indicates grid size mismatch (issue #272 regression)";
        }
    }
}

FUZZ_TEST(BatchSolverFuzz, GridSizeConsistency)
    .WithDomains(
        fuzztest::InRange<size_t>(21, 301),     // n_points (odd values work best)
        fuzztest::InRange<size_t>(1, 20),       // batch_size
        fuzztest::Arbitrary<bool>()             // use_shared_grid
    );

// ============================================================================
// Standard unit test to verify fuzz test setup works
// ============================================================================

TEST(BatchSolverFuzz, SmokeTest) {
    // Simple test to verify the fuzz test infrastructure compiles and runs
    BatchSolverNeverCrashes(101, 5, 100.0, 100.0, 1.0, 0.20, 0.05, true);
    OptionPricesNonNegative(100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false);
    AmericanOptionLowerBounds(100.0, 100.0, 1.0, 0.20, 0.05, false);
    GridSizeConsistency(101, 5, true);
}
