#include <gtest/gtest.h>
#include "src/option/iv_solver_fdm.hpp"
#include "src/option/iv_result.hpp"
#include "src/support/error_types.hpp"

using namespace mango;

// Test for new std::expected signature (Task 2.1)
// NOTE: This test calls solve_impl() directly to verify the new signature,
// bypassing the base class which still returns IVResult (will be updated in Task 3).
TEST(IVSolverFDMExpected, ReturnsExpectedType) {
    // Simple test to verify std::expected signature compiles
    IVQuery query{
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::PUT,
        10.0    // market_price
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);

    // Call solve_impl() directly (not solve() which goes through base class)
    auto result = solver.solve_impl(query);

    // Verify it compiles and returns expected type
    static_assert(std::is_same_v<decltype(result), std::expected<IVSuccess, IVError>>);

    // Verify placeholder implementation returns success
    EXPECT_TRUE(result.has_value());
    if (result.has_value()) {
        EXPECT_EQ(result->implied_vol, 0.20);  // Placeholder value
        EXPECT_EQ(result->iterations, 0);
    }
}
