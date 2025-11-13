/**
 * @file example_expected_validation.cpp
 * @brief Example demonstrating expected-based validation for AmericanOption constructors
 */

#include "src/option/american_option.hpp"
#include <iostream>
#include <iomanip>

using namespace mango;

int main() {
    std::cout << "=== AmericanOption Expected-Based Validation Example ===\n\n";

    // Example 1: Valid parameters
    {
        std::cout << "1. Valid parameters:\n";
        AmericanOptionParams valid_params{
            .strike = 100.0,
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.2,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT
        };

        // Create workspace
        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 101;
        constexpr size_t n_time = 1000;

        auto workspace = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
        if (!workspace) {
            std::cout << "   ✗ Failed to create workspace: " << workspace.error() << "\n\n";
            return 1;
        }

        // Using factory method with expected-based validation
        auto result = AmericanOptionSolver::create(valid_params, workspace.value());

        if (result.has_value()) {
            std::cout << "   ✓ Validation passed! Solver created successfully.\n";

            // Solve the option
            auto solution = result.value().solve();
            if (solution.has_value()) {
                std::cout << "   ✓ Option solved successfully.\n";
                std::cout << "   ✓ Option value: $" << std::fixed << std::setprecision(4)
                         << solution.value().value << "\n";
            }
        } else {
            std::cout << "   ✗ Validation failed: " << result.error() << "\n";
        }
        std::cout << "\n";
    }

    // Example 2: Invalid parameters (strike = 0)
    {
        std::cout << "2. Invalid parameters (strike = 0):\n";
        AmericanOptionParams invalid_params{
            .strike = 0.0,  // Invalid!
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.2,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT
        };

        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 101;
        constexpr size_t n_time = 1000;

        auto workspace = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
        if (!workspace) {
            std::cout << "   ✗ Failed to create workspace: " << workspace.error() << "\n\n";
            return 1;
        }

        auto result = AmericanOptionSolver::create(invalid_params, workspace.value());

        if (result.has_value()) {
            std::cout << "   ✓ Validation passed!\n";
        } else {
            std::cout << "   ✗ Validation failed: " << result.error() << "\n";
        }
        std::cout << "\n";
    }

    // Example 3: Invalid grid parameters
    {
        std::cout << "3. Invalid grid parameters (n_space too small):\n";
        AmericanOptionParams valid_params{
            .strike = 100.0,
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.2,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::CALL
        };

        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 5;  // Too small!
        constexpr size_t n_time = 1000;

        auto workspace = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);

        if (workspace.has_value()) {
            std::cout << "   ✓ Workspace created (unexpected)!\n";
        } else {
            std::cout << "   ✗ Workspace creation failed: " << workspace.error() << "\n";
        }
        std::cout << "\n";
    }

    // Example 4: Multiple validation errors (reports first one)
    {
        std::cout << "4. Multiple validation errors (reports first error):\n";
        AmericanOptionParams invalid_params{
            .strike = -50.0,  // Invalid!
            .spot = 0.0,      // Also invalid!
            .maturity = -1.0, // Also invalid!
            .volatility = -0.2,
            .rate = 0.05,
            .continuous_dividend_yield = -0.02,
            .option_type = OptionType::PUT
        };

        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 101;
        constexpr size_t n_time = 1000;

        auto workspace = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
        if (!workspace) {
            std::cout << "   ✗ Failed to create workspace: " << workspace.error() << "\n\n";
            return 1;
        }

        auto result = AmericanOptionSolver::create(invalid_params, workspace.value());

        if (result.has_value()) {
            std::cout << "   ✓ Validation passed!\n";
        } else {
            std::cout << "   ✗ Validation failed: " << result.error() << "\n";
            std::cout << "   (Note: Only the first validation error is reported)\n";
        }
        std::cout << "\n";
    }

    // Example 5: Using shared workspace for multiple solves
    {
        std::cout << "5. Using shared workspace for multiple solves:\n";

        // Create workspace once
        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 101;
        constexpr size_t n_time = 1000;

        auto workspace = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
        if (!workspace) {
            std::cout << "   ✗ Failed to create workspace: " << workspace.error() << "\n\n";
            return 1;
        }

        AmericanOptionParams params1{
            .strike = 100.0,
            .spot = 100.0,
            .maturity = 0.5,
            .volatility = 0.25,
            .rate = 0.03,
            .continuous_dividend_yield = 0.01,
            .option_type = OptionType::CALL
        };

        auto result1 = AmericanOptionSolver::create(params1, workspace.value());

        if (result1.has_value()) {
            std::cout << "   ✓ First solver created with shared workspace.\n";
        } else {
            std::cout << "   ✗ Validation failed: " << result1.error() << "\n";
        }
        std::cout << "\n";
    }

    // Example 6: Error propagation in solve()
    {
        std::cout << "6. Error propagation in solve():\n";
        AmericanOptionParams valid_params{
            .strike = 100.0,
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.2,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT
        };

        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 101;
        constexpr size_t n_time = 1000;

        auto workspace = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
        if (!workspace) {
            std::cout << "   ✗ Failed to create workspace: " << workspace.error() << "\n\n";
            return 1;
        }

        auto solver_result = AmericanOptionSolver::create(valid_params, workspace.value());
        if (!solver_result) {
            std::cout << "   ✗ Solver creation failed: " << solver_result.error() << "\n\n";
            return 1;
        }

        auto solution = solver_result.value().solve();
        if (solution.has_value()) {
            std::cout << "   ✓ Option solved successfully.\n";
            std::cout << "   ✓ Option value: $" << std::fixed << std::setprecision(4)
                     << solution.value().value << "\n";
        } else {
            std::cout << "   ✗ Solve failed: " << solution.error().message << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "=== Summary ===\n";
    std::cout << "The expected-based validation provides:\n";
    std::cout << "• Non-throwing validation with clear error messages\n";
    std::cout << "• Factory methods that return expected<T, E> for better error handling\n";
    std::cout << "• Workspace creation validates grid parameters (n_space, n_time)\n";
    std::cout << "• Solver creation validates option parameters (strike, spot, etc.)\n";
    std::cout << "• Shared workspace support for efficient batch solving\n\n";

    return 0;
}