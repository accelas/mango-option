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
        AmericanOptionParams valid_params(
            100.0,  // spot
            100.0,  // strike
            1.0,    // maturity
            0.05,   // rate
            0.02,   // dividend_yield
            OptionType::PUT,
            0.2     // volatility
        );

        // Create workspace
        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 101;
        constexpr size_t n_time = 1000;

        auto grid_spec = GridSpec<double>::uniform(x_min, x_max, n_space);
        if (!grid_spec.has_value()) {
            std::cout << "   ✗ Failed to create grid: " << grid_spec.error() << "\n\n";
            return 1;
        }
        auto workspace = AmericanSolverWorkspace::create(grid_spec.value().x_min(), grid_spec.value().x_max(), grid_spec.value().n_points(), n_time);
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
                         << solution.value().value_at(valid_params.spot) << "\n";
            }
        } else {
            std::cout << "   ✗ Validation failed: " << result.error() << "\n";
        }
        std::cout << "\n";
    }

    // Example 2: Invalid parameters (strike = 0)
    {
        std::cout << "2. Invalid parameters (strike = 0):\n";
        AmericanOptionParams invalid_params(
            100.0,  // spot
            0.0,    // strike (Invalid)
            1.0,    // maturity
            0.05,   // rate
            0.02,   // dividend_yield
            OptionType::PUT,
            0.2     // volatility
        );

        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 101;
        constexpr size_t n_time = 1000;

        auto grid_spec = GridSpec<double>::uniform(x_min, x_max, n_space);
        if (!grid_spec.has_value()) {
            std::cout << "   ✗ Failed to create grid: " << grid_spec.error() << "\n\n";
            return 1;
        }
        auto workspace = AmericanSolverWorkspace::create(grid_spec.value().x_min(), grid_spec.value().x_max(), grid_spec.value().n_points(), n_time);
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
        AmericanOptionParams valid_params(
            100.0,  // spot
            100.0,  // strike
            1.0,    // maturity
            0.05,   // rate
            0.02,   // dividend_yield
            OptionType::CALL,
            0.2     // volatility
        );

        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 5;  // Too small!
        constexpr size_t n_time = 1000;

        auto grid_spec = GridSpec<double>::uniform(x_min, x_max, n_space);
        std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string> workspace;
        if (grid_spec.has_value()) {
            workspace = AmericanSolverWorkspace::create(grid_spec.value().x_min(), grid_spec.value().x_max(), grid_spec.value().n_points(), n_time);
        } else {
            workspace = std::unexpected(grid_spec.error());
        }

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
        AmericanOptionParams invalid_params(
            0.0,          // spot (invalid)
            -50.0,        // strike (invalid)
            -1.0,         // maturity (invalid)
            0.05,         // rate
            -0.02,        // dividend_yield (invalid)
            OptionType::PUT,
            -0.2          // volatility (invalid)
        );

        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 101;
        constexpr size_t n_time = 1000;

        auto grid_spec = GridSpec<double>::uniform(x_min, x_max, n_space);
        if (!grid_spec.has_value()) {
            std::cout << "   ✗ Failed to create grid: " << grid_spec.error() << "\n\n";
            return 1;
        }
        auto workspace = AmericanSolverWorkspace::create(grid_spec.value().x_min(), grid_spec.value().x_max(), grid_spec.value().n_points(), n_time);
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

        auto grid_spec = GridSpec<double>::uniform(x_min, x_max, n_space);
        if (!grid_spec.has_value()) {
            std::cout << "   ✗ Failed to create grid: " << grid_spec.error() << "\n\n";
            return 1;
        }
        auto workspace = AmericanSolverWorkspace::create(grid_spec.value().x_min(), grid_spec.value().x_max(), grid_spec.value().n_points(), n_time);
        if (!workspace) {
            std::cout << "   ✗ Failed to create workspace: " << workspace.error() << "\n\n";
            return 1;
        }

        AmericanOptionParams params1(
            100.0,  // spot
            100.0,  // strike
            0.5,    // maturity
            0.03,   // rate
            0.01,   // dividend_yield
            OptionType::CALL,
            0.25    // volatility
        );

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
        AmericanOptionParams valid_params(
            100.0,  // spot
            100.0,  // strike
            1.0,    // maturity
            0.05,   // rate
            0.02,   // dividend_yield
            OptionType::PUT,
            0.2     // volatility
        );

        constexpr double x_min = -3.0;
        constexpr double x_max = 3.0;
        constexpr size_t n_space = 101;
        constexpr size_t n_time = 1000;

        auto grid_spec = GridSpec<double>::uniform(x_min, x_max, n_space);
        if (!grid_spec.has_value()) {
            std::cout << "   ✗ Failed to create grid: " << grid_spec.error() << "\n\n";
            return 1;
        }
        auto workspace = AmericanSolverWorkspace::create(grid_spec.value().x_min(), grid_spec.value().x_max(), grid_spec.value().n_points(), n_time);
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
                     << solution.value().value_at(valid_params.spot) << "\n";
        } else {
            std::cout << "   ✗ Solve failed: " << solution.error().message << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "=== Summary ===\n";
    std::cout << "The expected-based validation provides:\n";
    std::cout << "• Non-throwing validation with clear error messages\n";
    std::cout << "• Factory methods that return std::expected<T, E> for better error handling\n";
    std::cout << "• Workspace creation validates grid parameters (n_space, n_time)\n";
    std::cout << "• Solver creation validates option parameters (strike, spot, etc.)\n";
    std::cout << "• Shared workspace support for efficient batch solving\n\n";

    return 0;
}
