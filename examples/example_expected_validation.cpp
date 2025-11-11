/**
 * @file example_expected_validation.cpp
 * @brief Example demonstrating expected-based validation for AmericanOption constructors
 */

#include "src/pricing/american_option.hpp"
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

        AmericanOptionGrid valid_grid{};

        // Using factory method with expected-based validation
        auto result = AmericanOptionSolver::create(valid_params, valid_grid);

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

        AmericanOptionGrid valid_grid{};

        auto result = AmericanOptionSolver::create(invalid_params, valid_grid);

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

        AmericanOptionGrid invalid_grid{};
        invalid_grid.n_space = 5;  // Too small!

        auto result = AmericanOptionSolver::create(valid_params, invalid_grid);

        if (result.has_value()) {
            std::cout << "   ✓ Validation passed!\n";
        } else {
            std::cout << "   ✗ Validation failed: " << result.error() << "\n";
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

        AmericanOptionGrid valid_grid{};

        auto result = AmericanOptionSolver::create(invalid_params, valid_grid);

        if (result.has_value()) {
            std::cout << "   ✓ Validation passed!\n";
        } else {
            std::cout << "   ✗ Validation failed: " << result.error() << "\n";
            std::cout << "   (Note: Only the first validation error is reported)\n";
        }
        std::cout << "\n";
    }

    // Example 5: Using workspace mode with validation
    {
        std::cout << "5. Workspace mode with validation:\n";

        // Create workspace
        auto workspace = std::make_shared<SliceSolverWorkspace>(-3.0, 3.0, 101);

        AmericanOptionParams valid_params{
            .strike = 100.0,
            .spot = 100.0,
            .maturity = 0.5,
            .volatility = 0.25,
            .rate = 0.03,
            .continuous_dividend_yield = 0.01,
            .option_type = OptionType::CALL
        };

        AmericanOptionGrid grid_for_workspace{};  // Must match workspace dimensions
        grid_for_workspace.n_space = 101;  // Must match workspace

        auto result = AmericanOptionSolver::create_with_workspace(valid_params, grid_for_workspace, workspace);

        if (result.has_value()) {
            std::cout << "   ✓ Validation passed! Workspace solver created.\n";
        } else {
            std::cout << "   ✗ Validation failed: " << result.error() << "\n";
        }
        std::cout << "\n";
    }

    // Example 6: Backward compatibility (existing constructor still works)
    {
        std::cout << "6. Backward compatibility (existing constructor):\n";
        AmericanOptionParams valid_params{
            .strike = 100.0,
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.2,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT
        };

        AmericanOptionGrid valid_grid{};

        try {
            // This is the existing constructor - still works as before
            AmericanOptionSolver solver(valid_params, valid_grid);
            std::cout << "   ✓ Existing constructor works with valid parameters.\n";

            auto solution = solver.solve();
            if (solution.has_value()) {
                std::cout << "   ✓ Option solved successfully.\n";
            }
        } catch (const std::exception& e) {
            std::cout << "   ✗ Constructor failed: " << e.what() << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "=== Summary ===\n";
    std::cout << "The new expected-based validation provides:\n";
    std::cout << "• Non-throwing validation with clear error messages\n";
    std::cout << "• Factory methods that return expected<T, E> for better error handling\n";
    std::cout << "• Full backward compatibility with existing constructors\n";
    std::cout << "• Consistent validation logic between exception and expected paths\n";
    std::cout << "• Support for both standard and workspace modes\n\n";

    return 0;
}