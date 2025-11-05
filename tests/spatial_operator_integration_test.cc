#include "src/cpp/operators/operator_factory.hpp"
#include "src/cpp/operators/black_scholes_pde.hpp"
#include "src/cpp/spatial_operators.hpp"  // Old operators
#include "src/cpp/grid.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

TEST(SpatialOperatorIntegrationTest, UniformGridBlackScholes) {
    // Test composed operator on uniform grid in log-moneyness coordinates
    const size_t n = 101;
    const double x_min = -0.5;  // ln(S/K) range
    const double x_max = 0.5;
    const double dx = (x_max - x_min) / (n - 1);

    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = x_min + i * dx;
    }

    // PDE parameters
    const double sigma = 0.20;
    const double r = 0.05;
    const double d = 0.01;

    // Create composed operator
    auto grid = mango::GridView<double>(x);
    auto spatial_op = mango::operators::create_spatial_operator(
        mango::operators::BlackScholesPDE<double>(sigma, r, d),
        grid
    );

    // Test function: smooth option payoff-like function
    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        double S_over_K = std::exp(x[i]);
        u[i] = std::max(S_over_K - 1.0, 0.0);  // Call payoff
    }

    // Apply operator
    std::vector<double> Lu(n, 0.0);
    spatial_op.apply(0.0, u, Lu);

    // Verify operator produces reasonable values for interior points
    // For a call payoff, the operator should produce non-zero values in the ITM region
    bool found_nonzero = false;
    for (size_t i = 1; i < n - 1; ++i) {
        if (std::abs(Lu[i]) > 1e-10) {
            found_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(found_nonzero) << "Operator should produce non-zero values";
}

TEST(SpatialOperatorIntegrationTest, SecondOrderAccuracy) {
    // Verify second-order convergence for smooth function
    const double sigma = 0.20;
    const double r = 0.05;
    const double d = 0.01;

    // Test function: f(x) = sin(x), known exact derivatives
    auto test_func = [](double x) { return std::sin(x); };
    auto exact_first = [](double x) { return std::cos(x); };   // f'(x)
    auto exact_second = [](double x) { return -std::sin(x); };  // f''(x)

    std::vector<double> errors;
    std::vector<double> dxs = {0.1, 0.05, 0.025};

    for (double dx : dxs) {
        const size_t n = static_cast<size_t>(2.0 / dx) + 1;
        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = -1.0 + i * dx;
        }

        std::vector<double> u(n);
        for (size_t i = 0; i < n; ++i) {
            u[i] = test_func(x[i]);
        }

        auto grid = mango::GridView<double>(x);
        auto spatial_op = mango::operators::create_spatial_operator(
            mango::operators::BlackScholesPDE<double>(sigma, r, d),
            grid
        );

        std::vector<double> Lu(n, 0.0);
        spatial_op.apply(0.0, u, Lu);

        // Compute max error in interior
        double max_error = 0.0;
        const double half_sigma_sq = 0.5 * sigma * sigma;
        const double drift = r - d - half_sigma_sq;

        for (size_t i = 10; i < n - 10; ++i) {  // Avoid boundaries
            // Full Black-Scholes operator: Lu = (σ²/2)*f'' + (r-d-σ²/2)*f' - r*f
            // For sin(x): f'' = -sin(x), f' = cos(x), f = sin(x)
            double expected = half_sigma_sq * exact_second(x[i])
                            + drift * exact_first(x[i])
                            - r * test_func(x[i]);

            double error = std::abs(Lu[i] - expected);
            max_error = std::max(max_error, error);
        }
        errors.push_back(max_error);
    }

    // Verify second-order convergence: error ~ dx^2
    // error[1] / error[0] should be ~ (dx[1]/dx[0])^2 = 0.25
    double ratio = errors[1] / errors[0];
    EXPECT_LT(ratio, 0.3);   // Should be close to 0.25
    EXPECT_GT(ratio, 0.2);
}
