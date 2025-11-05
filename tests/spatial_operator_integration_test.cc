#include "src/cpp/operators/operator_factory.hpp"
#include "src/cpp/operators/black_scholes_pde.hpp"
#include "src/cpp/spatial_operators.hpp"  // Old operators
#include "src/cpp/grid.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

TEST(SpatialOperatorIntegrationTest, MatchesUniformGridOperator) {
    // Create uniform grid in log-moneyness coordinates
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

    // Create OLD operator
    auto old_op = mango::UniformGridBlackScholesOperator(sigma, r, d, dx);

    // Create NEW operator
    auto grid = mango::GridView<double>(x);
    auto new_op = mango::operators::create_spatial_operator(
        mango::operators::BlackScholesPDE<double>(sigma, r, d),
        grid
    );

    // Test function: smooth option payoff-like function
    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        double S_over_K = std::exp(x[i]);
        u[i] = std::max(S_over_K - 1.0, 0.0);  // Call payoff
    }

    // Apply OLD operator
    std::vector<double> Lu_old(n, 0.0);
    std::vector<double> dx_array(n-1, dx);  // Uniform spacing array
    old_op(0.0, x, u, Lu_old, dx_array);

    // Apply NEW operator
    std::vector<double> Lu_new(n, 0.0);
    new_op.apply(0.0, u, Lu_new);

    // Compare interior points (boundaries may differ in implementation)
    for (size_t i = 1; i < n - 1; ++i) {
        EXPECT_NEAR(Lu_new[i], Lu_old[i], 1e-12)
            << "Mismatch at point " << i << " (x=" << x[i] << ")";
    }
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
