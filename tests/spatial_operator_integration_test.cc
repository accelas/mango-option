#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
// Legacy spatial_operators.hpp removed
#include "src/pde/core/grid.hpp"
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

TEST(SpatialOperatorIntegrationTest, NonUniformGridAmericanOption) {
    // Test Black-Scholes operator with non-uniform grid concentrated near strike
    // This simulates pricing an American option with refined grid near the payoff kink

    // Create non-uniform grid using tanh transformation
    // Concentrates points near x=0 (at-the-money in log-moneyness)
    const size_t n = 101;
    const double x_min = -1.0;
    const double x_max = 1.0;

    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        // Map uniform ξ ∈ [0,1] to non-uniform x using tanh
        double xi = static_cast<double>(i) / (n - 1);  // Uniform in [0,1]
        double eta = 2.0 * xi - 1.0;  // Map to [-1, 1]
        double concentration = 2.0;   // Controls concentration (higher = more concentrated)
        x[i] = 0.5 * (x_max - x_min) * std::tanh(concentration * eta) / std::tanh(concentration)
             + 0.5 * (x_max + x_min);
    }

    // Verify grid is non-uniform
    bool is_uniform = true;
    double dx_first = x[1] - x[0];
    for (size_t i = 1; i < n - 1; ++i) {
        double dx = x[i+1] - x[i];
        if (std::abs(dx - dx_first) > 1e-10) {
            is_uniform = false;
            break;
        }
    }
    EXPECT_FALSE(is_uniform) << "Grid should be non-uniform";

    // Create Black-Scholes operator with typical parameters
    const double sigma = 0.25;
    const double r = 0.05;
    const double d = 0.0;  // No dividend for simplicity

    auto grid = mango::GridView<double>(x);
    auto spatial_op = mango::operators::create_spatial_operator(
        mango::operators::BlackScholesPDE<double>(sigma, r, d),
        grid
    );

    // American put payoff: max(K - S, 0) = max(1 - exp(x), 0) in log-moneyness
    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        double S_over_K = std::exp(x[i]);
        u[i] = std::max(1.0 - S_over_K, 0.0);  // Put payoff
    }

    // Apply Black-Scholes operator
    std::vector<double> Lu(n, 0.0);
    spatial_op.apply(0.0, u, Lu);

    // Verify operator produces reasonable values
    // 1. Boundaries should be zero (set by apply())
    EXPECT_NEAR(Lu[0], 0.0, 1e-12);
    EXPECT_NEAR(Lu[n-1], 0.0, 1e-12);

    // 2. Interior should have non-zero values near strike (x=0)
    // Find point closest to x=0
    size_t mid_idx = 0;
    double min_dist = std::abs(x[0]);
    for (size_t i = 1; i < n; ++i) {
        double dist = std::abs(x[i]);
        if (dist < min_dist) {
            min_dist = dist;
            mid_idx = i;
        }
    }

    // Near strike, operator should be active
    bool found_significant_value = false;
    for (size_t i = mid_idx - 5; i <= mid_idx + 5 && i < n - 1; ++i) {
        if (i >= 1 && std::abs(Lu[i]) > 1e-6) {
            found_significant_value = true;
            break;
        }
    }
    EXPECT_TRUE(found_significant_value) << "Operator should be active near strike";

    // 3. For ITM region (x < 0, S < K), put value should decay with time
    // This means Lu should have appropriate sign
    // For a put: V_t = -Lu (PDE: V_t = Lu), so Lu < 0 means value decays
    // Deep ITM puts lose time value, so we expect negative Lu
    size_t deep_itm_idx = n / 4;  // x < 0 region
    if (x[deep_itm_idx] < -0.3) {  // Reasonably deep ITM
        // Time decay should be negative (losing time value)
        // Lu includes -r*V term which should dominate for deep ITM
        EXPECT_LT(Lu[deep_itm_idx], 0.0) << "Deep ITM put should have negative Lu";
    }

    // 4. Verify non-uniform grid produces finite, reasonable values
    for (size_t i = 1; i < n - 1; ++i) {
        EXPECT_TRUE(std::isfinite(Lu[i])) << "Lu[" << i << "] should be finite";
        EXPECT_LT(std::abs(Lu[i]), 100.0) << "Lu[" << i << "] should be bounded";
    }
}

TEST(SpatialOperatorIntegrationTest, NonUniformGridSecondOrderAccuracy) {
    // Verify non-uniform grid maintains second-order accuracy
    // Uses a smooth test function (not a payoff with kink)

    const double sigma = 0.20;
    const double r = 0.05;
    const double d = 0.01;

    // Test function: f(x) = exp(x), smooth with known derivatives
    // f'(x) = exp(x), f''(x) = exp(x)
    auto test_func = [](double x) { return std::exp(x); };

    std::vector<double> errors;
    std::vector<size_t> grid_sizes = {51, 101, 201};

    for (size_t n : grid_sizes) {
        // Create non-uniform grid with tanh transformation
        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            double xi = static_cast<double>(i) / (n - 1);
            double eta = 2.0 * xi - 1.0;
            double concentration = 1.5;
            x[i] = std::tanh(concentration * eta) / std::tanh(concentration);
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

        for (size_t i = 5; i < n - 5; ++i) {  // Avoid boundaries
            // For f(x) = exp(x): f''(x) = exp(x), f'(x) = exp(x), f(x) = exp(x)
            double exact_value = std::exp(x[i]);
            double exact_first = exact_value;
            double exact_second = exact_value;

            double expected = half_sigma_sq * exact_second
                            + drift * exact_first
                            - r * exact_value;

            double error = std::abs(Lu[i] - expected);
            max_error = std::max(max_error, error);
        }
        errors.push_back(max_error);
    }

    // Verify second-order convergence
    // With grid refinement factor ~2x, error should reduce by ~4x
    double ratio_1 = errors[1] / errors[0];
    double ratio_2 = errors[2] / errors[1];

    // Should be approaching second-order (ratio ~ 0.25)
    EXPECT_LT(ratio_1, 0.35) << "First refinement should show second-order convergence";
    EXPECT_LT(ratio_2, 0.35) << "Second refinement should show second-order convergence";

    // Errors should be decreasing
    EXPECT_LT(errors[1], errors[0]);
    EXPECT_LT(errors[2], errors[1]);
}
