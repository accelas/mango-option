#include "src/cpp/american_obstacle.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

TEST(AmericanObstacleTest, PutObstacleValues) {
    mango::AmericanPutObstacle put_obstacle;

    // Test grid: x = ln(S/K) from -0.5 to 0.5 (S/K from ~0.6 to ~1.65)
    std::vector<double> x = {-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5};
    std::vector<double> psi(x.size());

    put_obstacle(0.0, x, psi);

    // Verify: ψ(x) = max(1 - exp(x), 0)
    for (size_t i = 0; i < x.size(); ++i) {
        double expected = std::max(1.0 - std::exp(x[i]), 0.0);
        EXPECT_NEAR(psi[i], expected, 1e-12)
            << "Put obstacle mismatch at x = " << x[i];
    }

    // Deep ITM put (x = -1.0, S/K ~ 0.37): ψ ~ 0.63
    x = {-1.0};
    psi.resize(1);
    put_obstacle(0.0, x, psi);
    EXPECT_NEAR(psi[0], 1.0 - std::exp(-1.0), 1e-12);

    // Deep OTM put (x = 1.0, S/K ~ 2.72): ψ = 0
    x = {1.0};
    put_obstacle(0.0, x, psi);
    EXPECT_NEAR(psi[0], 0.0, 1e-12);
}

TEST(AmericanObstacleTest, CallObstacleValues) {
    mango::AmericanCallObstacle call_obstacle;

    // Test grid: x = ln(S/K) from -0.5 to 0.5
    std::vector<double> x = {-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5};
    std::vector<double> psi(x.size());

    call_obstacle(0.0, x, psi);

    // Verify: ψ(x) = max(exp(x) - 1, 0)
    for (size_t i = 0; i < x.size(); ++i) {
        double expected = std::max(std::exp(x[i]) - 1.0, 0.0);
        EXPECT_NEAR(psi[i], expected, 1e-12)
            << "Call obstacle mismatch at x = " << x[i];
    }

    // Deep OTM call (x = -1.0, S/K ~ 0.37): ψ = 0
    x = {-1.0};
    psi.resize(1);
    call_obstacle(0.0, x, psi);
    EXPECT_NEAR(psi[0], 0.0, 1e-12);

    // Deep ITM call (x = 1.0, S/K ~ 2.72): ψ ~ 1.72
    x = {1.0};
    call_obstacle(0.0, x, psi);
    EXPECT_NEAR(psi[0], std::exp(1.0) - 1.0, 1e-12);
}

TEST(AmericanObstacleTest, PutCallSymmetry) {
    mango::AmericanPutObstacle put_obstacle;
    mango::AmericanCallObstacle call_obstacle;

    // At-the-money (x = 0): both obstacles should equal 0
    std::vector<double> x = {0.0};
    std::vector<double> psi_put(1), psi_call(1);

    put_obstacle(0.0, x, psi_put);
    call_obstacle(0.0, x, psi_call);

    EXPECT_NEAR(psi_put[0], 0.0, 1e-12);
    EXPECT_NEAR(psi_call[0], 0.0, 1e-12);

    // Mirror points: put(x) at ITM should equal call(-x) at ITM
    x = {-0.5, 0.5};
    psi_put.resize(2);
    psi_call.resize(2);

    put_obstacle(0.0, x, psi_put);
    call_obstacle(0.0, x, psi_call);

    // Put at x=-0.5 is ITM, call at x=0.5 is ITM
    EXPECT_GT(psi_put[0], 0.0);
    EXPECT_GT(psi_call[1], 0.0);
    EXPECT_NEAR(psi_put[1], 0.0, 1e-12);  // Put at x=0.5 is OTM
    EXPECT_NEAR(psi_call[0], 0.0, 1e-12);  // Call at x=-0.5 is OTM
}
