#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace mango {

TEST(BoundaryConditionDebug, PutLeftBoundary) {
    // Test parameters
    double K = 100.0;
    double r = 0.05;
    double T = 0.75;

    std::cout << "\n=== American Put Left Boundary Analysis ===" << std::endl;
    std::cout << "Strike K = " << K << std::endl;
    std::cout << "Rate r = " << r << std::endl;
    std::cout << "Maturity T = " << T << " years" << std::endl;

    // The left boundary in log-moneyness is when x → -∞, i.e., S → 0
    // The put value as S → 0 is:
    // - At maturity (t=T): V = K (intrinsic value, exercise immediately)
    // - Before maturity (t<T): V = ? (this is what we need to figure out)

    // Current implementation uses: V/K = max(1 - exp(x), 0)
    // which is the intrinsic value normalized by K

    double x_left = -7.0;  // Grid left boundary
    double S_left = K * std::exp(x_left);  // Spot at left boundary

    std::cout << "\n--- At left boundary x = " << x_left << " ---" << std::endl;
    std::cout << "Spot S = " << S_left << std::endl;
    std::cout << "Intrinsic value = K - S = " << (K - S_left) << std::endl;
    std::cout << "Intrinsic normalized = (K - S)/K = " << ((K - S_left) / K) << std::endl;
    std::cout << "Current BC formula = max(1 - exp(x), 0) = " << std::max(1.0 - std::exp(x_left), 0.0) << std::endl;

    std::cout << "\n--- Correct boundary value at different times ---" << std::endl;
    std::cout << std::setw(10) << "Time τ"
              << std::setw(20) << "Discounted K"
              << std::setw(20) << "Intrinsic K"
              << std::setw(20) << "Difference"
              << std::endl;

    for (double tau = 0.0; tau <= T; tau += 0.15) {
        double discounted_K = K * std::exp(-r * tau);
        double intrinsic_K = K;  // Since S ≈ 0
        double difference = intrinsic_K - discounted_K;

        std::cout << std::setw(10) << tau
                  << std::setw(20) << discounted_K
                  << std::setw(20) << intrinsic_K
                  << std::setw(20) << difference
                  << std::endl;
    }

    std::cout << "\n=== Analysis ===" << std::endl;
    std::cout << "For an American put with S → 0:" << std::endl;
    std::cout << "- At maturity (τ=0): Boundary value = K (intrinsic)" << std::endl;
    std::cout << "- Before maturity (τ>0 with r>0): Should we use K or K*exp(-r*τ)?" << std::endl;
    std::cout << "\nIf the boundary uses intrinsic K instead of discounted K*exp(-r*τ)," << std::endl;
    std::cout << "the PDE will produce values ABOVE the economically correct value!" << std::endl;
    std::cout << "\nFor τ = 0.75, the difference is: " << (K - K * std::exp(-r * T)) << std::endl;
}

}  // namespace mango
