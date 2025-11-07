#include "src/price_table_snapshot_collector.hpp"
#include "src/snapshot.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <chrono>

// Test with known analytical solution: European put Black-Scholes
TEST(PriceTableSnapshotCollectorTest, GammaFormulaValidation) {
    // Price table: 3 moneyness points
    std::vector<double> moneyness = {0.8, 1.0, 1.2};
    std::vector<double> tau = {0.5};
    const double K_ref = 100.0;

    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = std::span{moneyness},
        .tau = std::span{tau},
        .K_ref = K_ref,
        .exercise_type = mango::ExerciseType::EUROPEAN
    };

    mango::PriceTableSnapshotCollector collector(config);

    // Mock PDE solution in LOG-MONEYNESS coordinates
    // PDE grid is x = ln(S/K), NOT dollar values S
    std::vector<double> S_values = {60.0, 80.0, 100.0, 120.0, 140.0};
    std::vector<double> x(S_values.size());  // Log-moneyness grid
    std::vector<double> V(x.size());
    std::vector<double> dVdx(x.size());      // ∂V/∂x (NOT ∂V/∂S)
    std::vector<double> d2Vdx2(x.size());    // ∂²V/∂x² (NOT ∂²V/∂S²)

    // Convert S values to log-moneyness and compute derivatives
    for (size_t i = 0; i < x.size(); ++i) {
        double S = S_values[i];
        x[i] = std::log(S / K_ref);  // x = ln(S/K)

        // Function: V = S² (in dollar space)
        // In log-moneyness: V(x) = K²·exp(2x)
        V[i] = S * S;

        // Chain rule: ∂V/∂x = (∂V/∂S) · (∂S/∂x) = (2S) · (S) = 2S²
        dVdx[i] = 2.0 * S * S;

        // Second derivative: ∂²V/∂x² = ∂/∂x(2S²) = 2·2S·S = 4S²
        d2Vdx2[i] = 4.0 * S * S;
    }

    std::vector<double> dx_spacing(x.size() - 1);
    for (size_t i = 0; i < x.size() - 1; ++i) {
        dx_spacing[i] = x[i+1] - x[i];
    }

    std::vector<double> Lu(x.size(), 0.0);

    mango::Snapshot snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx_spacing},
        .solution = std::span{V},
        .spatial_operator = std::span{Lu},
        .first_derivative = std::span{dVdx},
        .second_derivative = std::span{d2Vdx2}
    };

    collector.collect(snapshot);

    // CRITICAL TEST: Verify gamma = ∂²V/∂S² = 2.0 everywhere
    // For V(S) = S², we have ∂²V/∂S² = 2 EVERYWHERE
    //
    // PDE snapshot provides ∂²V/∂S² directly (already in S-space)
    // No transformation needed! Just interpolate and use.

    auto gammas = collector.gammas();

    // Test ALL three moneyness points: ITM (0.8), ATM (1.0), OTM (1.2)
    for (size_t m_idx = 0; m_idx < 3; ++m_idx) {
        size_t idx = m_idx * 1 + 0;  // tau_idx=0
        double m = moneyness[m_idx];
        EXPECT_NEAR(gammas[idx], 2.0, 1e-6)
            << "Gamma must be 2.0 everywhere, failed at m=" << m;
    }
}

TEST(PriceTableSnapshotCollectorTest, ThetaInContinuationRegion) {
    // Test that theta = -L(V) in continuation region
    std::vector<double> moneyness = {1.0};
    std::vector<double> tau = {0.5};
    const double K_ref = 100.0;

    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = std::span{moneyness},
        .tau = std::span{tau},
        .K_ref = K_ref,
        .exercise_type = mango::ExerciseType::EUROPEAN
    };

    mango::PriceTableSnapshotCollector collector(config);

    // Mock data with known Lu
    std::vector<double> x = {80.0, 100.0, 120.0};
    std::vector<double> dx = {20.0, 20.0};
    std::vector<double> V = {20.0, 10.0, 5.0};
    std::vector<double> Lu = {-0.5, -0.3, -0.2};  // Known spatial operator
    std::vector<double> dV = {-0.5, -0.3, -0.2};
    std::vector<double> d2V = {0.01, 0.01, 0.01};

    mango::Snapshot snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx},
        .solution = std::span{V},
        .spatial_operator = std::span{Lu},
        .first_derivative = std::span{dV},
        .second_derivative = std::span{d2V}
    };

    collector.collect(snapshot);

    auto thetas = collector.thetas();

    // Theta should be -Lu = -(-0.3) = 0.3 at S=100 (m=1.0)
    EXPECT_NEAR(thetas[0], 0.3, 0.05);  // Allow interpolation error
}

TEST(PriceTableSnapshotCollectorTest, ThetaAtExerciseBoundary) {
    // Test that theta = NaN at exercise boundary for American options
    std::vector<double> moneyness = {0.5};  // Deep ITM
    std::vector<double> tau = {0.5};
    const double K_ref = 100.0;

    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = std::span{moneyness},
        .tau = std::span{tau},
        .K_ref = K_ref,
        .exercise_type = mango::ExerciseType::AMERICAN
    };

    mango::PriceTableSnapshotCollector collector(config);

    // At m=0.5, S=50, obstacle = max(100-50, 0) = 50
    std::vector<double> x = {30.0, 50.0, 70.0, 90.0, 110.0};
    std::vector<double> dx = {20.0, 20.0, 20.0, 20.0};
    std::vector<double> V(x.size());
    std::vector<double> Lu(x.size(), -0.1);
    std::vector<double> dV(x.size(), -1.0);
    std::vector<double> d2V(x.size(), 0.01);

    // Set V = obstacle at exercise boundary
    for (size_t i = 0; i < x.size(); ++i) {
        V[i] = std::max(K_ref - x[i], 0.0);  // At exercise boundary
    }

    mango::Snapshot snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx},
        .solution = std::span{V},
        .spatial_operator = std::span{Lu},
        .first_derivative = std::span{dV},
        .second_derivative = std::span{d2V}
    };

    collector.collect(snapshot);

    auto thetas = collector.thetas();

    // At exercise boundary, theta should be NaN
    EXPECT_TRUE(std::isnan(thetas[0]));
}

TEST(PriceTableSnapshotCollectorTest, VegaInterpolation) {
    // Test vega computation (placeholder - will be implemented later)
    std::vector<double> moneyness = {1.0};
    std::vector<double> tau = {0.5};
    const double K_ref = 100.0;

    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = std::span{moneyness},
        .tau = std::span{tau},
        .K_ref = K_ref,
        .exercise_type = mango::ExerciseType::EUROPEAN
    };

    mango::PriceTableSnapshotCollector collector(config);

    std::vector<double> x = {80.0, 100.0, 120.0};
    std::vector<double> dx = {20.0, 20.0};
    std::vector<double> V = {20.0, 10.0, 5.0};
    std::vector<double> Lu = {-0.1, -0.1, -0.1};
    std::vector<double> dV = {-0.5, -0.3, -0.2};
    std::vector<double> d2V = {0.01, 0.01, 0.01};

    mango::Snapshot snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx},
        .solution = std::span{V},
        .spatial_operator = std::span{Lu},
        .first_derivative = std::span{dV},
        .second_derivative = std::span{d2V}
    };

    collector.collect(snapshot);

    // Just verify no crash - vega not implemented yet
    SUCCEED();
}

TEST(PriceTableSnapshotCollectorTest, SnapshotOrdering) {
    // Test that user_index = tau_idx works correctly
    std::vector<double> moneyness = {1.0};
    std::vector<double> tau = {0.25, 0.5, 0.75};
    const double K_ref = 100.0;

    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = std::span{moneyness},
        .tau = std::span{tau},
        .K_ref = K_ref,
        .exercise_type = mango::ExerciseType::EUROPEAN
    };

    mango::PriceTableSnapshotCollector collector(config);

    std::vector<double> x = {80.0, 100.0, 120.0};
    std::vector<double> dx = {20.0, 20.0};
    std::vector<double> V = {20.0, 10.0, 5.0};
    std::vector<double> Lu = {-0.1, -0.1, -0.1};
    std::vector<double> dV = {-0.5, -0.3, -0.2};
    std::vector<double> d2V = {0.01, 0.01, 0.01};

    // Collect snapshots in order: tau_idx=0, 1, 2
    for (size_t tau_idx = 0; tau_idx < 3; ++tau_idx) {
        mango::Snapshot snapshot{
            .time = tau[tau_idx],
            .user_index = tau_idx,  // user_index IS tau_idx
            .spatial_grid = std::span{x},
            .dx = std::span{dx},
            .solution = std::span{V},
            .spatial_operator = std::span{Lu},
            .first_derivative = std::span{dV},
            .second_derivative = std::span{d2V}
        };

        collector.collect(snapshot);
    }

    auto prices = collector.prices();

    // Verify all prices collected (3 tau points × 1 moneyness = 3 entries)
    EXPECT_EQ(prices.size(), 3u);

    // All should have values (not zero)
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_GT(prices[i], 0.0);
    }
}

TEST(PriceTableSnapshotCollectorTest, InterpolatorsBuiltOnce) {
    // Verify performance optimization: interpolators built once per snapshot
    std::vector<double> moneyness(50);  // Many moneyness points
    for (size_t i = 0; i < 50; ++i) {
        moneyness[i] = 0.5 + i * 0.02;
    }
    std::vector<double> tau = {0.5};

    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = std::span{moneyness},
        .tau = std::span{tau},
        .K_ref = 100.0,
        .exercise_type = mango::ExerciseType::EUROPEAN
    };

    mango::PriceTableSnapshotCollector collector(config);

    // Simple mock data
    std::vector<double> x = {50.0, 100.0, 150.0};
    std::vector<double> dx = {50.0, 50.0};
    std::vector<double> V = {50.0, 10.0, 2.0};
    std::vector<double> Lu = {0.1, 0.2, 0.1};
    std::vector<double> dV = {-1.0, -0.5, -0.2};
    std::vector<double> d2V = {0.05, 0.03, 0.01};

    mango::Snapshot snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx},
        .solution = std::span{V},
        .spatial_operator = std::span{Lu},
        .first_derivative = std::span{dV},
        .second_derivative = std::span{d2V}
    };

    // This should complete quickly (not O(n²))
    auto start = std::chrono::high_resolution_clock::now();
    collector.collect(snapshot);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // With 50 moneyness points, should be <1ms if interpolators built once
    // Would be >>10ms if rebuilt in loop
    EXPECT_LT(duration_us, 10000) << "Interpolators likely rebuilt in loop (O(n²))";
}
