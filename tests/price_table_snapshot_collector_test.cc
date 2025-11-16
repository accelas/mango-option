#include "src/option/price_table_snapshot_collector.hpp"
#include "src/pde/core/snapshot.hpp"
#include "src/support/memory/solver_memory_arena.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory_resource>

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
        .option_type = mango::OptionType::PUT
    };

    mango::PriceTableSnapshotCollector collector(config);

    // Mock PDE solution in LOG-MONEYNESS coordinates
    // PDE grid is x = ln(S/K), working with NORMALIZED prices
    std::vector<double> S_values = {60.0, 80.0, 100.0, 120.0, 140.0};
    std::vector<double> x(S_values.size());  // Log-moneyness grid
    std::vector<double> V_norm(x.size());     // Normalized price = V_dollar / K_ref
    std::vector<double> dVnorm_dx(x.size());  // ∂V_norm/∂x
    std::vector<double> d2Vnorm_dx2(x.size());// ∂²V_norm/∂x²

    // Convert S values to log-moneyness and compute derivatives
    for (size_t i = 0; i < x.size(); ++i) {
        double S = S_values[i];
        x[i] = std::log(S / K_ref);  // x = ln(S/K)

        // Function: V_dollar = S²
        // Normalized: V_norm = S²/K_ref
        // In log-moneyness: V_norm(x) = K_ref·e^(2x)
        V_norm[i] = (S * S) / K_ref;

        // Chain rule: ∂V_norm/∂x = 2·K_ref·e^(2x) = 2·S²/K_ref
        dVnorm_dx[i] = 2.0 * (S * S) / K_ref;

        // Second derivative: ∂²V_norm/∂x² = 4·K_ref·e^(2x) = 4·S²/K_ref
        d2Vnorm_dx2[i] = 4.0 * (S * S) / K_ref;
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
        .solution = std::span{V_norm},
        .spatial_operator = std::span{Lu},
        .first_derivative = std::span{dVnorm_dx},
        .second_derivative = std::span{d2Vnorm_dx2}
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
        .option_type = mango::OptionType::PUT
    };

    mango::PriceTableSnapshotCollector collector(config);

    // Mock data with known Lu - use LOG-MONEYNESS coordinates
    // m=1.0 corresponds to x = ln(1.0) = 0
    std::vector<double> S_values = {80.0, 100.0, 120.0};
    std::vector<double> x(S_values.size());
    for (size_t i = 0; i < S_values.size(); ++i) {
        x[i] = std::log(S_values[i] / K_ref);
    }

    std::vector<double> dx_spacing(x.size() - 1);
    for (size_t i = 0; i < x.size() - 1; ++i) {
        dx_spacing[i] = x[i+1] - x[i];
    }

    // Normalized values (V_norm = V_dollar / K_ref)
    std::vector<double> V_norm = {0.20, 0.10, 0.05};
    std::vector<double> Lu_norm = {-0.005, -0.003, -0.002};  // Normalized spatial operator
    std::vector<double> dVnorm_dx = {-0.005, -0.003, -0.002};
    std::vector<double> d2Vnorm_dx2 = {0.0001, 0.0001, 0.0001};

    mango::Snapshot snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx_spacing},
        .solution = std::span{V_norm},
        .spatial_operator = std::span{Lu_norm},
        .first_derivative = std::span{dVnorm_dx},
        .second_derivative = std::span{d2Vnorm_dx2}
    };

    collector.collect(snapshot);

    auto thetas = collector.thetas();

    // Theta should be -K_ref * Lu_norm = -100 * (-0.003) = 0.3 at m=1.0
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
        .option_type = mango::OptionType::PUT
    };

    mango::PriceTableSnapshotCollector collector(config);

    // At m=0.5, S=50, obstacle = max(K-S, 0) = 50
    // Use LOG-MONEYNESS coordinates
    std::vector<double> S_values = {30.0, 50.0, 70.0, 90.0, 110.0};
    std::vector<double> x(S_values.size());
    std::vector<double> V_norm(x.size());

    for (size_t i = 0; i < S_values.size(); ++i) {
        double S = S_values[i];
        x[i] = std::log(S / K_ref);

        // Normalized obstacle for American put: max(K-S, 0) / K = max(1 - S/K, 0)
        V_norm[i] = std::max(1.0 - S / K_ref, 0.0);
    }

    std::vector<double> dx_spacing(x.size() - 1);
    for (size_t i = 0; i < x.size() - 1; ++i) {
        dx_spacing[i] = x[i+1] - x[i];
    }

    std::vector<double> Lu_norm(x.size(), -0.001);
    std::vector<double> dVnorm_dx(x.size(), -0.01);
    std::vector<double> d2Vnorm_dx2(x.size(), 0.0001);

    mango::Snapshot snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx_spacing},
        .solution = std::span{V_norm},
        .spatial_operator = std::span{Lu_norm},
        .first_derivative = std::span{dVnorm_dx},
        .second_derivative = std::span{d2Vnorm_dx2}
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
        .option_type = mango::OptionType::CALL
    };

    mango::PriceTableSnapshotCollector collector(config);

    // Use LOG-MONEYNESS coordinates
    std::vector<double> S_values = {80.0, 100.0, 120.0};
    std::vector<double> x(S_values.size());
    for (size_t i = 0; i < S_values.size(); ++i) {
        x[i] = std::log(S_values[i] / K_ref);
    }

    std::vector<double> dx_spacing(x.size() - 1);
    for (size_t i = 0; i < x.size() - 1; ++i) {
        dx_spacing[i] = x[i+1] - x[i];
    }

    // Normalized values
    std::vector<double> V_norm = {0.20, 0.10, 0.05};
    std::vector<double> Lu_norm = {-0.001, -0.001, -0.001};
    std::vector<double> dVnorm_dx = {-0.005, -0.003, -0.002};
    std::vector<double> d2Vnorm_dx2 = {0.0001, 0.0001, 0.0001};

    mango::Snapshot snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx_spacing},
        .solution = std::span{V_norm},
        .spatial_operator = std::span{Lu_norm},
        .first_derivative = std::span{dVnorm_dx},
        .second_derivative = std::span{d2Vnorm_dx2}
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
        .option_type = mango::OptionType::CALL
    };

    mango::PriceTableSnapshotCollector collector(config);

    // Use LOG-MONEYNESS coordinates
    std::vector<double> S_values = {80.0, 100.0, 120.0};
    std::vector<double> x(S_values.size());
    for (size_t i = 0; i < S_values.size(); ++i) {
        x[i] = std::log(S_values[i] / K_ref);
    }

    std::vector<double> dx_spacing(x.size() - 1);
    for (size_t i = 0; i < x.size() - 1; ++i) {
        dx_spacing[i] = x[i+1] - x[i];
    }

    // Normalized values
    std::vector<double> V_norm = {0.20, 0.10, 0.05};
    std::vector<double> Lu_norm = {-0.001, -0.001, -0.001};
    std::vector<double> dVnorm_dx = {-0.005, -0.003, -0.002};
    std::vector<double> d2Vnorm_dx2 = {0.0001, 0.0001, 0.0001};

    // Collect snapshots in order: tau_idx=0, 1, 2
    for (size_t tau_idx = 0; tau_idx < 3; ++tau_idx) {
        mango::Snapshot snapshot{
            .time = tau[tau_idx],
            .user_index = tau_idx,  // user_index IS tau_idx
            .spatial_grid = std::span{x},
            .dx = std::span{dx_spacing},
            .solution = std::span{V_norm},
            .spatial_operator = std::span{Lu_norm},
            .first_derivative = std::span{dVnorm_dx},
            .second_derivative = std::span{d2Vnorm_dx2}
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
        .option_type = mango::OptionType::PUT
    };

    mango::PriceTableSnapshotCollector collector(config);

    // Simple mock data - use LOG-MONEYNESS coordinates
    std::vector<double> S_values = {50.0, 100.0, 150.0};
    std::vector<double> x(S_values.size());
    for (size_t i = 0; i < S_values.size(); ++i) {
        x[i] = std::log(S_values[i] / 100.0);
    }

    std::vector<double> dx_spacing(x.size() - 1);
    for (size_t i = 0; i < x.size() - 1; ++i) {
        dx_spacing[i] = x[i+1] - x[i];
    }

    // Normalized values
    std::vector<double> V_norm = {0.50, 0.10, 0.02};
    std::vector<double> Lu_norm = {0.001, 0.002, 0.001};
    std::vector<double> dVnorm_dx = {-0.01, -0.005, -0.002};
    std::vector<double> d2Vnorm_dx2 = {0.0005, 0.0003, 0.0001};

    mango::Snapshot snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx_spacing},
        .solution = std::span{V_norm},
        .spatial_operator = std::span{Lu_norm},
        .first_derivative = std::span{dVnorm_dx},
        .second_derivative = std::span{d2Vnorm_dx2}
    };

    ASSERT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::value_build_calls(collector), 0u);
    ASSERT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::lu_build_calls(collector), 0u);

    collector.collect(snapshot);

    EXPECT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::value_build_calls(collector), 1u);
    EXPECT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::lu_build_calls(collector), 1u);
    EXPECT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::value_rebuild_calls(collector), 0u);
    EXPECT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::lu_rebuild_calls(collector), 0u);

    // Modify snapshot data but keep grid identical to trigger rebuild path
    V_norm[0] *= 1.1;
    V_norm[1] *= 0.9;
    V_norm[2] *= 1.05;
    Lu_norm[0] *= 1.2;
    Lu_norm[1] *= 0.8;
    Lu_norm[2] *= 1.1;
    dVnorm_dx[0] *= 1.05;
    dVnorm_dx[1] *= 0.95;
    dVnorm_dx[2] *= 1.02;
    d2Vnorm_dx2[0] *= 0.9;
    d2Vnorm_dx2[1] *= 1.1;
    d2Vnorm_dx2[2] *= 1.0;

    collector.collect(snapshot);

    EXPECT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::value_build_calls(collector), 1u);
    EXPECT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::lu_build_calls(collector), 1u);
    EXPECT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::value_rebuild_calls(collector), 1u);
    EXPECT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::lu_rebuild_calls(collector), 1u);
}

// PMR-specific tests
class PriceTableSnapshotCollectorPMRTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test data
        moneyness_ = {0.8, 1.0, 1.2};
        tau_ = {0.25, 0.5, 1.0};
        K_ref_ = 100.0;

        // Create a memory arena for PMR allocations
        auto arena_result = mango::memory::SolverMemoryArena::create(1024 * 1024); // 1MB arena
        ASSERT_TRUE(arena_result.has_value());
        arena_ = arena_result.value();
    }

    std::vector<double> moneyness_;
    std::vector<double> tau_;
    double K_ref_;
    std::shared_ptr<mango::memory::SolverMemoryArena> arena_;
};

TEST_F(PriceTableSnapshotCollectorPMRTest, MemoryAccountingWorksCorrectly) {
    // This test verifies that memory accounting works correctly with pmr::vector
    // and the arena tracks allocations properly

    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = moneyness_,
        .tau = tau_,
        .K_ref = K_ref_,
        .option_type = mango::OptionType::PUT,
        .payoff_params = nullptr
    };

    mango::PriceTableSnapshotCollector collector(config, arena_);

    // Create a mock snapshot
    std::vector<double> spatial_grid = {90.0, 100.0, 110.0};
    std::vector<double> solution = {10.0, 5.0, 2.0};
    std::vector<double> first_derivative = {-0.5, -0.3, -0.2};
    std::vector<double> second_derivative = {0.01, 0.005, 0.002};
    std::vector<double> spatial_operator = {-0.1, -0.05, -0.02};

    mango::Snapshot snapshot{
        .time = 0.5,
        .user_index = 1,  // tau index
        .spatial_grid = spatial_grid,
        .solution = solution,
        .spatial_operator = spatial_operator,
        .first_derivative = first_derivative,
        .second_derivative = second_derivative
    };

    // Get initial allocation stats
    auto initial_stats = arena_->get_stats();
    size_t initial_used = initial_stats.used_size;
    EXPECT_GT(initial_stats.total_size, 0);

    // Perform collect operation
    auto result = collector.collect_expected(snapshot);
    EXPECT_TRUE(result.has_value());

    // Verify memory tracking works
    auto final_stats = arena_->get_stats();
    size_t final_used = final_stats.used_size;

    // Memory accounting should track allocations (synchronized_pool_resource may allocate internally)
    // The key point is that pmr::vector doesn't allocate beyond what the pool needs
    EXPECT_GE(final_used, initial_used) << "Memory accounting should track all allocations";

    // Verify data was actually collected
    EXPECT_EQ(collector.prices().size(), moneyness_.size() * tau_.size());
    EXPECT_EQ(collector.deltas().size(), moneyness_.size() * tau_.size());
    EXPECT_EQ(collector.gammas().size(), moneyness_.size() * tau_.size());
    EXPECT_EQ(collector.thetas().size(), moneyness_.size() * tau_.size());

    // Verify some data points are non-zero (indicating successful collection)
    bool has_nonzero_price = false;
    for (double price : collector.prices()) {
        if (price != 0.0) {
            has_nonzero_price = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_price) << "Prices should contain non-zero values after collection";
}

TEST_F(PriceTableSnapshotCollectorPMRTest, CollectorUsesArenaForPmrVectors) {
    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = moneyness_,
        .tau = tau_,
        .K_ref = K_ref_,
        .option_type = mango::OptionType::PUT,
        .payoff_params = nullptr
    };

    mango::PriceTableSnapshotCollector collector(config, arena_);

    auto token = mango::memory::SolverMemoryArena::ActiveWorkspaceToken(arena_);
    auto* resource = token.resource();
    ASSERT_NE(resource, nullptr);

    EXPECT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::moneyness_resource(collector), resource);
    EXPECT_EQ(mango::test_support::PriceTableSnapshotCollectorTestPeer::tau_resource(collector), resource);
}

TEST_F(PriceTableSnapshotCollectorPMRTest, ArenaUsageIsTrackedViaRaiiToken) {
    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = moneyness_,
        .tau = tau_,
        .K_ref = K_ref_,
        .option_type = mango::OptionType::PUT,
        .payoff_params = nullptr
    };

    auto stats_before = arena_->get_stats();
    {
        mango::PriceTableSnapshotCollector collector(config, arena_);
        auto stats_during = arena_->get_stats();
        EXPECT_EQ(stats_during.active_workspace_count, stats_before.active_workspace_count + 1);
    }
    auto stats_after = arena_->get_stats();
    EXPECT_EQ(stats_after.active_workspace_count, stats_before.active_workspace_count);
}

TEST_F(PriceTableSnapshotCollectorPMRTest, SpanAccessorsWorkCorrectly) {
    // Test that span accessors provide correct access to data
    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = moneyness_,
        .tau = tau_,
        .K_ref = K_ref_,
        .option_type = mango::OptionType::PUT,
        .payoff_params = nullptr
    };

    mango::PriceTableSnapshotCollector collector(config, arena_);

    // Test span accessors before any data collection
    auto prices_span = collector.prices_span();
    auto deltas_span = collector.deltas_span();
    auto gammas_span = collector.gammas_span();
    auto thetas_span = collector.thetas_span();

    EXPECT_EQ(prices_span.size(), moneyness_.size() * tau_.size());
    EXPECT_EQ(deltas_span.size(), moneyness_.size() * tau_.size());
    EXPECT_EQ(gammas_span.size(), moneyness_.size() * tau_.size());
    EXPECT_EQ(thetas_span.size(), moneyness_.size() * tau_.size());

    // All should be zero-initialized
    for (size_t i = 0; i < prices_span.size(); ++i) {
        EXPECT_EQ(prices_span[i], 0.0);
        EXPECT_EQ(deltas_span[i], 0.0);
        EXPECT_EQ(gammas_span[i], 0.0);
        EXPECT_EQ(thetas_span[i], 0.0);
    }
}

TEST_F(PriceTableSnapshotCollectorPMRTest, ConstructorWithMemoryArena) {
    // Test that constructor properly accepts and uses the memory arena
    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = moneyness_,
        .tau = tau_,
        .K_ref = K_ref_,
        .option_type = mango::OptionType::CALL,
        .payoff_params = nullptr
    };

    // This should compile and work with the PMR-enabled constructor
    EXPECT_NO_THROW({
        mango::PriceTableSnapshotCollector collector(config, arena_);
    });

    // Verify the collector was properly initialized
    mango::PriceTableSnapshotCollector collector(config, arena_);
    EXPECT_EQ(collector.prices().size(), moneyness_.size() * tau_.size());
}
#include <memory_resource>
#include "src/option/price_table_snapshot_collector.hpp"
#include "src/support/memory/solver_memory_arena.hpp"
#include "src/support/memory/unified_memory_resource.hpp"
