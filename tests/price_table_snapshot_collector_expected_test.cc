#include "src/option/price_table_snapshot_collector.hpp"
#include "src/option/snapshot.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>

// Test error handling with expected pattern
class PriceTableSnapshotCollectorExpectedTest : public ::testing::Test {
protected:
    std::vector<double> moneyness = {0.8, 1.0, 1.2};
    std::vector<double> tau = {0.5};
    const double K_ref = 100.0;

    // Data storage for CreateValidSnapshot - ensures lifetime safety
    std::vector<double> snapshot_S_values;
    std::vector<double> snapshot_x;
    std::vector<double> snapshot_V_norm;
    std::vector<double> snapshot_dVnorm_dx;
    std::vector<double> snapshot_d2Vnorm_dx2;
    std::vector<double> snapshot_dx_spacing;
    std::vector<double> snapshot_Lu;

    mango::PriceTableSnapshotCollectorConfig CreateDefaultConfig() {
        return mango::PriceTableSnapshotCollectorConfig{
            .moneyness = std::span{moneyness},
            .tau = std::span{tau},
            .K_ref = K_ref,
            .option_type = mango::OptionType::PUT,
            .payoff_params = nullptr
        };
    }

    mango::Snapshot CreateValidSnapshot() {
        // Mock PDE solution in LOG-MONEYNESS coordinates
        snapshot_S_values = {60.0, 80.0, 100.0, 120.0, 140.0};
        snapshot_x.resize(snapshot_S_values.size());
        snapshot_V_norm.resize(snapshot_S_values.size());
        snapshot_dVnorm_dx.resize(snapshot_S_values.size());
        snapshot_d2Vnorm_dx2.resize(snapshot_S_values.size());

        // Convert S values to log-moneyness and compute derivatives
        for (size_t i = 0; i < snapshot_x.size(); ++i) {
            double S = snapshot_S_values[i];
            snapshot_x[i] = std::log(S / K_ref);
            snapshot_V_norm[i] = (S * S) / K_ref;
            snapshot_dVnorm_dx[i] = 2.0 * (S * S) / K_ref;
            snapshot_d2Vnorm_dx2[i] = 4.0 * (S * S) / K_ref;
        }

        snapshot_dx_spacing.resize(snapshot_x.size() - 1);
        for (size_t i = 0; i < snapshot_x.size() - 1; ++i) {
            snapshot_dx_spacing[i] = snapshot_x[i+1] - snapshot_x[i];
        }

        snapshot_Lu.assign(snapshot_x.size(), 0.0);

        return mango::Snapshot{
            .time = 0.5,
            .user_index = 0,
            .spatial_grid = std::span{snapshot_x},
            .dx = std::span{snapshot_dx_spacing},
            .solution = std::span{snapshot_V_norm},
            .spatial_operator = std::span{snapshot_Lu},
            .first_derivative = std::span{snapshot_dVnorm_dx},
            .second_derivative = std::span{snapshot_d2Vnorm_dx2}
        };
    }
};

// Test successful collection with expected pattern
TEST_F(PriceTableSnapshotCollectorExpectedTest, SuccessfulCollectionReturnsExpectedVoid) {
    mango::PriceTableSnapshotCollector collector(CreateDefaultConfig());
    auto snapshot = CreateValidSnapshot();

    // This should succeed and return std::expected<void, std::string> with no error
    auto result = collector.collect_expected(snapshot);

    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result.has_value() == false);  // No error

    // Verify data was actually collected
    auto prices = collector.prices();
    auto deltas = collector.deltas();
    auto gammas = collector.gammas();
    auto thetas = collector.thetas();

    EXPECT_EQ(prices.size(), 3u);  // 3 moneyness points
    EXPECT_EQ(deltas.size(), 3u);
    EXPECT_EQ(gammas.size(), 3u);
    EXPECT_EQ(thetas.size(), 3u);

    // Verify some data was actually computed (not just zeros)
    for (size_t i = 0; i < prices.size(); ++i) {
        EXPECT_GT(prices[i], 0.0);
        EXPECT_GT(gammas[i], 0.0);
    }
}

// Test error handling when interpolator build fails
TEST_F(PriceTableSnapshotCollectorExpectedTest, InterpolatorBuildFailure) {
    mango::PriceTableSnapshotCollector collector(CreateDefaultConfig());

    // Create snapshot with invalid grid (monotonicity issues)
    std::vector<double> x = {0.0, 0.5, 0.3, 1.0};  // Non-monotonic
    std::vector<double> dx_spacing(x.size() - 1);
    for (size_t i = 0; i < x.size() - 1; ++i) {
        dx_spacing[i] = x[i+1] - x[i];
    }

    std::vector<double> V_norm(x.size(), 0.1);
    std::vector<double> Lu(x.size(), 0.0);
    std::vector<double> dVnorm_dx(x.size(), 0.0);
    std::vector<double> d2Vnorm_dx2(x.size(), 0.0);

    mango::Snapshot invalid_snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx_spacing},
        .solution = std::span{V_norm},
        .spatial_operator = std::span{Lu},
        .first_derivative = std::span{dVnorm_dx},
        .second_derivative = std::span{d2Vnorm_dx2}
    };

    auto result = collector.collect_expected(invalid_snapshot);

    // Should return an error
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_value() == false);

    // Check error message
    if (!result.has_value()) {
        EXPECT_FALSE(result.error().empty());
        EXPECT_TRUE(result.error().find("interpolator") != std::string::npos ||
                   result.error().find("Failed") != std::string::npos);
    }
}

// Test error handling with mismatched array sizes
TEST_F(PriceTableSnapshotCollectorExpectedTest, MismatchedArraySizes) {
    mango::PriceTableSnapshotCollector collector(CreateDefaultConfig());

    // Create snapshot with mismatched array sizes
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> dx_spacing = {0.5, 0.5};  // Correct size
    std::vector<double> V_norm = {0.1, 0.2};      // Wrong size (should be 3)
    std::vector<double> Lu = {0.0, 0.0};          // Wrong size
    std::vector<double> dVnorm_dx = {0.0, 0.0};   // Wrong size
    std::vector<double> d2Vnorm_dx2 = {0.0, 0.0}; // Wrong size

    mango::Snapshot mismatched_snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx_spacing},
        .solution = std::span{V_norm},
        .spatial_operator = std::span{Lu},
        .first_derivative = std::span{dVnorm_dx},
        .second_derivative = std::span{d2Vnorm_dx2}
    };

    auto result = collector.collect_expected(mismatched_snapshot);

    // Should return an error due to size mismatch
    EXPECT_FALSE(result.has_value());
    if (!result.has_value()) {
        EXPECT_FALSE(result.error().empty());
    }
}

// Test multiple snapshots with expected pattern
TEST_F(PriceTableSnapshotCollectorExpectedTest, MultipleSnapshotsExpectedPattern) {
    std::vector<double> tau_multi = {0.25, 0.5, 0.75};

    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = std::span{moneyness},
        .tau = std::span{tau_multi},
        .K_ref = K_ref,
        .option_type = mango::OptionType::CALL,
        .payoff_params = nullptr
    };

    mango::PriceTableSnapshotCollector collector(config);

    // Collect multiple snapshots
    for (size_t tau_idx = 0; tau_idx < tau_multi.size(); ++tau_idx) {
        auto snapshot = CreateValidSnapshot();
        snapshot.user_index = tau_idx;
        snapshot.time = tau_multi[tau_idx];

        auto result = collector.collect_expected(snapshot);

        // Each collection should succeed
        EXPECT_TRUE(result.has_value()) << "Failed at tau_idx=" << tau_idx;
    }

    // Verify all data was collected
    auto prices = collector.prices();
    auto deltas = collector.deltas();
    auto gammas = collector.gammas();
    auto thetas = collector.thetas();

    EXPECT_EQ(prices.size(), 9u);  // 3 moneyness × 3 tau = 9
    EXPECT_EQ(deltas.size(), 9u);
    EXPECT_EQ(gammas.size(), 9u);
    EXPECT_EQ(thetas.size(), 9u);
}

// Test error propagation in rebuild scenarios
TEST_F(PriceTableSnapshotCollectorExpectedTest, RebuildErrorPropagation) {
    mango::PriceTableSnapshotCollector collector(CreateDefaultConfig());

    // First, successful collection to build interpolators
    auto snapshot1 = CreateValidSnapshot();
    auto result1 = collector.collect_expected(snapshot1);
    EXPECT_TRUE(result1.has_value());

    // Now try to rebuild with invalid data
    std::vector<double> x_bad = {0.0, 0.5, 0.3, 1.0};  // Non-monotonic
    std::vector<double> dx_spacing_bad(x_bad.size() - 1);
    for (size_t i = 0; i < x_bad.size() - 1; ++i) {
        dx_spacing_bad[i] = x_bad[i+1] - x_bad[i];
    }

    std::vector<double> V_norm_bad(x_bad.size(), 0.1);
    std::vector<double> Lu_bad(x_bad.size(), 0.0);
    std::vector<double> dVnorm_dx_bad(x_bad.size(), 0.0);
    std::vector<double> d2Vnorm_dx2_bad(x_bad.size(), 0.0);

    mango::Snapshot bad_snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x_bad},
        .dx = std::span{dx_spacing_bad},
        .solution = std::span{V_norm_bad},
        .spatial_operator = std::span{Lu_bad},
        .first_derivative = std::span{dVnorm_dx_bad},
        .second_derivative = std::span{d2Vnorm_dx2_bad}
    };

    auto result2 = collector.collect_expected(bad_snapshot);

    // Should fail during rebuild
    EXPECT_FALSE(result2.has_value());
    if (!result2.has_value()) {
        EXPECT_FALSE(result2.error().empty());
        EXPECT_TRUE(result2.error().find("rebuild") != std::string::npos ||
                   result2.error().find("Failed") != std::string::npos);
    }
}

// Test that successful operations don't affect subsequent calls
TEST_F(PriceTableSnapshotCollectorExpectedTest, SuccessfulOperationsAreIndependent) {
    mango::PriceTableSnapshotCollector collector(CreateDefaultConfig());

    // Create two valid snapshots with different data
    auto snapshot1 = CreateValidSnapshot();
    auto result1 = collector.collect_expected(snapshot1);
    EXPECT_TRUE(result1.has_value());

    auto snapshot2 = CreateValidSnapshot();
    snapshot2.time = 0.75;  // Different time
    auto result2 = collector.collect_expected(snapshot2);
    EXPECT_TRUE(result2.has_value());

    // Both should succeed independently
    EXPECT_TRUE(result1.has_value());
    EXPECT_TRUE(result2.has_value());
}