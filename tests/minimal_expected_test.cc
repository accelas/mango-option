#include "src/option/price_table_snapshot_collector.hpp"
#include "src/option/snapshot.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// Minimal test that copies the working test exactly
TEST(MinimalExpectedTest, CopyWorkingTest) {
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

    // Test the expected pattern version
    auto result = collector.collect_expected(snapshot);

    // Should succeed
    EXPECT_TRUE(result.has_value());

    // Verify data was actually collected
    auto gammas = collector.gammas();
    EXPECT_EQ(gammas.size(), 3u);

    // Test ALL three moneyness points: ITM (0.8), ATM (1.0), OTM (1.2)
    for (size_t m_idx = 0; m_idx < 3; ++m_idx) {
        size_t idx = m_idx * 1 + 0;  // tau_idx=0
        double m = moneyness[m_idx];
        EXPECT_NEAR(gammas[idx], 2.0, 1e-6)
            << "Gamma must be 2.0 everywhere, failed at m=" << m;
    }
}