#include "src/price_table_snapshot_collector.hpp"
#include "src/snapshot.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

// Copy the exact test setup from the failing test
class PriceTableSnapshotCollectorExpectedTest {
public:
    std::vector<double> moneyness = {0.8, 1.0, 1.2};
    std::vector<double> tau = {0.5};
    const double K_ref = 100.0;

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
        std::vector<double> S_values = {60.0, 80.0, 100.0, 120.0, 140.0};
        std::vector<double> x(S_values.size());
        std::vector<double> V_norm(x.size());
        std::vector<double> dVnorm_dx(x.size());
        std::vector<double> d2Vnorm_dx2(x.size());

        // Convert S values to log-moneyness and compute derivatives
        for (size_t i = 0; i < x.size(); ++i) {
            double S = S_values[i];
            x[i] = std::log(S / K_ref);
            V_norm[i] = (S * S) / K_ref;
            dVnorm_dx[i] = 2.0 * (S * S) / K_ref;
            d2Vnorm_dx2[i] = 4.0 * (S * S) / K_ref;
        }

        std::vector<double> dx_spacing(x.size() - 1);
        for (size_t i = 0; i < x.size() - 1; ++i) {
            dx_spacing[i] = x[i+1] - x[i];
        }

        std::vector<double> Lu(x.size(), 0.0);

        return mango::Snapshot{
            .time = 0.5,
            .user_index = 0,
            .spatial_grid = std::span{x},
            .dx = std::span{dx_spacing},
            .solution = std::span{V_norm},
            .spatial_operator = std::span{Lu},
            .first_derivative = std::span{dVnorm_dx},
            .second_derivative = std::span{d2Vnorm_dx2}
        };
    }
};

int main() {
    PriceTableSnapshotCollectorExpectedTest test;

    std::cout << "Creating collector with default config..." << std::endl;
    mango::PriceTableSnapshotCollector collector(test.CreateDefaultConfig());

    std::cout << "Creating valid snapshot..." << std::endl;
    auto snapshot = test.CreateValidSnapshot();

    std::cout << "Testing collect_expected..." << std::endl;
    auto result = collector.collect_expected(snapshot);

    if (!result.has_value()) {
        std::cout << "collect_expected failed with error: " << result.error() << std::endl;
        return 1;
    }

    std::cout << "collect_expected succeeded!" << std::endl;

    // Verify data was actually collected
    auto prices = collector.prices();
    auto gammas = collector.gammas();

    std::cout << "Prices size: " << prices.size() << std::endl;
    std::cout << "Gammas size: " << gammas.size() << std::endl;

    std::cout << "Prices: ";
    for (size_t i = 0; i < prices.size(); ++i) {
        std::cout << prices[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Gammas: ";
    for (size_t i = 0; i < gammas.size(); ++i) {
        std::cout << gammas[i] << " ";
    }
    std::cout << std::endl;

    // Verify some data was actually computed (not just zeros)
    for (size_t i = 0; i < prices.size(); ++i) {
        if (prices[i] <= 0.0) {
            std::cout << "FAIL: Found non-positive price at index " << i << ": " << prices[i] << std::endl;
            return 1;
        }
        if (gammas[i] <= 0.0) {
            std::cout << "FAIL: Found non-positive gamma at index " << i << ": " << gammas[i] << std::endl;
            return 1;
        }
    }

    std::cout << "SUCCESS: All checks passed!" << std::endl;
    return 0;
}