#include "src/price_table_snapshot_collector.hpp"
#include "src/snapshot.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

int main() {
    // Exact same data as the failing test
    std::vector<double> moneyness = {0.8, 1.0, 1.2};
    std::vector<double> tau = {0.5};
    const double K_ref = 100.0;

    mango::PriceTableSnapshotCollectorConfig config{
        .moneyness = std::span{moneyness},
        .tau = std::span{tau},
        .K_ref = K_ref,
        .option_type = mango::OptionType::PUT,
        .payoff_params = nullptr
    };

    mango::PriceTableSnapshotCollector collector(config);

    // Exact same snapshot creation as CreateValidSnapshot
    std::vector<double> S_values = {60.0, 80.0, 100.0, 120.0, 140.0};
    std::vector<double> x(S_values.size());
    std::vector<double> V_norm(x.size());
    std::vector<double> dVnorm_dx(x.size());
    std::vector<double> d2Vnorm_dx2(x.size());

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

    auto snapshot = mango::Snapshot{
        .time = 0.5,
        .user_index = 0,
        .spatial_grid = std::span{x},
        .dx = std::span{dx_spacing},
        .solution = std::span{V_norm},
        .spatial_operator = std::span{Lu},
        .first_derivative = std::span{dVnorm_dx},
        .second_derivative = std::span{d2Vnorm_dx2}
    };

    auto result = collector.collect_expected(snapshot);

    if (!result.has_value()) {
        std::cout << "FAILED: " << result.error() << std::endl;
        return 1;
    }

    // Verify data was actually collected - exact same checks as the failing test
    auto prices = collector.prices();
    auto gammas = collector.gammas();

    if (prices.size() != 3u) {
        std::cout << "FAIL: Expected prices size 3, got " << prices.size() << std::endl;
        return 1;
    }
    if (gammas.size() != 3u) {
        std::cout << "FAIL: Expected gammas size 3, got " << gammas.size() << std::endl;
        return 1;
    }

    // Check for zeros like the test does - this is where it fails
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