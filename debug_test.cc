#include "src/price_table_snapshot_collector.hpp"
#include "src/snapshot.hpp"
#include <vector>
#include <cmath>
#include <iostream>

int main() {
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

    // Use the exact same data as the working test
    std::vector<double> S_values = {60.0, 80.0, 100.0, 120.0, 140.0};
    std::vector<double> x(S_values.size());
    std::vector<double> V_norm(x.size());
    std::vector<double> dVnorm_dx(x.size());
    std::vector<double> d2Vnorm_dx2(x.size());

    // Convert S values to log-moneyness and compute derivatives exactly like working test
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

    std::cout << "x values: ";
    for (double val : x) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "V_norm values: ";
    for (double val : V_norm) std::cout << val << " ";
    std::cout << std::endl;

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

    std::cout << "Testing new collect_expected() method..." << std::endl;
    auto result = collector.collect_expected(snapshot);
    if (result.has_value()) {
        std::cout << "collect_expected() succeeded!" << std::endl;

        // Now check the actual data
        auto prices = collector.prices();
        auto gammas = collector.gammas();

        std::cout << "Prices size: " << prices.size() << std::endl;
        std::cout << "Prices: ";
        for (double p : prices) std::cout << p << " ";
        std::cout << std::endl;

        std::cout << "Gammas size: " << gammas.size() << std::endl;
        std::cout << "Gammas: ";
        for (double g : gammas) std::cout << g << " ";
        std::cout << std::endl;

        // Check if any are zero
        bool has_zero_price = false, has_zero_gamma = false;
        for (double p : prices) if (p == 0.0) has_zero_price = true;
        for (double g : gammas) if (g == 0.0) has_zero_gamma = true;

        std::cout << "Has zero prices: " << has_zero_price << std::endl;
        std::cout << "Has zero gammas: " << has_zero_gamma << std::endl;

    } else {
        std::cout << "collect_expected() failed: " << result.error() << std::endl;
    }

    return 0;
}