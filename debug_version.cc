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

    std::cout << "Creating collector..." << std::endl;
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

    std::cout << "Snapshot data before construction:" << std::endl;
    std::cout << "x.size() = " << x.size() << std::endl;
    std::cout << "x data: ";
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Checking monotonicity manually: ";
    bool ok = true;
    for (size_t i = 1; i < x.size() && ok; ++i) {
        if (x[i] <= x[i-1]) {
            std::cout << "FAILED at i=" << i << ": " << x[i] << " <= " << x[i-1] << std::endl;
            ok = false;
        }
    }
    if (ok) std::cout << "PASSED" << std::endl;

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

    std::cout << "Snapshot spatial_grid.size() = " << snapshot.spatial_grid.size() << std::endl;
    std::cout << "Snapshot spatial_grid data: ";
    for (size_t i = 0; i < snapshot.spatial_grid.size(); ++i) {
        std::cout << snapshot.spatial_grid[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Testing collect_expected..." << std::endl;
    auto result = collector.collect_expected(snapshot);

    if (!result.has_value()) {
        std::cout << "collect_expected failed with error: " << result.error() << std::endl;

        // Try to debug what went wrong by testing the interpolator directly
        mango::SnapshotInterpolator interp;
        auto interp_result = interp.build(snapshot.spatial_grid, snapshot.solution);
        if (interp_result.has_value()) {
            std::cout << "Direct interpolator test also failed: " << interp_result.value() << std::endl;
        } else {
            std::cout << "Direct interpolator test succeeded - problem is elsewhere" << std::endl;
        }

        return 1;
    }

    std::cout << "collect_expected succeeded!" << std::endl;
    return 0;
}