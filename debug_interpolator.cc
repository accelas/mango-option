#include "src/snapshot_interpolator.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

int main() {
    // Test the interpolator directly
    std::vector<double> S_values = {60.0, 80.0, 100.0, 120.0, 140.0};
    std::vector<double> x(S_values.size());
    std::vector<double> V_norm(x.size());
    const double K_ref = 100.0;

    for (size_t i = 0; i < x.size(); ++i) {
        double S = S_values[i];
        x[i] = std::log(S / K_ref);
        V_norm[i] = (S * S) / K_ref;
    }

    std::vector<double> dx_spacing(x.size() - 1);
    for (size_t i = 0; i < x.size() - 1; ++i) {
        dx_spacing[i] = x[i+1] - x[i];
    }

    std::cout << "Testing interpolator directly..." << std::endl;
    std::cout << "x values: ";
    for (double val : x) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "Checking monotonicity: ";
    bool monotonic = true;
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i] <= x[i-1]) {
            std::cout << "FAILED at index " << i << ": " << x[i] << " <= " << x[i-1] << std::endl;
            monotonic = false;
        }
    }
    if (monotonic) std::cout << "PASSED" << std::endl;

    mango::SnapshotInterpolator interp;
    auto error = interp.build(std::span{x}, std::span{V_norm});

    if (error.has_value()) {
        std::cout << "Interpolator build failed: " << error.value() << std::endl;
        return 1;
    } else {
        std::cout << "Interpolator build succeeded!" << std::endl;
        return 0;
    }
}