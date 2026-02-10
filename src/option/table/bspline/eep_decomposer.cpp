// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/eep_decomposer.hpp"
#include "mango/option/european_option.hpp"
#include <algorithm>
#include <cmath>

namespace mango {

void EEPDecomposer::decompose(PriceTensor& tensor, const PriceTableAxes& axes) const {
    const size_t Nm = axes.grids[0].size();
    const size_t Nt = axes.grids[1].size();
    const size_t Nv = axes.grids[2].size();
    const size_t Nr = axes.grids[3].size();

    for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
        double rate = axes.grids[3][r_idx];
        for (size_t v_idx = 0; v_idx < Nv; ++v_idx) {
            double sigma = axes.grids[2][v_idx];
            for (size_t j = 0; j < Nt; ++j) {
                double tau = axes.grids[1][j];
                for (size_t i = 0; i < Nm; ++i) {
                    double x = axes.grids[0][i];  // log-moneyness
                    double spot = std::exp(x) * K_ref;

                    // Current value is normalized price (V/K_ref)
                    double normalized_price = tensor.view[i, j, v_idx, r_idx];
                    double american_price = K_ref * normalized_price;

                    auto eu = EuropeanOptionSolver(
                        OptionSpec{.spot = spot, .strike = K_ref, .maturity = tau,
                            .rate = rate, .dividend_yield = dividend_yield,
                            .option_type = option_type}, sigma).solve().value();

                    double eep_raw = american_price - eu.value();

                    // Debiased softplus floor: smooth non-negativity with zero bias at eep_raw=0
                    constexpr double kSharpness = 100.0;
                    if (kSharpness * eep_raw > 500.0) {
                        // Overflow protection: softplus ≈ x for large x
                        tensor.view[i, j, v_idx, r_idx] = eep_raw;
                    } else {
                        double softplus = std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
                        double bias = std::log(2.0) / kSharpness;  // softplus(0) ≈ 0.00693
                        tensor.view[i, j, v_idx, r_idx] = std::max(0.0, softplus - bias);
                    }
                }
            }
        }
    }
}

}  // namespace mango
