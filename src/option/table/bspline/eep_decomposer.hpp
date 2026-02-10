// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"

namespace mango {

/// Build-time helper: converts a B-spline tensor of normalized American prices to EEP values.
struct EEPDecomposer {
    OptionType option_type;
    double K_ref;
    double dividend_yield;

    /// Transform tensor from V/K_ref (normalized American prices) to EEP values.
    ///
    /// @param tensor In/out: tensor of normalized prices (V/K_ref), overwritten with EEP
    /// @param axes Grid axes (axis 0 = log-moneyness, 1 = tau, 2 = sigma, 3 = rate)
    void decompose(PriceTensor& tensor, const PriceTableAxes& axes) const {
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
                        double american_price = K_ref * tensor.view[i, j, v_idx, r_idx];

                        tensor.view[i, j, v_idx, r_idx] = compute_eep(
                            american_price, spot, K_ref, tau, sigma, rate,
                            dividend_yield, option_type);
                    }
                }
            }
        }
    }
};

}  // namespace mango
