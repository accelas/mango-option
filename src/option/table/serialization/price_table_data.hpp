// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace mango {

/// Serializable representation of any PriceTable surface.
/// Plain vectors — no I/O dependencies.
struct PriceTableData {
    std::string surface_type;

    OptionType option_type = OptionType::PUT;
    double dividend_yield = 0.0;
    DividendSpec dividends;
    double maturity = 0.0;

    struct Segment {
        int32_t segment_id = 0;
        double K_ref = 0.0;
        double tau_start = 0.0, tau_end = 0.0;
        double tau_min = 0.0, tau_max = 0.0;
        std::string interp_type;  // "bspline" or "chebyshev"
        size_t ndim = 4;

        std::vector<double> domain_lo, domain_hi;
        std::vector<int32_t> num_pts;
        std::vector<std::vector<double>> grids;   // ndim vectors
        std::vector<std::vector<double>> knots;   // ndim vectors
        std::vector<double> values;               // coefficients or raw values
    };
    std::vector<Segment> segments;

    size_t n_pde_solves = 0;
    double precompute_time_seconds = 0.0;
};

}  // namespace mango
