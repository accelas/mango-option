// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"

#include <cmath>

namespace mango {

/// EEP accessor for a B-spline tensor + axes.
///
/// Linearizes the 4D (m, tau, sigma, rate) iteration in cache-friendly
/// order matching the row-major mdspan layout: m outermost, rate innermost.
class BSplineTensorAccessor {
public:
    BSplineTensorAccessor(PriceTensor& tensor, const PriceTableAxes& axes,
                          double K_ref)
        : tensor_(tensor), axes_(axes), K_ref_(K_ref),
          Nm_(axes.grids[0].size()), Nt_(axes.grids[1].size()),
          Nv_(axes.grids[2].size()), Nr_(axes.grids[3].size()) {}

    size_t size() const { return Nm_ * Nt_ * Nv_ * Nr_; }
    double strike() const { return K_ref_; }

    // Flat index layout: [m][tau][sigma][rate] (row-major, rate innermost)
    double american_price(size_t i) const {
        auto [mi, ti, vi, ri] = to_4d(i);
        return K_ref_ * tensor_.view[mi, ti, vi, ri];
    }

    double spot(size_t i) const {
        return std::exp(axes_.grids[0][to_4d(i).mi]) * K_ref_;
    }

    double tau(size_t i) const { return axes_.grids[1][to_4d(i).ti]; }
    double sigma(size_t i) const { return axes_.grids[2][to_4d(i).vi]; }
    double rate(size_t i) const { return axes_.grids[3][to_4d(i).ri]; }

    void set_value(size_t i, double v) {
        auto [mi, ti, vi, ri] = to_4d(i);
        tensor_.view[mi, ti, vi, ri] = v;
    }

private:
    struct Idx4D { size_t mi, ti, vi, ri; };

    // Row-major: rate varies fastest
    Idx4D to_4d(size_t flat) const {
        size_t ri = flat % Nr_;  flat /= Nr_;
        size_t vi = flat % Nv_;  flat /= Nv_;
        size_t ti = flat % Nt_;
        size_t mi = flat / Nt_;
        return {mi, ti, vi, ri};
    }

    PriceTensor& tensor_;
    const PriceTableAxes& axes_;
    double K_ref_;
    size_t Nm_, Nt_, Nv_, Nr_;
};

/// Build-time helper: converts a B-spline tensor of normalized American prices to EEP values.
struct EEPDecomposer {
    OptionType option_type;
    double K_ref;
    double dividend_yield;

    /// Transform tensor from V/K_ref (normalized American prices) to EEP values.
    void decompose(PriceTensor& tensor, const PriceTableAxes& axes) const {
        BSplineTensorAccessor accessor(tensor, axes, K_ref);
        analytical_eep_decompose(accessor, option_type, dividend_yield);
    }
};

}  // namespace mango
