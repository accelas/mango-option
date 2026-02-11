// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace mango {

/// Axes for dimensionless 3D price surface.
struct DimensionlessAxes {
    std::vector<double> log_moneyness;  ///< x = ln(S/K), sorted ascending
    std::vector<double> tau_prime;       ///< sigma^2 * tau / 2, sorted ascending, > 0
    std::vector<double> ln_kappa;        ///< ln(2r/sigma^2), sorted ascending
};

/// Result of building a dimensionless 3D surface.
struct DimensionlessBuildResult {
    std::shared_ptr<const PriceTableSurfaceND<3>> surface;
    int n_pde_solves = 0;
    double build_time_seconds = 0.0;
};

/// Build a 3D B-spline surface over (x, tau', ln kappa) using dimensionless PDE.
///
/// For each kappa value, solves the Black-Scholes PDE in dimensionless coordinates
/// (sigma_eff = sqrt(2), r_eff = kappa, spot = strike = K_ref) with snapshots at
/// all tau' grid points. The solutions are resampled onto the log-moneyness grid,
/// EEP decomposed, then fit with a 3D tensor-product B-spline.
///
/// @param axes Grid axes (each needs >= 4 points)
/// @param K_ref Reference strike for PDE solves
/// @param option_type PUT or CALL
/// @return Build result or error
[[nodiscard]] std::expected<DimensionlessBuildResult, PriceTableError>
build_dimensionless_surface(
    const DimensionlessAxes& axes,
    double K_ref,
    OptionType option_type);

}  // namespace mango
