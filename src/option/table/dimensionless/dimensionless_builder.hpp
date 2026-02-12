// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/support/error_types.hpp"
#include <array>
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

/// Result of solving the dimensionless PDE on a 3D grid.
///
/// Values are raw American V/K in dimensionless coordinates (NOT EEP-decomposed).
/// The caller is responsible for EEP decomposition before fitting if desired.
struct DimensionlessPDEResult {
    std::vector<double> values;  ///< Row-major (x × tau' × ln_kappa), American V/K
    int n_pde_solves = 0;
    double build_time_seconds = 0.0;
};

/// Solve the dimensionless PDE on arbitrary 3D grid nodes.
///
/// For each ln_kappa node, solves the Black-Scholes PDE in dimensionless
/// coordinates (sigma_eff = sqrt(2), r_eff = kappa) with snapshots at all
/// tau' nodes, then resamples onto the log-moneyness grid via cubic spline.
///
/// Returns raw American V/K values — no EEP decomposition.
///
/// @param axes Grid axes (each needs >= 2 points)
/// @param K_ref Reference strike for PDE solves
/// @param option_type PUT or CALL
/// @return Row-major American values or error
[[nodiscard]] std::expected<DimensionlessPDEResult, PriceTableError>
solve_dimensionless_pde(
    const DimensionlessAxes& axes,
    double K_ref,
    OptionType option_type);

/// Configuration for adaptive dimensionless surface building.
struct DimensionlessAdaptiveParams {
    double target_eep_error = 2e-3;    ///< Normalized EEP error threshold
    size_t max_iter = 10;              ///< Maximum refinement iterations
    size_t max_points_per_dim = 40;    ///< Grid ceiling per axis
    double refinement_factor = 1.5;    ///< Grid growth per iteration
    size_t lk_segments = 3;            ///< Split ln kappa axis into segments
    OptionType option_type = OptionType::PUT;

    double sigma_min = 0.10, sigma_max = 0.80;
    double rate_min = 0.005, rate_max = 0.10;
    double tau_min = 7.0 / 365, tau_max = 2.0;
    double moneyness_min = 0.65, moneyness_max = 1.50;  ///< S/K ratio
};

/// Segmented 3D dimensionless surface: splits the ln kappa axis into segments,
/// each with its own B-spline surface, and blends at segment boundaries.
class SegmentedDimensionlessSurface {
public:
    struct Segment {
        std::shared_ptr<const BSplineND<double, 3>> spline;
        double lk_min, lk_max;  ///< Physical ln kappa range (no headroom)
    };

    explicit SegmentedDimensionlessSurface(std::vector<Segment> segments)
        : segments_(std::move(segments)) {}

    [[nodiscard]] double value(const std::array<double, 3>& coords) const;
    [[nodiscard]] size_t num_segments() const noexcept { return segments_.size(); }
    [[nodiscard]] const std::vector<Segment>& segments() const noexcept { return segments_; }

private:
    std::vector<Segment> segments_;
};

/// Result of adaptive dimensionless surface building.
struct DimensionlessAdaptiveResult {
    std::shared_ptr<SegmentedDimensionlessSurface> surface;
    DimensionlessAxes final_axes;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    int total_pde_solves = 0;
    double total_build_time_seconds = 0.0;
    size_t iterations_used = 0;
    size_t num_segments = 0;
    std::array<double, 3> worst_probe = {};
    double worst_true_eep = 0.0;
    double worst_interp_eep = 0.0;
};

/// Build a 3D dimensionless surface with adaptive grid refinement.
[[nodiscard]] std::expected<DimensionlessAdaptiveResult, PriceTableError>
build_dimensionless_surface_adaptive(
    const DimensionlessAdaptiveParams& params = {},
    double K_ref = 100.0);

}  // namespace mango
