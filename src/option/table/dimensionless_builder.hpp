// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/price_table_metadata.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace mango {

/// Axes for dimensionless 3D price surface.
struct DimensionlessAxes {
    std::vector<double> log_moneyness;  ///< x = ln(S/K), sorted ascending
    std::vector<double> tau_prime;       ///< tau' = sigma^2 * tau / 2, sorted ascending, > 0
    std::vector<double> ln_kappa;        ///< ln(kappa) = ln(2r/sigma^2), sorted ascending
};

/// Result of building a dimensionless 3D surface.
struct DimensionlessBuildResult {
    std::shared_ptr<const PriceTableSurfaceND<3>> surface;
    PriceTableMetadata metadata;
    int n_pde_solves = 0;
    double build_time_seconds = 0.0;
};

/// Build a 3D B-spline surface over (x, tau', ln kappa) using dimensionless PDE.
///
/// For each kappa value, solves the Black-Scholes PDE in dimensionless coordinates
/// (sigma_eff = sqrt(2), r_eff = kappa, T_eff = tau_prime_max) with snapshots at
/// all tau' grid points. The solutions are resampled onto the log-moneyness grid
/// via cubic spline interpolation, optionally decomposed into EEP, then fit with
/// a 3D tensor-product B-spline.
///
/// @param axes Grid axes for the 3D surface (each axis needs >= 4 points)
/// @param K_ref Reference strike price for normalized PDE solves (spot = strike = K_ref)
/// @param option_type PUT or CALL
/// @param content Whether to store EarlyExercisePremium (default) or NormalizedPrice
/// @return Build result with surface, metadata, and diagnostics; or error
[[nodiscard]] std::expected<DimensionlessBuildResult, PriceTableError>
build_dimensionless_surface(
    const DimensionlessAxes& axes,
    double K_ref,
    OptionType option_type,
    SurfaceContent content = SurfaceContent::EarlyExercisePremium);

/// Configuration for adaptive dimensionless surface building.
///
/// The builder automatically finds the grid that achieves the target
/// interpolation accuracy. Physical domain bounds define the (σ, r, τ, S/K)
/// ranges to support; these are mapped to dimensionless coordinates internally.
struct DimensionlessAdaptiveParams {
    double target_eep_error = 2e-3;    ///< Normalized EEP error threshold (~$0.20 per $100 strike)
    size_t max_iter = 10;              ///< Maximum refinement iterations
    size_t max_points_per_dim = 40;    ///< Grid ceiling per axis (ln_kappa points = PDE solves)
    double refinement_factor = 1.5;    ///< Grid growth per iteration
    size_t lk_segments = 3;            ///< Split ln κ axis into this many segments
    OptionType option_type = OptionType::PUT;

    /// Physical domain bounds (mapped to dimensionless coordinates)
    double sigma_min = 0.10, sigma_max = 0.80;
    double rate_min = 0.005, rate_max = 0.10;
    double tau_min = 7.0 / 365, tau_max = 2.0;
    double moneyness_min = 0.65, moneyness_max = 1.50;  ///< S/K ratio
};

/// Segmented 3D dimensionless surface: splits the ln κ axis into segments,
/// each with its own B-spline surface, and blends at segment boundaries.
///
/// This overcomes the interpolatory B-spline's oscillation limit: each
/// segment covers a small ln κ range (~3 units) with moderate point count
/// (15-25 points), staying below the oscillation threshold while achieving
/// O(h⁴) accuracy with h ≈ 0.2 instead of h ≈ 0.7.
class SegmentedDimensionlessSurface {
public:
    struct Segment {
        std::shared_ptr<const PriceTableSurfaceND<3>> surface;
        double lk_min, lk_max;  ///< Physical ln κ range (no headroom)
    };

    explicit SegmentedDimensionlessSurface(std::vector<Segment> segments)
        : segments_(std::move(segments)) {}

    /// Evaluate EEP at (x, tau', ln_kappa) with segment dispatch and blending
    [[nodiscard]] double value(const std::array<double, 3>& coords) const;

    [[nodiscard]] size_t num_segments() const noexcept { return segments_.size(); }
    [[nodiscard]] const std::vector<Segment>& segments() const noexcept { return segments_; }

private:
    std::vector<Segment> segments_;
};

/// Result of adaptive dimensionless surface building.
struct DimensionlessAdaptiveResult {
    std::shared_ptr<SegmentedDimensionlessSurface> surface;
    PriceTableMetadata metadata;
    DimensionlessAxes final_axes;          ///< Grid after refinement (union of segments)
    double achieved_max_error = 0.0;       ///< Worst mid-cell EEP error
    double achieved_avg_error = 0.0;
    bool target_met = false;
    int total_pde_solves = 0;              ///< Build + validation PDE solves
    double total_build_time_seconds = 0.0;
    size_t iterations_used = 0;
    size_t num_segments = 0;               ///< Number of ln κ segments

    /// Worst probe location (for diagnostics)
    std::array<double, 3> worst_probe = {};  ///< {x, tau', ln_kappa}
    double worst_true_eep = 0.0;
    double worst_interp_eep = 0.0;
};

/// Build a 3D dimensionless surface with adaptive grid refinement.
///
/// Starts with a coarse grid, builds the surface, validates interpolation
/// quality at mid-cell points via reference PDE solves, and refines the
/// worst axis until the target EEP error is met.
///
/// @param params Adaptive configuration (target error, domain, iteration limits)
/// @param K_ref Reference strike price for normalized PDE solves
/// @return Adaptive result with surface, final grid, and diagnostics; or error
[[nodiscard]] std::expected<DimensionlessAdaptiveResult, PriceTableError>
build_dimensionless_surface_adaptive(
    const DimensionlessAdaptiveParams& params = {},
    double K_ref = 100.0);

}  // namespace mango
