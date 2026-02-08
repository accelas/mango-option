// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/table/price_tensor.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/math/bspline_nd_separable.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include <cmath>
#include <chrono>

namespace mango {

// ===========================================================================
// SegmentedDimensionlessSurface
// ===========================================================================

double SegmentedDimensionlessSurface::value(
    const std::array<double, 3>& coords) const
{
    double lk = coords[2];

    // Find the segment containing this ln κ value
    // If between segments, blend linearly over a transition zone
    for (size_t i = 0; i < segments_.size(); ++i) {
        if (lk <= segments_[i].lk_max || i == segments_.size() - 1) {
            double val = segments_[i].surface->value(coords);

            // Blend with next segment near upper boundary
            if (i + 1 < segments_.size()) {
                double blend_lo = segments_[i].lk_max;
                double blend_hi = segments_[i + 1].lk_min;
                // Overlap region: [next.lk_min, this.lk_max]
                // (segments overlap so blend_hi < blend_lo)
                if (lk >= blend_hi && lk <= blend_lo) {
                    double t = (lk - blend_hi) / (blend_lo - blend_hi);
                    double val_next = segments_[i + 1].surface->value(coords);
                    val = (1.0 - t) * val_next + t * val;
                }
            }

            return std::max(val, 0.0);
        }
    }

    // Fallback: last segment
    return std::max(segments_.back().surface->value(coords), 0.0);
}

std::expected<DimensionlessBuildResult, PriceTableError>
build_dimensionless_surface(
    const DimensionlessAxes& axes,
    double K_ref,
    OptionType option_type,
    SurfaceContent content)
{
    auto t_start = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------------
    // 1. Validate: each axis needs >= 4 points for cubic B-spline fitting
    // -----------------------------------------------------------------------
    const size_t Nm = axes.log_moneyness.size();
    const size_t Nt = axes.tau_prime.size();
    const size_t Nk = axes.ln_kappa.size();

    if (Nm < 4) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InsufficientGridPoints, 0, Nm});
    }
    if (Nt < 4) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InsufficientGridPoints, 1, Nt});
    }
    if (Nk < 4) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InsufficientGridPoints, 2, Nk});
    }

    // -----------------------------------------------------------------------
    // 2. Create batch: one PricingParams per kappa value
    //    Dimensionless mapping: sigma_eff = sqrt(2), r_eff = kappa,
    //    T_eff = tau_prime.back(), spot = strike = K_ref, q = 0
    // -----------------------------------------------------------------------
    const double sigma_eff = std::sqrt(2.0);
    // Set PDE maturity slightly beyond the last snapshot time so that the
    // snapshot at tau_prime.back() is captured as an interior snapshot rather
    // than coinciding with the final time step (which would be deduplicated).
    const double tau_prime_max = axes.tau_prime.back() * 1.01;

    std::vector<PricingParams> batch;
    batch.reserve(Nk);
    for (size_t k = 0; k < Nk; ++k) {
        double kappa = std::exp(axes.ln_kappa[k]);
        batch.emplace_back(
            OptionSpec{
                .spot = K_ref,
                .strike = K_ref,
                .maturity = tau_prime_max,
                .rate = kappa,
                .dividend_yield = 0.0,
                .option_type = option_type},
            sigma_eff);
    }

    // -----------------------------------------------------------------------
    // 3. Solve batch with snapshot times = tau_prime grid, shared grid
    // -----------------------------------------------------------------------
    BatchAmericanOptionSolver solver;
    // Use high accuracy to minimize PDE discretization noise — the B-spline
    // interpolant amplifies any noise in the training data.
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    solver.set_snapshot_times(std::span<const double>{axes.tau_prime});

    auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);

    if (!batch_result.all_succeeded()) {
        // Find first failure for diagnostics
        size_t first_fail_idx = 0;
        for (size_t i = 0; i < batch_result.results.size(); ++i) {
            if (!batch_result.results[i].has_value()) {
                first_fail_idx = i;
                break;
            }
        }
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::ExtractionFailed, first_fail_idx,
            batch_result.failed_count});
    }

    // -----------------------------------------------------------------------
    // 4. Extract tensor: PriceTensorND<3> with shape {Nm, Nt, Nk}
    // -----------------------------------------------------------------------
    auto tensor_result = PriceTensorND<3>::create({Nm, Nt, Nk});
    if (!tensor_result.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::TensorCreationFailed});
    }
    auto tensor = std::move(tensor_result.value());

    for (size_t k = 0; k < Nk; ++k) {
        const auto& result = batch_result.results[k].value();
        auto grid = result.grid();
        auto x_grid = grid->x();  // Spatial grid (log-moneyness)

        // Verify we got the expected number of snapshots
        if (result.num_snapshots() < Nt) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::ExtractionFailed, k,
                result.num_snapshots()});
        }

        for (size_t j = 0; j < Nt; ++j) {
            std::span<const double> spatial_solution = result.at_time(j);

            // Build cubic spline to resample PDE solution onto our moneyness grid
            CubicSpline<double> spline;
            auto build_error = spline.build(x_grid, spatial_solution);

            if (build_error.has_value()) {
                return std::unexpected(PriceTableError{
                    PriceTableErrorCode::ExtractionFailed, 0, k});
            }

            // Evaluate spline at each log-moneyness point.
            // The PDE solve is normalized (spot=strike=K_ref), so solution
            // gives V/K_ref directly as a function of log-moneyness.
            for (size_t i = 0; i < Nm; ++i) {
                tensor.view[i, j, k] = spline.eval(axes.log_moneyness[i]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 5. EEP decomposition (if content == EarlyExercisePremium)
    //    Both PDE solution and dimensionless_european return V/K_ref (normalized),
    //    so EEP = American - European.
    // -----------------------------------------------------------------------
    if (content == SurfaceContent::EarlyExercisePremium) {
        for (size_t i = 0; i < Nm; ++i) {
            const double x = axes.log_moneyness[i];
            for (size_t j = 0; j < Nt; ++j) {
                const double tp = axes.tau_prime[j];
                for (size_t k = 0; k < Nk; ++k) {
                    double kappa = std::exp(axes.ln_kappa[k]);
                    double eu = dimensionless_european(x, tp, kappa, option_type);
                    double eep = tensor.view[i, j, k] - eu;
                    tensor.view[i, j, k] = std::max(eep, 0.0);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 6. Fit B-spline: BSplineNDSeparable<double, 3>
    // -----------------------------------------------------------------------
    std::array<std::vector<double>, 3> grids = {
        axes.log_moneyness, axes.tau_prime, axes.ln_kappa};

    auto fitter_result = BSplineNDSeparable<double, 3>::create(std::move(grids));
    if (!fitter_result.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
    }

    // Extract values from tensor in row-major order
    size_t total = Nm * Nt * Nk;
    std::vector<double> values;
    values.reserve(total);
    for (size_t i = 0; i < Nm; ++i) {
        for (size_t j = 0; j < Nt; ++j) {
            for (size_t k = 0; k < Nk; ++k) {
                values.push_back(tensor.view[i, j, k]);
            }
        }
    }

    auto fit_result = fitter_result->fit(std::move(values));
    if (!fit_result.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
    }

    // -----------------------------------------------------------------------
    // 7. Build surface: PriceTableSurfaceND<3>
    // -----------------------------------------------------------------------
    PriceTableAxesND<3> surface_axes;
    surface_axes.grids[0] = axes.log_moneyness;
    surface_axes.grids[1] = axes.tau_prime;
    surface_axes.grids[2] = axes.ln_kappa;
    surface_axes.names = {"log_moneyness", "tau_prime", "ln_kappa"};

    PriceTableMetadata metadata{
        .K_ref = K_ref,
        .dividends = {},
        .m_min = axes.log_moneyness.front(),
        .m_max = axes.log_moneyness.back(),
        .content = content,
    };

    auto surface_result = PriceTableSurfaceND<3>::build(
        std::move(surface_axes),
        std::move(fit_result.value().coefficients),
        metadata);

    if (!surface_result.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::SurfaceBuildFailed});
    }

    // -----------------------------------------------------------------------
    // 8. Return result
    // -----------------------------------------------------------------------
    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    return DimensionlessBuildResult{
        .surface = std::move(surface_result.value()),
        .metadata = metadata,
        .n_pde_solves = static_cast<int>(Nk),
        .build_time_seconds = elapsed,
    };
}

}  // namespace mango
