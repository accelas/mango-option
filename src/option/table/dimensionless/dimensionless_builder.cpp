// SPDX-License-Identifier: MIT

#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"  // for PriceTensorND<3>
#include "mango/option/american_option_batch.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/math/bspline_nd_separable.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include <cmath>
#include <chrono>

namespace mango {

namespace {

/// Validate that all axes have >= 4 points for cubic B-spline.
std::expected<void, PriceTableError>
validate_axes(const DimensionlessAxes& axes) {
    if (axes.log_moneyness.size() < 4) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InsufficientGridPoints, 0,
            axes.log_moneyness.size()});
    }
    if (axes.tau_prime.size() < 4) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InsufficientGridPoints, 1,
            axes.tau_prime.size()});
    }
    if (axes.ln_kappa.size() < 4) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InsufficientGridPoints, 2,
            axes.ln_kappa.size()});
    }
    return {};
}

}  // namespace

std::expected<DimensionlessBuildResult, PriceTableError>
build_dimensionless_surface(
    const DimensionlessAxes& axes,
    double K_ref,
    OptionType option_type)
{
    auto start = std::chrono::steady_clock::now();

    // Step 1: Validate axes
    auto val = validate_axes(axes);
    if (!val.has_value()) {
        return std::unexpected(val.error());
    }

    const size_t Nm = axes.log_moneyness.size();
    const size_t Nt = axes.tau_prime.size();
    const size_t Nk = axes.ln_kappa.size();

    // Step 2: Create 3D tensor
    auto tensor_result = PriceTensorND<3>::create({Nm, Nt, Nk});
    if (!tensor_result.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::TensorCreationFailed});
    }
    auto tensor = std::move(tensor_result.value());

    // sigma_eff = sqrt(2) for dimensionless PDE
    const double sigma_eff = std::sqrt(2.0);

    // PDE maturity: slightly beyond last tau_prime snapshot
    const double pde_maturity = axes.tau_prime.back() * 1.01;

    // Step 3: For each kappa value, solve the dimensionless PDE
    int n_pde_solves = 0;

    for (size_t k = 0; k < Nk; ++k) {
        const double kappa = std::exp(axes.ln_kappa[k]);
        const double r_eff = kappa;  // r_eff = kappa = 2r/sigma^2

        // Create PricingParams in dimensionless coordinates
        PricingParams params(
            OptionSpec{
                .spot = K_ref,
                .strike = K_ref,
                .maturity = pde_maturity,
                .rate = r_eff,
                .dividend_yield = 0.0,
                .option_type = option_type},
            sigma_eff);

        // Set up batch solver with snapshot times and ultra accuracy
        BatchAmericanOptionSolver batch_solver;
        batch_solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
        batch_solver.set_snapshot_times(
            std::span<const double>{axes.tau_prime.data(), axes.tau_prime.size()});

        // Solve single PDE (batch of 1)
        std::vector<PricingParams> batch = {params};
        auto batch_result = batch_solver.solve_batch(batch, true);
        ++n_pde_solves;

        if (batch_result.failed_count > 0 || !batch_result.results[0].has_value()) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::ExtractionFailed, 2, k});
        }

        const auto& result = batch_result.results[0].value();
        auto grid = result.grid();
        auto pde_x = grid->x();  // log-moneyness grid from PDE solver

        // Step 5: For each snapshot time, resample PDE solution onto
        // our log-moneyness grid using cubic spline
        CubicSpline<double> spline;

        for (size_t j = 0; j < Nt; ++j) {
            auto solution = result.at_time(j);

            // Build cubic spline on PDE grid
            auto err = spline.build(pde_x, solution);
            if (err.has_value()) {
                return std::unexpected(PriceTableError{
                    PriceTableErrorCode::ExtractionFailed, 1, j});
            }

            // Resample onto our log-moneyness grid and EEP decompose
            for (size_t i = 0; i < Nm; ++i) {
                const double x = axes.log_moneyness[i];

                // PDE value (V/K in dimensionless coords)
                double pde_val = spline.eval(x);

                // European value in dimensionless coords
                double eu_val = dimensionless_european(
                    x, axes.tau_prime[j], kappa, option_type);

                // EEP = American - European
                double eep = pde_val - eu_val;

                // Clamp negative EEP to zero (numerical noise)
                if (eep < 0.0) eep = 0.0;

                tensor.view[i, j, k] = eep;
            }
        }
    }

    // Step 7: Extract values in row-major order for B-spline fitting
    const size_t total = Nm * Nt * Nk;
    std::vector<double> values(total);
    for (size_t i = 0; i < Nm; ++i) {
        for (size_t j = 0; j < Nt; ++j) {
            for (size_t k_idx = 0; k_idx < Nk; ++k_idx) {
                values[(i * Nt + j) * Nk + k_idx] = tensor.view[i, j, k_idx];
            }
        }
    }

    // Step 8: Fit 3D B-spline
    std::array<std::vector<double>, 3> grids = {
        axes.log_moneyness,
        axes.tau_prime,
        axes.ln_kappa
    };

    auto fitter_result = BSplineNDSeparable<double, 3>::create(grids);
    if (!fitter_result.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::FittingFailed});
    }

    auto fit_result = fitter_result->fit(std::move(values));
    if (!fit_result.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::FittingFailed});
    }

    // Step 9: Build surface
    PriceTableAxesND<3> surface_axes;
    surface_axes.grids[0] = axes.log_moneyness;
    surface_axes.grids[1] = axes.tau_prime;
    surface_axes.grids[2] = axes.ln_kappa;
    surface_axes.names = {"log_moneyness", "tau_prime", "ln_kappa"};

    auto surface = PriceTableSurfaceND<3>::build(
        std::move(surface_axes), std::move(fit_result->coefficients),
        1.0 /*K_ref=1.0 for dimensionless*/);

    if (!surface.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::SurfaceBuildFailed});
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    return DimensionlessBuildResult{
        .surface = std::move(surface.value()),
        .n_pde_solves = n_pde_solves,
        .build_time_seconds = elapsed
    };
}

}  // namespace mango
