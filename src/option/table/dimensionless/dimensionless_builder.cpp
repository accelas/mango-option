// SPDX-License-Identifier: MIT

#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include <cmath>
#include <chrono>

namespace mango {

std::expected<DimensionlessPDEResult, PriceTableError>
solve_dimensionless_pde(
    const DimensionlessAxes& axes,
    double K_ref,
    OptionType option_type)
{
    auto start = std::chrono::steady_clock::now();

    const size_t Nm = axes.log_moneyness.size();
    const size_t Nt = axes.tau_prime.size();
    const size_t Nk = axes.ln_kappa.size();

    if (Nm < 2 || Nt < 2 || Nk < 2) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InsufficientGridPoints, 0, std::min({Nm, Nt, Nk})});
    }

    const size_t total = Nm * Nt * Nk;
    std::vector<double> values(total, 0.0);

    const double sigma_eff = std::sqrt(2.0);
    const double pde_maturity = axes.tau_prime.back() * 1.01;
    int n_pde_solves = 0;

    for (size_t k = 0; k < Nk; ++k) {
        const double kappa = std::exp(axes.ln_kappa[k]);

        PricingParams params(
            OptionSpec{
                .spot = K_ref,
                .strike = K_ref,
                .maturity = pde_maturity,
                .rate = kappa,
                .dividend_yield = 0.0,
                .option_type = option_type},
            sigma_eff);

        BatchAmericanOptionSolver batch_solver;
        batch_solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
        batch_solver.set_snapshot_times(
            std::span<const double>{axes.tau_prime.data(), axes.tau_prime.size()});

        std::vector<PricingParams> batch = {params};
        auto batch_result = batch_solver.solve_batch(batch, true);
        ++n_pde_solves;

        if (batch_result.failed_count > 0 || !batch_result.results[0].has_value()) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::ExtractionFailed, 2, k});
        }

        const auto& result = batch_result.results[0].value();
        auto grid = result.grid();
        auto pde_x = grid->x();

        CubicSpline<double> spline;

        for (size_t j = 0; j < Nt; ++j) {
            auto solution = result.at_time(j);

            auto err = spline.build(pde_x, solution);
            if (err.has_value()) {
                return std::unexpected(PriceTableError{
                    PriceTableErrorCode::ExtractionFailed, 1, j});
            }

            for (size_t i = 0; i < Nm; ++i) {
                double pde_val = spline.eval(axes.log_moneyness[i]);
                values[(i * Nt + j) * Nk + k] = pde_val;
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    return DimensionlessPDEResult{
        .values = std::move(values),
        .n_pde_solves = n_pde_solves,
        .build_time_seconds = std::chrono::duration<double>(end - start).count(),
    };
}

}  // namespace mango
