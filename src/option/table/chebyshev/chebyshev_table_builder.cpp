// SPDX-License-Identifier: MIT
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"

#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/european_option.hpp"

#include <chrono>
#include <cmath>

namespace mango {

std::expected<ChebyshevTableResult, PriceTableError>
build_chebyshev_table(const ChebyshevTableConfig& config) {
    auto t0 = std::chrono::steady_clock::now();

    const size_t n_m     = config.num_pts[0];
    const size_t n_tau   = config.num_pts[1];
    const size_t n_sigma = config.num_pts[2];
    const size_t n_rate  = config.num_pts[3];

    // Generate CGL nodes per axis
    auto m_nodes     = chebyshev_nodes(n_m,     config.domain.lo[0], config.domain.hi[0]);
    auto tau_nodes   = chebyshev_nodes(n_tau,   config.domain.lo[1], config.domain.hi[1]);
    auto sigma_nodes = chebyshev_nodes(n_sigma, config.domain.lo[2], config.domain.hi[2]);
    auto rate_nodes  = chebyshev_nodes(n_rate,  config.domain.lo[3], config.domain.hi[3]);

    // Build batch: one PricingParams per (sigma, rate) pair
    std::vector<PricingParams> batch;
    batch.reserve(n_sigma * n_rate);
    for (size_t si = 0; si < n_sigma; ++si) {
        for (size_t ri = 0; ri < n_rate; ++ri) {
            batch.emplace_back(
                OptionSpec{
                    .spot = config.K_ref,
                    .strike = config.K_ref,
                    .maturity = config.domain.hi[1],  // max tau for full time domain
                    .rate = rate_nodes[ri],
                    .dividend_yield = config.dividend_yield,
                    .option_type = config.option_type,
                },
                sigma_nodes[si]);
        }
    }

    // Solve batch with snapshots at tau CGL nodes
    BatchAmericanOptionSolver solver;
    solver.set_snapshot_times(std::span<const double>(tau_nodes));
    auto batch_result = solver.solve_batch(
        std::span<const PricingParams>(batch), /*use_shared_grid=*/true);

    size_t n_pde_solves = batch.size() - batch_result.failed_count;

    // Extract EEP values at all CGL nodes
    // Tensor layout (row-major): [m, tau, sigma, rate]
    size_t total = n_m * n_tau * n_sigma * n_rate;
    std::vector<double> eep_values(total);

    for (size_t si = 0; si < n_sigma; ++si) {
        for (size_t ri = 0; ri < n_rate; ++ri) {
            size_t batch_idx = si * n_rate + ri;
            const auto& res = batch_result.results[batch_idx];

            if (!res.has_value()) {
                return std::unexpected(PriceTableError{
                    PriceTableErrorCode::ExtractionFailed,
                    batch_idx, 0});
            }

            auto grid = res->grid();
            auto x_grid = grid->x();  // spatial grid in log-moneyness

            for (size_t ti = 0; ti < n_tau; ++ti) {
                auto solution = res->at_time(ti);

                CubicSpline<double> spline;
                auto build_error = spline.build(x_grid, solution);
                if (build_error.has_value()) {
                    return std::unexpected(PriceTableError{
                        PriceTableErrorCode::ExtractionFailed,
                        batch_idx, ti});
                }

                for (size_t mi = 0; mi < n_m; ++mi) {
                    double m = m_nodes[mi];
                    double spot_node = config.K_ref * std::exp(m);
                    double tau = tau_nodes[ti];
                    double sigma = sigma_nodes[si];
                    double rate = rate_nodes[ri];

                    // American price in dollars (spline returns V/K_ref)
                    double am_price = spline.eval(m) * config.K_ref;

                    // European price in dollars
                    EuropeanOptionSolver eu_solver(
                        OptionSpec{
                            .spot = spot_node,
                            .strike = config.K_ref,
                            .maturity = tau,
                            .rate = rate,
                            .dividend_yield = config.dividend_yield,
                            .option_type = config.option_type},
                        sigma);
                    auto eu = eu_solver.solve();

                    double eep_raw = 0.0;
                    if (eu.has_value()) {
                        eep_raw = am_price - eu->value();
                    }

                    // Debiased softplus floor (matches EEPDecomposer)
                    constexpr double kSharpness = 100.0;
                    double eep;
                    if (kSharpness * eep_raw > 500.0) {
                        eep = eep_raw;
                    } else {
                        double sp = std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
                        double bias = std::log(2.0) / kSharpness;
                        eep = std::max(0.0, sp - bias);
                    }

                    size_t flat = mi * (n_tau * n_sigma * n_rate)
                                + ti * (n_sigma * n_rate)
                                + si * n_rate
                                + ri;
                    eep_values[flat] = eep;
                }
            }
        }
    }

    // Build Chebyshev interpolant from EEP values
    auto interp = ChebyshevInterpolant<4, TuckerTensor<4>>::build_from_values(
        std::span<const double>(eep_values),
        config.domain, config.num_pts, config.tucker_epsilon);

    // Wrap in EEPSurfaceAdapter + PriceTable
    ChebyshevLeaf leaf(
        std::move(interp),
        StandardTransform4D{},
        AnalyticalEEP(config.option_type, config.dividend_yield),
        config.K_ref);

    SurfaceBounds bounds{
        .m_min = config.domain.lo[0], .m_max = config.domain.hi[0],
        .tau_min = config.domain.lo[1], .tau_max = config.domain.hi[1],
        .sigma_min = config.domain.lo[2], .sigma_max = config.domain.hi[2],
        .rate_min = config.domain.lo[3], .rate_max = config.domain.hi[3],
    };

    ChebyshevSurface surface(
        std::move(leaf), bounds, config.option_type, config.dividend_yield);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    return ChebyshevTableResult{
        .surface = std::move(surface),
        .n_pde_solves = n_pde_solves,
        .build_seconds = elapsed,
    };
}

}  // namespace mango
