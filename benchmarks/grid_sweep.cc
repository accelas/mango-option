// SPDX-License-Identifier: MIT
// Grid density sweep: find where finer interpolation grid stops improving accuracy.
//
// Usage: bazel run -c opt //benchmarks:grid_sweep
//
// Sweeps PriceTableGridAccuracyParams.target_iv_error from coarse to fine,
// measuring interpolated IV accuracy against FDM ground truth.
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/price_table_grid_estimator.hpp"
#include "src/option/table/american_price_surface.hpp"
#include "src/option/iv_solver_fdm.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include <cstdio>
#include <chrono>
#include <cmath>
#include <vector>

using namespace mango;

int main() {
    // SPY-like option chain parameters
    const double spot = 680.0;
    const double rate = 0.04;
    const double div_yield = 0.011;

    // Representative strikes: 85%-115% of spot
    std::vector<double> strikes;
    for (double k = 0.85; k <= 1.15; k += 0.025) {
        strikes.push_back(spot * k);
    }

    std::vector<double> maturities = {0.02, 0.05, 0.1, 0.25, 0.5, 1.0};
    std::vector<double> vols = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50};
    std::vector<double> rates = {0.02, 0.04, 0.06};

    // FDM solver as ground truth (High accuracy)
    IVSolverFDMConfig fdm_config;
    fdm_config.root_config.max_iter = 100;
    fdm_config.root_config.tolerance = 1e-8;
    fdm_config.grid = grid_accuracy_profile(GridAccuracyProfile::High);
    IVSolverFDM fdm_solver(fdm_config);

    // Test queries: puts at various moneyness × maturity
    struct Query {
        double strike;
        double maturity;
        double vol_true;
    };
    std::vector<Query> test_cases;
    for (double tau : {0.1, 0.25, 1.0}) {
        for (double moneyness_ratio : {0.95, 1.0, 1.05}) {
            double K = spot / moneyness_ratio;
            test_cases.push_back({K, tau, 0.20});
        }
    }

    // Price options with High PDE accuracy to get market prices
    fprintf(stderr, "Computing reference prices...\n");
    std::vector<IVQuery> iv_queries;
    for (auto& tc : test_cases) {
        PricingParams params(spot, tc.strike, tc.maturity, rate, div_yield,
                                     OptionType::PUT, tc.vol_true);
        auto [gs, td] = estimate_grid_for_option(params, grid_accuracy_profile(GridAccuracyProfile::High));
        std::pmr::synchronized_pool_resource pool;
        std::pmr::vector<double> buf(PDEWorkspace::required_size(gs.n_points()), &pool);
        auto ws = PDEWorkspace::from_buffer(buf, gs.n_points()).value();
        auto custom = std::make_pair(gs, td);
        AmericanOptionSolver solver(params, ws, std::nullopt, custom);
        auto result = solver.solve();
        if (!result) continue;
        double price = result->value_at(spot);
        iv_queries.push_back(IVQuery(spot, tc.strike, tc.maturity, rate, div_yield,
                                      OptionType::PUT, price));
    }

    // Reference FDM IVs
    fprintf(stderr, "Computing reference IVs...\n");
    std::vector<double> ref_ivs;
    for (auto& q : iv_queries) {
        auto r = fdm_solver.solve_impl(q);
        ref_ivs.push_back(r.has_value() ? r->implied_vol : -1.0);
    }

    // Header
    printf("%-6s %-18s %-10s %-10s %-10s %-10s %-12s\n",
           "Trial", "Grid (mxTxσxr)", "PDE solves", "Build(ms)", "Max(bps)", "Avg(bps)", "Interp(us)");
    printf("%-6s %-18s %-10s %-10s %-10s %-10s %-12s\n",
           "-----", "------------------", "----------", "----------", "----------", "----------", "------------");
    fflush(stdout);

    // Sweep target_iv_error from coarse to fine
    std::vector<double> targets = {0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000007};

    for (size_t trial = 0; trial < targets.size(); trial++) {
        fprintf(stderr, "Trial %zu/%zu (target=%.1f bps)...\n", trial, targets.size(), targets[trial] * 10000.0);

        PriceTableGridAccuracyParams<4> grid_params;
        grid_params.target_iv_error = targets[trial];
        grid_params.min_points = 4;
        grid_params.max_points = 70;
        grid_params.curvature_weights = {1.0, 1.0, 2.5, 0.6};
        grid_params.scale_factor = 2.0;

        auto estimate = estimate_grid_from_chain_bounds(
            strikes, spot, maturities, vols, rates, grid_params);

        if (estimate.grids[0].empty()) {
            printf("%-6zu EMPTY GRID\n", trial);
            continue;
        }

        // Build price table (Medium PDE accuracy — sufficient for table nodes)
        auto pde_accuracy = grid_accuracy_profile(GridAccuracyProfile::Medium);
        auto builder_result = PriceTableBuilder<4>::from_vectors(
            estimate.grids[0], estimate.grids[1], estimate.grids[2], estimate.grids[3],
            spot, pde_accuracy, OptionType::PUT, div_yield);

        if (!builder_result) {
            printf("%-6zu BUILD FAILED\n", trial);
            continue;
        }

        auto& [builder, axes] = builder_result.value();
        auto t0 = std::chrono::steady_clock::now();
        auto table_result = builder.build(axes);
        auto t1 = std::chrono::steady_clock::now();
        double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!table_result) {
            printf("%-6zu TABLE BUILD FAILED\n", trial);
            continue;
        }

        auto aps = AmericanPriceSurface::create(table_result->surface, OptionType::PUT);
        if (!aps) {
            printf("%-6zu APS CREATE FAILED\n", trial);
            continue;
        }
        auto iv_solver_result = IVSolverInterpolatedStandard::create(std::move(*aps));
        if (!iv_solver_result) {
            printf("%-6zu IV SOLVER FAILED\n", trial);
            continue;
        }
        const auto& interp_solver = iv_solver_result.value();

        // Measure accuracy
        double max_err = 0.0, sum_err = 0.0;
        size_t valid = 0;
        bool is_last = (trial == targets.size() - 1);
        for (size_t i = 0; i < iv_queries.size(); i++) {
            if (ref_ivs[i] < 0) continue;
            auto interp_r = interp_solver.solve_impl(iv_queries[i]);
            if (!interp_r) continue;
            double err = std::abs(ref_ivs[i] - interp_r->implied_vol);
            max_err = std::max(max_err, err);
            sum_err += err;
            valid++;
        }
        double avg_err = valid > 0 ? sum_err / valid : 0.0;

        // Print per-query detail for finest grid
        if (is_last) {
            printf("\n  Per-query errors (finest grid):\n");
            printf("  %-8s %-8s %-8s %-10s %-10s\n", "m", "T", "ref_iv", "interp_iv", "err(bps)");
            for (size_t i = 0; i < iv_queries.size(); i++) {
                if (ref_ivs[i] < 0) continue;
                auto interp_r = interp_solver.solve_impl(iv_queries[i]);
                if (!interp_r) continue;
                double err = std::abs(ref_ivs[i] - interp_r->implied_vol);
                printf("  %-8.3f %-8.3f %-8.4f %-10.4f %-10.1f\n",
                    spot / iv_queries[i].strike, iv_queries[i].maturity,
                    ref_ivs[i], interp_r->implied_vol, err * 10000.0);
            }
            printf("\n");
        }

        // Measure interp speed
        auto t2 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 100; rep++) {
            for (auto& q : iv_queries) {
                auto r = interp_solver.solve_impl(q);
                asm volatile("" : : "r"(&r) : "memory");
            }
        }
        auto t3 = std::chrono::steady_clock::now();
        double interp_us = std::chrono::duration<double, std::micro>(t3 - t2).count()
                           / (100.0 * iv_queries.size());

        char grid_str[64];
        snprintf(grid_str, sizeof(grid_str), "%zux%zux%zux%zu",
                 axes.grids[0].size(), axes.grids[1].size(),
                 axes.grids[2].size(), axes.grids[3].size());

        printf("%-6zu %-18s %-10zu %-10.0f %-10.1f %-10.1f %-12.2f\n",
               trial, grid_str,
               table_result->n_pde_solves,
               build_ms,
               max_err * 10000.0,
               avg_err * 10000.0,
               interp_us);
        fflush(stdout);
    }

    return 0;
}
