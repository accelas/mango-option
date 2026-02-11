// SPDX-License-Identifier: MIT
/**
 * @file chebyshev_segmented_comparison.cc
 * @brief Compare old (type-erased) vs new (typed SplitSurface) Chebyshev segmented paths
 *
 * Measures build time and query time for both paths to verify the refactor
 * does not regress performance.
 *
 * Run with: bazel run //benchmarks:chebyshev_segmented_comparison
 */

#include "iv_benchmark_common.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace mango;
using namespace mango::bench;

// ============================================================================
// Shared config
// ============================================================================

static SegmentedAdaptiveConfig make_seg_config() {
    return {
        .spot = kSpot,
        .option_type = OptionType::PUT,
        .dividend_yield = kDivYield,
        .discrete_dividends = make_div_schedule(1.0),
        .maturity = 1.0,
        .kref_config = {.K_refs = {90.0, 100.0, 110.0}},
    };
}

static IVGrid make_log_domain() {
    return {
        .moneyness = {std::log(kSpot / 120.0), std::log(kSpot / 110.0),
                      std::log(kSpot / 100.0), std::log(kSpot / 90.0),
                      std::log(kSpot / 80.0)},
        .vol = {0.10, 0.15, 0.20, 0.30, 0.50},
        .rate = {0.03, 0.05, 0.07},
    };
}

// ============================================================================
// Build time comparison
// ============================================================================

static void compare_build_time() {
    std::printf("\n================================================================\n");
    std::printf("BUILD TIME COMPARISON\n");
    std::printf("================================================================\n\n");

    AdaptiveGridParams params;
    params.target_iv_error = 5e-4;
    params.max_iter = 3;

    auto seg_config = make_seg_config();
    auto domain = make_log_domain();

    // --- Old path (type-erased) ---
    {
        auto t0 = std::chrono::steady_clock::now();
        auto result = build_adaptive_chebyshev_segmented(params, seg_config, domain);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (result.has_value()) {
            std::printf("OLD (type-erased):  %8.1f ms  iters=%zu  pde=%zu  met=%s\n",
                        ms, result->iterations.size(), result->total_pde_solves,
                        result->target_met ? "yes" : "no");
            for (const auto& it : result->iterations) {
                std::printf("  iter %zu: grid [%zu, %zu, %zu, %zu] "
                            "max=%.1f bps avg=%.1f bps\n",
                            it.iteration,
                            it.grid_sizes[0], it.grid_sizes[1],
                            it.grid_sizes[2], it.grid_sizes[3],
                            it.max_error * 1e4, it.avg_error * 1e4);
            }
        } else {
            std::printf("OLD (type-erased):  FAILED\n");
        }
    }

    // --- New path (typed SplitSurface) ---
    {
        auto t0 = std::chrono::steady_clock::now();
        auto result = build_adaptive_chebyshev_segmented_typed(params, seg_config, domain);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (result.has_value()) {
            std::printf("NEW (typed split):  %8.1f ms  iters=%zu  pde=%zu  met=%s\n",
                        ms, result->iterations.size(), result->total_pde_solves,
                        result->target_met ? "yes" : "no");
            for (const auto& it : result->iterations) {
                std::printf("  iter %zu: grid [%zu, %zu, %zu, %zu] "
                            "max=%.1f bps avg=%.1f bps\n",
                            it.iteration,
                            it.grid_sizes[0], it.grid_sizes[1],
                            it.grid_sizes[2], it.grid_sizes[3],
                            it.max_error * 1e4, it.avg_error * 1e4);
            }
        } else {
            std::printf("NEW (typed split):  FAILED\n");
        }
    }
}

// ============================================================================
// Query time comparison
// ============================================================================

static void compare_query_time() {
    std::printf("\n================================================================\n");
    std::printf("QUERY TIME COMPARISON (IV solve)\n");
    std::printf("================================================================\n\n");

    // Build both solvers via factory
    auto seg_config = make_seg_config();
    auto domain = make_log_domain();
    AdaptiveGridParams params;
    params.target_iv_error = 5e-4;
    params.max_iter = 3;

    // Build old solver
    std::printf("Building old solver...\n");
    auto old_result = build_adaptive_chebyshev_segmented(params, seg_config, domain);
    if (!old_result.has_value()) {
        std::printf("Old build failed\n");
        return;
    }

    // Build new solver via factory
    std::printf("Building new solver...\n");
    IVSolverFactoryConfig factory_config{
        .option_type = OptionType::PUT,
        .spot = kSpot,
        .dividend_yield = kDivYield,
        .grid = {
            .moneyness = {0.80, 0.90, 1.00, 1.10, 1.20},
            .vol = {0.10, 0.15, 0.20, 0.30, 0.50},
            .rate = {0.03, 0.05, 0.07},
        },
        .adaptive = params,
        .backend = ChebyshevBackend{},
        .discrete_dividends = DiscreteDividendConfig{
            .maturity = 1.0,
            .discrete_dividends = make_div_schedule(1.0),
            .kref_config = {.K_refs = {90.0, 100.0, 110.0}},
        },
    };

    auto new_solver = make_interpolated_iv_solver(factory_config);
    if (!new_solver.has_value()) {
        std::printf("New solver build failed\n");
        return;
    }

    // Generate IV query set
    std::vector<IVQuery> queries;
    for (double K : {85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0}) {
        for (double tau : {0.1, 0.3, 0.5, 0.7, 0.9}) {
            for (double sigma : {0.15, 0.25, 0.40}) {
                // Compute a reference price from old surface for the query
                double price = old_result->price_fn(kSpot, K, tau, sigma, kRate);
                if (price > 0.01 && std::isfinite(price)) {
                    OptionSpec spec{
                        .spot = kSpot, .strike = K, .maturity = tau,
                        .rate = kRate, .dividend_yield = kDivYield,
                        .option_type = OptionType::PUT};
                    queries.emplace_back(spec, price);
                }
            }
        }
    }
    std::printf("Query set: %zu IV solves\n\n", queries.size());

    // --- Old path: Brent IV (no vega, just price_fn) ---
    {
        size_t solved = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (const auto& q : queries) {
            double iv = brent_solve_iv(
                [&](double vol) {
                    return old_result->price_fn(q.spot, q.strike, q.maturity, vol, kRate);
                },
                q.market_price);
            if (std::isfinite(iv)) ++solved;
        }
        auto t1 = std::chrono::steady_clock::now();
        double us_total = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::printf("OLD (Brent, type-erased price_fn):\n");
        std::printf("  total: %8.1f us  per-query: %6.2f us  solved: %zu/%zu\n",
                    us_total, us_total / queries.size(), solved, queries.size());
    }

    // --- New path: Newton IV (typed surface with vega) ---
    {
        size_t solved = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (const auto& q : queries) {
            auto result = new_solver->solve(q);
            if (result.has_value()) ++solved;
        }
        auto t1 = std::chrono::steady_clock::now();
        double us_total = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::printf("NEW (Newton, typed SplitSurface):\n");
        std::printf("  total: %8.1f us  per-query: %6.2f us  solved: %zu/%zu\n",
                    us_total, us_total / queries.size(), solved, queries.size());
    }

    // --- New path: batch ---
    {
        auto t0 = std::chrono::steady_clock::now();
        auto batch_result = new_solver->solve_batch(queries);
        auto t1 = std::chrono::steady_clock::now();
        double us_total = std::chrono::duration<double, std::micro>(t1 - t0).count();
        size_t solved = queries.size() - batch_result.failed_count;
        std::printf("NEW (Newton batch):\n");
        std::printf("  total: %8.1f us  per-query: %6.2f us  solved: %zu/%zu\n",
                    us_total, us_total / queries.size(), solved, queries.size());
    }
}

// ============================================================================
// Raw price/vega query time (no IV solve, just surface evaluation)
// ============================================================================

static void compare_raw_query_time() {
    std::printf("\n================================================================\n");
    std::printf("RAW QUERY TIME (price + vega evaluation, no IV solve)\n");
    std::printf("================================================================\n\n");

    auto seg_config = make_seg_config();
    auto domain = make_log_domain();
    AdaptiveGridParams params;
    params.target_iv_error = 5e-4;
    params.max_iter = 3;

    // Build both
    auto old_result = build_adaptive_chebyshev_segmented(params, seg_config, domain);
    auto new_result = build_adaptive_chebyshev_segmented_typed(params, seg_config, domain);

    if (!old_result.has_value() || !new_result.has_value()) {
        std::printf("Build failed\n");
        return;
    }

    // Query points
    struct QueryPt { double spot, strike, tau, sigma, rate; };
    std::vector<QueryPt> pts;
    for (double K : {85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0})
        for (double tau : {0.1, 0.3, 0.5, 0.7, 0.9})
            for (double sigma : {0.15, 0.25, 0.40})
                pts.push_back({kSpot, K, tau, sigma, kRate});

    constexpr int kReps = 1000;
    std::printf("Query points: %zu  x  %d reps\n\n", pts.size(), kReps);

    // --- Old: price only ---
    {
        volatile double sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < kReps; ++r)
            for (const auto& p : pts)
                sink = old_result->price_fn(p.spot, p.strike, p.tau, p.sigma, p.rate);
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count()
                    / (kReps * pts.size());
        std::printf("OLD price_fn:        %6.0f ns/query\n", ns);
        (void)sink;
    }

    // --- New: price ---
    {
        volatile double sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < kReps; ++r)
            for (const auto& p : pts)
                sink = new_result->surface.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count()
                    / (kReps * pts.size());
        std::printf("NEW surface.price(): %6.0f ns/query\n", ns);
        (void)sink;
    }

    // --- New: vega ---
    {
        volatile double sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < kReps; ++r)
            for (const auto& p : pts)
                sink = new_result->surface.vega(p.spot, p.strike, p.tau, p.sigma, p.rate);
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count()
                    / (kReps * pts.size());
        std::printf("NEW surface.vega():  %6.0f ns/query\n", ns);
        (void)sink;
    }
}

// ============================================================================
// Accuracy comparison: both paths vs FDM reference
// ============================================================================

static void compare_accuracy() {
    std::printf("\n================================================================\n");
    std::printf("ACCURACY: IV error vs FDM reference (bps)\n");
    std::printf("================================================================\n\n");

    auto seg_config = make_seg_config();
    auto domain = make_log_domain();
    AdaptiveGridParams params;
    params.target_iv_error = 5e-4;
    params.max_iter = 3;

    auto old_result = build_adaptive_chebyshev_segmented(params, seg_config, domain);
    auto new_result = build_adaptive_chebyshev_segmented_typed(params, seg_config, domain);

    if (!old_result.has_value() || !new_result.has_value()) {
        std::printf("Build failed\n");
        return;
    }

    auto divs = make_div_schedule(1.0);

    std::printf("  %6s %5s %5s | %10s %10s %10s | %8s %8s\n",
                "K", "tau", "sigma", "FDM_price", "OLD_price", "NEW_price",
                "OLD_bps", "NEW_bps");
    std::printf("  %s\n", std::string(80, '-').c_str());

    double old_max_bps = 0, new_max_bps = 0;
    double old_sum_bps = 0, new_sum_bps = 0;
    size_t count = 0;

    for (double K : {85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0}) {
        for (double tau : {0.1, 0.3, 0.7, 0.9}) {
            for (double sigma : {0.15, 0.25, 0.40}) {
                // FDM reference
                PricingParams pp;
                pp.spot = kSpot; pp.strike = K; pp.maturity = tau;
                pp.rate = kRate; pp.dividend_yield = kDivYield;
                pp.option_type = OptionType::PUT; pp.volatility = sigma;
                pp.discrete_dividends = divs;
                // Filter dividends to those in the future
                std::vector<Dividend> future_divs;
                double surface_maturity = 1.0;
                double cal_now = surface_maturity - tau;
                for (const auto& d : divs) {
                    if (d.calendar_time > cal_now)
                        future_divs.push_back(
                            Dividend{.calendar_time = d.calendar_time - cal_now,
                                     .amount = d.amount});
                }
                pp.discrete_dividends = future_divs;
                pp.maturity = tau;

                auto fdm = solve_american_option(pp);
                if (!fdm.has_value()) continue;
                double ref_price = fdm->value();
                if (ref_price < 0.01) continue;

                // Old surface price
                double old_price = old_result->price_fn(kSpot, K, tau, sigma, kRate);

                // New surface price
                double new_price = new_result->surface.price(kSpot, K, tau, sigma, kRate);

                // Recover IV from both via Brent, compare to FDM IV
                double fdm_iv = brent_solve_iv(
                    [&](double vol) -> double {
                        PricingParams p2 = pp;
                        p2.volatility = vol;
                        auto r = solve_american_option(p2);
                        return r.has_value() ? r->value() : std::nan("");
                    },
                    ref_price);

                double old_iv = brent_solve_iv(
                    [&](double vol) {
                        return old_result->price_fn(kSpot, K, tau, vol, kRate);
                    },
                    ref_price);

                double new_iv = brent_solve_iv(
                    [&](double vol) {
                        return new_result->surface.price(kSpot, K, tau, vol, kRate);
                    },
                    ref_price);

                if (!std::isfinite(fdm_iv) || !std::isfinite(old_iv) || !std::isfinite(new_iv))
                    continue;

                double old_err_bps = std::abs(old_iv - fdm_iv) * 1e4;
                double new_err_bps = std::abs(new_iv - fdm_iv) * 1e4;

                std::printf("  %6.0f %5.1f %5.2f | %10.4f %10.4f %10.4f | %8.1f %8.1f\n",
                            K, tau, sigma, ref_price, old_price, new_price,
                            old_err_bps, new_err_bps);

                old_max_bps = std::max(old_max_bps, old_err_bps);
                new_max_bps = std::max(new_max_bps, new_err_bps);
                old_sum_bps += old_err_bps;
                new_sum_bps += new_err_bps;
                ++count;
            }
        }
    }

    std::printf("  %s\n", std::string(80, '-').c_str());
    std::printf("  N=%zu  OLD: max=%.1f bps  avg=%.1f bps  |  NEW: max=%.1f bps  avg=%.1f bps\n",
                count, old_max_bps, old_sum_bps / count,
                new_max_bps, new_sum_bps / count);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("Chebyshev Segmented: Old (type-erased) vs New (typed SplitSurface)\n");
    std::printf("Spot=%.0f  Rate=%.2f  DivYield=%.2f  3 quarterly $0.50 dividends\n",
                kSpot, kRate, kDivYield);

    compare_build_time();
    compare_accuracy();
    compare_raw_query_time();
    compare_query_time();

    return 0;
}
