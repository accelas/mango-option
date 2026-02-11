// SPDX-License-Identifier: MIT
/**
 * @file interp_iv_safety.cc
 * @brief Interpolation IV safety diagnostic
 *
 * Maps interpolation IV error across the moneyness × maturity space
 * to establish where interpolated IV is safe to use.
 *
 * Workflow mirrors production use:
 *  1. Generate reference prices via BatchAmericanOptionSolver (chain mode)
 *  2. Build InterpolatedIVSolver via factory with adaptive IVGrid
 *  3. Recover IV via interpolated solver and FDM IVSolver
 *  4. Error = |interp_iv − fdm_iv| in basis points
 *
 * Run with: bazel run //benchmarks:interp_iv_safety
 */

#include "iv_benchmark_common.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

using namespace mango;
using namespace mango::bench;

// ============================================================================
// Test parameters
// ============================================================================

static constexpr std::array<double, 9> kStrikes = {
    80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0};

static constexpr std::array<double, 8> kMaturities = {
    7.0 / 365, 14.0 / 365, 30.0 / 365, 60.0 / 365,
    90.0 / 365, 180.0 / 365, 1.0, 2.0};

static constexpr std::array<double, 2> kVols = {0.15, 0.30};

static constexpr size_t kNS = kStrikes.size();
static constexpr size_t kNT = kMaturities.size();
static constexpr size_t kNV = kVols.size();

static const std::array<const char*, kNT> kMatLabels = {
    "  7d", " 14d", " 30d", " 60d", " 90d", "180d", "  1y", "  2y"};

// ============================================================================
// Step 1: Generate reference prices via batch chain solver
// ============================================================================

// prices[vol_idx][mat_idx][strike_idx]
using PriceGrid = std::array<std::array<std::array<double, kNS>, kNT>, kNV>;

static PriceGrid generate_prices(bool with_dividends) {
    PriceGrid prices{};
    BatchAmericanOptionSolver batch_solver;

    // Build all PricingParams across (vol, maturity, strike)
    // Chain solver groups by (σ, r, q, type, maturity) — one PDE per group
    std::vector<PricingParams> all_params;
    all_params.reserve(kNV * kNT * kNS);

    for (size_t vi = 0; vi < kNV; ++vi) {
        for (size_t ti = 0; ti < kNT; ++ti) {
            auto divs = with_dividends
                            ? make_div_schedule(kMaturities[ti])
                            : std::vector<Dividend>{};

            for (size_t si = 0; si < kNS; ++si) {
                PricingParams p;
                p.spot = kSpot;
                p.strike = kStrikes[si];
                p.maturity = kMaturities[ti];
                p.rate = kRate;
                p.dividend_yield = kDivYield;
                p.option_type = OptionType::PUT;
                p.volatility = kVols[vi];
                p.discrete_dividends = divs;
                all_params.push_back(std::move(p));
            }
        }
    }

    // Chain solving: use_shared_grid=true routes vanilla batches through
    // normalized chain path (one PDE per σ×T group = 16 PDEs for 144 options)
    auto result = batch_solver.solve_batch(all_params, /*use_shared_grid=*/true);

    // Map results back to 3D price grid
    size_t idx = 0;
    for (size_t vi = 0; vi < kNV; ++vi) {
        for (size_t ti = 0; ti < kNT; ++ti) {
            for (size_t si = 0; si < kNS; ++si) {
                if (result.results[idx].has_value()) {
                    prices[vi][ti][si] = result.results[idx]->value();
                } else {
                    prices[vi][ti][si] = std::nan("");
                }
                ++idx;
            }
        }
    }

    return prices;
}

// ============================================================================
// Step 2: Build interpolated IV solvers
// ============================================================================

// Vanilla: one solver covering all maturities via BSpline + adaptive grid
static AnyIVSolver build_vanilla_solver() {
    // Maturity grid for price table — deliberately offset from test maturities
    // so most test points require real interpolation
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = kSpot,
        .dividend_yield = kDivYield,
        .grid = IVGrid{
            .moneyness = {0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30},
            .vol = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50},
            .rate = {0.01, 0.03, 0.05, 0.10},
        },
        .adaptive = AdaptiveGridParams{.target_iv_error = 2e-5},  // 2 bps target
        .backend = BSplineBackend{
            .maturity_grid = {0.01, 0.03, 0.06, 0.12, 0.20,
                              0.35, 0.60, 1.0, 1.5, 2.0, 2.5},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    if (!solver.has_value()) {
        std::fprintf(stderr, "Failed to build vanilla interpolated solver\n");
        std::exit(1);
    }
    return std::move(*solver);
}

// Dividends: one solver per maturity via BSpline + discrete dividends + adaptive grid
static std::vector<std::pair<size_t, AnyIVSolver>> build_div_solvers() {
    std::vector<std::pair<size_t, AnyIVSolver>> solvers;

    for (size_t ti = 0; ti < kNT; ++ti) {
        double mat = kMaturities[ti];
        auto divs = make_div_schedule(mat);

        IVSolverFactoryConfig config{
            .option_type = OptionType::PUT,
            .spot = kSpot,
            .dividend_yield = kDivYield,
            .grid = IVGrid{
                .moneyness = {0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30},
                .vol = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50},
                .rate = {0.01, 0.03, 0.05, 0.10},
            },
            .adaptive = AdaptiveGridParams{.target_iv_error = 2e-5},
            .backend = BSplineBackend{},
            .discrete_dividends = DiscreteDividendConfig{
                .maturity = mat,
                .discrete_dividends = divs,
                .kref_config = {.K_refs = std::vector<double>(kStrikes.begin(), kStrikes.end())},
            },
        };

        auto solver = make_interpolated_iv_solver(config);
        if (!solver.has_value()) {
            std::fprintf(stderr, "  [skip] T=%s — solver build failed\n",
                         kMatLabels[ti]);
            continue;
        }
        solvers.emplace_back(ti, std::move(*solver));
    }
    return solvers;
}

// ============================================================================
// Step 3: FDM reference IV
// ============================================================================

// Manual Brent for dividend IV (IVSolver doesn't support discrete dividends)
static double solve_fdm_iv_div(double strike, double maturity,
                                double market_price,
                                const std::vector<Dividend>& divs) {
    return brent_solve_iv(
        [&](double vol) -> double {
            PricingParams p;
            p.spot = kSpot;
            p.strike = strike;
            p.maturity = maturity;
            p.rate = kRate;
            p.dividend_yield = kDivYield;
            p.option_type = OptionType::PUT;
            p.volatility = vol;
            p.discrete_dividends = divs;

            auto result = solve_american_option(p);
            if (!result.has_value()) return std::nan("");
            return result->value();
        },
        market_price);
}

// ============================================================================
// Step 4: Compute error grids
// ============================================================================

// errors[mat_idx][strike_idx] in bps (NaN for failed cases)
using ErrorTable = std::array<std::array<double, kNS>, kNT>;

static ErrorTable compute_errors_vanilla(const PriceGrid& prices,
                                          const AnyIVSolver& interp_solver,
                                          size_t vol_idx) {
    ErrorTable errors{};
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    // Build all IVQueries for batch FDM solving
    std::vector<IVQuery> queries;
    std::vector<std::pair<size_t, size_t>> query_map;  // (mat_idx, strike_idx)
    queries.reserve(kNT * kNS);

    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < kNS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) {
                errors[ti][si] = std::nan("");
                continue;
            }

            IVQuery q;
            q.spot = kSpot;
            q.strike = kStrikes[si];
            q.maturity = kMaturities[ti];
            q.rate = kRate;
            q.dividend_yield = kDivYield;
            q.option_type = OptionType::PUT;
            q.market_price = price;
            queries.push_back(q);
            query_map.emplace_back(ti, si);
        }
    }

    // Batch FDM IV (parallelized via OpenMP)
    auto fdm_results = fdm_solver.solve_batch(queries);

    // Batch interpolated IV (parallelized via OpenMP)
    auto interp_results = interp_solver.solve_batch(queries);

    // Compute errors
    for (size_t i = 0; i < queries.size(); ++i) {
        auto [ti, si] = query_map[i];

        if (!fdm_results.results[i].has_value() ||
            !interp_results.results[i].has_value()) {
            errors[ti][si] = std::nan("");
            continue;
        }

        double fdm_iv = fdm_results.results[i]->implied_vol;
        double interp_iv = interp_results.results[i]->implied_vol;
        errors[ti][si] = std::abs(interp_iv - fdm_iv) * 10000.0;
    }

    return errors;
}

static ErrorTable compute_errors_div(
    const PriceGrid& prices,
    const std::vector<std::pair<size_t, AnyIVSolver>>& div_solvers,
    size_t vol_idx) {
    ErrorTable errors{};

    // Initialize all to NaN
    for (auto& row : errors)
        for (auto& v : row)
            v = std::nan("");

    // Build lookup: mat_idx → solver index
    std::array<int, kNT> solver_idx{};
    solver_idx.fill(-1);
    for (size_t i = 0; i < div_solvers.size(); ++i) {
        solver_idx[div_solvers[i].first] = static_cast<int>(i);
    }

    for (size_t ti = 0; ti < kNT; ++ti) {
        if (solver_idx[ti] < 0) continue;  // no solver for this maturity

        double maturity = kMaturities[ti];
        auto divs = make_div_schedule(maturity);
        const auto& solver = div_solvers[static_cast<size_t>(solver_idx[ti])].second;

        // Build queries for this maturity
        std::vector<IVQuery> queries;
        std::vector<size_t> strike_indices;

        for (size_t si = 0; si < kNS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) continue;

            IVQuery q;
            q.spot = kSpot;
            q.strike = kStrikes[si];
            q.maturity = maturity;
            q.rate = kRate;
            q.dividend_yield = kDivYield;
            q.option_type = OptionType::PUT;
            q.market_price = price;
            queries.push_back(q);
            strike_indices.push_back(si);
        }

        // Batch interpolated IV
        auto interp_results = solver.solve_batch(queries);

        for (size_t i = 0; i < queries.size(); ++i) {
            size_t si = strike_indices[i];

            if (!interp_results.results[i].has_value()) continue;

            // FDM reference: manual Brent with dividends
            double fdm_iv = solve_fdm_iv_div(
                kStrikes[si], maturity,
                prices[vol_idx][ti][si], divs);

            if (std::isnan(fdm_iv)) continue;

            double interp_iv = interp_results.results[i]->implied_vol;
            errors[ti][si] = std::abs(interp_iv - fdm_iv) * 10000.0;
        }
    }

    return errors;
}

// ============================================================================
// Step 5: Print heatmap
// ============================================================================

static void print_heatmap(const char* title, const ErrorTable& errors) {
    std::printf("\n=== %s ===\n", title);

    // Header
    std::printf("          ");
    for (size_t si = 0; si < kNS; ++si) {
        std::printf("  K=%-3.0f ", kStrikes[si]);
    }
    std::printf("\n");

    size_t n_total = 0, n_failed = 0;
    double sum_sq = 0;

    for (size_t ti = 0; ti < kNT; ++ti) {
        std::printf("  T=%s  ", kMatLabels[ti]);
        for (size_t si = 0; si < kNS; ++si) {
            double e = errors[ti][si];
            n_total++;
            if (std::isnan(e)) {
                std::printf("   ---  ");
                n_failed++;
            } else {
                const char* marker = "";
                if (e > 200) marker = "***";
                else if (e > 50) marker = "**";
                else if (e > 10) marker = "*";

                std::printf("%6.1f%-3s", e, marker);
                sum_sq += e * e;
            }
        }
        std::printf("\n");
    }

    size_t n_valid = n_total - n_failed;
    double rms = n_valid > 0 ? std::sqrt(sum_sq / n_valid) : 0;
    std::printf("\n  Legend: * >10bps  ** >50bps  *** >200bps  --- solve failed\n");
    std::printf("  Overall RMS: %.1f bps (%zu/%zu succeeded)\n", rms, n_valid, n_total);
}

// ============================================================================
// TV/K filtered stats — filter out low-vega edge cases
// ============================================================================

/// TV/K mask: which (maturity, strike) points survive a given threshold.
/// Based purely on reference prices so all algorithms share the same mask.
using TVKMask = std::array<std::array<bool, kNS>, kNT>;

static TVKMask compute_tvk_mask(const PriceGrid& prices, size_t vol_idx,
                                 double threshold) {
    TVKMask mask{};
    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < kNS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) {
                mask[ti][si] = false;
                continue;
            }
            double K = kStrikes[si];
            double intrinsic = std::max(K - kSpot, 0.0);  // put
            double tv = price - intrinsic;
            mask[ti][si] = (tv / K) >= threshold;
        }
    }
    return mask;
}

/// Print RMS error for multiple algorithms at a given TV/K threshold.
/// All algorithms are filtered by the SAME mask (from reference prices).
struct AlgoErrors {
    const char* label;
    const ErrorTable* errors;
};

static void print_tvk_comparison(const PriceGrid& prices, size_t vol_idx,
                                  std::span<const AlgoErrors> algos) {
    static constexpr double kThresholds[] = {0.0, 1e-4, 1e-3, 5e-3};
    static constexpr const char* kThreshLabels[] = {
        "none", "1e-4", "1e-3", "5e-3"};

    std::printf("\n  TV/K filtered RMS (σ=%.0f%%):\n", kVols[vol_idx] * 100);

    // Header
    std::printf("  %-20s", "TV/K >=");
    for (const auto& a : algos)
        std::printf("  %14s", a.label);
    std::printf("\n");

    for (size_t fi = 0; fi < 4; ++fi) {
        auto mask = compute_tvk_mask(prices, vol_idx, kThresholds[fi]);

        // Count cells passing the TV/K filter
        size_t mask_count = 0;
        for (size_t ti = 0; ti < kNT; ++ti)
            for (size_t si = 0; si < kNS; ++si)
                if (mask[ti][si]) mask_count++;

        std::printf("  %-12s [%2zu/%zu]", kThreshLabels[fi],
                    mask_count, kNT * kNS);
        for (const auto& a : algos) {
            double sum_sq = 0;
            size_t n = 0;
            for (size_t ti = 0; ti < kNT; ++ti) {
                for (size_t si = 0; si < kNS; ++si) {
                    if (!mask[ti][si]) continue;
                    double e = (*a.errors)[ti][si];
                    if (std::isnan(e)) continue;
                    sum_sq += e * e;
                    n++;
                }
            }
            double rms = n > 0 ? std::sqrt(sum_sq / n) : 0;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.1f (%zu)", rms, n);
            std::printf("  %14s", buf);
        }
        std::printf("\n");
    }
}

// ============================================================================
// Chebyshev 4D
// ============================================================================

static ChebyshevTableResult build_chebyshev_surface() {
    ChebyshevTableConfig config{
        .num_pts = {20, 12, 12, 8},
        .domain = Domain<4>{
            .lo = {-0.50, 0.01, 0.05, 0.01},
            .hi = { 0.40, 2.50, 0.50, 0.10},
        },
        .K_ref = kSpot,
        .option_type = OptionType::PUT,
        .dividend_yield = kDivYield,
        .tucker_epsilon = 1e-8,
    };

    auto result = build_chebyshev_table(config);
    if (!result.has_value()) {
        std::fprintf(stderr, "Chebyshev build failed\n");
        std::exit(1);
    }

    std::printf("  PDE solves: %zu\n", result->n_pde_solves);
    std::printf("  Build time: %.2f s\n", result->build_seconds);

    return std::move(*result);
}

static ErrorTable compute_errors_chebyshev(
    const PriceGrid& prices,
    const ChebyshevTableResult& surface,
    size_t vol_idx) {
    ErrorTable errors{};
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    // Batch FDM IV
    std::vector<IVQuery> queries;
    std::vector<std::pair<size_t, size_t>> query_map;
    queries.reserve(kNT * kNS);

    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < kNS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) {
                errors[ti][si] = std::nan("");
                continue;
            }

            IVQuery q;
            q.spot = kSpot;
            q.strike = kStrikes[si];
            q.maturity = kMaturities[ti];
            q.rate = kRate;
            q.dividend_yield = kDivYield;
            q.option_type = OptionType::PUT;
            q.market_price = price;
            queries.push_back(q);
            query_map.emplace_back(ti, si);
        }
    }

    auto fdm_results = fdm_solver.solve_batch(queries);

    // Chebyshev IV via Brent
    for (size_t i = 0; i < queries.size(); ++i) {
        auto [ti, si] = query_map[i];

        if (!fdm_results.results[i].has_value()) {
            errors[ti][si] = std::nan("");
            continue;
        }

        double fdm_iv = fdm_results.results[i]->implied_vol;
        double strike = kStrikes[si];
        double maturity = kMaturities[ti];

        double cheb_iv = brent_solve_iv(
            [&](double vol) {
                return surface.price(kSpot, strike, maturity, vol, kRate);
            },
            queries[i].market_price);

        if (std::isnan(cheb_iv)) {
            errors[ti][si] = std::nan("");
            continue;
        }

        errors[ti][si] = std::abs(cheb_iv - fdm_iv) * 10000.0;
    }

    return errors;
}

static std::array<ErrorTable, kNV>
run_chebyshev_4d(const PriceGrid& prices) {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev 4D Tucker — vanilla (no dividends)\n");
    std::printf("================================================================\n\n");

    std::printf("--- Building Chebyshev 4D surface...\n");
    auto surface = build_chebyshev_surface();

    std::array<ErrorTable, kNV> all_errors{};
    std::printf("--- Computing Chebyshev IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Chebyshev 4D IV Error (bps) — σ=%.0f%%",
                      kVols[vi] * 100);
        all_errors[vi] = compute_errors_chebyshev(prices, surface, vi);
        print_heatmap(title, all_errors[vi]);
    }
    return all_errors;
}

// ============================================================================
// Chebyshev Adaptive — CC-level refinement via AdaptiveGridBuilder
// ============================================================================

static ErrorTable compute_errors_from_price_fn(
    const PriceGrid& prices,
    const std::function<double(double, double, double, double, double)>& price_fn,
    size_t vol_idx) {
    ErrorTable errors{};
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    std::vector<IVQuery> queries;
    std::vector<std::pair<size_t, size_t>> query_map;
    queries.reserve(kNT * kNS);

    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < kNS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) {
                errors[ti][si] = std::nan("");
                continue;
            }

            IVQuery q;
            q.spot = kSpot;
            q.strike = kStrikes[si];
            q.maturity = kMaturities[ti];
            q.rate = kRate;
            q.dividend_yield = kDivYield;
            q.option_type = OptionType::PUT;
            q.market_price = price;
            queries.push_back(q);
            query_map.emplace_back(ti, si);
        }
    }

    auto fdm_results = fdm_solver.solve_batch(queries);

    for (size_t i = 0; i < queries.size(); ++i) {
        auto [ti, si] = query_map[i];

        if (!fdm_results.results[i].has_value()) {
            errors[ti][si] = std::nan("");
            continue;
        }

        double fdm_iv = fdm_results.results[i]->implied_vol;
        double strike = kStrikes[si];
        double maturity = kMaturities[ti];

        double cheb_iv = brent_solve_iv(
            [&](double vol) {
                return price_fn(kSpot, strike, maturity, vol, kRate);
            },
            queries[i].market_price);

        if (std::isnan(cheb_iv)) {
            errors[ti][si] = std::nan("");
            continue;
        }

        errors[ti][si] = std::abs(cheb_iv - fdm_iv) * 10000.0;
    }

    return errors;
}

static std::array<ErrorTable, kNV>
run_chebyshev_adaptive(const PriceGrid& prices) {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev Adaptive — CC-level refinement\n");
    std::printf("================================================================\n\n");

    // Build OptionGrid from benchmark constants
    OptionGrid chain;
    chain.spot = kSpot;
    chain.dividend_yield = kDivYield;
    chain.strikes = std::vector<double>(kStrikes.begin(), kStrikes.end());
    chain.maturities = std::vector<double>(kMaturities.begin(), kMaturities.end());
    chain.implied_vols = std::vector<double>(kVols.begin(), kVols.end());
    chain.rates = {kRate};

    AdaptiveGridParams params;
    params.target_iv_error = 5e-4;  // 5 bps
    params.max_iter = 6;

    std::printf("--- Building adaptive Chebyshev surface (target=%.1f bps)...\n",
                params.target_iv_error * 1e4);

    auto result = build_adaptive_chebyshev(params, chain, OptionType::PUT);
    if (!result.has_value()) {
        std::fprintf(stderr, "Chebyshev adaptive build failed\n");
        std::array<ErrorTable, kNV> empty{};
        return empty;
    }

    // Print iteration stats
    std::printf("  Iterations: %zu, PDE solves: %zu, target_met: %s\n",
                result->iterations.size(),
                result->total_pde_solves,
                result->target_met ? "yes" : "no");
    for (const auto& it : result->iterations) {
        std::printf("  iter %zu: grid [%zu, %zu, %zu, %zu] "
                    "max_err=%.1f bps avg_err=%.1f bps PDE=%zu\n",
                    it.iteration,
                    it.grid_sizes[0], it.grid_sizes[1],
                    it.grid_sizes[2], it.grid_sizes[3],
                    it.max_error * 1e4, it.avg_error * 1e4,
                    it.pde_solves_table);
    }

    std::array<ErrorTable, kNV> all_errors{};
    std::printf("--- Computing adaptive Chebyshev IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Chebyshev Adaptive IV Error (bps) — σ=%.0f%%",
                      kVols[vi] * 100);
        all_errors[vi] = compute_errors_from_price_fn(
            prices, result->price_fn, vi);
        print_heatmap(title, all_errors[vi]);
    }
    return all_errors;
}

// ============================================================================
// Chebyshev Adaptive — Discrete Dividends (segmented, no EEP)
// ============================================================================

static std::array<ErrorTable, kNV>
run_chebyshev_dividends(const PriceGrid& prices) {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev Adaptive — Discrete Dividends (segmented)\n");
    std::printf("================================================================\n\n");

    AdaptiveGridParams params;
    params.target_iv_error = 5e-4;  // 5 bps
    params.max_iter = 6;

    SegmentedAdaptiveConfig config{
        .spot = kSpot,
        .option_type = OptionType::PUT,
        .dividend_yield = kDivYield,
        .discrete_dividends = make_div_schedule(1.0),
        .maturity = 1.0,
        .kref_config = {.K_refs = std::vector<double>(kStrikes.begin(), kStrikes.end())},
    };

    IVGrid domain{
        .moneyness = {std::log(kSpot / 120.0), std::log(kSpot / 115.0),
                      std::log(kSpot / 110.0), std::log(kSpot / 105.0),
                      std::log(kSpot / 100.0), std::log(kSpot / 95.0),
                      std::log(kSpot / 90.0), std::log(kSpot / 85.0),
                      std::log(kSpot / 80.0)},
        .vol = {0.10, 0.15, 0.20, 0.30, 0.50},
        .rate = {0.03, 0.05, 0.07},
    };

    std::printf("--- Building segmented Chebyshev surface (target=%.1f bps)...\n",
                params.target_iv_error * 1e4);

    auto result = build_adaptive_chebyshev_segmented(params, config, domain);
    if (!result.has_value()) {
        std::fprintf(stderr, "Chebyshev dividend build failed\n");
        std::array<ErrorTable, kNV> empty{};
        return empty;
    }

    std::printf("  Iterations: %zu, PDE solves: %zu, target_met: %s\n",
                result->iterations.size(),
                result->total_pde_solves,
                result->target_met ? "yes" : "no");
    for (const auto& it : result->iterations) {
        std::printf("  iter %zu: grid [%zu, %zu, %zu, %zu] "
                    "max_err=%.1f bps avg_err=%.1f bps\n",
                    it.iteration,
                    it.grid_sizes[0], it.grid_sizes[1],
                    it.grid_sizes[2], it.grid_sizes[3],
                    it.max_error * 1e4, it.avg_error * 1e4);
    }

    // Point diagnostic at T=1y (the surface's maturity) for both σ values
    auto diag_divs = make_div_schedule(1.0);
    std::printf("--- Diagnostic: surface vs FDM at T=1y (same dividends) ---\n");
    for (double sigma : {0.15, 0.30}) {
        std::printf("  σ=%.2f:\n", sigma);
        for (double K : {80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0}) {
            double surf = result->price_fn(kSpot, K, 1.0, sigma, kRate);
            PricingParams pp;
            pp.spot = kSpot; pp.strike = K; pp.maturity = 1.0;
            pp.rate = kRate; pp.dividend_yield = kDivYield;
            pp.option_type = OptionType::PUT; pp.volatility = sigma;
            pp.discrete_dividends = diag_divs;
            auto fdm = solve_american_option(pp);
            double ref = fdm.has_value() ? fdm->value() : -1.0;
            double pct_err = ref > 0.001 ? 100.0 * (surf - ref) / ref : 0.0;
            std::printf("    K=%3.0f: surf=%8.4f fdm=%8.4f diff=%+.4f (%.1f%%)\n",
                        K, surf, ref, surf - ref, pct_err);
        }
    }

    // Compute IV errors at each maturity using dividend-aware FDM reference.
    // The surface was built for maturity=1.0 with make_div_schedule(1.0).
    // At tau=T, the surface gives the value of the 1-year option with T time
    // remaining. For the reference, we solve the FDM with the same dividends.
    // Only dividends with calendar_time ≤ (maturity - tau_query) have been
    // "applied" at the query point, so the FDM reference with maturity=tau
    // needs the dividends that are still in the option's future.
    auto all_divs = make_div_schedule(1.0);
    double surface_maturity = 1.0;

    std::array<ErrorTable, kNV> all_errors{};
    std::printf("--- Computing Chebyshev dividend IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        auto& errors = all_errors[vi];
        for (auto& row : errors)
            for (auto& v : row)
                v = std::nan("");

        for (size_t ti = 0; ti < kNT; ++ti) {
            double tau = kMaturities[ti];
            if (tau > surface_maturity + 0.01) continue;

            // Filter dividends: keep those still in the future at this tau.
            // In the 1-year option at tau remaining, a dividend at cal_time t
            // is in the future if t > (surface_maturity - tau), i.e., the
            // calendar time hasn't passed yet.
            double cal_now = surface_maturity - tau;
            std::vector<Dividend> future_divs;
            for (const auto& d : all_divs) {
                if (d.calendar_time > cal_now)
                    future_divs.push_back(
                        Dividend{.calendar_time = d.calendar_time - cal_now,
                                 .amount = d.amount});
            }

            for (size_t si = 0; si < kNS; ++si) {
                double price = prices[vi][ti][si];
                if (std::isnan(price) || price <= 0) continue;

                // FDM reference with the same dividends
                double fdm_iv = solve_fdm_iv_div(
                    kStrikes[si], tau, price, future_divs);
                if (std::isnan(fdm_iv)) continue;

                // Surface IV via Brent inversion
                double cheb_iv = brent_solve_iv(
                    [&](double vol) {
                        return result->price_fn(
                            kSpot, kStrikes[si], tau, vol, kRate);
                    },
                    price);
                if (std::isnan(cheb_iv)) continue;

                errors[ti][si] = std::abs(cheb_iv - fdm_iv) * 10000.0;
            }
        }

        char title[128];
        std::snprintf(title, sizeof(title),
                      "Cheb Dividend IV Error (bps) — σ=%.0f%%",
                      kVols[vi] * 100);
        print_heatmap(title, all_errors[vi]);
    }
    return all_errors;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("Interpolation IV Safety Diagnostic\n");
    std::printf("===================================\n");
    std::printf("S=%.0f, r=%.2f, q=%.2f, PUT\n", kSpot, kRate, kDivYield);
    std::printf("Reference prices: BatchAmericanOptionSolver (chain mode)\n");
    std::printf("Interpolated IV:  adaptive IVGrid + make_interpolated_iv_solver\n");
    std::printf("FDM reference IV: IVSolver (vanilla) / Brent solver (dividends)\n");
    std::printf("Error = |interp_iv - fdm_iv| in basis points\n\n");

    std::printf("Strikes: ");
    for (double K : kStrikes) std::printf("%.0f ", K);
    std::printf("\nMaturities: ");
    for (size_t i = 0; i < kNT; ++i) std::printf("%s ", kMatLabels[i]);
    std::printf("\nVols: ");
    for (double v : kVols) std::printf("%.0f%% ", v * 100);
    std::printf("\n");

    // Step 1: Generate reference prices
    std::printf("\n--- Generating vanilla reference prices (batch chain solver)...\n");
    auto vanilla_prices = generate_prices(/*with_dividends=*/false);

    std::printf("--- Generating dividend reference prices (batch solver)...\n");
    auto div_prices = generate_prices(/*with_dividends=*/true);

    // Step 2: Build interpolated solvers
    std::printf("--- Building vanilla interpolated solver (adaptive)...\n");
    auto vanilla_solver = build_vanilla_solver();

    std::printf("--- Building dividend interpolated solvers (per-maturity)...\n");
    auto div_solvers = build_div_solvers();

    // Step 3 & 4: Compute errors and print heatmaps
    std::array<ErrorTable, kNV> vanilla_errors{};
    std::array<ErrorTable, kNV> div_errors{};

    std::printf("--- Computing vanilla IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Interpolation IV Error (bps) — σ=%.0f%%, no dividends",
                      kVols[vi] * 100);
        vanilla_errors[vi] = compute_errors_vanilla(vanilla_prices, vanilla_solver, vi);
        print_heatmap(title, vanilla_errors[vi]);
    }

    std::printf("\n--- Computing dividend IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Interpolation IV Error (bps) — σ=%.0f%%, quarterly $0.50 div",
                      kVols[vi] * 100);
        div_errors[vi] = compute_errors_div(div_prices, div_solvers, vi);
        print_heatmap(title, div_errors[vi]);
    }

    // Chebyshev 4D (fixed grid)
    auto cheb_errors = run_chebyshev_4d(vanilla_prices);

    // Chebyshev adaptive (CC-level refinement)
    auto cheb_adaptive_errors = run_chebyshev_adaptive(vanilla_prices);

    // Chebyshev adaptive dividends (segmented)
    auto cheb_div_errors = run_chebyshev_dividends(div_prices);

    // TV/K filtered comparison: same mask for all algorithms
    std::printf("\n================================================================\n");
    std::printf("TV/K Filtered Comparison — vanilla (no dividends)\n");
    std::printf("================================================================\n");

    for (size_t vi = 0; vi < kNV; ++vi) {
        AlgoErrors algos[] = {
            {"B-spline", &vanilla_errors[vi]},
            {"Cheb(fixed)", &cheb_errors[vi]},
            {"Cheb(adapt)", &cheb_adaptive_errors[vi]},
        };
        print_tvk_comparison(vanilla_prices, vi, algos);
    }

    std::printf("\n================================================================\n");
    std::printf("TV/K Filtered Comparison — discrete dividends\n");
    std::printf("================================================================\n");

    for (size_t vi = 0; vi < kNV; ++vi) {
        AlgoErrors algos[] = {
            {"B-spline(div)", &div_errors[vi]},
            {"Cheb(div)", &cheb_div_errors[vi]},
        };
        print_tvk_comparison(div_prices, vi, algos);
    }

    return 0;
}
