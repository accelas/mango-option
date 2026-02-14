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
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
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

static PriceGrid generate_prices(bool with_dividends, double div_yield = kDivYield) {
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
                p.dividend_yield = div_yield;
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
static AnyInterpIVSolver build_vanilla_solver() {
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

using BSplineDivSolver = InterpolatedIVSolver<BSplineMultiKRefSurface>;

// Dividends: one solver per maturity via BSpline + discrete dividends + adaptive grid.
// Uses the lower-level builder directly to capture convergence stats.
static std::vector<std::pair<size_t, BSplineDivSolver>> build_div_solvers() {
    std::vector<std::pair<size_t, BSplineDivSolver>> solvers;

    // S/K moneyness → log-moneyness for the builder
    const std::vector<double> sk_moneyness = {0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30};
    std::vector<double> log_m;
    log_m.reserve(sk_moneyness.size());
    for (double m : sk_moneyness) log_m.push_back(std::log(m));

    const std::vector<double> vols = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50};
    const std::vector<double> rates = {0.01, 0.03, 0.05, 0.10};
    const AdaptiveGridParams adaptive{.target_iv_error = 2e-5};

    for (size_t ti = 0; ti < kNT; ++ti) {
        double mat = kMaturities[ti];
        auto divs = make_div_schedule(mat);

        SegmentedAdaptiveConfig seg_config{
            .spot = kSpot,
            .option_type = OptionType::PUT,
            .dividend_yield = kDivYield,
            .discrete_dividends = divs,
            .maturity = mat,
            .kref_config = {.K_refs = std::vector<double>(kStrikes.begin(), kStrikes.end())},
        };

        auto result = build_adaptive_bspline_segmented(
            adaptive, seg_config, {log_m, vols, rates});
        if (!result.has_value()) {
            std::fprintf(stderr, "  [skip] T=%s — adaptive build failed\n",
                         kMatLabels[ti]);
            continue;
        }

        // Print convergence stats
        std::printf("  T=%s: iters=%zu target_met=%s max_err=%.1f bps "
                    "avg_err=%.1f bps PDE=%zu%s\n",
                    kMatLabels[ti],
                    result->iterations.size(),
                    result->target_met ? "yes" : "no",
                    result->achieved_max_error * 1e4,
                    result->achieved_avg_error * 1e4,
                    result->total_pde_solves,
                    result->used_retry ? " (retry)" : "");

        // Wrap in BSplineMultiKRefSurface → InterpolatedIVSolver
        SurfaceBounds bounds{
            .m_min = log_m.front(), .m_max = log_m.back(),
            .tau_min = 0.0, .tau_max = mat,
            .sigma_min = vols.front(), .sigma_max = vols.back(),
            .rate_min = rates.front(), .rate_max = rates.back(),
        };
        auto wrapper = BSplineMultiKRefSurface(
            std::move(result->surface), bounds, OptionType::PUT, kDivYield);
        auto solver = BSplineDivSolver::create(std::move(wrapper));
        if (!solver.has_value()) {
            std::fprintf(stderr, "  [skip] T=%s — solver wrap failed\n",
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
                                          const AnyInterpIVSolver& interp_solver,
                                          size_t vol_idx,
                                          double div_yield = kDivYield) {
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
            q.dividend_yield = div_yield;
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

template <typename Solver>
static ErrorTable compute_errors_div(
    const PriceGrid& prices,
    const std::vector<std::pair<size_t, Solver>>& div_solvers,
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

/// Generic error computation via any InterpolatedIVSolver.
/// The solver's built-in vega pre-check handles edge-case filtering.
template <typename Solver>
static ErrorTable compute_errors_via_solver(
    const PriceGrid& prices,
    const Solver& interp_solver,
    size_t vol_idx,
    double div_yield = kDivYield) {
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
            q.dividend_yield = div_yield;
            q.option_type = OptionType::PUT;
            q.market_price = price;
            queries.push_back(q);
            query_map.emplace_back(ti, si);
        }
    }

    auto fdm_results = fdm_solver.solve_batch(queries);
    auto interp_results = interp_solver.solve_batch(queries);

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

static std::array<ErrorTable, kNV>
run_chebyshev_4d(const PriceGrid& prices) {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev 4D — vanilla (no dividends)\n");
    std::printf("================================================================\n\n");

    std::printf("--- Building Chebyshev 4D surface...\n");
    auto surface = build_chebyshev_surface();

    // Wrap in InterpolatedIVSolver for consistent vega pre-check
    std::array<ErrorTable, kNV> all_errors{};
    std::printf("--- Computing Chebyshev IV errors...\n");

    auto solver = InterpolatedIVSolver<ChebyshevSurface>::create(
        std::move(surface.surface));
    if (!solver.has_value()) {
        std::fprintf(stderr, "Chebyshev 4D solver creation failed\n");
    } else {
        for (size_t vi = 0; vi < kNV; ++vi) {
            char title[128];
            std::snprintf(title, sizeof(title),
                          "Chebyshev 4D IV Error (bps) — σ=%.0f%%",
                          kVols[vi] * 100);
            all_errors[vi] = compute_errors_via_solver(prices, *solver, vi);
            print_heatmap(title, all_errors[vi]);
        }
    }
    return all_errors;
}

// ============================================================================
// Chebyshev Adaptive — CC-level refinement via AdaptiveGridBuilder
// ============================================================================

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

    // Wrap in InterpolatedIVSolver for consistent vega pre-check
    auto solver = InterpolatedIVSolver<ChebyshevRawSurface>::create(
        std::move(*result->surface));
    if (!solver.has_value()) {
        std::fprintf(stderr, "Chebyshev adaptive solver creation failed\n");
        return {};
    }

    std::array<ErrorTable, kNV> all_errors{};
    std::printf("--- Computing adaptive Chebyshev IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Chebyshev Adaptive IV Error (bps) — σ=%.0f%%",
                      kVols[vi] * 100);
        all_errors[vi] = compute_errors_via_solver(prices, *solver, vi);
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
            double surf = result->surface.price(kSpot, K, 1.0, sigma, kRate);
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

    // Wrap in InterpolatedIVSolver for consistent vega pre-check
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(
        std::move(result->surface));
    if (!solver.has_value()) {
        std::fprintf(stderr, "Chebyshev dividend solver creation failed\n");
        return {};
    }

    // Compute IV errors at each maturity using dividend-aware FDM reference.
    // The surface was built for maturity=1.0 with make_div_schedule(1.0).
    // FDM reference needs dividends still in the option's future at each tau.
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

                // Surface IV via InterpolatedIVSolver (vega pre-check built in)
                IVQuery q;
                q.spot = kSpot;
                q.strike = kStrikes[si];
                q.maturity = tau;
                q.rate = kRate;
                q.dividend_yield = kDivYield;
                q.option_type = OptionType::PUT;
                q.market_price = price;
                auto iv_result = solver->solve(q);
                if (!iv_result.has_value()) continue;

                errors[ti][si] = std::abs(iv_result->implied_vol - fdm_iv) * 10000.0;
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
// q=0 comparison: 4D B-spline vs 3D dimensionless (B-spline & Chebyshev)
// ============================================================================

static AnyInterpIVSolver build_bspline_q0() {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = kSpot,
        .dividend_yield = 0.0,
        .grid = IVGrid{
            .moneyness = {0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30},
            .vol = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50},
            .rate = {0.01, 0.03, 0.05, 0.10},
        },
        .adaptive = AdaptiveGridParams{.target_iv_error = 2e-5},
        .backend = BSplineBackend{
            .maturity_grid = {0.01, 0.03, 0.06, 0.12, 0.20,
                              0.35, 0.60, 1.0, 1.5, 2.0, 2.5},
        },
    };
    auto solver = make_interpolated_iv_solver(config);
    if (!solver.has_value()) {
        std::fprintf(stderr, "4D B-spline (q=0) build failed\n");
        std::exit(1);
    }
    return std::move(*solver);
}

static std::expected<AnyInterpIVSolver, ValidationError>
build_dimless_3d(DimensionlessBackend::Interpolant interp) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = kSpot,
        .dividend_yield = 0.0,
        .grid = IVGrid{
            .moneyness = {0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30},
            .vol = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50},
            .rate = {0.01, 0.03, 0.05, 0.10},
        },
        .backend = DimensionlessBackend{.maturity = 2.5, .interpolant = interp},
    };
    return make_interpolated_iv_solver(config);
}

// ============================================================================
// CLI path selection
// ============================================================================

static std::string_view parse_path(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--path=", 7) == 0) {
            return std::string_view(argv[i] + 7);
        }
    }
    return "all";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    auto path = parse_path(argc, argv);
    bool run_all = (path == "all");

    if (!run_all) {
        std::printf("Running path: %.*s\n\n",
                    static_cast<int>(path.size()), path.data());
    }
    std::printf("Interpolation IV Safety Diagnostic\n");
    std::printf("===================================\n");
    std::printf("S=%.0f, r=%.2f, q=%.2f, PUT\n", kSpot, kRate, kDivYield);
    std::printf("Reference prices: BatchAmericanOptionSolver (chain mode)\n");
    std::printf("Interpolated IV:  InterpolatedIVSolver (all backends, vega pre-check)\n");
    std::printf("FDM reference IV: IVSolver (vanilla) / Brent solver (dividends)\n");
    std::printf("Error = |interp_iv - fdm_iv| in basis points\n\n");

    std::printf("Strikes: ");
    for (double K : kStrikes) std::printf("%.0f ", K);
    std::printf("\nMaturities: ");
    for (size_t i = 0; i < kNT; ++i) std::printf("%s ", kMatLabels[i]);
    std::printf("\nVols: ");
    for (double v : kVols) std::printf("%.0f%% ", v * 100);
    std::printf("\n");

    std::printf("Usage: interp_iv_safety [--path=all|bspline|chebyshev|q0|dividends]\n\n");

    // Step 1: Generate reference prices (always needed)
    PriceGrid vanilla_prices{}, div_prices{};
    bool need_vanilla = run_all || path == "bspline" || path == "chebyshev";
    bool need_divs = run_all || path == "dividends";
    bool need_q0 = run_all || path == "q0";

    if (need_vanilla) {
        std::printf("--- Generating vanilla reference prices (batch chain solver)...\n");
        vanilla_prices = generate_prices(/*with_dividends=*/false);
    }
    if (need_divs) {
        std::printf("--- Generating dividend reference prices (batch solver)...\n");
        div_prices = generate_prices(/*with_dividends=*/true);
    }

    // Per-path error tables
    std::array<ErrorTable, kNV> vanilla_errors{};
    std::array<ErrorTable, kNV> div_errors{};
    std::array<ErrorTable, kNV> cheb_errors{};
    std::array<ErrorTable, kNV> cheb_adaptive_errors{};
    std::array<ErrorTable, kNV> cheb_div_errors{};
    std::array<ErrorTable, kNV> q0_bs4d_errors{};
    std::array<ErrorTable, kNV> q0_dim3d_bs_errors{};
    std::array<ErrorTable, kNV> q0_dim3d_ch_errors{};

    // B-spline adaptive (vanilla + dividends)
    if (run_all || path == "bspline") {
        std::printf("--- Building vanilla interpolated solver (adaptive)...\n");
        auto vanilla_solver = build_vanilla_solver();

        std::printf("--- Computing vanilla IV errors...\n");
        for (size_t vi = 0; vi < kNV; ++vi) {
            char title[128];
            std::snprintf(title, sizeof(title),
                          "Interpolation IV Error (bps) — σ=%.0f%%, no dividends",
                          kVols[vi] * 100);
            vanilla_errors[vi] = compute_errors_vanilla(vanilla_prices, vanilla_solver, vi);
            print_heatmap(title, vanilla_errors[vi]);
        }
    }

    if (run_all || path == "dividends") {
        std::printf("--- Building dividend interpolated solvers (per-maturity)...\n");
        auto div_solvers = build_div_solvers();

        std::printf("\n--- Computing dividend IV errors...\n");
        for (size_t vi = 0; vi < kNV; ++vi) {
            char title[128];
            std::snprintf(title, sizeof(title),
                          "Interpolation IV Error (bps) — σ=%.0f%%, quarterly $0.50 div",
                          kVols[vi] * 100);
            div_errors[vi] = compute_errors_div(div_prices, div_solvers, vi);
            print_heatmap(title, div_errors[vi]);
        }
    }

    // Chebyshev 4D
    if (run_all || path == "chebyshev") {
        cheb_errors = run_chebyshev_4d(vanilla_prices);
        cheb_adaptive_errors = run_chebyshev_adaptive(vanilla_prices);
    }

    // Chebyshev dividends
    if (run_all || path == "dividends") {
        cheb_div_errors = run_chebyshev_dividends(div_prices);
    }

    // q=0 comparison: 4D B-spline vs dimensionless 3D (B-spline & Chebyshev)
    PriceGrid q0_prices{};
    if (need_q0) {
        std::printf("\n================================================================\n");
        std::printf("q=0 Comparison: 4D B-spline vs Dimensionless 3D\n");
        std::printf("================================================================\n");

        std::printf("--- Generating q=0 reference prices...\n");
        q0_prices = generate_prices(/*with_dividends=*/false, /*div_yield=*/0.0);

        std::printf("--- Building 4D B-spline (q=0, adaptive)...\n");
        auto bs4d_solver = build_bspline_q0();
        for (size_t vi = 0; vi < kNV; ++vi) {
            q0_bs4d_errors[vi] = compute_errors_vanilla(q0_prices, bs4d_solver, vi, 0.0);
            char title[128];
            std::snprintf(title, sizeof(title),
                          "4D B-spline (q=0) IV Error (bps) — σ=%.0f%%",
                          kVols[vi] * 100);
            print_heatmap(title, q0_bs4d_errors[vi]);
        }

        std::printf("\n--- Building dimensionless 3D B-spline (q=0)...\n");
        auto dim3d_bs = build_dimless_3d(DimensionlessBackend::Interpolant::BSpline);
        if (dim3d_bs.has_value()) {
            for (size_t vi = 0; vi < kNV; ++vi) {
                q0_dim3d_bs_errors[vi] = compute_errors_vanilla(
                    q0_prices, *dim3d_bs, vi, 0.0);
                char title[128];
                std::snprintf(title, sizeof(title),
                              "Dim3D B-spline (q=0) IV Error (bps) — σ=%.0f%%",
                              kVols[vi] * 100);
                print_heatmap(title, q0_dim3d_bs_errors[vi]);
            }
        } else {
            std::fprintf(stderr, "Dimensionless 3D B-spline build failed\n");
        }

        std::printf("\n--- Building dimensionless 3D Chebyshev (q=0)...\n");
        auto dim3d_ch = build_dimless_3d(DimensionlessBackend::Interpolant::Chebyshev);
        if (dim3d_ch.has_value()) {
            for (size_t vi = 0; vi < kNV; ++vi) {
                q0_dim3d_ch_errors[vi] = compute_errors_vanilla(
                    q0_prices, *dim3d_ch, vi, 0.0);
                char title[128];
                std::snprintf(title, sizeof(title),
                              "Dim3D Chebyshev (q=0) IV Error (bps) — σ=%.0f%%",
                              kVols[vi] * 100);
                print_heatmap(title, q0_dim3d_ch_errors[vi]);
            }
        } else {
            std::fprintf(stderr, "Dimensionless 3D Chebyshev build failed\n");
        }
    }

    // TV/K filtered comparison — vanilla backends (q=0.02)
    if (need_vanilla) {
        std::printf("\n================================================================\n");
        std::printf("TV/K Filtered Comparison — vanilla (q=%.2f)\n", kDivYield);
        std::printf("================================================================\n");

        for (size_t vi = 0; vi < kNV; ++vi) {
            std::vector<AlgoErrors> vol_algos;
            if (run_all || path == "bspline")
                vol_algos.push_back({"B-spline", &vanilla_errors[vi]});
            if (run_all || path == "chebyshev") {
                vol_algos.push_back({"Cheb(fixed)", &cheb_errors[vi]});
                vol_algos.push_back({"Cheb(adapt)", &cheb_adaptive_errors[vi]});
            }
            if (!vol_algos.empty())
                print_tvk_comparison(vanilla_prices, vi, vol_algos);
        }
    }

    // TV/K filtered comparison — q=0 (4D vs 3D dimensionless)
    if (need_q0) {
        std::printf("\n================================================================\n");
        std::printf("TV/K Filtered Comparison — q=0 (4D B-spline vs Dim3D)\n");
        std::printf("================================================================\n");

        for (size_t vi = 0; vi < kNV; ++vi) {
            std::vector<AlgoErrors> vol_algos;
            vol_algos.push_back({"BS-4D(q=0)", &q0_bs4d_errors[vi]});
            vol_algos.push_back({"Dim3D-BS", &q0_dim3d_bs_errors[vi]});
            vol_algos.push_back({"Dim3D-Ch", &q0_dim3d_ch_errors[vi]});
            print_tvk_comparison(q0_prices, vi, vol_algos);
        }
    }

    if (need_divs) {
        std::printf("\n================================================================\n");
        std::printf("TV/K Filtered Comparison — discrete dividends\n");
        std::printf("================================================================\n");

        for (size_t vi = 0; vi < kNV; ++vi) {
            std::vector<AlgoErrors> vol_algos;
            if (run_all || path == "dividends") {
                vol_algos.push_back({"B-spline(div)", &div_errors[vi]});
                vol_algos.push_back({"Cheb(div)", &cheb_div_errors[vi]});
            }
            if (!vol_algos.empty())
                print_tvk_comparison(div_prices, vi, vol_algos);
        }
    }

    return 0;
}
