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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_inner.hpp"
#include "chebyshev_eep_inner.hpp"
#include "chebyshev_4d_eep_inner.hpp"

using namespace mango;
using namespace mango::bench;

static std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = lo + (hi - lo) * i / (n - 1);
    return v;
}

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

// Vanilla: one solver covering all maturities via StandardIVPath + adaptive grid
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
        .path = StandardIVPath{
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

// Dividends: one solver per maturity via SegmentedIVPath + adaptive grid
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
            .path = SegmentedIVPath{
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
// 3D Dimensionless surface helpers (q=0 comparison)
// ============================================================================

static PriceGrid generate_prices_q0() {
    PriceGrid prices{};
    BatchAmericanOptionSolver batch_solver;
    std::vector<PricingParams> all_params;
    all_params.reserve(kNV * kNT * kNS);

    for (size_t vi = 0; vi < kNV; ++vi) {
        for (size_t ti = 0; ti < kNT; ++ti) {
            for (size_t si = 0; si < kNS; ++si) {
                PricingParams p;
                p.spot = kSpot;
                p.strike = kStrikes[si];
                p.maturity = kMaturities[ti];
                p.rate = kRate;
                p.dividend_yield = 0.0;  // q=0 for dimensionless comparison
                p.option_type = OptionType::PUT;
                p.volatility = kVols[vi];
                all_params.push_back(std::move(p));
            }
        }
    }

    auto result = batch_solver.solve_batch(all_params, /*use_shared_grid=*/true);
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

static DimensionlessEEPInner build_3d_surface() {
    auto t0 = std::chrono::steady_clock::now();
    DimensionlessAxes axes;
    // x: log-moneyness covering S/K from ~0.60 to ~1.5
    axes.log_moneyness = linspace(-0.50, 0.40, 25);
    // tau' = sigma^2*tau/2: covers sigma=[0.10,0.50], tau=[30d,2y]
    axes.tau_prime = linspace(0.005, 0.125, 20);
    // ln(kappa) = ln(2r/sigma^2): r=0.05, sigma=[0.10,0.50]
    // kappa range ~[0.14, 16.4] → ln_kappa ~[-2.0, 2.8]
    axes.ln_kappa = linspace(-2.0, 2.8, 30);

    auto result = build_dimensionless_surface(
        axes, kSpot, OptionType::PUT, SurfaceContent::EarlyExercisePremium);
    if (!result) {
        std::fprintf(stderr, "3D surface build failed (code=%d)\n",
                     static_cast<int>(result.error().code));
        std::exit(1);
    }
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::printf("  3D surface: %d PDE solves, %.3fs build\n",
                result->n_pde_solves, elapsed);

    return DimensionlessEEPInner(result->surface, OptionType::PUT, kSpot, 0.0);
}

static AnyIVSolver build_vanilla_solver_q0() {
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
        .path = StandardIVPath{
            .maturity_grid = {0.01, 0.03, 0.06, 0.12, 0.20,
                              0.35, 0.60, 1.0, 1.5, 2.0, 2.5},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    if (!solver) {
        std::fprintf(stderr, "Failed to build 4D q=0 solver\n");
        std::exit(1);
    }
    return std::move(*solver);
}

static ErrorTable compute_errors_3d(const PriceGrid& prices,
                                     const DimensionlessEEPInner& inner_3d,
                                     size_t vol_idx) {
    ErrorTable errors{};
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < kNS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) {
                errors[ti][si] = std::nan("");
                continue;
            }

            // FDM reference IV at q=0
            IVQuery fdm_q;
            fdm_q.spot = kSpot;
            fdm_q.strike = kStrikes[si];
            fdm_q.maturity = kMaturities[ti];
            fdm_q.rate = kRate;
            fdm_q.dividend_yield = 0.0;
            fdm_q.option_type = OptionType::PUT;
            fdm_q.market_price = price;

            auto fdm_result = fdm_solver.solve(fdm_q);
            if (!fdm_result) {
                errors[ti][si] = std::nan("");
                continue;
            }

            // 3D interpolated IV via Brent
            double iv_3d = brent_solve_iv(
                [&](double vol) -> double {
                    PriceQuery q{.spot = kSpot, .strike = kStrikes[si],
                                 .tau = kMaturities[ti], .sigma = vol,
                                 .rate = kRate};
                    return inner_3d.price(q);
                },
                price);

            if (!std::isfinite(iv_3d)) {
                errors[ti][si] = std::nan("");
                continue;
            }

            errors[ti][si] = std::abs(iv_3d - fdm_result->implied_vol) * 10000.0;
        }
    }
    return errors;
}

// ============================================================================
// Step 4b: Chebyshev-Tucker 3D surface (same domain as B-spline 3D)
// ============================================================================

static ChebyshevEEPInner build_chebyshev_3d_surface() {
    ChebyshevEEPConfig cfg;  // use wide defaults

    auto t0 = std::chrono::steady_clock::now();
    auto result = build_chebyshev_eep(cfg, kSpot, OptionType::PUT);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    auto ranks = result.interp.ranks();
    std::printf("  Chebyshev surface: %d PDE solves, %.3fs build, "
                "ranks=(%zu,%zu,%zu)\n",
                result.n_pde_solves, elapsed, ranks[0], ranks[1], ranks[2]);

    return ChebyshevEEPInner(
        std::move(result.interp), OptionType::PUT, kSpot, 0.0);
}

static ErrorTable compute_errors_chebyshev(const PriceGrid& prices,
                                            const ChebyshevEEPInner& inner,
                                            size_t vol_idx) {
    ErrorTable errors{};
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < kNS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) {
                errors[ti][si] = std::nan("");
                continue;
            }

            IVQuery fdm_q;
            fdm_q.spot = kSpot;
            fdm_q.strike = kStrikes[si];
            fdm_q.maturity = kMaturities[ti];
            fdm_q.rate = kRate;
            fdm_q.dividend_yield = 0.0;
            fdm_q.option_type = OptionType::PUT;
            fdm_q.market_price = price;

            auto fdm_result = fdm_solver.solve(fdm_q);
            if (!fdm_result) {
                errors[ti][si] = std::nan("");
                continue;
            }

            double iv_cheb = brent_solve_iv(
                [&](double vol) -> double {
                    PriceQuery q{.spot = kSpot, .strike = kStrikes[si],
                                 .tau = kMaturities[ti], .sigma = vol,
                                 .rate = kRate};
                    return inner.price(q);
                },
                price);

            if (!std::isfinite(iv_cheb)) {
                errors[ti][si] = std::nan("");
                continue;
            }

            errors[ti][si] = std::abs(iv_cheb - fdm_result->implied_vol) * 10000.0;
        }
    }
    return errors;
}

// ============================================================================
// Step 4c: Chebyshev-Tucker 4D surface (ln(S/K), tau, sigma, rate)
// ============================================================================

static Chebyshev4DEEPInner build_chebyshev_4d_surface() {
    Chebyshev4DEEPConfig cfg;  // use defaults: 10x10x15x6, epsilon=1e-8

    auto t0 = std::chrono::steady_clock::now();
    auto result = build_chebyshev_4d_eep(cfg, kSpot, OptionType::PUT);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    auto ranks = result.interp.ranks();
    std::printf("  Chebyshev 4D surface: %d PDE solves, %.3fs build, "
                "ranks=(%zu,%zu,%zu,%zu)\n",
                result.n_pde_solves, elapsed,
                ranks[0], ranks[1], ranks[2], ranks[3]);

    return Chebyshev4DEEPInner(
        std::move(result.interp), OptionType::PUT, kSpot, 0.0);
}

static ErrorTable compute_errors_chebyshev_4d(const PriceGrid& prices,
                                               const Chebyshev4DEEPInner& inner,
                                               size_t vol_idx) {
    ErrorTable errors{};
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < kNS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) {
                errors[ti][si] = std::nan("");
                continue;
            }

            IVQuery fdm_q;
            fdm_q.spot = kSpot;
            fdm_q.strike = kStrikes[si];
            fdm_q.maturity = kMaturities[ti];
            fdm_q.rate = kRate;
            fdm_q.dividend_yield = 0.0;
            fdm_q.option_type = OptionType::PUT;
            fdm_q.market_price = price;

            auto fdm_result = fdm_solver.solve(fdm_q);
            if (!fdm_result) {
                errors[ti][si] = std::nan("");
                continue;
            }

            double iv_cheb4d = brent_solve_iv(
                [&](double vol) -> double {
                    PriceQuery q{.spot = kSpot, .strike = kStrikes[si],
                                 .tau = kMaturities[ti], .sigma = vol,
                                 .rate = kRate};
                    return inner.price(q);
                },
                price);

            if (!std::isfinite(iv_cheb4d)) {
                errors[ti][si] = std::nan("");
                continue;
            }

            errors[ti][si] = std::abs(iv_cheb4d - fdm_result->implied_vol) * 10000.0;
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
    std::printf("--- Computing vanilla IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Interpolation IV Error (bps) — σ=%.0f%%, no dividends",
                      kVols[vi] * 100);
        auto errors = compute_errors_vanilla(vanilla_prices, vanilla_solver, vi);
        print_heatmap(title, errors);
    }

    std::printf("\n--- Computing dividend IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Interpolation IV Error (bps) — σ=%.0f%%, quarterly $0.50 div",
                      kVols[vi] * 100);
        auto errors = compute_errors_div(div_prices, div_solvers, vi);
        print_heatmap(title, errors);
    }

    // ================================================================
    // 3D Dimensionless vs 4D Standard (q=0)
    // ================================================================
    std::printf("\n\n================================================================\n");
    std::printf("3D Dimensionless vs 4D Standard — q=0, no dividends\n");
    std::printf("================================================================\n\n");

    std::printf("--- Generating q=0 reference prices...\n");
    auto q0_prices = generate_prices_q0();

    std::printf("--- Building 3D dimensionless surface...\n");
    auto inner_3d = build_3d_surface();

    std::printf("--- Building 4D standard surface (q=0)...\n");
    auto solver_4d_q0 = build_vanilla_solver_q0();

    std::printf("--- Computing errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title_3d[128], title_4d[128];
        std::snprintf(title_3d, sizeof(title_3d),
                      "3D Dimensionless IV Error (bps) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);
        std::snprintf(title_4d, sizeof(title_4d),
                      "4D Standard IV Error (bps) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);

        auto errors_3d = compute_errors_3d(q0_prices, inner_3d, vi);
        auto errors_4d = compute_errors_vanilla(q0_prices, solver_4d_q0, vi, 0.0);

        print_heatmap(title_3d, errors_3d);
        print_heatmap(title_4d, errors_4d);
    }

    // ================================================================
    // Chebyshev-Tucker 3D vs B-spline 3D (q=0)
    // ================================================================
    std::printf("\n\n================================================================\n");
    std::printf("Chebyshev-Tucker 3D vs B-spline 3D — q=0, no dividends\n");
    std::printf("================================================================\n\n");

    std::printf("--- Building Chebyshev-Tucker 3D surface...\n");
    auto inner_cheb = build_chebyshev_3d_surface();

    std::printf("--- Computing Chebyshev IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title_cheb[128];
        std::snprintf(title_cheb, sizeof(title_cheb),
                      "Chebyshev-Tucker IV Error (bps) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);

        auto errors_cheb = compute_errors_chebyshev(q0_prices, inner_cheb, vi);
        print_heatmap(title_cheb, errors_cheb);
    }

    // ================================================================
    // Chebyshev-Tucker 4D (ln(S/K), tau, sigma, rate)
    // ================================================================
    std::printf("\n\n================================================================\n");
    std::printf("Chebyshev-Tucker 4D — q=0, no dividends\n");
    std::printf("================================================================\n\n");

    std::printf("--- Building Chebyshev-Tucker 4D surface...\n");
    auto inner_cheb4d = build_chebyshev_4d_surface();

    std::printf("--- Computing Chebyshev 4D IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title_cheb4d[128];
        std::snprintf(title_cheb4d, sizeof(title_cheb4d),
                      "Chebyshev-Tucker 4D IV Error (bps) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);

        auto errors_cheb4d = compute_errors_chebyshev_4d(q0_prices, inner_cheb4d, vi);
        print_heatmap(title_cheb4d, errors_cheb4d);
    }

    return 0;
}
