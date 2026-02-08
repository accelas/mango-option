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
 *
 * Individual sections:
 *   bazel run //benchmarks:interp_iv_safety -- vanilla
 *   bazel run //benchmarks:interp_iv_safety -- dividends
 *   bazel run //benchmarks:interp_iv_safety -- bspline-3d
 *   bazel run //benchmarks:interp_iv_safety -- bspline-4d
 *   bazel run //benchmarks:interp_iv_safety -- cheb3d
 *   bazel run //benchmarks:interp_iv_safety -- cheb4d
 *
 * Multiple:  ... -- cheb3d cheb4d
 */

#include "iv_benchmark_common.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/table/standard_surface.hpp"
#include "mango/option/table/adaptive_grid_builder.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
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
    Chebyshev4DEEPConfig cfg;
    cfg.num_x = 40;
    cfg.num_tau = 15;

    // Coordinate transforms (toggle individually)
    cfg.use_sinh_x = false;      // sinh mapping clusters x-nodes near ATM
    cfg.use_sqrt_tau = false;    // sqrt(tau) clusters nodes at short maturities
    cfg.use_log_eep = false;     // log(EEP+eps) smooths sharp EEP transition

    auto t0 = std::chrono::steady_clock::now();
    auto result = build_chebyshev_4d_eep(cfg, kSpot, OptionType::PUT);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    auto ranks = result.interp.ranks();
    std::printf("  Chebyshev 4D surface: %d PDE solves, %.3fs build, "
                "ranks=(%zu,%zu,%zu,%zu)\n",
                result.n_pde_solves, elapsed,
                ranks[0], ranks[1], ranks[2], ranks[3]);
    std::printf("  Transforms: sinh_x=%s sqrt_tau=%s log_eep=%s\n",
                cfg.use_sinh_x ? "ON" : "off",
                cfg.use_sqrt_tau ? "ON" : "off",
                cfg.use_log_eep ? "ON" : "off");

    return Chebyshev4DEEPInner(
        std::move(result.interp), OptionType::PUT, kSpot, 0.0,
        result.transforms);
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
// Generic Brent-based error computation (works with any price function)
// ============================================================================

template <typename PriceFn>
static ErrorTable compute_errors_brent(const PriceGrid& prices,
                                        PriceFn&& price_fn,
                                        size_t vol_idx,
                                        double div_yield = 0.0) {
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
            fdm_q.dividend_yield = div_yield;
            fdm_q.option_type = OptionType::PUT;
            fdm_q.market_price = price;

            auto fdm_result = fdm_solver.solve(fdm_q);
            if (!fdm_result) {
                errors[ti][si] = std::nan("");
                continue;
            }

            double iv = brent_solve_iv(
                [&](double vol) -> double {
                    return price_fn(kSpot, kStrikes[si], kMaturities[ti], vol, kRate);
                },
                price);

            if (!std::isfinite(iv)) {
                errors[ti][si] = std::nan("");
                continue;
            }

            errors[ti][si] = std::abs(iv - fdm_result->implied_vol) * 10000.0;
        }
    }
    return errors;
}

// Build B-spline 4D surface wrapper for Brent-based comparison
static StandardSurfaceWrapper build_bspline_4d_wrapper() {
    auto t0 = std::chrono::steady_clock::now();

    OptionGrid chain;
    chain.spot = kSpot;
    chain.dividend_yield = 0.0;
    chain.strikes = {76.9, 83.3, 90.9, 100.0, 111.1, 125.0, 142.9};
    chain.maturities = {0.01, 0.03, 0.06, 0.12, 0.20,
                        0.35, 0.60, 1.0, 1.5, 2.0, 2.5};
    chain.implied_vols = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50};
    chain.rates = {0.01, 0.03, 0.05, 0.10};

    AdaptiveGridParams params;
    params.target_iv_error = 2e-5;
    AdaptiveGridBuilder builder(params);
    GridAccuracyParams accuracy = make_grid_accuracy(GridAccuracyProfile::High);

    auto result = builder.build(chain, accuracy, OptionType::PUT);
    if (!result) {
        std::fprintf(stderr, "B-spline 4D surface build failed\n");
        std::exit(1);
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::printf("  B-spline 4D surface: %zu PDE solves, %.3fs build\n",
                result->total_pde_solves, elapsed);

    auto wrapper = make_standard_wrapper(result->surface, OptionType::PUT);
    if (!wrapper) {
        std::fprintf(stderr, "B-spline wrapper failed: %s\n", wrapper.error().c_str());
        std::exit(1);
    }
    return std::move(*wrapper);
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
// Diagnostic: dual price + IV error computation
// ============================================================================

struct DualError {
    double price_err;  // dollars
    double iv_err;     // bps
};
using DualErrorTable = std::array<std::array<DualError, kNS>, kNT>;

template <typename PriceFn>
static DualErrorTable compute_dual_errors(const PriceGrid& prices,
                                           PriceFn&& price_fn,
                                           size_t vol_idx,
                                           double div_yield = 0.0) {
    DualErrorTable errors{};
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < kNS; ++si) {
            double ref_price = prices[vol_idx][ti][si];
            if (std::isnan(ref_price) || ref_price <= 0) {
                errors[ti][si] = {std::nan(""), std::nan("")};
                continue;
            }

            // FDM reference IV
            IVQuery fdm_q;
            fdm_q.spot = kSpot;
            fdm_q.strike = kStrikes[si];
            fdm_q.maturity = kMaturities[ti];
            fdm_q.rate = kRate;
            fdm_q.dividend_yield = div_yield;
            fdm_q.option_type = OptionType::PUT;
            fdm_q.market_price = ref_price;

            auto fdm_result = fdm_solver.solve(fdm_q);
            if (!fdm_result) {
                errors[ti][si] = {std::nan(""), std::nan("")};
                continue;
            }
            double ref_iv = fdm_result->implied_vol;

            // Price error: evaluate interpolant at reference IV
            double interp_price = price_fn(kSpot, kStrikes[si],
                                            kMaturities[ti], ref_iv, kRate);
            double price_err = std::abs(interp_price - ref_price);

            // IV error: Brent inversion
            double interp_iv = brent_solve_iv(
                [&](double vol) -> double {
                    return price_fn(kSpot, kStrikes[si],
                                     kMaturities[ti], vol, kRate);
                },
                ref_price);

            double iv_err = std::isfinite(interp_iv)
                ? std::abs(interp_iv - ref_iv) * 10000.0
                : std::nan("");

            errors[ti][si] = {price_err, iv_err};
        }
    }
    return errors;
}

static void print_dual_heatmap(const char* title, const DualErrorTable& errors) {
    std::printf("\n=== %s ===\n", title);
    std::printf("  Format: $price_err / iv_bps\n\n");

    // Header
    std::printf("          ");
    for (size_t si = 0; si < kNS; ++si) {
        std::printf("    K=%-3.0f     ", kStrikes[si]);
    }
    std::printf("\n");

    for (size_t ti = 0; ti < kNT; ++ti) {
        std::printf("  T=%s  ", kMatLabels[ti]);
        for (size_t si = 0; si < kNS; ++si) {
            auto [pe, ie] = errors[ti][si];
            if (std::isnan(pe) && std::isnan(ie)) {
                std::printf("     ---      ");
            } else if (std::isnan(ie)) {
                std::printf(" $%5.3f/---   ", pe);
            } else {
                std::printf(" $%5.3f/%4.0f  ", pe, ie);
            }
        }
        std::printf("\n");
    }

    // Summary: median vega amplification ratio where both are valid
    std::vector<double> ratios;
    for (size_t ti = 0; ti < kNT; ++ti)
        for (size_t si = 0; si < kNS; ++si) {
            auto [pe, ie] = errors[ti][si];
            if (std::isfinite(pe) && std::isfinite(ie) && pe > 1e-6)
                ratios.push_back(ie / (pe * 10000.0));  // bps per dollar
        }
    if (!ratios.empty()) {
        std::sort(ratios.begin(), ratios.end());
        double median = ratios[ratios.size() / 2];
        std::printf("\n  Median IV/price amplification: %.0f bps per $0.01 price error\n",
                    median * 100.0);
    }
}

// ============================================================================
// Section runners
// ============================================================================

static void run_vanilla() {
    std::printf("\n--- Generating vanilla reference prices (batch chain solver)...\n");
    auto prices = generate_prices(/*with_dividends=*/false);

    std::printf("--- Building vanilla interpolated solver (adaptive)...\n");
    auto solver = build_vanilla_solver();

    std::printf("--- Computing vanilla IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Interpolation IV Error (bps) — σ=%.0f%%, no dividends",
                      kVols[vi] * 100);
        auto errors = compute_errors_vanilla(prices, solver, vi);
        print_heatmap(title, errors);
    }
}

static void run_dividends() {
    std::printf("\n--- Generating dividend reference prices (batch solver)...\n");
    auto prices = generate_prices(/*with_dividends=*/true);

    std::printf("--- Building dividend interpolated solvers (per-maturity)...\n");
    auto solvers = build_div_solvers();

    std::printf("--- Computing dividend IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Interpolation IV Error (bps) — σ=%.0f%%, quarterly $0.50 div",
                      kVols[vi] * 100);
        auto errors = compute_errors_div(prices, solvers, vi);
        print_heatmap(title, errors);
    }
}

static const PriceGrid& get_q0_prices() {
    static PriceGrid prices = [] {
        std::printf("--- Generating q=0 reference prices...\n");
        return generate_prices_q0();
    }();
    return prices;
}

static void run_bspline_3d() {
    std::printf("\n================================================================\n");
    std::printf("B-spline 3D Dimensionless — q=0, no dividends\n");
    std::printf("================================================================\n\n");

    const auto& q0_prices = get_q0_prices();

    std::printf("--- Building 3D dimensionless surface...\n");
    auto inner_3d = build_3d_surface();

    std::printf("--- Computing errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "3D Dimensionless IV Error (bps) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);
        auto errors = compute_errors_3d(q0_prices, inner_3d, vi);
        print_heatmap(title, errors);
    }
}

static void run_bspline_4d() {
    std::printf("\n================================================================\n");
    std::printf("B-spline 4D Standard (Brent) — q=0, no dividends\n");
    std::printf("================================================================\n\n");

    const auto& q0_prices = get_q0_prices();

    std::printf("--- Building 4D standard surface (q=0)...\n");
    auto wrapper = build_bspline_4d_wrapper();

    std::printf("--- Computing errors (Brent)...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "4D Standard IV Error (bps, Brent) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);
        auto errors = compute_errors_brent(q0_prices,
            [&](double S, double K, double tau, double sigma, double r) {
                return wrapper.price(S, K, tau, sigma, r);
            }, vi);
        print_heatmap(title, errors);
    }
}

static void run_cheb3d() {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev-Tucker 3D — q=0, no dividends\n");
    std::printf("================================================================\n\n");

    const auto& q0_prices = get_q0_prices();

    std::printf("--- Building Chebyshev-Tucker 3D surface...\n");
    auto inner = build_chebyshev_3d_surface();

    std::printf("--- Computing Chebyshev 3D IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Chebyshev-Tucker 3D IV Error (bps) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);
        auto errors = compute_errors_chebyshev(q0_prices, inner, vi);
        print_heatmap(title, errors);
    }
}

static void run_cheb4d() {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev-Tucker 4D (Brent) — q=0, no dividends\n");
    std::printf("================================================================\n\n");

    const auto& q0_prices = get_q0_prices();

    std::printf("--- Building Chebyshev-Tucker 4D surface...\n");
    auto inner = build_chebyshev_4d_surface();

    std::printf("--- Computing Chebyshev 4D IV errors (Brent)...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Chebyshev-Tucker 4D IV Error (bps, Brent) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);
        auto errors = compute_errors_brent(q0_prices,
            [&](double S, double K, double tau, double sigma, double r) {
                PriceQuery q{.spot = S, .strike = K, .tau = tau,
                             .sigma = sigma, .rate = r};
                return inner.price(q);
            }, vi);
        print_heatmap(title, errors);
    }
}

static void run_cheb4d_diag() {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev 4D Diagnostics — price vs IV error, smooth floor\n");
    std::printf("================================================================\n\n");

    const auto& q0_prices = get_q0_prices();

    // --- Diagnostic A: dual price + IV error heatmap (hard max floor) ---
    std::printf("--- [A] Building Chebyshev 4D (hard max floor)...\n");
    auto inner = build_chebyshev_4d_surface();

    auto price_fn = [&](double S, double K, double tau, double sigma, double r) {
        PriceQuery q{.spot = S, .strike = K, .tau = tau,
                     .sigma = sigma, .rate = r};
        return inner.price(q);
    };

    std::printf("--- [A] Computing dual price + IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Price($) vs IV(bps) — σ=%.0f%%, hard max floor",
                      kVols[vi] * 100);
        auto dual = compute_dual_errors(q0_prices, price_fn, vi);
        print_dual_heatmap(title, dual);
    }

    // --- Diagnostic B: softplus-only floor (no hard max) ---
    std::printf("\n--- [B] Building Chebyshev 4D (smooth softplus-only floor)...\n");

    Chebyshev4DEEPConfig cfg_smooth;
    cfg_smooth.num_x = 40;
    cfg_smooth.num_tau = 15;
    cfg_smooth.use_hard_max = false;
    cfg_smooth.use_sinh_x = false;
    cfg_smooth.use_sqrt_tau = false;
    cfg_smooth.use_log_eep = false;

    auto result_smooth = build_chebyshev_4d_eep(cfg_smooth, kSpot, OptionType::PUT);
    auto ranks = result_smooth.interp.ranks();
    std::printf("  Smooth floor: %d PDE, ranks=(%zu,%zu,%zu,%zu)\n",
                result_smooth.n_pde_solves,
                ranks[0], ranks[1], ranks[2], ranks[3]);

    Chebyshev4DEEPInner inner_smooth(
        std::move(result_smooth.interp), OptionType::PUT, kSpot, 0.0,
        result_smooth.transforms);

    std::printf("--- [B] Computing IV errors (smooth floor)...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Chebyshev 4D IV Error (bps) — σ=%.0f%%, smooth softplus-only",
                      kVols[vi] * 100);
        auto errors = compute_errors_brent(q0_prices,
            [&](double S, double K, double tau, double sigma, double r) {
                PriceQuery q{.spot = S, .strike = K, .tau = tau,
                             .sigma = sigma, .rate = r};
                return inner_smooth.price(q);
            }, vi);
        print_heatmap(title, errors);
    }
}

static PiecewiseChebyshev4DEEPInner build_piecewise_chebyshev_4d_surface() {
    PiecewiseChebyshev4DConfig cfg;
    // Default breaks: [-0.50, -0.10, 0.15, 0.40], 15 nodes/seg

    auto t0 = std::chrono::steady_clock::now();
    auto result = build_piecewise_chebyshev_4d_eep(cfg, kSpot, OptionType::PUT);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::printf("  Piecewise 4D: %d PDE solves, %zu segments, %.3fs build\n",
                result.n_pde_solves, result.segments.size(), elapsed);
    for (size_t s = 0; s < result.segments.size(); ++s) {
        auto r = result.segments[s].ranks();
        std::printf("    seg %zu [%.2f, %.2f]: ranks=(%zu,%zu,%zu,%zu)\n",
                    s, result.x_bounds[s], result.x_bounds[s + 1],
                    r[0], r[1], r[2], r[3]);
    }

    return PiecewiseChebyshev4DEEPInner(
        std::move(result.segments), std::move(result.x_bounds),
        OptionType::PUT, kSpot, 0.0);
}

static void run_cheb4d_pw() {
    std::printf("\n================================================================\n");
    std::printf("Piecewise Chebyshev 4D (Brent) — q=0, no dividends\n");
    std::printf("================================================================\n\n");

    const auto& q0_prices = get_q0_prices();

    std::printf("--- Building Piecewise Chebyshev 4D surface...\n");
    auto inner = build_piecewise_chebyshev_4d_surface();

    std::printf("--- Computing IV errors (Brent)...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Piecewise Cheb 4D IV Error (bps, Brent) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);
        auto errors = compute_errors_brent(q0_prices,
            [&](double S, double K, double tau, double sigma, double r) {
                PriceQuery q{.spot = S, .strike = K, .tau = tau,
                             .sigma = sigma, .rate = r};
                return inner.price(q);
            }, vi);
        print_heatmap(title, errors);
    }
}

// ============================================================================
// Main
// ============================================================================

static constexpr const char* kSections[] = {
    "vanilla", "dividends", "bspline-3d", "bspline-4d", "cheb3d", "cheb4d",
    "cheb4d-diag", "cheb4d-pw"
};

int main(int argc, char* argv[]) {
    // Parse section names from argv. No args = print help.
    std::unordered_set<std::string> selected;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            selected.clear();
            break;
        }
        selected.insert(argv[i]);
    }
    if (selected.empty()) {
        std::printf("Usage: %s <section> [section...]\n\nSections:\n", argv[0]);
        for (const char* s : kSections) std::printf("  %s\n", s);
        std::printf("\nSpecify one or more sections to run.\n");
        return 0;
    }
    auto want = [&](const char* name) { return selected.count(name) > 0; };

    std::printf("Interpolation IV Safety Diagnostic\n");
    std::printf("===================================\n");
    std::printf("S=%.0f, r=%.2f, q=%.2f, PUT\n", kSpot, kRate, kDivYield);
    std::printf("Error = |interp_iv - fdm_iv| in basis points\n\n");

    std::printf("Strikes: ");
    for (double K : kStrikes) std::printf("%.0f ", K);
    std::printf("\nMaturities: ");
    for (size_t i = 0; i < kNT; ++i) std::printf("%s ", kMatLabels[i]);
    std::printf("\nVols: ");
    for (double v : kVols) std::printf("%.0f%% ", v * 100);
    std::printf("\n");

    if (want("vanilla"))    run_vanilla();
    if (want("dividends"))  run_dividends();
    if (want("bspline-3d")) run_bspline_3d();
    if (want("bspline-4d")) run_bspline_4d();
    if (want("cheb3d"))      run_cheb3d();
    if (want("cheb4d"))      run_cheb4d();
    if (want("cheb4d-diag")) run_cheb4d_diag();
    if (want("cheb4d-pw"))   run_cheb4d_pw();

    return 0;
}
