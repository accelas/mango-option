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
 * Paths (--path=...): all, bspline, chebyshev, q0, dividends, kref.
 *
 * The dividends path prices two discrete-dividend reference grids: the
 * B-spline rows use a per-maturity schedule on a fixed quarterly $0.50
 * calendar (kQuarterlyPerMaturity), the Chebyshev rows use a single 1y
 * build rolled forward to each shorter maturity (kRolledFrom1y). Grid,
 * K_refs and target IV error come from documented_adaptive_dividend_config()
 * in tests/iv_solver_factory_slow_test.cc (see kDoc* below); the yield and
 * the quarterly schedule itself are the benchmark's own.
 *
 * The kref path (--path=kref) sweeps the manual segmented B-spline's K_ref
 * spacing against a same-query blend-policy control, reporting the
 * MultiKRefSplit blend's own IV-equivalent error apart from the surface's.
 * See docs/API_GUIDE.md, "Measured K_ref spacing baseline", for the
 * committed run's numbers and their conditions.
 *
 * Run with: bazel run //benchmarks:interp_iv_safety
 */

#include "iv_benchmark_common.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/dividend_utils.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using namespace mango;
using namespace mango::bench;

// ============================================================================
// Failure reporting (D3): every build/wrap failure prints its error code
// and marks the run as failed so main() can exit non-zero.
// ============================================================================

static bool g_build_failed = false;

static const char* code_name(PriceTableErrorCode c) {
    switch (c) {
        case PriceTableErrorCode::InvalidConfig:          return "InvalidConfig";
        case PriceTableErrorCode::InsufficientGridPoints: return "InsufficientGridPoints";
        case PriceTableErrorCode::GridNotSorted:          return "GridNotSorted";
        case PriceTableErrorCode::NonPositiveValue:       return "NonPositiveValue";
        case PriceTableErrorCode::EmptyBatch:             return "EmptyBatch";
        case PriceTableErrorCode::ExtractionFailed:       return "ExtractionFailed";
        case PriceTableErrorCode::RepairFailed:           return "RepairFailed";
        case PriceTableErrorCode::FittingFailed:          return "FittingFailed";
        case PriceTableErrorCode::SurfaceBuildFailed:     return "SurfaceBuildFailed";
        case PriceTableErrorCode::SerializationFailed:    return "SerializationFailed";
        case PriceTableErrorCode::ArenaAllocationFailed:  return "ArenaAllocationFailed";
        case PriceTableErrorCode::TensorCreationFailed:   return "TensorCreationFailed";
        case PriceTableErrorCode::ValidationFailed:       return "ValidationFailed";
        case PriceTableErrorCode::NoViableSurface:        return "NoViableSurface";
    }
    return "?";  // unreachable; -Wswitch flags a new enumerator above
}

// Both helpers print to stderr and stdout: stderr surfaces the failure to a
// terminal immediately (even if stdout is buffered or the run is piped
// through `tee`), while stdout keeps it inline with the rest of the run's
// output so a redirected transcript (e.g. `... > log.txt 2>&1`) still shows
// the failure next to the table it broke, instead of only in a separate
// stderr capture that may not be kept.
static void report_build_failure(const char* what, const PriceTableError& e) {
    g_build_failed = true;
    std::fprintf(stderr, "  [FAILED] %s: %s (axis=%zu count=%zu)\n",
                 what, code_name(e.code), e.axis_index, e.count);
    std::printf("  [FAILED] %s: %s (axis=%zu count=%zu)\n",
                what, code_name(e.code), e.axis_index, e.count);
}

static void report_wrap_failure(const char* what, const ValidationError& e) {
    g_build_failed = true;
    std::fprintf(stderr, "  [FAILED] %s: ValidationErrorCode %d\n",
                 what, static_cast<int>(e.code));
    std::printf("  [FAILED] %s: ValidationErrorCode %d\n",
                what, static_cast<int>(e.code));
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
// Generic per-path containers (strike count is a template parameter so the
// dividends path can carry its own strike set)
// ============================================================================

// prices[vol_idx][mat_idx][strike_idx]
template <size_t NS>
using PriceGridN = std::array<std::array<std::array<double, NS>, kNT>, kNV>;

// errors[mat_idx][strike_idx] in bps (NaN for failed cases)
template <size_t NS>
using ErrorTableN = std::array<std::array<double, NS>, kNT>;

template <size_t NS>
using TVKMaskN = std::array<std::array<bool, NS>, kNT>;

using PriceGrid = PriceGridN<kNS>;    // vanilla/q0 keep these aliases
using ErrorTable = ErrorTableN<kNS>;

/// Maturity -> schedule for that contract; nullopt = maturity not covered
/// by this path (its prices are NaN).
using ScheduleFn = std::function<std::optional<std::vector<Dividend>>(double maturity)>;

// ============================================================================
// Step 1: Generate reference prices via batch chain solver
// ============================================================================

template <size_t NS>
static PriceGridN<NS> generate_prices(const std::array<double, NS>& strikes,
                                      const ScheduleFn& schedule,
                                      double div_yield) {
    PriceGridN<NS> prices{};
    for (auto& v : prices) for (auto& t : v) t.fill(std::nan(""));

    BatchAmericanOptionSolver batch_solver;
    std::vector<PricingParams> all_params;
    std::vector<std::array<size_t, 3>> index;  // (vi, ti, si)
    all_params.reserve(kNV * kNT * NS);

    for (size_t vi = 0; vi < kNV; ++vi) {
        for (size_t ti = 0; ti < kNT; ++ti) {
            auto divs = schedule(kMaturities[ti]);
            if (!divs) continue;  // maturity not covered: row stays NaN
            for (size_t si = 0; si < NS; ++si) {
                PricingParams p;
                p.spot = kSpot;
                p.strike = strikes[si];
                p.maturity = kMaturities[ti];
                p.rate = kRate;
                p.dividend_yield = div_yield;
                p.option_type = OptionType::PUT;
                p.volatility = kVols[vi];
                p.discrete_dividends = *divs;
                all_params.push_back(std::move(p));
                index.push_back({vi, ti, si});
            }
        }
    }

    auto result = batch_solver.solve_batch(all_params, /*use_shared_grid=*/true);
    for (size_t i = 0; i < index.size(); ++i) {
        auto [vi, ti, si] = index[i];
        if (result.results[i].has_value())
            prices[vi][ti][si] = result.results[i]->value();
    }
    return prices;
}

static const ScheduleFn kNoDividends = [](double) {
    return std::optional<std::vector<Dividend>>{std::vector<Dividend>{}};
};

// ============================================================================
// Dividends path: the documented adaptive discrete-dividend configuration.
// Grid, K_refs and target are verbatim from documented_adaptive_dividend_config()
// in tests/iv_solver_factory_slow_test.cc (the nightly pin); if that helper
// changes, change these too. The yield (kDivYield) and the schedule
// (quarterly_div_schedule) are the benchmark's own.
// ============================================================================
static constexpr std::array<double, 7> kDivStrikes = {
    93.0, 95.0, 97.5, 100.0, 102.5, 105.0, 107.0};   // all inside S/K in [0.92, 1.08]
static constexpr size_t kNDS = kDivStrikes.size();
static const std::vector<double> kDocMoneyness = {0.92, 0.95, 1.0, 1.05, 1.08};
static const std::vector<double> kDocVols      = {0.10, 0.15, 0.20, 0.30};
static const std::vector<double> kDocRates     = {0.02, 0.03, 0.05, 0.07};
static const std::vector<double> kDocKRefs     = {90.0, 92.5, 95.0, 97.5, 100.0,
                                                  102.5, 105.0, 107.5, 110.0};
static constexpr double kDocTargetIVError = 1e-3;

static std::vector<double> doc_log_moneyness() {
    std::vector<double> out;
    for (double m : kDocMoneyness) out.push_back(std::log(m));
    return out;
}

static const ScheduleFn kQuarterlyPerMaturity = [](double T) {
    return std::optional{quarterly_div_schedule(T)};
};
static const ScheduleFn kRolledFrom1y = [](double T) -> std::optional<std::vector<Dividend>> {
    if (T > 1.0 + 1e-9) return std::nullopt;
    return rolled_dividends(quarterly_div_schedule(1.0), 1.0, T);
};

/// Per-row dividend-count labels alongside a per-row coverage flag, so
/// print_heatmap can exclude schedule-uncovered rows from its aggregate
/// (Minor 3) without string-matching the "(not covered)" text.
struct DividendRowLabels {
    std::array<std::string, kNT> text;
    std::array<bool, kNT> covered;
};

static DividendRowLabels dividend_count_labels(const ScheduleFn& schedule) {
    DividendRowLabels out{};
    for (size_t ti = 0; ti < kNT; ++ti) {
        auto d = schedule(kMaturities[ti]);
        out.covered[ti] = d.has_value();
        out.text[ti] = d ? "(" + std::to_string(d->size()) + " div)" : "(not covered)";
    }
    return out;
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
        report_wrap_failure("vanilla interpolated solver", solver.error());
        std::exit(1);
    }
    return std::move(*solver);
}

using BSplineDivSolver = InterpolatedIVSolver<BSplineMultiKRefSurface>;

// Dividends: one solver per maturity via BSpline + discrete dividends + adaptive grid.
// Uses the lower-level builder directly to capture convergence stats. Grid,
// K_refs and target are the documented adaptive discrete-dividend config
// (see kDoc* above); the schedule is a fixed quarterly $0.50 calendar.
static std::vector<std::pair<size_t, BSplineDivSolver>> build_div_solvers() {
    std::vector<std::pair<size_t, BSplineDivSolver>> solvers;
    const auto log_m = doc_log_moneyness();
    const AdaptiveGridParams adaptive{.target_iv_error = kDocTargetIVError};

    for (size_t ti = 0; ti < kNT; ++ti) {
        const double mat = kMaturities[ti];
        auto divs = quarterly_div_schedule(mat);
        char label[48];
        std::snprintf(label, sizeof(label), "B-spline dividends T=%s", kMatLabels[ti]);

        SegmentedAdaptiveConfig seg_config{
            .spot = kSpot,
            .option_type = OptionType::PUT,
            .dividend_yield = kDivYield,
            .discrete_dividends = divs,
            .maturity = mat,
            .kref_config = {.K_refs = kDocKRefs},
        };
        auto result = build_adaptive_bspline_segmented(
            adaptive, seg_config, {log_m, kDocVols, kDocRates});
        if (!result.has_value()) { report_build_failure(label, result.error()); continue; }

        std::printf("  T=%s (%zu div): iters=%zu target_met=%s max_err=%.1f bps "
                    "avg_err=%.1f bps measured=%zu PDE=%zu%s\n",
                    kMatLabels[ti], divs.size(), result->iterations.size(),
                    result->target_met ? "yes" : "no",
                    result->achieved_max_error * 1e4, result->achieved_avg_error * 1e4,
                    result->diagnostics.holdout_points_measured,
                    result->total_pde_solves, result->used_retry ? " (retry)" : "");

        // Published bounds are the builder's measured sample domain (spec D2
        // of #454), not the input arrays. In particular sample_bounds.tau_max
        // == mat here: expand_segmented_domain (adaptive_refinement.cpp)
        // clamps max_tau to the requested maturity, and the solver's own
        // schedule validation rolls query dividends from tau_max, so this
        // wrapper's roll point always lines up with the contract it was
        // built for.
        auto wrapper = BSplineMultiKRefSurface(
            std::move(result->surface), result->sample_bounds, OptionType::PUT, kDivYield);
        auto solver = BSplineDivSolver::create(std::move(wrapper), {}, divs);
        if (!solver.has_value()) { report_wrap_failure(label, solver.error()); continue; }
        solvers.emplace_back(ti, std::move(*solver));
    }
    std::printf("  built %zu/%zu per-maturity solvers:", solvers.size(), kNT);
    for (const auto& [ti, _] : solvers) std::printf(" %s", kMatLabels[ti]);
    std::printf("\n");
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

/// Per-table failure-reason counts (spec D3 bullet 4). A NaN cell in a
/// dividends error table can hide any of three distinct causes — a bad
/// reference price/IV, the interpolated solver rejecting the query, or a
/// row no schedule ever covers — and printing "0.0" or a bare "---" makes
/// them indistinguishable. `attempted` counts cells in covered rows only;
/// `succeeded + ref_fail + inv_fail == attempted`; `uncovered` counts cells
/// in rows the schedule (or, for the B-spline path, the per-maturity
/// builder) never produced at all, so `attempted + uncovered` is the full
/// table population.
struct ErrorCounts {
    size_t attempted = 0, succeeded = 0, ref_fail = 0, inv_fail = 0, uncovered = 0;
    std::map<IVErrorCode, size_t> inv_fail_codes;
};

// Local mirror of mango::iv_error_message() (src/support/error_types.hpp),
// using a return-per-case switch rather than a switch-assigned local: GCC's
// -Wmaybe-uninitialized cannot prove the header's version is fully
// initialized once inlined into a std::map iteration under -O3, and that
// header must not change here. Falls back to the integer code, mirroring
// this file's own code_name() above, if a mapping is ever missing.
static const char* iv_error_code_name(IVErrorCode code) {
    switch (code) {
        case IVErrorCode::NegativeSpot:             return "NegativeSpot";
        case IVErrorCode::NegativeStrike:           return "NegativeStrike";
        case IVErrorCode::NegativeMaturity:         return "NegativeMaturity";
        case IVErrorCode::NegativeMarketPrice:      return "NegativeMarketPrice";
        case IVErrorCode::ArbitrageViolation:       return "ArbitrageViolation";
        case IVErrorCode::InvalidGridConfig:        return "InvalidGridConfig";
        case IVErrorCode::OptionTypeMismatch:       return "OptionTypeMismatch";
        case IVErrorCode::DividendYieldMismatch:    return "DividendYieldMismatch";
        case IVErrorCode::DiscreteDividendMismatch: return "DiscreteDividendMismatch";
        case IVErrorCode::MaxIterationsExceeded:    return "MaxIterationsExceeded";
        case IVErrorCode::BracketingFailed:         return "BracketingFailed";
        case IVErrorCode::NumericalInstability:     return "NumericalInstability";
        case IVErrorCode::VegaTooSmall:             return "VegaTooSmall";
        case IVErrorCode::PDESolveFailed:           return "PDESolveFailed";
        case IVErrorCode::MultipleRoots:            return "MultipleRoots";
    }
    return nullptr;  // unreachable; -Wswitch flags a new enumerator above
}

static void print_error_counts(const ErrorCounts& c) {
    std::printf("  counts: attempted=%zu succeeded=%zu ref-fail=%zu inv-fail=%zu uncovered=%zu\n",
                c.attempted, c.succeeded, c.ref_fail, c.inv_fail, c.uncovered);
    if (!c.inv_fail_codes.empty()) {
        std::printf("  inv-fail by code:");
        for (const auto& [code, n] : c.inv_fail_codes) {
            const char* name = iv_error_code_name(code);
            if (name) std::printf(" %s=%zu", name, n);
            else      std::printf(" %d=%zu", static_cast<int>(code), n);
        }
        std::printf("\n");
    }
}

template <size_t NS, typename Solver>
static std::pair<ErrorTableN<NS>, ErrorCounts> compute_errors_div(
    const PriceGridN<NS>& prices,
    const std::array<double, NS>& strikes,
    const std::vector<std::pair<size_t, Solver>>& div_solvers,
    size_t vol_idx) {
    ErrorTableN<NS> errors{};
    ErrorCounts counts;

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
        if (solver_idx[ti] < 0) { counts.uncovered += NS; continue; }  // no solver for this maturity

        double maturity = kMaturities[ti];
        auto divs = quarterly_div_schedule(maturity);
        const auto& solver = div_solvers[static_cast<size_t>(solver_idx[ti])].second;

        // Build queries for this maturity
        std::vector<IVQuery> queries;
        std::vector<size_t> strike_indices;

        for (size_t si = 0; si < NS; ++si) {
            counts.attempted++;
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) { counts.ref_fail++; continue; }

            IVQuery q;
            q.spot = kSpot;
            q.strike = strikes[si];
            q.maturity = maturity;
            q.rate = kRate;
            q.dividend_yield = kDivYield;
            q.option_type = OptionType::PUT;
            q.market_price = price;
            q.discrete_dividends = divs;
            queries.push_back(q);
            strike_indices.push_back(si);
        }

        // Batch interpolated IV
        auto interp_results = solver.solve_batch(queries);

        for (size_t i = 0; i < queries.size(); ++i) {
            size_t si = strike_indices[i];

            if (!interp_results.results[i].has_value()) {
                counts.inv_fail++;
                counts.inv_fail_codes[interp_results.results[i].error().code]++;
                continue;
            }

            // FDM reference: manual Brent with dividends
            double fdm_iv = solve_fdm_iv_div(
                strikes[si], maturity,
                prices[vol_idx][ti][si], divs);

            if (std::isnan(fdm_iv)) { counts.ref_fail++; continue; }

            double interp_iv = interp_results.results[i]->implied_vol;
            errors[ti][si] = std::abs(interp_iv - fdm_iv) * 10000.0;
            counts.succeeded++;
        }
    }

    return {errors, counts};
}

// ============================================================================
// Step 5: Print heatmap
// ============================================================================

template <size_t NS>
static void print_heatmap(const char* title, const std::array<double, NS>& strikes,
                          const ErrorTableN<NS>& errors,
                          const std::array<std::string, kNT>* row_suffix = nullptr,
                          const std::array<bool, kNT>* row_covered = nullptr) {
    std::printf("\n=== %s ===\n", title);
    std::printf("          ");
    for (double K : strikes) {
        // Integral strikes keep the historical "K=100" header; fractional
        // ones (dividends path) print one decimal.
        if (std::fmod(K, 1.0) == 0.0) std::printf("  K=%-3.0f ", K);
        else                          std::printf(" K=%-5.1f", K);
    }
    std::printf("\n");

    // Rows the schedule never covers (row_covered[ti] == false) carry no
    // data at all; counting their cells toward n_total/n_failed made an
    // entirely absent row look like a row of failed solves (Minor 3).
    size_t n_total = 0, n_failed = 0, n_uncovered = 0;
    double sum_sq = 0;
    for (size_t ti = 0; ti < kNT; ++ti) {
        bool covered = !row_covered || (*row_covered)[ti];
        std::printf("  T=%s  ", kMatLabels[ti]);
        for (size_t si = 0; si < NS; ++si) {
            double e = errors[ti][si];
            if (!covered) { std::printf("   ---  "); n_uncovered++; continue; }
            n_total++;
            if (std::isnan(e)) { std::printf("   ---  "); n_failed++; continue; }
            const char* marker = e > 200 ? "***" : e > 50 ? "**" : e > 10 ? "*" : "";
            std::printf("%6.1f%-3s", e, marker);
            sum_sq += e * e;
        }
        if (row_suffix) std::printf("  %s", (*row_suffix)[ti].c_str());
        std::printf("\n");
    }
    size_t n_valid = n_total - n_failed;
    std::printf("\n  Legend: * >10bps  ** >50bps  *** >200bps  --- solve failed\n");
    char uncovered_suffix[48] = "";
    if (n_uncovered > 0)
        std::snprintf(uncovered_suffix, sizeof(uncovered_suffix),
                      ", %zu cells not covered", n_uncovered);
    if (n_valid == 0)
        std::printf("  Overall RMS: n/a (0/%zu succeeded%s)\n", n_total, uncovered_suffix);
    else
        std::printf("  Overall RMS: %.1f bps (%zu/%zu succeeded%s)\n",
                    std::sqrt(sum_sq / n_valid), n_valid, n_total, uncovered_suffix);
}

// ============================================================================
// TV/K filtered stats — filter out low-vega edge cases
// ============================================================================

/// TV/K mask: which (maturity, strike) points survive a given threshold.
/// Based purely on reference prices so all algorithms share the same mask.
template <size_t NS>
static TVKMaskN<NS> compute_tvk_mask(const PriceGridN<NS>& prices,
                                     const std::array<double, NS>& strikes,
                                     size_t vol_idx, double threshold) {
    TVKMaskN<NS> mask{};
    for (size_t ti = 0; ti < kNT; ++ti)
        for (size_t si = 0; si < NS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) { mask[ti][si] = false; continue; }
            double intrinsic = std::max(strikes[si] - kSpot, 0.0);  // put
            mask[ti][si] = ((price - intrinsic) / strikes[si]) >= threshold;
        }
    return mask;
}

/// Print RMS error for multiple algorithms at a given TV/K threshold.
/// All algorithms are filtered by the SAME mask (from reference prices).
template <size_t NS>
struct AlgoErrorsN { const char* label; const ErrorTableN<NS>* errors; };  // null => n/a

template <size_t NS>
static void print_tvk_comparison(const PriceGridN<NS>& prices,
                                 const std::array<double, NS>& strikes,
                                 size_t vol_idx,
                                 std::span<const AlgoErrorsN<NS>> algos) {
    static constexpr double kThresholds[] = {0.0, 1e-4, 1e-3, 5e-3};
    static constexpr const char* kThreshLabels[] = {"none", "1e-4", "1e-3", "5e-3"};
    std::printf("\n  TV/K filtered RMS (σ=%.0f%%):\n", kVols[vol_idx] * 100);
    std::printf("  %-20s", "TV/K >=");
    for (const auto& a : algos) std::printf("  %14s", a.label);
    std::printf("\n");
    for (size_t fi = 0; fi < 4; ++fi) {
        auto mask = compute_tvk_mask(prices, strikes, vol_idx, kThresholds[fi]);
        size_t mask_count = 0;
        for (auto& row : mask) for (bool b : row) mask_count += b;
        std::printf("  %-12s [%2zu/%zu]", kThreshLabels[fi], mask_count, kNT * NS);
        for (const auto& a : algos) {
            if (!a.errors) { std::printf("  %14s", "n/a (no surf)"); continue; }
            double sum_sq = 0; size_t n = 0;
            for (size_t ti = 0; ti < kNT; ++ti)
                for (size_t si = 0; si < NS; ++si) {
                    if (!mask[ti][si]) continue;
                    double e = (*a.errors)[ti][si];
                    if (std::isnan(e)) continue;
                    sum_sq += e * e; n++;
                }
            char buf[32];
            if (n == 0) std::snprintf(buf, sizeof(buf), "n/a (0)");
            else        std::snprintf(buf, sizeof(buf), "%.1f (%zu)", std::sqrt(sum_sq / n), n);
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
        report_build_failure("Chebyshev 4D", result.error());
        std::exit(1);
    }

    std::printf("  PDE solves: %zu\n", result->n_pde_solves);
    std::printf("  Build time: %.2f s\n", result->build_seconds);

    return std::move(*result);
}

/// Generic error computation via any InterpolatedIVSolver.
/// The solver's built-in vega pre-check handles edge-case filtering.
template <size_t NS, typename Solver>
static ErrorTableN<NS> compute_errors_via_solver(
    const PriceGridN<NS>& prices,
    const std::array<double, NS>& strikes,
    const Solver& interp_solver,
    size_t vol_idx,
    double div_yield = kDivYield) {
    ErrorTableN<NS> errors{};
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    std::vector<IVQuery> queries;
    std::vector<std::pair<size_t, size_t>> query_map;
    queries.reserve(kNT * NS);

    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < NS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) {
                errors[ti][si] = std::nan("");
                continue;
            }

            IVQuery q;
            q.spot = kSpot;
            q.strike = strikes[si];
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
        report_wrap_failure("Chebyshev 4D solver", solver.error());
    } else {
        for (size_t vi = 0; vi < kNV; ++vi) {
            char title[128];
            std::snprintf(title, sizeof(title),
                          "Chebyshev 4D IV Error (bps) — σ=%.0f%%",
                          kVols[vi] * 100);
            all_errors[vi] = compute_errors_via_solver(prices, kStrikes, *solver, vi);
            print_heatmap(title, kStrikes, all_errors[vi]);
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
        report_build_failure("Chebyshev adaptive", result.error());
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
        report_wrap_failure("Chebyshev adaptive solver", solver.error());
        return {};
    }

    std::array<ErrorTable, kNV> all_errors{};
    std::printf("--- Computing adaptive Chebyshev IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Chebyshev Adaptive IV Error (bps) — σ=%.0f%%",
                      kVols[vi] * 100);
        all_errors[vi] = compute_errors_via_solver(prices, kStrikes, *solver, vi);
        print_heatmap(title, kStrikes, all_errors[vi]);
    }
    return all_errors;
}

// ============================================================================
// Chebyshev Adaptive — Discrete Dividends (segmented, no EEP)
// ============================================================================

static std::optional<std::array<ErrorTableN<kNDS>, kNV>>
run_chebyshev_dividends(const PriceGridN<kNDS>& prices) {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev Adaptive — Discrete Dividends (segmented)\n");
    std::printf("================================================================\n\n");

    AdaptiveGridParams params{.target_iv_error = kDocTargetIVError};
    const auto divs_1y = quarterly_div_schedule(1.0);
    SegmentedAdaptiveConfig config{
        .spot = kSpot,
        .option_type = OptionType::PUT,
        .dividend_yield = kDivYield,
        .discrete_dividends = divs_1y,
        .maturity = 1.0,
        .kref_config = {.K_refs = kDocKRefs},
    };
    IVGrid domain{.moneyness = doc_log_moneyness(), .vol = kDocVols, .rate = kDocRates};
    std::printf("--- Building segmented Chebyshev surface (documented config, "
                "target=%.1f bps, %zu div)...\n", params.target_iv_error * 1e4, divs_1y.size());
    auto result = build_adaptive_chebyshev_segmented(params, config, domain);
    if (!result.has_value()) {
        report_build_failure("Chebyshev dividends", result.error());
        return {};   // callers treat an empty optional/table as "no surface" (see main)
    }

    std::printf("  Iterations: %zu, PDE solves: %zu, target_met: %s, "
                "achieved max=%.1f bps avg=%.1f bps measured=%zu\n",
                result->iterations.size(), result->total_pde_solves,
                result->target_met ? "yes" : "no",
                result->achieved_max_error * 1e4, result->achieved_avg_error * 1e4,
                result->diagnostics.holdout_points_measured);
    for (const auto& it : result->iterations) {
        std::printf("  iter %zu: grid [%zu, %zu, %zu, %zu] "
                    "max_err=%.1f bps avg_err=%.1f bps\n",
                    it.iteration,
                    it.grid_sizes[0], it.grid_sizes[1],
                    it.grid_sizes[2], it.grid_sizes[3],
                    it.max_error * 1e4, it.avg_error * 1e4);
    }

    // Point diagnostic at T=1y (the surface's maturity) for both σ values
    std::printf("--- Diagnostic: surface vs FDM at T=1y (same dividends) ---\n");
    for (double sigma : {0.15, 0.30}) {
        std::printf("  σ=%.2f:\n", sigma);
        for (double K : kDivStrikes) {
            double surf = result->surface.price(kSpot, K, 1.0, sigma, kRate);
            PricingParams pp;
            pp.spot = kSpot; pp.strike = K; pp.maturity = 1.0;
            pp.rate = kRate; pp.dividend_yield = kDivYield;
            pp.option_type = OptionType::PUT; pp.volatility = sigma;
            pp.discrete_dividends = divs_1y;
            auto fdm = solve_american_option(pp);
            double ref = fdm.has_value() ? fdm->value() : -1.0;
            double pct_err = ref > 0.001 ? 100.0 * (surf - ref) / ref : 0.0;
            std::printf("    K=%5.1f: surf=%8.4f fdm=%8.4f diff=%+.4f (%.1f%%)\n",
                        K, surf, ref, surf - ref, pct_err);
        }
    }

    // Wrap in InterpolatedIVSolver for consistent vega pre-check. The
    // surface already carries its published sample bounds; it only needs
    // the build schedule for query-time roll validation.
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(
        std::move(result->surface), {}, divs_1y);
    if (!solver.has_value()) { report_wrap_failure("Chebyshev dividends", solver.error()); return {}; }

    // Compute IV errors at each maturity using dividend-aware FDM reference.
    // The surface was built for maturity=1.0 with quarterly_div_schedule(1.0);
    // each other maturity's schedule is that calendar rolled forward.
    std::array<ErrorTableN<kNDS>, kNV> all_errors{};
    std::printf("--- Computing Chebyshev dividend IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        auto& errors = all_errors[vi];
        for (auto& row : errors)
            for (auto& v : row)
                v = std::nan("");

        ErrorCounts counts;
        for (size_t ti = 0; ti < kNT; ++ti) {
            double tau = kMaturities[ti];
            auto rolled = kRolledFrom1y(tau);
            if (!rolled) { counts.uncovered += kNDS; continue; }   // T > 1: not covered
            for (size_t si = 0; si < kNDS; ++si) {
                counts.attempted++;
                double price = prices[vi][ti][si];       // from the rolled reference grid
                if (std::isnan(price) || price <= 0) { counts.ref_fail++; continue; }
                double fdm_iv = solve_fdm_iv_div(kDivStrikes[si], tau, price, *rolled);
                if (std::isnan(fdm_iv)) { counts.ref_fail++; continue; }
                IVQuery q;
                q.spot = kSpot;
                q.strike = kDivStrikes[si];
                q.maturity = tau;
                q.rate = kRate;
                q.dividend_yield = kDivYield;
                q.option_type = OptionType::PUT;
                q.market_price = price;
                q.discrete_dividends = *rolled;
                auto iv_result = solver->solve(q);
                if (!iv_result.has_value()) {
                    counts.inv_fail++;
                    counts.inv_fail_codes[iv_result.error().code]++;
                    continue;
                }
                errors[ti][si] = std::abs(iv_result->implied_vol - fdm_iv) * 10000.0;
                counts.succeeded++;
            }
        }

        char title[160];
        std::snprintf(title, sizeof(title),
                      "Cheb Dividend IV Error (bps) — σ=%.0f%%, 1y calendar rolled",
                      kVols[vi] * 100);
        auto labels = dividend_count_labels(kRolledFrom1y);
        print_heatmap(title, kDivStrikes, errors, &labels.text, &labels.covered);
        print_error_counts(counts);
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
        report_wrap_failure("4D B-spline (q=0)", solver.error());
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
// K_ref spacing sweep (--path=kref): blend policy vs surface, with a
// same-query FDM control. Spec: docs/superpowers/specs/2026-09-12-interp-iv-safety-dividends-462-design.md D4.
//
// Terms: an *anchor* is a strike equal to a K_ref; a *mid-anchor* is the
// midpoint between two adjacent K_refs; the *blend policy* is
// MultiKRefSplit (query each bracketing K_ref surface at (spot, K_ref),
// normalize by K_ref, interpolate linearly in strike, multiply by strike).
// ============================================================================
namespace kref {

constexpr double kSpanLo = 80.0, kSpanHi = 120.0;
constexpr double kWindowLo = 85.0, kWindowHi = 115.0;
constexpr std::array<double, 4> kSpacings = {10.0, 5.0, 2.5, 1.25};
constexpr std::array<double, 4> kSweepMaturities = {0.20, 0.30, 0.60, 1.0};
constexpr std::array<double, 2> kSweepVols = {0.15, 0.30};
constexpr double kVegaFloor = 1e-4;      // AdaptiveGridParams::vega_floor default
constexpr double kTVKThreshold = 1e-4;   // make_iv_score_fn's threshold
constexpr double kQualifyBps = 10.0;     // D5 classification threshold
constexpr size_t kBaseMoneynessKnots = 41;
constexpr int kBaseTauPoints = 5;

struct Ref { double price = 0.0, vega = 0.0; bool ok = false; };

/// FDM reference price and central-bump vega (same bump as
/// make_fd_vega_refs_fn: eps = max(1e-4, 0.01*sigma)). `accuracy` nullopt =
/// the solver's automatic grid; set = an explicit GridAccuracyParams.
static Ref fdm_ref(double K, double T, double sigma,
                   const std::vector<Dividend>& divs,
                   std::optional<GridAccuracyParams> accuracy) {
    auto price_at = [&](double sg) -> std::optional<double> {
        PricingParams p;
        p.spot = kSpot; p.strike = K; p.maturity = T; p.rate = kRate;
        p.dividend_yield = kDivYield; p.option_type = OptionType::PUT;
        p.volatility = sg; p.discrete_dividends = divs;
        std::optional<PDEGridSpec> grid;
        if (accuracy) grid = PDEGridSpec{*accuracy};
        auto solver = AmericanOptionSolver::create(p, grid);
        if (!solver) return std::nullopt;
        auto r = solver->solve();
        if (!r || !std::isfinite(r->value())) return std::nullopt;
        return r->value();
    };
    Ref out;
    const double eps = std::max(1e-4, 0.01 * sigma);
    const double sigma_dn = std::max(1e-4, sigma - eps);
    auto p0 = price_at(sigma);
    auto pu = price_at(sigma + eps);
    auto pd = price_at(sigma_dn);
    if (!p0 || !pu || !pd) return out;
    out.price = *p0;
    out.vega = (*pu - *pd) / ((sigma + eps) - sigma_dn);
    out.ok = std::isfinite(out.vega);
    return out;
}

static std::vector<double> krefs_for(double delta) {
    std::vector<double> ks;
    for (double k = kSpanLo; k <= kSpanHi + 1e-9; k += delta) ks.push_back(k);
    return ks;
}

/// Manual (non-adaptive) multi-K_ref segmented B-spline surface on fixed
/// input knots. Mirrors build_multi_kref_manual + manual_segmented_bounds in
/// src/option/price_table_factory.cpp.
static std::expected<BSplineMultiKRefSurface, PriceTableError>
build_manual(const std::vector<double>& krefs, double T, size_t n_m, int tau_pts) {
    std::vector<double> log_m(n_m);
    for (size_t i = 0; i < n_m; ++i)
        log_m[i] = -0.30 + 0.60 * static_cast<double>(i) / static_cast<double>(n_m - 1);
    const std::vector<double> vols  = {0.10, 0.15, 0.20, 0.30, 0.50};
    const std::vector<double> rates = {0.02, 0.03, 0.05, 0.07};  // builder needs >= 4 knots
    DividendSpec dividends{.dividend_yield = kDivYield,
                           .discrete_dividends = quarterly_div_schedule(T)};
    std::vector<BSplineMultiKRefEntry> entries;
    entries.reserve(krefs.size());
    for (double k : krefs) {
        SegmentedPriceTableBuilder::Config cfg{
            .K_ref = k, .option_type = OptionType::PUT, .dividends = dividends,
            .grid = IVGrid{.moneyness = log_m, .vol = vols, .rate = rates},
            .maturity = T, .tau_points_per_segment = tau_pts,
        };
        auto surface = SegmentedPriceTableBuilder::build(cfg);
        if (!surface) return std::unexpected(surface.error());
        entries.push_back({.K_ref = k, .surface = std::move(*surface)});
    }
    auto inner = build_multi_kref_surface(std::move(entries));
    if (!inner) return std::unexpected(inner.error());
    SurfaceBounds bounds{.m_min = -0.30, .m_max = 0.30, .tau_min = 0.0, .tau_max = T,
                         .sigma_min = vols.front(), .sigma_max = vols.back(),
                         .rate_min = rates.front(), .rate_max = rates.back()};
    return BSplineMultiKRefSurface(std::move(*inner), bounds, OptionType::PUT, kDivYield);
}

struct Query { double K; bool anchor; double L, H; };

static std::vector<Query> queries_for(const std::vector<double>& krefs) {
    std::vector<Query> qs;
    for (size_t i = 0; i < krefs.size(); ++i) {
        if (krefs[i] >= kWindowLo && krefs[i] <= kWindowHi)
            qs.push_back({krefs[i], true, krefs[i], krefs[i]});   // anchor: control = itself
        if (i + 1 < krefs.size()) {
            double mid = 0.5 * (krefs[i] + krefs[i + 1]);
            if (mid >= kWindowLo && mid <= kWindowHi)
                qs.push_back({mid, false, krefs[i], krefs[i + 1]});
        }
    }
    return qs;
}

/// The blend policy applied to exact prices: query each bracketing K_ref
/// surface at (spot, K_ref), normalize by K_ref, interpolate linearly in
/// strike, multiply by strike. `w` is 0 at an anchor, where L == H == K and
/// the expression collapses to that strike's own reference price.
static double blend_control(double K, double L, double H, double w,
                            double price_L, double price_H) {
    return K * ((1.0 - w) * price_L / L + w * price_H / H);
}

/// Accumulator for one (delta, T, sigma, anchor/mid) population.
struct Stat {
    size_t q = 0, elig = 0, ref_fail = 0, low_tv = 0, surf_nonfinite = 0;
    // The sweep's vega test is signed, which is stricter than make_iv_score_fn's
    // |vega| test only for negative FD vega; neg_vega counts that difference.
    size_t low_vega = 0, neg_vega = 0;
    double blend_max = 0, blend_sq = 0;   // |B_fdm - P_fdm| / vega, bps
    double blend_signed_sum = 0;          // signed (B_fdm - P_fdm) / vega, bps
    double surf_max = 0, surf_sq = 0;     // |P_hat - B_fdm| / vega, bps
    size_t surf_n = 0;
    size_t inv_n = 0, inv_fail = 0; double inv_max = 0;
    // ref-sens (mid-anchors only): fine_n counts eligible queries whose finer
    // references all solved; fine_skip the eligible ones they did not cover,
    // so a population mismatch against `elig` is visible rather than silent.
    double blend_max_fine = 0; size_t fine_n = 0, fine_skip = 0;
    bool complete() const { return elig >= 1 && 100 * elig >= 90 * q; }
    double blend_rms() const { return elig ? std::sqrt(blend_sq / elig) : std::nan(""); }
    double blend_mean() const { return elig ? blend_signed_sum / static_cast<double>(elig)
                                            : std::nan(""); }
    double surf_rms()  const { return surf_n ? std::sqrt(surf_sq / surf_n) : std::nan(""); }
};

static void fmt(char* buf, size_t n, double v) {
    if (std::isnan(v)) std::snprintf(buf, n, "%10s", "n/a");
    else std::snprintf(buf, n, "%10.2f", v);
}

/// Reference cache keyed by (T, sigma, K, fine): each (K, sigma, T) is solved
/// once per accuracy no matter how many spacings share it.
using RefKey = std::tuple<int, int, long, int>;
static Ref cached_ref(std::map<RefKey, Ref>& cache, double K, double T, double sigma,
                      const std::vector<Dividend>& divs, bool fine) {
    RefKey key{static_cast<int>(std::lround(T * 1e4)), static_cast<int>(std::lround(sigma * 1e4)),
               std::lround(K * 1e3), fine ? 1 : 0};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto r = fdm_ref(K, T, sigma, divs,
                     fine ? std::optional{make_grid_accuracy(GridAccuracyProfile::Ultra)} : std::nullopt);
    cache.emplace(key, r);
    return r;
}

struct RowResult { Stat mid, anchor; bool built = false; double seconds = 0; };

static RowResult run_row(double delta, double T, double sigma, size_t n_m, int tau_pts,
                         std::map<RefKey, Ref>& cache) {
    RowResult row;
    auto t0 = std::chrono::steady_clock::now();
    const auto divs = quarterly_div_schedule(T);
    const auto krefs = krefs_for(delta);
    auto surface = build_manual(krefs, T, n_m, tau_pts);
    if (!surface) {
        char what[64]; std::snprintf(what, sizeof(what), "kref sweep delta=%.2f T=%.2f", delta, T);
        report_build_failure(what, surface.error());
        return row;
    }
    // InterpolatedIVSolver keeps its surface private, so keep a copy for
    // direct pricing (PriceTable and SplitSurface are value types).
    const BSplineMultiKRefSurface surf = *surface;
    auto solver = InterpolatedIVSolver<BSplineMultiKRefSurface>::create(std::move(*surface), {}, divs);
    if (!solver) {
        char what[64]; std::snprintf(what, sizeof(what), "kref sweep delta=%.2f T=%.2f", delta, T);
        report_wrap_failure(what, solver.error());
        return row;
    }
    row.built = true;

    for (const Query& qy : queries_for(krefs)) {
        Stat& st = qy.anchor ? row.anchor : row.mid;
        st.q++;
        Ref rk = cached_ref(cache, qy.K, T, sigma, divs, false);
        Ref rl = qy.anchor ? rk : cached_ref(cache, qy.L, T, sigma, divs, false);
        Ref rh = qy.anchor ? rk : cached_ref(cache, qy.H, T, sigma, divs, false);
        if (!rk.ok || !rl.ok || !rh.ok) { st.ref_fail++; continue; }
        const double intrinsic = intrinsic_value(kSpot, qy.K, OptionType::PUT);
        if ((rk.price - intrinsic) / qy.K < kTVKThreshold) { st.low_tv++; continue; }
        if (rk.vega < 0.0) { st.neg_vega++; continue; }
        if (rk.vega < kVegaFloor) { st.low_vega++; continue; }
        st.elig++;

        // Anchors take w = 0 against their own reference: no (H - L) division.
        const double w = qy.anchor ? 0.0 : (qy.K - qy.L) / (qy.H - qy.L);
        const double b_fdm = blend_control(qy.K, qy.L, qy.H, w, rl.price, rh.price);
        const double blend_signed_bps = (b_fdm - rk.price) / rk.vega * 1e4;
        const double blend_bps = std::abs(blend_signed_bps);
        st.blend_max = std::max(st.blend_max, blend_bps);
        st.blend_sq += blend_bps * blend_bps;
        st.blend_signed_sum += blend_signed_bps;

        const double p_hat = surf.price(kSpot, qy.K, T, sigma, kRate);
        if (!std::isfinite(p_hat)) { st.surf_nonfinite++; }
        else {
            const double surf_bps = std::abs(p_hat - b_fdm) / rk.vega * 1e4;
            st.surf_max = std::max(st.surf_max, surf_bps);
            st.surf_sq += surf_bps * surf_bps; st.surf_n++;

            double iv_fdm = solve_fdm_iv_div(qy.K, T, rk.price, divs);
            IVQuery q; q.spot = kSpot; q.strike = qy.K; q.maturity = T; q.rate = kRate;
            q.dividend_yield = kDivYield; q.option_type = OptionType::PUT;
            q.market_price = rk.price; q.discrete_dividends = divs;
            auto iv = solver->solve(q);
            if (std::isnan(iv_fdm) || !iv) st.inv_fail++;
            else { st.inv_n++; st.inv_max = std::max(st.inv_max, std::abs(iv->implied_vol - iv_fdm) * 1e4); }
        }

        if (!qy.anchor) {   // ref-sens: same query, finer references
            Ref fk = cached_ref(cache, qy.K, T, sigma, divs, true);
            Ref fl = cached_ref(cache, qy.L, T, sigma, divs, true);
            Ref fh = cached_ref(cache, qy.H, T, sigma, divs, true);
            if (fk.ok && fl.ok && fh.ok && fk.vega >= kVegaFloor) {
                const double b_fine = blend_control(qy.K, qy.L, qy.H, w, fl.price, fh.price);
                st.blend_max_fine = std::max(st.blend_max_fine, std::abs(b_fine - fk.price) / fk.vega * 1e4);
                st.fine_n++;
            } else {
                st.fine_skip++;
            }
        }
    }
    row.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return row;
}

static void print_legend() {
    std::printf("  legend\n");
    std::printf("    Δ         K_ref spacing in dollars; K_refs are 80, 80+Δ, ..., 120\n");
    std::printf("    T         maturity in years (0.20 carries no dividend: the control row)\n");
    std::printf("    q/elig    queries in the window / the eligible subset (reference-only test:\n");
    std::printf("              finite P_FDM(K), P_FDM(L), P_FDM(H) and vega, TV/K >= %.0e, vega >= %.0e)\n",
                kTVKThreshold, kVegaFloor);
    std::printf("    blendmax  max |B_FDM - P_FDM| / vega_FDM in bps over the eligible queries,\n");
    std::printf("              B_FDM = K[(1-w)P_FDM(L)/L + w P_FDM(H)/H]: the blend policy's own error\n");
    std::printf("    blendmaxU the same maximum with every reference re-solved at\n");
    std::printf("              GridAccuracyProfile::Ultra, over the same mid-anchor population\n");
    std::printf("    blendmean signed mean of (B_FDM - P_FDM) / vega_FDM in bps over that population:\n");
    std::printf("              blendmax and blendrms are absolute, so this is where the sign shows\n");
    std::printf("    blendrms  rms of |B_FDM - P_FDM| / vega_FDM over the same population\n");
    std::printf("    surfmax   max |P_hat - B_FDM| / vega_FDM in bps: the surface's own error,\n");
    std::printf("    surfrms   and its rms; population = eligible queries with a finite P_hat\n");
    std::printf("    inv n     inversions where both IV_interp and IV_FDM converged\n");
    std::printf("    inv max   max |IV_interp - IV_FDM| in bps over that population\n");
    std::printf("    anchors   blendmax/blendrms are identically zero at an anchor (w = 0 against\n");
    std::printf("              its own reference), so only the surface and inversion columns print\n");
    std::printf("    ref-sens  |blendmaxU - blendmax|: an observed reference sensitivity,\n");
    std::printf("              not a discretization-error bound\n");
    std::printf("    status    pass/fail = mid-anchor blendmax vs %.0f bps; inconclusive = the gap to\n", kQualifyBps);
    std::printf("              %.0f bps is within ref-sens, or any eligible mid-anchor lacks a usable\n", kQualifyBps);
    std::printf("              Ultra reference (ref-sens-skip > 0); incomplete = eligible mid-anchors < 90%% of q\n");
    std::printf("    n/a       an empty population\n");
    std::printf("  exclusions (printed under any row that has them): ref-fail = a reference solve\n");
    std::printf("    failed or returned a non-finite value; low-tv = TV/K below the threshold;\n");
    std::printf("    low-vega = 0 <= vega < floor; neg-vega = vega < 0 (the sweep's signed test is\n");
    std::printf("    stricter than make_iv_score_fn's |vega| test only here); surf-nonfinite = the\n");
    std::printf("    surface returned a non-finite price; inv-fail = an inversion did not converge;\n");
    std::printf("    ref-sens-skip = an eligible mid-anchor whose Ultra references were unusable.\n");
}

static void print_header() {
    std::printf("  %15s%s   %s\n", "", "---------------------------------------- mid-anchors ----------------------------------------", "-------------------- anchors --------------------");
    std::printf("  %6s %5s | %4s %5s %10s %10s %10s %10s %10s %10s %5s %10s"
                " | %4s %5s %10s %10s %5s %10s | %10s %-12s %7s\n",
                "\u0394", "T", "q", "elig", "blendmax", "blendmaxU", "blendmean", "blendrms",
                "surfmax", "surfrms", "inv n", "inv max",
                "q", "elig", "surfmax", "surfrms", "inv n", "inv max",
                "ref-sens", "status", "time");
}

static void print_row(const char* delta_label, const char* t_label, const RowResult& r) {
    char b1[16], b1u[16], b1m[16], b2[16], b3[16], b4[16], b5[16], b6[16], b7[16], b8[16], b9[16];
    if (!r.built) {
        std::printf("  %6s %5s | (no surface)\n", delta_label, t_label);
        return;
    }
    const Stat& m = r.mid; const Stat& a = r.anchor;
    fmt(b1, 16, m.elig ? m.blend_max : std::nan(""));
    fmt(b1u, 16, m.fine_n ? m.blend_max_fine : std::nan(""));
    fmt(b1m, 16, m.blend_mean());
    fmt(b2, 16, m.blend_rms());
    fmt(b3, 16, m.surf_n ? m.surf_max : std::nan("")); fmt(b4, 16, m.surf_rms());
    fmt(b5, 16, m.inv_n ? m.inv_max : std::nan(""));
    fmt(b6, 16, a.surf_n ? a.surf_max : std::nan("")); fmt(b7, 16, a.surf_rms());
    fmt(b8, 16, a.inv_n ? a.inv_max : std::nan(""));
    // The sensitivity is only meaningful over the identical population: if
    // any eligible mid-anchor lacked a usable Ultra reference (fine_skip > 0)
    // the two maxima describe different query sets, so the row cannot be
    // classified and is reported inconclusive.
    const double sens = (m.elig && m.fine_n == m.elig)
        ? std::abs(m.blend_max_fine - m.blend_max) : std::nan("");
    fmt(b9, 16, sens);
    const char* status = !m.complete() ? "incomplete"
        : (std::isnan(sens) || std::abs(m.blend_max - kQualifyBps) <= sens) ? "inconclusive"
        : (m.blend_max <= kQualifyBps ? "pass" : "fail");
    std::printf("  %6s %5s | %4zu %5zu %s %s %s %s %s %s %5zu %s"
                " | %4zu %5zu %s %s %5zu %s | %s %-12s %6.0fs\n",
                delta_label, t_label, m.q, m.elig, b1, b1u, b1m, b2, b3, b4, m.inv_n, b5,
                a.q, a.elig, b6, b7, a.inv_n, b8, b9, status, r.seconds);
    if (m.ref_fail || m.low_vega || m.neg_vega || m.low_tv || m.surf_nonfinite || m.inv_fail ||
        m.fine_skip ||
        a.ref_fail || a.low_vega || a.neg_vega || a.low_tv || a.surf_nonfinite || a.inv_fail)
        std::printf("         excluded: mid ref-fail=%zu low-tv=%zu low-vega=%zu neg-vega=%zu"
                    " surf-nonfinite=%zu inv-fail=%zu ref-sens-skip=%zu"
                    " | anchor ref-fail=%zu low-tv=%zu low-vega=%zu neg-vega=%zu"
                    " surf-nonfinite=%zu inv-fail=%zu\n",
                    m.ref_fail, m.low_tv, m.low_vega, m.neg_vega, m.surf_nonfinite, m.inv_fail,
                    m.fine_skip,
                    a.ref_fail, a.low_tv, a.low_vega, a.neg_vega, a.surf_nonfinite, a.inv_fail);
}

}  // namespace kref

static void run_kref_sweep() {
    using namespace kref;
    const auto sweep_start = std::chrono::steady_clock::now();
    std::printf("\n================================================================\n");
    std::printf("K_ref spacing sweep — manual segmented B-spline, quarterly $0.50 calendar\n");
    std::printf("================================================================\n");
    std::printf("S=%.0f, PUT, r=%.0f%%, q=%.0f%%; window K in [%.0f, %.0f] inclusive;\n",
                kSpot, kRate * 100, kDivYield * 100, kWindowLo, kWindowHi);
    std::printf("surfaces: SegmentedPriceTableBuilder on %zu log-moneyness knots over [-0.30, 0.30],\n",
                kBaseMoneynessKnots);
    std::printf("vol {0.10, 0.15, 0.20, 0.30, 0.50}, rate {0.02, 0.03, 0.05, 0.07}, "
                "tau_points_per_segment=%d\n", kBaseTauPoints);
    std::printf("all error columns are IV-equivalent estimates in bps except 'inv', "
                "which is a true IV error\n\n");
    print_legend();
    std::map<RefKey, Ref> cache;
    for (double sigma : kSweepVols) {
        std::printf("\n  σ=%.0f%%\n", sigma * 100);
        print_header();
        for (double delta : kSpacings) {
            for (double T : kSweepMaturities) {
                char dl[8], tl[8];
                std::snprintf(dl, sizeof(dl), "%.2f", delta);
                std::snprintf(tl, sizeof(tl), "%.2f", T);
                auto row = run_row(delta, T, sigma, kBaseMoneynessKnots, kBaseTauPoints, cache);
                print_row(dl, tl, row);
            }
        }
        auto fine = run_row(2.5, 1.0, sigma, 81, 9, cache);
        print_row("fine", "1.00", fine);
        std::printf("  (fine = Δ 2.5, T 1.00 with 81 moneyness knots and 9 tau points per segment; "
                    "a surf max change > 2x\n   means the surface floor is not converged in those axes. "
                    "The blend columns do not depend on the surface.)\n");
    }
    const double total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sweep_start).count();
    std::printf("\n  K_ref sweep total: %.0f s (%.1f min)\n", total, total / 60.0);
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
    std::printf("\nDividend strikes: ");
    for (double K : kDivStrikes) std::printf("%.1f ", K);
    std::printf("\nMaturities: ");
    for (size_t i = 0; i < kNT; ++i) std::printf("%s ", kMatLabels[i]);
    std::printf("\nVols: ");
    for (double v : kVols) std::printf("%.0f%% ", v * 100);
    std::printf("\n");

    std::printf("Usage: interp_iv_safety [--path=all|bspline|chebyshev|q0|dividends|kref]\n\n");

    // Step 1: Generate reference prices (always needed)
    PriceGrid vanilla_prices{};
    bool need_vanilla = run_all || path == "bspline" || path == "chebyshev";
    bool need_divs = run_all || path == "dividends";
    bool need_q0 = run_all || path == "q0";
    bool need_kref = run_all || path == "kref";

    if (need_vanilla) {
        std::printf("--- Generating vanilla reference prices (batch chain solver)...\n");
        vanilla_prices = generate_prices(kStrikes, kNoDividends, kDivYield);
    }

    // Each dividends backend prices the contract it actually builds: the
    // B-spline path is one solver per maturity against that maturity's own
    // quarterly calendar, while the Chebyshev path is a single fixed-expiry
    // (1y) surface whose dividends are the 1y calendar rolled to each query
    // maturity. Sharing one price grid between them would validate the
    // wrong schedule for one of the two.
    PriceGridN<kNDS> bs_div_prices{}, cheb_div_prices{};
    if (need_divs) {
        std::printf("--- Generating dividend reference prices: per-maturity quarterly calendar (B-spline)...\n");
        bs_div_prices = generate_prices(kDivStrikes, kQuarterlyPerMaturity, kDivYield);
        std::printf("--- Generating dividend reference prices: 1y calendar rolled to each maturity (Chebyshev)...\n");
        cheb_div_prices = generate_prices(kDivStrikes, kRolledFrom1y, kDivYield);
    }

    // Per-path error tables
    std::array<ErrorTable, kNV> vanilla_errors{};
    std::array<ErrorTable, kNV> cheb_errors{};
    std::array<ErrorTable, kNV> cheb_adaptive_errors{};
    std::array<ErrorTable, kNV> q0_bs4d_errors{};
    std::array<ErrorTable, kNV> q0_dim3d_bs_errors{};
    std::array<ErrorTable, kNV> q0_dim3d_ch_errors{};
    std::optional<std::array<ErrorTableN<kNDS>, kNV>> div_errors, cheb_div_errors;

    // B-spline adaptive (vanilla)
    if (run_all || path == "bspline") {
        std::printf("--- Building vanilla interpolated solver (adaptive)...\n");
        auto vanilla_solver = build_vanilla_solver();

        std::printf("--- Computing vanilla IV errors...\n");
        for (size_t vi = 0; vi < kNV; ++vi) {
            char title[128];
            std::snprintf(title, sizeof(title),
                          "Interpolation IV Error (bps) — σ=%.0f%%, no dividends",
                          kVols[vi] * 100);
            vanilla_errors[vi] = compute_errors_via_solver(vanilla_prices, kStrikes, vanilla_solver, vi);
            print_heatmap(title, kStrikes, vanilla_errors[vi]);
        }
    }

    if (run_all || path == "dividends") {
        std::printf("--- Building dividend interpolated solvers (per-maturity)...\n");
        auto div_solvers = build_div_solvers();
        if (!div_solvers.empty()) {
            div_errors.emplace();
            auto labels = dividend_count_labels(kQuarterlyPerMaturity);
            std::printf("\n--- Computing dividend IV errors...\n");
            for (size_t vi = 0; vi < kNV; ++vi) {
                auto [errors, counts] = compute_errors_div(bs_div_prices, kDivStrikes, div_solvers, vi);
                (*div_errors)[vi] = errors;
                char title[160];
                std::snprintf(title, sizeof(title),
                    "Interpolation IV Error (bps) — σ=%.0f%%, quarterly $0.50 calendar (B-spline per-maturity)",
                    kVols[vi] * 100);
                print_heatmap(title, kDivStrikes, (*div_errors)[vi], &labels.text, &labels.covered);
                print_error_counts(counts);
            }
        }
        cheb_div_errors = run_chebyshev_dividends(cheb_div_prices);
    }

    // Chebyshev 4D
    if (run_all || path == "chebyshev") {
        cheb_errors = run_chebyshev_4d(vanilla_prices);
        cheb_adaptive_errors = run_chebyshev_adaptive(vanilla_prices);
    }

    // q=0 comparison: 4D B-spline vs dimensionless 3D (B-spline & Chebyshev)
    PriceGrid q0_prices{};
    if (need_q0) {
        std::printf("\n================================================================\n");
        std::printf("q=0 Comparison: 4D B-spline vs Dimensionless 3D\n");
        std::printf("================================================================\n");

        std::printf("--- Generating q=0 reference prices...\n");
        q0_prices = generate_prices(kStrikes, kNoDividends, /*div_yield=*/0.0);

        std::printf("--- Building 4D B-spline (q=0, adaptive)...\n");
        auto bs4d_solver = build_bspline_q0();
        for (size_t vi = 0; vi < kNV; ++vi) {
            q0_bs4d_errors[vi] = compute_errors_via_solver(q0_prices, kStrikes, bs4d_solver, vi, 0.0);
            char title[128];
            std::snprintf(title, sizeof(title),
                          "4D B-spline (q=0) IV Error (bps) — σ=%.0f%%",
                          kVols[vi] * 100);
            print_heatmap(title, kStrikes, q0_bs4d_errors[vi]);
        }

        std::printf("\n--- Building dimensionless 3D B-spline (q=0)...\n");
        auto dim3d_bs = build_dimless_3d(DimensionlessBackend::Interpolant::BSpline);
        if (dim3d_bs.has_value()) {
            for (size_t vi = 0; vi < kNV; ++vi) {
                q0_dim3d_bs_errors[vi] = compute_errors_via_solver(
                    q0_prices, kStrikes, *dim3d_bs, vi, 0.0);
                char title[128];
                std::snprintf(title, sizeof(title),
                              "Dim3D B-spline (q=0) IV Error (bps) — σ=%.0f%%",
                              kVols[vi] * 100);
                print_heatmap(title, kStrikes, q0_dim3d_bs_errors[vi]);
            }
        } else {
            report_wrap_failure("Dimensionless 3D B-spline (q=0)", dim3d_bs.error());
        }

        std::printf("\n--- Building dimensionless 3D Chebyshev (q=0)...\n");
        auto dim3d_ch = build_dimless_3d(DimensionlessBackend::Interpolant::Chebyshev);
        if (dim3d_ch.has_value()) {
            for (size_t vi = 0; vi < kNV; ++vi) {
                q0_dim3d_ch_errors[vi] = compute_errors_via_solver(
                    q0_prices, kStrikes, *dim3d_ch, vi, 0.0);
                char title[128];
                std::snprintf(title, sizeof(title),
                              "Dim3D Chebyshev (q=0) IV Error (bps) — σ=%.0f%%",
                              kVols[vi] * 100);
                print_heatmap(title, kStrikes, q0_dim3d_ch_errors[vi]);
            }
        } else {
            report_wrap_failure("Dimensionless 3D Chebyshev (q=0)", dim3d_ch.error());
        }
    }

    if (need_kref) run_kref_sweep();

    // TV/K filtered comparison — vanilla backends (q=0.02)
    if (need_vanilla) {
        std::printf("\n================================================================\n");
        std::printf("TV/K Filtered Comparison — vanilla (q=%.2f)\n", kDivYield);
        std::printf("================================================================\n");

        for (size_t vi = 0; vi < kNV; ++vi) {
            std::vector<AlgoErrorsN<kNS>> vol_algos;
            if (run_all || path == "bspline")
                vol_algos.push_back({"B-spline", &vanilla_errors[vi]});
            if (run_all || path == "chebyshev") {
                vol_algos.push_back({"Cheb(fixed)", &cheb_errors[vi]});
                vol_algos.push_back({"Cheb(adapt)", &cheb_adaptive_errors[vi]});
            }
            if (!vol_algos.empty())
                print_tvk_comparison<kNS>(vanilla_prices, kStrikes, vi, vol_algos);
        }
    }

    // TV/K filtered comparison — q=0 (4D vs 3D dimensionless)
    if (need_q0) {
        std::printf("\n================================================================\n");
        std::printf("TV/K Filtered Comparison — q=0 (4D B-spline vs Dim3D)\n");
        std::printf("================================================================\n");

        for (size_t vi = 0; vi < kNV; ++vi) {
            std::vector<AlgoErrorsN<kNS>> vol_algos;
            vol_algos.push_back({"BS-4D(q=0)", &q0_bs4d_errors[vi]});
            vol_algos.push_back({"Dim3D-BS", &q0_dim3d_bs_errors[vi]});
            vol_algos.push_back({"Dim3D-Ch", &q0_dim3d_ch_errors[vi]});
            print_tvk_comparison<kNS>(q0_prices, kStrikes, vi, vol_algos);
        }
    }

    if (need_divs) {
        std::printf("\n================================================================\n");
        std::printf("TV/K Filtered Comparison — discrete dividends\n");
        std::printf("================================================================\n");

        for (size_t vi = 0; vi < kNV; ++vi) {
            std::printf("\n  [B-spline per-maturity, reference = quarterly calendar per maturity]");
            std::array<AlgoErrorsN<kNDS>, 1> a{{{"B-spline(div)", div_errors ? &(*div_errors)[vi] : nullptr}}};
            print_tvk_comparison<kNDS>(bs_div_prices, kDivStrikes, vi, a);
            std::printf("\n  [Chebyshev fixed-expiry 1y, reference = 1y calendar rolled]");
            std::array<AlgoErrorsN<kNDS>, 1> c{{{"Cheb(div)", cheb_div_errors ? &(*cheb_div_errors)[vi] : nullptr}}};
            print_tvk_comparison<kNDS>(cheb_div_prices, kDivStrikes, vi, c);
        }
    }

    return g_build_failed ? 1 : 0;
}
