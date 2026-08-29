// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/support/error_types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <utility>
#include <vector>

TEST(AdaptiveGridParamsTest, DefaultMaxIterIsEight) {
    EXPECT_EQ(mango::AdaptiveGridParams{}.max_iter, 8u);
}

TEST(BuildDiagnosticsTest, DefaultsAreEmpty) {
    mango::BuildDiagnostics d;
    EXPECT_FALSE(d.target_met);
    EXPECT_EQ(d.holdout_points, 0u);
}

// ===========================================================================
// Task 3: PrepareRefsFn / ScoreErrorFn split (spec D4)
// ===========================================================================

// Score equivalence with the old arithmetic:
TEST(ScoreFnTest, MatchesComputeIvError) {
    mango::AdaptiveGridParams p;  // target 2e-5, floor 1e-4
    auto score = mango::make_iv_score_fn(p, mango::OptionType::PUT);
    mango::ErrorRefs refs{.ref_price = 5.0, .vega = 20.0};
    // price_error 0.01 / vega 20 = 5e-4
    EXPECT_NEAR(score(5.01, refs, 100.0, 100.0, 1.0, 0.2, 0.05), 5e-4, 1e-12);
}

TEST(ScoreFnTest, TvkFilterZeroesDeepItm) {
    mango::AdaptiveGridParams p;
    auto score = mango::make_iv_score_fn(p, mango::OptionType::PUT);
    // K=100, S=100 put ref 0.005 -> TV/K = 5e-5 < 1e-4 -> filtered
    mango::ErrorRefs refs{.ref_price = 0.005, .vega = 1.0};
    EXPECT_EQ(score(1.0, refs, 100.0, 100.0, 0.01, 0.2, 0.05), 0.0);
}

TEST(PrepareRefsTest, PropagatesSolveFailure) {
    mango::ValidateFn failing = [](double, double, double, double, double)
        -> std::expected<double, mango::SolverError> {
        return std::unexpected(mango::SolverError{});
    };
    auto prep = mango::make_fd_vega_refs_fn(mango::AdaptiveGridParams{}, failing);
    EXPECT_FALSE(prep(100, 100, 1.0, 0.2, 0.05).has_value());
}

// ===========================================================================
// Task 4: RefineFn/RefineOutcome + B-spline refiner rewrite (spec D6, D2)
// ===========================================================================

TEST(BSplineRefineFnTest, NoOpAtCapReturnsUnchanged) {
    mango::AdaptiveGridParams p;
    p.max_points_per_dim = 4;
    auto fn = mango::make_bspline_refine_fn(p);
    std::vector<double> m{0.0, 0.1, 0.2, 0.3}, t{0.1, 0.5, 1.0, 2.0},
                        v{0.1, 0.2, 0.3, 0.4}, r{0.01, 0.03, 0.05, 0.08};
    auto out = fn(0, {}, m, t, v, r);
    EXPECT_FALSE(out.changed);
    EXPECT_EQ(out.changed_dim, -1);
    EXPECT_EQ(m.size(), 4u);
}

TEST(BSplineRefineFnTest, UniformWhenNoFocus) {
    mango::AdaptiveGridParams p;
    p.refinement_factor = 2.0;  // enough budget to fill every gap
    auto fn = mango::make_bspline_refine_fn(p);
    std::vector<double> m{0.0, 0.3, 0.6, 1.0};
    std::vector<double> t{0.1, 0.5, 1.0, 2.0};
    std::vector<double> v{0.1, 0.2, 0.3, 0.4};
    std::vector<double> r{0.01, 0.03, 0.05, 0.08};

    auto out = fn(0, {}, m, t, v, r);

    EXPECT_TRUE(out.changed);
    EXPECT_EQ(out.changed_dim, 0);
    // Grew toward size * refinement_factor (= 8), capped by the number of
    // gaps actually available to insert into (3) in a single pass.
    size_t target = std::min<size_t>(
        static_cast<size_t>(4 * p.refinement_factor), p.max_points_per_dim);
    EXPECT_GT(target, 4u);
    EXPECT_EQ(m.size(), 7u);

    // Uniform refinement (no focus) spreads new points across the whole
    // axis, not just one localized region.
    EXPECT_TRUE(std::any_of(m.begin(), m.end(),
        [](double x) { return x > 0.0 && x < 0.3; }));
    EXPECT_TRUE(std::any_of(m.begin(), m.end(),
        [](double x) { return x > 0.3 && x < 0.6; }));
    EXPECT_TRUE(std::any_of(m.begin(), m.end(),
        [](double x) { return x > 0.6 && x < 1.0; }));
}

TEST(BSplineRefineFnTest, FocusIntervalTargetsBin) {
    mango::AdaptiveGridParams p;
    p.refinement_factor = 3.0;  // plenty of insertion budget
    auto fn = mango::make_bspline_refine_fn(p);
    std::vector<double> m{0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    std::vector<double> t{0.1, 0.5, 1.0, 2.0};
    std::vector<double> v{0.1, 0.2, 0.3, 0.4};
    std::vector<double> r{0.01, 0.03, 0.05, 0.08};
    const std::vector<double> original = m;

    std::vector<std::pair<double, double>> focus = {{0.55, 0.85}};
    auto out = fn(0, focus, m, t, v, r);

    EXPECT_TRUE(out.changed);
    EXPECT_EQ(out.changed_dim, 0);
    EXPECT_GT(m.size(), original.size());

    // Every newly inserted point lies inside the provided focus interval.
    for (double x : m) {
        bool is_original =
            std::find(original.begin(), original.end(), x) != original.end();
        if (!is_original) {
            EXPECT_GE(x, 0.55);
            EXPECT_LE(x, 0.85);
        }
    }
}

// ===========================================================================
// Task 5: fit domain vs. sample domain separation (spec D2, D3)
// ===========================================================================

// Regression: headroom used to be 3 * width / (n_strikes - 1), which for a
// 7-strike chain gave 3 * w / 6 -- an order of magnitude too wide.  Spec D3
// requires the *expected seeded moneyness density* instead.
TEST(ExtractChainDomainTest, HeadroomUsesExpectedKnots) {
    mango::OptionGrid chain;
    chain.spot = 100.0;
    chain.strikes = {80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0};
    chain.maturities = {0.1, 1.0};
    chain.implied_vols = {0.1, 0.4};
    chain.rates = {0.02, 0.08};

    auto ctx = mango::extract_chain_domain(chain, 60);
    ASSERT_TRUE(ctx.has_value());

    const double w = ctx->sample_bounds.m_max - ctx->sample_bounds.m_min;
    const double expected_h = 3.0 * w / 59.0;
    EXPECT_NEAR(ctx->bounds.m_max - ctx->sample_bounds.m_max, expected_h, 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.m_min - ctx->bounds.m_min, expected_h, 1e-12);

    // The old (n_strikes - 1) rule would have been 3 * w / 6 -- far wider.
    EXPECT_LT(expected_h, 3.0 * w / 6.0);

    // tau / vol / rate: fit == sample (headroom on moneyness only).
    EXPECT_EQ(ctx->bounds.tau_min, ctx->sample_bounds.tau_min);
    EXPECT_EQ(ctx->bounds.tau_max, ctx->sample_bounds.tau_max);
    EXPECT_EQ(ctx->bounds.sigma_min, ctx->sample_bounds.sigma_min);
    EXPECT_EQ(ctx->bounds.sigma_max, ctx->sample_bounds.sigma_max);
    EXPECT_EQ(ctx->bounds.rate_min, ctx->sample_bounds.rate_min);
    EXPECT_EQ(ctx->bounds.rate_max, ctx->sample_bounds.rate_max);
}

// sample_bounds is the user's own range (after minimum-spread widening,
// which is a usability floor rather than headroom).
TEST(ExtractChainDomainTest, SampleBoundsAreTheUserRange) {
    mango::OptionGrid chain;
    chain.spot = 100.0;
    chain.strikes = {80.0, 100.0, 120.0};
    chain.maturities = {0.25, 1.0};
    chain.implied_vols = {0.15, 0.35};
    chain.rates = {0.02, 0.08};

    auto ctx = mango::extract_chain_domain(chain, 60);
    ASSERT_TRUE(ctx.has_value());

    EXPECT_NEAR(ctx->sample_bounds.m_min, std::log(100.0 / 120.0), 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.m_max, std::log(100.0 / 80.0), 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.sigma_min, 0.15, 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.sigma_max, 0.35, 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.rate_min, 0.02, 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.rate_max, 0.08, 1e-12);
    // tau minimum spread is 0.5: [0.25, 1.0] is 0.75 wide, so untouched.
    EXPECT_NEAR(ctx->sample_bounds.tau_min, 0.25, 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.tau_max, 1.0, 1e-12);
}

// Spec D2: run_refinement draws every validation sample from sample_bounds
// while the grids it builds still span the (wider) fit domain.
TEST(RunRefinementDomainTest, ValidationSamplesStayInSampleBounds) {
    mango::AdaptiveGridParams p;
    p.validation_samples = 32;
    p.max_iter = 1;
    p.min_moneyness_points = 6;

    mango::RefinementContext ctx{
        .spot = 100.0,
        .dividend_yield = 0.0,
        .option_type = mango::OptionType::PUT,
        .bounds = {.m_min = -1.0, .m_max = 1.0,
                   .tau_min = 0.1, .tau_max = 2.0,
                   .sigma_min = 0.1, .sigma_max = 0.5,
                   .rate_min = 0.01, .rate_max = 0.09},
        .sample_bounds = {.m_min = -0.2, .m_max = 0.2,
                          .tau_min = 0.1, .tau_max = 2.0,
                          .sigma_min = 0.1, .sigma_max = 0.5,
                          .rate_min = 0.01, .rate_max = 0.09},
    };

    std::vector<double> queried_m;
    std::vector<double> built_m_grid;

    mango::BuildFn build_fn =
        [&](std::span<const double> m, std::span<const double>,
            std::span<const double>, std::span<const double>)
        -> std::expected<mango::SurfaceHandle, mango::PriceTableError> {
        built_m_grid.assign(m.begin(), m.end());
        return mango::SurfaceHandle{
            .price = [&queried_m](double spot, double strike, double,
                                  double, double) {
                queried_m.push_back(std::log(spot / strike));
                return 1.0;
            },
            .pde_solves = 0,
        };
    };

    mango::RefineFn refine_fn =
        [](size_t, std::span<const std::pair<double, double>>,
           std::vector<double>&, std::vector<double>&,
           std::vector<double>&, std::vector<double>&) {
            return mango::RefineOutcome{.changed = false, .changed_dim = -1};
        };

    mango::PrepareRefsFn prepare_refs =
        [](double, double, double, double, double)
        -> std::expected<mango::ErrorRefs, mango::SolverError> {
        return mango::ErrorRefs{.ref_price = 1.0, .vega = 20.0};
    };
    mango::ScoreErrorFn score =
        [](double, const mango::ErrorRefs&, double, double, double,
           double, double) { return 0.0; };

    auto result = mango::run_refinement(p, build_fn, refine_fn, ctx,
                                        prepare_refs, score);
    ASSERT_TRUE(result.has_value());

    // Grids span the fit domain ...
    ASSERT_FALSE(built_m_grid.empty());
    EXPECT_NEAR(built_m_grid.front(), ctx.bounds.m_min, 1e-12);
    EXPECT_NEAR(built_m_grid.back(), ctx.bounds.m_max, 1e-12);

    // ... but every validation sample lies inside the user domain.
    ASSERT_FALSE(queried_m.empty());
    for (double m : queried_m) {
        EXPECT_GE(m, ctx.sample_bounds.m_min - 1e-12);
        EXPECT_LE(m, ctx.sample_bounds.m_max + 1e-12);
    }
}

// ===========================================================================
// Task 6: run_refinement core loop -- holdout, retention, viability,
// measured backtracking walk, build diagnostics (spec D4/D5/D6/D7).
//
// Synthetic-callback harness (no PDE solves):
//   prepare_refs -> {ref_price = analytic_ref(point), vega = 1}
//   score        -> |interp - ref_price|, so a candidate's holdout maximum
//                   is *exactly* its scripted error.
//   build_fn     -> surface priced at analytic_ref + scripted error.  The
//                   script is a function of the grid SIZES it is handed, so
//                   the same grids always produce the same surface (the
//                   determinism D5's final rebuild relies on).
//   refine_fn    -> grows the requested axis by one point.
// ===========================================================================
namespace {

using GridSizes = std::array<size_t, 4>;
using Deltas = std::array<int, 4>;

constexpr double kHarnessSpot = 100.0;
constexpr GridSizes kSeedSizes{4, 5, 6, 7};

// Deliberately independent of sigma: the monotonicity scan (D7) then sees a
// flat sigma profile unless a test overrides the price explicitly.
double analytic_ref(double spot, double strike, double tau, double rate) {
    return 10.0 + std::log(spot / strike) + tau + rate;
}

Deltas delta(const GridSizes& s) {
    Deltas d{};
    for (size_t i = 0; i < 4; ++i) {
        d[i] = static_cast<int>(s[i]) - static_cast<int>(kSeedSizes[i]);
    }
    return d;
}

struct SurfaceScript {
    bool build_ok = true;
    double fresh_err = 0.0;
    double holdout_err = 0.0;
    bool nan_fresh = false;    ///< NaN price at the first fresh sample
    bool nan_holdout = false;  ///< NaN price at the first holdout point
};

class Harness {
public:
    Harness() {
        params.target_iv_error = 1e-4;
        params.validation_samples = 8;
        params.max_iter = 8;
        params.min_moneyness_points = 4;

        ctx = mango::RefinementContext{
            .spot = kHarnessSpot,
            .dividend_yield = 0.0,
            .option_type = mango::OptionType::PUT,
            .bounds = {.m_min = -0.4, .m_max = 0.4,
                       .tau_min = 0.1, .tau_max = 1.0,
                       .sigma_min = 0.1, .sigma_max = 0.5,
                       .rate_min = 0.01, .rate_max = 0.09},
            .sample_bounds = {.m_min = -0.2, .m_max = 0.2,
                              .tau_min = 0.1, .tau_max = 1.0,
                              .sigma_min = 0.1, .sigma_max = 0.5,
                              .rate_min = 0.01, .rate_max = 0.09},
        };

        // Exact seed grids with a distinct size per axis, so "which axis
        // grew" is unambiguous in the scripts below.
        initial.exact = true;
        initial.moneyness = mango::linspace(-0.4, 0.4, kSeedSizes[0]);
        initial.tau = mango::linspace(0.1, 1.0, kSeedSizes[1]);
        initial.vol = mango::linspace(0.1, 0.5, kSeedSizes[2]);
        initial.rate = mango::linspace(0.01, 0.09, kSeedSizes[3]);
    }

    // ---- configuration -----------------------------------------------
    mango::AdaptiveGridParams params;
    mango::RefinementContext ctx;
    mango::InitialGrids initial;
    std::function<SurfaceScript(const GridSizes&, size_t build_call)> script;
    std::set<size_t> fail_setup_refs;  ///< holdout setup indices that fail
    std::set<size_t> noop_axes;        ///< axes whose refine is a no-op
    std::function<double(double, double, double, double, double)> price_override;
    bool use_levels = false;           ///< emulate Chebyshev level counters

    // ---- observations -------------------------------------------------
    size_t prepare_calls = 0;
    size_t setup_calls = 0;
    size_t build_calls = 0;
    size_t snapshot_calls = 0;
    size_t restore_calls = 0;
    bool setup_done = false;
    std::array<int, 4> levels{};
    std::vector<GridSizes> built_sizes;
    std::vector<size_t> refine_axes;
    std::vector<GridSizes> refine_input_sizes;
    std::vector<std::array<int, 4>> refine_levels;
    std::vector<std::vector<std::pair<double, double>>> refine_focus;
    std::vector<std::array<double, 4>> queried;  ///< m, tau, sigma, rate
    std::set<std::array<double, 4>> holdout_keys;

    std::expected<mango::RefinementResult, mango::PriceTableError> run() {
        return mango::run_refinement(params, build_fn(), refine_fn(), ctx,
                                     prepare_fn(), score_fn(), initial,
                                     hooks());
    }

    mango::PrepareRefsFn prepare_fn() {
        return [this](double spot, double strike, double tau, double sigma,
                      double rate)
            -> std::expected<mango::ErrorRefs, mango::SolverError> {
            ++prepare_calls;
            if (!setup_done) {
                size_t idx = setup_calls++;
                if (fail_setup_refs.count(idx) > 0) {
                    return std::unexpected(mango::SolverError{});
                }
                holdout_keys.insert({strike, tau, sigma, rate});
            }
            return mango::ErrorRefs{
                .ref_price = analytic_ref(spot, strike, tau, rate),
                .vega = 1.0};
        };
    }

    mango::ScoreErrorFn score_fn() {
        return [](double interp, const mango::ErrorRefs& refs, double, double,
                  double, double, double) {
            return std::abs(interp - refs.ref_price);
        };
    }

    mango::BuildFn build_fn() {
        return [this](std::span<const double> m, std::span<const double> t,
                      std::span<const double> v, std::span<const double> r)
            -> std::expected<mango::SurfaceHandle, mango::PriceTableError> {
            setup_done = true;
            GridSizes sizes{m.size(), t.size(), v.size(), r.size()};
            size_t call = build_calls++;
            SurfaceScript s = script ? script(sizes, call) : SurfaceScript{};
            if (!s.build_ok) {
                return std::unexpected(mango::PriceTableError{
                    mango::PriceTableErrorCode::FittingFailed});
            }
            built_sizes.push_back(sizes);
            auto fresh_seen = std::make_shared<size_t>(0);
            auto hold_seen = std::make_shared<size_t>(0);
            return mango::SurfaceHandle{
                .price = [this, s, fresh_seen, hold_seen](
                             double spot, double strike, double tau,
                             double sigma, double rate) -> double {
                    queried.push_back(
                        {std::log(spot / strike), tau, sigma, rate});
                    if (price_override) {
                        return price_override(spot, strike, tau, sigma, rate);
                    }
                    double base = analytic_ref(spot, strike, tau, rate);
                    if (holdout_keys.count({strike, tau, sigma, rate}) > 0) {
                        if (s.nan_holdout && (*hold_seen)++ == 0) {
                            return std::numeric_limits<double>::quiet_NaN();
                        }
                        return base + s.holdout_err;
                    }
                    if (s.nan_fresh && (*fresh_seen)++ == 0) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    return base + s.fresh_err;
                },
                .pde_solves = 1,
            };
        };
    }

    mango::RefineFn refine_fn() {
        return [this](size_t dim,
                      std::span<const std::pair<double, double>> focus,
                      std::vector<double>& m, std::vector<double>& t,
                      std::vector<double>& v,
                      std::vector<double>& r) -> mango::RefineOutcome {
            refine_axes.push_back(dim);
            refine_input_sizes.push_back({m.size(), t.size(), v.size(),
                                          r.size()});
            refine_levels.push_back(levels);
            refine_focus.emplace_back(focus.begin(), focus.end());
            if (noop_axes.count(dim) > 0) {
                return mango::RefineOutcome{.changed = false,
                                            .changed_dim = -1};
            }
            std::vector<double>* grids[4] = {&m, &t, &v, &r};
            std::vector<double>& g = *grids[dim];
            if (use_levels) {
                levels[dim] += 1;
                g = mango::linspace(
                    axis_lo(dim), axis_hi(dim),
                    kSeedSizes[dim] + static_cast<size_t>(levels[dim]));
            } else {
                // Insert the midpoint of the widest gap (deterministic).
                size_t at = 0;
                double widest = -1.0;
                for (size_t i = 0; i + 1 < g.size(); ++i) {
                    if (g[i + 1] - g[i] > widest) {
                        widest = g[i + 1] - g[i];
                        at = i;
                    }
                }
                g.insert(g.begin() + static_cast<std::ptrdiff_t>(at) + 1,
                         (g[at] + g[at + 1]) / 2.0);
            }
            return mango::RefineOutcome{.changed = true,
                                        .changed_dim = static_cast<int>(dim)};
        };
    }

    mango::RefineStateHooks hooks() {
        if (!use_levels) return {};
        return mango::RefineStateHooks{
            .snapshot = [this]() -> std::shared_ptr<const void> {
                ++snapshot_calls;
                return std::static_pointer_cast<const void>(
                    std::make_shared<const std::array<int, 4>>(levels));
            },
            .restore = [this](const std::shared_ptr<const void>& snap) {
                ++restore_calls;
                if (snap) {
                    levels = *std::static_pointer_cast<const std::array<int, 4>>(
                        snap);
                }
            },
        };
    }

    double axis_lo(size_t d) const {
        const auto& b = ctx.bounds;
        return d == 0 ? b.m_min : d == 1 ? b.tau_min
                                : d == 2 ? b.sigma_min
                                         : b.rate_min;
    }
    double axis_hi(size_t d) const {
        const auto& b = ctx.bounds;
        return d == 0 ? b.m_max : d == 1 ? b.tau_max
                                : d == 2 ? b.sigma_max
                                         : b.rate_max;
    }

    /// Grid sizes of the returned result.
    static GridSizes result_sizes(const mango::RefinementResult& r) {
        return {r.moneyness.size(), r.tau.size(), r.vol.size(),
                r.rate.size()};
    }
};

/// Script helper: error keyed by the growth vector relative to the seed.
std::function<SurfaceScript(const GridSizes&, size_t)> by_growth(
    std::vector<std::pair<Deltas, SurfaceScript>> table,
    SurfaceScript fallback) {
    return [table = std::move(table), fallback](const GridSizes& s, size_t) {
        Deltas d = delta(s);
        for (const auto& [key, val] : table) {
            if (key == d) return val;
        }
        return fallback;
    };
}

}  // namespace

// ---------------------------------------------------------------------------
// Parameter validation (spec D3)
// ---------------------------------------------------------------------------
TEST(RunRefinementTest, ParamValidation) {
    auto expect_invalid = [](Harness& h, const char* what) {
        auto r = h.run();
        ASSERT_FALSE(r.has_value()) << what;
        EXPECT_EQ(r.error().code, mango::PriceTableErrorCode::InvalidConfig)
            << what;
    };

    {   Harness h; h.params.target_iv_error = 0.0;
        expect_invalid(h, "target_iv_error == 0"); }
    {   Harness h; h.params.target_iv_error = -1e-4;
        expect_invalid(h, "target_iv_error < 0"); }
    {   Harness h;
        h.params.target_iv_error = std::numeric_limits<double>::infinity();
        expect_invalid(h, "target_iv_error non-finite"); }
    {   Harness h; h.params.vega_floor = 0.0;
        expect_invalid(h, "vega_floor == 0"); }
    {   Harness h; h.params.vega_floor = std::numeric_limits<double>::quiet_NaN();
        expect_invalid(h, "vega_floor non-finite"); }
    {   Harness h; h.params.refinement_factor = 1.0;
        expect_invalid(h, "refinement_factor == 1"); }
    {   Harness h; h.params.refinement_factor = 0.5;
        expect_invalid(h, "refinement_factor < 1"); }
    {   Harness h; h.params.max_iter = 0;
        expect_invalid(h, "max_iter == 0"); }
    {   Harness h; h.params.validation_samples = 7;
        expect_invalid(h, "validation_samples < 8"); }
    {   Harness h; h.params.min_moneyness_points = 3;
        expect_invalid(h, "min_moneyness_points < 4"); }
    // sample_bounds must be non-degenerate on every axis.
    {   Harness h; h.ctx.sample_bounds.m_max = h.ctx.sample_bounds.m_min;
        expect_invalid(h, "degenerate moneyness range"); }
    {   Harness h; h.ctx.sample_bounds.tau_max = h.ctx.sample_bounds.tau_min;
        expect_invalid(h, "degenerate tau range"); }
    {   Harness h; h.ctx.sample_bounds.sigma_max = h.ctx.sample_bounds.sigma_min;
        expect_invalid(h, "degenerate sigma range"); }
    {   Harness h; h.ctx.sample_bounds.rate_max = h.ctx.sample_bounds.rate_min;
        expect_invalid(h, "degenerate rate range"); }
    {   Harness h;
        h.ctx.sample_bounds.rate_max = std::numeric_limits<double>::infinity();
        expect_invalid(h, "non-finite rate bound"); }
}

// ---------------------------------------------------------------------------
// Budget edges (spec D5)
// ---------------------------------------------------------------------------
TEST(RunRefinementTest, MaxIterOneReturnsSeed) {
    Harness h;
    h.params.max_iter = 1;
    h.script = by_growth({}, SurfaceScript{.holdout_err = 0.05});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(Harness::result_sizes(*r), kSeedSizes);
    EXPECT_EQ(h.build_calls, 1u);
    EXPECT_TRUE(h.refine_axes.empty());
    EXPECT_EQ(r->diagnostics.total_iterations, 1u);
    EXPECT_EQ(r->diagnostics.picked_iteration, 0u);
    EXPECT_FALSE(r->diagnostics.final_rebuild);
    EXPECT_NEAR(r->achieved_max_error, 0.05, 1e-12);
    EXPECT_FALSE(r->target_met);
}

TEST(RunRefinementTest, ConvergenceAtIterationZero) {
    Harness h;
    h.script = by_growth({}, SurfaceScript{});  // zero error everywhere

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(h.build_calls, 1u);
    EXPECT_TRUE(h.refine_axes.empty());
    EXPECT_EQ(r->diagnostics.picked_iteration, 0u);
    EXPECT_EQ(r->diagnostics.total_iterations, 1u);
    EXPECT_TRUE(r->target_met);
}

// ---------------------------------------------------------------------------
// Retention (spec D5)
// ---------------------------------------------------------------------------
namespace {
// seed 0.05 -> axis-0 refinement 0.01 (best) -> everything afterwards worse.
std::function<SurfaceScript(const GridSizes&, size_t)> retention_script() {
    return by_growth(
        {
            {{0, 0, 0, 0}, {.holdout_err = 0.05}},
            {{1, 0, 0, 0}, {.holdout_err = 0.01}},
            {{2, 0, 0, 0}, {.holdout_err = 0.08}},
        },
        SurfaceScript{.holdout_err = 0.09});
}
}  // namespace

TEST(RunRefinementTest, RetentionPicksBestViable) {
    Harness h;
    h.script = retention_script();

    auto r = h.run();
    ASSERT_TRUE(r.has_value());

    // Iteration 1 (one extra moneyness point) is the best candidate.
    EXPECT_EQ(r->diagnostics.picked_iteration, 1u);
    EXPECT_EQ(Harness::result_sizes(*r), (GridSizes{5, 5, 6, 7}));
    EXPECT_NEAR(r->achieved_max_error, 0.01, 1e-12);
    EXPECT_NEAR(r->achieved_avg_error, 0.01, 1e-12);
    EXPECT_NEAR(r->diagnostics.achieved_max_error, 0.01, 1e-12);
    EXPECT_FALSE(r->target_met);

    // ... and it is not the last surface built, so it is rebuilt once.
    EXPECT_TRUE(r->diagnostics.final_rebuild);
    ASSERT_FALSE(h.built_sizes.empty());
    EXPECT_EQ(h.built_sizes.back(), (GridSizes{5, 5, 6, 7}));
}

TEST(RunRefinementTest, FinalRebuildStatsExcluded) {
    Harness h;
    h.script = retention_script();

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    ASSERT_TRUE(r->diagnostics.final_rebuild);

    const auto& stats = r->diagnostics.iterations;
    ASSERT_FALSE(stats.empty());
    EXPECT_EQ(stats.back().refined_dim, -2);

    size_t rebuild_entries = 0;
    for (const auto& s : stats) {
        if (s.refined_dim == -2) ++rebuild_entries;
    }
    EXPECT_EQ(rebuild_entries, 1u);
    EXPECT_EQ(r->diagnostics.total_iterations, stats.size() - 1);
    // The rebuild does not count against the budget.
    EXPECT_LE(r->diagnostics.total_iterations, h.params.max_iter);
    EXPECT_EQ(h.build_calls, stats.size());
}

// ---------------------------------------------------------------------------
// Viability gate (spec D5)
// ---------------------------------------------------------------------------
TEST(RunRefinementTest, ViabilityBoundRejectsAll) {
    Harness h;
    // Every candidate is far above kViabilityBound (0.20).
    h.script = by_growth({}, SurfaceScript{.holdout_err = 0.5});

    auto r = h.run();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::PriceTableErrorCode::NoViableSurface);
    EXPECT_GT(h.build_calls, 1u);  // exploration was attempted
}

TEST(RunRefinementTest, ViabilityBoundIsIndependentOfTarget) {
    Harness h;
    // A 50 bps surface against a 0.7 bps target: not target_met, but viable.
    h.params.target_iv_error = 7e-5;
    h.script = by_growth({}, SurfaceScript{.holdout_err = 0.005});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_FALSE(r->target_met);
    EXPECT_NEAR(r->achieved_max_error, 0.005, 1e-12);
    EXPECT_LT(0.005, mango::kViabilityBound);
}

TEST(RunRefinementTest, NonFiniteHoldoutDisqualifies) {
    Harness h;
    h.script = by_growth(
        {
            {{0, 0, 0, 0}, {.holdout_err = 0.05}},
            // Lowest score of all, but one holdout price is NaN.
            {{1, 0, 0, 0}, {.holdout_err = 0.001, .nan_holdout = true}},
        },
        SurfaceScript{.holdout_err = 0.06});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->diagnostics.picked_iteration, 0u);
    EXPECT_EQ(Harness::result_sizes(*r), kSeedSizes);
    EXPECT_NEAR(r->achieved_max_error, 0.05, 1e-12);
}

TEST(RunRefinementTest, NonFiniteFreshDisqualifies) {
    Harness h;
    h.script = by_growth(
        {
            {{0, 0, 0, 0}, {.holdout_err = 0.05}},
            // Whole holdout finite and best-scoring, but one fresh in-domain
            // sample is NaN -> non-viable, never returned.
            {{1, 0, 0, 0}, {.holdout_err = 0.001, .nan_fresh = true}},
        },
        SurfaceScript{.holdout_err = 0.06});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->diagnostics.picked_iteration, 0u);
    EXPECT_NEAR(r->achieved_max_error, 0.05, 1e-12);
}

TEST(RunRefinementTest, NonViableSeedRecoveredViaLaterAxis) {
    Harness h;
    h.params.max_iter = 4;
    h.script = by_growth(
        {
            {{0, 0, 0, 0}, {.holdout_err = 0.5}},    // seed: non-viable
            {{1, 0, 0, 0}, {.holdout_err = 0.499}},  // sub-2% improvement
            {{1, 1, 0, 0}, {.holdout_err = 0.498}},  // sub-2% improvement
            {{1, 1, 1, 0}, {.holdout_err = 0.10}},   // viable at last
        },
        SurfaceScript{.holdout_err = 0.6});

    auto r = h.run();
    ASSERT_TRUE(r.has_value()) << "a non-viable seed must still be refinable";
    EXPECT_NEAR(r->achieved_max_error, 0.10, 1e-12);
    EXPECT_EQ(Harness::result_sizes(*r), (GridSizes{5, 6, 7, 7}));
    EXPECT_EQ(r->diagnostics.picked_iteration, 3u);
    EXPECT_FALSE(r->diagnostics.final_rebuild);  // picked == last built
}

// ---------------------------------------------------------------------------
// Measured walk (spec D6)
// ---------------------------------------------------------------------------
TEST(RunRefinementTest, SubThresholdAdvancesBaseOnly) {
    Harness h;
    h.params.max_iter = 3;
    h.script = by_growth(
        {
            {{0, 0, 0, 0}, {.holdout_err = 0.10}},
            {{1, 0, 0, 0}, {.holdout_err = 0.0995}},  // 0.5% better: no restart
        },
        SurfaceScript{.holdout_err = 0.12});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    ASSERT_GE(h.refine_axes.size(), 2u);
    // The walk did not restart: axis 0 is marked tried, axis 1 is next ...
    EXPECT_EQ(h.refine_axes[0], 0u);
    EXPECT_EQ(h.refine_axes[1], 1u);
    // ... but the improved candidate did become the exploration base.
    EXPECT_EQ(h.refine_input_sizes[1], (GridSizes{5, 5, 6, 7}));
}

TEST(RunRefinementTest, FivePercentRestartsWalk) {
    Harness h;
    h.params.max_iter = 3;
    h.script = by_growth(
        {
            {{0, 0, 0, 0}, {.holdout_err = 0.10}},
            {{1, 0, 0, 0}, {.holdout_err = 0.095}},  // 5% better: restart
        },
        SurfaceScript{.holdout_err = 0.12});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    ASSERT_GE(h.refine_axes.size(), 2u);
    // tried[] cleared -> axis 0 is picked again, now from the new base.
    EXPECT_EQ(h.refine_axes[0], 0u);
    EXPECT_EQ(h.refine_axes[1], 0u);
    EXPECT_EQ(h.refine_input_sizes[1], (GridSizes{5, 5, 6, 7}));
}

TEST(RunRefinementTest, BacktrackReachesThirdAxis) {
    Harness h;
    h.params.max_iter = 6;
    h.script = by_growth(
        {
            {{0, 0, 0, 0}, {.holdout_err = 0.10}},
            {{1, 0, 0, 0}, {.holdout_err = 0.0999}},  // sub-2%: axis 0 tried
            {{1, 1, 0, 0}, {.holdout_err = 0.0998}},  // sub-2%: axis 1 tried
            {{1, 1, 1, 0}, {.holdout_err = 0.05}},    // axis 2 pays off
        },
        SurfaceScript{.holdout_err = 0.15});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    ASSERT_GE(h.refine_axes.size(), 4u);
    EXPECT_EQ(h.refine_axes[0], 0u);
    EXPECT_EQ(h.refine_axes[1], 1u);
    EXPECT_EQ(h.refine_axes[2], 2u);
    // The 50% improvement restarted the walk: axis 0 is offered again.
    EXPECT_EQ(h.refine_axes[3], 0u);
    EXPECT_NEAR(r->achieved_max_error, 0.05, 1e-12);
    EXPECT_EQ(Harness::result_sizes(*r), (GridSizes{5, 6, 7, 7}));
}

TEST(RunRefinementTest, NoOpRefineConsumesNoBuild) {
    Harness h;
    h.noop_axes = {0, 1, 2, 3};
    h.script = by_growth({}, SurfaceScript{.holdout_err = 0.05});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    // Every axis was offered exactly once and none consumed a build.
    EXPECT_EQ(h.refine_axes.size(), 4u);
    EXPECT_EQ(h.build_calls, 1u);
    EXPECT_EQ(r->diagnostics.total_iterations, 1u);
    EXPECT_EQ(Harness::result_sizes(*r), kSeedSizes);
}

TEST(RunRefinementTest, NoOpAxesSkippedThenBuild) {
    Harness h;
    h.params.max_iter = 2;
    h.noop_axes = {0, 1};
    h.script = by_growth(
        {
            {{0, 0, 0, 0}, {.holdout_err = 0.05}},
            {{0, 0, 1, 0}, {.holdout_err = 0.04}},
        },
        SurfaceScript{.holdout_err = 0.05});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(h.refine_axes.size(), 3u);  // 0 (no-op), 1 (no-op), 2 (built)
    EXPECT_EQ(h.build_calls, 2u);
    EXPECT_EQ(r->diagnostics.total_iterations, 2u);
    EXPECT_NEAR(r->achieved_max_error, 0.04, 1e-12);
}

// ---------------------------------------------------------------------------
// Build failures (spec D5)
// ---------------------------------------------------------------------------
TEST(RunRefinementTest, SeedBuildFailurePropagates) {
    Harness h;
    h.script = by_growth({}, SurfaceScript{.build_ok = false});

    auto r = h.run();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::PriceTableErrorCode::FittingFailed);
    EXPECT_EQ(h.build_calls, 1u);
}

TEST(RunRefinementTest, TrialBuildFailureContinuesExploration) {
    Harness h;
    h.params.max_iter = 5;
    h.script = by_growth(
        {
            {{0, 0, 0, 0}, {.holdout_err = 0.5}},
            {{1, 0, 0, 0}, {.build_ok = false}},     // axis 0 trial fails
            {{0, 1, 0, 0}, {.holdout_err = 0.499}},  // axis 1: sub-2%
            {{0, 1, 1, 0}, {.holdout_err = 0.10}},   // axis 2: first viable
        },
        SurfaceScript{.holdout_err = 0.6});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->diagnostics.build_failure_fallback);
    EXPECT_NEAR(r->achieved_max_error, 0.10, 1e-12);
    EXPECT_EQ(Harness::result_sizes(*r), (GridSizes{4, 6, 7, 7}));

    // The failed trial is recorded, consumes budget, and names its axis.
    size_t failed = 0;
    for (const auto& s : r->diagnostics.iterations) {
        if (s.build_failed) {
            ++failed;
            EXPECT_EQ(s.refined_dim, 0);
        }
    }
    EXPECT_EQ(failed, 1u);

    // After the failure the exploration base (the seed) was restored, so the
    // axis-1 trial refines the seed grids, not the failed axis-0 grids.
    ASSERT_GE(h.refine_input_sizes.size(), 2u);
    EXPECT_EQ(h.refine_axes[1], 1u);
    EXPECT_EQ(h.refine_input_sizes[1], kSeedSizes);
}

TEST(RunRefinementTest, AllTrialBuildsFailNoViable) {
    Harness h;
    h.script = [](const GridSizes& s, size_t) {
        if (delta(s) == Deltas{0, 0, 0, 0}) {
            return SurfaceScript{.holdout_err = 0.5};  // seed: non-viable
        }
        return SurfaceScript{.build_ok = false};
    };

    auto r = h.run();
    ASSERT_FALSE(r.has_value());
    // The terminal error is NoViableSurface, not the trial build's error.
    EXPECT_EQ(r.error().code, mango::PriceTableErrorCode::NoViableSurface);
}

TEST(RunRefinementTest, FinalRebuildFailurePropagates) {
    Harness h;
    // Same trace as RetentionPicksBestViable, but the picked candidate's
    // rebuild (the second build of those grids) fails.
    h.script = [](const GridSizes& s, size_t call) {
        Deltas d = delta(s);
        if (d == Deltas{0, 0, 0, 0}) return SurfaceScript{.holdout_err = 0.05};
        if (d == Deltas{1, 0, 0, 0}) {
            if (call > 1) return SurfaceScript{.build_ok = false};
            return SurfaceScript{.holdout_err = 0.01};
        }
        return SurfaceScript{.holdout_err = 0.09};
    };

    auto r = h.run();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::PriceTableErrorCode::FittingFailed);
}

// ---------------------------------------------------------------------------
// Backend state hooks (spec D6)
// ---------------------------------------------------------------------------
TEST(RunRefinementTest, StateHooksRestoredOnBacktrack) {
    Harness h;
    h.use_levels = true;
    h.params.max_iter = 4;
    h.script = by_growth(
        {
            {{0, 0, 0, 0}, {.holdout_err = 0.10}},
            {{1, 0, 0, 0}, {.holdout_err = 0.12}},  // rejected trial
            {{0, 1, 0, 0}, {.holdout_err = 0.05}},  // >=2%: restarts the walk
        },
        SurfaceScript{.holdout_err = 0.15});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(h.snapshot_calls, 0u);
    EXPECT_GT(h.restore_calls, 0u);

    ASSERT_GE(h.refine_levels.size(), 3u);
    EXPECT_EQ(h.refine_levels[0], (std::array<int, 4>{0, 0, 0, 0}));
    // The rejected axis-0 trial's level counter was rolled back.
    EXPECT_EQ(h.refine_levels[1], (std::array<int, 4>{0, 0, 0, 0}));
    // After the restart, axis 0 is retried from the new base's state, so its
    // next level is 1 (not 2).
    EXPECT_EQ(h.refine_axes[2], 0u);
    EXPECT_EQ(h.refine_levels[2], (std::array<int, 4>{0, 1, 0, 0}));
    ASSERT_GE(h.built_sizes.size(), 4u);
    EXPECT_EQ(h.built_sizes[3], (GridSizes{5, 6, 6, 7}));
}

// ---------------------------------------------------------------------------
// Holdout (spec D4)
// ---------------------------------------------------------------------------
TEST(RunRefinementTest, ConvergenceRequiresHoldout) {
    Harness h;
    h.params.max_iter = 3;
    // Fresh samples are exact; the holdout is not.
    h.script = by_growth({}, SurfaceScript{.fresh_err = 0.0,
                                           .holdout_err = 0.001});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_FALSE(r->target_met) << "fresh convergence alone must not certify";
    EXPECT_GT(h.build_calls, 1u) << "the loop must keep refining";
    EXPECT_NEAR(r->achieved_max_error, 0.001, 1e-12);
}

TEST(RunRefinementTest, HoldoutRefsPreparedOnce) {
    Harness h;
    h.params.max_iter = 1;
    h.script = by_growth({}, SurfaceScript{.holdout_err = 0.05});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    // holdout setup (8) + one iteration of fresh validation (8).
    EXPECT_EQ(h.setup_calls, h.params.validation_samples);
    EXPECT_EQ(h.prepare_calls, 2 * h.params.validation_samples);
    EXPECT_EQ(r->diagnostics.holdout_points, h.params.validation_samples);
    EXPECT_EQ(r->diagnostics.holdout_points_invalid, 0u);
}

TEST(RunRefinementTest, HoldoutRefsNotRepreparedPerIteration) {
    Harness h;
    h.params.max_iter = 3;
    h.script = by_growth({}, SurfaceScript{.holdout_err = 0.05});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    const size_t n = h.params.validation_samples;
    EXPECT_EQ(h.prepare_calls, n * (1 + r->diagnostics.total_iterations));
}

TEST(RunRefinementTest, HoldoutValidityThresholds) {
    {   // 3 valid points < max(4, 8/4) -> the holdout cannot certify.
        Harness h;
        h.fail_setup_refs = {0, 1, 2, 3, 4};
        h.script = by_growth({}, SurfaceScript{.holdout_err = 0.05});
        auto r = h.run();
        ASSERT_FALSE(r.has_value());
        EXPECT_EQ(r.error().code,
                  mango::PriceTableErrorCode::ValidationFailed);
        EXPECT_EQ(h.build_calls, 0u) << "no build before a usable holdout";
    }
    {   // Exactly at the minimum: build proceeds, invalid points counted.
        Harness h;
        h.fail_setup_refs = {0, 1, 2, 3};
        h.params.max_iter = 1;
        h.script = by_growth({}, SurfaceScript{.holdout_err = 0.05});
        auto r = h.run();
        ASSERT_TRUE(r.has_value());
        EXPECT_EQ(r->diagnostics.holdout_points, 4u);
        EXPECT_EQ(r->diagnostics.holdout_points_invalid, 4u);
    }
}

TEST(RunRefinementTest, SamplesInsideSampleBounds) {
    Harness h;
    h.params.max_iter = 3;
    // Fresh error above target so the error bins (and therefore the focus
    // intervals handed to refine_fn) are populated.
    h.script = by_growth({}, SurfaceScript{.fresh_err = 0.01,
                                           .holdout_err = 0.01});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    const auto& sb = h.ctx.sample_bounds;
    ASSERT_FALSE(h.queried.empty());
    for (const auto& q : h.queried) {
        EXPECT_GE(q[0], sb.m_min - 1e-12);
        EXPECT_LE(q[0], sb.m_max + 1e-12);
        EXPECT_GE(q[1], sb.tau_min - 1e-12);
        EXPECT_LE(q[1], sb.tau_max + 1e-12);
        EXPECT_GE(q[2], sb.sigma_min - 1e-12);
        EXPECT_LE(q[2], sb.sigma_max + 1e-12);
        EXPECT_GE(q[3], sb.rate_min - 1e-12);
        EXPECT_LE(q[3], sb.rate_max + 1e-12);
    }

    // Focus intervals are physical coordinates inside sample_bounds.
    ASSERT_FALSE(h.refine_focus.empty());
    bool any_focus = false;
    for (size_t i = 0; i < h.refine_focus.size(); ++i) {
        size_t dim = h.refine_axes[i];
        double lo = dim == 0 ? sb.m_min : dim == 1 ? sb.tau_min
                             : dim == 2 ? sb.sigma_min
                                        : sb.rate_min;
        double hi = dim == 0 ? sb.m_max : dim == 1 ? sb.tau_max
                             : dim == 2 ? sb.sigma_max
                                        : sb.rate_max;
        for (const auto& iv : h.refine_focus[i]) {
            any_focus = true;
            EXPECT_GE(iv.first, lo - 1e-12);
            EXPECT_LE(iv.second, hi + 1e-12);
            EXPECT_LT(iv.first, iv.second);
        }
    }
    EXPECT_TRUE(any_focus) << "above-target fresh samples must populate bins";
}

// ---------------------------------------------------------------------------
// Monotonicity diagnostics (spec D7)
// ---------------------------------------------------------------------------
TEST(RunRefinementTest, MonotonicityScanCleanSurface) {
    Harness h;
    h.params.max_iter = 1;
    h.script = by_growth({}, SurfaceScript{.holdout_err = 0.05});

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->diagnostics.monotonicity_violations, 0u);
    EXPECT_EQ(r->diagnostics.monotonicity_points_invalid, 0u);
    EXPECT_DOUBLE_EQ(r->diagnostics.worst_vega_slope, 0.0);
}

TEST(RunRefinementTest, MonotonicityScanCountsViolations) {
    Harness h;
    h.params.max_iter = 1;
    h.script = by_growth({}, SurfaceScript{});
    // Price falls with sigma: 6 violating steps per holdout point.
    h.price_override = [](double spot, double strike, double tau, double sigma,
                          double rate) {
        return analytic_ref(spot, strike, tau, rate) - 0.01 * sigma;
    };

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->diagnostics.monotonicity_violations,
              6 * r->diagnostics.holdout_points);
    EXPECT_NEAR(r->diagnostics.worst_vega_slope, -0.01, 1e-9);
}

TEST(RunRefinementTest, MonotonicityScanCountsInvalidPrices) {
    Harness h;
    h.params.max_iter = 1;
    h.script = by_growth({}, SurfaceScript{});
    // NaN at exactly one scan sigma (never at a holdout point's own sigma).
    const double bad_sigma =
        mango::linspace(h.ctx.sample_bounds.sigma_min,
                        h.ctx.sample_bounds.sigma_max, 7)[3];
    h.price_override = [bad_sigma](double spot, double strike, double tau,
                                   double sigma, double rate) {
        if (sigma == bad_sigma) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return analytic_ref(spot, strike, tau, rate);
    };

    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->diagnostics.monotonicity_points_invalid,
              r->diagnostics.holdout_points);
    EXPECT_EQ(r->diagnostics.monotonicity_violations, 0u);
}
