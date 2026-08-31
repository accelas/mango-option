// SPDX-License-Identifier: MIT
//
// Per-PR CI tests for the IV solver factory: wiring, error paths, and
// regression invariants.  Every case here exists to expose a software error,
// not to measure computation results — the accuracy pins (documented-config
// viability, manual-vs-adaptive comparison) live in
// iv_solver_factory_slow_test.cc and run in the nightly slow suite.
#include <gtest/gtest.h>
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include <cmath>

using namespace mango;

namespace {

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

constexpr double SPOT = 100.0;
constexpr double DIVIDEND_YIELD = 0.02;
constexpr OptionType TYPE = OptionType::PUT;

IVSolverFactoryConfig make_base_config() {
    IVSolverFactoryConfig config;
    config.option_type = TYPE;
    config.spot = SPOT;
    config.dividend_yield = DIVIDEND_YIELD;
    config.grid = IVGrid{
        .moneyness = {0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2},
        .vol = {0.10, 0.15, 0.20, 0.25, 0.30},
        .rate = {0.02, 0.03, 0.05, 0.07},
    };
    config.backend = BSplineBackend{.maturity_grid = {0.1, 0.25, 0.5, 0.75, 1.0}};
    return config;
}

AnyInterpIVSolver build_solver(const IVSolverFactoryConfig& config) {
    auto result = make_interpolated_iv_solver(config);
    EXPECT_TRUE(result.has_value()) << "Solver build failed";
    return std::move(*result);
}

std::vector<IVQuery> make_test_queries() {
    std::vector<IVQuery> queries;
    for (double K : {95.0, 100.0, 105.0}) {
        for (double T : {0.25, 0.5, 1.0}) {
            PricingParams params(
                OptionSpec{.spot = SPOT, .strike = K, .maturity = T,
                           .rate = 0.05, .dividend_yield = DIVIDEND_YIELD,
                           .option_type = TYPE},
                0.20);
            auto result = solve_american_option(params);
            if (result.has_value()) {
                queries.push_back(IVQuery(
                    OptionSpec{.spot = SPOT, .strike = K, .maturity = T,
                               .rate = 0.05, .dividend_yield = DIVIDEND_YIELD,
                               .option_type = TYPE},
                    result->value()));
            }
        }
    }
    return queries;
}

// ---------------------------------------------------------------------------
// Standard path: manual wiring, then one consolidated adaptive smoke
// ---------------------------------------------------------------------------

TEST(IVSolverFactoryTest, ManualBuilds) {
    auto solver = make_interpolated_iv_solver(make_base_config());
    ASSERT_TRUE(solver.has_value())
        << "Error code: " << static_cast<int>(solver.error().code);
}

TEST(IVSolverFactoryTest, ManualSolvesIV) {
    auto solver = build_solver(make_base_config());
    auto queries = make_test_queries();
    ASSERT_FALSE(queries.empty());

    for (const auto& query : queries) {
        auto result = solver.solve(query);
        ASSERT_TRUE(result.has_value())
            << "IV solve failed for K=" << query.strike
            << " T=" << query.maturity;
        EXPECT_NEAR(result->implied_vol, 0.20, 0.02)
            << "K=" << query.strike << " T=" << query.maturity;
    }
}

TEST(IVSolverFactoryTest, ManualBatchSolve) {
    auto solver = build_solver(make_base_config());

    std::vector<IVQuery> queries(3);
    for (auto& q : queries) {
        q.spot = SPOT;
        q.strike = 100.0;
        q.maturity = 0.5;
        q.rate = RateSpec{0.05};
        q.dividend_yield = DIVIDEND_YIELD;
        q.option_type = TYPE;
        q.market_price = 6.0;
    }

    auto batch_result = solver.solve_batch(queries);
    EXPECT_EQ(batch_result.results.size(), 3u);
}

// One adaptive build carries every adaptive-path wiring assertion: build
// succeeds, diagnostics are exposed (spec D7), single solves recover the
// true vol, and batch solve round-trips.  Historically these were four
// separate cases (Builds/Adaptive, SolvesIV/Adaptive, BatchSolve/Adaptive,
// AdaptivePathExposesDiagnostics), each paying ~18-50s for its own build of
// the same surface.  The 0.02 tolerance is a smoke bound distinguishing a
// correctly wired surface from garbage, not an accuracy pin — accuracy pins
// live in iv_solver_factory_slow_test.cc (nightly slow suite).
TEST(IVSolverFactoryTest, AdaptiveEndToEndSmoke) {
    auto config = make_base_config();
    config.adaptive = AdaptiveGridParams{
        .target_iv_error = 0.002,
        .max_iter = 5,
        .validation_samples = 32,
    };
    auto solver = build_solver(config);

    // Diagnostics exposed through the convenience factory (spec D7).
    auto diag = solver.build_diagnostics();
    ASSERT_TRUE(diag.has_value());
    EXPECT_GE(diag->total_iterations, 1u);

    // Single solves.
    auto queries = make_test_queries();
    ASSERT_FALSE(queries.empty());
    for (const auto& query : queries) {
        auto result = solver.solve(query);
        ASSERT_TRUE(result.has_value())
            << "IV solve failed for K=" << query.strike
            << " T=" << query.maturity;
        EXPECT_NEAR(result->implied_vol, 0.20, 0.02)
            << "K=" << query.strike << " T=" << query.maturity;
    }

    // Batch solve on the same surface.
    std::vector<IVQuery> batch(3);
    for (auto& q : batch) {
        q.spot = SPOT;
        q.strike = 100.0;
        q.maturity = 0.5;
        q.rate = RateSpec{0.05};
        q.dividend_yield = DIVIDEND_YIELD;
        q.option_type = TYPE;
        q.market_price = 6.0;
    }
    auto batch_result = solver.solve_batch(batch);
    EXPECT_EQ(batch_result.results.size(), 3u);
}

// ---------------------------------------------------------------------------
// Segmented path (manual and adaptive)
// ---------------------------------------------------------------------------

TEST(IVSolverFactorySegmented, DiscreteDividends) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .grid = IVGrid{
            .moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
            .vol = {0.10, 0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        .backend = BSplineBackend{},
        .discrete_dividends = DiscreteDividendConfig{
            .maturity = 1.0,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}},
            .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value()) << "Factory should succeed with discrete dividends";

    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.5;
    query.rate = RateSpec{0.05};
    query.option_type = OptionType::PUT;
    query.market_price = 7.0;

    auto result = solver->solve(query);
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 3.0);
    }
}

TEST(IVSolverFactorySegmented, AdaptiveDiscreteDividends) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.02,
        // The assembled multi-K_ref surface blends K_ref-struck prices
        // linearly in strike, so the K_refs must both span and resolve the
        // queryable strike range.  The default +/-30 % moneyness grid with
        // K_refs {80, 100, 120} measures 8,278 (827,756 bps) on the final
        // validation and is refused by the viability gate (spec D9).
        .grid = IVGrid{
            .moneyness = {0.92, 0.95, 1.0, 1.05, 1.08},
            .vol = {0.10, 0.15, 0.20, 0.30},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        .adaptive = AdaptiveGridParams{
            .target_iv_error = 0.005,  // 50 bps for test speed
            .max_iter = 2,
            .validation_samples = 16,
        },
        .backend = BSplineBackend{},
        .discrete_dividends = DiscreteDividendConfig{
            .maturity = 1.0,
            .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
            .kref_config = {.K_refs = {90.0, 95.0, 100.0, 105.0, 110.0}},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value())
        << "Factory should succeed with adaptive + discrete dividends";

    // Solve IV for a known option
    OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 0.5,
        .rate = 0.05, .dividend_yield = 0.02,
        .option_type = OptionType::PUT
    };

    PricingParams pricing_params(spec, 0.20);
    pricing_params.discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}};
    auto ref = solve_american_option(pricing_params);
    ASSERT_TRUE(ref.has_value());

    IVQuery query(spec, ref->value());
    auto result = solver->solve(query);
    // The market price comes from a reference solve, so a solve failure here
    // would be a real regression — assert, don't skip.
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.0);
    EXPECT_LT(result->implied_vol, 3.0);
}

// The documentation pins for the adaptive discrete-dividend config published
// in CLAUDE.md (Pattern 4) and docs/API_GUIDE.md —
// IVSolverFactorySegmented.DocumentedAdaptiveDiscreteDividendConfig and
// .DocumentedConfigOnBSplineBackendRefuses — live in
// iv_solver_factory_slow_test.cc (nightly slow suite): they measure accuracy
// against the viability bound, which is computation stress, not an invariant
// per-PR CI needs to re-establish.

// ---------------------------------------------------------------------------
// Chebyshev backend: continuous and discrete dividend paths
// ---------------------------------------------------------------------------

TEST(IVSolverFactoryChebyshev, ContinuousBuildsAndSolves) {

    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.02,
        .grid = IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.10, 0.20, 0.30},
            .rate = {0.03, 0.05},
        },
        .backend = ChebyshevBackend{
            .maturity = 1.0,
            .num_pts = {9, 9, 7, 5},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value())
        << "Chebyshev continuous build failed: code "
        << static_cast<int>(solver.error().code);

    // Round-trip: price an ATM put at known vol, then recover IV
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.5,
                   .rate = 0.05, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT},
        0.20);
    auto ref = solve_american_option(params);
    ASSERT_TRUE(ref.has_value());

    IVQuery query(static_cast<const OptionSpec&>(params), ref->value());
    auto result = solver->solve(query);
    ASSERT_TRUE(result.has_value())
        << "Chebyshev IV solve failed";
    EXPECT_NEAR(result->implied_vol, 0.20, 0.02);
}

TEST(IVSolverFactoryChebyshev, SegmentedBuildsAndSolves) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.02,
        // Moneyness kept near the money.  Measured on this dividend config
        // with the D5 holdout drawn from the user's own range (D2): S/K in
        // [0.8, 1.2] scores 3892 bps max error, which the viability gate
        // (2000 bps) correctly rejects; the same build over S/K in
        // [0.9, 1.1] scores 99 bps.  The segmented Chebyshev accuracy cliff
        // away from the money is pre-existing and tracked separately -- this
        // test covers the factory wiring.
        .grid = IVGrid{
            .moneyness = {0.9, 0.95, 1.0, 1.05, 1.1},
            .vol = {0.15, 0.20, 0.30},
            .rate = {0.03, 0.05},
        },
        .adaptive = AdaptiveGridParams{
            .target_iv_error = 0.005,
            .max_iter = 2,
            .validation_samples = 16,
        },
        .backend = ChebyshevBackend{},
        .discrete_dividends = DiscreteDividendConfig{
            .maturity = 1.0,
            .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
            .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value())
        << "Chebyshev segmented build failed: code "
        << static_cast<int>(solver.error().code);

    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.5;
    query.rate = RateSpec{0.05};
    query.dividend_yield = 0.02;
    query.option_type = OptionType::PUT;
    query.market_price = 7.0;

    auto result = solver->solve(query);
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 3.0);
    }
}

// The manual-vs-adaptive accuracy comparison
// (IVSolverFactoryComparison.AccuracyManualVsAdaptive) lives in
// iv_solver_factory_slow_test.cc (nightly slow suite): it measures IV error
// against the true vol across two full builds, which is computation stress,
// not a wiring invariant.

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: an adaptive build that refuses must surface as a refusal
// through the factory, not as a grid-shape complaint.
// Bug: `to_validation_error` had no arm for `PriceTableErrorCode::
// NoViableSurface` / `ValidationFailed`, so both fell through the `default:`
// arm and reached callers as `ValidationErrorCode::InvalidGridSize` -- a
// caller told to fix its grid sizes when the real answer is that no candidate
// surface passed the D5 viability gate.  This survived 11 task reviews
// because only the reverse mapping (ValidationError -> PriceTableError) was
// tested; nothing exercised the forward direction end to end.
//
// The config is `AdaptiveGridBuilderTest.BuildSegmentedLargeDividend`'s --
// $20 of absolute dividends against a $100 spot, which measures 583,897 bps
// against the 0.20 viability bound -- routed through the public factory.
TEST(IVSolverFactorySegmented, AdaptiveRefusalSurfacesAsNoViableSurface) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.0,
        .grid = IVGrid{
            .moneyness = {0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5},
            .vol = {0.05, 0.10, 0.20, 0.30, 0.50},
            .rate = {0.01, 0.03, 0.05, 0.10},
        },
        .adaptive = AdaptiveGridParams{
            .target_iv_error = 0.005,
            .max_iter = 2,
            .validation_samples = 16,
        },
        .backend = BSplineBackend{
            .maturity_grid = {0.1, 0.25, 0.5, 1.0},
        },
        .discrete_dividends = DiscreteDividendConfig{
            .maturity = 1.0,
            .discrete_dividends = {
                Dividend{.calendar_time = 0.25, .amount = 10.0},
                Dividend{.calendar_time = 0.75, .amount = 10.0}},
            .kref_config = {.K_refs = {70.0, 100.0, 130.0}},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_FALSE(solver.has_value())
        << "an unusable surface must not be returned";
    EXPECT_EQ(solver.error().code, ValidationErrorCode::NoViableSurface)
        << "adaptive refusal reached the caller as code "
        << static_cast<int>(solver.error().code);
}

// ---------------------------------------------------------------------------
// Build diagnostics surfaced through the convenience factory (spec D7)
// ---------------------------------------------------------------------------

TEST(IVSolverFactoryBuildDiagnostics, ManualPathHasNoDiagnostics) {
    auto solver = build_solver(make_base_config());
    EXPECT_FALSE(solver.build_diagnostics().has_value());
}

// The adaptive-path counterpart (diagnostics ARE exposed) is asserted by
// IVSolverFactoryTest.AdaptiveEndToEndSmoke on its single adaptive build.

// Regression: a continuous surface must loudly reject a query carrying a
// discrete dividend schedule (#448 / #440 item 1)
// Bug: the schedule was silently ignored and the query priced dividend-free.
TEST(IVSolverFactoryDividendValidation, ContinuousSurfaceRejectsDiscreteQuery) {
    mango::IVSolverFactoryConfig config{
        .option_type = mango::OptionType::PUT,
        .spot = 100.0,
        .grid = mango::IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.10, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.04, 0.06, 0.08},
        },
        .backend = mango::BSplineBackend{
            .maturity_grid = {0.25, 0.5, 0.75, 1.0},
        },
    };
    auto solver = mango::make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value());

    mango::IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.8;
    query.rate = mango::RateSpec{0.04};
    query.option_type = mango::OptionType::PUT;
    query.market_price = 6.0;
    query.discrete_dividends = {{.calendar_time = 0.5, .amount = 1.5}};

    auto result = solver->solve(query);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::IVErrorCode::DiscreteDividendMismatch);

    // Same query without the schedule still solves (sanity that the
    // rejection is schedule-driven, not incidental).
    query.discrete_dividends.clear();
    auto ok = solver->solve(query);
    EXPECT_TRUE(ok.has_value());
}

// Cheap continuous BSpline surface for exercising InterpolatedIVSolver::create()
// directly (bypassing the factory, which already canonicalizes schedules).
BSplinePriceTable make_direct_create_wrapper() {
    constexpr double K_ref = 100.0;
    std::vector<double> m_grid = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
    std::vector<double> tau_grid = {0.25, 0.5, 1.0, 2.0};
    std::vector<double> vol_grid = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.04, 0.06, 0.08};

    auto result = PriceTableBuilder::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, K_ref,
        GridAccuracyParams{}, OptionType::PUT, 0.0);
    EXPECT_TRUE(result.has_value()) << "Failed to build price table";
    auto [builder, axes] = std::move(result.value());
    auto table = builder.build(axes,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            BSplineTensorAccessor accessor(tensor, a, K_ref);
            eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));
        });
    EXPECT_TRUE(table.has_value()) << "Failed to build EEP table";
    auto wrapper = make_bspline_surface(table->spline, K_ref, 0.0, OptionType::PUT);
    EXPECT_TRUE(wrapper.has_value());
    return std::move(*wrapper);
}

// Regression: an explicitly supplied build schedule was stored without
// canonicalization; an unsorted-but-correct schedule caused spurious
// DiscreteDividendMismatch rejections (PR #449 pre-merge review, round 3)
// Bug: validate_query assumes the stored schedule is sorted and merged;
//      create() now canonicalizes explicit schedules at the choke point.
TEST(IVSolverFactoryDividendValidation, DirectCreateCanonicalizesUnsortedSchedule) {
    // Deliberately unsorted (0.5 before 0.25) — create() must canonicalize
    // this before storing it in build_dividends_.
    std::vector<Dividend> unsorted_build_schedule = {
        {.calendar_time = 0.5, .amount = 2.0},
        {.calendar_time = 0.25, .amount = 1.0},
    };

    auto solver_result = InterpolatedIVSolver<BSplinePriceTable>::create(
        make_direct_create_wrapper(), InterpolatedIVSolverConfig{}, unsorted_build_schedule);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 1.0;  // comfortably above 0.5, within [0.25, 2.0]
    query.rate = RateSpec{0.04};
    query.option_type = OptionType::PUT;
    query.market_price = 6.0;
    // Sorted equivalent of the unsorted build schedule.
    query.discrete_dividends = {
        {.calendar_time = 0.25, .amount = 1.0},
        {.calendar_time = 0.5, .amount = 2.0},
    };

    auto result = solver.solve(query);
    if (!result.has_value()) {
        EXPECT_NE(result.error().code, IVErrorCode::DiscreteDividendMismatch)
            << "Unsorted-but-equivalent build schedule should not trigger a mismatch";
    }

    // A query missing an entry from the build schedule must still be rejected.
    query.discrete_dividends = {{.calendar_time = 0.25, .amount = 1.0}};
    auto missing = solver.solve(query);
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, IVErrorCode::DiscreteDividendMismatch);
}

}  // namespace
