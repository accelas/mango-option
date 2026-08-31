// SPDX-License-Identifier: MIT
//
// Accuracy pins for the IV solver factory.  These cases measure computation
// results (documented-config viability, manual-vs-adaptive accuracy) rather
// than software invariants, and each pays for one or more full-size adaptive
// builds (~140s CPU total locally).  They run in the nightly slow suite
// (tags = ["manual", "slow"]), not in per-PR CI: a nightly failure still
// catches documentation rot and accuracy regressions, without putting
// minutes of PDE solves on every pull request.  Factory wiring and
// error-path invariants live in iv_solver_factory_test.cc.
#include <gtest/gtest.h>
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/american_option.hpp"
#include <cmath>

using namespace mango;

namespace {

// ---------------------------------------------------------------------------
// Shared helpers (mirrors iv_solver_factory_test.cc)
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

/// The adaptive discrete-dividend configuration published in CLAUDE.md
/// (Pattern 4) and docs/API_GUIDE.md ("Discrete Dividends with Adaptive
/// Grid").  Shared by the two tests below so the pinning and the
/// documented-limitation companion cannot drift apart.
IVSolverFactoryConfig documented_adaptive_dividend_config() {
    return IVSolverFactoryConfig{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.01,
        .grid = IVGrid{
            .moneyness = {0.92, 0.95, 1.0, 1.05, 1.08},
            .vol = {0.10, 0.15, 0.20, 0.30},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        // Verbatim from the docs: the default max_iter (8) and
        // validation_samples (64), not a relaxed pair.
        .adaptive = AdaptiveGridParams{.target_iv_error = 0.001},
        .backend = ChebyshevBackend{},
        .discrete_dividends = DiscreteDividendConfig{
            .maturity = 1.0,
            .discrete_dividends = {
                Dividend{.calendar_time = 0.25, .amount = 1.50},
                Dividend{.calendar_time = 0.50, .amount = 1.50}},
            .kref_config = {.K_refs = {90.0, 92.5, 95.0, 97.5, 100.0,
                                       102.5, 105.0, 107.5, 110.0}},
        },
    };
}

// The documented adaptive discrete-dividend config, pinned so the
// documentation cannot silently rot into a configuration the viability gate
// refuses.  Everything a reader would copy is verbatim -- including
// `AdaptiveGridParams`, which is *not* relaxed here: the whole point of the
// pin is that the published parameters are the ones that were measured.
//
// The pairing of moneyness grid and K_refs is the fragile part.  The
// assembled surface blends K_ref-struck prices linearly in strike, so the
// K_refs must span *and resolve* the strike range the moneyness grid implies:
// S/K in [0.92, 1.08] means strikes in [92.6, 108.7], served here by K_refs
// at 2.5 % spacing across [90, 110].
//
// Measured on this config: **0.0549 (549 bps) max, 0.0145 avg, 64 of 64
// holdout points measured**, against the 0.20 viability bound -- roughly 3.6x
// of margin.  `target_met` is false (549 bps does not reach the 10 bps
// target), which is honest and expected: viability, not the target, is what
// gates the build.  Runtime ~57 s.
TEST(IVSolverFactorySegmented, DocumentedAdaptiveDiscreteDividendConfig) {
    auto config = documented_adaptive_dividend_config();

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value())
        << "the documented adaptive discrete-dividend config must build a "
           "viable surface: code "
        << static_cast<int>(solver.error().code);

    auto diag = solver->build_diagnostics();
    ASSERT_TRUE(diag.has_value()) << "an adaptive build must report diagnostics";
    EXPECT_GT(diag->holdout_points_measured, 0u)
        << "a surface measured nowhere certifies nothing";
    EXPECT_LE(diag->achieved_max_error, 0.20)
        << "measured " << diag->achieved_max_error * 1e4 << " bps against the "
           "0.20 viability bound";
    // Generous headroom over the measured 0.0549 -- this pins the config
    // against silent degradation, not against ordinary numerical drift.
    EXPECT_LE(diag->achieved_max_error, 0.10)
        << "the documented config measured 549 bps when it was written; "
           "measuring " << diag->achieved_max_error * 1e4
        << " bps means it has degraded materially";

    OptionSpec spec{
        .spot = 100.0, .strike = 95.0, .maturity = 0.5,
        .rate = 0.05, .dividend_yield = 0.01,
        .option_type = OptionType::PUT
    };
    PricingParams pricing_params(spec, 0.20);
    pricing_params.discrete_dividends = config.discrete_dividends->discrete_dividends;
    auto ref = solve_american_option(pricing_params);
    ASSERT_TRUE(ref.has_value());

    IVQuery query(spec, ref->value());
    auto result = solver->solve(query);
    ASSERT_TRUE(result.has_value())
        << "the documented config must also solve, not merely build: code "
        << static_cast<int>(result.error().code);
    EXPECT_GT(result->implied_vol, 0.0);
    EXPECT_LT(result->implied_vol, 3.0);
}

// The documented limitation, pinned: the *same* config on `BSplineBackend`
// does not build.  This is why the documentation recommends `ChebyshevBackend`
// for adaptive discrete-dividend surfaces.
//
// The segmented multi-K_ref B-spline fit degrades badly at low vol on the
// tau segments after a dividend.  At the documented parameters the assembled
// surface measures **1.550 (15,500 bps) max** and the bumped-grid retry
// measures 4.079, against the 0.20 bound.  The worst points cluster at
// sigma <= 0.127 and tau in (0.64, 0.94) -- one returns exactly 0.0 for a put
// worth $7.62, another returns 44.47 for one worth $7.91.  Denser grids make
// it worse, not better, so the D9 retry cannot rescue it.
//
// This was always true; it was not always visible.  Before the reference
// solves filtered their dividend schedule by the sampled maturity, every
// sample below the last dividend date lost its reference, and the surviving
// long-tau tail happened to miss the pathology at the relaxed parameters the
// old version of this test used.
//
// Tracked as the MultiKRefSplit blend / segmented-fit follow-ups.  When one
// of them lands this test will start failing, which is the intended signal:
// re-measure, and if the B-spline path is viable again, promote it back into
// the documentation.
TEST(IVSolverFactorySegmented, DocumentedConfigOnBSplineBackendRefuses) {
    auto config = documented_adaptive_dividend_config();
    config.backend = BSplineBackend{.maturity_grid = {0.1, 0.25, 0.5, 1.0}};

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_FALSE(solver.has_value())
        << "a surface measuring 15,500 bps must not be returned";
    EXPECT_EQ(solver.error().code, ValidationErrorCode::NoViableSurface);
}

// ---------------------------------------------------------------------------
// Side-by-side accuracy comparison
// ---------------------------------------------------------------------------

TEST(IVSolverFactoryComparison, AccuracyManualVsAdaptive) {
    auto manual_config = make_base_config();

    auto adaptive_config = make_base_config();
    adaptive_config.adaptive = AdaptiveGridParams{
        .target_iv_error = 0.002,
        .max_iter = 5,
        .validation_samples = 32,
    };

    auto manual = build_solver(manual_config);
    auto adaptive = build_solver(adaptive_config);
    auto queries = make_test_queries();
    constexpr double TRUE_VOL = 0.20;

    double manual_max_err = 0.0, adaptive_max_err = 0.0;
    size_t count = 0;

    for (const auto& query : queries) {
        auto m = manual.solve(query);
        auto a = adaptive.solve(query);
        if (!m.has_value() || !a.has_value()) continue;

        manual_max_err = std::max(manual_max_err, std::abs(m->implied_vol - TRUE_VOL));
        adaptive_max_err = std::max(adaptive_max_err, std::abs(a->implied_vol - TRUE_VOL));
        count++;
    }

    ASSERT_GT(count, 0u);

    EXPECT_LT(manual_max_err, 0.05);
    EXPECT_LT(adaptive_max_err, 0.05);
}

}  // namespace
