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
#include <iostream>
#include <chrono>

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
// Corrected fixed-expiry oracle, 2026-09-06: max 0.00744049 (74.4 bps),
// 64 measured / 0 invalid points. The 0.001 (10 bps) target is still unmet;
// this gate retains current viability admission and the historical ceiling.
// Default fastbuild at 2 threads took 1445.6s on the shared test host; the live
// stack showed bounded final assembly over 9 K_refs after refinement finished.
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
    // Preserve the existing ceiling for this documented configuration.
    EXPECT_LE(diag->achieved_max_error, 0.10)
        << "the corrected oracle measured 74.4 bps; now measuring "
        << diag->achieved_max_error * 1e4
        << " bps means it has degraded materially";

    OptionSpec spec{
        .spot = 100.0, .strike = 95.0, .maturity = 0.6,
        .rate = 0.05, .dividend_yield = 0.01,
        .option_type = OptionType::PUT
    };
    PricingParams pricing_params(spec, 0.20);
    // At tau=.6, .4 years elapsed: d=.25 is past; d=.5 is .1 ahead.
    // Explicit offsets make this a separate oracle for schedule conversion.
    pricing_params.discrete_dividends = {{0.1, 1.5}};
    auto ref = solve_american_option(pricing_params);
    ASSERT_TRUE(ref.has_value());

    IVQuery query(spec, ref->value(), pricing_params.discrete_dividends);
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
// Re-measured after #485 with the fixed-expiry oracle: still NoViableSurface.
// The old 15,500 bps value used a different oracle and is no longer a valid
// accuracy claim. Failed builds currently expose the typed refusal without
// candidate error diagnostics. Revisit after #488/#458/#460.
TEST(IVSolverFactorySegmented, DocumentedConfigOnBSplineBackendRefuses) {
    auto config = documented_adaptive_dividend_config();
    config.backend = BSplineBackend{.maturity_grid = {0.1, 0.25, 0.5, 1.0}};

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_FALSE(solver.has_value())
        << "the documented B-spline configuration must retain honest refusal";
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
