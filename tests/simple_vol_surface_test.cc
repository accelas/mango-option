// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/simple/vol_surface.hpp"
#include "mango/simple/chain_builder.hpp"
#include "mango/simple/sources/yfinance.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/interpolated_iv_solver.hpp"

using namespace mango::simple;

TEST(VolSurfaceTest, ComputeSmileFromChain) {
    // Build a simple chain
    auto chain = ChainBuilder<YFinanceSource>{}
        .symbol("SPY")
        .spot(580.50)
        .quote_time("2024-06-21T10:30:00")
        .settlement(Settlement::PM)
        .dividend_yield(0.013)
        .build();

    // Add a single option for testing
    ExpirySlice slice;
    slice.expiry = Timestamp{"2024-06-28"};  // 1 week out
    slice.settlement = Settlement::PM;

    OptionLeg call;
    call.type = OptionType::CALL;
    call.strike = Price{580.0};
    call.bid = Price{5.50};
    call.ask = Price{5.70};
    slice.options.push_back(call);

    chain.expiries.push_back(std::move(slice));

    MarketContext ctx;
    ctx.rate = 0.053;
    ctx.valuation_time = Timestamp{"2024-06-21T10:30:00"};

    // This requires a precomputed price table, so we test the structure
    // In real usage, you'd provide a solver
    VolatilitySurface surface;
    surface.symbol = chain.symbol;
    surface.spot = *chain.spot;

    EXPECT_EQ(surface.symbol, "SPY");
}

TEST(VolSmileTest, SmilePointStructure) {
    VolatilitySmile::Point pt;
    pt.type = OptionType::CALL;
    pt.strike = Price{580.0};
    pt.moneyness = 0.0;  // ATM
    pt.iv_mid = 0.15;

    EXPECT_EQ(pt.type, OptionType::CALL);
    EXPECT_DOUBLE_EQ(pt.strike.to_double(), 580.0);
    EXPECT_TRUE(pt.iv_mid.has_value());
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: compute_vol_surface dropped discrete dividend schedules (#448)
// Bug: only the double alternative of DividendSpec was read; the
//      vector<Dividend> alternative left div_yield = 0 and never populated
//      IVQuery::discrete_dividends. These tests pin the conversion helper.
TEST(DividendConversionTest, ConvertsExDatesToSortedYearOffsets) {
    Timestamp val{"2026-01-01T00:00:00"};
    std::vector<Dividend> divs = {
        {.ex_date = Timestamp{"2026-07-02T00:00:00"}, .amount = Price{1.50}},
        {.ex_date = Timestamp{"2026-04-02T00:00:00"}, .amount = Price{1.25}},
    };
    auto out = convert_discrete_dividends(divs, val, 1.0);
    ASSERT_EQ(out.size(), 2u);
    // Sorted by calendar time even though the input was not.
    EXPECT_NEAR(out[0].calendar_time, 91.0 / 365.0, 0.01);
    EXPECT_DOUBLE_EQ(out[0].amount, 1.25);
    EXPECT_NEAR(out[1].calendar_time, 182.0 / 365.0, 0.01);
    EXPECT_DOUBLE_EQ(out[1].amount, 1.50);
}

TEST(DividendConversionTest, DropsPastAndPostExpiryDividends) {
    Timestamp val{"2026-01-01T00:00:00"};
    std::vector<Dividend> divs = {
        // Already gone ex before valuation: excluded.
        {.ex_date = Timestamp{"2025-12-15T00:00:00"}, .amount = Price{1.00}},
        // Inside the window: kept.
        {.ex_date = Timestamp{"2026-03-01T00:00:00"}, .amount = Price{1.50}},
        // After expiry (tau_max = 0.5): excluded.
        {.ex_date = Timestamp{"2027-06-01T00:00:00"}, .amount = Price{2.00}},
    };
    auto out = convert_discrete_dividends(divs, val, 0.5);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_DOUBLE_EQ(out[0].amount, 1.50);
    EXPECT_GT(out[0].calendar_time, 0.0);
    EXPECT_LE(out[0].calendar_time, 0.5);
}

TEST(DividendConversionTest, UnparseableExDateIsDropped) {
    Timestamp val{"2026-01-01T00:00:00"};
    std::vector<Dividend> divs = {
        {.ex_date = Timestamp{"not-a-date"}, .amount = Price{1.00}},
    };
    auto out = convert_discrete_dividends(divs, val, 1.0);
    EXPECT_TRUE(out.empty());
}

// ===========================================================================
// End-to-end regression tests: compute_vol_surface + discrete dividends (#448)
// ===========================================================================

namespace {

// Chain with one expiry (~0.80y), a few strikes around spot=100, and one
// discrete dividend at ~0.50y. Market prices are FDM American prices at
// KNOWN_VOL with that dividend, so the IVs a correct pipeline recovers
// are ~KNOWN_VOL.
constexpr double kKnownVol = 0.20;

OptionChain make_discrete_dividend_chain() {
    std::vector<Dividend> divs = {
        {.ex_date = Timestamp{"2026-07-02T00:00:00"}, .amount = Price{1.50}},
    };
    auto chain = ChainBuilder<YFinanceSource>{}
        .symbol("XYZ")
        .spot(100.0)
        .quote_time("2026-01-01T00:00:00")
        .discrete_dividends(divs)
        .build();

    Timestamp val{"2026-01-01T00:00:00"};
    ExpirySlice slice;
    slice.expiry = Timestamp{"2026-10-20T00:00:00"};
    double tau = compute_tau(val, slice.expiry);
    auto solver_divs = convert_discrete_dividends(divs, val, tau);

    for (double strike : {90.0, 95.0, 100.0, 105.0, 110.0}) {
        mango::PricingParams params(
            mango::OptionSpec{.spot = 100.0, .strike = strike,
                .maturity = tau, .rate = 0.05,
                .option_type = mango::OptionType::PUT},
            kKnownVol, solver_divs);
        auto priced = mango::solve_american_option(params);
        if (!priced.has_value()) continue;
        OptionLeg leg;
        leg.type = mango::OptionType::PUT;
        leg.strike = Price{strike};
        double mid = priced->value();
        leg.bid = Price{mid};
        leg.ask = Price{mid};
        slice.options.push_back(leg);
    }
    chain.expiries.push_back(std::move(slice));
    return chain;
}

}  // namespace

// Regression: compute_vol_surface solved the whole surface dividend-free
// when the chain carried a discrete schedule (#448)
// Bug: the vector<Dividend> alternative of DividendSpec was never read;
//      every IVQuery went out with dividend_yield=0 and no schedule.
//      With solver-side validation (#440 item 1) a dividend-free surface
//      must now reject those queries loudly instead of returning IVs
//      biased by the full dividend effect.
TEST(VolSurfaceDividendTest, ContinuousSolverRejectsDiscreteChain) {
    auto chain = make_discrete_dividend_chain();

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

    MarketContext ctx;
    ctx.rate = 0.05;
    ctx.valuation_time = Timestamp{"2026-01-01T00:00:00"};

    auto surface = compute_vol_surface(chain, ctx, &*solver);
    ASSERT_TRUE(surface.has_value());
    ASSERT_FALSE(surface->smiles.empty());
    for (const auto& smile : surface->smiles) {
        for (const auto& pt : smile.points) {
            EXPECT_FALSE(pt.iv_mid.has_value())
                << "dividend-free surface must not produce an IV for a "
                   "discrete-dividend chain (strike "
                << pt.strike.to_double() << ")";
        }
    }
}

// Happy path: a segmented solver built with the SAME schedule accepts the
// queries and recovers the known vol.
//
// Uses the default auto K_ref grid (not an explicit sparse list): sparse
// explicit K_ref grids (e.g. 3 points) carry up to ~0.04 IV blending error
// at mid-anchor strikes -- a pre-existing table-accuracy limitation outside
// this branch's scope. This test covers the schedule wiring end to end, not
// sparse-K_ref accuracy.
TEST(VolSurfaceDividendTest, SegmentedSolverRecoversKnownVol) {
    auto chain = make_discrete_dividend_chain();
    Timestamp val{"2026-01-01T00:00:00"};
    const auto& divs = std::get<std::vector<Dividend>>(*chain.dividends);
    auto solver_divs = convert_discrete_dividends(divs, val, 1.0);

    mango::IVSolverFactoryConfig config{
        .option_type = mango::OptionType::PUT,
        .spot = 100.0,
        .grid = mango::IVGrid{
            .moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
            .vol = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        .backend = mango::BSplineBackend{},
        .discrete_dividends = mango::DiscreteDividendConfig{
            .maturity = 1.0,
            .discrete_dividends = solver_divs,
            // default K_ref config (auto, log-spaced)
        },
    };
    auto solver = mango::make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value());

    MarketContext ctx;
    ctx.rate = 0.05;
    ctx.valuation_time = val;

    auto surface = compute_vol_surface(chain, ctx, &*solver);
    ASSERT_TRUE(surface.has_value());
    ASSERT_FALSE(surface->smiles.empty());
    size_t checked = 0;
    for (const auto& smile : surface->smiles) {
        for (const auto& pt : smile.points) {
            ASSERT_TRUE(pt.iv_mid.has_value())
                << "matching schedule must solve (strike "
                << pt.strike.to_double() << ")";
            EXPECT_NEAR(*pt.iv_mid, kKnownVol, 0.02)
                << "strike " << pt.strike.to_double();
            ++checked;
        }
    }
    EXPECT_GE(checked, 3u);
}
