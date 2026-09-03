// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include "mango/option/grid_spec_types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace mango {
namespace {

PricingParams put(double sigma, double T, double spot = 100.0) {
    return PricingParams(OptionSpec{.spot = spot, .strike = 100.0, .maturity = T,
        .rate = 0.05, .dividend_yield = 0.0, .option_type = OptionType::PUT}, sigma);
}

// Literal "same grid": bounds, type, every generated coordinate, time steps.
void expect_same_grid(const std::pair<GridSpec<double>, TimeDomain>& a,
                      const std::pair<GridSpec<double>, TimeDomain>& b) {
    EXPECT_DOUBLE_EQ(a.first.x_min(), b.first.x_min());
    EXPECT_DOUBLE_EQ(a.first.x_max(), b.first.x_max());
    EXPECT_EQ(a.first.n_points(), b.first.n_points());
    EXPECT_EQ(a.first.type(), b.first.type());
    auto ga = a.first.generate(); auto gb = b.first.generate();
    auto sa = ga.view().span();   auto sb = gb.view().span();
    ASSERT_EQ(sa.size(), sb.size());
    for (size_t i = 0; i < sa.size(); ++i) EXPECT_EQ(sa[i], sb[i]) << "i=" << i;
    EXPECT_EQ(a.second.n_steps(), b.second.n_steps());
}

TEST(LogMoneynessRange, OfIsOrderIndependentAndEmptyOrNonFiniteIsNullopt) {
    std::vector<double> permuted = {0.0, -0.51, 0.10};
    auto r = LogMoneynessRange::of(permuted);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->lo, -0.51);
    EXPECT_DOUBLE_EQ(r->hi, 0.10);
    EXPECT_FALSE(LogMoneynessRange::of({}).has_value());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> with_nan = {0.0, nan, 0.5};
    std::vector<double> with_inf = {-inf, 0.0};
    EXPECT_FALSE(LogMoneynessRange::of(with_nan).has_value());
    EXPECT_FALSE(LogMoneynessRange::of(with_inf).has_value());
}

TEST(LogMoneynessRange, ReachIsMeasuredFromX0) {
    LogMoneynessRange r{-0.5, 0.5};
    EXPECT_DOUBLE_EQ(r.reach_from(0.0), 0.5);
    EXPECT_DOUBLE_EQ(r.reach_from(0.2), 0.7);    // inside: to the far endpoint
    EXPECT_DOUBLE_EQ(r.reach_from(2.0), 2.5);    // outside: to the far endpoint
}

// D11 rule on a single normalized contract: edge = max(1.1*reach, reach + 3s).
TEST(EstimatePdeGrid, SingleContractCoversWithItsOwnSigmaSqrtT) {
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.5, 0.5};
    auto [grid, td] = estimate_pde_grid(put(0.20, 0.25), acc);   // s = 0.1
    EXPECT_NEAR(grid.x_min(), -(0.5 + 3.0 * 0.1), 1e-9);
    EXPECT_NEAR(grid.x_max(),  (0.5 + 3.0 * 0.1), 1e-9);
}

// Review round 1: the range is absolute ln(S/K); a contract off the money
// must still get [lo - 3s, hi + 3s] inside its domain (x0 = ln 1.2 ~ 0.182).
TEST(EstimatePdeGrid, OffTheMoneyContractStillCoversTheAbsoluteRange) {
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.5, 0.5};
    auto [grid, td] = estimate_pde_grid(put(0.20, 0.25, /*spot=*/120.0), acc);  // s = 0.1
    EXPECT_LE(grid.x_min(), -0.8 + 1e-9);
    EXPECT_GE(grid.x_max(),  0.8 - 1e-9);
}

TEST(EstimatePdeGrid, ClearanceIsConfigurable) {
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.5, 0.5};
    acc.coverage_clearance_sigmas = 6.0;
    auto [grid, td] = estimate_pde_grid(put(0.20, 0.25), acc);   // s = 0.1
    EXPECT_NEAR(grid.x_min(), -(0.5 + 6.0 * 0.1), 1e-9);
}

TEST(EstimatePdeGrid, CoverageInsideNSigmaDomainLeavesGridUnchanged) {
    GridAccuracyParams plain;
    GridAccuracyParams covered = plain;
    covered.log_moneyness_coverage = LogMoneynessRange{-0.51, 0.51};   // 0.51/0.5 + 3 = 4.02 < 5
    expect_same_grid(estimate_pde_grid(put(0.50, 1.0), plain),
                     estimate_pde_grid(put(0.50, 1.0), covered));
}

// Non-finite input never reaches the grid arithmetic: a NaN or infinite
// endpoint, or a NaN or infinite clearance, disables coverage (the plain
// n_sigma grid results); a negative clearance counts as zero.
TEST(EstimatePdeGrid, NonFiniteInputDisablesCoverageAndNegativeClearanceIsZero) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    GridAccuracyParams plain;
    const auto reference = estimate_pde_grid(put(0.20, 0.25), plain);
    for (double bad : {nan, inf, -inf}) {
        GridAccuracyParams endpoint = plain;
        endpoint.log_moneyness_coverage = LogMoneynessRange{bad, 0.5};
        expect_same_grid(estimate_pde_grid(put(0.20, 0.25), endpoint), reference);
        GridAccuracyParams clearance = plain;
        clearance.log_moneyness_coverage = LogMoneynessRange{-2.0, 2.0};
        clearance.coverage_clearance_sigmas = bad;
        expect_same_grid(estimate_pde_grid(put(0.20, 0.25), clearance), reference);
    }
    GridAccuracyParams neg;
    neg.log_moneyness_coverage = LogMoneynessRange{-2.0, 2.0};
    neg.coverage_clearance_sigmas = -4.0;
    auto [g3, t3] = estimate_pde_grid(put(0.20, 0.25), neg);
    EXPECT_NEAR(g3.x_min(), -(2.0 * 1.1), 1e-9);   // the 10% floor alone
}

// The novel part of the fold: contracts with different x0.  Contract A is
// ATM with the largest s (0.4*sqrt(0.25) = 0.2); contract B sits at
// x0 = ln 0.6 ~ -0.511 with s = 0.05 and owns the FARTHEST required reach
// (1.411 vs A's 0.9).
//
// Containment of [lo - 3*s_max, hi + 3*s_max] = [-0.9, 1.5] is guaranteed by
// the s_max contract alone, so those two assertions do NOT discriminate: they
// would hold even for a fold that ignored the max over contracts.  What pins
// the max is the REALIZED edge.  The fold takes n_sigma =
// max_i required_i(s_max) = max(1.1 * 1.4108256/0.2, 1.4108256/0.2 + 3)
// = 10.0541281, so contract A's half-width -- and hence the union -- is
// 10.0541281 * 0.2 = 2.0108256.  A fold that measured the reach from x0 = 0,
// or from the first contract only, or from the s_max contract only, would use
// reach 0.9, give n_sigma 7.5 and an edge of +-1.5, which this pin rejects.
TEST(EstimateBatchPdeGrid, HeterogeneousX0BatchCoversTheAbsoluteRange) {
    std::vector<PricingParams> batch = {put(0.40, 0.25, /*spot=*/100.0),
                                        put(0.10, 0.25, /*spot=*/60.0)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.3, 0.9};
    auto [grid, td] = estimate_batch_pde_grid(batch, acc);
    EXPECT_LE(grid.x_min(), -0.9 + 1e-9);
    EXPECT_GE(grid.x_max(),  1.5 - 1e-9);

    const double x0_b = std::log(0.6);
    const double reach_b = std::max(std::abs(-0.3 - x0_b), std::abs(0.9 - x0_b));
    const double s_max = 0.40 * std::sqrt(0.25);
    const double n_sigma = std::max(1.1 * reach_b / s_max, reach_b / s_max + 3.0);
    const double expected_edge = n_sigma * s_max;   // ~2.0108256
    EXPECT_NEAR(grid.x_min(), -expected_edge, 1e-9);
    EXPECT_NEAR(grid.x_max(),  expected_edge, 1e-9);
}

// Batch: one n_sigma for all, from the largest sigma*sqrt(T); the union's edge
// is max(1.1*reach, reach + 3*s_max).  Identical to widening n_sigma by hand.
TEST(EstimateBatchPdeGrid, CoverageEdgeIsReachPlusThreeSigmaMaxSqrtT) {
    std::vector<PricingParams> batch = {put(0.10, 0.1), put(0.15, 0.1), put(0.20, 0.1)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.51, 0.51};
    const double s = 0.20 * std::sqrt(0.1);
    const double expected_edge = std::max(0.51 * 1.1, 0.51 + 3.0 * s);   // ~0.700
    auto [grid, td] = estimate_batch_pde_grid(batch, acc);
    EXPECT_NEAR(grid.x_min(), -expected_edge, 1e-9);
    EXPECT_NEAR(grid.x_max(),  expected_edge, 1e-9);
    GridAccuracyParams manual;
    manual.n_sigma = std::max(manual.n_sigma, expected_edge / s);
    auto [grid2, td2] = estimate_batch_pde_grid(batch, manual);
    EXPECT_DOUBLE_EQ(grid.x_min(), grid2.x_min());
    EXPECT_EQ(grid.n_points(), grid2.n_points());
    EXPECT_EQ(td.n_steps(), td2.n_steps());
}

TEST(EstimateBatchPdeGridConfig, WrapsTheSharedGrid) {
    std::vector<PricingParams> batch = {put(0.10, 0.1), put(0.20, 0.1)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.51, 0.51};
    auto [grid, td] = estimate_batch_pde_grid(batch, acc);
    auto config = estimate_batch_pde_grid_config(batch, acc);
    EXPECT_DOUBLE_EQ(config.grid_spec.x_min(), grid.x_min());
    EXPECT_EQ(config.n_time, td.n_steps());
    EXPECT_TRUE(config.mandatory_times.empty());
}

// Exact goldens recorded from the retired covering-grid helper on the
// parent revision (its identity test proved the fold reproduces them bit
// for bit), so the fold cannot drift now that the helper is deleted.
TEST(EstimateBatchPdeGrid, GoldensMatchTheRetiredHelper) {
    // (a) clamp-binding Ultra chain batch (T2-like): sigma nodes over
    //     [0.01, 0.225] at T = 0.694375, coverage [-1.0881, 1.0881].
    {
        std::vector<PricingParams> batch = {put(0.01, 0.694375), put(0.1175, 0.694375),
                                            put(0.225, 0.694375)};
        GridAccuracyParams acc = make_grid_accuracy(GridAccuracyProfile::Ultra);
        acc.log_moneyness_coverage = LogMoneynessRange{-1.0881, 1.0881};
        auto [grid, td] = estimate_batch_pde_grid(batch, acc);
        EXPECT_DOUBLE_EQ(grid.x_min(), -1.6505718742968398);
        EXPECT_DOUBLE_EQ(grid.x_max(), 1.6505718742968398);
        EXPECT_EQ(grid.n_points(), 5001u);
        EXPECT_EQ(td.n_steps(), 20000u);
    }
    // (b) dividend batch: sigma {0.05, 0.15}, T = 0.2525, one dividend
    //     {calendar_time 0.1, amount 1.0}, coverage [-0.8452, 0.8250].
    {
        std::vector<PricingParams> batch = {put(0.05, 0.2525), put(0.15, 0.2525)};
        for (auto& p : batch) {
            p.discrete_dividends = {Dividend{.calendar_time = 0.1, .amount = 1.0}};
        }
        GridAccuracyParams acc = make_grid_accuracy(GridAccuracyProfile::Ultra);
        acc.log_moneyness_coverage = LogMoneynessRange{-0.8452, 0.8250};
        auto [grid, td] = estimate_batch_pde_grid(batch, acc);
        EXPECT_DOUBLE_EQ(grid.x_min(), -1.1009491447542108);
        EXPECT_DOUBLE_EQ(grid.x_max(), 1.0713222014752199);
        EXPECT_EQ(grid.n_points(), 5001u);
        EXPECT_EQ(td.n_steps(), 20001u);
    }
}

}  // namespace
}  // namespace mango
