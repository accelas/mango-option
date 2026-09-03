// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <variant>
#include "mango/option/table/covering_grid.hpp"

namespace mango {
namespace {

// ===========================================================================
// Regression tests for issue #437 (moneyness coverage helpers), moved here
// from price_table_builder_test.cc when the helpers became backend-neutral
// (#480).
// ===========================================================================

// Regression: adaptive cached path skipped moneyness-coverage widening
// Bug (#437): solve_missing_slices never raised n_sigma, so the PDE domain
// could undershoot the moneyness axis and extract_tensor extrapolated tails.
TEST(EnsureMoneynessCoverage, WidensNSigmaWhenAxisUndershoots) {
    mango::GridAccuracyParams accuracy;  // default n_sigma = 5.0
    std::vector<mango::PricingParams> batch{mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.1,
                          .rate = 0.05, .dividend_yield = 0.0,
                          .option_type = mango::OptionType::PUT},
        0.20)};
    std::vector<double> log_m = {-0.51, 0.0, 0.51};
    mango::detail::ensure_moneyness_coverage(accuracy, batch, log_m);
    const double ssqt = 0.20 * std::sqrt(0.1);
    const double expected = std::max(0.51 / ssqt * 1.1, 0.51 / ssqt + 3.0);
    EXPECT_NEAR(accuracy.n_sigma, expected, 1e-12);  // ~11.06
}

TEST(EnsureMoneynessCoverage, LeavesNSigmaWhenCovered) {
    mango::GridAccuracyParams accuracy;  // default n_sigma = 5.0
    std::vector<mango::PricingParams> batch{mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                          .rate = 0.05, .dividend_yield = 0.0,
                          .option_type = mango::OptionType::PUT},
        0.50)};
    std::vector<double> log_m = {-0.51, 0.0, 0.51};
    mango::detail::ensure_moneyness_coverage(accuracy, batch, log_m);
    // max(0.51/0.5*1.1, 0.51/0.5+3) = 4.02 < 5
    EXPECT_DOUBLE_EQ(accuracy.n_sigma, 5.0);
}

// Regression: exported helper must not read front()/back() of empty spans
// or divide a wide axis by the 1e-10 floor for an empty batch.
TEST(EnsureMoneynessCoverage, EmptyInputsAreNoOps) {
    mango::GridAccuracyParams accuracy;
    std::vector<mango::PricingParams> batch{mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.1,
                          .rate = 0.05, .dividend_yield = 0.0,
                          .option_type = mango::OptionType::PUT},
        0.20)};
    std::vector<double> log_m = {-0.51, 0.0, 0.51};

    mango::detail::ensure_moneyness_coverage(accuracy, {}, log_m);
    EXPECT_DOUBLE_EQ(accuracy.n_sigma, 5.0);

    mango::detail::ensure_moneyness_coverage(accuracy, batch, {});
    EXPECT_DOUBLE_EQ(accuracy.n_sigma, 5.0);
}

// Regression: widened accuracy alone is bypassed by per-normalized-group
// grid estimation; the materialized concrete grid is what guarantees
// coverage for EVERY slice of a multi-sigma batch.
TEST(MaterializeCoveringGrid, ConcreteGridCoversAxisForMultiSigmaBatch) {
    mango::GridAccuracyParams accuracy;
    std::vector<mango::PricingParams> batch;
    for (double sigma : {0.10, 0.15, 0.20}) {
        batch.push_back(mango::PricingParams(
            mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.1,
                              .rate = 0.05, .dividend_yield = 0.0,
                              .option_type = mango::OptionType::PUT},
            sigma));
    }
    std::vector<double> log_m = {-0.51, 0.0, 0.51};
    auto spec = mango::detail::materialize_covering_grid(accuracy, batch, log_m);
    auto* config = std::get_if<mango::PDEGridConfig>(&spec);
    ASSERT_NE(config, nullptr);
    EXPECT_LE(config->grid_spec.x_min(), -0.51);
    EXPECT_GE(config->grid_spec.x_max(), 0.51);
    EXPECT_GT(config->n_time, 0u);
    EXPECT_TRUE(config->mandatory_times.empty());
}

// Regression (#480 D3): the reach is the largest-magnitude node wherever it
// sits in the span.  A merely reversed array still has its extremes at
// front()/back(); only an INTERIOR extreme distinguishes minmax from the
// old front()/back() reach, which read {0.0, -0.51, 0.10} as reach 0.10.
TEST(EnsureMoneynessCoverage, ReachIsOrderIndependent) {
    std::vector<mango::PricingParams> batch{mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.1,
                          .rate = 0.05, .dividend_yield = 0.0,
                          .option_type = mango::OptionType::PUT},
        0.20)};
    std::vector<double> sorted = {-0.51, 0.0, 0.10};
    std::vector<double> permuted = {0.0, -0.51, 0.10};

    mango::GridAccuracyParams from_sorted, from_permuted;
    mango::detail::ensure_moneyness_coverage(from_sorted, batch, sorted);
    mango::detail::ensure_moneyness_coverage(from_permuted, batch, permuted);

    const double ssqt = 0.20 * std::sqrt(0.1);
    // ~11.06
    const double expected = std::max(0.51 / ssqt * 1.1, 0.51 / ssqt + 3.0);
    EXPECT_NEAR(from_sorted.n_sigma, expected, 1e-12);
    EXPECT_NEAR(from_permuted.n_sigma, expected, 1e-12);
}

// Regression (#480 follow-on): the boundary must clear the outermost node by
// a few diffusion lengths, not a fixed fraction of the reach.  With reach
// 0.5 and sigma*sqrt(T) = 0.1 the 10% rule alone would put the edge 0.05
// past the node -- half a diffusion length -- and boundary error diffused
// into the edge nodes (measured 0.84 per $100 at sigma ~0.19 on the
// segmented Chebyshev fit).  Required: half-width >= reach + 3*sigma*sqrt(T).
TEST(EnsureMoneynessCoverage, BoundaryClearsOuterNodeByThreeSigmaSqrtT) {
    mango::GridAccuracyParams accuracy;
    std::vector<mango::PricingParams> batch{mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.25,
                          .rate = 0.05, .dividend_yield = 0.0,
                          .option_type = mango::OptionType::PUT},
        0.20)};                                  // sigma*sqrt(T) = 0.1
    std::vector<double> log_m = {-0.5, 0.0, 0.5};
    mango::detail::ensure_moneyness_coverage(accuracy, batch, log_m);
    EXPECT_NEAR(accuracy.n_sigma, 0.5 / 0.1 + 3.0, 1e-12);   // 8.0 > 5.5

    auto spec = mango::detail::materialize_covering_grid(
        mango::GridAccuracyParams{}, batch, log_m);
    auto* config = std::get_if<mango::PDEGridConfig>(&spec);
    ASSERT_NE(config, nullptr);
    EXPECT_LE(config->grid_spec.x_min(), -(0.5 + 3.0 * 0.1) + 1e-9);
    EXPECT_GE(config->grid_spec.x_max(),  (0.5 + 3.0 * 0.1) - 1e-9);
}

}  // namespace
}  // namespace mango
