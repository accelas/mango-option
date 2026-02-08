// SPDX-License-Identifier: MIT
/**
 * @file dimensionless_inner_test.cc
 * @brief Tests for DimensionlessEEPInner query adapter
 *
 * Validates that DimensionlessEEPInner correctly maps physical queries to
 * dimensionless coordinates, reconstructs American prices from EEP + European,
 * and computes vega via the chain rule with two B-spline partials.
 */

#include "mango/option/table/dimensionless_inner.hpp"
#include "mango/option/table/dimensionless_builder.hpp"
#include "mango/option/american_option.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <memory>

namespace mango {
namespace {

class DimensionlessInnerTest : public ::testing::Test {
protected:
    void SetUp() override {
        DimensionlessAxes axes;
        // Dense grid for good interpolation accuracy at test query points.
        // Test queries use sigma=0.20, r=0.05, tau=1.0, which maps to:
        //   tau'     = 0.02
        //   ln_kappa = ln(2*0.05/0.04) ~= 0.916
        axes.log_moneyness = {-0.40, -0.30, -0.20, -0.15, -0.10, -0.05, 0.0,
                               0.05, 0.10, 0.15, 0.20, 0.30, 0.40};
        axes.tau_prime = {0.005, 0.01, 0.015, 0.02, 0.03, 0.04,
                          0.06, 0.08, 0.10, 0.125, 0.15};
        axes.ln_kappa = {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5,
                          0.8, 1.0, 1.5, 2.0, 2.5};

        auto result = build_dimensionless_surface(
            axes, K_ref_, OptionType::PUT, SurfaceContent::EarlyExercisePremium);
        ASSERT_TRUE(result.has_value())
            << "Build failed with error code: "
            << static_cast<int>(result.error().code)
            << " axis_index=" << result.error().axis_index
            << " count=" << result.error().count;

        inner_ = std::make_unique<DimensionlessEEPInner>(
            result->surface, OptionType::PUT, K_ref_, 0.0);
    }

    static constexpr double K_ref_ = 100.0;
    std::unique_ptr<DimensionlessEEPInner> inner_;
};

// ===========================================================================
// Test 1: Price is positive and bounded for ATM put
// ===========================================================================
TEST_F(DimensionlessInnerTest, PriceIsPositive) {
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 1.0,
                 .sigma = 0.20, .rate = 0.05};
    double price = inner_->price(q);
    EXPECT_GT(price, 0.0) << "ATM put price should be positive";
    EXPECT_LT(price, 20.0) << "ATM put price should be bounded";
}

// ===========================================================================
// Test 2: Vega is positive and bounded for ATM put
// ===========================================================================
TEST_F(DimensionlessInnerTest, VegaIsPositive) {
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 1.0,
                 .sigma = 0.20, .rate = 0.05};
    double vega = inner_->vega(q);
    EXPECT_GT(vega, 0.0) << "ATM put vega should be positive";
    EXPECT_LT(vega, 100.0) << "ATM put vega should be bounded";
}

// ===========================================================================
// Test 3: Reconstructed price matches direct PDE solve
// ===========================================================================
TEST_F(DimensionlessInnerTest, PriceMatchesDirectPDE) {
    struct TestCase {
        double spot;
        double strike;
        double tau;
        double sigma;
        double rate;
        const char* label;
    };

    // Three test points: ATM, ITM put (low spot), OTM put (high spot)
    TestCase cases[] = {
        {100.0, 100.0, 1.0, 0.20, 0.05, "ATM"},
        { 90.0, 100.0, 1.0, 0.20, 0.05, "ITM put"},
        {110.0, 100.0, 1.0, 0.20, 0.05, "OTM put"},
    };

    for (const auto& tc : cases) {
        PriceQuery q{.spot = tc.spot, .strike = tc.strike, .tau = tc.tau,
                     .sigma = tc.sigma, .rate = tc.rate};
        double reconstructed = inner_->price(q);

        // Reference: direct PDE solve
        PricingParams params(
            OptionSpec{
                .spot = tc.spot, .strike = tc.strike, .maturity = tc.tau,
                .rate = tc.rate, .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            tc.sigma);
        auto ref = solve_american_option(params);
        ASSERT_TRUE(ref.has_value()) << "PDE solve failed for " << tc.label;
        double reference = ref->value();

        EXPECT_NEAR(reconstructed, reference, 0.30)
            << "Price mismatch for " << tc.label
            << "\n  reconstructed=" << reconstructed
            << "\n  reference=" << reference
            << "\n  difference=" << std::abs(reconstructed - reference);
    }
}

// ===========================================================================
// Test 4: Chain-rule vega matches finite-difference vega
// ===========================================================================
TEST_F(DimensionlessInnerTest, VegaChainRuleConsistency) {
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 1.0,
                 .sigma = 0.20, .rate = 0.05};

    double chain_vega = inner_->vega(q);

    // Finite-difference vega
    double h = 1e-4;
    PriceQuery q_up = q;
    q_up.sigma = q.sigma + h;
    PriceQuery q_dn = q;
    q_dn.sigma = q.sigma - h;
    double fd_vega = (inner_->price(q_up) - inner_->price(q_dn)) / (2.0 * h);

    // Allow 5% relative + 0.01 absolute tolerance
    double tol = 0.05 * std::abs(fd_vega) + 0.01;
    EXPECT_NEAR(chain_vega, fd_vega, tol)
        << "Chain-rule vega does not match finite-difference vega"
        << "\n  chain_rule=" << chain_vega
        << "\n  finite_diff=" << fd_vega
        << "\n  tolerance=" << tol;
}

}  // namespace
}  // namespace mango
