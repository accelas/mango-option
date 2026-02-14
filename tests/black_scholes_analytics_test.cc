// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/european_option.hpp"
#include <cmath>

namespace mango {
namespace {

TEST(BlackScholesAnalyticsTest, VegaATMPut) {
    // ATM put: S=K=100, τ=1, σ=0.20, r=0.05
    double vega = bs_vega(100.0, 100.0, 1.0, 0.20, 0.05);
    // Expected: S * sqrt(τ) * N'(d1) where d1 = (r + σ²/2)τ / (σ√τ) = 0.15
    // N'(0.15) ≈ 0.3752, so vega ≈ 100 * 1 * 0.3752 ≈ 37.52
    EXPECT_NEAR(vega, 37.52, 0.1);
}

TEST(BlackScholesAnalyticsTest, VegaOTMPut) {
    // OTM put: S=100, K=80, τ=0.5, σ=0.25, r=0.03
    double vega = bs_vega(100.0, 80.0, 0.5, 0.25, 0.03);
    // Lower vega for OTM
    EXPECT_NEAR(vega, 10.1, 0.5);
}

TEST(BlackScholesAnalyticsTest, VegaShortMaturity) {
    // Very short maturity: τ=0.01
    double vega = bs_vega(100.0, 100.0, 0.01, 0.20, 0.05);
    // Vega scales with sqrt(τ), should be ~1/10 of 1-year
    EXPECT_LT(vega, 5.0);
}

TEST(BlackScholesAnalyticsTest, VegaDeepITM) {
    // Deep ITM put: S=100, K=150
    double vega = bs_vega(100.0, 150.0, 1.0, 0.20, 0.05);
    // Still positive but lower than ATM
    EXPECT_NEAR(vega, 9.4, 0.5);
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: Missing dividend yield parameter in bs_d1 and bs_vega
// Bug: Original implementation ignored dividend yield, affecting both d1 calculation
//      and the dividend discount factor in vega
TEST(BlackScholesAnalyticsTest, VegaWithDividendYield) {
    // ATM with 2% dividend yield
    // With q=0.02: d1 = [ln(100/100) + (0.05 - 0.02 + 0.02)τ] / (0.20√1) = 0.25
    // Vega = 100 * e^(-0.02*1) * √1 * N'(0.25)
    // e^(-0.02) ≈ 0.9802, N'(0.25) ≈ 0.3867
    // Vega ≈ 100 * 0.9802 * 1.0 * 0.3867 ≈ 37.9
    double vega = bs_vega(100.0, 100.0, 1.0, 0.20, 0.05, 0.02);
    EXPECT_GT(vega, 35.0);
    EXPECT_LT(vega, 40.0);
}

}  // namespace
}  // namespace mango
