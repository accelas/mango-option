// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/european_option.hpp"
#include "src/option/option_concepts.hpp"
#include <cmath>

namespace {

using mango::EuropeanOptionSolver;
using mango::EuropeanOptionResult;
using mango::PricingParams;
using mango::OptionType;

// ===========================================================================
// Concept satisfaction (compile-time checks)
// ===========================================================================
static_assert(mango::OptionResult<EuropeanOptionResult>,
              "EuropeanOptionResult must satisfy OptionResult");
static_assert(mango::OptionResultWithVega<EuropeanOptionResult>,
              "EuropeanOptionResult must satisfy OptionResultWithVega");

// ===========================================================================
// Helper: standard ATM put parameters
// ===========================================================================
PricingParams atm_put_params() {
    PricingParams p;
    p.spot = 100.0;
    p.strike = 100.0;
    p.maturity = 1.0;
    p.rate = 0.05;
    p.dividend_yield = 0.02;
    p.type = OptionType::PUT;
    p.volatility = 0.20;
    return p;
}

PricingParams atm_call_params() {
    PricingParams p = atm_put_params();
    p.type = OptionType::CALL;
    return p;
}

// ===========================================================================
// 1. ATM put known value
// ===========================================================================
TEST(EuropeanOptionTest, ATMPutKnownValue) {
    auto params = atm_put_params();
    EuropeanOptionSolver solver(params);
    auto result = solver.solve();

    // S=K=100, τ=1, σ=0.20, r=0.05, q=0.02
    // Expected ≈ 6.34 (Black-Scholes closed form)
    EXPECT_NEAR(result.value(), 6.34, 0.01);
}

// ===========================================================================
// 2. Put-call parity: C - P = S·exp(-qT) - K·exp(-rT)
// ===========================================================================
TEST(EuropeanOptionTest, PutCallParity) {
    auto put_params = atm_put_params();
    auto call_params = atm_call_params();

    EuropeanOptionSolver put_solver(put_params);
    EuropeanOptionSolver call_solver(call_params);

    auto put = put_solver.solve();
    auto call = call_solver.solve();

    double S = put_params.spot;
    double K = put_params.strike;
    double r = 0.05;
    double q = put_params.dividend_yield;
    double T = put_params.maturity;

    double parity = call.value() - put.value() - (S * std::exp(-q * T) - K * std::exp(-r * T));
    EXPECT_NEAR(parity, 0.0, 1e-10);
}

// ===========================================================================
// 3. Delta bounds: put ∈ [-1, 0], call ∈ [0, 1]
// ===========================================================================
TEST(EuropeanOptionTest, DeltaBounds) {
    auto put = EuropeanOptionSolver(atm_put_params()).solve();
    auto call = EuropeanOptionSolver(atm_call_params()).solve();

    EXPECT_GE(put.delta(), -1.0);
    EXPECT_LE(put.delta(), 0.0);

    EXPECT_GE(call.delta(), 0.0);
    EXPECT_LE(call.delta(), 1.0);
}

// ===========================================================================
// 4. Gamma non-negative
// ===========================================================================
TEST(EuropeanOptionTest, GammaNonNegative) {
    auto put = EuropeanOptionSolver(atm_put_params()).solve();
    auto call = EuropeanOptionSolver(atm_call_params()).solve();

    EXPECT_GE(put.gamma(), 0.0);
    EXPECT_GE(call.gamma(), 0.0);

    // Gamma should be the same for put and call
    EXPECT_NEAR(put.gamma(), call.gamma(), 1e-12);
}

// ===========================================================================
// 5. Vega non-negative
// ===========================================================================
TEST(EuropeanOptionTest, VegaNonNegative) {
    auto put = EuropeanOptionSolver(atm_put_params()).solve();
    auto call = EuropeanOptionSolver(atm_call_params()).solve();

    EXPECT_GE(put.vega(), 0.0);
    EXPECT_GE(call.vega(), 0.0);

    // Vega should be the same for put and call
    EXPECT_NEAR(put.vega(), call.vega(), 1e-12);
}

// ===========================================================================
// 6. value_at(spot) == value()
// ===========================================================================
TEST(EuropeanOptionTest, ValueAtSpotEqualsValue) {
    auto result = EuropeanOptionSolver(atm_put_params()).solve();
    EXPECT_NEAR(result.value_at(result.spot()), result.value(), 1e-12);
}

// ===========================================================================
// 7. Delta vs finite differences
// ===========================================================================
TEST(EuropeanOptionTest, DeltaVsFiniteDifferences) {
    double eps = 0.01;
    auto params = atm_put_params();

    auto result = EuropeanOptionSolver(params).solve();
    double analytic_delta = result.delta();

    // Central difference: (V(S+eps) - V(S-eps)) / (2*eps)
    double v_up = result.value_at(params.spot + eps);
    double v_down = result.value_at(params.spot - eps);
    double fd_delta = (v_up - v_down) / (2.0 * eps);

    EXPECT_NEAR(analytic_delta, fd_delta, 1e-6);
}

// ===========================================================================
// 8. Accessors return correct values
// ===========================================================================
TEST(EuropeanOptionTest, AccessorsReturnCorrectValues) {
    auto params = atm_put_params();
    auto result = EuropeanOptionSolver(params).solve();

    EXPECT_DOUBLE_EQ(result.spot(), 100.0);
    EXPECT_DOUBLE_EQ(result.strike(), 100.0);
    EXPECT_DOUBLE_EQ(result.maturity(), 1.0);
    EXPECT_DOUBLE_EQ(result.volatility(), 0.20);
    EXPECT_EQ(result.option_type(), OptionType::PUT);
}

// ===========================================================================
// 9. Deep ITM put ≈ K·exp(-rT) - S·exp(-qT)
// ===========================================================================
TEST(EuropeanOptionTest, DeepITMPut) {
    PricingParams params;
    params.spot = 50.0;
    params.strike = 100.0;
    params.maturity = 1.0;
    params.rate = 0.05;
    params.dividend_yield = 0.02;
    params.type = OptionType::PUT;
    params.volatility = 0.20;

    auto result = EuropeanOptionSolver(params).solve();
    double expected = 100.0 * std::exp(-0.05) - 50.0 * std::exp(-0.02);
    EXPECT_NEAR(result.value(), expected, 0.01);
}

// ===========================================================================
// 10. Deep OTM put ≈ 0
// ===========================================================================
TEST(EuropeanOptionTest, DeepOTMPut) {
    PricingParams params;
    params.spot = 200.0;
    params.strike = 100.0;
    params.maturity = 1.0;
    params.rate = 0.05;
    params.dividend_yield = 0.02;
    params.type = OptionType::PUT;
    params.volatility = 0.20;

    auto result = EuropeanOptionSolver(params).solve();
    EXPECT_NEAR(result.value(), 0.0, 0.01);
}

// ===========================================================================
// 11. Negative spot rejected by create()
// ===========================================================================
TEST(EuropeanOptionTest, NegativeSpotRejected) {
    PricingParams params = atm_put_params();
    params.spot = -10.0;

    auto result = EuropeanOptionSolver::create(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::ValidationErrorCode::InvalidSpotPrice);
}

// ===========================================================================
// 12. Rho has correct sign
// ===========================================================================
TEST(EuropeanOptionTest, RhoSign) {
    // Rho for put should be negative (higher rates decrease put value)
    auto put = EuropeanOptionSolver(atm_put_params()).solve();
    EXPECT_LT(put.rho(), 0.0);

    // Rho for call should be positive (higher rates increase call value)
    auto call = EuropeanOptionSolver(atm_call_params()).solve();
    EXPECT_GT(call.rho(), 0.0);
}

// ===========================================================================
// Edge case: zero maturity returns intrinsic value
// ===========================================================================
TEST(EuropeanOptionTest, ZeroMaturityReturnsIntrinsic) {
    PricingParams params = atm_put_params();
    params.maturity = 0.0;

    // ITM put: intrinsic = max(K - S, 0) = 0 for ATM
    auto result = EuropeanOptionSolver(params).solve();
    EXPECT_DOUBLE_EQ(result.value(), 0.0);

    // ITM put
    params.spot = 90.0;
    result = EuropeanOptionSolver(params).solve();
    EXPECT_DOUBLE_EQ(result.value(), 10.0);

    // Greeks should be zero (except delta for deep ITM)
    EXPECT_DOUBLE_EQ(result.gamma(), 0.0);
    EXPECT_DOUBLE_EQ(result.vega(), 0.0);
    EXPECT_DOUBLE_EQ(result.theta(), 0.0);
    EXPECT_DOUBLE_EQ(result.rho(), 0.0);
}

// ===========================================================================
// Edge case: zero volatility returns intrinsic-like value
// ===========================================================================
TEST(EuropeanOptionTest, ZeroVolatilityReturnsDiscountedIntrinsic) {
    PricingParams params;
    params.spot = 90.0;
    params.strike = 100.0;
    params.maturity = 1.0;
    params.rate = 0.05;
    params.dividend_yield = 0.0;
    params.type = OptionType::PUT;
    params.volatility = 0.0;

    auto result = EuropeanOptionSolver(params).solve();
    // ITM put with zero vol: K*exp(-rT) - S*exp(-qT)
    double expected = 100.0 * std::exp(-0.05) - 90.0;
    EXPECT_NEAR(result.value(), expected, 1e-10);
}

// ===========================================================================
// Gamma via finite differences
// ===========================================================================
TEST(EuropeanOptionTest, GammaVsFiniteDifferences) {
    double eps = 0.01;
    auto params = atm_put_params();

    auto result = EuropeanOptionSolver(params).solve();
    double analytic_gamma = result.gamma();

    // Central difference for gamma: (V(S+eps) - 2V(S) + V(S-eps)) / eps^2
    double v_up = result.value_at(params.spot + eps);
    double v_mid = result.value_at(params.spot);
    double v_down = result.value_at(params.spot - eps);
    double fd_gamma = (v_up - 2.0 * v_mid + v_down) / (eps * eps);

    EXPECT_NEAR(analytic_gamma, fd_gamma, 1e-4);
}

}  // namespace
