// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/eep_layer.hpp"
#include "mango/option/table/surface_concepts.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

using namespace mango;

// ===========================================================================
// AnalyticalEEP: European Greeks match direct EuropeanOptionSolver
// ===========================================================================

TEST(AnalyticalEEPTest, EuropeanDeltaMatchesDirect) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;
    double eep_delta = eep.european_delta(S, K, tau, sigma, rate);
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = rate, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, sigma).solve().value();
    EXPECT_NEAR(eep_delta, eu.delta(), 1e-12);
}

TEST(AnalyticalEEPTest, EuropeanGammaMatchesDirect) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;
    double eep_gamma = eep.european_gamma(S, K, tau, sigma, rate);
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = rate, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, sigma).solve().value();
    EXPECT_NEAR(eep_gamma, eu.gamma(), 1e-12);
}

TEST(AnalyticalEEPTest, EuropeanThetaMatchesDirect) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;
    double eep_theta = eep.european_theta(S, K, tau, sigma, rate);
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = rate, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, sigma).solve().value();
    EXPECT_NEAR(eep_theta, eu.theta(), 1e-12);
}

TEST(AnalyticalEEPTest, EuropeanRhoMatchesDirect) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;
    double eep_rho = eep.european_rho(S, K, tau, sigma, rate);
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = rate, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, sigma).solve().value();
    EXPECT_NEAR(eep_rho, eu.rho(), 1e-12);
}

// Test with a call option too
TEST(AnalyticalEEPTest, CallGreeksMatchDirect) {
    AnalyticalEEP eep(OptionType::CALL, 0.01);
    double S = 105.0, K = 100.0, tau = 0.5, sigma = 0.25, rate = 0.03;
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = rate, .dividend_yield = 0.01,
            .option_type = OptionType::CALL}, sigma).solve().value();

    EXPECT_NEAR(eep.european_delta(S, K, tau, sigma, rate), eu.delta(), 1e-12);
    EXPECT_NEAR(eep.european_gamma(S, K, tau, sigma, rate), eu.gamma(), 1e-12);
    EXPECT_NEAR(eep.european_theta(S, K, tau, sigma, rate), eu.theta(), 1e-12);
    EXPECT_NEAR(eep.european_rho(S, K, tau, sigma, rate), eu.rho(), 1e-12);
}

// ===========================================================================
// Concept static assertion
// ===========================================================================

TEST(AnalyticalEEPTest, SatisfiesUpdatedEEPStrategyConcept) {
    static_assert(EEPStrategy<AnalyticalEEP>);
}

// ===========================================================================
// EEPLayer: Greek/Gamma with mock leaf
// ===========================================================================

/// Minimal mock leaf that returns constant values for testing EEPLayer wiring.
struct MockLeaf {
    double price_val = 0.5;
    double vega_val = 0.1;
    double raw_val = 0.1;   // > 0 means EEP is active
    double greek_val = 0.05;
    double gamma_val = 0.02;

    [[nodiscard]] double price(double, double, double, double, double) const {
        return price_val;
    }
    [[nodiscard]] double vega(double, double, double, double, double) const {
        return vega_val;
    }
    [[nodiscard]] double raw_value(double, double, double, double, double) const {
        return raw_val;
    }

    [[nodiscard]] std::expected<double, GreekError>
    greek(Greek, const PricingParams&) const {
        return greek_val;
    }

    [[nodiscard]] std::expected<double, GreekError>
    gamma(const PricingParams&) const {
        return gamma_val;
    }

    // Stubs required by EEPLayer
    struct FakeInterp {};
    [[nodiscard]] const FakeInterp& interpolant() const noexcept {
        static FakeInterp fi;
        return fi;
    }
    [[nodiscard]] double K_ref() const noexcept { return 100.0; }
};

TEST(EEPLayerGreeksTest, GreekCombinesLeafAndEuropean) {
    MockLeaf leaf;
    leaf.raw_val = 0.1;    // EEP active
    leaf.greek_val = 0.05;
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    EEPLayer layer(std::move(leaf), std::move(eep));

    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, 0.20);

    auto result = layer.greek(Greek::Delta, params);
    ASSERT_TRUE(result.has_value());

    // Expected: leaf greek + European delta
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, 0.20).solve().value();
    EXPECT_NEAR(*result, 0.05 + eu.delta(), 1e-12);
}

TEST(EEPLayerGreeksTest, GreekReturnsEuropeanOnlyWhenRawZero) {
    MockLeaf leaf;
    leaf.raw_val = 0.0;  // EEP inactive (deep OTM)
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    EEPLayer layer(std::move(leaf), std::move(eep));

    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, 0.20);

    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, 0.20).solve().value();

    // All Greeks should return European-only when raw <= 0
    for (auto g : {Greek::Delta, Greek::Vega, Greek::Theta, Greek::Rho}) {
        auto result = layer.greek(g, params);
        ASSERT_TRUE(result.has_value());
        double expected = [&] {
            switch (g) {
                case Greek::Delta: return eu.delta();
                case Greek::Vega:  return eu.vega();
                case Greek::Theta: return eu.theta();
                case Greek::Rho:   return eu.rho();
            }
            __builtin_unreachable();
        }();
        EXPECT_NEAR(*result, expected, 1e-12)
            << "Failed for Greek enum value " << static_cast<int>(g);
    }
}

TEST(EEPLayerGreeksTest, GammaCombinesLeafAndEuropean) {
    MockLeaf leaf;
    leaf.raw_val = 0.1;
    leaf.gamma_val = 0.02;
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    EEPLayer layer(std::move(leaf), std::move(eep));

    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, 0.20);

    auto result = layer.gamma(params);
    ASSERT_TRUE(result.has_value());

    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, 0.20).solve().value();
    EXPECT_NEAR(*result, 0.02 + eu.gamma(), 1e-12);
}

TEST(EEPLayerGreeksTest, GammaReturnsEuropeanOnlyWhenRawNegative) {
    MockLeaf leaf;
    leaf.raw_val = -0.01;  // Negative raw (deep OTM)
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    EEPLayer layer(std::move(leaf), std::move(eep));

    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, 0.20);

    auto result = layer.gamma(params);
    ASSERT_TRUE(result.has_value());

    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, 0.20).solve().value();
    EXPECT_NEAR(*result, eu.gamma(), 1e-12);
}

// Test sign conventions: put delta should be negative
TEST(AnalyticalEEPTest, PutDeltaIsNegative) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double delta = eep.european_delta(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_LT(delta, 0.0);
}

// Test gamma is always positive
TEST(AnalyticalEEPTest, GammaIsPositive) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double gamma = eep.european_gamma(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(gamma, 0.0);
}

// Test theta is negative for ATM put (time decay)
TEST(AnalyticalEEPTest, PutThetaIsNegative) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double theta = eep.european_theta(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_LT(theta, 0.0);
}

// Test rho is negative for put
TEST(AnalyticalEEPTest, PutRhoIsNegative) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double rho = eep.european_rho(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_LT(rho, 0.0);
}
