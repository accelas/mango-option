// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/table/dimensionless/dimensionless_3d_accessor.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/european_option.hpp"
#include <cmath>

namespace mango {
namespace {

TEST(DimensionlessBuilderTest, SolvesPutPDE) {
    DimensionlessAxes axes;
    axes.log_moneyness = {-0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5};
    axes.tau_prime = {0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.14};
    axes.ln_kappa = {-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5};

    auto result = solve_dimensionless_pde(axes, 100.0, OptionType::PUT);
    ASSERT_TRUE(result.has_value())
        << "PDE solve failed: code=" << static_cast<int>(result.error().code);

    EXPECT_EQ(result->n_pde_solves, static_cast<int>(axes.ln_kappa.size()));

    // Values vector should have Nm * Nt * Nk entries
    const size_t Nm = axes.log_moneyness.size();
    const size_t Nt = axes.tau_prime.size();
    const size_t Nk = axes.ln_kappa.size();
    EXPECT_EQ(result->values.size(), Nm * Nt * Nk);

    // V/K at ATM (x=0, index 4), moderate tau' (index 3), kappa=1 (index 3)
    // flat = (4 * 8 + 3) * 8 + 3 = (32 + 3) * 8 + 3 = 35 * 8 + 3 = 283
    size_t idx = (4 * Nt + 3) * Nk + 3;
    EXPECT_GT(result->values[idx], 0.0);
    EXPECT_LT(result->values[idx], 0.5);
}

TEST(DimensionlessBuilderTest, EEPMatchesPDE) {
    DimensionlessAxes axes;
    axes.log_moneyness = {-0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5};
    axes.tau_prime = {0.01, 0.02, 0.03125, 0.04, 0.06, 0.08, 0.10, 0.14};
    axes.ln_kappa = {-1.5, -1.0, -0.5, 0.0, 0.247, 0.5, 1.0, 1.5};

    auto pde = solve_dimensionless_pde(axes, 100.0, OptionType::PUT);
    ASSERT_TRUE(pde.has_value());

    // EEP decompose: converts V/K to dollar EEP in-place
    Dimensionless3DAccessor accessor(pde->values, axes, 100.0);
    eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));

    // Physical parameters: sigma=0.25, rate=0.04, maturity=1.0
    const double sigma = 0.25, rate = 0.04, maturity = 1.0;
    const double K_ref = 100.0;

    // Solve American option in physical coordinates
    auto am = solve_american_option(PricingParams(
        OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = maturity,
            .rate = rate, .dividend_yield = 0.0,
            .option_type = OptionType::PUT}, sigma));
    ASSERT_TRUE(am.has_value());

    // Solve European option in physical coordinates
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = maturity,
            .rate = rate, .dividend_yield = 0.0,
            .option_type = OptionType::PUT}, sigma).solve();
    ASSERT_TRUE(eu.has_value());

    // Reference dollar EEP = American - European
    double ref_dollar_eep = am->value() - eu->value();

    // Find flat index for (x=0, tau_prime=0.03125, ln_kappa=0.247)
    // x=0 is at index 4, tau_prime=0.03125 is at index 2, ln_kappa=0.247 is at index 4
    // flat = (4 * 8 + 2) * 8 + 4 = 34 * 8 + 4 = 276
    const size_t Nk = axes.ln_kappa.size();
    const size_t Nt = axes.tau_prime.size();
    size_t flat_idx = (4 * Nt + 2) * Nk + 4;

    EXPECT_NEAR(pde->values[flat_idx], ref_dollar_eep, 0.2);
}

TEST(DimensionlessBuilderTest, RejectsInsufficientGrid) {
    DimensionlessAxes axes;
    axes.log_moneyness = {0.0};  // Only 1 -- too few for PDE
    axes.tau_prime = {0.01, 0.02, 0.04, 0.06};
    axes.ln_kappa = {-1.0, 0.0, 0.5, 1.0};

    auto result = solve_dimensionless_pde(axes, 100.0, OptionType::PUT);
    ASSERT_FALSE(result.has_value());
}

// Regression (#480, S4): solve_dimensionless_pde solved each kappa slice
// gridless with sigma_eff = sqrt(2) and T = 1.01 * tau'_max, so the PDE
// half-width was 5 * sqrt(2) * sqrt(0.00404) ~= 0.45 while the x nodes
// reach +-0.7: both endpoint nodes were cubic-spline extrapolations.
// Oracle: the same contract solved directly at spot = K e^x (High profile),
// dollar value divided by K (the table stores V/K).
// Pre-fix max abs error on this branch's parent: 0.4827 (V/K) at
// x=-0.7, ln_kappa=-1.0 (same magnitude at ln_kappa=0.0 and 0.5).
TEST(DimensionlessBuilderTest, TailsMatchFdmAtExtremeMoneyness) {
    DimensionlessAxes axes;
    axes.log_moneyness = {-0.7, -0.35, 0.0, 0.35, 0.7};
    axes.tau_prime = {0.002, 0.004};
    axes.ln_kappa = {-1.0, 0.0, 0.5};
    const double K = 100.0;

    auto pde = solve_dimensionless_pde(axes, K, OptionType::PUT);
    ASSERT_TRUE(pde.has_value());

    const size_t Nt = axes.tau_prime.size();
    const size_t Nk = axes.ln_kappa.size();
    const size_t j = Nt - 1;                 // tau' = 0.004
    const double tp = axes.tau_prime[j];

    // Tolerance (V/K): measured post-fix max |got-ref| = 3.73e-12 (x=0.7,
    // ln_kappa=-1.0); 1e-6 gives >250x headroom above that (cross-toolchain
    // floor for Newton-iteration/compiler noise) while staying far below
    // 1/50 of the pre-fix 0.4827 error (~9.65e-3).
    constexpr double TOL = 1e-6;

    for (size_t i : {size_t{0}, axes.log_moneyness.size() - 1}) {
        const double x = axes.log_moneyness[i];
        for (size_t k = 0; k < Nk; ++k) {
            const double kappa = std::exp(axes.ln_kappa[k]);
            PricingParams p(
                OptionSpec{.spot = K * std::exp(x), .strike = K,
                           .maturity = tp, .rate = kappa,
                           .dividend_yield = 0.0,
                           .option_type = OptionType::PUT},
                std::sqrt(2.0));
            auto solver = AmericanOptionSolver::create(
                p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
            ASSERT_TRUE(solver.has_value());
            auto ref = solver->solve();
            ASSERT_TRUE(ref.has_value());
            const double got = pde->values[(i * Nt + j) * Nk + k];
            EXPECT_NEAR(got, ref->value() / K, TOL)
                << "x=" << x << " ln_kappa=" << axes.ln_kappa[k];
        }
    }
}

}  // namespace
}  // namespace mango
