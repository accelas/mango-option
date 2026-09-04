// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_adaptive_detail.hpp"
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/american_option.hpp"
#include <algorithm>
#include <cmath>

namespace mango {
namespace {

TEST(DimensionlessAdaptiveTest, BuildsAndConverges) {
    DimensionlessAdaptiveParams params;
    params.target_eep_error = 2e-3;
    params.max_iter = 5;
    params.option_type = OptionType::PUT;
    params.sigma_min = 0.12;
    params.sigma_max = 0.50;
    params.rate_min = 0.01;
    params.rate_max = 0.08;
    params.tau_min = 0.1;
    params.tau_max = 1.5;
    params.moneyness_min = 0.80;
    params.moneyness_max = 1.20;

    auto result = build_dimensionless_surface_adaptive(params, 100.0);
    ASSERT_TRUE(result.has_value());

    EXPECT_GT(result->total_pde_solves, 0);
    EXPECT_GT(result->surface->num_segments(), 0u);
    EXPECT_GT(result->iterations_used, 0u);
}

// Regression (#480, S5): the adaptive loop's ground-truth probe solved a
// normalized contract with maturity max(1.01 tau'_0, 0.02) gridless, so
// its half-width was 5 * sqrt(2) * sqrt(0.02) ~= 1.0 and any probe with
// |x0| > 1 read a cubic-spline extrapolation -- a wrong reference that
// silently misdirects refinement.  This probe sits 0.3 beyond that edge.
// Oracle: the same contract solved directly at spot = K e^x0 (High
// profile); its dollar value is divided by K before subtracting the
// normalized European, matching what the probe returns.
// Pre-fix abs error on this branch's parent: 0.3412 (got 0.3461 vs
// ref 0.0050), from the gridless probe's cubic-spline extrapolation.
TEST(DimensionlessAdaptiveTest, ReferenceEepCoversFarProbe) {
    const double x0 = -1.3, tp = 0.005, lk = 0.0, K = 100.0;
    const double kappa = std::exp(lk);

    const double got = detail::dimensionless_reference_eep(
        x0, tp, lk, K, OptionType::PUT);

    PricingParams p(
        OptionSpec{.spot = K * std::exp(x0), .strike = K, .maturity = tp,
                   .rate = kappa, .dividend_yield = 0.0,
                   .option_type = OptionType::PUT},
        std::sqrt(2.0));
    auto solver = AmericanOptionSolver::create(
        p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
    ASSERT_TRUE(solver.has_value());
    auto am = solver->solve();
    ASSERT_TRUE(am.has_value());
    const double ref = std::max(
        am->value() / K - dimensionless_european(x0, tp, kappa, OptionType::PUT),
        0.0);

    // Tolerance (V/K): post-fix deviation measures ~6.5e-14 (Ultra
    // covering-grid solve vs. the High-profile oracle, both converged to
    // essentially the same value); 1e-4 is >=10x that with cross-toolchain
    // slack, and well under 1/50 of the pre-fix 0.3412 abs error.
    constexpr double TOL = 1e-4;
    EXPECT_NEAR(got, ref, TOL);
    EXPECT_GT(ref, 0.0) << "probe must sit where the EEP is non-trivial";
}

}  // namespace
}  // namespace mango
