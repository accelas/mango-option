// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/american_option.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"

using namespace mango;

// Regression #485: cardinal evaluations expose the raw sampled rows, so
// no off-node tensor fit or reference-strike blend can mask timeline errors.
// Bug: PDE maturity padding shifted the dividend relative to every sample.
TEST(ChebyshevPDECacheTest, FixedExpiryRowsMatchRolledContracts) {
    const std::vector<Dividend> dividends = {{0.25, 3.0}};
    const std::vector<double> bounds = {0.1, 0.7495, 0.7505, 1.0};
    const std::vector<bool> gaps = {false, true, false};
    const std::vector<double> m = {-0.7, 0.0, 0.7};
    const std::vector<double> tau = {0.1, 0.42475, 0.7495, 0.7505, 0.87525, 1.0};
    const std::vector<double> sigma = {0.1, 0.2};
    const std::vector<double> rate = {0.03, 0.05};
    for (auto type : {OptionType::PUT, OptionType::CALL}) {
        auto pieces = build_chebyshev_segmented_pieces(
            100.0, type, 0.0, dividends, bounds, gaps, m, tau, sigma, rate);
        ASSERT_TRUE(pieces.has_value());
        EXPECT_EQ(pieces->pde_solves, 4u);
        for (size_t j = 0; j < tau.size(); ++j) {
            for (double vol : sigma) {
                PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0,
                    .maturity = tau[j], .rate = 0.05, .option_type = type}, vol);
                // Independent fixed-expiry oracle: event at backward .75.
                if (tau[j] > 0.75) p.discrete_dividends = {{tau[j] - 0.75, 3.0}};
                auto acc = make_grid_accuracy(GridAccuracyProfile::High);
                auto direct = AmericanOptionSolver::create(p, PDEGridSpec{acc});
                ASSERT_TRUE(direct.has_value());
                auto ref = direct->solve();
                ASSERT_TRUE(ref.has_value());
                acc = make_grid_accuracy(GridAccuracyProfile::Ultra);
                auto finer = AmericanOptionSolver::create(p, PDEGridSpec{acc});
                ASSERT_TRUE(finer.has_value());
                auto converged = finer->solve();
                ASSERT_TRUE(converged.has_value());
                ASSERT_NEAR(ref->value(), converged->value(), 0.001);
                const size_t leaf = j < 3 ? 0 : 1;
                const double local_tau = tau[j] - pieces->tau_split.tau_start()[leaf];
                const double got = 100.0 * pieces->leaves[leaf].price(
                    100.0, 100.0, local_tau, vol, 0.05);
                EXPECT_NEAR(got, converged->value(), 0.003);
            }
        }
    }
}
