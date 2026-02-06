// SPDX-License-Identifier: MIT
#include "mango/option/american_option.hpp"
#include "mango/pde/operators/centered_difference.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>

namespace mango {
namespace {

namespace {

struct SweepCase {
    double moneyness;
    double maturity;
    double vol;
};

constexpr double kSpot = 100.0;
constexpr double kRate = 0.04;
constexpr double kDividendYield = 0.0;

bool run_gamma_oscillation_case(const SweepCase& test_case) {
    const double strike = kSpot / test_case.moneyness;
    PricingParams params(
        OptionSpec{.spot = kSpot, .strike = strike, .maturity = test_case.maturity,
            .rate = kRate, .dividend_yield = kDividendYield, .option_type = OptionType::PUT},
        test_case.vol);

    auto [grid_spec, time_domain] = estimate_pde_grid(params);
    size_t n = grid_spec.n_points();

    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace.has_value()) {
        ADD_FAILURE() << "Workspace creation failed";
        return false;
    }

    auto solver = AmericanOptionSolver::create(params, workspace.value()).value();
    auto result = solver.solve();
    if (!result.has_value()) {
        ADD_FAILURE() << "Solver failed";
        return false;
    }

    auto grid = result->grid();
    auto solution = grid->solution();
    auto x_grid = grid->x();

    operators::CenteredDifference<double> diff(grid->spacing());
    std::vector<double> dv_dx(n, 0.0);
    std::vector<double> d2v_dx2(n, 0.0);
    diff.compute_first_derivative(solution, dv_dx, 1, n - 2);
    diff.compute_second_derivative(solution, d2v_dx2, 1, n - 2);

    size_t total = 0;
    size_t sign_flips = 0;
    double last_gamma = 0.0;
    bool has_last = false;
    double max_abs_gamma = 0.0;

    constexpr double kCenterWidth = 2.0;

    for (size_t i = 1; i + 1 < n; ++i) {
        double x = x_grid[i];
        if (std::abs(x) > kCenterWidth) {
            continue;
        }

        double S = strike * std::exp(x);
        double K_over_S2 = strike / (S * S);
        double gamma = K_over_S2 * (d2v_dx2[i] - dv_dx[i]);

        if (!std::isfinite(gamma)) {
            continue;
        }

        max_abs_gamma = std::max(max_abs_gamma, std::abs(gamma));
    }

    const double eps = std::max(1e-10, 1e-3 * max_abs_gamma);

    for (size_t i = 1; i + 1 < n; ++i) {
        double x = x_grid[i];
        if (std::abs(x) > kCenterWidth) {
            continue;
        }

        double S = strike * std::exp(x);
        double K_over_S2 = strike / (S * S);
        double gamma = K_over_S2 * (d2v_dx2[i] - dv_dx[i]);

        if (!std::isfinite(gamma) || std::abs(gamma) < eps) {
            continue;
        }

        ++total;
        if (has_last && gamma * last_gamma < 0.0 &&
            std::abs(last_gamma) >= eps) {
            ++sign_flips;
        }

        last_gamma = gamma;
        has_last = true;
    }

    if (total <= 5u || max_abs_gamma < 1e-8) {
        return false;
    }

    EXPECT_LE(sign_flips, 6u)
        << "m=" << test_case.moneyness
        << " T=" << test_case.maturity
        << " vol=" << test_case.vol;
    return true;
}

}  // namespace

TEST(AmericanOptionGammaOscillationTest, SweepNearExpiryPuts) {
    const std::array<double, 3> maturities = {1.0 / 365.0, 2.0 / 365.0, 7.0 / 365.0};
    const std::array<double, 3> vols = {0.4, 0.8, 1.2};
    const std::array<double, 3> moneyness = {0.7, 1.0, 1.3};

    size_t checked = 0;
    for (double m : moneyness) {
        for (double t : maturities) {
            for (double v : vols) {
                if (run_gamma_oscillation_case(SweepCase{m, t, v})) {
                    ++checked;
                }
            }
        }
    }

    ASSERT_GT(checked, 0u);
}

}  // namespace
}  // namespace mango
