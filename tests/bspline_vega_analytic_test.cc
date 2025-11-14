/**
 * @file bspline_vega_analytic_test.cc
 * @brief Test analytic B-spline vega derivative against finite difference
 */

#include "src/interpolation/bspline_4d.hpp"
#include "src/interpolation/bspline_fitter_4d.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mango;

namespace {

// Analytic Black-Scholes for test surface
double analytic_bs_price(double S, double K, double tau, double sigma, double r) {
    if (tau <= 0.0) {
        return std::max(K - S, 0.0);
    }

    const double sqrt_tau = std::sqrt(tau);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau);
    const double d2 = d1 - sigma * sqrt_tau;

    auto Phi = [](double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    };

    return K * std::exp(-r * tau) * Phi(-d2) - S * Phi(-d1);
}

// Fit B-spline to analytic surface
std::unique_ptr<BSpline4D_FMA> fit_test_surface() {
    const double K_ref = 100.0;

    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.10, 0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate_grid = {0.0, 0.025, 0.05, 0.10};

    const size_t Nm = m_grid.size();
    const size_t Nt = tau_grid.size();
    const size_t Nv = sigma_grid.size();
    const size_t Nr = rate_grid.size();

    // Generate prices from analytic Black-Scholes
    std::vector<double> prices(Nm * Nt * Nv * Nr);
    for (size_t i = 0; i < Nm; ++i) {
        for (size_t j = 0; j < Nt; ++j) {
            for (size_t k = 0; k < Nv; ++k) {
                for (size_t l = 0; l < Nr; ++l) {
                    const size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                    prices[idx] = analytic_bs_price(
                        m_grid[i] * K_ref,
                        K_ref,
                        tau_grid[j],
                        sigma_grid[k],
                        rate_grid[l]);
                }
            }
        }
    }

    // Fit B-spline
    auto fitter_result = BSplineFitter4D::create(m_grid, tau_grid, sigma_grid, rate_grid);
    EXPECT_TRUE(fitter_result.has_value());

    auto fit_result = fitter_result.value().fit(prices);
    EXPECT_TRUE(fit_result.success);

    return std::make_unique<BSpline4D_FMA>(
        m_grid, tau_grid, sigma_grid, rate_grid, fit_result.coefficients);
}

} // namespace

TEST(BSplineVegaAnalytic, AnalyticMatchesFiniteDifference) {
    auto spline = fit_test_surface();

    // Test at several points
    constexpr double epsilon = 1e-4;
    constexpr double tolerance = 1e-4; // Reasonable FD tolerance

    std::vector<std::tuple<double, double, double, double>> test_points = {
        {1.03, 0.5, 0.22, 0.05},   // Interior point
        {0.95, 0.25, 0.15, 0.025}, // Near boundaries
        {1.10, 1.5, 0.28, 0.08},   // Different region
        {0.85, 0.75, 0.12, 0.02},  // Low sigma
        {1.15, 1.8, 0.29, 0.09},   // High everything
    };

    for (const auto& [m, tau, sigma, r] : test_points) {
        double price_analytic, vega_analytic;
        spline->eval_price_and_vega_analytic(m, tau, sigma, r, price_analytic, vega_analytic);

        double price_fd, vega_fd;
        spline->eval_price_and_vega_triple(m, tau, sigma, r, epsilon, price_fd, vega_fd);

        // Prices should be identical (both evaluate at σ)
        EXPECT_NEAR(price_analytic, price_fd, 1e-12)
            << "Price mismatch at (m=" << m << ", τ=" << tau << ", σ=" << sigma << ", r=" << r << ")";

        // Vegas should match within FD truncation error
        EXPECT_NEAR(vega_analytic, vega_fd, tolerance)
            << "Vega mismatch at (m=" << m << ", τ=" << tau << ", σ=" << sigma << ", r=" << r << ")"
            << "\n  Analytic: " << vega_analytic
            << "\n  FD:       " << vega_fd
            << "\n  Diff:     " << std::abs(vega_analytic - vega_fd);
    }
}

TEST(BSplineVegaAnalytic, BoundaryValues) {
    auto spline = fit_test_surface();

    // Test that analytic derivatives produce reasonable values at boundaries
    // (cannot compare to FD because clamping makes FD biased at boundaries)

    const auto& m_grid = spline->moneyness_grid();
    const auto& tau_grid = spline->maturity_grid();
    const auto& sigma_grid = spline->volatility_grid();
    const auto& rate_grid = spline->rate_grid();

    // Left sigma boundary (low vol)
    {
        double price, vega;
        spline->eval_price_and_vega_analytic(
            1.0, 0.5, sigma_grid.front(), 0.05,
            price, vega);

        // Vega should be positive and finite
        EXPECT_GT(vega, 0.0) << "Vega should be positive";
        EXPECT_TRUE(std::isfinite(vega)) << "Vega should be finite";
        EXPECT_LT(vega, 1000.0) << "Vega should be reasonable magnitude";
    }

    // Right sigma boundary (high vol)
    {
        double price, vega;
        spline->eval_price_and_vega_analytic(
            1.0, 0.5, sigma_grid.back(), 0.05,
            price, vega);

        // Vega should be positive and finite
        EXPECT_GT(vega, 0.0) << "Vega should be positive";
        EXPECT_TRUE(std::isfinite(vega)) << "Vega should be finite";
        EXPECT_LT(vega, 1000.0) << "Vega should be reasonable magnitude";
    }

    // Corner case: all boundaries
    {
        double price, vega;
        spline->eval_price_and_vega_analytic(
            m_grid.front(), tau_grid.front(), sigma_grid.front(), rate_grid.front(),
            price, vega);

        EXPECT_TRUE(std::isfinite(vega)) << "Vega should be finite at corner";
    }
}
