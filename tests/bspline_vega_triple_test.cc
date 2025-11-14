#include "src/interpolation/bspline_4d.hpp"
#include "src/interpolation/bspline_fitter_4d.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace mango;

namespace {

// Helper: Analytic Black-Scholes for test surface
double analytic_bs(double S, double K, double tau, double sigma, double r) {
    if (tau <= 0.0) return std::max(K - S, 0.0);
    const double sqrt_tau = std::sqrt(tau);
    const double d1 = (std::log(S/K) + (r + 0.5*sigma*sigma)*tau) / (sigma*sqrt_tau);
    const double d2 = d1 - sigma*sqrt_tau;
    auto Phi = [](double x) { return 0.5 * (1.0 + std::erf(x/std::sqrt(2.0))); };
    return K*std::exp(-r*tau)*Phi(-d2) - S*Phi(-d1);
}

// Fixture with fitted B-spline surface
class BSplineVegaTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Small test grids (minimum 4 points for cubic B-splines)
        m_grid = {0.85, 0.9, 1.0, 1.1};
        tau_grid = {0.25, 0.5, 0.75, 1.0};
        sigma_grid = {0.15, 0.20, 0.25, 0.30};
        rate_grid = {0.02, 0.03, 0.04, 0.05};

        // Generate prices
        K_ref = 100.0;
        const size_t Nm = m_grid.size();
        const size_t Nt = tau_grid.size();
        const size_t Nv = sigma_grid.size();
        const size_t Nr = rate_grid.size();

        std::vector<double> prices(Nm * Nt * Nv * Nr);
        for (size_t i = 0; i < Nm; ++i) {
            for (size_t j = 0; j < Nt; ++j) {
                for (size_t k = 0; k < Nv; ++k) {
                    for (size_t l = 0; l < Nr; ++l) {
                        const size_t idx = ((i*Nt + j)*Nv + k)*Nr + l;
                        prices[idx] = analytic_bs(
                            m_grid[i]*K_ref, K_ref, tau_grid[j],
                            sigma_grid[k], rate_grid[l]);
                    }
                }
            }
        }

        // Fit B-spline
        auto fitter_result = BSplineFitter4D::create(m_grid, tau_grid, sigma_grid, rate_grid);
        ASSERT_TRUE(fitter_result.has_value()) << fitter_result.error();

        auto fit_result = fitter_result.value().fit(prices);
        ASSERT_TRUE(fit_result.success) << fit_result.error_message;

        spline = std::make_unique<BSpline4D_FMA>(
            m_grid, tau_grid, sigma_grid, rate_grid, fit_result.coefficients);
    }

    double K_ref;
    std::vector<double> m_grid, tau_grid, sigma_grid, rate_grid;
    std::unique_ptr<BSpline4D_FMA> spline;
};

TEST_F(BSplineVegaTest, EvalPriceAndVegaTriple_MatchesScalar) {
    constexpr double m = 1.05;
    constexpr double tau = 0.75;
    constexpr double sigma = 0.20;
    constexpr double r = 0.04;
    constexpr double epsilon = 1e-4;

    // Scalar reference: 3 separate evals
    double price_down_scalar = spline->eval(m, tau, sigma - epsilon, r);
    double price_scalar = spline->eval(m, tau, sigma, r);
    double price_up_scalar = spline->eval(m, tau, sigma + epsilon, r);
    double vega_scalar = (price_up_scalar - price_down_scalar) / (2.0 * epsilon);

    // NEW METHOD (to be implemented): single-pass triple eval
    double price_triple, vega_triple;
    spline->eval_price_and_vega_triple(m, tau, sigma, r, epsilon, price_triple, vega_triple);

    // Should match scalar within FP tolerance
    EXPECT_NEAR(price_triple, price_scalar, 1e-12);
    EXPECT_NEAR(vega_triple, vega_scalar, 1e-10);  // Slightly looser for derivative
}

TEST_F(BSplineVegaTest, EvalPriceAndVegaTripleSIMD_MatchesScalar) {
    constexpr double m = 1.05;
    constexpr double tau = 0.75;
    constexpr double sigma = 0.20;
    constexpr double r = 0.04;
    constexpr double epsilon = 1e-4;

    // Scalar reference
    double price_scalar, vega_scalar;
    spline->eval_price_and_vega_triple(m, tau, sigma, r, epsilon, price_scalar, vega_scalar);

    // SIMD version (to be implemented)
    double price_simd, vega_simd;
    spline->eval_price_and_vega_triple_simd(m, tau, sigma, r, epsilon, price_simd, vega_simd);

    // Should match scalar within FP rounding tolerance
    EXPECT_NEAR(price_simd, price_scalar, 1e-14);
    EXPECT_NEAR(vega_simd, vega_scalar, 1e-14);
}

} // namespace
