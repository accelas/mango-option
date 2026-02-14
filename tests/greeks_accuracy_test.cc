// SPDX-License-Identifier: MIT
/// @file greeks_accuracy_test.cc
/// @brief Integration tests: B-spline surface Greeks vs FDM and Black-Scholes reference

#include <gtest/gtest.h>
#include <cmath>
#include <memory>

#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_3d_accessor.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/math/bspline/bspline_nd_separable.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/bspline/bspline_basis.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

using namespace mango;

// ===========================================================================
// Test fixture: build a B-spline PriceTable once for all accuracy tests
// ===========================================================================

class GreeksAccuracyTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Grid covering a reasonable parameter range for American puts.
        // Log-moneyness: ln(S/K) for S/K in {0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20}
        std::vector<double> log_moneyness = {
            std::log(0.80), std::log(0.85), std::log(0.90), std::log(0.95),
            std::log(1.00), std::log(1.05), std::log(1.10), std::log(1.15),
            std::log(1.20)};
        std::vector<double> maturity = {0.10, 0.25, 0.50, 0.75, 1.00};
        std::vector<double> vol      = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40};
        std::vector<double> rate     = {0.02, 0.03, 0.05, 0.07};

        constexpr double K_ref = 100.0;
        constexpr double div_yield = 0.02;

        // Build with auto-estimated PDE grid + EEP decomposition
        auto setup = PriceTableBuilder::from_vectors(
            log_moneyness, maturity, vol, rate, K_ref,
            GridAccuracyParams{},
            OptionType::PUT,
            div_yield,
            0.0);  // max_failure_rate

        ASSERT_TRUE(setup.has_value())
            << "from_vectors failed: code=" << static_cast<int>(setup.error().code);

        auto& [builder, axes] = *setup;

        auto result = builder.build(axes,
            [&](PriceTensor& tensor, const PriceTableAxes& a) {
                BSplineTensorAccessor accessor(tensor, a, K_ref);
                eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, div_yield));
            });

        ASSERT_TRUE(result.has_value())
            << "build failed: code=" << static_cast<int>(result.error().code);
        ASSERT_NE(result->spline, nullptr);

        auto wrapper = make_bspline_surface(
            result->spline, result->K_ref,
            result->dividends.dividend_yield, OptionType::PUT);
        ASSERT_TRUE(wrapper.has_value())
            << "make_bspline_surface failed: " << wrapper.error();

        surface_ = std::make_unique<BSplinePriceTable>(std::move(*wrapper));
    }

    /// ATM put with dividend yield (exercises early exercise premium path)
    static PricingParams atm_put_with_div() {
        return PricingParams(
            OptionSpec{
                .spot = 100.0, .strike = 100.0, .maturity = 0.5,
                .rate = 0.05, .dividend_yield = 0.02,
                .option_type = OptionType::PUT},
            0.20);
    }

    /// Near-European put: high rate relative to dividend yield, short maturity.
    /// When r >> q and maturity is short, early exercise premium is small,
    /// so American price and Greeks are close to Black-Scholes European.
    /// Uses dividend_yield = 0.02 to match the surface's baked-in dividend.
    static PricingParams near_european_put() {
        return PricingParams(
            OptionSpec{
                .spot = 100.0, .strike = 100.0, .maturity = 0.25,
                .rate = 0.07, .dividend_yield = 0.02,
                .option_type = OptionType::PUT},
            0.20);
    }

    static std::unique_ptr<BSplinePriceTable> surface_;
};

std::unique_ptr<BSplinePriceTable> GreeksAccuracyTest::surface_;

// ===========================================================================
// Delta accuracy: surface vs FDM reference
// ===========================================================================

TEST_F(GreeksAccuracyTest, DeltaMatchesFDM) {
    auto params = atm_put_with_div();

    auto surface_delta = surface_->delta(params);
    ASSERT_TRUE(surface_delta.has_value()) << "surface delta failed";

    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value())
        << "FDM solve failed: " << static_cast<int>(fdm.error().code);
    double fdm_delta = fdm->delta();

    // Within 1% relative error
    double tol = std::abs(fdm_delta) * 0.01;
    EXPECT_NEAR(*surface_delta, fdm_delta, tol)
        << "Surface delta=" << *surface_delta << " FDM delta=" << fdm_delta
        << " rel_err=" << std::abs(*surface_delta - fdm_delta) / std::abs(fdm_delta);
}

// ===========================================================================
// Gamma accuracy: surface vs FDM reference
// ===========================================================================

TEST_F(GreeksAccuracyTest, GammaMatchesFDM) {
    auto params = atm_put_with_div();

    auto surface_gamma = surface_->gamma(params);
    ASSERT_TRUE(surface_gamma.has_value()) << "surface gamma failed";

    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value())
        << "FDM solve failed: " << static_cast<int>(fdm.error().code);
    double fdm_gamma = fdm->gamma();

    // Within 5% relative error (gamma is noisier due to second derivative)
    double tol = std::abs(fdm_gamma) * 0.05;
    EXPECT_NEAR(*surface_gamma, fdm_gamma, tol)
        << "Surface gamma=" << *surface_gamma << " FDM gamma=" << fdm_gamma
        << " rel_err=" << std::abs(*surface_gamma - fdm_gamma) / std::abs(fdm_gamma);
}

// ===========================================================================
// Theta accuracy: surface theta vs finite-difference on surface prices
// ===========================================================================

TEST_F(GreeksAccuracyTest, ThetaReasonable) {
    auto params = atm_put_with_div();

    auto surface_theta = surface_->theta(params);
    ASSERT_TRUE(surface_theta.has_value()) << "surface theta failed";

    // Theta should be negative for options (time decay)
    EXPECT_LT(*surface_theta, 0.0) << "Put theta should be negative (time decay)";

    // Cross-check: compute theta via finite difference on the surface itself
    // theta = -dV/dtau (negative because increasing tau increases value)
    double tau = params.maturity;
    double sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);
    double spot = params.spot;
    double strike = params.strike;
    double dtau = 0.001;

    double price_base = surface_->price(spot, strike, tau, sigma, rate);
    double price_up   = surface_->price(spot, strike, tau + dtau, sigma, rate);
    double fd_theta = -(price_up - price_base) / dtau;

    // Within 10% relative error (theta FD is sensitive to step size)
    double tol = std::abs(fd_theta) * 0.10;
    EXPECT_NEAR(*surface_theta, fd_theta, std::max(tol, 0.01))
        << "Surface theta=" << *surface_theta << " FD theta=" << fd_theta;
}

// ===========================================================================
// Theta vs FDM: compare against two-solve finite difference
// ===========================================================================

TEST_F(GreeksAccuracyTest, ThetaMatchesFDMFiniteDifference) {
    auto params = atm_put_with_div();

    auto surface_theta = surface_->theta(params);
    ASSERT_TRUE(surface_theta.has_value()) << "surface theta failed";

    // Compute FDM theta via two PDE solves at different maturities
    double dtau = 0.001;
    auto fdm1 = solve_american_option(params);
    ASSERT_TRUE(fdm1.has_value());

    PricingParams params2 = params;
    params2.maturity = params.maturity + dtau;
    auto fdm2 = solve_american_option(params2);
    ASSERT_TRUE(fdm2.has_value());

    double fdm_theta = -(fdm2->value() - fdm1->value()) / dtau;

    // Within 10% relative error
    double tol = std::abs(fdm_theta) * 0.10;
    EXPECT_NEAR(*surface_theta, fdm_theta, std::max(tol, 0.01))
        << "Surface theta=" << *surface_theta << " FDM FD theta=" << fdm_theta;
}

// ===========================================================================
// Near-European case: surface Greeks match FDM American, and both are close
// to Black-Scholes analytical (since EEP is small for these parameters)
// ===========================================================================

TEST_F(GreeksAccuracyTest, NearEuropeanDeltaMatchesFDM) {
    // Short maturity, high rate relative to dividend yield -> small EEP
    auto params = near_european_put();

    auto surface_delta = surface_->delta(params);
    ASSERT_TRUE(surface_delta.has_value()) << "surface delta failed";

    // Primary check: surface matches FDM American
    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value());
    double fdm_delta = fdm->delta();

    double tol_fdm = std::abs(fdm_delta) * 0.02;
    EXPECT_NEAR(*surface_delta, fdm_delta, tol_fdm)
        << "Surface delta=" << *surface_delta << " FDM delta=" << fdm_delta;

    // Secondary check: FDM American delta is close to European BS delta
    // (validates our "near-European" assumption)
    auto eu_result = EuropeanOptionSolver(params).solve();
    ASSERT_TRUE(eu_result.has_value());
    double eu_delta = eu_result->delta();

    double tol_eu = std::abs(eu_delta) * 0.05;
    EXPECT_NEAR(fdm_delta, eu_delta, tol_eu)
        << "FDM delta=" << fdm_delta << " BS delta=" << eu_delta
        << " (confirming near-European regime)";
}

TEST_F(GreeksAccuracyTest, NearEuropeanGammaMatchesFDM) {
    auto params = near_european_put();

    auto surface_gamma = surface_->gamma(params);
    ASSERT_TRUE(surface_gamma.has_value()) << "surface gamma failed";

    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value());
    double fdm_gamma = fdm->gamma();

    // Within 5% relative error
    double tol = std::abs(fdm_gamma) * 0.05;
    EXPECT_NEAR(*surface_gamma, fdm_gamma, tol)
        << "Surface gamma=" << *surface_gamma << " FDM gamma=" << fdm_gamma;

    // Sanity: FDM gamma close to European gamma
    auto eu_result = EuropeanOptionSolver(params).solve();
    ASSERT_TRUE(eu_result.has_value());
    EXPECT_NEAR(fdm_gamma, eu_result->gamma(), std::abs(eu_result->gamma()) * 0.10)
        << "FDM gamma=" << fdm_gamma << " BS gamma=" << eu_result->gamma();
}

TEST_F(GreeksAccuracyTest, NearEuropeanThetaReasonable) {
    auto params = near_european_put();

    auto surface_theta = surface_->theta(params);
    ASSERT_TRUE(surface_theta.has_value()) << "surface theta failed";

    // Theta should be negative for puts (time decay)
    EXPECT_LT(*surface_theta, 0.0);

    // Cross-check via FD on the surface
    double tau = params.maturity;
    double sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);
    double dtau = 0.001;
    double price_base = surface_->price(params.spot, params.strike, tau, sigma, rate);
    double price_up   = surface_->price(params.spot, params.strike, tau + dtau, sigma, rate);
    double fd_theta = -(price_up - price_base) / dtau;

    double tol = std::abs(fd_theta) * 0.10;
    EXPECT_NEAR(*surface_theta, fd_theta, std::max(tol, 0.01))
        << "Surface theta=" << *surface_theta << " FD theta=" << fd_theta;
}

TEST_F(GreeksAccuracyTest, NearEuropeanPriceMatchesFDM) {
    auto params = near_european_put();

    double rate = get_zero_rate(params.rate, params.maturity);
    double surface_price = surface_->price(
        params.spot, params.strike, params.maturity,
        params.volatility, rate);

    // Primary check: surface matches FDM American
    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value());
    double fdm_price = fdm->value();

    double tol_fdm = std::abs(fdm_price) * 0.01;
    EXPECT_NEAR(surface_price, fdm_price, tol_fdm)
        << "Surface price=" << surface_price << " FDM price=" << fdm_price;

    // Secondary: FDM American price close to European (confirming small EEP)
    auto eu_result = EuropeanOptionSolver(params).solve();
    ASSERT_TRUE(eu_result.has_value());
    double eu_price = eu_result->value();

    // American >= European always; gap should be small (< 5% of price)
    EXPECT_GE(fdm_price, eu_price - 1e-6);
    EXPECT_LT(fdm_price - eu_price, eu_price * 0.05)
        << "EEP too large for near-European test: FDM=" << fdm_price
        << " EU=" << eu_price;
}

// ===========================================================================
// Off-ATM checks: ITM and OTM delta accuracy
// ===========================================================================

TEST_F(GreeksAccuracyTest, ITMPutDeltaMatchesFDM) {
    PricingParams params(
        OptionSpec{
            .spot = 90.0, .strike = 100.0, .maturity = 0.5,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto surface_delta = surface_->delta(params);
    ASSERT_TRUE(surface_delta.has_value()) << "surface delta failed";

    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value());
    double fdm_delta = fdm->delta();

    // ITM: within 2% relative error
    double tol = std::abs(fdm_delta) * 0.02;
    EXPECT_NEAR(*surface_delta, fdm_delta, tol)
        << "ITM Surface delta=" << *surface_delta << " FDM delta=" << fdm_delta;
}

TEST_F(GreeksAccuracyTest, OTMPutDeltaMatchesFDM) {
    PricingParams params(
        OptionSpec{
            .spot = 110.0, .strike = 100.0, .maturity = 0.5,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto surface_delta = surface_->delta(params);
    ASSERT_TRUE(surface_delta.has_value()) << "surface delta failed";

    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value());
    double fdm_delta = fdm->delta();

    // OTM: within 2% relative error
    double tol = std::abs(fdm_delta) * 0.02;
    EXPECT_NEAR(*surface_delta, fdm_delta, tol)
        << "OTM Surface delta=" << *surface_delta << " FDM delta=" << fdm_delta;
}

// ===========================================================================
// Rho accuracy: surface vs finite-difference on surface prices
// ===========================================================================

TEST_F(GreeksAccuracyTest, RhoConsistentWithFiniteDifference) {
    auto params = atm_put_with_div();

    auto surface_rho = surface_->rho(params);
    ASSERT_TRUE(surface_rho.has_value()) << "surface rho failed";

    // FD rho: bump rate by a small amount
    double rate = get_zero_rate(params.rate, params.maturity);
    double dr = 0.0001;  // 1bp bump
    double price_base = surface_->price(
        params.spot, params.strike, params.maturity, params.volatility, rate);
    double price_up = surface_->price(
        params.spot, params.strike, params.maturity, params.volatility, rate + dr);
    double fd_rho = (price_up - price_base) / dr;

    // Rho should be negative for puts
    EXPECT_LT(*surface_rho, 0.0) << "Put rho should be negative";

    // Within 10% relative error
    double tol = std::abs(fd_rho) * 0.10;
    EXPECT_NEAR(*surface_rho, fd_rho, std::max(tol, 0.01))
        << "Surface rho=" << *surface_rho << " FD rho=" << fd_rho;
}

// ===========================================================================
// Fixture 2: Chebyshev 4D surface Greeks
// Exercises the FD fallback path for gamma (ChebyshevInterpolant has no
// eval_second_partial).
// ===========================================================================

class Chebyshev4DGreeksTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ChebyshevTableConfig config{
            .num_pts = {12, 8, 8, 5},
            .domain = Domain<4>{
                .lo = {-0.30, 0.02, 0.05, 0.01},
                .hi = { 0.30, 2.00, 0.50, 0.10},
            },
            .K_ref = 100.0,
            .option_type = OptionType::PUT,
            .dividend_yield = 0.02,
            // tucker_epsilon=0 forces RawTensor (no Tucker decomposition).
            // Tucker uses Eigen SVD which triggers the known AVX-512 alignment
            // bug when linked with -march=native translation units (see MEMORY.md).
            .tucker_epsilon = 0.0,
        };
        auto result = build_chebyshev_table(config);
        ASSERT_TRUE(result.has_value())
            << "build_chebyshev_table failed";
        surface_ = std::make_unique<ChebyshevTableResult>(std::move(*result));
    }

    static PricingParams atm_put() {
        return PricingParams(
            OptionSpec{
                .spot = 100.0, .strike = 100.0, .maturity = 0.5,
                .rate = 0.05, .dividend_yield = 0.02,
                .option_type = OptionType::PUT},
            0.20);
    }

    static std::unique_ptr<ChebyshevTableResult> surface_;
};

std::unique_ptr<ChebyshevTableResult> Chebyshev4DGreeksTest::surface_;

TEST_F(Chebyshev4DGreeksTest, DeltaMatchesFDM) {
    auto params = atm_put();

    auto surface_delta = surface_->delta(params);
    ASSERT_TRUE(surface_delta.has_value()) << "Chebyshev delta failed";

    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value());
    double fdm_delta = fdm->delta();

    // Chebyshev is less accurate than B-spline: allow 2% relative error
    double tol = std::abs(fdm_delta) * 0.02;
    EXPECT_NEAR(*surface_delta, fdm_delta, tol)
        << "Chebyshev delta=" << *surface_delta << " FDM delta=" << fdm_delta
        << " rel_err=" << std::abs(*surface_delta - fdm_delta) / std::abs(fdm_delta);
}

TEST_F(Chebyshev4DGreeksTest, GammaIsPositiveATM) {
    auto params = atm_put();

    auto surface_gamma = surface_->gamma(params);
    ASSERT_TRUE(surface_gamma.has_value())
        << "Chebyshev gamma failed (FD fallback)";

    EXPECT_GT(*surface_gamma, 0.0)
        << "Gamma should be positive for ATM put";
}

TEST_F(Chebyshev4DGreeksTest, GammaMatchesFDM) {
    auto params = atm_put();

    auto surface_gamma = surface_->gamma(params);
    ASSERT_TRUE(surface_gamma.has_value()) << "Chebyshev gamma failed";

    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value());
    double fdm_gamma = fdm->gamma();

    // FD fallback for gamma is noisier: allow 10% relative error
    double tol = std::abs(fdm_gamma) * 0.10;
    EXPECT_NEAR(*surface_gamma, fdm_gamma, tol)
        << "Chebyshev gamma=" << *surface_gamma << " FDM gamma=" << fdm_gamma
        << " rel_err=" << std::abs(*surface_gamma - fdm_gamma) / std::abs(fdm_gamma);
}

TEST_F(Chebyshev4DGreeksTest, ThetaIsNegative) {
    auto params = atm_put();

    auto surface_theta = surface_->theta(params);
    ASSERT_TRUE(surface_theta.has_value()) << "Chebyshev theta failed";

    EXPECT_LT(*surface_theta, 0.0)
        << "Put theta should be negative (time decay)";
}

TEST_F(Chebyshev4DGreeksTest, RhoIsNegativeForPut) {
    auto params = atm_put();

    auto surface_rho = surface_->rho(params);
    ASSERT_TRUE(surface_rho.has_value()) << "Chebyshev rho failed";

    EXPECT_LT(*surface_rho, 0.0)
        << "Put rho should be negative";
}

// ===========================================================================
// Fixture 3: Dimensionless 3D B-spline surface Greeks
// Exercises DimensionlessTransform3D chain rule weights.
// ===========================================================================

class Dimensionless3DGreeksTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        DimensionlessAxes axes;
        axes.log_moneyness = {-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30};
        axes.tau_prime = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16};
        axes.ln_kappa = {-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8};

        // 1. PDE solve
        auto pde = solve_dimensionless_pde(axes, K_ref_, OptionType::PUT);
        ASSERT_TRUE(pde.has_value())
            << "PDE solve failed: code=" << static_cast<int>(pde.error().code);

        // 2. EEP decompose
        Dimensionless3DAccessor accessor(pde->values, axes, K_ref_);
        eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));

        // 3. Fit B-spline
        std::array<std::vector<double>, 3> grids = {
            axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
        auto fitter = BSplineNDSeparable<double, 3>::create(grids);
        ASSERT_TRUE(fitter.has_value());
        auto fit = fitter->fit(std::move(pde->values));
        ASSERT_TRUE(fit.has_value());

        // 4. Build BSplineND<double, 3>
        std::array<std::vector<double>, 3> bspline_grids = {
            axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
        std::array<std::vector<double>, 3> bspline_knots;
        for (size_t i = 0; i < 3; ++i) {
            bspline_knots[i] = clamped_knots_cubic(bspline_grids[i]);
        }
        auto spline = BSplineND<double, 3>::create(
            bspline_grids, std::move(bspline_knots), std::move(fit->coefficients));
        ASSERT_TRUE(spline.has_value());

        auto spline_ptr = std::make_shared<const BSplineND<double, 3>>(
            std::move(spline.value()));

        // 5. Wrap in layered PriceTable
        SharedBSplineInterp<3> interp(std::move(spline_ptr));
        DimensionlessTransform3D xform;
        BSpline3DTransformLeaf leaf(std::move(interp), xform, K_ref_);
        AnalyticalEEP eep(OptionType::PUT, 0.0);
        BSpline3DLeaf eep_leaf(std::move(leaf), std::move(eep));

        const double sigma_min = 0.10;
        const double sigma_max = 0.80;
        SurfaceBounds bounds{
            .m_min = axes.log_moneyness.front(),
            .m_max = axes.log_moneyness.back(),
            .tau_min = 2.0 * axes.tau_prime.front() / (sigma_max * sigma_max),
            .tau_max = 2.0 * axes.tau_prime.back() / (sigma_min * sigma_min),
            .sigma_min = sigma_min,
            .sigma_max = sigma_max,
            .rate_min = 0.005,
            .rate_max = 0.10,
        };

        table_ = std::make_unique<BSpline3DPriceTable>(
            std::move(eep_leaf), bounds, OptionType::PUT, 0.0);
    }

    /// ATM put with dividend_yield=0.0 (matching the 3D surface)
    static PricingParams atm_put() {
        return PricingParams(
            OptionSpec{
                .spot = 100.0, .strike = 100.0, .maturity = 1.0,
                .rate = 0.05, .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            0.20);
    }

    static constexpr double K_ref_ = 100.0;
    static std::unique_ptr<BSpline3DPriceTable> table_;
};

std::unique_ptr<BSpline3DPriceTable> Dimensionless3DGreeksTest::table_;

TEST_F(Dimensionless3DGreeksTest, DeltaMatchesFDM) {
    auto params = atm_put();

    auto surface_delta = table_->delta(params);
    ASSERT_TRUE(surface_delta.has_value()) << "3D delta failed";

    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value());
    double fdm_delta = fdm->delta();

    // 3D approximation has ~312 bps coupling error: allow 5% relative error
    double tol = std::abs(fdm_delta) * 0.05;
    EXPECT_NEAR(*surface_delta, fdm_delta, tol)
        << "3D delta=" << *surface_delta << " FDM delta=" << fdm_delta
        << " rel_err=" << std::abs(*surface_delta - fdm_delta) / std::abs(fdm_delta);
}

TEST_F(Dimensionless3DGreeksTest, GammaIsPositive) {
    auto params = atm_put();

    auto surface_gamma = table_->gamma(params);
    ASSERT_TRUE(surface_gamma.has_value()) << "3D gamma failed";

    EXPECT_GT(*surface_gamma, 0.0)
        << "Gamma should be positive for ATM put";
}

TEST_F(Dimensionless3DGreeksTest, GammaMatchesFDM) {
    auto params = atm_put();

    auto surface_gamma = table_->gamma(params);
    ASSERT_TRUE(surface_gamma.has_value()) << "3D gamma failed";

    auto fdm = solve_american_option(params);
    ASSERT_TRUE(fdm.has_value());
    double fdm_gamma = fdm->gamma();

    // Broader tolerance for 3D surface: 10% relative error
    double tol = std::abs(fdm_gamma) * 0.10;
    EXPECT_NEAR(*surface_gamma, fdm_gamma, tol)
        << "3D gamma=" << *surface_gamma << " FDM gamma=" << fdm_gamma
        << " rel_err=" << std::abs(*surface_gamma - fdm_gamma) / std::abs(fdm_gamma);
}

TEST_F(Dimensionless3DGreeksTest, ThetaMatchesFD) {
    auto params = atm_put();

    auto surface_theta = table_->theta(params);
    ASSERT_TRUE(surface_theta.has_value()) << "3D theta failed";

    // Theta should be negative (time decay)
    EXPECT_LT(*surface_theta, 0.0)
        << "Put theta should be negative (time decay)";

    // Cross-check via finite difference on the surface
    double rate = get_zero_rate(params.rate, params.maturity);
    double dtau = 0.001;
    double price_base = table_->price(
        params.spot, params.strike, params.maturity, params.volatility, rate);
    double price_up = table_->price(
        params.spot, params.strike, params.maturity + dtau, params.volatility, rate);
    double fd_theta = -(price_up - price_base) / dtau;

    double tol = std::abs(fd_theta) * 0.10;
    EXPECT_NEAR(*surface_theta, fd_theta, std::max(tol, 0.01))
        << "3D theta=" << *surface_theta << " FD theta=" << fd_theta;
}

TEST_F(Dimensionless3DGreeksTest, AllGreeksAreFinite) {
    auto params = atm_put();

    auto d = table_->delta(params);
    auto g = table_->gamma(params);
    auto t = table_->theta(params);
    auto r = table_->rho(params);

    ASSERT_TRUE(d.has_value()) << "delta failed";
    ASSERT_TRUE(g.has_value()) << "gamma failed";
    ASSERT_TRUE(t.has_value()) << "theta failed";
    ASSERT_TRUE(r.has_value()) << "rho failed";

    EXPECT_TRUE(std::isfinite(*d)) << "delta=" << *d;
    EXPECT_TRUE(std::isfinite(*g)) << "gamma=" << *g;
    EXPECT_TRUE(std::isfinite(*t)) << "theta=" << *t;
    EXPECT_TRUE(std::isfinite(*r)) << "rho=" << *r;
}

// ===========================================================================
// Fixture 4: Segmented surface Greeks (discrete dividends)
// Exercises SplitSurface routing with TauSegmentSplit.
// ===========================================================================

namespace {

std::vector<double> log_m_grid_helper(std::initializer_list<double> moneyness) {
    std::vector<double> out;
    out.reserve(moneyness.size());
    for (double m : moneyness) {
        out.push_back(std::log(m));
    }
    return out;
}

}  // namespace

class SegmentedGreeksTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        SegmentedPriceTableBuilder::Config config{
            .K_ref = 100.0,
            .option_type = OptionType::PUT,
            .dividends = {
                .dividend_yield = 0.0,
                .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}},
            },
            .grid = IVGrid{
                .moneyness = log_m_grid_helper({0.80, 0.85, 0.90, 0.95, 1.00,
                                                1.05, 1.10, 1.15, 1.20}),
                .vol = {0.15, 0.20, 0.25, 0.30, 0.40},
                .rate = {0.03, 0.05, 0.07, 0.09},
            },
            .maturity = 1.0,
        };

        auto result = SegmentedPriceTableBuilder::build(config);
        ASSERT_TRUE(result.has_value()) << "Segmented build failed";
        surface_ = std::make_unique<BSplineSegmentedSurface>(std::move(*result));
    }

    /// ATM put querying across the dividend boundary (tau=0.8 > t_div=0.5)
    static PricingParams atm_put() {
        return PricingParams(
            OptionSpec{
                .spot = 100.0, .strike = 100.0, .maturity = 0.8,
                .rate = 0.05, .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            0.20);
    }

    static std::unique_ptr<BSplineSegmentedSurface> surface_;
};

std::unique_ptr<BSplineSegmentedSurface> SegmentedGreeksTest::surface_;

TEST_F(SegmentedGreeksTest, DeltaIsNegativeForPut) {
    auto params = atm_put();

    auto delta = surface_->greek(Greek::Delta, params);
    ASSERT_TRUE(delta.has_value()) << "Segmented delta failed";

    EXPECT_LT(*delta, 0.0)
        << "Put delta should be negative, got " << *delta;
}

TEST_F(SegmentedGreeksTest, GammaIsPositive) {
    auto params = atm_put();

    auto gamma = surface_->gamma(params);
    ASSERT_TRUE(gamma.has_value()) << "Segmented gamma failed";

    EXPECT_GT(*gamma, 0.0)
        << "Gamma should be positive for ATM put, got " << *gamma;
}

TEST_F(SegmentedGreeksTest, PriceIsFiniteAndPositive) {
    auto params = atm_put();
    double rate = get_zero_rate(params.rate, params.maturity);

    double price = surface_->price(
        params.spot, params.strike, params.maturity,
        params.volatility, rate);

    EXPECT_TRUE(std::isfinite(price)) << "Price must be finite";
    EXPECT_GT(price, 0.0) << "ATM put price must be positive";
}
