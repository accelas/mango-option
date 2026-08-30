// SPDX-License-Identifier: MIT
/// @file eep_integration_test.cc
/// @brief End-to-end integration tests for EEP decomposition feature

#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include "mango/option/american_option.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

// ===========================================================================
// End-to-end integration tests for EEP decomposition
// ===========================================================================

/// Build a price table and verify that the reconstructed
/// American price from BSplinePriceTable matches a direct PDE solve.
TEST(EEPIntegrationTest, ReconstructedPriceMatchesPDE) {
    // Grid covering a modest range for the price table
    // Each axis needs >= 4 points for B-spline fitting
    std::vector<double> log_moneyness = {std::log(0.90), std::log(0.95), std::log(1.00), std::log(1.05), std::log(1.10)};
    std::vector<double> maturity  = {0.25, 0.50, 0.75, 1.00};
    std::vector<double> vol       = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate      = {0.02, 0.03, 0.04, 0.05};

    double K_ref = 100.0;

    // Build with auto-estimated PDE grid, then apply EEP decomposition
    auto setup = PriceTableBuilder::from_vectors(
        log_moneyness, maturity, vol, rate, K_ref,
        GridAccuracyParams{},   // auto-estimate PDE grid
        OptionType::PUT,
        0.0,   // dividend_yield
        0.0);  // max_failure_rate

    ASSERT_TRUE(setup.has_value())
        << "from_vectors failed: code=" << static_cast<int>(setup.error().code);

    auto& [builder, axes] = *setup;
    auto result = builder.build(axes,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            BSplineTensorAccessor accessor(tensor, a, K_ref);
            eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));
        });
    ASSERT_TRUE(result.has_value())
        << "build failed: code=" << static_cast<int>(result.error().code);
    ASSERT_NE(result->spline, nullptr);


    // Wrap in BSplinePriceTable for reconstruction
    auto wrapper_result = make_bspline_surface(result->spline, result->K_ref, result->dividends.dividend_yield, OptionType::PUT);
    ASSERT_TRUE(wrapper_result.has_value())
        << "make_bspline_surface failed: " << wrapper_result.error();
    auto wrapper = std::move(*wrapper_result);

    // Test point: ATM put, 1-year, 20% vol, 5% rate
    double S     = 100.0;
    double K     = 100.0;
    double tau   = 1.0;
    double sigma = 0.20;
    double r     = 0.05;

    double reconstructed = wrapper.price(S, K, tau, sigma, r);
    EXPECT_GT(reconstructed, 0.0) << "Reconstructed price should be positive";

    // Direct PDE solve for comparison
    PricingParams params(OptionSpec{.spot = S, .strike = K, .maturity = tau, .rate = r, .option_type = OptionType::PUT}, sigma);
    auto solver = AmericanOptionSolver::create(params).value();
    auto pde_result = solver.solve();
    ASSERT_TRUE(pde_result.has_value())
        << "PDE solve failed: " << static_cast<int>(pde_result.error().code);

    double pde_price = pde_result->value();
    EXPECT_GT(pde_price, 0.0);

    // Compare: reconstructed should be within ~1% of PDE price
    double tol = 0.01 * pde_price;
    EXPECT_NEAR(reconstructed, pde_price, tol)
        << "Reconstructed=" << reconstructed
        << " PDE=" << pde_price
        << " tolerance=" << tol;
}

TEST(EEPIntegrationTest, ControlledPremiumIsPreservedAcrossTypesAndExpiries) {
    constexpr double kKRef = 100.0;
    constexpr double kKnownPremium = 0.02;
    constexpr double kSigma = 0.20;
    constexpr double kRate = 0.05;

    const std::vector<double> log_moneyness = {
        std::log(0.90),
        std::log(0.95),
        0.0,
        std::log(1.05),
        std::log(1.10),
    };
    const std::vector<double> maturity = {0.04, 0.25, 0.50, 1.00};
    const std::vector<double> volatility = {0.15, 0.20, 0.25, 0.30};
    const std::vector<double> rate = {0.02, 0.03, 0.04, 0.05};

    for (OptionType type : {OptionType::CALL, OptionType::PUT}) {
        SCOPED_TRACE(::testing::Message()
                     << "option_type=" << static_cast<int>(type));

        auto setup = PriceTableBuilder::from_vectors(
            log_moneyness,
            maturity,
            volatility,
            rate,
            kKRef,
            GridAccuracyParams{},
            type,
            0.0,
            0.0);
        ASSERT_TRUE(setup.has_value())
            << "from_vectors failed: code="
            << static_cast<int>(setup.error().code);

        auto& [builder, axes] = *setup;
        auto result = builder.build(
            axes,
            [=](PriceTensor& tensor, const PriceTableAxes& current_axes) {
                const AnalyticalEEP analytical(type, 0.0);
                for (size_t mi = 0; mi < current_axes.grids[0].size(); ++mi) {
                    for (size_t ti = 0; ti < current_axes.grids[1].size(); ++ti) {
                        for (size_t vi = 0; vi < current_axes.grids[2].size(); ++vi) {
                            for (size_t ri = 0; ri < current_axes.grids[3].size(); ++ri) {
                                const double spot =
                                    std::exp(current_axes.grids[0][mi]) * kKRef;
                                const double european = analytical.european_price(
                                    spot,
                                    kKRef,
                                    current_axes.grids[1][ti],
                                    current_axes.grids[2][vi],
                                    current_axes.grids[3][ri]);
                                tensor.view[mi, ti, vi, ri] =
                                    (european + kKnownPremium) / kKRef;
                            }
                        }
                    }
                }

                BSplineTensorAccessor accessor(
                    tensor, current_axes, kKRef);
                eep_decompose(accessor, analytical);
            });
        ASSERT_TRUE(result.has_value())
            << "build failed: code="
            << static_cast<int>(result.error().code);

        auto surface = make_bspline_surface(
            result->spline,
            result->K_ref,
            result->dividends.dividend_yield,
            type);
        ASSERT_TRUE(surface.has_value())
            << "make_bspline_surface failed: " << surface.error();

        const AnalyticalEEP analytical(type, 0.0);
        for (double tau : {0.04, 1.00}) {
            SCOPED_TRACE(::testing::Message() << "tau=" << tau);
            const double european = analytical.european_price(
                kKRef, kKRef, tau, kSigma, kRate);
            const double reconstructed =
                surface->price(kKRef, kKRef, tau, kSigma, kRate);

            EXPECT_NEAR(
                reconstructed,
                european + kKnownPremium,
                1.0e-8);
            EXPECT_GE(reconstructed, european);
        }
    }
}

/// Build a price table and verify that the raw EEP
/// surface produces non-negative values at all grid points.
/// The construction-time projection should guarantee this.
TEST(EEPIntegrationTest, ProjectionEnsuresNonNegative) {
    // Small grid — each axis needs >= 4 points for B-spline fitting
    std::vector<double> log_moneyness = {std::log(0.90), std::log(0.95), std::log(1.00), std::log(1.10)};
    std::vector<double> maturity  = {0.25, 0.50, 0.75, 1.00};
    std::vector<double> vol       = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate      = {0.02, 0.03, 0.04, 0.05};

    double K_ref = 100.0;

    auto setup = PriceTableBuilder::from_vectors(
        log_moneyness, maturity, vol, rate, K_ref,
        GridAccuracyParams{},
        OptionType::PUT,
        0.0,   // dividend_yield
        0.0);  // max_failure_rate

    ASSERT_TRUE(setup.has_value())
        << "from_vectors failed: code=" << static_cast<int>(setup.error().code);

    auto& [builder, axes] = *setup;
    auto result = builder.build(axes,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            BSplineTensorAccessor accessor(tensor, a, K_ref);
            eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));
        });
    ASSERT_TRUE(result.has_value())
        << "build failed: code=" << static_cast<int>(result.error().code);
    ASSERT_NE(result->spline, nullptr);


    // Query the raw EEP spline at every grid point combination.
    // The B-spline is fitted to projected EEP data, so values at
    // grid points should be non-negative (or very close due to fitting error).
    const auto& spline = *result->spline;

    size_t negative_count = 0;
    double most_negative = 0.0;

    for (double m : spline.grid(0)) {
        for (double tau : spline.grid(1)) {
            for (double sigma : spline.grid(2)) {
                for (double r : spline.grid(3)) {
                    double val = spline.eval({m, tau, sigma, r});
                    if (val < -1e-10) {
                        ++negative_count;
                        most_negative = std::min(most_negative, val);
                    }
                }
            }
        }
    }

    EXPECT_EQ(negative_count, 0u)
        << "Found " << negative_count << " negative EEP values; "
        << "most negative = " << most_negative;
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Note: The old MakeStandardWrapperRejectsNormalizedPrice test was removed.
// SurfaceContent enum was deleted — the type system now enforces correct
// EEP reconstruction via EEPLayer<..., AnalyticalEEP> vs TransformLeaf.

}  // namespace
}  // namespace mango
