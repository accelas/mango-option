// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <cmath>
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/eep_decomposer.hpp"

using namespace mango;

TEST(PriceTableBuilderTest, DefaultBuildProducesNormalizedPrice) {
    std::vector<double> m_grid = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
    std::vector<double> tau_grid = {0.1, 0.25, 0.5, 1.0};
    std::vector<double> vol_grid = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.03, 0.04, 0.05};

    auto setup = PriceTableBuilder::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, 100.0,
        GridAccuracyParams{}, OptionType::PUT);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;

    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());

    // Default build produces NormalizedPrice
    EXPECT_EQ(result->surface->metadata().content, SurfaceContent::NormalizedPrice);
}

TEST(PriceTableBuilderTest, BuildWithEEPTransform) {
    std::vector<double> m_grid = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
    std::vector<double> tau_grid = {0.1, 0.25, 0.5, 1.0};
    std::vector<double> vol_grid = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.03, 0.04, 0.05};

    auto setup = PriceTableBuilder::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, 100.0,
        GridAccuracyParams{}, OptionType::PUT);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;

    // Build with EEP decomposition
    EEPDecomposer decomposer{OptionType::PUT, 100.0, 0.0};
    auto result = builder.build(axes, SurfaceContent::EarlyExercisePremium,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            decomposer.decompose(tensor, a);
        });
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->surface->metadata().content, SurfaceContent::EarlyExercisePremium);
}
