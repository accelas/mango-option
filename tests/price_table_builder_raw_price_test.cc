// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <cmath>
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/american_price_surface.hpp"

using namespace mango;

TEST(PriceTableBuilderTest, BuildWithNormalizedPriceMode) {
    std::vector<double> m_grid = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
    std::vector<double> tau_grid = {0.1, 0.25, 0.5, 1.0};
    std::vector<double> vol_grid = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.03, 0.04, 0.05};

    auto setup = PriceTableBuilder<4>::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, 100.0,
        GridAccuracyParams{}, OptionType::PUT);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;

    builder.set_surface_content(SurfaceContent::NormalizedPrice);
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());

    // Verify metadata says NormalizedPrice
    EXPECT_EQ(result->surface->metadata().content, SurfaceContent::NormalizedPrice);
}

TEST(PriceTableBuilderTest, DefaultBehaviorUnchanged) {
    // Regression: default build (EEP mode, no custom IC) still works
    std::vector<double> m_grid = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
    std::vector<double> tau_grid = {0.1, 0.25, 0.5, 1.0};
    std::vector<double> vol_grid = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.03, 0.04, 0.05};

    auto setup = PriceTableBuilder<4>::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, 100.0,
        GridAccuracyParams{}, OptionType::PUT);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;

    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->surface->metadata().content, SurfaceContent::EarlyExercisePremium);
}
