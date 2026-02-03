// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/american_price_surface.hpp"

using namespace mango;

TEST(PriceTableBuilderTest, BuildWithRawPriceMode) {
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_grid = {0.1, 0.25, 0.5, 1.0};
    std::vector<double> vol_grid = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.03, 0.04, 0.05};

    auto setup = PriceTableBuilder<4>::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, 100.0,
        GridAccuracyParams{}, OptionType::PUT);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;

    builder.set_surface_content(SurfaceContent::RawPrice);
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());

    // Verify metadata says RawPrice
    EXPECT_EQ(result->surface->metadata().content, SurfaceContent::RawPrice);
}

TEST(PriceTableBuilderTest, BuildWithCustomIC) {
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_grid = {0.0, 0.1, 0.25, 0.5};  // includes tau=0 (4 points)
    std::vector<double> vol_grid = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.03, 0.04, 0.05};

    auto custom_ic = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(1.0 - std::exp(x[i]), 0.0) + 0.005;
        }
    };

    auto setup = PriceTableBuilder<4>::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, 100.0,
        GridAccuracyParams{}, OptionType::PUT);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;

    builder.set_initial_condition(custom_ic);
    builder.set_allow_tau_zero(true);
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
}

TEST(PriceTableBuilderTest, DefaultBehaviorUnchanged) {
    // Regression: default build (EEP mode, no custom IC) still works
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
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
