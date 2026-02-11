// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
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

}  // namespace
}  // namespace mango
