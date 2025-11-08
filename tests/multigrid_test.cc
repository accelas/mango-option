#include "src/multigrid.hpp"
#include "src/grid.hpp"
#include <gtest/gtest.h>

TEST(MultiGridBufferTest, TwoAxisCreation) {
    mango::MultiGridBuffer mgrid;

    // Add moneyness axis: log-spaced [0.7, 1.3] with 10 points
    auto m_spec = mango::GridSpec<>::log_spaced(0.7, 1.3, 10);
    ASSERT_TRUE(m_spec.has_value());
    auto result1 = mgrid.add_axis(mango::GridAxis::Moneyness, *m_spec);
    EXPECT_TRUE(result1.has_value());

    // Add maturity axis: linear [0.027, 2.0] with 20 points
    auto tau_spec = mango::GridSpec<>::uniform(0.027, 2.0, 20);
    ASSERT_TRUE(tau_spec.has_value());
    auto result2 = mgrid.add_axis(mango::GridAxis::Maturity, *tau_spec);
    EXPECT_TRUE(result2.has_value());

    // Verify axes were added
    EXPECT_TRUE(mgrid.has_axis(mango::GridAxis::Moneyness));
    EXPECT_TRUE(mgrid.has_axis(mango::GridAxis::Maturity));
    EXPECT_FALSE(mgrid.has_axis(mango::GridAxis::Volatility));

    // Verify axis sizes
    EXPECT_EQ(mgrid.axis_size(mango::GridAxis::Moneyness), 10);
    EXPECT_EQ(mgrid.axis_size(mango::GridAxis::Maturity), 20);

    // Verify total grid points
    EXPECT_EQ(mgrid.total_points(), 200);  // 10 × 20
}

TEST(MultiGridBufferTest, AccessAxisData) {
    mango::MultiGridBuffer mgrid;

    auto m_spec = mango::GridSpec<>::uniform(0.8, 1.2, 5);
    ASSERT_TRUE(m_spec.has_value());
    auto result = mgrid.add_axis(mango::GridAxis::Moneyness, *m_spec);
    EXPECT_TRUE(result.has_value());

    // Get view of moneyness axis
    auto m_view = mgrid.axis_view(mango::GridAxis::Moneyness);
    ASSERT_TRUE(m_view.has_value());
    auto m_view_span = *m_view;

    // Check endpoints
    EXPECT_DOUBLE_EQ(m_view_span[0], 0.8);
    EXPECT_DOUBLE_EQ(m_view_span[4], 1.2);

    // Check uniform spacing
    double expected_spacing = (1.2 - 0.8) / 4.0;  // 0.1
    for (size_t i = 0; i < 4; ++i) {
        double spacing = m_view_span[i+1] - m_view_span[i];
        EXPECT_NEAR(spacing, expected_spacing, 1e-10);
    }
}

TEST(MultiGridBufferTest, FiveDimensionalPriceTable) {
    mango::MultiGridBuffer mgrid;

    // 5D price table: moneyness × maturity × volatility × rate × dividend
    auto moneyness_spec = mango::GridSpec<>::log_spaced(0.7, 1.3, 50);
    ASSERT_TRUE(moneyness_spec.has_value());
    auto result1 = mgrid.add_axis(mango::GridAxis::Moneyness,  *moneyness_spec);
    EXPECT_TRUE(result1.has_value());

    auto maturity_spec = mango::GridSpec<>::uniform(0.027, 2.0, 30);
    ASSERT_TRUE(maturity_spec.has_value());
    auto result2 = mgrid.add_axis(mango::GridAxis::Maturity,   *maturity_spec);
    EXPECT_TRUE(result2.has_value());

    auto volatility_spec = mango::GridSpec<>::uniform(0.10, 0.80, 20);
    ASSERT_TRUE(volatility_spec.has_value());
    auto result3 = mgrid.add_axis(mango::GridAxis::Volatility, *volatility_spec);
    EXPECT_TRUE(result3.has_value());

    auto rate_spec = mango::GridSpec<>::uniform(0.0, 0.10, 10);
    ASSERT_TRUE(rate_spec.has_value());
    auto result4 = mgrid.add_axis(mango::GridAxis::Rate,       *rate_spec);
    EXPECT_TRUE(result4.has_value());

    auto dividend_spec = mango::GridSpec<>::uniform(0.0, 0.05, 5);
    ASSERT_TRUE(dividend_spec.has_value());
    auto result5 = mgrid.add_axis(mango::GridAxis::Dividend,   *dividend_spec);
    EXPECT_TRUE(result5.has_value());

    // Verify all axes present
    EXPECT_EQ(mgrid.n_axes(), 5);
    EXPECT_TRUE(mgrid.has_axis(mango::GridAxis::Dividend));

    // Verify total points
    size_t expected_total = 50 * 30 * 20 * 10 * 5;  // 1,500,000 points
    EXPECT_EQ(mgrid.total_points(), expected_total);

    // Verify dividend axis spacing
    auto div_view = mgrid.axis_view(mango::GridAxis::Dividend);
    ASSERT_TRUE(div_view.has_value());
    auto div_view_span = *div_view;
    EXPECT_EQ(div_view_span.size(), 5);
    EXPECT_DOUBLE_EQ(div_view_span[0], 0.0);
    EXPECT_DOUBLE_EQ(div_view_span[4], 0.05);
}
