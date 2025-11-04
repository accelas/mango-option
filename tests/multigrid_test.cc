#include "src/cpp/multigrid.hpp"
#include "src/cpp/grid.hpp"
#include <gtest/gtest.h>

TEST(MultiGridBufferTest, TwoAxisCreation) {
    mango::MultiGridBuffer mgrid;

    // Add moneyness axis: log-spaced [0.7, 1.3] with 10 points
    auto m_spec = mango::GridSpec<>::log_spaced(0.7, 1.3, 10);
    mgrid.add_axis(mango::GridAxis::Moneyness, m_spec);

    // Add maturity axis: linear [0.027, 2.0] with 20 points
    auto tau_spec = mango::GridSpec<>::uniform(0.027, 2.0, 20);
    mgrid.add_axis(mango::GridAxis::Maturity, tau_spec);

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
    mgrid.add_axis(mango::GridAxis::Moneyness, m_spec);

    // Get view of moneyness axis
    auto m_view = mgrid.axis_view(mango::GridAxis::Moneyness);

    // Check endpoints
    EXPECT_DOUBLE_EQ(m_view[0], 0.8);
    EXPECT_DOUBLE_EQ(m_view[4], 1.2);

    // Check uniform spacing
    double expected_spacing = (1.2 - 0.8) / 4.0;  // 0.1
    for (size_t i = 0; i < 4; ++i) {
        double spacing = m_view[i+1] - m_view[i];
        EXPECT_NEAR(spacing, expected_spacing, 1e-10);
    }
}
