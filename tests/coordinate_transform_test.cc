#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "../src/price_table.h"

// Expose internal function for testing
void transform_query_to_grid(
    CoordinateSystem coord_system,
    double m_raw, double tau_raw, double sigma_raw, double r_raw,
    double *m_grid, double *tau_grid, double *sigma_grid, double *r_grid);
}

TEST(CoordinateSystemTest, EnumValues) {
    EXPECT_EQ(COORD_RAW, 0);
    EXPECT_EQ(COORD_LOG_SQRT, 1);
    EXPECT_EQ(COORD_LOG_VARIANCE, 2);
}

TEST(MemoryLayoutTest, EnumValues) {
    EXPECT_EQ(LAYOUT_M_OUTER, 0);
    EXPECT_EQ(LAYOUT_M_INNER, 1);
    EXPECT_EQ(LAYOUT_BLOCKED, 2);
}

TEST(TransformTest, RawPassthrough) {
    double m_grid, tau_grid, sigma_grid, r_grid;
    transform_query_to_grid(COORD_RAW, 1.05, 0.5, 0.25, 0.03,
                            &m_grid, &tau_grid, &sigma_grid, &r_grid);

    EXPECT_DOUBLE_EQ(m_grid, 1.05);
    EXPECT_DOUBLE_EQ(tau_grid, 0.5);
    EXPECT_DOUBLE_EQ(sigma_grid, 0.25);
    EXPECT_DOUBLE_EQ(r_grid, 0.03);
}

TEST(TransformTest, LogSqrtTransform) {
    double m_grid, tau_grid, sigma_grid, r_grid;
    transform_query_to_grid(COORD_LOG_SQRT, 1.05, 0.5, 0.25, 0.03,
                            &m_grid, &tau_grid, &sigma_grid, &r_grid);

    EXPECT_NEAR(m_grid, log(1.05), 1e-10);
    EXPECT_NEAR(tau_grid, sqrt(0.5), 1e-10);
    EXPECT_DOUBLE_EQ(sigma_grid, 0.25);
    EXPECT_DOUBLE_EQ(r_grid, 0.03);
}

TEST(TransformTest, ZeroMoneynessHandling) {
    double m_grid, tau_grid, sigma_grid, r_grid;
    transform_query_to_grid(COORD_LOG_SQRT, 0.0, 0.5, 0.25, 0.03,
                            &m_grid, &tau_grid, &sigma_grid, &r_grid);

    EXPECT_TRUE(std::isinf(m_grid));  // log(0) = -inf
}
