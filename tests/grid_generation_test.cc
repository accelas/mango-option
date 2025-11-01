#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>

extern "C" {
#include "src/grid_generation.h"
}

class GridGenerationTest : public ::testing::Test {
protected:
    const double tol = 1e-10;

    void ExpectSorted(const double *grid, size_t n) {
        for (size_t i = 0; i < n - 1; i++) {
            EXPECT_LT(grid[i], grid[i+1]) << "Grid not sorted at index " << i;
        }
    }

    void ExpectEndpoints(const double *grid, size_t n, double min, double max) {
        EXPECT_NEAR(grid[0], min, tol) << "First point not equal to min";
        EXPECT_NEAR(grid[n-1], max, tol) << "Last point not equal to max";
    }

    void ExpectInBounds(const double *grid, size_t n, double min, double max) {
        for (size_t i = 0; i < n; i++) {
            EXPECT_GE(grid[i], min - tol) << "Point " << i << " below min";
            EXPECT_LE(grid[i], max + tol) << "Point " << i << " above max";
        }
    }
};

// Uniform Grid Tests

TEST_F(GridGenerationTest, UniformBasic) {
    double *grid = grid_uniform(0.0, 1.0, 11);
    ASSERT_NE(grid, nullptr);

    ExpectEndpoints(grid, 11, 0.0, 1.0);
    ExpectSorted(grid, 11);

    // Check uniform spacing
    GridMetrics metrics = grid_compute_metrics(grid, 11);
    EXPECT_NEAR(metrics.spacing_ratio, 1.0, 0.01) << "Not uniform";
    EXPECT_NEAR(metrics.avg_spacing, 0.1, tol);

    free(grid);
}

TEST_F(GridGenerationTest, UniformNegativeRange) {
    double *grid = grid_uniform(-1.0, 1.0, 21);
    ASSERT_NE(grid, nullptr);

    ExpectEndpoints(grid, 21, -1.0, 1.0);
    ExpectSorted(grid, 21);

    free(grid);
}

TEST_F(GridGenerationTest, UniformInvalidInput) {
    // n < 2
    EXPECT_EQ(grid_uniform(0.0, 1.0, 1), nullptr);

    // max <= min
    EXPECT_EQ(grid_uniform(1.0, 0.0, 10), nullptr);
    EXPECT_EQ(grid_uniform(1.0, 1.0, 10), nullptr);
}

// Logarithmic Grid Tests

TEST_F(GridGenerationTest, LogBasic) {
    double *grid = grid_log(0.1, 10.0, 20);
    ASSERT_NE(grid, nullptr);

    ExpectEndpoints(grid, 20, 0.1, 10.0);
    ExpectSorted(grid, 20);

    // Log spacing: points should be denser near min
    double spacing_near_min = grid[1] - grid[0];
    double spacing_near_max = grid[19] - grid[18];
    EXPECT_LT(spacing_near_min, spacing_near_max);

    free(grid);
}

TEST_F(GridGenerationTest, LogMoneyness) {
    // Typical moneyness range
    double *grid = grid_log(0.7, 1.3, 30);
    ASSERT_NE(grid, nullptr);

    ExpectEndpoints(grid, 30, 0.7, 1.3);
    ExpectSorted(grid, 30);

    free(grid);
}

TEST_F(GridGenerationTest, LogInvalidInput) {
    // min <= 0
    EXPECT_EQ(grid_log(0.0, 1.0, 10), nullptr);
    EXPECT_EQ(grid_log(-0.1, 1.0, 10), nullptr);

    // max <= min
    EXPECT_EQ(grid_log(1.0, 0.5, 10), nullptr);
}

// Chebyshev Grid Tests

TEST_F(GridGenerationTest, ChebyshevBasic) {
    double *grid = grid_chebyshev(-1.0, 1.0, 10);
    ASSERT_NE(grid, nullptr);

    ExpectEndpoints(grid, 10, -1.0, 1.0);
    ExpectSorted(grid, 10);

    // Chebyshev concentrates at boundaries
    GridMetrics metrics = grid_compute_metrics(grid, 10);
    EXPECT_GT(metrics.spacing_ratio, 2.0) << "Should be non-uniform";

    free(grid);
}

TEST_F(GridGenerationTest, ChebyshevSymmetric) {
    double *grid = grid_chebyshev(0.0, 1.0, 15);
    ASSERT_NE(grid, nullptr);

    ExpectEndpoints(grid, 15, 0.0, 1.0);
    ExpectSorted(grid, 15);

    // Check symmetry: spacing near 0 should match spacing near 1
    double spacing_left = grid[1] - grid[0];
    double spacing_right = grid[14] - grid[13];
    EXPECT_NEAR(spacing_left, spacing_right, 0.01);

    free(grid);
}

// Tanh Concentration Tests

TEST_F(GridGenerationTest, TanhCenterATM) {
    // Tanh creates concentration - test that grid is non-uniform
    double *grid = grid_tanh_center(0.7, 1.3, 20, 1.0, 3.0);
    ASSERT_NE(grid, nullptr);

    ExpectEndpoints(grid, 20, 0.7, 1.3);
    ExpectSorted(grid, 20);

    // Grid should be non-uniform (concentration creates varying spacing)
    GridMetrics metrics = grid_compute_metrics(grid, 20);
    EXPECT_GT(metrics.spacing_ratio, 1.5) << "Should have non-uniform spacing";

    free(grid);
}

TEST_F(GridGenerationTest, TanhDifferentStrengths) {
    // Weak concentration
    double *grid_weak = grid_tanh_center(0.7, 1.3, 20, 1.0, 1.0);
    ASSERT_NE(grid_weak, nullptr);

    // Strong concentration
    double *grid_strong = grid_tanh_center(0.7, 1.3, 20, 1.0, 5.0);
    ASSERT_NE(grid_strong, nullptr);

    // Strong should have higher spacing ratio
    GridMetrics metrics_weak = grid_compute_metrics(grid_weak, 20);
    GridMetrics metrics_strong = grid_compute_metrics(grid_strong, 20);
    EXPECT_LT(metrics_weak.spacing_ratio, metrics_strong.spacing_ratio);

    free(grid_weak);
    free(grid_strong);
}

TEST_F(GridGenerationTest, TanhInvalidInput) {
    // Center out of range
    EXPECT_EQ(grid_tanh_center(0.7, 1.3, 20, 0.5, 3.0), nullptr);
    EXPECT_EQ(grid_tanh_center(0.7, 1.3, 20, 1.5, 3.0), nullptr);

    // Invalid strength
    EXPECT_EQ(grid_tanh_center(0.7, 1.3, 20, 1.0, 0.0), nullptr);
    EXPECT_EQ(grid_tanh_center(0.7, 1.3, 20, 1.0, 15.0), nullptr);
}

// Sinh One-Sided Concentration Tests

TEST_F(GridGenerationTest, SinhOneSidedShortMaturity) {
    // Concentrate at tau = 0 (short maturities)
    double *grid = grid_sinh_onesided(0.027, 2.0, 15, 2.5);
    ASSERT_NE(grid, nullptr);

    ExpectEndpoints(grid, 15, 0.027, 2.0);
    ExpectSorted(grid, 15);

    // First spacing should be smaller than last spacing
    double first_spacing = grid[1] - grid[0];
    double last_spacing = grid[14] - grid[13];
    EXPECT_LT(first_spacing, last_spacing);

    free(grid);
}

TEST_F(GridGenerationTest, SinhDifferentStrengths) {
    // Weak concentration
    double *grid_weak = grid_sinh_onesided(0.0, 1.0, 20, 1.0);
    ASSERT_NE(grid_weak, nullptr);

    // Strong concentration
    double *grid_strong = grid_sinh_onesided(0.0, 1.0, 20, 4.0);
    ASSERT_NE(grid_strong, nullptr);

    // Compare first spacings
    double spacing_weak = grid_weak[1] - grid_weak[0];
    double spacing_strong = grid_strong[1] - grid_strong[0];
    EXPECT_LT(spacing_strong, spacing_weak);

    free(grid_weak);
    free(grid_strong);
}

TEST_F(GridGenerationTest, SinhInvalidInput) {
    // Invalid strength
    EXPECT_EQ(grid_sinh_onesided(0.0, 1.0, 20, 0.0), nullptr);
    EXPECT_EQ(grid_sinh_onesided(0.0, 1.0, 20, 15.0), nullptr);
}

// Grid Validation Tests

TEST_F(GridGenerationTest, ValidateCorrectGrid) {
    double *grid = grid_uniform(0.0, 1.0, 11);
    ASSERT_NE(grid, nullptr);

    EXPECT_TRUE(grid_validate(grid, 11, 0.0, 1.0));

    free(grid);
}

TEST_F(GridGenerationTest, ValidateNonUniformGrid) {
    double *grid = grid_tanh_center(0.7, 1.3, 20, 1.0, 3.0);
    ASSERT_NE(grid, nullptr);

    EXPECT_TRUE(grid_validate(grid, 20, 0.7, 1.3));

    free(grid);
}

TEST_F(GridGenerationTest, ValidateUnsortedGrid) {
    double grid[] = {0.0, 0.5, 0.3, 1.0};  // Not sorted
    EXPECT_FALSE(grid_validate(grid, 4, 0.0, 1.0));
}

TEST_F(GridGenerationTest, ValidateDuplicateValues) {
    double grid[] = {0.0, 0.5, 0.5, 1.0};  // Duplicate
    EXPECT_FALSE(grid_validate(grid, 4, 0.0, 1.0));
}

TEST_F(GridGenerationTest, ValidateWrongEndpoints) {
    double grid[] = {0.1, 0.5, 0.9};  // Wrong endpoints
    EXPECT_FALSE(grid_validate(grid, 3, 0.0, 1.0));
}

// Grid Metrics Tests

TEST_F(GridGenerationTest, MetricsUniform) {
    double *grid = grid_uniform(0.0, 1.0, 11);
    ASSERT_NE(grid, nullptr);

    GridMetrics metrics = grid_compute_metrics(grid, 11);

    EXPECT_NEAR(metrics.avg_spacing, 0.1, tol);
    EXPECT_NEAR(metrics.min_spacing, 0.1, tol);
    EXPECT_NEAR(metrics.max_spacing, 0.1, tol);
    EXPECT_NEAR(metrics.spacing_ratio, 1.0, 0.01);

    free(grid);
}

TEST_F(GridGenerationTest, MetricsNonUniform) {
    double *grid = grid_tanh_center(0.7, 1.3, 20, 1.0, 3.0);
    ASSERT_NE(grid, nullptr);

    GridMetrics metrics = grid_compute_metrics(grid, 20);

    EXPECT_GT(metrics.max_spacing, metrics.min_spacing);
    EXPECT_GT(metrics.spacing_ratio, 1.5) << "Should be non-uniform";
    EXPECT_LT(metrics.spacing_ratio, 10.0) << "Shouldn't be too extreme";

    // Average spacing should match total range / (n-1)
    double expected_avg = (1.3 - 0.7) / 19.0;
    EXPECT_NEAR(metrics.avg_spacing, expected_avg, 0.01);

    free(grid);
}

TEST_F(GridGenerationTest, MetricsInvalidInput) {
    GridMetrics metrics = grid_compute_metrics(nullptr, 10);
    EXPECT_TRUE(std::isnan(metrics.avg_spacing));

    metrics = grid_compute_metrics(nullptr, 1);
    EXPECT_TRUE(std::isnan(metrics.avg_spacing));
}

// GridSpec API Tests

TEST_F(GridGenerationTest, GridGenerateUniform) {
    GridSpec spec = {
        .type = GRID_UNIFORM,
        .min = 0.0,
        .max = 1.0,
        .n_points = 11
    };

    double *grid = grid_generate(&spec);
    ASSERT_NE(grid, nullptr);

    EXPECT_TRUE(grid_validate(grid, 11, 0.0, 1.0));

    free(grid);
}

TEST_F(GridGenerationTest, GridGenerateLog) {
    GridSpec spec = {
        .type = GRID_LOG,
        .min = 0.5,
        .max = 2.0,
        .n_points = 20
    };

    double *grid = grid_generate(&spec);
    ASSERT_NE(grid, nullptr);

    EXPECT_TRUE(grid_validate(grid, 20, 0.5, 2.0));

    free(grid);
}

TEST_F(GridGenerationTest, GridGenerateTanh) {
    GridSpec spec = {
        .type = GRID_TANH_CENTER,
        .min = 0.7,
        .max = 1.3,
        .n_points = 25,
        .tanh_params = {
            .center = 1.0,
            .strength = 3.5
        }
    };

    double *grid = grid_generate(&spec);
    ASSERT_NE(grid, nullptr);

    EXPECT_TRUE(grid_validate(grid, 25, 0.7, 1.3));

    free(grid);
}

TEST_F(GridGenerationTest, GridGenerateSinh) {
    GridSpec spec = {
        .type = GRID_SINH_ONESIDED,
        .min = 0.027,
        .max = 2.0,
        .n_points = 20,
        .sinh_params = {
            .strength = 3.0
        }
    };

    double *grid = grid_generate(&spec);
    ASSERT_NE(grid, nullptr);

    EXPECT_TRUE(grid_validate(grid, 20, 0.027, 2.0));

    free(grid);
}

TEST_F(GridGenerationTest, GridGenerateNullSpec) {
    EXPECT_EQ(grid_generate(nullptr), nullptr);
}

// Integration Tests

TEST_F(GridGenerationTest, CompareUniformVsLog) {
    double *uniform = grid_uniform(0.7, 1.3, 30);
    double *log_grid = grid_log(0.7, 1.3, 30);

    ASSERT_NE(uniform, nullptr);
    ASSERT_NE(log_grid, nullptr);

    // Log spacing should be denser near min
    double uniform_first = uniform[1] - uniform[0];
    double log_first = log_grid[1] - log_grid[0];
    EXPECT_LT(log_first, uniform_first);

    // Log spacing should be sparser near max
    double uniform_last = uniform[29] - uniform[28];
    double log_last = log_grid[29] - log_grid[28];
    EXPECT_GT(log_last, uniform_last);

    free(uniform);
    free(log_grid);
}

TEST_F(GridGenerationTest, TanhVsSinhConcentration) {
    // Tanh concentrates at center, sinh concentrates at min
    double *tanh_grid = grid_tanh_center(0.0, 1.0, 20, 0.5, 3.0);
    double *sinh_grid = grid_sinh_onesided(0.0, 1.0, 20, 3.0);

    ASSERT_NE(tanh_grid, nullptr);
    ASSERT_NE(sinh_grid, nullptr);

    // Sinh should have tighter concentration at start
    double tanh_first = tanh_grid[1] - tanh_grid[0];
    double sinh_first = sinh_grid[1] - sinh_grid[0];
    EXPECT_LT(sinh_first, tanh_first);

    free(tanh_grid);
    free(sinh_grid);
}
