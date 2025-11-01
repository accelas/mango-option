#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "src/grid_presets.h"
}

class GridPresetsTest : public ::testing::Test {};

TEST_F(GridPresetsTest, PresetUniform) {
    GridConfig config = grid_preset_get(
        GRID_PRESET_UNIFORM,
        0.7, 1.3,      // moneyness
        0.027, 2.0,    // maturity
        0.10, 0.80,    // volatility
        0.0, 0.10,     // rate
        0.0, 0.0);     // no dividend

    EXPECT_EQ(config.moneyness.type, GRID_UNIFORM);
    EXPECT_EQ(config.moneyness.n_points, 30);
    EXPECT_EQ(config.maturity.n_points, 25);
    EXPECT_EQ(config.volatility.n_points, 15);
    EXPECT_EQ(config.rate.n_points, 10);
    EXPECT_EQ(config.dividend.n_points, 0);
}

TEST_F(GridPresetsTest, PresetLogStandard) {
    GridConfig config = grid_preset_get(
        GRID_PRESET_LOG_STANDARD,
        0.7, 1.3, 0.027, 2.0, 0.10, 0.80, 0.0, 0.10, 0.0, 0.0);

    EXPECT_EQ(config.moneyness.type, GRID_LOG);
    EXPECT_EQ(config.moneyness.n_points, 30);
}

TEST_F(GridPresetsTest, PresetAdaptiveFast) {
    GridConfig config = grid_preset_get(
        GRID_PRESET_ADAPTIVE_FAST,
        0.7, 1.3, 0.027, 2.0, 0.10, 0.80, 0.0, 0.10, 0.0, 0.0);

    EXPECT_EQ(config.moneyness.type, GRID_TANH_CENTER);
    EXPECT_EQ(config.moneyness.n_points, 12);
    EXPECT_EQ(config.maturity.type, GRID_SINH_ONESIDED);
    EXPECT_EQ(config.maturity.n_points, 10);
    EXPECT_EQ(config.volatility.n_points, 8);
    EXPECT_EQ(config.rate.n_points, 5);

    // Total: 12 × 10 × 8 × 5 = 4,800
}

TEST_F(GridPresetsTest, PresetAdaptiveBalanced) {
    GridConfig config = grid_preset_get(
        GRID_PRESET_ADAPTIVE_BALANCED,
        0.7, 1.3, 0.027, 2.0, 0.10, 0.80, 0.0, 0.10, 0.0, 0.0);

    EXPECT_EQ(config.moneyness.n_points, 20);
    EXPECT_EQ(config.maturity.n_points, 15);
    EXPECT_EQ(config.volatility.n_points, 10);
    EXPECT_EQ(config.rate.n_points, 5);

    // Total: 20 × 15 × 10 × 5 = 15,000
}

TEST_F(GridPresetsTest, PresetAdaptiveAccurate) {
    GridConfig config = grid_preset_get(
        GRID_PRESET_ADAPTIVE_ACCURATE,
        0.7, 1.3, 0.027, 2.0, 0.10, 0.80, 0.0, 0.10, 0.0, 0.0);

    EXPECT_EQ(config.moneyness.n_points, 25);
    EXPECT_EQ(config.maturity.n_points, 20);
    EXPECT_EQ(config.volatility.n_points, 12);
    EXPECT_EQ(config.rate.n_points, 5);

    // Total: 25 × 20 × 12 × 5 = 30,000
}

TEST_F(GridPresetsTest, GenerateAllGrids4D) {
    GridConfig config = grid_preset_get(
        GRID_PRESET_ADAPTIVE_FAST,
        0.7, 1.3, 0.027, 2.0, 0.10, 0.80, 0.0, 0.10, 0.0, 0.0);

    GeneratedGrids grids = grid_generate_all(&config);

    EXPECT_NE(grids.moneyness, nullptr);
    EXPECT_EQ(grids.n_moneyness, 12);

    EXPECT_NE(grids.maturity, nullptr);
    EXPECT_EQ(grids.n_maturity, 10);

    EXPECT_NE(grids.volatility, nullptr);
    EXPECT_EQ(grids.n_volatility, 8);

    EXPECT_NE(grids.rate, nullptr);
    EXPECT_EQ(grids.n_rate, 5);

    EXPECT_EQ(grids.dividend, nullptr);
    EXPECT_EQ(grids.n_dividend, 0);

    EXPECT_EQ(grids.total_points, 4800);

    // Validate generated grids
    EXPECT_DOUBLE_EQ(grids.moneyness[0], 0.7);
    EXPECT_DOUBLE_EQ(grids.moneyness[11], 1.3);

    EXPECT_DOUBLE_EQ(grids.maturity[0], 0.027);
    EXPECT_DOUBLE_EQ(grids.maturity[9], 2.0);

    grid_free_all(&grids);

    // After free, pointers should be NULL
    EXPECT_EQ(grids.moneyness, nullptr);
    EXPECT_EQ(grids.maturity, nullptr);
}

TEST_F(GridPresetsTest, GenerateAllGrids5D) {
    GridConfig config = grid_preset_get(
        GRID_PRESET_ADAPTIVE_BALANCED,
        0.7, 1.3, 0.027, 2.0, 0.10, 0.80, 0.0, 0.10,
        0.0, 0.05);  // With dividend

    GeneratedGrids grids = grid_generate_all(&config);

    EXPECT_EQ(grids.n_moneyness, 20);
    EXPECT_EQ(grids.n_maturity, 15);
    EXPECT_EQ(grids.n_volatility, 10);
    EXPECT_EQ(grids.n_rate, 5);
    EXPECT_EQ(grids.n_dividend, 4);

    // Total: 20 × 15 × 10 × 5 × 4 = 60,000
    EXPECT_EQ(grids.total_points, 60000);

    EXPECT_NE(grids.dividend, nullptr);
    EXPECT_DOUBLE_EQ(grids.dividend[0], 0.0);
    EXPECT_DOUBLE_EQ(grids.dividend[3], 0.05);

    grid_free_all(&grids);
}

TEST_F(GridPresetsTest, PresetNames) {
    EXPECT_STREQ(grid_preset_name(GRID_PRESET_UNIFORM), "Uniform");
    EXPECT_STREQ(grid_preset_name(GRID_PRESET_ADAPTIVE_FAST), "Adaptive Fast");
    EXPECT_STREQ(grid_preset_name(GRID_PRESET_ADAPTIVE_BALANCED), "Adaptive Balanced");
    EXPECT_STREQ(grid_preset_name(GRID_PRESET_ADAPTIVE_ACCURATE), "Adaptive Accurate");
}

TEST_F(GridPresetsTest, PresetDescriptions) {
    const char* desc = grid_preset_description(GRID_PRESET_ADAPTIVE_BALANCED);
    EXPECT_NE(desc, nullptr);
    EXPECT_GT(strlen(desc), 0);
}

TEST_F(GridPresetsTest, ComparePresetSizes) {
    // Uniform baseline
    GridConfig uniform = grid_preset_get(
        GRID_PRESET_UNIFORM,
        0.7, 1.3, 0.027, 2.0, 0.10, 0.80, 0.0, 0.10, 0.0, 0.0);
    GeneratedGrids grids_uniform = grid_generate_all(&uniform);

    // Adaptive Fast
    GridConfig fast = grid_preset_get(
        GRID_PRESET_ADAPTIVE_FAST,
        0.7, 1.3, 0.027, 2.0, 0.10, 0.80, 0.0, 0.10, 0.0, 0.0);
    GeneratedGrids grids_fast = grid_generate_all(&fast);

    // Fast should be significantly smaller
    EXPECT_EQ(grids_uniform.total_points, 112500);  // 30×25×15×10
    EXPECT_EQ(grids_fast.total_points, 4800);       // 12×10×8×5

    // Fast is ~23× smaller
    double ratio = (double)grids_uniform.total_points / (double)grids_fast.total_points;
    EXPECT_NEAR(ratio, 23.4, 1.0);

    grid_free_all(&grids_uniform);
    grid_free_all(&grids_fast);
}
