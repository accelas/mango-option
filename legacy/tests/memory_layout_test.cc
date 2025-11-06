#include <gtest/gtest.h>

extern "C" {
#include "../src/price_table.h"
}

// Test helper to create minimal table
static OptionPriceTable* create_test_table(MemoryLayout layout) {
    double m[] = {0.9, 1.0, 1.1};
    double tau[] = {0.25, 0.5};
    double sigma[] = {0.2, 0.3};
    double r[] = {0.02, 0.05};

    return price_table_create_ex(
        m, 3, tau, 2, sigma, 2, r, 2, nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW, layout);
}

TEST(StrideCalculationTest, LayoutMOuter) {
    OptionPriceTable *table = create_test_table(LAYOUT_M_OUTER);

    // [m][tau][sigma][r] order
    EXPECT_EQ(table->stride_m, 2 * 2 * 2);  // n_tau * n_sigma * n_r = 8
    EXPECT_EQ(table->stride_tau, 2 * 2);    // n_sigma * n_r = 4
    EXPECT_EQ(table->stride_sigma, 2);      // n_r = 2
    EXPECT_EQ(table->stride_r, 1);
    EXPECT_EQ(table->stride_q, 0);          // 4D mode

    price_table_destroy(table);
}

TEST(StrideCalculationTest, LayoutMInner) {
    OptionPriceTable *table = create_test_table(LAYOUT_M_INNER);

    // [r][sigma][tau][m] order
    EXPECT_EQ(table->stride_m, 1);          // Innermost
    EXPECT_EQ(table->stride_tau, 3);        // n_m = 3
    EXPECT_EQ(table->stride_sigma, 3 * 2);  // n_m * n_tau = 6
    EXPECT_EQ(table->stride_r, 3 * 2 * 2);  // n_m * n_tau * n_sigma = 12
    EXPECT_EQ(table->stride_q, 0);

    price_table_destroy(table);
}

TEST(SliceExtractionTest, MoneynessSliceContiguous) {
    OptionPriceTable *table = create_test_table(LAYOUT_M_INNER);

    // Populate some test data
    for (size_t i = 0; i < 3; i++) {
        price_table_set(table, i, 0, 0, 0, 0, 100.0 + i);
    }

    double slice[3];
    bool contiguous;
    int fixed[] = {-1, 0, 0, 0, 0};  // Vary moneyness, fix others

    int status = price_table_extract_slice(
        table, SLICE_DIM_MONEYNESS, fixed, slice, &contiguous);

    EXPECT_EQ(status, 0);
    EXPECT_TRUE(contiguous);  // LAYOUT_M_INNER → stride_m = 1
    EXPECT_DOUBLE_EQ(slice[0], 100.0);
    EXPECT_DOUBLE_EQ(slice[1], 101.0);
    EXPECT_DOUBLE_EQ(slice[2], 102.0);

    price_table_destroy(table);
}

TEST(SliceExtractionTest, MoneynessSliceStrided) {
    OptionPriceTable *table = create_test_table(LAYOUT_M_OUTER);

    // Populate test data
    for (size_t i = 0; i < 3; i++) {
        price_table_set(table, i, 0, 0, 0, 0, 200.0 + i);
    }

    double slice[3];
    bool contiguous;
    int fixed[] = {-1, 0, 0, 0, 0};

    int status = price_table_extract_slice(
        table, SLICE_DIM_MONEYNESS, fixed, slice, &contiguous);

    EXPECT_EQ(status, 0);
    EXPECT_FALSE(contiguous);  // LAYOUT_M_OUTER → stride_m = 8
    EXPECT_DOUBLE_EQ(slice[0], 200.0);
    EXPECT_DOUBLE_EQ(slice[1], 201.0);
    EXPECT_DOUBLE_EQ(slice[2], 202.0);

    price_table_destroy(table);
}
