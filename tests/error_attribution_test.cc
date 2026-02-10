// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_slice_cache.hpp"

namespace mango {
namespace {

TEST(ErrorBinsTest, RecordsSingleError) {
    ErrorBins bins;
    // Error at position (0.5, 0.5, 0.5, 0.5) - all middle bins
    bins.record_error({0.5, 0.5, 0.5, 0.5}, 0.001, 0.0005);

    // Error exceeds threshold, should be recorded
    EXPECT_GT(bins.dim_error_mass[0], 0.0);
}

TEST(ErrorBinsTest, IdentifiesWorstDimension) {
    ErrorBins bins;
    // Concentrate errors in dimension 2 (volatility)
    // Dimension 2 = middle region (bin 2)
    for (int i = 0; i < 10; ++i) {
        bins.record_error({0.1 * i, 0.1 * i, 0.5, 0.1 * i}, 0.002, 0.001);
    }

    size_t worst = bins.worst_dimension();
    // Dim 2 has all errors concentrated in bin 2
    // Other dims have errors spread across bins
    // So dim 2 should have highest concentration
    EXPECT_EQ(worst, 2);
}

TEST(ErrorBinsTest, IgnoresErrorsBelowThreshold) {
    ErrorBins bins;
    bins.record_error({0.5, 0.5, 0.5, 0.5}, 0.0001, 0.001);  // Below threshold

    // Should not be recorded
    EXPECT_DOUBLE_EQ(bins.dim_error_mass[0], 0.0);
}

TEST(ErrorBinsTest, FindsProblematicBins) {
    ErrorBins bins;
    // Add errors in bins 0 and 1 of dimension 0
    for (int i = 0; i < 5; ++i) {
        bins.record_error({0.1, 0.5, 0.5, 0.5}, 0.002, 0.001);  // bin 0
        bins.record_error({0.25, 0.5, 0.5, 0.5}, 0.002, 0.001); // bin 1
    }

    auto problematic = bins.problematic_bins(0, 3);
    // Bins 0 and 1 should have count >= 3
    EXPECT_TRUE(std::find(problematic.begin(), problematic.end(), 0) != problematic.end());
    EXPECT_TRUE(std::find(problematic.begin(), problematic.end(), 1) != problematic.end());
}

TEST(ErrorBinsTest, NormalizedPositionOutOfRange) {
    ErrorBins bins;
    // Position outside [0,1] should be clamped
    bins.record_error({-0.1, 1.5, 0.5, 0.5}, 0.002, 0.001);

    // Should still work without crash
    EXPECT_GT(bins.dim_error_mass[0], 0.0);
}

TEST(ErrorBinsTest, ResetClearsAllData) {
    ErrorBins bins;
    bins.record_error({0.5, 0.5, 0.5, 0.5}, 0.002, 0.001);

    EXPECT_GT(bins.dim_error_mass[0], 0.0);

    bins.reset();

    EXPECT_DOUBLE_EQ(bins.dim_error_mass[0], 0.0);
    EXPECT_DOUBLE_EQ(bins.dim_error_mass[1], 0.0);
    EXPECT_DOUBLE_EQ(bins.dim_error_mass[2], 0.0);
    EXPECT_DOUBLE_EQ(bins.dim_error_mass[3], 0.0);

    for (size_t d = 0; d < 4; ++d) {
        for (size_t b = 0; b < 5; ++b) {
            EXPECT_EQ(bins.bin_counts[d][b], 0);
        }
    }
}

TEST(ErrorBinsTest, WorstDimensionWithNoErrors) {
    ErrorBins bins;
    // With no errors recorded, should return 0 (default)
    EXPECT_EQ(bins.worst_dimension(), 0);
}

}  // namespace
}  // namespace mango
