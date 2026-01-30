// SPDX-License-Identifier: MIT
#include "src/support/error_types.hpp"
#include <gtest/gtest.h>
#include <sstream>

using namespace mango;

TEST(InterpolationErrorTest, WorkspaceCreationFailedCode) {
    InterpolationError err(InterpolationErrorCode::WorkspaceCreationFailed,
                           "Buffer too small: 100 < 200 required");

    EXPECT_EQ(err.code, InterpolationErrorCode::WorkspaceCreationFailed);
    EXPECT_EQ(err.message, "Buffer too small: 100 < 200 required");
}

TEST(InterpolationErrorTest, BackwardCompatibleConstructor) {
    // Old-style construction should still work
    InterpolationError err(InterpolationErrorCode::FittingFailed, 100, 5, 0.001);

    EXPECT_EQ(err.code, InterpolationErrorCode::FittingFailed);
    EXPECT_EQ(err.grid_size, 100u);
    EXPECT_EQ(err.index, 5u);
    EXPECT_DOUBLE_EQ(err.max_residual, 0.001);
    EXPECT_TRUE(err.message.empty());
}

TEST(InterpolationErrorTest, OutputStreamWithMessage) {
    InterpolationError err(InterpolationErrorCode::WorkspaceCreationFailed,
                           "Test message");

    std::ostringstream oss;
    oss << err;
    std::string output = oss.str();

    // The operator<< outputs the code as an integer, not the enum name
    // Just verify the message is present
    EXPECT_NE(output.find("Test message"), std::string::npos);
    EXPECT_NE(output.find("code="), std::string::npos);
}

TEST(InterpolationErrorTest, ConvertToPriceTableError) {
    InterpolationError err(InterpolationErrorCode::WorkspaceCreationFailed,
                           "Buffer allocation failed");

    PriceTableError pte = convert_to_price_table_error(err);

    EXPECT_EQ(pte.code, PriceTableErrorCode::ArenaAllocationFailed);
}
