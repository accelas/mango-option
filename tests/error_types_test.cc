// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/support/error_types.hpp"

using namespace mango;

// ===========================================================================
// ValidationError -> IVError conversion
// ===========================================================================

TEST(ErrorConversionTest, ValidationInvalidSpotToIVError) {
    ValidationError err(ValidationErrorCode::InvalidSpotPrice, -100.0);
    IVError iv_err = convert_to_iv_error(err);
    EXPECT_EQ(iv_err.code, IVErrorCode::NegativeSpot);
}

TEST(ErrorConversionTest, ValidationInvalidStrikeToIVError) {
    ValidationError err(ValidationErrorCode::InvalidStrike, 0.0);
    IVError iv_err = convert_to_iv_error(err);
    EXPECT_EQ(iv_err.code, IVErrorCode::NegativeStrike);
}

TEST(ErrorConversionTest, ValidationInvalidMaturityToIVError) {
    ValidationError err(ValidationErrorCode::InvalidMaturity, -1.0);
    IVError iv_err = convert_to_iv_error(err);
    EXPECT_EQ(iv_err.code, IVErrorCode::NegativeMaturity);
}

TEST(ErrorConversionTest, ValidationInvalidMarketPriceToIVError) {
    ValidationError err(ValidationErrorCode::InvalidMarketPrice, -5.0);
    IVError iv_err = convert_to_iv_error(err);
    EXPECT_EQ(iv_err.code, IVErrorCode::NegativeMarketPrice);
}

TEST(ErrorConversionTest, ValidationOutOfRangeToIVError) {
    ValidationError err(ValidationErrorCode::OutOfRange, 200.0);
    IVError iv_err = convert_to_iv_error(err);
    EXPECT_EQ(iv_err.code, IVErrorCode::ArbitrageViolation);
}

TEST(ErrorConversionTest, ValidationGridSizeToIVError) {
    ValidationError err(ValidationErrorCode::InvalidGridSize, 2.0);
    IVError iv_err = convert_to_iv_error(err);
    EXPECT_EQ(iv_err.code, IVErrorCode::InvalidGridConfig);
}

// ===========================================================================
// SolverError -> IVError conversion
// ===========================================================================

TEST(ErrorConversionTest, SolverConvergenceFailureToIVError) {
    SolverError err{.code = SolverErrorCode::ConvergenceFailure, .iterations = 42, .residual = 0.01};
    IVError iv_err = convert_to_iv_error(err);
    // All SolverErrors map to PDESolveFailed
    EXPECT_EQ(iv_err.code, IVErrorCode::PDESolveFailed);
    EXPECT_EQ(iv_err.iterations, 42);
    EXPECT_DOUBLE_EQ(iv_err.final_error, 0.01);
}

TEST(ErrorConversionTest, SolverLinearSolveFailureToIVError) {
    SolverError err{.code = SolverErrorCode::LinearSolveFailure, .iterations = 5, .residual = 1e10};
    IVError iv_err = convert_to_iv_error(err);
    EXPECT_EQ(iv_err.code, IVErrorCode::PDESolveFailed);
    EXPECT_EQ(iv_err.iterations, 5);
}

// ===========================================================================
// InterpolationError -> PriceTableError conversion
// ===========================================================================

TEST(ErrorConversionTest, InterpolationInsufficientPointsToPriceTableError) {
    InterpolationError err(InterpolationErrorCode::InsufficientGridPoints, 3);
    PriceTableError pt_err = convert_to_price_table_error(err);
    EXPECT_EQ(pt_err.code, PriceTableErrorCode::InsufficientGridPoints);
}

TEST(ErrorConversionTest, InterpolationFittingFailedToPriceTableError) {
    InterpolationError err(InterpolationErrorCode::FittingFailed, 10, 0, 1.5);
    PriceTableError pt_err = convert_to_price_table_error(err);
    EXPECT_EQ(pt_err.code, PriceTableErrorCode::FittingFailed);
}

TEST(ErrorConversionTest, InterpolationWorkspaceFailedToPriceTableError) {
    InterpolationError err(InterpolationErrorCode::WorkspaceCreationFailed, "test");
    PriceTableError pt_err = convert_to_price_table_error(err);
    EXPECT_EQ(pt_err.code, PriceTableErrorCode::ArenaAllocationFailed);
}

// ===========================================================================
// ValidationError -> PriceTableError conversion
// ===========================================================================

TEST(ErrorConversionTest, ValidationInvalidGridSizeToPriceTableError) {
    ValidationError err(ValidationErrorCode::InvalidGridSize, 2.0);
    PriceTableError pt_err = convert_to_price_table_error(err);
    EXPECT_EQ(pt_err.code, PriceTableErrorCode::InsufficientGridPoints);
}

TEST(ErrorConversionTest, ValidationInvalidStrikeToPriceTableError) {
    ValidationError err(ValidationErrorCode::InvalidStrike, -1.0);
    PriceTableError pt_err = convert_to_price_table_error(err);
    EXPECT_EQ(pt_err.code, PriceTableErrorCode::NonPositiveValue);
}

TEST(ErrorConversionTest, ValidationUnsortedGridToPriceTableError) {
    ValidationError err(ValidationErrorCode::UnsortedGrid);
    PriceTableError pt_err = convert_to_price_table_error(err);
    EXPECT_EQ(pt_err.code, PriceTableErrorCode::GridNotSorted);
}

// ===========================================================================
// map_expected_to_iv_error
// ===========================================================================

TEST(MapExpectedTest, SuccessValuePreservedForIVError) {
    std::expected<double, ValidationError> ok_result{42.0};
    auto mapped = map_expected_to_iv_error(ok_result);

    ASSERT_TRUE(mapped.has_value());
    EXPECT_DOUBLE_EQ(mapped.value(), 42.0);
}

TEST(MapExpectedTest, ErrorMappedForIVError) {
    std::expected<double, ValidationError> err_result{
        std::unexpected(ValidationError(ValidationErrorCode::InvalidSpotPrice, -1.0))
    };
    auto mapped = map_expected_to_iv_error(err_result);

    ASSERT_FALSE(mapped.has_value());
    EXPECT_EQ(mapped.error().code, IVErrorCode::NegativeSpot);
}

TEST(MapExpectedTest, SolverErrorMappedForIVError) {
    std::expected<double, SolverError> err_result{
        std::unexpected(SolverError{.code = SolverErrorCode::ConvergenceFailure, .iterations = 10})
    };
    auto mapped = map_expected_to_iv_error(err_result);

    ASSERT_FALSE(mapped.has_value());
    EXPECT_EQ(mapped.error().code, IVErrorCode::PDESolveFailed);
}

TEST(MapExpectedTest, SuccessValuePreservedForPriceTableError) {
    std::expected<double, InterpolationError> ok_result{3.14};
    auto mapped = map_expected_to_price_table_error(ok_result);

    ASSERT_TRUE(mapped.has_value());
    EXPECT_DOUBLE_EQ(mapped.value(), 3.14);
}

TEST(MapExpectedTest, ErrorMappedForPriceTableError) {
    std::expected<double, InterpolationError> err_result{
        std::unexpected(InterpolationError(InterpolationErrorCode::FittingFailed))
    };
    auto mapped = map_expected_to_price_table_error(err_result);

    ASSERT_FALSE(mapped.has_value());
    EXPECT_EQ(mapped.error().code, PriceTableErrorCode::FittingFailed);
}
