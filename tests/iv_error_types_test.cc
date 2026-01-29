#include <gtest/gtest.h>
#include "src/support/error_types.hpp"

using namespace mango;

TEST(IVErrorTest, ErrorCodeConstruction) {
    IVError error{
        .code = IVErrorCode::NegativeSpot,
        .iterations = 0,
        .final_error = 0.0
    };

    EXPECT_EQ(error.code, IVErrorCode::NegativeSpot);
    EXPECT_EQ(error.iterations, 0);
    EXPECT_EQ(error.final_error, 0.0);
    EXPECT_FALSE(error.last_vol.has_value());
}

TEST(IVErrorTest, ErrorWithLastVol) {
    IVError error{
        .code = IVErrorCode::MaxIterationsExceeded,
        .iterations = 100,
        .final_error = 1e-5,
        .last_vol = 0.25
    };

    EXPECT_EQ(error.code, IVErrorCode::MaxIterationsExceeded);
    EXPECT_EQ(error.iterations, 100);
    EXPECT_DOUBLE_EQ(error.final_error, 1e-5);
    ASSERT_TRUE(error.last_vol.has_value());
    EXPECT_DOUBLE_EQ(*error.last_vol, 0.25);
}

TEST(IVErrorTest, ValidationErrorCodes) {
    // Test all validation error codes exist
    IVError e1{.code = IVErrorCode::NegativeSpot};
    IVError e2{.code = IVErrorCode::NegativeStrike};
    IVError e3{.code = IVErrorCode::NegativeMaturity};
    IVError e4{.code = IVErrorCode::NegativeMarketPrice};
    IVError e5{.code = IVErrorCode::ArbitrageViolation};

    EXPECT_EQ(e1.code, IVErrorCode::NegativeSpot);
    EXPECT_EQ(e2.code, IVErrorCode::NegativeStrike);
    EXPECT_EQ(e3.code, IVErrorCode::NegativeMaturity);
    EXPECT_EQ(e4.code, IVErrorCode::NegativeMarketPrice);
    EXPECT_EQ(e5.code, IVErrorCode::ArbitrageViolation);
}

TEST(IVErrorTest, ConvergenceErrorCodes) {
    // Test all convergence error codes exist
    IVError e1{.code = IVErrorCode::MaxIterationsExceeded};
    IVError e2{.code = IVErrorCode::BracketingFailed};
    IVError e3{.code = IVErrorCode::NumericalInstability};

    EXPECT_EQ(e1.code, IVErrorCode::MaxIterationsExceeded);
    EXPECT_EQ(e2.code, IVErrorCode::BracketingFailed);
    EXPECT_EQ(e3.code, IVErrorCode::NumericalInstability);
}

TEST(IVErrorTest, SolverErrorCodes) {
    // Test solver error codes exist
    IVError e1{.code = IVErrorCode::PDESolveFailed};

    EXPECT_EQ(e1.code, IVErrorCode::PDESolveFailed);
}

// ============================================================================
// Error Conversion Tests
// ============================================================================

TEST(ErrorConversionTest, ValidationErrorToIVError) {
    // Test mapping of validation codes to IV codes
    ValidationError strike_err(ValidationErrorCode::InvalidStrike, -100.0);
    auto iv1 = convert_to_iv_error(strike_err);
    EXPECT_EQ(iv1.code, IVErrorCode::NegativeStrike);
    EXPECT_DOUBLE_EQ(iv1.final_error, -100.0);

    ValidationError spot_err(ValidationErrorCode::InvalidSpotPrice, 0.0);
    auto iv2 = convert_to_iv_error(spot_err);
    EXPECT_EQ(iv2.code, IVErrorCode::NegativeSpot);

    ValidationError maturity_err(ValidationErrorCode::InvalidMaturity, -0.5);
    auto iv3 = convert_to_iv_error(maturity_err);
    EXPECT_EQ(iv3.code, IVErrorCode::NegativeMaturity);

    ValidationError price_err(ValidationErrorCode::InvalidMarketPrice, -10.0);
    auto iv4 = convert_to_iv_error(price_err);
    EXPECT_EQ(iv4.code, IVErrorCode::NegativeMarketPrice);

    ValidationError vol_err(ValidationErrorCode::InvalidVolatility, -0.2);
    auto iv5 = convert_to_iv_error(vol_err);
    EXPECT_EQ(iv5.code, IVErrorCode::ArbitrageViolation);

    ValidationError grid_err(ValidationErrorCode::InvalidGridSize, 2);
    auto iv6 = convert_to_iv_error(grid_err);
    EXPECT_EQ(iv6.code, IVErrorCode::InvalidGridConfig);
}

TEST(ErrorConversionTest, SolverErrorToIVError) {
    SolverError solver_err{
        .code = SolverErrorCode::ConvergenceFailure,
        .iterations = 50,
        .residual = 1e-3
    };

    auto iv = convert_to_iv_error(solver_err);
    EXPECT_EQ(iv.code, IVErrorCode::PDESolveFailed);
    EXPECT_EQ(iv.iterations, 50);
    EXPECT_DOUBLE_EQ(iv.final_error, 1e-3);
}

TEST(ErrorConversionTest, InterpolationErrorToPriceTableError) {
    InterpolationError grid_err(InterpolationErrorCode::InsufficientGridPoints, 3, 0);
    auto pt1 = convert_to_price_table_error(grid_err);
    EXPECT_EQ(pt1.code, PriceTableErrorCode::InsufficientGridPoints);
    EXPECT_EQ(pt1.count, 3);

    InterpolationError sort_err(InterpolationErrorCode::GridNotSorted, 10, 2);
    auto pt2 = convert_to_price_table_error(sort_err);
    EXPECT_EQ(pt2.code, PriceTableErrorCode::GridNotSorted);
    EXPECT_EQ(pt2.axis_index, 2);

    InterpolationError fit_err(InterpolationErrorCode::FittingFailed, 100, 1, 0.05);
    auto pt3 = convert_to_price_table_error(fit_err);
    EXPECT_EQ(pt3.code, PriceTableErrorCode::FittingFailed);
}

TEST(ErrorConversionTest, ValidationErrorToPriceTableError) {
    ValidationError strike_err(ValidationErrorCode::InvalidStrike, -100.0);
    auto pt1 = convert_to_price_table_error(strike_err);
    EXPECT_EQ(pt1.code, PriceTableErrorCode::NonPositiveValue);

    ValidationError grid_err(ValidationErrorCode::InvalidGridSize, 2, 1);
    auto pt2 = convert_to_price_table_error(grid_err);
    EXPECT_EQ(pt2.code, PriceTableErrorCode::InsufficientGridPoints);

    ValidationError sort_err(ValidationErrorCode::UnsortedGrid, 0, 3);
    auto pt3 = convert_to_price_table_error(sort_err);
    EXPECT_EQ(pt3.code, PriceTableErrorCode::GridNotSorted);
}

TEST(ErrorConversionTest, MapExpectedValidationToIV) {
    // Test with success case
    std::expected<double, ValidationError> success_result = 42.0;
    auto iv_success = map_expected_to_iv_error(success_result);
    ASSERT_TRUE(iv_success.has_value());
    EXPECT_DOUBLE_EQ(iv_success.value(), 42.0);

    // Test with failure case
    std::expected<double, ValidationError> fail_result =
        std::unexpected(ValidationError(ValidationErrorCode::InvalidStrike, -100.0));
    auto iv_fail = map_expected_to_iv_error(fail_result);
    ASSERT_FALSE(iv_fail.has_value());
    EXPECT_EQ(iv_fail.error().code, IVErrorCode::NegativeStrike);
}

TEST(ErrorConversionTest, MapExpectedSolverToIV) {
    // Test with success case
    std::expected<int, SolverError> success_result = 123;
    auto iv_success = map_expected_to_iv_error(success_result);
    ASSERT_TRUE(iv_success.has_value());
    EXPECT_EQ(iv_success.value(), 123);

    // Test with failure case
    std::expected<int, SolverError> fail_result = std::unexpected(SolverError{
        .code = SolverErrorCode::LinearSolveFailure,
        .iterations = 10,
        .residual = 0.1
    });
    auto iv_fail = map_expected_to_iv_error(fail_result);
    ASSERT_FALSE(iv_fail.has_value());
    EXPECT_EQ(iv_fail.error().code, IVErrorCode::PDESolveFailed);
    EXPECT_EQ(iv_fail.error().iterations, 10);
}

TEST(ErrorConversionTest, MapExpectedInterpolationToPriceTable) {
    // Test with success case
    std::expected<std::string, InterpolationError> success_result = "success";
    auto pt_success = map_expected_to_price_table_error(success_result);
    ASSERT_TRUE(pt_success.has_value());
    EXPECT_EQ(pt_success.value(), "success");

    // Test with failure case
    std::expected<std::string, InterpolationError> fail_result =
        std::unexpected(InterpolationError(InterpolationErrorCode::FittingFailed, 50, 2, 0.01));
    auto pt_fail = map_expected_to_price_table_error(fail_result);
    ASSERT_FALSE(pt_fail.has_value());
    EXPECT_EQ(pt_fail.error().code, PriceTableErrorCode::FittingFailed);
}
