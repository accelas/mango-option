#include <gtest/gtest.h>
#include "src/support/error_types.hpp"

using namespace mango;

TEST(IVErrorTest, ErrorCodeConstruction) {
    IVError error{
        .code = IVErrorCode::NegativeSpot,
        .message = "Spot price must be positive",
        .iterations = 0,
        .final_error = 0.0
    };

    EXPECT_EQ(error.code, IVErrorCode::NegativeSpot);
    EXPECT_EQ(error.message, "Spot price must be positive");
    EXPECT_EQ(error.iterations, 0);
    EXPECT_EQ(error.final_error, 0.0);
    EXPECT_FALSE(error.last_vol.has_value());
}

TEST(IVErrorTest, ErrorWithLastVol) {
    IVError error{
        .code = IVErrorCode::MaxIterationsExceeded,
        .message = "Reached max iterations",
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
    IVError e1{.code = IVErrorCode::NegativeSpot, .message = ""};
    IVError e2{.code = IVErrorCode::NegativeStrike, .message = ""};
    IVError e3{.code = IVErrorCode::NegativeMaturity, .message = ""};
    IVError e4{.code = IVErrorCode::NegativeMarketPrice, .message = ""};
    IVError e5{.code = IVErrorCode::ArbitrageViolation, .message = ""};

    EXPECT_EQ(e1.code, IVErrorCode::NegativeSpot);
    EXPECT_EQ(e2.code, IVErrorCode::NegativeStrike);
    EXPECT_EQ(e3.code, IVErrorCode::NegativeMaturity);
    EXPECT_EQ(e4.code, IVErrorCode::NegativeMarketPrice);
    EXPECT_EQ(e5.code, IVErrorCode::ArbitrageViolation);
}

TEST(IVErrorTest, ConvergenceErrorCodes) {
    // Test all convergence error codes exist
    IVError e1{.code = IVErrorCode::MaxIterationsExceeded, .message = ""};
    IVError e2{.code = IVErrorCode::BracketingFailed, .message = ""};
    IVError e3{.code = IVErrorCode::NumericalInstability, .message = ""};

    EXPECT_EQ(e1.code, IVErrorCode::MaxIterationsExceeded);
    EXPECT_EQ(e2.code, IVErrorCode::BracketingFailed);
    EXPECT_EQ(e3.code, IVErrorCode::NumericalInstability);
}

TEST(IVErrorTest, SolverErrorCodes) {
    // Test solver error codes exist
    IVError e1{.code = IVErrorCode::PDESolveFailed, .message = ""};

    EXPECT_EQ(e1.code, IVErrorCode::PDESolveFailed);
}
