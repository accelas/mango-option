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
