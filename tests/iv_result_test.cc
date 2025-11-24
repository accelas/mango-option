#include <gtest/gtest.h>
#include "src/option/iv_result.hpp"
#include "src/support/error_types.hpp"
#include <expected>
#include <vector>

using namespace mango;

TEST(IVSuccessTest, BasicConstruction) {
    IVSuccess success{
        .implied_vol = 0.25,
        .iterations = 12,
        .final_error = 1e-8
    };

    EXPECT_DOUBLE_EQ(success.implied_vol, 0.25);
    EXPECT_EQ(success.iterations, 12);
    EXPECT_DOUBLE_EQ(success.final_error, 1e-8);
    EXPECT_FALSE(success.vega.has_value());
}

TEST(IVSuccessTest, ConstructionWithVega) {
    IVSuccess success{
        .implied_vol = 0.30,
        .iterations = 8,
        .final_error = 5e-9,
        .vega = 45.3
    };

    EXPECT_DOUBLE_EQ(success.implied_vol, 0.30);
    EXPECT_EQ(success.iterations, 8);
    EXPECT_DOUBLE_EQ(success.final_error, 5e-9);
    ASSERT_TRUE(success.vega.has_value());
    EXPECT_DOUBLE_EQ(*success.vega, 45.3);
}

TEST(IVSuccessTest, ZeroIterations) {
    // Edge case: convergence on first iteration
    IVSuccess success{
        .implied_vol = 0.20,
        .iterations = 0,
        .final_error = 0.0
    };

    EXPECT_DOUBLE_EQ(success.implied_vol, 0.20);
    EXPECT_EQ(success.iterations, 0);
    EXPECT_DOUBLE_EQ(success.final_error, 0.0);
}

TEST(IVSuccessTest, HighIterations) {
    // Edge case: near max iterations
    IVSuccess success{
        .implied_vol = 0.15,
        .iterations = 99,
        .final_error = 1e-6
    };

    EXPECT_DOUBLE_EQ(success.implied_vol, 0.15);
    EXPECT_EQ(success.iterations, 99);
    EXPECT_DOUBLE_EQ(success.final_error, 1e-6);
}

TEST(IVSuccessTest, VolatilityRange) {
    // Test various realistic volatility values
    IVSuccess low_vol{.implied_vol = 0.05, .iterations = 10, .final_error = 1e-7};
    IVSuccess med_vol{.implied_vol = 0.30, .iterations = 12, .final_error = 1e-8};
    IVSuccess high_vol{.implied_vol = 2.00, .iterations = 15, .final_error = 1e-7};

    EXPECT_DOUBLE_EQ(low_vol.implied_vol, 0.05);
    EXPECT_DOUBLE_EQ(med_vol.implied_vol, 0.30);
    EXPECT_DOUBLE_EQ(high_vol.implied_vol, 2.00);
}

// ============================================================================
// BatchIVResult Tests
// ============================================================================

TEST(BatchIVResultTest, AllSucceeded) {
    std::vector<std::expected<IVSuccess, IVError>> results;
    results.push_back(IVSuccess{.implied_vol = 0.20, .iterations = 10, .final_error = 1e-8});
    results.push_back(IVSuccess{.implied_vol = 0.25, .iterations = 12, .final_error = 1e-9});
    results.push_back(IVSuccess{.implied_vol = 0.18, .iterations = 8, .final_error = 1e-7});

    mango::BatchIVResult batch{
        .results = std::move(results),
        .failed_count = 0
    };

    EXPECT_TRUE(batch.all_succeeded());
    EXPECT_EQ(batch.failed_count, 0);
    EXPECT_EQ(batch.results.size(), 3);
}

TEST(BatchIVResultTest, SomeFailures) {
    std::vector<std::expected<IVSuccess, IVError>> results;
    results.push_back(IVSuccess{.implied_vol = 0.20, .iterations = 10, .final_error = 1e-8});
    results.push_back(std::unexpected(IVError{
        .code = IVErrorCode::MaxIterationsExceeded,
        .iterations = 100,
        .final_error = 1e-2
    }));
    results.push_back(IVSuccess{.implied_vol = 0.18, .iterations = 8, .final_error = 1e-7});

    mango::BatchIVResult batch{
        .results = std::move(results),
        .failed_count = 1
    };

    EXPECT_FALSE(batch.all_succeeded());
    EXPECT_EQ(batch.failed_count, 1);
    EXPECT_EQ(batch.results.size(), 3);

    // Verify specific results
    ASSERT_TRUE(batch.results[0].has_value());
    EXPECT_DOUBLE_EQ(batch.results[0]->implied_vol, 0.20);

    ASSERT_FALSE(batch.results[1].has_value());
    EXPECT_EQ(batch.results[1].error().code, IVErrorCode::MaxIterationsExceeded);

    ASSERT_TRUE(batch.results[2].has_value());
    EXPECT_DOUBLE_EQ(batch.results[2]->implied_vol, 0.18);
}

TEST(BatchIVResultTest, AllFailed) {
    std::vector<std::expected<IVSuccess, IVError>> results;
    results.push_back(std::unexpected(IVError{
        .code = IVErrorCode::NegativeSpot
    }));
    results.push_back(std::unexpected(IVError{
        .code = IVErrorCode::ArbitrageViolation
    }));

    mango::BatchIVResult batch{
        .results = std::move(results),
        .failed_count = 2
    };

    EXPECT_FALSE(batch.all_succeeded());
    EXPECT_EQ(batch.failed_count, 2);
    EXPECT_EQ(batch.results.size(), 2);
}

TEST(BatchIVResultTest, EmptyBatch) {
    mango::BatchIVResult batch{
        .results = {},
        .failed_count = 0
    };

    EXPECT_TRUE(batch.all_succeeded());
    EXPECT_EQ(batch.failed_count, 0);
    EXPECT_EQ(batch.results.size(), 0);
}
