// SPDX-License-Identifier: MIT
#include "mango/option/detail/price_table_error_mapping.hpp"
#include <gtest/gtest.h>
#include "mango/option/table/refinement_work.hpp"

using mango::PriceTableError;
using mango::PriceTableErrorCode;
using mango::ValidationErrorCode;

// Regression: distinct build failures must not surface as InvalidGridSize
// Bug: to_validation_error's default: arm mapped fitting, repair,
//      extraction, serialization, allocation, and config failures all to
//      InvalidGridSize — a grid-size lie that destroyed diagnostics.
TEST(PriceTableErrorMappingTest, BuildFailuresMapToPriceTableBuildFailed) {
    for (auto code : {PriceTableErrorCode::InvalidConfig,
                      PriceTableErrorCode::EmptyBatch,
                      PriceTableErrorCode::ExtractionFailed,
                      PriceTableErrorCode::RepairFailed,
                      PriceTableErrorCode::FittingFailed,
                      PriceTableErrorCode::SurfaceBuildFailed,
                      PriceTableErrorCode::SerializationFailed,
                      PriceTableErrorCode::ArenaAllocationFailed,
                      PriceTableErrorCode::TensorCreationFailed}) {
        auto ve = mango::detail::to_validation_error(
            PriceTableError{code, 0, 0});
        EXPECT_EQ(ve.code, ValidationErrorCode::PriceTableBuildFailed)
            << "code " << static_cast<int>(code);
    }
}

TEST(PriceTableErrorMappingTest, SpecificArmsUnchanged) {
    EXPECT_EQ(mango::detail::to_validation_error(
                  PriceTableError{PriceTableErrorCode::NonPositiveValue, 0, 0}).code,
              ValidationErrorCode::InvalidBounds);
    EXPECT_EQ(mango::detail::to_validation_error(
                  PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 0, 3}).code,
              ValidationErrorCode::InvalidGridSize);
    EXPECT_EQ(mango::detail::to_validation_error(
                  PriceTableError{PriceTableErrorCode::GridNotSorted, 0, 0}).code,
              ValidationErrorCode::InvalidGridSize);
}

TEST(PriceTableErrorMappingTest, WorkSurvivesPublicFailureMappingAndRoundTrip) {
    mango::RefinementWork work;
    work.references.requests = 7;
    work.references.failed_requests = 2;
    work.references.pde.reset();
    PriceTableError source{PriceTableErrorCode::NoViableSurface, 2, 4};
    source.work = std::make_shared<const mango::RefinementWork>(work);
    auto mapped = mango::detail::to_validation_error(source);
    ASSERT_TRUE(mapped.work);
    EXPECT_EQ(mapped.code, ValidationErrorCode::NoViableSurface);
    EXPECT_EQ(mapped.work->references.requests, 7u);
    EXPECT_FALSE(mapped.work->total_pde_attempts());
    auto restored = mango::convert_to_price_table_error(mapped);
    ASSERT_TRUE(restored.work);
    EXPECT_EQ(restored.work->references.failed_requests, 2u);
    EXPECT_FALSE(restored.work->total_pde_attempts());
}
