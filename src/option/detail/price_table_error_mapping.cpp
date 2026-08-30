// SPDX-License-Identifier: MIT
#include "mango/option/detail/price_table_error_mapping.hpp"

namespace mango::detail {

ValidationError to_validation_error(const PriceTableError& error) {
    switch (error.code) {
        case PriceTableErrorCode::NonPositiveValue:
            return ValidationError{ValidationErrorCode::InvalidBounds, 0.0,
                                   error.axis_index};
        case PriceTableErrorCode::InsufficientGridPoints:
        case PriceTableErrorCode::GridNotSorted:
            return ValidationError{ValidationErrorCode::InvalidGridSize,
                                   static_cast<double>(error.count),
                                   error.axis_index};
        case PriceTableErrorCode::InvalidConfig:
        case PriceTableErrorCode::EmptyBatch:
        case PriceTableErrorCode::ExtractionFailed:
        case PriceTableErrorCode::RepairFailed:
        case PriceTableErrorCode::FittingFailed:
        case PriceTableErrorCode::SurfaceBuildFailed:
        case PriceTableErrorCode::SerializationFailed:
        case PriceTableErrorCode::ArenaAllocationFailed:
        case PriceTableErrorCode::TensorCreationFailed:
            return ValidationError{ValidationErrorCode::PriceTableBuildFailed,
                                   static_cast<double>(error.count),
                                   error.axis_index};
        // An adaptive build that refuses (spec D5/D9) must reach the caller
        // as a refusal, not as a generic build failure: these are the forward
        // direction of the round trip pinned in error_types.hpp.
        case PriceTableErrorCode::NoViableSurface:
            return ValidationError{ValidationErrorCode::NoViableSurface,
                                   static_cast<double>(error.count),
                                   error.axis_index};
        case PriceTableErrorCode::ValidationFailed:
            return ValidationError{ValidationErrorCode::AdaptiveValidationFailed,
                                   static_cast<double>(error.count),
                                   error.axis_index};
    }
    // Unreachable: the switch is exhaustive (-Werror=switch enforces it).
    return ValidationError{ValidationErrorCode::PriceTableBuildFailed, 0.0, 0};
}

}  // namespace mango::detail
