// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <expected>
#include <ostream>

namespace mango {

/// High-level solver error categories surfaced through expected results
enum class SolverErrorCode {
    ConvergenceFailure,
    LinearSolveFailure,
    InvalidConfiguration,
    Unknown
};

/// Detailed solver error passed through expected failure path
struct SolverError {
    SolverErrorCode code{SolverErrorCode::Unknown};
    size_t iterations{0};
    double residual{0.0};  // Code-dependent diagnostic: for ConvergenceFailure,
                           // the solution-norm-normalized RMS Newton step delta
                           // (not a PDE residual); infinity for
                           // LinearSolveFailure; other codes may carry other
                           // values (e.g. grid-validation data).
};

/// Error codes for parameter validation failures
enum class ValidationErrorCode {
    InvalidStrike,
    InvalidSpotPrice,
    InvalidMaturity,
    InvalidVolatility,
    InvalidRate,
    InvalidDividend,
    InvalidMarketPrice,
    InvalidGridSize,
    InvalidBounds,
    OutOfRange,
    InvalidGridSpacing,
    UnsortedGrid,
    ZeroWidthGrid,
    OptionTypeMismatch,
    DividendYieldMismatch,
    DiscreteDividendMismatch,
    PriceTableBuildFailed,     ///< Price table construction failed (non-grid cause)

    // Adaptive price-table build errors (surfaced from PriceTableErrorCode;
    // see PriceTableErrorCode::NoViableSurface / ValidationFailed).
    // Appended after PriceTableBuildFailed: merged ordinals stay stable
    // (Python exposes the numeric value; see ADR 0001).
    NoViableSurface,           ///< No adaptive-build candidate passed the D5 viability gate
    AdaptiveValidationFailed   ///< Adaptive build's holdout validation set could not measure error
};

/// Detailed validation error for parameter validation failures
struct ValidationError {
    ValidationErrorCode code;
    double value;  // The invalid value that was provided
    size_t index;  // Optional index for array/grid errors (0 if not applicable)

    ValidationError(ValidationErrorCode code,
                   double value = 0.0,
                   size_t index = 0)
        : code(code), value(value), index(index) {}
};

/// Error codes for allocation failures
enum class AllocationErrorCode {
    WorkspaceAllocationFailed,
    GridAllocationFailed,
    BufferAllocationFailed,
    MemoryExhausted,
    InvalidSize
};

/// Detailed allocation error for memory allocation failures
struct AllocationError {
    AllocationErrorCode code;
    size_t requested_size;

    AllocationError(AllocationErrorCode code,
                   size_t requested_size)
        : code(code), requested_size(requested_size) {}
};

/// Error codes for interpolation and fitting failures
enum class InterpolationErrorCode {
    InsufficientGridPoints,
    GridNotSorted,
    ZeroWidthGrid,
    ValueSizeMismatch,
    BufferSizeMismatch,
    DimensionMismatch,
    CoefficientSizeMismatch,
    NaNInput,
    InfInput,
    FittingFailed,
    EvaluationFailed,
    ExtrapolationNotAllowed,
    WorkspaceCreationFailed   // from_bytes() failed
};

/// Detailed interpolation error for fitting and evaluation failures
struct InterpolationError {
    InterpolationErrorCode code;
    size_t grid_size;      ///< Grid size involved
    size_t index;          ///< Index or axis where error occurred
    double max_residual;   ///< Maximum residual for fitting errors
    std::string message;   ///< Empty for most errors; used for workspace errors

    // Existing constructor (backward compatible)
    InterpolationError(InterpolationErrorCode code,
                      size_t grid_size = 0,
                      size_t index = 0,
                      double max_residual = 0.0)
        : code(code), grid_size(grid_size), index(index), max_residual(max_residual) {}

    // Constructor with message for workspace errors
    InterpolationError(InterpolationErrorCode code, std::string msg)
        : code(code), grid_size(0), index(0), max_residual(0.0), message(std::move(msg)) {}
};

/// IV solver error categories
enum class IVErrorCode {
    // Validation errors
    NegativeSpot,
    NegativeStrike,
    NegativeMaturity,
    NegativeMarketPrice,
    ArbitrageViolation,
    InvalidGridConfig,
    OptionTypeMismatch,
    DividendYieldMismatch,
    DiscreteDividendMismatch,

    // Convergence errors
    MaxIterationsExceeded,
    BracketingFailed,
    NumericalInstability,
    VegaTooSmall,           ///< Vega below threshold — IV undefined

    // Solver errors
    PDESolveFailed,

    // Query-time screening errors
    MultipleRoots           ///< IV inversion screen (D8) found more than one root
};

/// Detailed IV solver error with diagnostics
struct IVError {
    IVErrorCode code;
    size_t iterations = 0;           ///< Iterations before failure
    double final_error = 0.0;        ///< Residual at failure
    std::optional<double> last_vol;  ///< Last volatility candidate tried
};

/// Human-readable message for an IVErrorCode. Returns a stable string
/// literal (no allocation). Deliberately exhaustive (no default case) so
/// appending a new IVErrorCode value without a matching arm here is a
/// build failure (-Wswitch under -Werror) rather than a silent gap.
inline const char* iv_error_message(IVErrorCode code) {
    const char* msg;
    switch (code) {
        case IVErrorCode::NegativeSpot: msg = "Negative spot price"; break;
        case IVErrorCode::NegativeStrike: msg = "Negative strike price"; break;
        case IVErrorCode::NegativeMaturity: msg = "Negative maturity"; break;
        case IVErrorCode::NegativeMarketPrice: msg = "Negative market price"; break;
        case IVErrorCode::ArbitrageViolation: msg = "Arbitrage violation"; break;
        case IVErrorCode::InvalidGridConfig: msg = "Invalid grid configuration"; break;
        case IVErrorCode::OptionTypeMismatch: msg = "Option type mismatch"; break;
        case IVErrorCode::DividendYieldMismatch: msg = "Dividend yield mismatch"; break;
        case IVErrorCode::DiscreteDividendMismatch:
            msg = "Discrete dividend schedule mismatch";
            break;
        case IVErrorCode::MaxIterationsExceeded: msg = "Maximum iterations exceeded"; break;
        case IVErrorCode::BracketingFailed: msg = "Bracketing failed"; break;
        case IVErrorCode::NumericalInstability: msg = "Numerical instability"; break;
        case IVErrorCode::VegaTooSmall: msg = "Vega too small"; break;
        case IVErrorCode::PDESolveFailed: msg = "PDE solve failed"; break;
        case IVErrorCode::MultipleRoots:
            msg = "Multiple roots detected during IV inversion";
            break;
    }
    return msg;
}

/// Error codes for price table operations
enum class PriceTableErrorCode {
    // Validation errors
    InvalidConfig,             ///< Configuration validation failed
    InsufficientGridPoints,    ///< Grid axis has < 4 points
    GridNotSorted,             ///< Grid not sorted ascending
    NonPositiveValue,          ///< Value must be positive (moneyness, tau, sigma, K_ref)

    // Build errors
    EmptyBatch,                ///< make_batch returned empty
    ExtractionFailed,          ///< Tensor extraction failed
    RepairFailed,              ///< Failed to repair failed slices
    FittingFailed,             ///< B-spline fitting failed
    SurfaceBuildFailed,        ///< Surface construction failed

    // Serialization errors
    SerializationFailed,       ///< Arrow serialization failed
    ArenaAllocationFailed,     ///< Failed to allocate PMR arena
    TensorCreationFailed,      ///< Failed to create price tensor

    // Adaptive build errors (see design D5)
    ValidationFailed,          ///< Holdout validation set could not measure error
    NoViableSurface            ///< No candidate surface passed the viability gate
};

/// Detailed price table error with axis information
struct PriceTableError {
    PriceTableErrorCode code;
    size_t axis_index;         ///< Axis index for grid errors (0-3 for 4D)
    size_t count;              ///< Count for size-related errors

    PriceTableError(PriceTableErrorCode code,
                   size_t axis_index = 0,
                   size_t count = 0)
        : code(code), axis_index(axis_index), count(count) {}
};

/// Combined error type that can hold any of our specific error types
using ErrorVariant = std::variant<
    ValidationError,
    SolverError,
    AllocationError,
    InterpolationError,
    PriceTableError,
    std::string  // Generic error message
>;

/// Get error code as integer for diagnostics
inline int error_code(const ErrorVariant& error) {
    return std::visit([](const auto& e) -> int {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, ValidationError>) {
            return static_cast<int>(e.code);
        } else if constexpr (std::is_same_v<T, SolverError>) {
            return static_cast<int>(e.code);
        } else if constexpr (std::is_same_v<T, AllocationError>) {
            return static_cast<int>(e.code);
        } else if constexpr (std::is_same_v<T, InterpolationError>) {
            return static_cast<int>(e.code);
        } else if constexpr (std::is_same_v<T, PriceTableError>) {
            return static_cast<int>(e.code);
        } else {
            return -1;  // Generic string error
        }
    }, error);
}

/// Output stream operator for ValidationError
inline std::ostream& operator<<(std::ostream& os, const ValidationError& err) {
    os << "ValidationError{code=" << static_cast<int>(err.code)
       << ", value=" << err.value
       << ", index=" << err.index << "}";
    return os;
}

/// Output stream operator for SolverError
inline std::ostream& operator<<(std::ostream& os, const SolverError& err) {
    os << "SolverError{code=" << static_cast<int>(err.code)
       << ", iterations=" << err.iterations
       << ", residual=" << err.residual << "}";
    return os;
}

/// Output stream operator for InterpolationError
inline std::ostream& operator<<(std::ostream& os, const InterpolationError& err) {
    os << "InterpolationError{code=" << static_cast<int>(err.code)
       << ", grid_size=" << err.grid_size
       << ", index=" << err.index
       << ", max_residual=" << err.max_residual;
    if (!err.message.empty()) {
        os << ", message=\"" << err.message << "\"";
    }
    os << "}";
    return os;
}

/// Output stream operator for PriceTableError
inline std::ostream& operator<<(std::ostream& os, const PriceTableError& err) {
    os << "PriceTableError{code=" << static_cast<int>(err.code)
       << ", axis_index=" << err.axis_index
       << ", count=" << err.count << "}";
    return os;
}

// ============================================================================
// Error Conversion Functions
// ============================================================================
//
// Naming convention:
//   convert_to_<target>(<source>)  - Convert error struct to another type
//   map_expected_to_<target>()     - Transform std::expected error type
//
// Supported conversions:
//   ValidationError  → IVError, PriceTableError
//   SolverError      → IVError
//   InterpolationError → PriceTableError
// ============================================================================

/// Convert ValidationError to IVError
inline IVError convert_to_iv_error(const ValidationError& err) {
    IVErrorCode code;
    switch (err.code) {
        case ValidationErrorCode::InvalidStrike:
            code = IVErrorCode::NegativeStrike;
            break;
        case ValidationErrorCode::InvalidSpotPrice:
            code = IVErrorCode::NegativeSpot;
            break;
        case ValidationErrorCode::InvalidMaturity:
            code = IVErrorCode::NegativeMaturity;
            break;
        case ValidationErrorCode::InvalidMarketPrice:
            code = IVErrorCode::NegativeMarketPrice;
            break;
        case ValidationErrorCode::InvalidVolatility:
        case ValidationErrorCode::InvalidRate:
        case ValidationErrorCode::InvalidDividend:
        case ValidationErrorCode::OutOfRange:
            code = IVErrorCode::ArbitrageViolation;
            break;
        case ValidationErrorCode::InvalidGridSize:
        case ValidationErrorCode::InvalidBounds:
        case ValidationErrorCode::InvalidGridSpacing:
        case ValidationErrorCode::UnsortedGrid:
        case ValidationErrorCode::ZeroWidthGrid:
            code = IVErrorCode::InvalidGridConfig;
            break;
        case ValidationErrorCode::OptionTypeMismatch:
            code = IVErrorCode::OptionTypeMismatch;
            break;
        case ValidationErrorCode::DividendYieldMismatch:
            code = IVErrorCode::DividendYieldMismatch;
            break;
        case ValidationErrorCode::DiscreteDividendMismatch:
            code = IVErrorCode::DiscreteDividendMismatch;
            break;
        case ValidationErrorCode::PriceTableBuildFailed:
        case ValidationErrorCode::NoViableSurface:
        case ValidationErrorCode::AdaptiveValidationFailed:
            // Adaptive-build failures are grid/config-shaped problems, not a
            // bad market input — same bucket as the other grid-config codes.
            code = IVErrorCode::InvalidGridConfig;
            break;
    }
    return IVError{
        .code = code,
        .iterations = 0,
        .final_error = err.value,
        .last_vol = std::nullopt
    };
}

/// Convert SolverError to IVError
inline IVError convert_to_iv_error(const SolverError& err) {
    return IVError{
        .code = IVErrorCode::PDESolveFailed,
        .iterations = err.iterations,
        .final_error = err.residual,
        .last_vol = std::nullopt
    };
}

/// Convert InterpolationError to PriceTableError
inline PriceTableError convert_to_price_table_error(const InterpolationError& err) {
    // Initialized (not left uninitialized like the switches below) because
    // deep inlining at -O3 from price_table_factory.cpp's new call sites
    // defeats GCC's exhaustiveness analysis and trips -Wmaybe-uninitialized;
    // the switch is still exhaustive and -Werror=switch still enforces it.
    PriceTableErrorCode code = PriceTableErrorCode::FittingFailed;
    switch (err.code) {
        case InterpolationErrorCode::InsufficientGridPoints:
            code = PriceTableErrorCode::InsufficientGridPoints;
            break;
        case InterpolationErrorCode::GridNotSorted:
        case InterpolationErrorCode::ZeroWidthGrid:
            code = PriceTableErrorCode::GridNotSorted;
            break;
        case InterpolationErrorCode::ValueSizeMismatch:
        case InterpolationErrorCode::BufferSizeMismatch:
        case InterpolationErrorCode::DimensionMismatch:
        case InterpolationErrorCode::CoefficientSizeMismatch:
        case InterpolationErrorCode::NaNInput:
        case InterpolationErrorCode::InfInput:
        case InterpolationErrorCode::FittingFailed:
        case InterpolationErrorCode::EvaluationFailed:
        case InterpolationErrorCode::ExtrapolationNotAllowed:
            code = PriceTableErrorCode::FittingFailed;
            break;
        case InterpolationErrorCode::WorkspaceCreationFailed:
            code = PriceTableErrorCode::ArenaAllocationFailed;
            break;
    }
    return PriceTableError{code, err.index, err.grid_size};
}

/// Convert ValidationError to PriceTableError
inline PriceTableError convert_to_price_table_error(const ValidationError& err) {
    PriceTableErrorCode code;
    switch (err.code) {
        case ValidationErrorCode::InvalidStrike:
        case ValidationErrorCode::InvalidSpotPrice:
        case ValidationErrorCode::InvalidMaturity:
        case ValidationErrorCode::InvalidVolatility:
        case ValidationErrorCode::InvalidMarketPrice:
        case ValidationErrorCode::InvalidRate:
        case ValidationErrorCode::InvalidDividend:
            code = PriceTableErrorCode::NonPositiveValue;
            break;
        case ValidationErrorCode::InvalidGridSize:
            code = PriceTableErrorCode::InsufficientGridPoints;
            break;
        case ValidationErrorCode::InvalidBounds:
        case ValidationErrorCode::OutOfRange:
        case ValidationErrorCode::InvalidGridSpacing:
            code = PriceTableErrorCode::InvalidConfig;
            break;
        case ValidationErrorCode::UnsortedGrid:
        case ValidationErrorCode::ZeroWidthGrid:
            code = PriceTableErrorCode::GridNotSorted;
            break;
        case ValidationErrorCode::OptionTypeMismatch:
        case ValidationErrorCode::DividendYieldMismatch:
        case ValidationErrorCode::DiscreteDividendMismatch:
            code = PriceTableErrorCode::InvalidConfig;
            break;
        case ValidationErrorCode::PriceTableBuildFailed:
            code = PriceTableErrorCode::SurfaceBuildFailed;
            break;
        case ValidationErrorCode::NoViableSurface:
            // Mirrors the forward mapping in
            // detail/price_table_error_mapping.cpp's to_validation_error():
            // PriceTableErrorCode::NoViableSurface ->
            // ValidationErrorCode::NoViableSurface.
            code = PriceTableErrorCode::NoViableSurface;
            break;
        case ValidationErrorCode::AdaptiveValidationFailed:
            // Mirrors PriceTableErrorCode::ValidationFailed ->
            // ValidationErrorCode::AdaptiveValidationFailed.
            code = PriceTableErrorCode::ValidationFailed;
            break;
    }
    return PriceTableError{code, err.index, 0};
}

// ============================================================================
// std::expected Error Mapping Functions
// ============================================================================

/// Map std::expected<T, ValidationError> to std::expected<T, IVError>
template<typename T>
std::expected<T, IVError> map_expected_to_iv_error(
    const std::expected<T, ValidationError>& result)
{
    if (result.has_value()) {
        return result.value();
    }
    return std::unexpected(convert_to_iv_error(result.error()));
}

template<typename T>
std::expected<T, IVError> map_expected_to_iv_error(
    std::expected<T, ValidationError>&& result)
{
    if (result.has_value()) {
        return std::move(result.value());
    }
    return std::unexpected(convert_to_iv_error(result.error()));
}

/// Map std::expected<T, SolverError> to std::expected<T, IVError>
template<typename T>
std::expected<T, IVError> map_expected_to_iv_error(
    const std::expected<T, SolverError>& result)
{
    if (result.has_value()) {
        return result.value();
    }
    return std::unexpected(convert_to_iv_error(result.error()));
}

template<typename T>
std::expected<T, IVError> map_expected_to_iv_error(
    std::expected<T, SolverError>&& result)
{
    if (result.has_value()) {
        return std::move(result.value());
    }
    return std::unexpected(convert_to_iv_error(result.error()));
}

/// Map std::expected<T, InterpolationError> to std::expected<T, PriceTableError>
template<typename T>
std::expected<T, PriceTableError> map_expected_to_price_table_error(
    const std::expected<T, InterpolationError>& result)
{
    if (result.has_value()) {
        return result.value();
    }
    return std::unexpected(convert_to_price_table_error(result.error()));
}

template<typename T>
std::expected<T, PriceTableError> map_expected_to_price_table_error(
    std::expected<T, InterpolationError>&& result)
{
    if (result.has_value()) {
        return std::move(result.value());
    }
    return std::unexpected(convert_to_price_table_error(result.error()));
}

/// Map std::expected<T, ValidationError> to std::expected<T, PriceTableError>
template<typename T>
std::expected<T, PriceTableError> map_expected_to_price_table_error(
    const std::expected<T, ValidationError>& result)
{
    if (result.has_value()) {
        return result.value();
    }
    return std::unexpected(convert_to_price_table_error(result.error()));
}

template<typename T>
std::expected<T, PriceTableError> map_expected_to_price_table_error(
    std::expected<T, ValidationError>&& result)
{
    if (result.has_value()) {
        return std::move(result.value());
    }
    return std::unexpected(convert_to_price_table_error(result.error()));
}

} // namespace mango