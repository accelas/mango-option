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
    Stage1ConvergenceFailure,
    Stage2ConvergenceFailure,
    LinearSolveFailure,
    InvalidConfiguration,
    InvalidState,
    Unknown
};

/// Detailed solver error passed through expected failure path
struct SolverError {
    SolverErrorCode code{SolverErrorCode::Unknown};
    size_t iterations{0};
    double residual{0.0};  // Final residual at failure
};

/// Error codes for parameter validation failures
enum class ValidationErrorCode {
    InvalidStrike,
    InvalidSpotPrice,
    InvalidMaturity,
    InvalidVolatility,
    InvalidRate,
    InvalidDividend,
    InvalidGridSize,
    InvalidBounds,
    OutOfRange,
    InvalidGridSpacing,
    UnsortedGrid,
    ZeroWidthGrid
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
    FittingFailed,
    EvaluationFailed,
    ExtrapolationNotAllowed
};

/// Detailed interpolation error for fitting and evaluation failures
struct InterpolationError {
    InterpolationErrorCode code;
    size_t grid_size;      // Grid size involved
    double max_residual;   // Maximum residual for fitting errors

    InterpolationError(InterpolationErrorCode code,
                      size_t grid_size = 0,
                      double max_residual = 0.0)
        : code(code), grid_size(grid_size), max_residual(max_residual) {}
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

    // Convergence errors
    MaxIterationsExceeded,
    BracketingFailed,
    NumericalInstability,

    // Solver errors
    PDESolveFailed
};

/// Detailed IV solver error with diagnostics
struct IVError {
    IVErrorCode code;
    size_t iterations = 0;           ///< Iterations before failure
    double final_error = 0.0;        ///< Residual at failure
    std::optional<double> last_vol;  ///< Last volatility candidate tried
};

/// Combined error type that can hold any of our specific error types
using ErrorVariant = std::variant<
    ValidationError,
    SolverError,
    AllocationError,
    InterpolationError,
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

} // namespace mango