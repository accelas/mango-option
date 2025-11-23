#pragma once

#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <expected>

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
    std::string message;
    size_t iterations{0};
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
    std::string parameter_name;
    std::string message;
    double value;  // The invalid value that was provided

    ValidationError(ValidationErrorCode code,
                   const std::string& parameter_name,
                   const std::string& message,
                   double value = 0.0)
        : code(code), parameter_name(parameter_name), message(message), value(value) {}
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
    std::string allocation_type;

    AllocationError(AllocationErrorCode code,
                   size_t requested_size,
                   const std::string& allocation_type)
        : code(code), requested_size(requested_size), allocation_type(allocation_type) {}
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
    std::string message;
    std::optional<std::string> details;

    InterpolationError(InterpolationErrorCode code,
                      const std::string& message,
                      const std::optional<std::string>& details = std::nullopt)
        : code(code), message(message), details(details) {}
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
    std::string message;
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

/// Convert error variant to string representation
inline std::string error_to_string(const ErrorVariant& error) {
    return std::visit([](const auto& e) -> std::string {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, ValidationError>) {
            return "Validation error: " + e.parameter_name + " - " + e.message;
        } else if constexpr (std::is_same_v<T, SolverError>) {
            return "Solver error: " + e.message;
        } else if constexpr (std::is_same_v<T, AllocationError>) {
            return "Allocation error: " + e.allocation_type + " failed for " +
                   std::to_string(e.requested_size) + " bytes";
        } else if constexpr (std::is_same_v<T, InterpolationError>) {
            std::string result = "Interpolation error: " + e.message;
            if (e.details.has_value()) {
                result += " (" + e.details.value() + ")";
            }
            return result;
        } else if constexpr (std::is_same_v<T, std::string>) {
            return e;
        }
    }, error);
}

} // namespace mango