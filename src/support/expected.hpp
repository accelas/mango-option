#pragma once

#include <string>
#include <cstddef>
#include <variant>
#include <optional>
#include <expected>

namespace mango {

template<typename T, typename E>
using expected = std::expected<T, E>;

template<typename E>
using unexpected = std::unexpected<E>;

using std::unexpected;

/// High-level solver error categories surfaced through expected results.
enum class SolverErrorCode {
    Stage1ConvergenceFailure,
    Stage2ConvergenceFailure,
    LinearSolveFailure,
    InvalidConfiguration,
    InvalidState,  // Added for state validation errors
    Unknown
};

/// Detailed solver error passed through expected failure path.
struct SolverError {
    SolverErrorCode code{SolverErrorCode::Unknown};
    std::string message;
    size_t iterations{0};
};

}  // namespace mango

