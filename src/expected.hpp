#pragma once

#include <string>
#include <cstddef>
#include "3rd/tl/expected.hpp"

namespace mango {

template<typename T, typename E>
using expected = tl::expected<T, E>;

template<typename E>
using unexpected = tl::unexpected<E>;

using tl::unexpected;

/// High-level solver error categories surfaced through expected results.
enum class SolverErrorCode {
    Stage1ConvergenceFailure,
    Stage2ConvergenceFailure,
    LinearSolveFailure,
    InvalidConfiguration,
    Unknown
};

/// Detailed solver error passed through expected failure path.
struct SolverError {
    SolverErrorCode code{SolverErrorCode::Unknown};
    std::string message;
    size_t iterations{0};
};

}  // namespace mango

