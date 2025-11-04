#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace mango {

/// Configuration for all root-finding methods
///
/// Unified configuration allowing different methods to coexist.
/// Each method uses only its relevant parameters.
struct RootFindingConfig {
    /// Maximum iterations for any method
    size_t max_iter = 100;

    /// Relative convergence tolerance
    double tolerance = 1e-6;

    // Newton-specific parameters
    double jacobian_fd_epsilon = 1e-7;  ///< Finite difference step for Jacobian

    // Brent-specific parameters
    double brent_tol_abs = 1e-6;  ///< Absolute tolerance for Brent's method

    // Future methods can add parameters here
};

/// Result from any root-finding method
///
/// Provides consistent interface for convergence status,
/// iteration count, and diagnostic information.
struct RootFindingResult {
    /// Convergence status
    bool converged;

    /// Number of iterations performed
    size_t iterations;

    /// Final error measure (method-dependent)
    double final_error;

    /// Optional failure diagnostic message
    std::optional<std::string> failure_reason;
};

}  // namespace mango
