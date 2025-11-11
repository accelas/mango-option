#pragma once

#include "common/ivcalc_trace.h"
#include <cstddef>
#include <optional>
#include <string>
#include <concepts>
#include <cmath>
#include <limits>
#include <algorithm>

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

    /// Optional root value (for scalar root-finding methods like Brent)
    /// Not used by Newton-Raphson (which operates on solution vectors)
    std::optional<double> root;
};

/// Concept for Brent objective functions
///
/// Brent's method works with any callable that takes a double and returns a double.
/// This includes lambdas, function objects, function pointers, and std::function.
template<typename F>
concept BrentObjective = requires(F f, double x) {
    { f(x) } -> std::convertible_to<double>;
};

/// Find root using Brent's method
///
/// Combines bisection, secant, and inverse quadratic interpolation for robust
/// scalar root-finding without derivatives.
///
/// **Properties:**
/// - Guaranteed convergence if root exists in [a, b]
/// - Superlinear convergence rate
/// - Does not require derivatives
/// - More robust than Newton for difficult scalar problems
///
/// **Use cases:**
/// - Implied volatility calculation
/// - American option critical price
/// - Any scalar equation f(x) = 0 with bracketed root
///
/// **Precondition:** f(a) and f(b) must have opposite signs
///
/// @tparam F Function type satisfying BrentObjective concept
/// @param f Function to find root of
/// @param a Left bracket
/// @param b Right bracket
/// @param config Root-finding configuration
/// @return Result with root (if converged) and convergence status
///
/// Reference: Brent, R. (1973). "Algorithms for Minimization without Derivatives"
template<BrentObjective F>
RootFindingResult brent_find_root(F&& f, double a, double b,
                                  const RootFindingConfig& config) {
    // Evaluate endpoints
    double fa = f(a);
    double fb = f(b);

    // Emit start trace
    MANGO_TRACE_BRENT_START(a, b, config.brent_tol_abs, config.max_iter);

    // Check for NaN/Inf at endpoints (indicates invalid input or function failure)
    if (!std::isfinite(fa) || !std::isfinite(fb)) {
        return RootFindingResult{
            .converged = false,
            .iterations = 0,
            .final_error = std::numeric_limits<double>::quiet_NaN(),
            .failure_reason = "Function returned non-finite value (NaN or Inf)",
            .root = std::nullopt
        };
    }

    // Check if root is bracketed
    if (fa * fb > 0.0) {
        return RootFindingResult{
            .converged = false,
            .iterations = 0,
            .final_error = std::min(std::abs(fa), std::abs(fb)),
            .failure_reason = "Root not bracketed",
            .root = std::nullopt
        };
    }

    // Check if endpoints are roots
    if (std::abs(fa) < config.brent_tol_abs) {
        MANGO_TRACE_BRENT_COMPLETE(a, 0, 1);
        return RootFindingResult{
            .converged = true,
            .iterations = 0,
            .final_error = std::abs(fa),
            .failure_reason = std::nullopt,
            .root = a
        };
    }

    if (std::abs(fb) < config.brent_tol_abs) {
        MANGO_TRACE_BRENT_COMPLETE(b, 0, 1);
        return RootFindingResult{
            .converged = true,
            .iterations = 0,
            .final_error = std::abs(fb),
            .failure_reason = std::nullopt,
            .root = b
        };
    }

    // Ensure |f(b)| < |f(a)|
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;
    double fc = fa;
    bool mflag = true;
    double d = 0.0;

    for (size_t iter = 0; iter < config.max_iter; ++iter) {
        // Emit iteration trace
        [[maybe_unused]] double interval_width = std::abs(b - a);
        MANGO_TRACE_BRENT_ITER(iter, b, fb, interval_width);

        // Check convergence criteria
        if (std::abs(fb) < config.brent_tol_abs ||
            std::abs(b - a) < config.brent_tol_abs) {
            MANGO_TRACE_BRENT_COMPLETE(b, iter + 1, 1);
            return RootFindingResult{
                .converged = true,
                .iterations = iter + 1,
                .final_error = std::abs(fb),
                .failure_reason = std::nullopt,
                .root = b
            };
        }

        double s;

        // Decide between interpolation and bisection
        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Check conditions for using interpolation vs bisection
        double bisect = (3.0 * a + b) / 4.0;

        // Condition 1: s is not between (3a+b)/4 and b
        bool condition1 = !((s > bisect && s < b) || (s < bisect && s > b));

        // Condition 2: mflag is set and |s-b| >= |b-c|/2
        bool condition2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;

        // Condition 3: mflag is not set and |s-b| >= |c-d|/2
        bool condition3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2.0;

        // Condition 4: mflag is set and |b-c| < tolerance
        bool condition4 = mflag && std::abs(b - c) < config.brent_tol_abs;

        // Condition 5: mflag is not set and |c-d| < tolerance
        bool condition5 = !mflag && std::abs(c - d) < config.brent_tol_abs;

        if (condition1 || condition2 || condition3 || condition4 || condition5) {
            // Use bisection
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        // Evaluate function at s
        double fs = f(s);

        // Check for NaN (indicates PDE solver failure or invalid input)
        if (!std::isfinite(fs)) {
            MANGO_TRACE_BRENT_COMPLETE(s, iter + 1, 0);
            return RootFindingResult{
                .converged = false,
                .iterations = iter + 1,
                .final_error = std::numeric_limits<double>::quiet_NaN(),
                .failure_reason = "Function returned non-finite value (NaN or Inf)",
                .root = std::nullopt
            };
        }

        // Update for next iteration
        d = c;
        c = b;
        fc = fb;

        // Update bracket
        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        // Ensure |f(b)| < |f(a)|
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }

    // Max iterations reached
    MANGO_TRACE_BRENT_COMPLETE(b, config.max_iter, 0);
    return RootFindingResult{
        .converged = false,
        .iterations = config.max_iter,
        .final_error = std::abs(fb),
        .failure_reason = "Max iterations reached",
        .root = b
    };
}

}  // namespace mango
