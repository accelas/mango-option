#pragma once

#include <Kokkos_Core.hpp>
#include <expected>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <optional>
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

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

/// Success result from root-finding methods
///
/// Returned when the root-finding algorithm converges successfully.
struct RootFindingSuccess {
    /// The root value (solution where f(x) ≈ 0)
    double root;

    /// Number of iterations performed
    size_t iterations;

    /// Final error measure (|f(root)| for scalar methods)
    double final_error;
};

/// Error codes for root-finding failures
enum class RootFindingErrorCode {
    InvalidBracket,          ///< Initial bracket doesn't bracket the root
    MaxIterationsExceeded,   ///< Algorithm didn't converge in max_iter
    NumericalInstability,    ///< Numerical issues (NaN, division by zero)
    NoProgress              ///< Algorithm stopped making progress
};

/// Error result from root-finding methods
///
/// Returned when the root-finding algorithm fails to converge.
struct RootFindingError {
    /// Error code identifying the failure type
    RootFindingErrorCode code;

    /// Number of iterations performed before failure
    size_t iterations;

    /// Final error measure at failure point
    double final_error;

    /// Last value tried before failure (for diagnostics)
    /// Useful for understanding where the algorithm got stuck
    std::optional<double> last_value;
};

/// Result from any root-finding method
///
/// Uses std::expected for type-safe error handling without exceptions.
/// Success case contains the root value and convergence diagnostics.
/// Error case contains detailed failure information.
using RootFindingResult = std::expected<RootFindingSuccess, RootFindingError>;

/// Find root using Brent's method (host version with std::function)
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
/// @param f Function to find root of
/// @param a Left bracket
/// @param b Right bracket
/// @param config Root-finding configuration
/// @return Result with root (if converged) and convergence status
///
/// Reference: Brent, R. (1973). "Algorithms for Minimization without Derivatives"
inline RootFindingResult brent_find_root(const std::function<double(double)>& f,
                                         double a, double b,
                                         const RootFindingConfig& config) {
    // Evaluate endpoints
    double fa = f(a);
    double fb = f(b);

    // Check for NaN/Inf at endpoints (indicates invalid input or function failure)
    if (!std::isfinite(fa) || !std::isfinite(fb)) {
        return std::unexpected(RootFindingError{
            .code = RootFindingErrorCode::NumericalInstability,
            .iterations = 0,
            .final_error = std::numeric_limits<double>::quiet_NaN(),
            .last_value = std::nullopt
        });
    }

    // Check if root is bracketed
    if (fa * fb > 0.0) {
        return std::unexpected(RootFindingError{
            .code = RootFindingErrorCode::InvalidBracket,
            .iterations = 0,
            .final_error = std::min(std::abs(fa), std::abs(fb)),
            .last_value = std::nullopt
        });
    }

    // Check if endpoints are roots
    if (std::abs(fa) < config.brent_tol_abs) {
        return RootFindingSuccess{
            .root = a,
            .iterations = 0,
            .final_error = std::abs(fa)
        };
    }

    if (std::abs(fb) < config.brent_tol_abs) {
        return RootFindingSuccess{
            .root = b,
            .iterations = 0,
            .final_error = std::abs(fb)
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
        // Check convergence criteria
        if (std::abs(fb) < config.brent_tol_abs ||
            std::abs(b - a) < config.brent_tol_abs) {
            return RootFindingSuccess{
                .root = b,
                .iterations = iter + 1,
                .final_error = std::abs(fb)
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
            return std::unexpected(RootFindingError{
                .code = RootFindingErrorCode::NumericalInstability,
                .iterations = iter + 1,
                .final_error = std::numeric_limits<double>::quiet_NaN(),
                .last_value = s
            });
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
    return std::unexpected(RootFindingError{
        .code = RootFindingErrorCode::MaxIterationsExceeded,
        .iterations = config.max_iter,
        .final_error = std::abs(fb),
        .last_value = b
    });
}

/// Find root using bounded Newton-Raphson method (host version with std::function)
///
/// Iteratively refines an initial guess using Newton's method with bounds enforcement.
/// Uses the update rule: x_{n+1} = x_n - f(x_n)/f'(x_n), clamped to [x_min, x_max].
///
/// **Properties:**
/// - Quadratic convergence when close to root (if derivative is accurate)
/// - Requires derivative information (analytic or finite-difference)
/// - Bounds prevent divergence and ensure stability
/// - Detects flat regions (small derivative) and bounds hits
///
/// **Use cases:**
/// - Implied volatility with B-spline price surface (fast derivative via chain rule)
/// - Any bounded optimization where derivative is cheap to compute
/// - Problems where initial guess is good but needs refinement
///
/// **Advantages over Brent:**
/// - Faster convergence (quadratic vs superlinear) near root
/// - Better when derivative is cheap (e.g., automatic differentiation)
/// - No bracketing required, just initial guess and bounds
///
/// **Disadvantages:**
/// - Requires derivative (Brent doesn't)
/// - Can fail if derivative is zero or very small
/// - Less robust globally (Brent guarantees convergence if bracketed)
///
/// @param f Function to find root of (finds x where f(x) = 0)
/// @param df Derivative of f (df/dx)
/// @param x0 Initial guess
/// @param x_min Lower bound (x will stay >= x_min)
/// @param x_max Upper bound (x will stay <= x_max)
/// @param config Root-finding configuration (uses max_iter, tolerance)
/// @return Result with root (if converged), final derivative, and convergence status
inline RootFindingResult newton_find_root(const std::function<double(double)>& f,
                                          const std::function<double(double)>& df,
                                          double x0,
                                          double x_min, double x_max,
                                          const RootFindingConfig& config) {
    // Validate bounds
    if (x_min >= x_max) {
        return std::unexpected(RootFindingError{
            .code = RootFindingErrorCode::InvalidBracket,
            .iterations = 0,
            .final_error = std::numeric_limits<double>::quiet_NaN(),
            .last_value = std::nullopt
        });
    }

    // Clamp initial guess to bounds
    double x = std::clamp(x0, x_min, x_max);

    for (size_t iter = 0; iter < config.max_iter; ++iter) {
        // Evaluate function and derivative at current point
        const double fx = f(x);
        const double dfx = df(x);

        // Check for non-finite values
        if (!std::isfinite(fx) || !std::isfinite(dfx)) {
            return std::unexpected(RootFindingError{
                .code = RootFindingErrorCode::NumericalInstability,
                .iterations = iter + 1,
                .final_error = std::numeric_limits<double>::quiet_NaN(),
                .last_value = x
            });
        }

        // Compute absolute error
        const double error_abs = std::abs(fx);

        // Check convergence
        if (error_abs < config.tolerance) {
            return RootFindingSuccess{
                .root = x,
                .iterations = iter + 1,
                .final_error = error_abs
            };
        }

        // Check for numerical issues (flat derivative)
        if (std::abs(dfx) < 1e-10) {
            return std::unexpected(RootFindingError{
                .code = RootFindingErrorCode::NoProgress,
                .iterations = iter + 1,
                .final_error = error_abs,
                .last_value = x
            });
        }

        // Newton step: x_{n+1} = x_n - f(x_n)/f'(x_n)
        const double x_new = x - fx / dfx;

        // Enforce bounds
        const double x_clamped = std::clamp(x_new, x_min, x_max);

        // Check if bounds are hit repeatedly (may indicate convergence issues)
        if (x_new < x_min || x_new > x_max) {
            if (iter > 10) {
                return std::unexpected(RootFindingError{
                    .code = RootFindingErrorCode::NoProgress,
                    .iterations = iter + 1,
                    .final_error = error_abs,
                    .last_value = x_clamped
                });
            }
        }

        x = x_clamped;
    }

    // Max iterations reached
    const double fx_final = f(x);
    return std::unexpected(RootFindingError{
        .code = RootFindingErrorCode::MaxIterationsExceeded,
        .iterations = config.max_iter,
        .final_error = std::abs(fx_final),
        .last_value = x
    });
}

/// Device-callable Brent's method using function pointers
///
/// This version can be called from device code (in Kokkos kernels).
/// Uses function pointers instead of std::function for device compatibility.
///
/// @tparam F Function pointer type for objective function
/// @param f Function to find root of
/// @param a Left bracket
/// @param b Right bracket
/// @param config Root-finding configuration
/// @return Result with root (if converged) and convergence status
template<typename F>
KOKKOS_INLINE_FUNCTION
RootFindingResult brent_find_root_device(F f, double a, double b,
                                         const RootFindingConfig& config) {
    // Evaluate endpoints
    double fa = f(a);
    double fb = f(b);

    // Check for NaN/Inf at endpoints
    if (!Kokkos::isfinite(fa) || !Kokkos::isfinite(fb)) {
        return std::unexpected(RootFindingError{
            .code = RootFindingErrorCode::NumericalInstability,
            .iterations = 0,
            .final_error = 0.0,  // Can't use quiet_NaN in device code
            .last_value = std::nullopt
        });
    }

    // Check if root is bracketed
    if (fa * fb > 0.0) {
        return std::unexpected(RootFindingError{
            .code = RootFindingErrorCode::InvalidBracket,
            .iterations = 0,
            .final_error = (Kokkos::fabs(fa) < Kokkos::fabs(fb)) ? Kokkos::fabs(fa) : Kokkos::fabs(fb),
            .last_value = std::nullopt
        });
    }

    // Check if endpoints are roots
    if (Kokkos::fabs(fa) < config.brent_tol_abs) {
        return RootFindingSuccess{
            .root = a,
            .iterations = 0,
            .final_error = Kokkos::fabs(fa)
        };
    }

    if (Kokkos::fabs(fb) < config.brent_tol_abs) {
        return RootFindingSuccess{
            .root = b,
            .iterations = 0,
            .final_error = Kokkos::fabs(fb)
        };
    }

    // Ensure |f(b)| < |f(a)|
    if (Kokkos::fabs(fa) < Kokkos::fabs(fb)) {
        double tmp = a; a = b; b = tmp;
        double tmpf = fa; fa = fb; fb = tmpf;
    }

    double c = a;
    double fc = fa;
    bool mflag = true;
    double d = 0.0;

    for (size_t iter = 0; iter < config.max_iter; ++iter) {
        // Check convergence criteria
        if (Kokkos::fabs(fb) < config.brent_tol_abs ||
            Kokkos::fabs(b - a) < config.brent_tol_abs) {
            return RootFindingSuccess{
                .root = b,
                .iterations = iter + 1,
                .final_error = Kokkos::fabs(fb)
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
        bool condition2 = mflag && Kokkos::fabs(s - b) >= Kokkos::fabs(b - c) / 2.0;

        // Condition 3: mflag is not set and |s-b| >= |c-d|/2
        bool condition3 = !mflag && Kokkos::fabs(s - b) >= Kokkos::fabs(c - d) / 2.0;

        // Condition 4: mflag is set and |b-c| < tolerance
        bool condition4 = mflag && Kokkos::fabs(b - c) < config.brent_tol_abs;

        // Condition 5: mflag is not set and |c-d| < tolerance
        bool condition5 = !mflag && Kokkos::fabs(c - d) < config.brent_tol_abs;

        if (condition1 || condition2 || condition3 || condition4 || condition5) {
            // Use bisection
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        // Evaluate function at s
        double fs = f(s);

        // Check for NaN
        if (!Kokkos::isfinite(fs)) {
            return std::unexpected(RootFindingError{
                .code = RootFindingErrorCode::NumericalInstability,
                .iterations = iter + 1,
                .final_error = 0.0,
                .last_value = s
            });
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
        if (Kokkos::fabs(fa) < Kokkos::fabs(fb)) {
            double tmp = a; a = b; b = tmp;
            double tmpf = fa; fa = fb; fb = tmpf;
        }
    }

    // Max iterations reached
    return std::unexpected(RootFindingError{
        .code = RootFindingErrorCode::MaxIterationsExceeded,
        .iterations = config.max_iter,
        .final_error = Kokkos::fabs(fb),
        .last_value = b
    });
}

/// Device-callable Newton's method using function pointers
///
/// This version can be called from device code (in Kokkos kernels).
/// Uses function pointers instead of std::function for device compatibility.
///
/// @tparam F Function pointer type for objective function
/// @tparam DF Function pointer type for derivative function
/// @param f Function to find root of
/// @param df Derivative of f
/// @param x0 Initial guess
/// @param x_min Lower bound
/// @param x_max Upper bound
/// @param config Root-finding configuration
/// @return Result with root (if converged) and convergence status
template<typename F, typename DF>
KOKKOS_INLINE_FUNCTION
RootFindingResult newton_find_root_device(F f, DF df,
                                          double x0,
                                          double x_min, double x_max,
                                          const RootFindingConfig& config) {
    // Validate bounds
    if (x_min >= x_max) {
        return std::unexpected(RootFindingError{
            .code = RootFindingErrorCode::InvalidBracket,
            .iterations = 0,
            .final_error = 0.0,
            .last_value = std::nullopt
        });
    }

    // Clamp initial guess to bounds
    double x = (x0 < x_min) ? x_min : ((x0 > x_max) ? x_max : x0);

    for (size_t iter = 0; iter < config.max_iter; ++iter) {
        // Evaluate function and derivative at current point
        const double fx = f(x);
        const double dfx = df(x);

        // Check for non-finite values
        if (!Kokkos::isfinite(fx) || !Kokkos::isfinite(dfx)) {
            return std::unexpected(RootFindingError{
                .code = RootFindingErrorCode::NumericalInstability,
                .iterations = iter + 1,
                .final_error = 0.0,
                .last_value = x
            });
        }

        // Compute absolute error
        const double error_abs = Kokkos::fabs(fx);

        // Check convergence
        if (error_abs < config.tolerance) {
            return RootFindingSuccess{
                .root = x,
                .iterations = iter + 1,
                .final_error = error_abs
            };
        }

        // Check for numerical issues (flat derivative)
        if (Kokkos::fabs(dfx) < 1e-10) {
            return std::unexpected(RootFindingError{
                .code = RootFindingErrorCode::NoProgress,
                .iterations = iter + 1,
                .final_error = error_abs,
                .last_value = x
            });
        }

        // Newton step: x_{n+1} = x_n - f(x_n)/f'(x_n)
        const double x_new = x - fx / dfx;

        // Enforce bounds
        const double x_clamped = (x_new < x_min) ? x_min : ((x_new > x_max) ? x_max : x_new);

        // Check if bounds are hit repeatedly (may indicate convergence issues)
        if (x_new < x_min || x_new > x_max) {
            if (iter > 10) {
                return std::unexpected(RootFindingError{
                    .code = RootFindingErrorCode::NoProgress,
                    .iterations = iter + 1,
                    .final_error = error_abs,
                    .last_value = x_clamped
                });
            }
        }

        x = x_clamped;
    }

    // Max iterations reached
    const double fx_final = f(x);
    return std::unexpected(RootFindingError{
        .code = RootFindingErrorCode::MaxIterationsExceeded,
        .iterations = config.max_iter,
        .final_error = Kokkos::fabs(fx_final),
        .last_value = x
    });
}

}  // namespace mango::kokkos
