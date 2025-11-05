#pragma once

#include "root_finding.hpp"
#include <concepts>
#include <cmath>
#include <limits>

namespace mango {

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
    // STUB: Return failure for now
    // Full implementation will be added in Task 3
    return RootFindingResult{
        .converged = false,
        .iterations = 0,
        .final_error = std::numeric_limits<double>::infinity(),
        .failure_reason = "Not implemented yet"
    };
}

}  // namespace mango
