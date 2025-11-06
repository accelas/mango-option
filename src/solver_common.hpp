/**
 * @file solver_common.hpp
 * @brief Common types and concepts for linear system solvers
 *
 * Shared definitions used across ThomasSolver, BandedLU, and other solvers.
 */

#pragma once

#include <concepts>
#include <string_view>
#include <optional>

namespace mango {

/// Floating point concept for template constraints
template<typename T>
concept FloatingPoint = std::floating_point<T>;

}  // namespace mango
