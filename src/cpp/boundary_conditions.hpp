#pragma once

#include <concepts>
#include <functional>

namespace mango {
namespace bc {

// Tag types for boundary conditions
struct dirichlet_tag {};
struct neumann_tag {};
struct robin_tag {};

// Boundary side enum for orientation-dependent BCs
enum class BoundarySide { Left, Right };

// Type trait to extract tag from BC type
template<typename BC>
using boundary_tag_t = typename BC::tag;

} // namespace bc

/**
 * BoundaryCondition concept: Types that provide boundary values
 *
 * Requirements:
 * - Must have a 'tag' type (dirichlet_tag, neumann_tag, or robin_tag)
 * - Must have apply() method with uniform signature
 */
template<typename T>
concept BoundaryCondition = requires {
    typename bc::boundary_tag_t<T>;
};

} // namespace mango
