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

/**
 * DirichletBC: Specifies boundary value u(x,t) = g(x,t)
 *
 * Template parameter Func should be callable with signature:
 *   double operator()(double t, double x) const
 */
template<typename Func>
class DirichletBC {
public:
    using tag = bc::dirichlet_tag;

    explicit DirichletBC(Func f) : func_(std::move(f)) {}

    // Natural interface - returns boundary value
    double value(double t, double x) const {
        return func_(t, x);
    }

    // Solver interface - UNIFORM signature for all BC types
    // Parameters: u (boundary value), x (position), t (time),
    //             dx (grid spacing), u_interior (neighbor value),
    //             D (diffusion coeff), side (boundary orientation)
    // Dirichlet only needs u, x, t but signature must match for polymorphism
    void apply(double& u, double x, double t,
               [[maybe_unused]] double dx,
               [[maybe_unused]] double u_interior,
               [[maybe_unused]] double D,
               [[maybe_unused]] bc::BoundarySide side) const {
        u = value(t, x);  // Directly set boundary value
    }

private:
    Func func_;  // Can capture state, no constraints
};

// Deduction guide for CTAD
template<typename Func>
DirichletBC(Func) -> DirichletBC<Func>;

} // namespace mango
