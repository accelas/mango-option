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

/**
 * NeumannBC: Specifies boundary gradient ∂u/∂x = g(x,t)
 *
 * Uses ghost-point method with orientation-aware formulas:
 * - Left boundary:  (u[1] - u[0]) / dx = g  →  u[0] = u[1] - g·dx
 * - Right boundary: (u[n-1] - u[n-2]) / dx = g  →  u[n-1] = u[n-2] + g·dx
 *
 * Requires diffusion coefficient D for proper ghost-point construction.
 */
template<typename Func>
class NeumannBC {
public:
    using tag = bc::neumann_tag;

    NeumannBC(Func f, double diffusion_coeff)
        : func_(std::move(f)), diffusion_coeff_(diffusion_coeff) {}

    // Natural interface - returns gradient
    double gradient(double t, double x) const {
        return func_(t, x);
    }

    double diffusion_coeff() const { return diffusion_coeff_; }

    // Solver interface - UNIFORM signature for all BC types
    // Neumann uses gradient, dx, and side to enforce du/dx = g via ghost point method
    void apply(double& u, double x, double t, double dx, double u_interior,
               [[maybe_unused]] double D, bc::BoundarySide side) const {
        // Ghost point method: enforce gradient by setting boundary value
        // Left boundary:  (u[1] - u[0]) / dx = g  →  u[0] = u[1] - g·dx
        // Right boundary: (u[n-1] - u[n-2]) / dx = g  →  u[n-1] = u[n-2] + g·dx
        double g = gradient(t, x);
        if (side == bc::BoundarySide::Left) {
            u = u_interior - g * dx;  // Forward difference
        } else {  // Right
            u = u_interior + g * dx;  // Backward difference
        }
    }

private:
    Func func_;
    double diffusion_coeff_;
};

// Deduction guide
template<typename Func>
NeumannBC(Func, double) -> NeumannBC<Func>;

} // namespace mango
