// SPDX-License-Identifier: MIT
#pragma once

namespace mango::operators {

/// LaplacianPDE: Pure diffusion operator L(u) = D·∂²u/∂x²
///
/// Time-independent heat/diffusion equation:
/// ∂u/∂t = D·∂²u/∂x²
///
/// @tparam T Value type (default: double)
template<typename T = double>
class LaplacianPDE {
public:
    explicit LaplacianPDE(T diffusion_coeff)
        : D_(diffusion_coeff)
    {}

    /// Evaluate operator at a point: L(u) = D·d²u/dx²
    ///
    /// @param d2u Second derivative ∂²u/∂x² at point
    /// @param du First derivative ∂u/∂x at point (unused)
    /// @param u Value at point (unused)
    /// @return Operator value D·d²u/dx²
    T operator()(T d2u, [[maybe_unused]] T du, [[maybe_unused]] T u) const {
        return D_ * d2u;
    }

private:
    T D_;  // Diffusion coefficient
};

} // namespace mango::operators
