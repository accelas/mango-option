<!-- SPDX-License-Identifier: MIT -->
# Root Cause Analysis: Neumann BC Mass Non-Conservation

## Executive Summary

The Neumann boundary condition implementation violates mass conservation not because the spatial operator is wrong, but because the **boundary points are evolved by algebraic constraints** rather than by the PDE. This creates an inconsistency in the time-stepping scheme.

## Key Finding

**The spatial operator IS conservative:**
- Σ L(u)_i · Δx ≈ 0 (within machine precision ~10⁻¹⁶)
- The discrete Laplacian telescopes correctly
- Setting L(u)₀ = L(u)ₙ₋₁ = 0 does not break spatial conservation

**The time-stepping is NOT conservative:**
- Mass increases by ~1.96% over 100 time steps
- Boundary points u₀ and uₙ₋₁ are evolved differently than interior points
- This breaks the temporal discretization's conservation property

## Detailed Analysis

### Spatial Conservation (✓ Works)

The discrete mass balance requires:
$$\frac{d}{dt}\left(\sum_{i=0}^{n-1} u_i \Delta x\right) = \sum_{i=0}^{n-1} \frac{du_i}{dt} \Delta x = \sum_{i=0}^{n-1} L(u)_i \Delta x$$

For a conservative spatial operator, we need $\sum_i L(u)_i = 0$.

**Verification:**
```
Step 0:  Σ L(u)·dx = -1.99e-16  ✓ (machine precision)
Step 1:  Σ L(u)·dx =  6.83e-16  ✓
Step 5:  Σ L(u)·dx =  1.39e-15  ✓
```

The spatial operator satisfies discrete conservation perfectly.

### Temporal Evolution (✗ Broken)

The TR-BDF2 scheme should evolve ALL points according to:
$$u_i^{n+1} - c \cdot L(u^{n+1})_i = \text{RHS}_i$$

where c is the time coefficient and RHS comes from previous stages.

**For interior points (i = 1, ..., n-2):**

The Newton system solves:
$$\left(\mathbf{I} - c \cdot \mathbf{J}\right) \delta\mathbf{u} = \mathbf{F}$$

where:
- F_i = RHS_i - u_old_i + c·L(u_old)_i (residual of implicit equation)
- This correctly evolves the PDE

**For boundary points (i = 0, n-1) with Neumann BC:**

The Newton system is modified:
- Row 0: `diag[0] = 1, upper[0] = -1, lower[0] = 0`
- Residual: `F₀ = -u₀ + u₁ - Δx·g`

This enforces the **algebraic constraint** u₀ = u₁ directly, rather than evolving u₀ according to the PDE.

### The Inconsistency

At each time step, the solver does:

**Interior points:**
$$\Delta u_i = \text{(Newton correction from PDE evolution)}$$
$$u_i^{new} = u_i^{old} + \Delta u_i$$

**Boundary points:**
$$\Delta u_0 = u_1^{new} - u_0^{old} \quad \text{(from algebraic constraint)}$$
$$u_0^{new} = u_1^{new}$$

The boundary points are **not** being integrated by the TR-BDF2 scheme for the PDE du/dt = L(u). Instead, they're being set equal to their neighbor at each Newton iteration.

This means:
- Interior points follow: $\frac{du_i}{dt} \approx L(u)_i$ (via TR-BDF2)
- Boundary points follow: $u_0 = u_1$, $u_{n-1} = u_{n-2}$ (algebraic constraint)

The boundary constraint does NOT imply that $\frac{du_0}{dt} = \frac{du_1}{dt}$!

### Why Mass Increases

Since the boundary points are constrained to equal their neighbors, their rate of change is determined by the neighbor's evolution, not by their own PDE evolution.

If u₁ is evolving (spreading due to diffusion), then u₀ is forced to follow it. But this creates extra "mass" at the boundary because:
- u₁ evolves according to the PDE with boundary effect from u₀
- u₀ is forced to equal u₁
- The combined system has more degrees of freedom than it should

The system is essentially doing:
1. Evolve u₁ using the PDE with stencil (u₀, u₁, u₂)
2. Set u₀ = u₁
3. This increases both u₀ and u₁, creating net mass

### Mathematical Proof of Non-Conservation

Consider a simplified TR-BDF2 step. For conservation, we need:
$$\sum_i (u_i^{n+1} - u_i^n) = \sum_i c \cdot L(u^{n+1})_i + \sum_i c \cdot L(u^n)_i$$

Since Σ L(u) = 0 (verified), the RHS = 0, so we should have Σ u^{n+1} = Σ u^n.

But with the current implementation:
- For i = 1, ..., n-2: The equation $u_i - c·L(u)_i = \text{RHS}_i$ is solved correctly
- For i = 0: The equation $u_0 - u_1 = 0$ is enforced (NOT the PDE!)
- For i = n-1: The equation $u_{n-1} - u_{n-2} = 0$ is enforced

So the system being solved is:
$$\begin{pmatrix}
1 & -1 & 0 & \cdots & 0 \\
\text{(I-cJ)}_{1,0} & \text{(I-cJ)}_{1,1} & \text{(I-cJ)}_{1,2} & \cdots & 0 \\
\vdots & & \ddots & & \vdots \\
0 & \cdots & 0 & -1 & 1
\end{pmatrix}
\begin{pmatrix}
\delta u_0 \\
\delta u_1 \\
\vdots \\
\delta u_{n-1}
\end{pmatrix}
=
\begin{pmatrix}
F_0^{BC} \\
F_1^{PDE} \\
\vdots \\
F_{n-1}^{BC}
\end{pmatrix}$$

The boundary rows (0 and n-1) do NOT correspond to the PDE evolution equations. This breaks the conservation structure.

## Numerical Evidence

**Test case:**
- Initial mass: M₀ = 0.2507
- Final mass (100 steps): Mf = 0.2556
- Increase: 1.96%

**Mass evolution:**
```
Step 0:  M/M₀ = 1.0000
Step 1:  M/M₀ = 1.00000537
Step 10: M/M₀ = 1.00156
Step 20: M/M₀ = 1.00616
Step 100: M/M₀ = 1.01961
```

The mass increases steadily and consistently, not due to numerical instability but due to the fundamental inconsistency in how boundary points are evolved.

## Solution

The ghost point method resolves this by:
1. Extending the domain to include fictitious points u₋₁ and uₙ
2. Setting u₋₁ = u₁ and uₙ = uₙ₋₂ (reflecting the zero-flux BC)
3. Evolving ALL physical points (i = 0, ..., n-1) using the PDE
4. Updating ghost points after each step

This ensures that boundary points are evolved by the PDE, not by algebraic constraints, maintaining conservation.

## References

- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.
  - Chapter 2.12: "Boundary conditions for diffusion equations"
  - Recommends ghost point method for Neumann BCs

- Morton, K. W., & Mayers, D. F. (2005). *Numerical Solution of Partial Differential Equations*. Cambridge University Press.
  - Section 3.5: "Conservative finite difference schemes"
  - Discusses importance of maintaining conservation at boundary
