# Neumann Boundary Condition Mass Conservation Problem

## Problem Statement

The PDE solver with Neumann (zero-flux) boundary conditions fails to conserve mass in a closed diffusion system. This violates fundamental physical conservation laws.

## Mathematical Formulation

### PDE Problem

Consider the pure diffusion equation on a 1D domain:

$$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, 1], \quad t > 0$$

where $D = 0.1$ is the diffusion coefficient.

**Boundary conditions (Neumann - zero flux):**
$$\frac{\partial u}{\partial x}\bigg|_{x=0} = 0, \quad \frac{\partial u}{\partial x}\bigg|_{x=1} = 0$$

**Initial condition (Gaussian pulse):**
$$u_0(x) = \exp\left(-50(x - 0.5)^2\right)$$

**Time domain:**
- $t \in [0, 1]$
- Time step: $\Delta t = 0.01$
- Total steps: 100

**Spatial discretization:**
- Grid points: $n = 101$
- Spacing: $\Delta x = 0.01$

### Conservation Property

For a closed system with zero-flux boundary conditions, the **total mass must be conserved**:

$$M(t) = \int_0^1 u(x, t) \, dx = \text{constant}$$

**Proof:** Integrate the PDE over the domain:
$$\frac{d}{dt} \int_0^1 u \, dx = D \int_0^1 \frac{\partial^2 u}{\partial x^2} \, dx = D \left[ \frac{\partial u}{\partial x}\bigg|_{x=1} - \frac{\partial u}{\partial x}\bigg|_{x=0} \right] = 0$$

The last equality follows from the Neumann boundary conditions. Therefore, $\frac{dM}{dt} = 0$, so mass is conserved.

### Discrete Mass

The discrete mass is computed using the trapezoidal rule:
$$M = \sum_{i=0}^{n-1} u_i \cdot \Delta x$$

**Expected:** $M(t) = M(0)$ for all $t$

**Observed:** $M(t_{\text{final}}) / M(0) \approx 1.020$ (2% error, exceeds tolerance of 1%)

## Numerical Method

### TR-BDF2 Time Stepping

The solver uses a two-stage TR-BDF2 scheme with $\gamma = 2 - \sqrt{2} \approx 0.5858$.

**Stage 1 (Trapezoidal rule):**
$$u^* - \frac{\gamma \Delta t}{2} L(u^*) = u^n + \frac{\gamma \Delta t}{2} L(u^n)$$

**Stage 2 (BDF2):**
$$u^{n+1} - \frac{(1-\gamma)\Delta t}{2-\gamma} L(u^{n+1}) = \frac{1}{\gamma(2-\gamma)}u^* - \frac{(1-\gamma)^2}{\gamma(2-\gamma)}u^n$$

where $L(u) = D \frac{\partial^2 u}{\partial x^2}$ is the spatial operator.

### Spatial Discretization

Interior points use standard centered finite differences:
$$\frac{\partial^2 u}{\partial x^2}\bigg|_{x_i} \approx \frac{u_{i-1} - 2u_i + u_{i+1}}{\Delta x^2}$$

### Current Neumann BC Implementation

The Neumann boundary condition $\frac{\partial u}{\partial x} = 0$ is discretized using:

**At left boundary ($i = 0$):**
$$\frac{u_1 - u_0}{\Delta x} = 0 \quad \Rightarrow \quad u_0 = u_1$$

**At right boundary ($i = n-1$):**
$$\frac{u_{n-1} - u_{n-2}}{\Delta x} = 0 \quad \Rightarrow \quad u_{n-1} = u_{n-2}$$

### Newton-Raphson System

Each implicit stage is solved using Newton iteration:
$$\mathbf{F}(\mathbf{u}) = \mathbf{u} - c \cdot L(\mathbf{u}) - \mathbf{r} = \mathbf{0}$$

where $c$ is the time coefficient and $\mathbf{r}$ is the RHS from the previous stage.

The linearized system at each Newton iteration:
$$\left(\mathbf{I} - c \cdot \mathbf{J}\right) \delta\mathbf{u} = \mathbf{F}(\mathbf{u}_{\text{old}})$$

where $\mathbf{J} = \frac{\partial L}{\partial \mathbf{u}}$ is the Jacobian of the spatial operator.

### Boundary Condition Enforcement

**Matrix rows (algebraic constraints):**

Left Neumann ($i = 0$):
$$\text{diag}[0] = 1.0, \quad \text{upper}[0] = -1.0$$
This enforces: $u_0 - u_1 = 0$

Right Neumann ($i = n-1$):
$$\text{diag}[n-1] = 1.0, \quad \text{lower}[n-2] = -1.0$$
This enforces: $u_{n-1} - u_{n-2} = 0$

**Residual computation:**

Left Neumann:
$$r_0 = -u_0 + u_1 - \Delta x \cdot g$$
where $g = 0$ for zero flux.

Right Neumann:
$$r_{n-1} = -u_{n-1} + u_{n-2} + \Delta x \cdot g$$

## Observed Behavior

**Test result:**
```
mass_final / mass_initial = 1.0196 (expected: 1.0 ± 0.01)
FAILED: difference is 0.0196, exceeds tolerance 0.01
```

**Analysis:**
- Mass increases by approximately 2% over 100 time steps
- This indicates that the Neumann boundary implementation is not maintaining the zero-flux property correctly
- The solution should spread out (diffusion) but total integral should remain constant

## Theoretical Issues

### Null Space Problem

For **pure Neumann boundary conditions** (both boundaries are Neumann), the continuous PDE has a **null space**: the solution is determined only up to an additive constant.

Mathematically, if $u(x,t)$ is a solution, then $u(x,t) + C$ is also a solution for any constant $C$.

This means:
1. The discrete system matrix may be **singular** or **near-singular**
2. Without proper handling, the solution can drift in the null space direction
3. Standard iterative solvers may not converge properly

### Discrete Conservation

For the discrete system to conserve mass, we need:
$$\sum_{i=0}^{n-1} \frac{du_i}{dt} \cdot \Delta x = 0$$

This requires that the discrete spatial operator satisfies:
$$\sum_{i=0}^{n-1} L(u)_i \cdot \Delta x = 0$$

**For interior points:** The sum of finite difference terms telescopes correctly.

**For boundary points:** The current implementation sets $L(u)_0 = L(u)_{n-1} = 0$, but this may not be compatible with the constraint equations $u_0 = u_1$ and $u_{n-1} = u_{n-2}$ in a way that preserves the telescoping property.

### Coupling Issue

The current implementation creates a **coupled system** where:
1. Row 0 enforces the algebraic constraint $u_0 = u_1$
2. Row 1 evolves $u_1$ according to the PDE using the stencil involving $u_0$, $u_1$, and $u_2$

Since row 0 modifies the matrix structure (setting `diag[0] = 1, upper[0] = -1`), but row 1 still depends on $u_0$ through the finite difference stencil, there may be an inconsistency in how the boundary couples to the interior evolution.

## Questions for Expert Review

1. **Discrete conservation:** Does the current discretization of Neumann BCs guarantee that $\sum_i L(u)_i = 0$ as required for mass conservation?

2. **Boundary operator:** Should $L(u)_0$ and $L(u)_{n-1}$ be defined using one-sided differences or ghost points, rather than being set to zero?

3. **Matrix structure:** Is there an inconsistency between:
   - Enforcing $u_0 = u_1$ via matrix row 0
   - Computing $L(u)_1$ using the stencil that depends on $u_0$?

4. **Null space:** For pure Neumann problems, should we explicitly constrain the solution (e.g., fix the total mass or pin one value) to remove the null space?

5. **Ghost point method:** Would using ghost points $u_{-1}$ and $u_n$ with:
   - $u_{-1} = u_1$ (reflecting zero flux at left)
   - $u_n = u_{n-2}$ (reflecting zero flux at right)
   - Applying standard stencil at all interior points including $i=0$ and $i=n-1$

   provide better conservation properties?

6. **Alternative formulation:** Should the Neumann BC be enforced weakly (through the residual only) rather than strongly (through matrix constraints)?

## Numerical Data

**Initial mass:**
$$M_0 = \sum_{i=0}^{100} \exp(-50(x_i - 0.5)^2) \cdot \Delta x$$

For the Gaussian pulse centered at $x = 0.5$ with width parameter 50:
$$M_0 \approx \int_0^1 \exp(-50(x-0.5)^2) \, dx \approx \sqrt{\frac{\pi}{50}} \approx 0.2507$$

**Expected final mass:**
$$M_{\text{final}} = M_0$$

**Observed final mass:**
$$M_{\text{final}} \approx 1.0196 \cdot M_0$$

**Error:**
$$\text{Relative error} = \frac{M_{\text{final}} - M_0}{M_0} \approx 1.96\%$$

This exceeds the tolerance of 1% and indicates a systematic issue with the Neumann BC implementation.

## Summary

The Neumann boundary condition implementation violates mass conservation for a pure diffusion problem with zero-flux boundaries. The 2% mass increase suggests that the current discretization is not maintaining the zero-flux property at the discrete level, allowing spurious mass generation at the boundaries.

Possible solutions include:
- Ghost point method for natural incorporation of Neumann BCs
- Correcting the discrete operator at boundaries to ensure telescoping sum
- Explicitly constraining the null space for pure Neumann problems
- Using weak enforcement of Neumann BCs through natural boundary conditions
