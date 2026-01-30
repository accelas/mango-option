<!-- SPDX-License-Identifier: MIT -->
# TR-BDF2 Time-Stepping Stagnation Problem

## Problem Statement

A TR-BDF2 (Trapezoidal Rule - Backward Differentiation Formula 2) time-stepping scheme exhibits solution stagnation after approximately 10-20 time steps when solving parabolic PDEs. The numerical solution stops evolving despite having large spatial operator values, suggesting the time-stepping mechanism is reverting to a fixed point rather than advancing forward in time.

## Mathematical Formulation

### PDE Problem

We solve the semi-discrete parabolic PDE:

$$\frac{\partial u}{\partial t} = L(u)$$

where $L$ is a spatial differential operator. For testing, we use:

$$L(u) = \frac{\partial^2 u}{\partial x^2} - u + 1$$

with Dirichlet boundary conditions:
- $u(0,t) = 0$
- $u(1,t) = 0$
- $u(x,0) = 0$

The steady-state solution is analytically known:

$$u_{\infty}(x) = 1 - \frac{\cosh(x - 0.5)}{\cosh(0.5)}$$

### TR-BDF2 Scheme

The TR-BDF2 method is a composite two-stage implicit scheme with parameter $\gamma = 2 - \sqrt{2} \approx 0.5858$.

**Stage 1 (Trapezoidal Rule):** Advance from $t^n$ to $t^n + \gamma \Delta t$

$$u^* - \frac{\gamma \Delta t}{2} L(u^*) = u^n + \frac{\gamma \Delta t}{2} L(u^n)$$

**Stage 2 (BDF2):** Advance from $t^n$ to $t^{n+1}$ using intermediate solution $u^*$

Define $\alpha = 1 - \gamma \approx 0.4142$. The BDF2 formula is:

$$(1 + 2\alpha) u^{n+1} - (1 + \alpha) u^* + \alpha u^n = (1 - \alpha) \Delta t \cdot L(u^{n+1})$$

Rearranging into implicit form:

$$u^{n+1} - \frac{(1-\alpha)\Delta t}{1+2\alpha} L(u^{n+1}) = \frac{(1+\alpha)u^* - \alpha u^n}{1+2\alpha}$$

### Numerical Constants

For the test problem:
- $\Delta t = 0.01$
- $\gamma \approx 0.5858$
- $\alpha \approx 0.4142$
- Stage 1 coefficient: $\gamma \Delta t / 2 \approx 0.002929$
- Stage 2 coefficient: $(1-\alpha)\Delta t/(1+2\alpha) \approx 0.003172$

## Newton-Raphson Solution Method

Each implicit equation $u - c \cdot L(u) = r$ (where $c$ is the time-step coefficient and $r$ is the right-hand side) is solved using Newton-Raphson iteration:

1. Compute Jacobian matrix $J$ where $J_{ij} = \frac{\partial L_i}{\partial u_j}$ using finite differences
2. Build system matrix: $A = I - c \cdot J$
3. For each iteration $k$:
   - Compute residual: $\text{res} = r - u^{(k)} + c \cdot L(u^{(k)})$
   - Solve: $A \cdot \delta u = \text{res}$
   - Update: $u^{(k+1)} = u^{(k)} + \delta u$
4. Converge when $\|\text{res}\| < 10^{-6}$

**Note:** The Jacobian $J$ is computed once at the beginning of each implicit solve and reused for all Newton iterations (quasi-Newton approach).

## Observed Behavior

### Symptom: Solution Stagnation

After approximately 10-20 time steps, the numerical solution ceases to evolve:

| Time Step | $u(0.5, t)$ | $L(u)(0.5)$ | Expected $du/dt$ |
|-----------|-------------|-------------|------------------|
| 0         | 0.000000    | 1.000000    | 1.000            |
| 1         | 0.007683    | 0.988691    | 0.989            |
| 10        | 0.015829    | 0.934960    | 0.935            |
| 20        | 0.015841    | 0.934826    | 0.935            |
| 100       | 0.015841    | 0.934826    | 0.935 ← **STUCK**|

**Key observation:** $L(u) \approx 0.935$ indicates $du/dt$ should be large, but $u$ is not changing between time steps.

### Detailed Newton Iteration Pattern

Debug instrumentation reveals the following pattern within a single time step:

**Stage 1 (TR):** Starting from $u^n$, mid-point value $u(0.5) = 0.01583$

| Iteration | $u(0.5)$  | Residual Norm |
|-----------|-----------|---------------|
| 0         | 0.01583   | $4.3 \times 10^{-3}$ |
| 1         | 0.02124   | $2.4 \times 10^{-14}$ ← **Converged** |

**Stage 2 (BDF2):** Starting from $u^* = 0.02124$

| Iteration | $u(0.5)$  | Residual Norm |
|-----------|-----------|---------------|
| 0         | 0.02124   | $4.3 \times 10^{-3}$ |
| 1         | 0.01584   | $2.3 \times 10^{-14}$ ← **Converged back!** |

**Critical observation:** Stage 2 reverts the solution nearly back to $u^n$, undoing Stage 1's progress. The net change over the full time step is:

$$u^{n+1}(0.5) - u^n(0.5) = 0.01584 - 0.01583 = 0.00001$$

This is approximately **1000× smaller** than expected from $du/dt \approx 0.935 \times \Delta t \approx 0.00935$.

## Mathematical Verification of Stage 1

The Stage 1 implicit equation should satisfy:

$$u^* - \frac{\gamma \Delta t}{2} L(u^*) = u^n + \frac{\gamma \Delta t}{2} L(u^n)$$

Numerical check at mid-point after Stage 1 convergence:
- $u^n(0.5) = 0.01584$
- $u^*(0.5) = 0.02124$
- $L(u^n)(0.5) = 0.9348$
- $L(u^*)(0.5) = 0.9246$
- $c = \gamma \Delta t / 2 = 0.002929$

**LHS:** $u^* - c \cdot L(u^*) = 0.02124 - 0.002929 \times 0.9246 = 0.01853$

**RHS:** $u^n + c \cdot L(u^n) = 0.01584 + 0.002929 \times 0.9348 = 0.01858$

**Difference:** $|0.01853 - 0.01858| = 5 \times 10^{-4}$ ← **NOT SATISFIED!**

Despite Newton reporting convergence with residual norm $\sim 10^{-14}$, the Stage 1 equation has a discrepancy of $5 \times 10^{-4}$.

## Questions for Expert Review

1. **TR-BDF2 Formulation:** Is the Stage 2 BDF2 equation correctly formulated? Specifically, is the coefficient
   $$\frac{(1-\alpha)\Delta t}{1+2\alpha}$$
   correct for the spatial operator term?

2. **Stage Coupling:** Should Stage 2 use the exact Stage 1 solution $u^*$, or is there an intermediate transformation required?

3. **Jacobian Reuse:** For the quasi-Newton approach with Jacobian computed once per stage:
   - Is it valid to reuse $J$ evaluated at the initial guess for all Newton iterations?
   - Could Jacobian staleness cause convergence to incorrect solutions?

4. **BDF2 Stability:** Could the BDF2 stage have a stability issue that causes it to:
   - Converge to a solution that approximates $u^n$ rather than advancing forward?
   - Produce an oscillatory pattern between stages?

5. **Discretization Error:** Are there known issues with TR-BDF2 when applied to reaction-diffusion equations of the form $L(u) = \nabla^2 u - u + f$?

6. **Alternative Formulation:** Should we consider the "L-stable" variant of TR-BDF2 or a different stage coupling?

## Numerical Data for Verification

### Stage 2 RHS Computation

With $\alpha = 0.4142$, $u^* = 0.02124$, $u^n = 0.01583$:

$$\text{RHS} = \frac{(1+\alpha)u^* - \alpha u^n}{1+2\alpha} = \frac{1.4142 \times 0.02124 - 0.4142 \times 0.01583}{1.8284}$$

$$= \frac{0.03003 - 0.00656}{1.8284} = \frac{0.02347}{1.8284} = 0.01284$$

This RHS value (0.01284) is **less than both** $u^n$ (0.01583) and $u^*$ (0.02124), which may contribute to the regression behavior.

### Expected vs. Actual Behavior

For a properly advancing scheme starting from $u(0.5) = 0.01583$ with $du/dt \approx 0.935$:

- **Expected after $\Delta t = 0.01$:** $u \approx 0.01583 + 0.935 \times 0.01 = 0.0252$
- **Actual Stage 1 result:** $u^* = 0.02124$ (reasonable intermediate)
- **Actual Stage 2 result:** $u^{n+1} = 0.01584$ ← **Regression to starting value**

The scheme appears to have an attractor at $u \approx 0.0158$ where Stage 2 consistently pulls the solution back regardless of Stage 1's advancement.

## Additional Context

- **Grid:** Uniform spatial grid with $\Delta x = 0.02$ (51 points)
- **Boundary Treatment:** Dirichlet BCs enforced in Newton system matrix
- **Spatial Discretization:** Standard centered finite differences for $\partial^2/\partial x^2$
- **Convergence Criterion:** $\|\text{residual}\|_2 / \sqrt{n} < 10^{-6}$ (absolute tolerance)

The Newton solver consistently achieves $\|\text{residual}\| \sim 10^{-14}$, indicating excellent numerical convergence, yet the converged solution does not satisfy the intended PDE time-stepping equation.
