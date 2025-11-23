# Mathematical Foundations

Mathematical formulations and numerical methods underlying the mango-iv library.

**For software architecture, see [ARCHITECTURE.md](ARCHITECTURE.md)**
**For usage examples, see [API_GUIDE.md](API_GUIDE.md)**

## Table of Contents

1. [Black-Scholes PDE](#black-scholes-pde)
2. [TR-BDF2 Time Stepping](#tr-bdf2-time-stepping)
3. [American Option Constraints](#american-option-constraints)
4. [Grid Generation Strategies](#grid-generation-strategies)
5. [B-Spline Interpolation](#b-spline-interpolation)
6. [Root Finding Methods](#root-finding-methods)
7. [Convergence Analysis](#convergence-analysis)

---

## Black-Scholes PDE

### Backward Time Formulation

American options solve the Black-Scholes PDE backward in time from maturity to present:

```
∂V/∂τ = (1/2)σ²S² ∂²V/∂S² + (r-d)S ∂V/∂S - rV,  τ ∈ [0, T]
V(S,T) = payoff(S)
V(S,τ) ≥ intrinsic(S)     [American constraint]
```

Where:
- τ = T - t (time to maturity, forward time variable)
- V(S,τ) = option value
- S = spot price
- σ = volatility
- r = risk-free rate
- d = dividend yield

### Log-Price Transformation

Substituting x = ln(S/K) reduces the PDE to constant-coefficient form:

```
∂V/∂τ = (σ²/2) ∂²V/∂x² + (r-d-σ²/2) ∂V/∂x - rV
```

**Coefficients:**
- Second derivative (diffusion): σ²/2
- First derivative (drift): r - d - σ²/2
- Zeroth order (decay): -r

**Advantages:**
- Constant coefficients (don't depend on S)
- Natural moneyness scaling (x=0 is ATM)
- Better numerical stability
- Symmetric domain around strike

### Boundary Conditions

**Left boundary** (x → -∞, S → 0):
- Call: V(0, τ) = 0 (worthless)
- Put: V(0, τ) = K·e^(-rτ) (discounted strike)

**Right boundary** (x → ∞, S → ∞):
- Call: V ≈ S (parity, exercise value)
- Put: V(∞, τ) = 0 (worthless)

**Terminal condition** (τ = T):
```
V(x, T) = payoff(e^x · K)
  Call: max(e^x · K - K, 0) = K · max(e^x - 1, 0)
  Put:  max(K - e^x · K, 0) = K · max(1 - e^x, 0)
```

---

## TR-BDF2 Time Stepping

### Two-Stage Composite Scheme

TR-BDF2 combines trapezoidal rule and BDF2 for L-stable, second-order accurate time stepping:

**Stage 1** (Trapezoidal to t_n + γ·dt):
```
u_stage = u_n + γ·dt·[(1/2)L(u_n, t_n) + (1/2)L(u_stage, t_n + γ·dt)]
```

**Stage 2** (BDF2 to t_n+1):
```
u_{n+1} = [(1+2γ)/(1+γ)]·u_stage - [γ²/(1+γ)]·u_n + [dt/(1+γ)]·L(u_{n+1}, t_{n+1})
```

Where γ = 2 - √2 ≈ 0.5858 (chosen for L-stability)

### Properties

**Stability:**
- L-stable (all eigenvalues in left half-plane decay exponentially)
- Unconditionally stable (large dt possible for diffusion-dominated problems)
- Damping factor for spurious high-frequency modes

**Accuracy:**
- Second-order accurate in time (O(dt²))
- Spatial accuracy limited by finite difference stencil (O(dx²))

**Computational Cost:**
- 2 implicit solves per time step (Stage 1 + Stage 2)
- Each solve requires Newton iteration (~3-5 iterations typical)
- Tridiagonal linear system per Newton iteration (O(n) via Thomas)

---

## American Option Constraints

### Obstacle Condition

American options enforce early exercise via obstacle (variational inequality):

```
V(x,τ) ≥ ψ(x)  for all τ ∈ [0, T]
```

Where ψ(x) is the intrinsic value (payoff from immediate exercise):
- Call: ψ(x) = max(e^x · K - K, 0)
- Put: ψ(x) = max(K - e^x · K, 0)

### Projection Method

After each Newton iteration, project solution onto feasible set:

```
u^{k+1} ← max(u^{k+1}, ψ)
```

This enforces V ≥ intrinsic value at all grid points.

**Order of operations** (critical):
1. Update interior points (Newton iteration)
2. Apply boundary conditions
3. Apply obstacle projection

### Early Exercise Boundary

Free boundary S_exercise(τ) separates continuation and exercise regions:

```
S_exercise(τ) = sup{S : V(S,τ) > intrinsic(S)}
```

**Put options:**
- Exercise when S < S_exercise (deep ITM)
- Continue when S > S_exercise (OTM or shallow ITM)

**Call options:**
- Exercise when S > S_exercise (deep ITM)
- Continue when S < S_exercise (OTM or shallow ITM)

**Dividend impact:**
- Discrete dividends create jumps in early exercise boundary
- Calls may optimally exercise just before ex-dividend
- Puts benefit from holding (dividend reduces spot price)

---

## Grid Generation Strategies

### Uniform Grids

Equally spaced points in log-moneyness:

```
x_i = x_min + i·dx,  dx = (x_max - x_min)/(n-1)
```

**Use cases:**
- Simple testing and debugging
- Problems without localized features

**Limitations:**
- Wastes resolution far from strike
- Poor for steep gradients near ATM

### Sinh-Spaced Grids

Hyperbolic sine transformation concentrates points near center:

```
ξ_i = -1 + 2i/(n-1)            [uniform in ξ ∈ [-1, 1]]
x_i = x_c + (Δx/α)·sinh(α·ξ_i) [sinh-spaced in x]
```

Where:
- x_c = (x_min + x_max)/2 (center point)
- Δx = (x_max - x_min)/2 (half-width)
- α controls concentration (typical: 2-3)

**Properties:**
- dx_min ≈ (Δx/n)·exp(-α) near center
- dx_max ≈ (Δx/n)·exp(+α) at boundaries
- Smooth, monotonic spacing

**Use cases:**
- Single American option (concentrate near strike)
- IV calculation (steep gamma near ATM)
- Recommended for most applications

### Multi-Sinh Grids

Weighted combination of multiple sinh transformations:

```
x_i = Σ_k w_k · sinh_k(ξ_i)  [normalized weights]
```

Each cluster k specifies:
- center_x: Location for concentration
- alpha: Strength of clustering
- weight: Relative importance

**Use cases:**
- Price tables covering multiple strikes (e.g., ATM + deep ITM)
- Batch pricing with different moneyness
- When single-center sinh leaves important regions coarse

**Cluster merging:**
Automatically merges clusters closer than 0.3/α_avg to prevent wasted resolution.

**Guidelines:**
- Use single sinh unless strikes differ by >20% (Δx ≥ 0.18)
- Each additional cluster adds ~10% computational cost
- Validate spacing via monotonicity checks

---

## B-Spline Interpolation

### Cubic B-Spline Basis

Cox-de Boor recursive formula for degree-3 splines:

```
N_{i,0}(x) = 1 if x ∈ [t_i, t_{i+1}), else 0
N_{i,k}(x) = [(x-t_i)/(t_{i+k}-t_i)]·N_{i,k-1}(x) + [(t_{i+k+1}-x)/(t_{i+k+1}-t_{i+1})]·N_{i+1,k-1}(x)
```

**Properties:**
- Compact support (non-zero on 4 adjacent intervals for cubic)
- C² continuity (2 continuous derivatives)
- Partition of unity (Σ_i N_i(x) = 1)
- Local control (changing one coefficient affects 4 intervals)

### 4D Separable Fitting

Price table uses tensor-product B-splines:

```
P(m, τ, σ, r) = Σ_{i,j,k,l} c_{ijkl} · N_i(m) · N_j(τ) · N_k(σ) · N_l(r)
```

**Separable algorithm** (4 sequential 1D fits):
1. Fix (τ, σ, r), fit moneyness → c_{*jkl}
2. Fix (σ, r), fit maturity on previous coefficients → c_{*,*,kl}
3. Fix r, fit volatility → c_{*,*,*,l}
4. Fit rate → c_{*,*,*,*}

**Collocation system** (banded matrix per dimension):
```
Σ_j c_j N_j(x_i) = f_i  for data points (x_i, f_i)
```

**Banded solver:**
- Cubic splines → 4-diagonal collocation matrix
- Banded LU: O(n) time vs O(n³) dense
- 7.8× speedup on large grids (50×30×20×10)

### Greeks via Differentiation

B-splines provide analytical derivatives:

**Delta** (∂P/∂m):
```
∂P/∂m = Σ_{i,j,k,l} c_{ijkl} · N_i'(m) · N_j(τ) · N_k(σ) · N_l(r)
```

**Vega** (∂P/∂σ):
```
∂P/∂σ = Σ_{i,j,k,l} c_{ijkl} · N_i(m) · N_j(τ) · N_k'(σ) · N_l(r)
```

**Gamma** (∂²P/∂m²):
```
∂²P/∂m² = Σ_{i,j,k,l} c_{ijkl} · N_i''(m) · N_j(τ) · N_k(σ) · N_l(r)
```

Cost: Same as price evaluation (~500ns, no additional computation)

---

## Root Finding Methods

### Brent's Method

Combines bisection, secant, and inverse quadratic interpolation for robust root-finding:

**Algorithm:**
1. Start with bracketing interval [a, b] where f(a)·f(b) < 0
2. Choose interpolation (secant or inverse quadratic) if safe
3. Fall back to bisection if interpolation fails or bracket too wide
4. Iterate until |f(x)| < tolerance or |b-a| < ε

**Properties:**
- Guaranteed convergence if root bracketed
- Superlinear convergence rate (~1.6)
- No derivatives required
- More robust than Newton for IV calculation

**Use cases:**
- American IV calculation (nested with PDE pricing)
- ~5-8 iterations typical for vol ∈ [0.01, 3.0]

### Newton's Method

Quadratic convergence for smooth functions:

**Iteration:**
```
x_{k+1} = x_k - f(x_k)/f'(x_k)
```

**Use cases:**
- TR-BDF2 implicit stages (analytical Jacobian available)
- Interpolated IV solver (fast vega from B-spline derivatives)

**Newton for PDEs:**
Solves F(u) = u - dt·α·L(u) - RHS = 0 at each implicit stage:
- Jacobian: J = I - dt·α·∂L/∂u (tridiagonal for 1D)
- Linear solve: Thomas algorithm (O(n))
- Convergence: 3-5 iterations typical

---

## Convergence Analysis

### Spatial Discretization Error

Centered finite differences for spatial operators:

**Second derivative:**
```
∂²u/∂x² |_i ≈ (u_{i+1} - 2u_i + u_{i-1})/dx²
```
Truncation error: O(dx²)

**First derivative:**
```
∂u/∂x |_i ≈ (u_{i+1} - u_{i-1})/(2dx)
```
Truncation error: O(dx²)

**Non-uniform grids:**
Finite difference weights adjusted for variable spacing:
```
∂²u/∂x² |_i = w_left·u_{i-1} + w_center·u_i + w_right·u_{i+1}
```

### Temporal Discretization Error

TR-BDF2 provides second-order accuracy:
- Local truncation error: O(dt³)
- Global error: O(dt²)

**Stability criterion** (diffusion-dominated):
```
dt ≤ C·dx_min²/D
```
Where:
- C ≈ 1/2 for explicit methods (FTCS)
- C = ∞ for TR-BDF2 (unconditionally stable)

**Practical limit:**
For sinh grids with α=2, use dt ≈ 0.75·dx_min for accuracy:
```
dt ≈ 0.75 · (Δx/n)·exp(-α)
```

### Obstacle Convergence

Projection method converges under mild regularity:

**Theorem** (Kinderlehrer-Stampacchia):
If obstacle ψ ∈ C² and initial condition u₀ ≥ ψ, then:
- Solution exists and is unique
- V(x,τ) ≥ ψ(x) for all τ
- Free boundary S_exercise(τ) is Lipschitz continuous

**Numerical convergence:**
- O(dx²) spatial accuracy maintained with obstacle
- Projection is non-expansive (doesn't amplify errors)
- May slow Newton iterations (active set changes)

### Grid Independence Study

Verify solution converged by refining grid until results stabilize:

**Procedure:**
1. Solve on coarse grid (n=100, dt=1e-3)
2. Refine spatially (n=200, dt=1e-3)
3. Refine temporally (n=200, dt=5e-4)
4. Check |V_refined - V_coarse| < tolerance

**Typical convergence:**
- ATM options: 1e-3 price error at n=141, dt=1e-3
- Deep ITM/OTM: Require finer grids (steep boundaries)
- Greeks: Need 2-3× finer grids than prices

---

## References

1. **TR-BDF2 Scheme**: Bank et al. (1985), "Transient Simulation of Silicon Devices and Circuits"
2. **American Options**: Wilmott, "Derivatives", Chapter 11
3. **Finite Differences**: LeVeque, "Finite Difference Methods for ODEs and PDEs"
4. **B-Splines**: de Boor, "A Practical Guide to Splines"
5. **Obstacle Problems**: Kinderlehrer & Stampacchia, "An Introduction to Variational Inequalities"
6. **Root Finding**: Press et al., "Numerical Recipes", Chapter 9

**For implementation details, see [ARCHITECTURE.md](ARCHITECTURE.md)**
**For usage examples, see [API_GUIDE.md](API_GUIDE.md)**
