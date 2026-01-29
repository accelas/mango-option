# Mathematical Foundations

Mathematical formulations and numerical methods underlying the mango-option library.

**For software architecture, see [ARCHITECTURE.md](ARCHITECTURE.md)**
**For usage examples, see [API_GUIDE.md](API_GUIDE.md)**

## Table of Contents

1. [Black-Scholes PDE](#black-scholes-pde)
2. [TR-BDF2 Time Stepping](#tr-bdf2-time-stepping)
   - [Rannacher Startup](#rannacher-startup)
3. [American Option Constraints](#american-option-constraints)
4. [Grid Generation Strategies](#grid-generation-strategies)
   - [PDE Grid Estimation](#pde-grid-estimation)
   - [Price Table Grid Estimation](#price-table-grid-estimation)
   - [Adaptive Grid Refinement](#adaptive-grid-refinement)
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

### Rannacher Startup

TR-BDF2's trapezoidal Stage 1 preserves high-frequency modes because it has no numerical dissipation at the Nyquist frequency (amplification factor |R(z)| → 1 as z → −∞ along the imaginary axis). For smooth initial data this is harmless. For option pricing, the terminal payoff has a discontinuous first derivative (kink at the strike), which excites all frequencies equally. The trapezoidal rule propagates these modes unchanged, producing **oscillations in gamma** near the strike during the first few time steps.

Rannacher (1984) showed that replacing the first TR-BDF2 step with fully implicit backward Euler eliminates these oscillations. Backward Euler is L-stable with strong high-frequency damping: |R(z)| → 0 as z → −∞. One full step of backward Euler is sufficient to smooth the payoff kink before TR-BDF2 takes over.

**Implementation.** The first time step is replaced by two half-steps of implicit Euler:

```
Step 0 (Rannacher):
  u^{1/2} = u⁰ + (dt/2)·L(u^{1/2})     [implicit Euler, half-step]
  u¹      = u^{1/2} + (dt/2)·L(u¹)      [implicit Euler, half-step]

Steps 1 to N (standard TR-BDF2):
  Stage 1: trapezoidal to t_n + γ·dt
  Stage 2: BDF2 to t_{n+1}
```

Two half-steps rather than one full step provide better accuracy while retaining the damping properties of backward Euler. The half-step size matches the implicit Euler stability region more naturally to the problem's diffusion scale.

**Effect on accuracy.** Backward Euler is first-order, so the startup step introduces O(dt) local error. Since it applies to only one step out of N, the global contribution is O(dt/N) = O(dt²), preserving the overall second-order convergence of TR-BDF2.

**Configuration.** Rannacher startup is enabled by default (`TRBDF2Config::rannacher_startup = true`). The number of startup steps is fixed at one (two half-steps). Disabling it reverts to pure TR-BDF2 for all steps.

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

The library provides three spatial grid types (uniform, sinh-spaced, multi-sinh) and two automatic estimation strategies: one for single-option PDE solves, one for 4D price table pre-computation.

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

### PDE Grid Estimation

`estimate_grid_for_option()` constructs a sinh-spaced spatial grid and uniform time grid adapted to the option parameters. The goal is second-order accuracy in both space and time with minimal points.

**Domain bounds.** The PDE is solved in log-moneyness coordinates x = log(S/K). The domain extends ±n_σ standard deviations from the current log-moneyness:

```
x₀ = log(S/K)
x_min = x₀ - n_σ · σ√T
x_max = x₀ + n_σ · σ√T
```

With n_σ = 5 (default), this covers >99.99997% of the terminal distribution. The domain width scales with σ√T, so higher volatility or longer maturity automatically produces a wider grid.

**Spatial resolution.** Centered finite differences have truncation error O(Δx²). To achieve a target price error proportional to `tol`, the spacing must satisfy:

```
Δx_target = σ · √tol
N_x = ⌈(x_max - x_min) / Δx_target⌉
```

Scaling Δx with σ ensures that the number of points adapts to volatility: higher σ widens the domain but also coarsens Δx proportionally, keeping N_x stable. The √tol relationship means reducing error by 10× requires ~3.2× more points. N_x is clamped to [min_spatial_points, max_spatial_points] and rounded to odd (for centered stencils).

The N_x points are then placed via sinh spacing (see above), concentrating resolution near ATM where gamma is highest.

**Temporal resolution.** TR-BDF2 is L-stable and unconditionally stable, so there is no strict CFL constraint. However, second-order temporal accuracy requires Δt ~ O(Δx). The time step is coupled to the minimum spatial spacing (at the sinh cluster center):

```
Δx_min ≈ Δx_avg · exp(-α)
Δt = c_t · Δx_min
N_t = ⌈T / Δt⌉
```

With c_t = 0.75 (default) and α = 2.0, this ensures the temporal error does not dominate the spatial error. Coupling to Δx_min rather than Δx_avg prevents accuracy loss in the clustered region where the solution has the steepest gradients.

**Default parameters (`GridAccuracyParams`):**

| Parameter | Default | Effect |
|-----------|---------|--------|
| n_sigma | 5.0 | Domain half-width in σ√T units |
| alpha | 2.0 | Sinh clustering strength (~7× center-to-edge ratio) |
| tol | 1e-2 | Spatial truncation error target |
| c_t | 0.75 | Time-space coupling factor |
| min_spatial_points | 100 | Lower bound on N_x |
| max_spatial_points | 1200 | Upper bound on N_x |
| max_time_steps | 5000 | Upper bound on N_t |

For short-dated SPY options (σ ≈ 0.15, T ≈ 0.09), the defaults produce a 101 × 150 grid (15,150 elements).

### Price Table Grid Estimation

For price table pre-computation, grid density directly impacts IV accuracy. The library provides automatic estimation based on B-spline interpolation error theory.

**Theoretical basis:**
Cubic B-spline interpolation error is O(h⁴ · f''''(x)), where h is grid spacing and f'''' is the fourth derivative (curvature). Dimensions with higher curvature require finer grids.

**Curvature-based budget allocation:**
Each dimension receives points proportional to its curvature weight:

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Volatility (σ) | 1.5 | Highest curvature—vega non-linearity |
| Moneyness (m) | 1.0 | Moderate—log-transform handles ATM peak |
| Maturity (τ) | 1.0 | Baseline—moderate √τ behavior |
| Rate (r) | 0.6 | Lowest—nearly linear discounting |

**Grid density formula:**
```
base_points = (scale_factor / target_error)^(1/4)
n_dim = clamp(base_points × weight_dim, min_points, max_points)
```

Where:
- scale_factor ≈ 2.0 (calibrated empirically)
- target_error = desired IV accuracy (e.g., 0.001 for 10 bps)
- min_points = 4 (B-spline minimum)
- max_points = 50 (cost ceiling)

**Grid spacing strategies by dimension:**
- **Moneyness:** Log-uniform (matches internal storage)
- **Maturity:** √τ-uniform (concentrates near short expiries)
- **Volatility:** Uniform (highest curvature needs regular spacing)
- **Rate:** Uniform (nearly linear response)

**Empirical calibration:**
A 13×18×8 grid achieves ~4 bps average IV error across typical parameter ranges, which validates the scale_factor and weight choices.

### Adaptive Grid Refinement

When initial estimates are insufficient, iterative refinement improves accuracy:

**Algorithm:**
```
1. Build initial grid from curvature-based estimate
2. Sample N validation points via Latin Hypercube
3. For each sample:
   a. Evaluate price from B-spline surface
   b. Solve fresh FD problem for reference price
   c. Compute IV error metric
4. If max(error) ≤ target: done
5. Else: refine dimension with highest error attribution
6. Repeat until target met or max_iterations reached
```

**Error attribution:**
For each dimension, compute error contribution via partial derivative sensitivity analysis. The dimension causing largest error increase gets refined.

**Refinement strategy:**
```
n_dim_new = ceil(n_dim × refinement_factor)
```

Where refinement_factor ≈ 1.3 provides geometric growth.

**Convergence:**
- Typical: 2-3 iterations for 5 bps target
- Maximum: 5 iterations before declaring failure
- Validation: Fresh FD solves (not self-referential spline comparison)

**Hybrid error metric:**
For numerical stability, the error metric handles low-vega regions specially:

```
if vega > vega_floor:
    error = |IV_interpolated - IV_reference|  # Direct IV error
else:
    error = |price_interpolated - price_reference| / vega_floor  # Price-scaled
```

This prevents numerical instability when vega approaches zero (deep ITM/OTM).

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

### Clamped Knot Vectors

For n data points x₀...xₙ₋₁, the clamped knot vector has n+4 entries:

```
t = [x₀, x₀, x₀, x₀, t₁, ..., tₘ, xₙ₋₁, xₙ₋₁, xₙ₋₁, xₙ₋₁]
```

**Endpoint clamping** (multiplicity p+1 = 4 for cubics):
- Forces the B-spline curve to interpolate exactly at x₀ and xₙ₋₁
- Creates boundary conditions: B(x₀) = f₀ and B(xₙ₋₁) = fₙ₋₁
- Essential for price table interpolation (ensures exact values at grid boundaries)

**Interior knot placement:**
Interior knots t₁...tₘ are positioned strictly between data sites to satisfy the Schoenberg-Whitney condition (ensures collocation matrix is non-singular):

```cpp
// Proportional placement with epsilon clamping
T ratio = (idx + 1) / (n_interior + 1);
T pos = ratio * n_intervals;
int low = floor(pos);
T frac = pos - low;
T knot = (1 - frac) * x[low] + frac * x[low+1];

// Clamp to interior: avoid coinciding with data sites
T eps = max(1e-12 * spacing, machine_epsilon);
knot = clamp(knot, x[low] + eps, x[low+1] - eps);
```

**Why epsilon clamping matters:**
- Prevents knot coinciding exactly with data site x_i
- Avoids singular collocation matrix (repeated knot-data overlap)
- Maintains numerical stability (machine epsilon relative to spacing)

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

### Projected Thomas Algorithm (Brennan-Schwartz)

For American options, we must solve the Linear Complementarity Problem (LCP):

```
A·u = d
u ≥ ψ (obstacle constraint, e.g., payoff)
(A·u - d)ᵀ·(u - ψ) = 0 (complementarity)
```

**Key insight:** For tridiagonal systems with M-matrix structure (TR-BDF2 produces this), we can enforce the obstacle constraint DURING backward substitution, achieving single-pass convergence.

**Algorithm:**

1. **Forward elimination** (standard Thomas):
   ```
   c'[0] = c[0] / b[0]
   d'[0] = d[0] / b[0]

   For i = 1 to n-1:
     denom = b[i] - a[i-1]·c'[i-1]
     c'[i] = c[i] / denom
     d'[i] = (d[i] - a[i-1]·d'[i-1]) / denom
   ```

2. **Projected backward substitution** (KEY DIFFERENCE):
   ```
   u[n-1] = max(d'[n-1], ψ[n-1])

   For i = n-2 down to 0:
     unconstrained = d'[i] - c'[i]·u[i+1]
     u[i] = max(unconstrained, ψ[i])  // Projection at each step
   ```

**Why this works:**

The naive "solve then project" approach **violates the tridiagonal coupling**:
```
WRONG:
  1. Solve A·u = d unconstrained
  2. Project: u[i] = max(u[i], ψ[i])
  → Result: A·u ≠ d at projected nodes!
```

Projected Thomas **respects coupling** by projecting during backward substitution:
- Constraint enforced at EACH STEP, not after
- For M-matrices (positive diagonal, non-positive off-diagonals), the max(·, ψ) projection is monotone
- Backward substitution propagates constraints correctly through tridiagonal structure
- **Provably converges in single pass** for well-posed problems

**Performance:**
- Time: O(n) (same as standard Thomas)
- Space: O(n) workspace for c', d' arrays
- Iterations: **1 (always)** for M-matrices with proper dt

**Deep ITM exercise region locking:**

For nodes deep in-the-money (ψ > 0.95·intrinsic), convert to Dirichlet constraints to prevent diffusion from lifting values above intrinsic:

```cpp
if (psi[i] > 0.95 && u[i] ≈ psi[i]) {
    // Lock to intrinsic value
    jacobian_diag[i] = 1.0;
    jacobian_lower[i-1] = 0.0;
    jacobian_upper[i] = 0.0;
    rhs[i] = psi[i];
}
```

This ensures deep ITM American puts price at intrinsic value (99.75) instead of being erroneously lifted by diffusion (115.97).

**References:**
- Brennan & Schwartz (1977): "The Valuation of American Put Options"
- Hintermüller & Ito (2006): "A primal-dual active set strategy for general constrained optimization problems"

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
