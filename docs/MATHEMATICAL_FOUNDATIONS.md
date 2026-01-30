<!-- SPDX-License-Identifier: MIT -->
# Mathematical Foundations

Mathematical formulations and numerical methods underlying the mango-option library.

**For software architecture, see [ARCHITECTURE.md](ARCHITECTURE.md)**
**For usage examples, see [API_GUIDE.md](API_GUIDE.md)**

## Table of Contents

**Part I — PDE Pricing:** Everything needed to price a single American option.

1. [Black-Scholes PDE](#1-black-scholes-pde)
2. [Spatial Discretization](#2-spatial-discretization)
3. [TR-BDF2 Time Stepping](#3-tr-bdf2-time-stepping)
4. [American Option Constraints](#4-american-option-constraints)
5. [Grid Generation](#5-grid-generation)

**Part II — Price Tables & Interpolation:** Pre-computing prices across parameter space for fast queries.

6. [B-Spline Interpolation](#6-b-spline-interpolation)
7. [Price Table Grid Estimation](#7-price-table-grid-estimation)
8. [Implied Volatility](#8-implied-volatility)

**Part III — Analysis**

9. [Convergence Analysis](#9-convergence-analysis)

---

# Part I — PDE Pricing

The goal: given an American option's parameters (spot, strike, volatility, rate, maturity), compute its fair value and Greeks. We do this by solving a PDE on a finite grid, stepping backward in time from the known terminal payoff.

The sections below follow the computation pipeline: formulate the PDE, discretize space, step through time, enforce the American early-exercise constraint, and finally choose a good grid.

---

## 1. Black-Scholes PDE

### Backward Time Formulation

An American option's value V(S, t) satisfies the Black-Scholes PDE. We work in backward time tau = T - t (time remaining to maturity), which turns the problem into a forward-in-time evolution — easier to reason about numerically:

```
dV/dtau = (1/2) sigma^2 S^2 d^2V/dS^2 + (r - d) S dV/dS - rV
```

with initial condition V(S, 0) = payoff(S) and the American constraint V >= intrinsic at all times.

Here S is the spot price, sigma the volatility, r the risk-free rate, and d the continuous dividend yield. The three terms on the right have intuitive meanings: diffusion (volatility spreads the distribution), drift (the risk-neutral growth rate), and discounting (a dollar tomorrow is worth less today).

### Log-Price Transformation

Working directly in S is inconvenient — the coefficients depend on S, and the domain is [0, infinity). Substituting x = ln(S/K) fixes both problems:

```
dV/dtau = (sigma^2 / 2) d^2V/dx^2 + (r - d - sigma^2/2) dV/dx - rV
```

Now all coefficients are constants, x = 0 corresponds to at-the-money (S = K), and the domain is symmetric around the strike. This is the form we actually discretize and solve.

### Boundary Conditions

At the edges of our finite computational domain:

**Left boundary** (x -> -infinity, S -> 0):
- Call: V = 0 (worthless far out-of-the-money)
- Put: V = K e^{-r tau} (deep in-the-money, exercise is certain)

**Right boundary** (x -> +infinity, S -> infinity):
- Call: V ~ S (deep in-the-money)
- Put: V = 0 (worthless far out-of-the-money)

**Terminal condition** (tau = 0, i.e., at maturity):
```
Call: V(x, 0) = K max(e^x - 1, 0)
Put:  V(x, 0) = K max(1 - e^x, 0)
```

This terminal condition is the starting point for our backward-in-time solve.

---

## 2. Spatial Discretization

With the PDE in log-price coordinates, we need to approximate the spatial derivatives on a discrete grid.

### Centered Finite Differences

On a uniform grid with spacing dx, the standard second-order approximations are:

**Second derivative:**
```
d^2u/dx^2 |_i ~ (u_{i+1} - 2 u_i + u_{i-1}) / dx^2     error: O(dx^2)
```

**First derivative:**
```
du/dx |_i ~ (u_{i+1} - u_{i-1}) / (2 dx)                 error: O(dx^2)
```

Both are second-order accurate, meaning halving dx reduces the error by 4x. This is important for the grid estimation strategy later.

### Non-Uniform Grids

Sinh-spaced grids (section 5) have variable spacing. The finite difference weights must account for this:

```
d^2u/dx^2 |_i = w_left u_{i-1} + w_center u_i + w_right u_{i+1}
```

where the weights depend on the local spacings dx_{i-1} and dx_i. The truncation error remains O(dx^2) as long as the spacing varies smoothly — which sinh grids guarantee by construction.

### Assembling the Spatial Operator

Combining the second derivative, first derivative, and zeroth-order terms from the Black-Scholes PDE into a single operator L:

```
L(u)_i = (sigma^2/2) d^2u/dx^2 |_i + (r - d - sigma^2/2) du/dx |_i - r u_i
```

This produces a tridiagonal matrix: each grid point couples only to its immediate neighbors. This structure is what makes the solve fast — O(n) per linear solve instead of O(n^3).

---

## 3. TR-BDF2 Time Stepping

We now have a spatial discretization L. The PDE becomes a system of ODEs:

```
du/dtau = L(u)
```

We need to march this forward in tau from the terminal payoff. The choice of time-stepping scheme matters a lot for stability and accuracy.

### Why TR-BDF2?

Explicit methods (forward Euler, RK4) require tiny time steps for stability — the CFL condition forces dt ~ dx^2 for diffusion problems, which is impractical with fine grids. Implicit methods remove this restriction but typically sacrifice either accuracy or damping properties.

TR-BDF2 is a composite two-stage scheme that gives us everything:
- **L-stable**: spurious high-frequency modes decay exponentially
- **Second-order accurate**: O(dt^2) global error
- **Unconditionally stable**: dt is limited only by accuracy, not stability

### The Two-Stage Scheme

Each time step from t_n to t_{n+1} proceeds in two stages:

**Stage 1 — Trapezoidal rule** (advance to t_n + gamma dt):
```
u_stage = u_n + gamma dt [(1/2) L(u_n) + (1/2) L(u_stage)]
```

**Stage 2 — BDF2** (advance to t_{n+1}):
```
u_{n+1} = [(1+2gamma)/(1+gamma)] u_stage
         - [gamma^2/(1+gamma)] u_n
         + [dt/(1+gamma)] L(u_{n+1})
```

The parameter gamma = 2 - sqrt(2) ~ 0.5858 is chosen specifically to make the scheme L-stable.

### Newton Iteration for Implicit Stages

Both stages are implicit — u_stage and u_{n+1} appear on both sides of their equations. We solve each by Newton's method.

Rearranging Stage 1 as F(u) = 0:

```
F(u) = u - u_n - gamma dt [(1/2) L(u_n) + (1/2) L(u)]
```

Newton's iteration:
```
J delta_u = -F(u^k)
u^{k+1} = u^k + delta_u
```

where J = I - (gamma dt / 2) dL/du is the Jacobian. Since L is tridiagonal, so is J — each Newton step costs just one tridiagonal solve via the Thomas algorithm (O(n)).

Convergence is typically fast: 3-5 Newton iterations per implicit stage. The Jacobian is assembled analytically from the spatial operator coefficients, avoiding the cost of finite-difference Jacobian approximation.

### Rannacher Startup

There is one subtlety. The trapezoidal rule in Stage 1 has no numerical dissipation at the Nyquist frequency — it preserves high-frequency modes exactly. For smooth initial data, this is fine. But option payoffs have a kink at the strike (discontinuous first derivative), which excites all frequencies equally. The result: **oscillations in gamma** near the strike during the first few time steps.

Rannacher (1984) showed that replacing the first TR-BDF2 step with backward Euler eliminates this. Backward Euler is strongly dissipative — it damps high-frequency modes aggressively, smoothing the payoff kink before TR-BDF2 takes over.

The implementation uses two half-steps of backward Euler:

```
Step 0 (Rannacher):
  u^{1/2} = u^0 + (dt/2) L(u^{1/2})      [implicit Euler, half-step]
  u^1     = u^{1/2} + (dt/2) L(u^1)       [implicit Euler, half-step]

Steps 1 to N (standard TR-BDF2):
  Stage 1: trapezoidal to t_n + gamma dt
  Stage 2: BDF2 to t_{n+1}
```

Two half-steps rather than one full step preserve better accuracy while retaining the damping properties.

Does this hurt the overall accuracy? Backward Euler is first-order, introducing O(dt) local error. But it applies to only one step out of N, so the global contribution is O(dt/N) = O(dt^2) — the same order as TR-BDF2. The overall second-order convergence is preserved.

---

## 4. American Option Constraints

European options just solve the PDE and read off the answer. American options add a constraint: the holder can exercise at any time, so the option value must never fall below the immediate exercise (intrinsic) value.

### The Obstacle Condition

At every grid point and every time step:

```
V(x, tau) >= psi(x)
```

where psi(x) is the intrinsic value:
- Call: psi(x) = max(K e^x - K, 0)
- Put: psi(x) = max(K - K e^x, 0)

This turns the PDE into a **variational inequality** — a free boundary problem where the exercise boundary is part of the solution, not an input.

### Projected Thomas Algorithm (Brennan-Schwartz)

The question is: how do we enforce the obstacle constraint while solving the tridiagonal systems from Newton iteration?

The naive approach — solve the unconstrained system, then project — breaks the tridiagonal coupling:

```
WRONG:
  1. Solve A u = d ignoring the obstacle
  2. Set u[i] = max(u[i], psi[i])
  Result: A u != d at projected nodes. The solution is inconsistent.
```

The Projected Thomas algorithm, due to Brennan & Schwartz (1977), does something more elegant. It enforces the constraint **during** backward substitution rather than after.

**Forward elimination** (identical to standard Thomas):
```
c'[0] = c[0] / b[0]
d'[0] = d[0] / b[0]

For i = 1 to n-1:
  denom = b[i] - a[i-1] c'[i-1]
  c'[i] = c[i] / denom
  d'[i] = (d[i] - a[i-1] d'[i-1]) / denom
```

**Projected backward substitution** (the key difference):
```
u[n-1] = max(d'[n-1], psi[n-1])

For i = n-2 down to 0:
  unconstrained = d'[i] - c'[i] u[i+1]
  u[i] = max(unconstrained, psi[i])
```

The max() at each step is the projection. Because the matrix A is an M-matrix (positive diagonal, non-positive off-diagonals — guaranteed by TR-BDF2's discretization), this projection is monotone and the algorithm converges in a **single pass**. Same O(n) cost as standard Thomas, no iteration needed.

### Why This Works

The insight is about information flow. In backward substitution, u[i] depends on u[i+1]. If we project u[i+1] upward (to the obstacle), this propagates correctly through the tridiagonal coupling — the constraint at one node affects its neighbors in a consistent way.

For M-matrices, the off-diagonal elements are non-positive, so increasing u[i+1] can only decrease the unconstrained value at u[i]. This means the projection is monotone: once a node is clamped to the obstacle, nodes to its left see a larger u[i+1] and thus a smaller unconstrained value, making them more likely to also hit the obstacle. This is exactly the early-exercise region propagating inward from deep in-the-money.

### Deep ITM Locking

One practical detail: for nodes deep in-the-money where psi is close to the maximum intrinsic value, numerical diffusion can erroneously lift the solution above intrinsic. We prevent this by converting deep ITM nodes to Dirichlet constraints:

```
if psi[i] > 0.95 * max_intrinsic and u[i] ~ psi[i]:
    lock row i to u[i] = psi[i]
```

This ensures, for example, that a deep ITM put with intrinsic value 99.75 prices at 99.75 rather than being lifted to 115.97 by diffusion.

### The Early Exercise Boundary

The free boundary S_exercise(tau) separates two regions:

- **Continuation region**: V > psi (hold the option, it's worth more alive)
- **Exercise region**: V = psi (exercise immediately)

For puts, the exercise region is S < S_exercise (deep ITM). For calls, it's S > S_exercise. The Projected Thomas algorithm finds this boundary implicitly — we never compute it directly; it emerges from the constraint enforcement.

---

## 5. Grid Generation

The grid determines both accuracy and cost. Too coarse and the solution is inaccurate; too fine and the solve is slow. The library provides three grid types and an automatic estimation strategy.

### Uniform Grids

Equally spaced points in log-moneyness:

```
x_i = x_min + i dx,    dx = (x_max - x_min) / (n-1)
```

Simple and useful for testing, but wasteful for option pricing: most of the interesting behavior (gamma peak, exercise boundary) is concentrated near the strike, while far OTM/ITM regions are nearly linear.

### Sinh-Spaced Grids

The workhorse grid for option pricing. A hyperbolic sine transformation concentrates points near a center while maintaining smooth spacing:

```
xi_i = -1 + 2i/(n-1)                       [uniform in [-1, 1]]
x_i  = x_c + (Dx/alpha) sinh(alpha xi_i)   [sinh-spaced in x]
```

where x_c is the center (typically 0 = ATM), Dx is the half-width, and alpha controls the concentration. With alpha = 2:

- Spacing near center: dx_min ~ (Dx/n) exp(-alpha) — about 7x finer than uniform
- Spacing at boundaries: dx_max ~ (Dx/n) exp(alpha) — about 7x coarser than uniform

This puts resolution where it matters (near the strike) and saves points where it doesn't (far tails). The spacing varies smoothly and monotonically, so the non-uniform finite difference weights remain well-conditioned.

### Multi-Sinh Grids

When pricing across multiple strikes (e.g., for price tables), a single concentration center is insufficient. Multi-sinh grids superpose several sinh transformations:

```
x_i = sum_k w_k sinh_k(xi_i)     [normalized weights]
```

Each cluster specifies a center, alpha, and weight. Clusters closer than 0.3/alpha_avg are automatically merged to avoid wasted resolution. Use this when strikes differ by more than ~20%.

### Automatic PDE Grid Estimation

`estimate_grid_for_option()` builds a sinh grid tailored to a specific option. The logic:

**Domain bounds.** Extend +/- n_sigma standard deviations from the current log-moneyness:

```
x_min = ln(S/K) - n_sigma sigma sqrt(T)
x_max = ln(S/K) + n_sigma sigma sqrt(T)
```

With n_sigma = 5 (default), this covers >99.99997% of the terminal distribution. The width scales with sigma sqrt(T) — higher volatility or longer maturity automatically produces a wider domain.

**Spatial resolution.** The centered difference truncation error is O(dx^2). To achieve a target error proportional to tol:

```
dx_target = sigma sqrt(tol)
N_x = ceil((x_max - x_min) / dx_target)
```

Scaling dx with sigma keeps N_x stable across volatilities: higher sigma widens the domain but proportionally coarsens the target spacing. The sqrt(tol) relationship means 10x better accuracy costs ~3.2x more points. N_x is clamped to [100, 1200].

**Temporal resolution.** TR-BDF2 is unconditionally stable, so there is no CFL constraint. But second-order accuracy requires dt ~ O(dx_min). The time step couples to the finest spatial spacing:

```
dt = c_t dx_min,    where dx_min ~ dx_avg exp(-alpha)
N_t = ceil(T / dt)
```

With c_t = 0.75 and alpha = 2.0, this ensures temporal error doesn't dominate spatial error in the clustered region where gradients are steepest.

**Default parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| n_sigma | 5.0 | Domain half-width in sigma sqrt(T) units |
| alpha | 2.0 | Sinh clustering strength (~7x center-to-edge ratio) |
| tol | 1e-2 | Spatial truncation error target |
| c_t | 0.75 | Time-space coupling factor |
| min_spatial_points | 100 | Lower bound on N_x |
| max_spatial_points | 1200 | Upper bound on N_x |
| max_time_steps | 5000 | Upper bound on N_t |

For a short-dated SPY option (sigma ~ 0.15, T ~ 0.09), the defaults produce a 101 x 150 grid.

---

# Part II — Price Tables & Interpolation

Part I gave us a PDE solver that prices one option in ~1-2ms. For implied volatility — which requires pricing the option repeatedly at different volatilities until the price matches the market — this is too slow. A single IV solve takes ~15ms (5-8 Brent iterations x 2ms each), and a trading desk needs thousands of IVs per second.

The solution: pre-compute prices across a 4D parameter grid (moneyness, maturity, volatility, rate), fit a B-spline surface, and evaluate the surface at ~500ns per query. This section covers the interpolation machinery, grid estimation, and IV extraction.

---

## 6. B-Spline Interpolation

### Why B-Splines?

We need a smooth interpolant over a 4D grid of pre-computed prices. Requirements:
- C^2 continuity (smooth Greeks via differentiation)
- Local support (changing one region doesn't affect distant regions)
- Fast evaluation (~hundreds of nanoseconds)

Cubic B-splines satisfy all three. They provide C^2 continuity, each basis function is non-zero on only 4 adjacent intervals, and evaluation requires only local data.

### Cubic B-Spline Basis

The B-spline basis functions are defined recursively (Cox-de Boor):

```
N_{i,0}(x) = 1 if x in [t_i, t_{i+1}),  else 0
N_{i,k}(x) = [(x - t_i)/(t_{i+k} - t_i)] N_{i,k-1}(x)
            + [(t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1})] N_{i+1,k-1}(x)
```

Key properties:
- **Compact support**: each cubic basis function spans 4 intervals
- **Partition of unity**: sum_i N_i(x) = 1 everywhere
- **C^2 continuity**: two continuous derivatives (sufficient for delta, gamma, vega)
- **Local control**: modifying one coefficient affects only 4 intervals

### Clamped Knot Vectors

For n data points x_0 ... x_{n-1}, the clamped knot vector repeats the endpoints with multiplicity p+1 = 4:

```
t = [x_0, x_0, x_0, x_0,  t_1, ..., t_m,  x_{n-1}, x_{n-1}, x_{n-1}, x_{n-1}]
```

This forces the B-spline to interpolate exactly at the endpoints — essential for price tables where boundary values must be exact.

Interior knots are placed proportionally between data sites, with epsilon clamping to avoid coinciding with data sites (which would make the collocation matrix singular):

```
knot = clamp(proportional_position, x[low] + eps, x[low+1] - eps)
```

This satisfies the Schoenberg-Whitney condition, guaranteeing a non-singular collocation system.

### 4D Separable Fitting

The price table uses tensor-product B-splines:

```
P(m, tau, sigma, r) = sum_{i,j,k,l} c_{ijkl} N_i(m) N_j(tau) N_k(sigma) N_l(r)
```

Fitting all 4 dimensions simultaneously would require solving a dense (n_m n_tau n_sigma n_r)^2 system — completely impractical for a 50x30x20x10 grid (300K unknowns).

The separable algorithm exploits the tensor-product structure with 4 sequential 1D fits:

1. Fix (tau, sigma, r), fit moneyness for each slice -> c_{*,j,k,l}
2. Fix (sigma, r), fit maturity on the coefficients from step 1 -> c_{*,*,k,l}
3. Fix r, fit volatility -> c_{*,*,*,l}
4. Fit rate -> c_{*,*,*,*}

Each 1D fit solves a banded collocation system. Cubic splines produce a 4-diagonal matrix, solved in O(n) via banded LU. The total cost is O(n_m n_tau n_sigma n_r) — linear in the grid size.

### Greeks via Differentiation

A major benefit of B-splines: derivatives are analytic. To compute delta (dP/dm), differentiate the basis functions in m while leaving the others untouched:

```
dP/dm     = sum c_{ijkl} N_i'(m) N_j(tau) N_k(sigma) N_l(r)      [delta]
dP/dsigma = sum c_{ijkl} N_i(m)  N_j(tau) N_k'(sigma) N_l(r)     [vega]
d^2P/dm^2 = sum c_{ijkl} N_i''(m) N_j(tau) N_k(sigma) N_l(r)     [gamma]
```

Each derivative costs the same as a price evaluation (~500ns) because we evaluate one differentiated 1D basis and three undifferentiated ones.

---

## 7. Price Table Grid Estimation

The 4D grid density directly controls IV accuracy. Too coarse and the B-spline interpolation introduces significant error; too fine and pre-computation takes days. The library provides both automatic estimation and iterative refinement.

### Curvature-Based Budget Allocation

The cubic B-spline interpolation error is O(h^4 f''''(x)), where h is the grid spacing and f'''' is the fourth derivative. Dimensions with higher curvature need finer grids.

Each dimension receives points proportional to its curvature weight:

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Volatility (sigma) | 1.5 | Highest curvature — vega non-linearity |
| Moneyness (m) | 1.0 | Moderate — log-transform handles ATM peak |
| Maturity (tau) | 1.0 | Baseline — sqrt(tau) behavior |
| Rate (r) | 0.6 | Lowest — nearly linear discounting |

The formula:
```
base_points = (scale_factor / target_error)^{1/4}
n_dim = clamp(base_points x weight_dim, 4, 50)
```

where scale_factor ~ 2.0 (calibrated empirically) and target_error is the desired IV accuracy (e.g., 0.001 for 10 bps).

Grid spacing strategies vary by dimension:
- **Moneyness**: log-uniform (matches the log-price coordinate)
- **Maturity**: sqrt(tau)-uniform (concentrates near short expiries where theta is steepest)
- **Volatility**: uniform (highest curvature needs regular spacing)
- **Rate**: uniform (nearly linear response)

### Adaptive Grid Refinement

When the initial estimate isn't accurate enough, iterative refinement improves it:

```
1. Build initial grid from curvature-based estimate
2. Sample N validation points via Latin Hypercube
3. For each sample:
   a. Evaluate price from B-spline surface
   b. Solve a fresh PDE for the reference price
   c. Compute IV error
4. If max(error) <= target: done
5. Else: refine the dimension with highest error attribution
6. Repeat until target met or max_iterations reached
```

The key detail is step 3b: validation uses fresh PDE solves, not the spline itself. This prevents the refinement from chasing its own tail.

**Error attribution** identifies which dimension to refine by computing partial derivative sensitivity — the dimension contributing the most error gets more points. The refinement factor is ~1.3x (geometric growth), and convergence typically takes 2-3 iterations for a 5 bps target.

**Low-vega handling.** Deep ITM/OTM options have near-zero vega, making IV error numerically unstable. The error metric switches to price-scaled error in these regions:

```
if vega > vega_floor:
    error = |IV_interpolated - IV_reference|
else:
    error = |price_interpolated - price_reference| / vega_floor
```

---

## 8. Implied Volatility

Implied volatility is the volatility that, when plugged into the pricing model, reproduces the observed market price. Computing it requires inverting the price-to-volatility mapping — a root-finding problem.

The library provides two approaches with very different speed/accuracy tradeoffs.

### FDM-Based IV (Brent's Method)

The direct approach: use the PDE solver as a black box and find the root of

```
f(sigma) = V_model(sigma) - V_market = 0
```

We use Brent's method, which combines bisection, secant, and inverse quadratic interpolation. It requires a bracketing interval [sigma_lo, sigma_hi] where f changes sign, then iterates:

1. Choose interpolation step (secant or inverse quadratic) if it's safe
2. Fall back to bisection if interpolation fails or the bracket is too wide
3. Stop when |f(sigma)| < tolerance or the bracket is smaller than epsilon

**Properties:**
- Guaranteed convergence (as long as the root is bracketed)
- Superlinear convergence rate (~1.6)
- No derivatives required (important — dV/dsigma from the PDE is expensive)
- Typically 5-8 iterations for sigma in [0.01, 3.0]

Each iteration calls the PDE solver (~2ms), so a single IV solve costs ~15ms. Accurate but too slow for production use with thousands of queries.

### Interpolated IV (Newton on B-Spline Surface)

The fast approach: pre-compute a 4D price table (Part II, sections 6-7), then solve for IV using Newton's method on the B-spline surface.

```
sigma_{k+1} = sigma_k - [P(m, tau, sigma_k, r) - V_market] / [dP/dsigma(m, tau, sigma_k, r)]
```

The key advantage: both P and dP/dsigma come from B-spline evaluation (~500ns each), not PDE solves. Newton's method converges quadratically (error squares each iteration), typically in 3-4 iterations.

**Performance comparison:**

| Method | Time per IV | Use case |
|--------|------------|----------|
| FDM (Brent) | ~15ms | Ground truth, validation, few queries |
| Interpolated (Newton) | ~3.5us | Production, many queries |

The interpolated solver is ~5,000x faster, at the cost of pre-computation time and interpolation error (typically 10-60 bps depending on grid profile).

---

# Part III — Analysis

## 9. Convergence Analysis

### Overall Error Budget

The total pricing error has three independent contributions:

```
error_total ~ C_x dx^2 + C_t dt^2 + C_obstacle
```

- **Spatial**: O(dx^2) from centered differences
- **Temporal**: O(dt^2) from TR-BDF2
- **Obstacle**: the projection is non-expansive (doesn't amplify errors) and the free boundary is Lipschitz continuous (Kinderlehrer-Stampacchia)

The grid estimation strategy (section 5) balances these by coupling dt to dx_min, ensuring neither dominates.

### Grid Independence

To verify convergence, refine the grid until results stabilize:

1. Solve on coarse grid (n = 100, dt = 1e-3)
2. Refine spatially (n = 200, dt = 1e-3)
3. Refine temporally (n = 200, dt = 5e-4)
4. Check |V_refined - V_coarse| < tolerance

Typical behavior:
- ATM options: 1e-3 price error at n = 141, dt = 1e-3
- Deep ITM/OTM: require finer grids (steep exercise boundaries)
- Greeks: need 2-3x finer grids than prices (higher-order quantities are more sensitive)

### Interpolation Error (Price Tables)

For the B-spline price tables, the additional interpolation error is O(h^4) per dimension. The separable fitting preserves this order. In practice, the dominant error source is the volatility dimension (highest curvature), which is why it receives 1.5x weight in the grid budget.

Adaptive refinement (section 7) provides a verified error bound by testing against fresh PDE solves at random parameter combinations.

---

## References

1. **TR-BDF2**: Bank et al. (1985), "Transient Simulation of Silicon Devices and Circuits"
2. **Rannacher Startup**: Rannacher (1984), "Finite Element Solution of Diffusion Problems with Irregular Data"
3. **Projected Thomas / LCP**: Brennan & Schwartz (1977), "The Valuation of American Put Options"
4. **American Options**: Wilmott, "Derivatives", Chapter 11
5. **Finite Differences**: LeVeque, "Finite Difference Methods for ODEs and PDEs"
6. **B-Splines**: de Boor, "A Practical Guide to Splines"
7. **Obstacle Problems**: Kinderlehrer & Stampacchia, "An Introduction to Variational Inequalities"
8. **Root Finding**: Press et al., "Numerical Recipes", Chapter 9

**For implementation details, see [ARCHITECTURE.md](ARCHITECTURE.md)**
**For usage examples, see [API_GUIDE.md](API_GUIDE.md)**
