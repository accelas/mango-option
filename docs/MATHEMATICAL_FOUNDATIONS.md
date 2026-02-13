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
5. [Discrete Dividends](#5-discrete-dividends)
6. [Grid Generation](#6-grid-generation)

**Part II — Price Tables & Interpolation:** Pre-computing prices across parameter space for fast queries.

7. [B-Spline Interpolation](#7-b-spline-interpolation)
8. [EEP Decomposition](#8-eep-decomposition)
9. [Segmented Price Surfaces](#9-segmented-price-surfaces)
10. [Price Table Grid Estimation](#10-price-table-grid-estimation)
11. [Implied Volatility](#11-implied-volatility)
12. [Interpolated Greeks via Chain Rule](#12-interpolated-greeks-via-chain-rule)

**Part III — Analysis**

13. [Convergence Analysis](#13-convergence-analysis)

---

# Part I — PDE Pricing

The goal: given an American option's parameters (spot, strike, volatility, rate, maturity), compute its fair value and Greeks. We do this by solving a PDE on a finite grid, stepping backward in time from the known terminal payoff.

The sections below follow the computation pipeline: formulate the PDE, discretize space, step through time, enforce the American early-exercise constraint, and finally choose a good grid.

---

## 1. Black-Scholes PDE

### Backward Time Formulation

An American option's value $V(S, t)$ satisfies the Black-Scholes PDE in the **continuation region** (where holding is optimal). In the **exercise region** (where immediate exercise is optimal), $V$ equals the intrinsic value. Together, these form a variational inequality — the American option problem:

$$\frac{\partial V}{\partial \tau} \geq \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r - d)S\frac{\partial V}{\partial S} - rV, \qquad V \geq \psi, \qquad \left(\frac{\partial V}{\partial \tau} - \mathcal{L}V\right)(V - \psi) = 0$$

where $\psi$ is the intrinsic (payoff) value and $\mathcal{L}$ is the Black-Scholes operator. The complementarity condition (third equation) says: at each point, either the PDE holds as equality (continuation) or the option is at intrinsic (exercise), but not both. The boundary between these regions — the **early exercise boundary** — is a free boundary that emerges from the solution (section 4).

For the numerical method, we work in backward time $\tau = T - t$ (time remaining to maturity), which turns the problem into a forward-in-time evolution. In the continuation region, $V$ satisfies:

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r - d)S\frac{\partial V}{\partial S} - rV$$

with initial condition $V(S, 0) = \psi(S)$ and the constraint $V \geq \psi$ enforced at every time step (section 4).

Here $S$ is the spot price, $\sigma$ the volatility, $r$ the risk-free rate, and $d$ the continuous dividend yield. The three terms on the right have intuitive meanings: diffusion (volatility spreads the distribution), drift (the risk-neutral growth rate), and discounting (a dollar tomorrow is worth less today).

### Log-Price Transformation

Working directly in $S$ is inconvenient — the coefficients depend on $S$, and the domain is $[0, \infty)$. Substituting $x = \ln(S/K)$ fixes both problems:

$$\frac{\partial V}{\partial \tau} = \frac{\sigma^2}{2}\frac{\partial^2 V}{\partial x^2} + \left(r - d - \frac{\sigma^2}{2}\right)\frac{\partial V}{\partial x} - rV$$

Now all coefficients are constants, $x = 0$ corresponds to at-the-money ($S = K$), and the domain is symmetric around the strike. This is the form we actually discretize and solve.

### Boundary Conditions

At the edges of our finite computational domain:

**Left boundary** ($x \to -\infty$, $S \to 0$):
- Call: $V = 0$ (worthless far out-of-the-money)
- Put: $V = Ke^{-r\tau}$ (deep in-the-money, exercise is certain)

**Right boundary** ($x \to +\infty$, $S \to \infty$):
- Call: $V \sim S$ (deep in-the-money)
- Put: $V = 0$ (worthless far out-of-the-money)

**Terminal condition** ($\tau = 0$, i.e., at maturity):

$$\text{Call: } V(x, 0) = K\max(e^x - 1,\; 0)$$

$$\text{Put: } V(x, 0) = K\max(1 - e^x,\; 0)$$

This terminal condition is the starting point for our backward-in-time solve.

---

## 2. Spatial Discretization

With the PDE in log-price coordinates, we need to approximate the spatial derivatives on a discrete grid.

### Centered Finite Differences

On a uniform grid with spacing $\Delta x$, the standard second-order approximations are:

**Second derivative:**

$$\left.\frac{\partial^2 u}{\partial x^2}\right|_i \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2} \qquad \text{error: } O(\Delta x^2)$$

**First derivative:**

$$\left.\frac{\partial u}{\partial x}\right|_i \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x} \qquad \text{error: } O(\Delta x^2)$$

Both are second-order accurate, meaning halving $\Delta x$ reduces the error by 4×. This is important for the grid estimation strategy later.

### Non-Uniform Grids

Sinh-spaced grids (section 6) have variable spacing. The finite difference weights must account for this:

$$\left.\frac{\partial^2 u}{\partial x^2}\right|_i = w_\text{left} u_{i-1} + w_\text{center} u_i + w_\text{right} u_{i+1}$$

where the weights depend on the local spacings $\Delta x_{i-1}$ and $\Delta x_i$. The truncation error remains $O(\Delta x^2)$ as long as the spacing varies smoothly — which sinh grids guarantee by construction.

### Assembling the Spatial Operator

Combining the second derivative, first derivative, and zeroth-order terms from the Black-Scholes PDE into a single operator $\mathcal{L}$:

$$\mathcal{L}(u)_i = \frac{\sigma^2}{2}\left.\frac{\partial^2 u}{\partial x^2}\right|_i + \left(r - d - \frac{\sigma^2}{2}\right)\left.\frac{\partial u}{\partial x}\right|_i - ru_i$$

This produces a tridiagonal matrix: each grid point couples only to its immediate neighbors. This structure is what makes the solve fast — $O(n)$ per linear solve instead of $O(n^3)$.

---

## 3. TR-BDF2 Time Stepping

We now have a spatial discretization $\mathcal{L}$. The PDE becomes a system of ODEs:

$$\frac{du}{d\tau} = \mathcal{L}(u)$$

We need to march this forward in $\tau$ from the terminal payoff. The choice of time-stepping scheme matters a lot for stability and accuracy.

### Why TR-BDF2?

Explicit methods (forward Euler, RK4) require tiny time steps for stability — the CFL condition forces $\Delta t \sim \Delta x^2$ for diffusion problems, which is impractical with fine grids. Implicit methods remove this restriction but typically sacrifice either accuracy or damping properties.

TR-BDF2 is a composite two-stage scheme that gives us everything:
- **L-stable**: spurious high-frequency modes decay exponentially
- **Second-order accurate**: $O(\Delta t^2)$ global error
- **Unconditionally stable**: $\Delta t$ is limited only by accuracy, not stability

### The Two-Stage Scheme

Each time step from $t_n$ to $t_{n+1}$ proceeds in two stages:

**Stage 1 — Trapezoidal rule** (advance to $t_n + \gamma\Delta t$):

$$u_\text{stage} = u_n + \gamma\Delta t\left[\tfrac{1}{2}\mathcal{L}(u_n) + \tfrac{1}{2}\mathcal{L}(u_\text{stage})\right]$$

**Stage 2 — BDF2** (advance to $t_{n+1}$):

$$u_{n+1} = \frac{1+2\gamma}{1+\gamma}u_\text{stage} - \frac{\gamma^2}{1+\gamma}u_n + \frac{\Delta t}{1+\gamma}\mathcal{L}(u_{n+1})$$

The parameter $\gamma = 2 - \sqrt{2} \approx 0.5858$ is chosen specifically to make the scheme L-stable.

### Newton Iteration for Implicit Stages

Both stages are implicit — $u_\text{stage}$ and $u_{n+1}$ appear on both sides of their equations. We solve each by Newton's method.

Rearranging Stage 1 as $F(u) = 0$:

$$F(u) = u - u_n - \gamma\Delta t\left[\tfrac{1}{2}\mathcal{L}(u_n) + \tfrac{1}{2}\mathcal{L}(u)\right]$$

Newton's iteration:

$$J\delta u = -F(u^k), \qquad u^{k+1} = u^k + \delta u$$

where $J = I - \frac{\gamma\Delta t}{2}\frac{\partial\mathcal{L}}{\partial u}$ is the Jacobian. Since $\mathcal{L}$ is tridiagonal, so is $J$ — each Newton step costs just one tridiagonal solve via the Thomas algorithm ($O(n)$).

Convergence is typically fast: 3–5 Newton iterations per implicit stage. The Jacobian is assembled analytically from the spatial operator coefficients, avoiding the cost of finite-difference Jacobian approximation.

### Rannacher Startup

There is one subtlety. The trapezoidal rule in Stage 1 has no numerical dissipation at the Nyquist frequency — it preserves high-frequency modes exactly. For smooth initial data, this is fine. But option payoffs have a kink at the strike (discontinuous first derivative), which excites all frequencies equally. The result: **oscillations in gamma** near the strike during the first few time steps.

Rannacher (1984) showed that replacing the first TR-BDF2 step with backward Euler eliminates this. Backward Euler is strongly dissipative — it damps high-frequency modes aggressively, smoothing the payoff kink before TR-BDF2 takes over.

The implementation uses two half-steps of backward Euler:

**Step 0 (Rannacher):**

$$u^{1/2} = u^0 + \tfrac{\Delta t}{2}\mathcal{L}(u^{1/2})$$

$$u^1 = u^{1/2} + \tfrac{\Delta t}{2}\mathcal{L}(u^1)$$

**Steps 1 to $N$ (standard TR-BDF2):**
- Stage 1: trapezoidal to $t_n + \gamma\Delta t$
- Stage 2: BDF2 to $t_{n+1}$

Two half-steps rather than one full step preserve better accuracy while retaining the damping properties.

Does this hurt the overall accuracy? Backward Euler is first-order, introducing $O(\Delta t)$ local error. But it applies to only one step out of $N$, so the global contribution is $O(\Delta t / N) = O(\Delta t^2)$ — the same order as TR-BDF2. The overall second-order convergence is preserved.

---

## 4. American Option Constraints

European options just solve the PDE and read off the answer. American options add a constraint: the holder can exercise at any time, so the option value must never fall below the immediate exercise (intrinsic) value.

### The Obstacle Condition

At every grid point and every time step:

$$V(x, \tau) \geq \psi(x)$$

where $\psi(x)$ is the intrinsic value:
- Call: $\psi(x) = \max(Ke^x - K,\; 0)$
- Put: $\psi(x) = \max(K - Ke^x,\; 0)$

This turns the PDE into a **variational inequality** — a free boundary problem where the exercise boundary is part of the solution, not an input.

### Projected Thomas Algorithm (Brennan-Schwartz)

The question is: how do we enforce the obstacle constraint while solving the tridiagonal systems from Newton iteration?

The naive approach — solve the unconstrained system, then project — breaks the tridiagonal coupling:

> **WRONG:**
> 1. Solve $Au = d$ ignoring the obstacle
> 2. Set $u_i = \max(u_i, \psi_i)$
>
> Result: $Au \neq d$ at projected nodes. The solution is inconsistent.

The Projected Thomas algorithm, due to Brennan & Schwartz (1977), does something more elegant. It enforces the constraint **during** backward substitution rather than after.

**Forward elimination** (identical to standard Thomas):

$$c'_0 = \frac{c_0}{b_0}, \qquad d'_0 = \frac{d_0}{b_0}$$

For $i = 1$ to $n-1$:

$$w = b_i - a_{i-1}c'_{i-1}, \qquad c'_i = \frac{c_i}{w}, \qquad d'_i = \frac{d_i - a_{i-1}d'_{i-1}}{w}$$

**Projected backward substitution** (the key difference):

$$u_{n-1} = \max(d'_{n-1},\; \psi_{n-1})$$

For $i = n-2$ down to $0$:

$$u_i = \max\!\big(d'_i - c'_iu_{i+1},\;\; \psi_i\big)$$

The $\max$ at each step is the projection. Because the matrix $A$ is an M-matrix (positive diagonal, non-positive off-diagonals — guaranteed by TR-BDF2's discretization), this projection is monotone and the algorithm converges in a **single pass**. Same $O(n)$ cost as standard Thomas, no iteration needed.

### Why This Works

The insight is about information flow. In backward substitution, $u_i$ depends on $u_{i+1}$. If we project $u_{i+1}$ upward (to the obstacle), this propagates correctly through the tridiagonal coupling — the constraint at one node affects its neighbors in a consistent way.

For M-matrices, the off-diagonal elements are non-positive, so increasing $u_{i+1}$ can only decrease the unconstrained value at $u_i$. This means the projection is monotone: once a node is clamped to the obstacle, nodes to its left see a larger $u_{i+1}$ and thus a smaller unconstrained value, making them more likely to also hit the obstacle. This is exactly the early-exercise region propagating inward from deep in-the-money.

### Deep ITM Locking

One practical detail: for nodes deep in-the-money where $\psi$ is close to the maximum intrinsic value, numerical diffusion can erroneously lift the solution above intrinsic. We prevent this by converting deep ITM nodes to Dirichlet constraints:

$$\text{if } \psi_i > 0.95 \cdot \psi_\max \text{ and } u_i \approx \psi_i\text{: lock } u_i = \psi_i$$

This ensures, for example, that a deep ITM put with intrinsic value 99.75 prices at 99.75 rather than being lifted to 115.97 by diffusion.

### The Early Exercise Boundary

The free boundary $S^*(\tau)$ separates two regions:

- **Continuation region**: $V > \psi$ (hold the option, it's worth more alive)
- **Exercise region**: $V = \psi$ (exercise immediately)

For puts, the exercise region is $S < S^*$ (deep ITM). For calls, it's $S > S^*$. The Projected Thomas algorithm finds this boundary implicitly — we never compute it directly; it emerges from the constraint enforcement.

---

## 5. Discrete Dividends

Sections 1–4 assume a continuous dividend yield $d$ that enters the PDE coefficients. Real equities pay discrete cash dividends at known dates. A dividend $D$ paid at calendar time $t_d$ causes the spot to drop: $S(t_d^+) = S(t_d^-) - D$. The option value must satisfy the jump condition across the dividend:

$$V(S, t_d^-) = V(S - D, t_d^+)$$

This cannot be handled by adjusting the PDE coefficients — it requires modifying the solution at discrete points in time.

### Temporal Events

The solver handles discrete dividends as **temporal events**: callbacks that modify the solution vector at mandatory time steps. The time grid is constructed so that each dividend date $t_d$ falls exactly on a time step boundary (converted to backward time $\tau_d = T - t_d$).

At each dividend event, the solver:

1. Completes the TR-BDF2 step up to $\tau_d$
2. Applies the jump condition (see below)
3. Re-applies boundary conditions and obstacle constraints
4. Continues the PDE solve to the next event or $\tau = T$

### The Jump Condition in Log-Moneyness

In log-moneyness coordinates $x = \ln(S/K)$, the spot drop $S \to S - D$ maps to a nonlinear shift. Define the normalized dividend $\delta = D/K$. For each grid point $x_i$:

$$x_i' = \ln\!\big(e^{x_i} - \delta\big)$$

The jump condition becomes:

$$u_i \leftarrow \hat{u}(x_i')$$

where $\hat{u}$ is a cubic spline interpolant of the current solution. The spline is rebuilt from the solution vector at each dividend event (reusing the same grid, so the rebuild is $O(n)$ with zero allocation).

**Boundary cases.** When $e^{x_i} \leq \delta$ (the spot drops to zero or below after the dividend), the spline cannot be evaluated. The solver uses a fallback:
- Put: $u_i = 1.0$ (normalized deep ITM value — exercise is certain)
- Call: $u_i = 0.0$ (worthless)

When $x_i'$ falls below the grid domain $x_\min$, the same fallback applies. When $x_i'$ exceeds $x_\max$, it is clamped to $x_\max$.

### Grid Adjustments

Discrete dividends require two modifications to the grid estimation (section 6).

**Domain widening.** The shift $x \to \ln(e^x - \delta)$ moves points leftward. Without adjustment, points near $x_\min$ would shift outside the domain. The estimator extends the left boundary:

$$x_\min' = \begin{cases} \ln\!\big(e^{x_\min} - \delta_\max\big) & \text{if } e^{x_\min} > \delta_\max \\ x_\min - 1 & \text{otherwise (conservative extension)} \end{cases}$$

where $\delta_\max = \max_k D_k / K$ is the largest normalized dividend.

**Mandatory time steps.** Each dividend date $\tau_d$ is inserted as a mandatory point in the time grid. The interval between consecutive dividends (or between a dividend and the domain boundary) is subdivided uniformly with $\Delta t \leq \Delta t_\text{target}$, ensuring temporal accuracy is maintained within each segment.

### Interaction with Continuous Yield

Continuous and discrete dividends combine naturally. The continuous yield $d$ enters the Black-Scholes PDE coefficients (drift term $r - d - \sigma^2/2$) and is active throughout the solve. Discrete dividends operate orthogonally through temporal events. The two mechanisms do not interfere.

---

## 6. Grid Generation

The grid determines both accuracy and cost. Too coarse and the solution is inaccurate; too fine and the solve is slow. The library provides three grid types and an automatic estimation strategy.

### Uniform Grids

Equally spaced points in log-moneyness:

$$x_i = x_\min + i\Delta x, \qquad \Delta x = \frac{x_\max - x_\min}{n - 1}$$

Simple and useful for testing, but wasteful for option pricing: most of the interesting behavior (gamma peak, exercise boundary) is concentrated near the strike, while far OTM/ITM regions are nearly linear.

### Sinh-Spaced Grids

The workhorse grid for option pricing. A hyperbolic sine transformation concentrates points near a center while maintaining smooth spacing:

$$\xi_i = -1 + \frac{2i}{n-1} \qquad \text{(uniform in } [-1, 1]\text{)}$$

$$x_i = x_c + \frac{\Delta x}{\alpha}\sinh(\alpha\xi_i) \qquad \text{(sinh-spaced in } x\text{)}$$

where $x_c$ is the center (typically $0$ = ATM), $\Delta x$ is the half-width, and $\alpha$ controls the concentration. With $\alpha = 2$:

- Spacing near center: $\Delta x_\min \sim (\Delta x / n)e^{-\alpha}$ — about 7× finer than uniform
- Spacing at boundaries: $\Delta x_\max \sim (\Delta x / n)e^{\alpha}$ — about 7× coarser than uniform

This puts resolution where it matters (near the strike) and saves points where it doesn't (far tails). The spacing varies smoothly and monotonically, so the non-uniform finite difference weights remain well-conditioned.

### Multi-Sinh Grids

When pricing across multiple strikes (e.g., for price tables), a single concentration center is insufficient. Multi-sinh grids superpose several sinh transformations:

$$x_i = \sum_k w_k\text{sinh}_k(\xi_i) \qquad \text{(normalized weights)}$$

Each cluster specifies a center, $\alpha$, and weight. Clusters closer than $0.3/\bar{\alpha}$ are automatically merged to avoid wasted resolution. Use this when strikes differ by more than ~20%.

### Automatic PDE Grid Estimation

`estimate_pde_grid()` builds a sinh grid tailored to a specific option. The logic:

**Domain bounds.** Extend $\pm n_\sigma$ standard deviations from the current log-moneyness:

$$x_\min = \ln(S/K) - n_\sigma\sigma\sqrt{T}, \qquad x_\max = \ln(S/K) + n_\sigma\sigma\sqrt{T}$$

With $n_\sigma = 5$ (default), this covers >99.99997% of the terminal distribution. The width scales with $\sigma\sqrt{T}$ — higher volatility or longer maturity automatically produces a wider domain.

**Spatial resolution.** The centered difference truncation error is $O(\Delta x^2)$. To achieve a target error proportional to $\varepsilon$:

$$\Delta x_\text{target} = \sigma\sqrt{\varepsilon}, \qquad N_x = \left\lceil\frac{x_\max - x_\min}{\Delta x_\text{target}}\right\rceil$$

Scaling $\Delta x$ with $\sigma$ keeps $N_x$ stable across volatilities: higher $\sigma$ widens the domain but proportionally coarsens the target spacing. The $\sqrt{\varepsilon}$ relationship means 10× better accuracy costs ~3.2× more points. $N_x$ is clamped to $[100, 1200]$.

**Temporal resolution.** TR-BDF2 is unconditionally stable, so there is no CFL constraint. But second-order accuracy requires $\Delta t \sim O(\Delta x_\min)$. The time step couples to the finest spatial spacing:

$$\Delta t = c_t\Delta x_\min, \qquad \text{where } \Delta x_\min \sim \Delta x_\text{avg}e^{-\alpha}$$

$$N_t = \left\lceil T / \Delta t \right\rceil$$

With $c_t = 0.75$ and $\alpha = 2.0$, this ensures temporal error doesn't dominate spatial error in the clustered region where gradients are steepest.

**Default parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| $n_\sigma$ | 5.0 | Domain half-width in $\sigma\sqrt{T}$ units |
| $\alpha$ | 2.0 | Sinh clustering strength (~7× center-to-edge ratio) |
| $\varepsilon$ | $10^{-2}$ | Spatial truncation error target |
| $c_t$ | 0.75 | Time-space coupling factor |
| min_spatial_points | 100 | Lower bound on $N_x$ |
| max_spatial_points | 1200 | Upper bound on $N_x$ |
| max_time_steps | 5000 | Upper bound on $N_t$ |

For a short-dated SPY option ($\sigma \approx 0.15$, $T \approx 0.09$), the defaults produce a $101 \times 150$ grid.

---

# Part II — Price Tables & Interpolation

Part I gave us a PDE solver that prices one option in ~1–2ms. For implied volatility — which requires pricing the option repeatedly at different volatilities until the price matches the market — this is too slow. A single IV solve takes ~15ms (5–8 Brent iterations × 2ms each), and a trading desk needs thousands of IVs per second.

The solution: pre-compute prices across a 4D parameter grid (moneyness, maturity, volatility, rate), fit a B-spline surface, and evaluate the surface at ~500ns per query. This section covers the interpolation machinery, grid estimation, and IV extraction.

---

## 7. B-Spline Interpolation

### Why B-Splines?

We need a smooth interpolant over a 4D grid of pre-computed prices. Requirements:
- $C^2$ continuity (smooth Greeks via differentiation)
- Local support (changing one region doesn't affect distant regions)
- Fast evaluation (~hundreds of nanoseconds)

Cubic B-splines satisfy all three. They provide $C^2$ continuity, each basis function is non-zero on only 4 adjacent intervals, and evaluation requires only local data.

### Cubic B-Spline Basis

The B-spline basis functions are defined recursively (Cox-de Boor):

$$N_{i,0}(x) = \begin{cases} 1 & \text{if } x \in [t_i, t_{i+1}) \\ 0 & \text{otherwise} \end{cases}$$

$$N_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i}N_{i,k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}}N_{i+1,k-1}(x)$$

Key properties:
- **Compact support**: each cubic basis function spans 4 intervals
- **Partition of unity**: $\sum_i N_i(x) = 1$ everywhere
- **$C^2$ continuity**: two continuous derivatives (sufficient for delta, gamma, vega)
- **Local control**: modifying one coefficient affects only 4 intervals

### Clamped Knot Vectors

For $n$ data points $x_0, \ldots, x_{n-1}$, the clamped knot vector repeats the endpoints with multiplicity $p+1 = 4$:

$$\mathbf{t} = [\underbrace{x_0, x_0, x_0, x_0}_{p+1},\; t_1, \ldots, t_m,\; \underbrace{x_{n-1}, x_{n-1}, x_{n-1}, x_{n-1}}_{p+1}]$$

This forces the B-spline to interpolate exactly at the endpoints — essential for price tables where boundary values must be exact.

Interior knots are placed proportionally between data sites, with epsilon clamping to avoid coinciding with data sites (which would make the collocation matrix singular):

$$t_j = \text{clamp}(t_j^\text{proportional},\; x_\text{low} + \epsilon,\; x_{\text{low}+1} - \epsilon)$$

This satisfies the Schoenberg-Whitney condition, guaranteeing a non-singular collocation system.

### 4D Separable Fitting

The price table uses tensor-product B-splines:

$$P(m, \tau, \sigma, r) = \sum_{i,j,k,l} c_{ijkl}N_i(m)N_j(\tau)N_k(\sigma)N_l(r)$$

Fitting all 4 dimensions simultaneously would require solving a dense $(n_m \cdot n_\tau \cdot n_\sigma \cdot n_r)^2$ system — completely impractical for a $50 \times 30 \times 20 \times 10$ grid (300K unknowns).

The separable algorithm exploits the tensor-product structure with 4 sequential 1D fits:

1. Fix $(\tau, \sigma, r)$, fit moneyness for each slice → $c_{*,j,k,l}$
2. Fix $(\sigma, r)$, fit maturity on the coefficients from step 1 → $c_{*,*,k,l}$
3. Fix $r$, fit volatility → $c_{*,*,*,l}$
4. Fit rate → $c_{*,*,*,*}$

Each 1D fit solves a banded collocation system. Cubic splines produce a 4-diagonal matrix, solved in $O(n)$ via banded LU. The total cost is $O(n_m \cdot n_\tau \cdot n_\sigma \cdot n_r)$ — linear in the grid size.

### Greeks via Differentiation

A major benefit of B-splines: derivatives are analytic. To compute delta ($\partial P / \partial m$), differentiate the basis functions in $m$ while leaving the others untouched:

$$\Delta = \frac{\partial P}{\partial m} = \sum c_{ijkl}N_i'(m)N_j(\tau)N_k(\sigma)N_l(r)$$

$$\nu = \frac{\partial P}{\partial \sigma} = \sum c_{ijkl}N_i(m)N_j(\tau)N_k'(\sigma)N_l(r)$$

$$\Gamma = \frac{\partial^2 P}{\partial m^2} = \sum c_{ijkl}N_i''(m)N_j(\tau)N_k(\sigma)N_l(r)$$

Each derivative costs the same as a price evaluation (~500ns) because we evaluate one differentiated 1D basis and three undifferentiated ones.

---

## 8. EEP Decomposition

The price table does not store raw American option prices. Instead, it stores the **Early Exercise Premium** (EEP): the difference between the American and European prices.

### Decomposition

Any American option price can be written as:

$$P_\text{Am}(S, K, \tau, \sigma, r) = \text{EEP}(m, \tau, \sigma, r) \cdot \frac{K}{K_\text{ref}} + P_\text{Eu}(S, K, \tau, \sigma, r, q)$$

where $m = S/K$ is moneyness, $K_\text{ref}$ is a fixed reference strike, and $q$ is the continuous dividend yield.

This decomposition has three advantages:

1. **Smoothness.** The EEP varies slowly across parameter space — it lacks the sharp kink at $S = K$ that the full price inherits from the payoff. Smoother functions interpolate better.

2. **Strike homogeneity.** The EEP depends on strike only through moneyness $m = S/K$, so one 4D surface covers all strikes. The factor $K/K_\text{ref}$ rescales back to absolute dollar terms.

3. **European exactness.** The European component $P_\text{Eu}$ is computed analytically (Black-Scholes), so interpolation error affects only the EEP — typically a small fraction of the total price.

### Log-Moneyness Transform

Internally, the B-spline interpolates in log-moneyness $x = \ln(m)$ rather than $m$. This provides symmetric resolution around ATM (where $x = 0$) and reduces interpolation error by 20–40% in the tails.

All user-facing APIs accept moneyness $m$; the transform is applied internally.

### Greeks via Chain Rule

Because the price is reconstructed from an interpolated EEP and an analytic European component, each Greek requires a chain rule through the decomposition. Let $E(m, \tau, \sigma, r)$ denote the EEP B-spline and $g(x) = E(e^x, \tau, \sigma, r)$ the EEP in log-moneyness.

**Delta** ($\partial P / \partial S$):

$$\Delta = \frac{1}{K_\text{ref}} \cdot \frac{\partial E}{\partial m} + \Delta_\text{Eu}$$

where $\partial E / \partial m = g'(x) / m$ (chain rule from log-moneyness).

**Gamma** ($\partial^2 P / \partial S^2$):

$$\Gamma = \frac{1}{K_\text{ref} \cdot K} \cdot \frac{\partial^2 E}{\partial m^2} + \Gamma_\text{Eu}$$

where $\partial^2 E / \partial m^2 = (g''(x) - g'(x)) / m^2$. Both $g'$ and $g''$ are analytic B-spline derivatives (first and second order), so gamma avoids finite differencing entirely.

**Theta** ($\partial P / \partial t$, calendar time):

$$\Theta = -\frac{K}{K_\text{ref}} \cdot \frac{\partial E}{\partial \tau} + \Theta_\text{Eu}$$

The sign flip converts from time-to-expiry $\tau$ to calendar time $t$. Both the EEP and European theta use the same convention ($dV/dt$, negative for time decay).

**Vega** ($\partial P / \partial \sigma$):

$$\nu = \frac{K}{K_\text{ref}} \cdot \frac{\partial E}{\partial \sigma} + \nu_\text{Eu}$$

### Measured Accuracy

Interpolated Greeks vs. PDE solver reference across 20 (strike × maturity) combinations ($S = 100$, $\sigma = 0.20$, $r = 0.05$, $q = 0.02$):

| Greek | Max Abs Error | Max Rel Error | Notes |
|-------|---------------|---------------|-------|
| Price | \$0.086 | 1.9% | |
| Delta | 0.0087 | 2.8% | |
| Gamma | 0.0024 | 7.3% | Worst at short $\tau$; < 1.3% for $\tau \geq 0.5$ |
| Theta | \$0.15 | 3.3% | |

When accuracy is critical, use the PDE solver directly (`AmericanOptionSolver`).

---

## 9. Segmented Price Surfaces

Discrete dividends (section 5) break the smoothness that B-spline interpolation requires. A single surface cannot fit the price discontinuity at a dividend date. The segmented surface builder solves this by partitioning the maturity axis at dividend dates and fitting a separate B-spline surface per segment.

### Maturity Partitioning

For $N$ dividends at calendar times $t_1 < t_2 < \cdots < t_N$, the backward-time boundaries are:

$$0 = \tau_0 < \tau_N < \tau_{N-1} < \cdots < \tau_1 < T$$

where $\tau_k = T - t_k$. Each segment covers $[\tau_{k+1}, \tau_k]$ and has its own B-spline surface.

**Segment 0** ($\tau \in [0, \tau_N]$, nearest to expiry): built in EEP mode (section 8) with the payoff as initial condition.

**Segment $k > 0$** ($\tau \in [\tau_{N-k+1}, \tau_{N-k}]$): built in raw-price mode. Its initial condition is the previous segment's surface evaluated at the post-dividend spot:

$$V_k(m, \sigma, r)\big|_{\tau = \tau_{N-k+1}} = V_{k-1}\!\left(m + \frac{D_{N-k+1}}{K_\text{ref}},\; \sigma,\; r\right)\bigg|_{\tau = \tau_{N-k+1}}$$

The moneyness shift $m \to m + D/K_\text{ref}$ accounts for the spot being higher before the dividend — the same jump condition as in the PDE solver (section 5), expressed in moneyness coordinates.

### Query-Time Evaluation

At query time, the surface finds the segment covering the requested $\tau$ and evaluates it directly. For the EEP segment, a spot adjustment subtracts dividends that will be paid between the query time and the segment boundary:

$$S_\text{adj} = S - \sum_{\{k : t_\text{query} < t_k \leq t_\text{boundary}\}} D_k$$

### Multiple Reference Strikes

Cash dividends break the scale invariance that American options normally have in strike (the EEP decomposition assumes $P \propto K$, which fails when $D/K$ varies with $K$). The builder constructs surfaces at several reference strikes and interpolates across them with Catmull-Rom splines in $\ln(K_\text{ref})$, producing a `SegmentedMultiKRefSurface`.

### Adaptive Grid Refinement for Segmented Surfaces

The adaptive grid builder (section 10) extends to segmented surfaces via a probe-and-max strategy. Rather than building the full multi-K_ref surface at each refinement iteration, the builder:

1. **Selects 2–3 probe K_ref values** from the full list: the lowest, highest, and the one closest to ATM (deduplicated if ATM coincides with an endpoint).

2. **Runs independent refinement loops** on each probe, building single-K_ref `BSplineSegmentedSurface` instances. Each probe validates at strike = K_ref (the only strike that single-K_ref raw segments can price exactly).

3. **Takes the per-axis maximum** grid sizes across probes — the worst-case K_ref determines each axis.

4. **Builds the full `SegmentedMultiKRefSurface`** once, using uniform grids at the maximum sizes with `skip_moneyness_expansion = true` (the domain was pre-expanded in step 1).

5. **Final validation** at arbitrary strikes against fresh PDE reference prices. If the error exceeds the target, all grids are bumped by one refinement step and the surface is rebuilt (one retry).

The moneyness domain is pre-expanded before probing using the worst-case (smallest) K_ref: $m_\text{min}' = \max(m_\text{min} - \sum D_k / K_\text{ref,min},\; 0.01)$. This ensures all K_refs share the same expanded domain.

The tau axis is refined via the `tau_points_per_segment` scalar (minimum 4 for B-spline), which the refinement loop increments when tau is the worst dimension.

---

## 10. Price Table Grid Estimation

The 4D grid density directly controls IV accuracy. Too coarse and the B-spline interpolation introduces significant error; too fine and pre-computation becomes prohibitively expensive. The challenge is choosing per-axis point counts that hit a target accuracy without wasting PDE solves.

The library offers two grid specification modes:

- **Manual grid.** The user supplies explicit grid vectors for moneyness, volatility, and rate (each requiring $\geq 4$ points for the cubic B-spline). Predefined accuracy profiles translate a qualitative accuracy level into concrete grid sizes derived from the curvature-based formula below.
- **Adaptive grid.** The user specifies a target IV error $\varepsilon_\text{target}$ and domain bounds. The builder automatically determines grid density via iterative refinement, validated against fresh PDE solves. This works for both the standard path (continuous dividends) and the segmented path (discrete dividends; see section 9 for the probe-and-max strategy). This removes the need for manual tuning at the cost of additional PDE solves during construction.

Both modes share the same maturity grid (supplied via the path configuration) and produce the same `BSplineND<double, 4>` — the difference is only in how the per-axis point counts are chosen.

### Curvature-Based Budget Allocation

Cubic B-spline interpolation error on a uniform grid of spacing $h$ is bounded by

$$\|f - s\|_\infty \leq C \cdot h^4 \cdot \|f^{(4)}\|_\infty$$

where $f^{(4)}$ is the fourth derivative (curvature) of the underlying function and $C$ is a constant depending on the spline order. Inverting for grid spacing: to achieve error $\varepsilon$, we need $h \sim \varepsilon^{1/4}$, so the number of points scales as $n \sim \varepsilon^{-1/4}$.

In a multi-dimensional price table, the four axes have very different curvature profiles. The American option price surface is most non-linear in volatility (vega curvature), moderately non-linear in moneyness (gamma peak near ATM, dampened by log-transform) and maturity ($\sqrt{\tau}$ behavior), and nearly linear in the risk-free rate (discounting). We assign curvature weights $w_d$ to reflect this:

| Dimension | Weight $w_d$ | Rationale |
|-----------|:---:|-----------|
| Moneyness ($m$) | 1.0 | Moderate curvature; log-transform absorbs ATM peak |
| Maturity ($\tau$) | 1.0 | Baseline; $\sqrt{\tau}$ dependence |
| Volatility ($\sigma$) | 2.5 | Highest curvature — vega non-linearity, vanna, volga |
| Rate ($r$) | 0.6 | Nearly linear discounting effect |

The per-dimension point count is:

$$n_\text{base} = \left(\frac{s}{\varepsilon_\text{target}}\right)^{1/4}, \qquad n_d = \text{clamp}\!\left(\lceil n_\text{base} \cdot w_d \rceil,\; 4,\; n_\text{max}\right)$$

where $s = 2.0$ is a scale factor calibrated from benchmark data. The calibration reference: a grid of $13 \times 18 \times 8$ (moneyness $\times$ volatility $\times$ rate) achieves approximately 4.3 bps average IV error. With weights $[1.0, 1.0, 2.5, 0.6]$ and $n_\text{base} = 12$, the formula reproduces $n_\sigma = \lceil 12 \times 2.5 \rceil = 30$ for the volatility axis. The fourth-root relationship means accuracy improves slowly with grid size (halving the error requires $16\times$ the points), so the weights matter more than raw point count.

**Grid spacing strategies.** The distribution of points within each axis exploits the coordinate transform used internally:

- **Moneyness**: log-uniform spacing. Points are uniform in $\log(m)$, matching the log-moneyness coordinate used by the B-spline. This concentrates points near ATM where gamma peaks.
- **Maturity**: $\sqrt{\tau}$-uniform spacing. Points are uniform in $\sqrt{\tau}$, concentrating near short expiries where theta is steepest and the early exercise boundary moves fastest.
- **Volatility**: uniform spacing. The price surface's dependence on $\sigma$ has the highest fourth derivative; regular spacing is the safest choice for the B-spline error bound.
- **Rate**: uniform spacing. Nearly linear dependence means uniform spacing wastes the fewest points.

### Predefined Accuracy Profiles

For manual grid construction, four profiles translate a qualitative accuracy level into concrete `PriceTableGridAccuracyParams`. Each profile sets $\varepsilon_\text{target}$ and $n_\text{max}$; the curvature-based formula above then determines the per-axis point counts.

| Profile | $\varepsilon_\text{target}$ | $n_\text{max}$ | Typical grid $(m \times \tau \times \sigma \times r)$ | Estimated PDE solves |
|---------|:---:|:---:|:---:|:---:|
| Low | $5 \times 10^{-4}$ (50 bps) | 80 | $8 \times 8 \times 20 \times 5$ | ~100 |
| Medium | $1 \times 10^{-4}$ (10 bps) | 120 | $12 \times 12 \times 30 \times 8$ | ~240 |
| High | $2 \times 10^{-5}$ (2 bps) | 160 | $18 \times 18 \times 45 \times 11$ | ~495 |
| Ultra | $7 \times 10^{-6}$ (0.7 bps) | 200 | $24 \times 24 \times 58 \times 14$ | ~812 |

All profiles use the same curvature weights $[1.0,\, 1.0,\, 2.5,\, 0.6]$ and minimum of 4 points per axis (the cubic B-spline minimum). The "typical grid" column shows approximate sizes from the formula; actual sizes depend on domain bounds.

The computational cost scales as $n_\sigma \times n_r$ PDE solves (one solve per volatility-rate pair, shared across all moneyness and maturity points via snapshot extraction). Moving from Medium to High roughly doubles the PDE solve count.

### Adaptive Grid Refinement

The adaptive builder automates grid selection by iteratively refining an initial seed grid until a target IV error is met, using fresh PDE solves for validation at each iteration. The algorithm has five stages per iteration.

#### Stage 1: Seed Grid

The initial grid is a small uniform grid: 5 points each for moneyness, maturity, and volatility, and 4 points for rate (the B-spline minimum). Domain bounds are extracted from the input option chain's strikes, maturities, implied volatilities, and rates. Each dimension's bounds are expanded by a minimum spread to ensure the linspace produces distinct points:

| Dimension | Minimum spread | Rationale |
|-----------|:---:|-----------|
| Moneyness | 0.10 | $\pm 5\%$ around ATM |
| Maturity | 0.50 years | $\pm 0.25$ years |
| Volatility | 0.10 | $\pm 5\%$ vol |
| Rate | 0.04 | $\pm 2\%$ rate |

Lower bounds for moneyness, maturity, and volatility are clamped to $10^{-6}$ to avoid negative or zero values that the B-spline fitting rejects.

#### Stage 2: Build Price Table

At each iteration, a full `BSplineND<double, 4>` is built from the current grid vectors. The builder computes batch PDE solves for all $(\sigma, r)$ parameter pairs, extracts price snapshots at each maturity, fits the 4D B-spline, and produces the surface.

A slice cache keyed by $(\sigma, r)$ pairs (rounded to 6 decimal places) avoids re-solving PDE slices that were computed in previous iterations. When a grid refinement changes only one dimension (e.g., adding moneyness points), all previously computed $(\sigma, r)$ slices remain valid. The cache is invalidated only if the maturity grid changes, since maturity affects the PDE time-stepping.

#### Stage 3: Validation via Latin Hypercube Sampling

The validation step compares the B-spline surface against fresh PDE reference solves at $N$ random points (default $N = 64$). The key design choice: validation solves are independent of the table's own PDE solves. The surface cannot validate itself — fresh solves prevent the refinement from chasing interpolation artifacts.

**Latin Hypercube Sampling (LHS)** generates the validation points. In standard Monte Carlo, random points can cluster in some regions and leave others unsampled. LHS avoids this by stratifying each dimension independently: for $N$ samples, dimension $d$ is divided into $N$ equal strata, and each stratum contains exactly one sample. The position within each stratum is chosen uniformly at random, and strata are shuffled independently per dimension.

Formally, for dimension $d$ with permutation $\pi_d$ of $\{0, \ldots, N-1\}$:

$$x_i^{(d)} = \frac{\pi_d(i) + U_i^{(d)}}{N}, \qquad U_i^{(d)} \sim \text{Uniform}(0, 1)$$

This guarantees that projecting all $N$ points onto any single axis produces exactly one point per stratum — no clustering, no gaps.

After the first iteration, the sampling becomes **two-phase**: half the budget ($N/2$ points) uses standard LHS for broad coverage, and the other half uses **targeted sampling** focused on bins that had high errors in the previous iteration. Targeted samples are drawn from the problematic bins' sub-intervals, concentrating validation where it matters most.

#### Stage 4: Error Metric

For each validation point, the error metric converts the price difference between the B-spline surface and the fresh PDE solve into an IV-equivalent error:

$$\text{error} = \begin{cases} \displaystyle\frac{|P_\text{interp} - P_\text{ref}|}{\nu} & \text{if } \nu \geq \nu_\text{floor} \\[10pt] \displaystyle\frac{|P_\text{interp} - P_\text{ref}|}{\nu_\text{floor}} & \text{if } \nu < \nu_\text{floor} \text{ and } |P_\text{interp} - P_\text{ref}| > \varepsilon_\text{target} \cdot \nu_\text{floor} \\[10pt] \text{(skip)} & \text{otherwise} \end{cases}$$

where $\nu$ is the European Black-Scholes vega at the sample point and $\nu_\text{floor} = 10^{-4}$.

The first case is the standard first-order Taylor approximation $\Delta\sigma \approx \Delta P / \nu$. The second case handles deep ITM/OTM options where vega is near zero and IV is numerically ill-defined; here the floor prevents division by a tiny number. The third case skips samples where the price error is small enough that even with the floor, the point would pass — these are uninteresting for refinement decisions.

Using European vega (instead of the true American vega) is a deliberate approximation. For ATM and OTM options the two are nearly identical. For deep ITM American puts, European vega overestimates vega, which underestimates the IV error — a conservative choice that may slightly under-refine those regions. Computing true American vega would require two additional PDE solves per validation point, an unacceptable cost.

**Convergence check.** If $\max(\text{error}) \leq \varepsilon_\text{target}$ across all validation points, the grid is accurate enough and the algorithm terminates. Otherwise, the error data is passed to the error attribution stage.

#### Stage 5: Error Attribution and Targeted Refinement

The goal is to answer: *which dimension should receive more grid points?* Adding points uniformly across all dimensions is wasteful — typically one dimension dominates the error.

**Bin-based error tracking.** Each dimension is divided into $B = 5$ equal-width bins over its normalized $[0, 1]$ range. For every validation point where the error exceeds $\varepsilon_\text{target}$, the algorithm increments the bin count for each dimension at the point's position and accumulates the error mass:

$$\text{bin\_counts}[d][\lfloor p_d \cdot B \rfloor] \mathrel{+}= 1, \qquad \text{error\_mass}[d] \mathrel{+}= e$$

where $p_d \in [0, 1]$ is the normalized position in dimension $d$ and $e$ is the IV error.

**Dimension selection.** The worst dimension is the one with the highest score:

$$\text{score}_d = \underbrace{\frac{\max_b\, \text{bin\_counts}[d][b]}{\sum_b \text{bin\_counts}[d][b]}}_{\text{concentration}} \times \underbrace{\text{error\_mass}[d]}_{\text{magnitude}}$$

The concentration ratio measures how localized the errors are in that dimension. If errors are spread uniformly across all 5 bins, concentration is $1/5 = 0.2$; if all errors fall in one bin, concentration is $1.0$. Multiplying by error mass ensures the algorithm prefers dimensions with both localized *and* large errors — a dimension with one tiny error in one bin scores lower than a dimension with many large errors concentrated in two bins.

**Targeted midpoint insertion.** Once the worst dimension is identified, the algorithm inserts midpoints between existing grid points, but only in regions corresponding to problematic bins (bins with $\geq 2$ high-error samples). This avoids wasting points in regions that are already well-resolved. The total number of new points per iteration is bounded by:

$$n_\text{new} \leq \lfloor n_\text{current} \times (f - 1) \rfloor$$

where $f = 1.3$ is the refinement factor (30% geometric growth), and the absolute ceiling is $n_\text{max} = 50$ points per dimension (configurable). This geometric growth prevents runaway grid inflation.

If no problematic bins are identified (errors are spread uniformly), the algorithm falls back to uniform midpoint insertion across the entire dimension.

**Iteration budget.** The default maximum is 5 iterations. Convergence typically occurs in 2–3 iterations for a 5 bps target, since each iteration both adds points where needed and benefits from the slice cache (only new $(\sigma, r)$ pairs require fresh PDE solves). The total PDE solve cost is the sum of table solves (new slices only, thanks to caching) plus validation solves ($N$ per iteration).

---

## 11. Implied Volatility

Implied volatility is the volatility that, when plugged into the pricing model, reproduces the observed market price. Computing it requires inverting the price-to-volatility mapping — a root-finding problem.

The library provides two approaches with very different speed/accuracy tradeoffs.

### FDM-Based IV (Brent's Method)

The direct approach: use the PDE solver as a black box and find the root of

$$f(\sigma) = V_\text{model}(\sigma) - V_\text{market} = 0$$

We use Brent's method, which combines bisection, secant, and inverse quadratic interpolation. It requires a bracketing interval $[\sigma_\text{lo}, \sigma_\text{hi}]$ where $f$ changes sign, then iterates:

1. Choose interpolation step (secant or inverse quadratic) if it's safe
2. Fall back to bisection if interpolation fails or the bracket is too wide
3. Stop when $|f(\sigma)| < \varepsilon$ or the bracket is smaller than machine epsilon

**Properties:**
- Guaranteed convergence (as long as the root is bracketed)
- Superlinear convergence rate (~1.6)
- No derivatives required (important — $\partial V / \partial \sigma$ from the PDE is expensive)
- Typically 5–8 iterations for $\sigma \in [0.01, 3.0]$

Each iteration calls the PDE solver (~2ms), so a single IV solve costs ~15ms. Accurate but too slow for production use with thousands of queries.

### Interpolated IV (Newton on B-Spline Surface)

The fast approach: pre-compute a 4D price table (Part II, sections 7–8), then solve for IV using Newton's method on the B-spline surface.

$$\sigma_{k+1} = \sigma_k - \frac{P(m, \tau, \sigma_k, r) - V_\text{market}}{\partial P / \partial\sigma(m, \tau, \sigma_k, r)}$$

The key advantage: both $P$ and $\partial P / \partial\sigma$ come from B-spline evaluation (~500ns each), not PDE solves. Newton's method converges quadratically (error squares each iteration), typically in 3–4 iterations.

**Performance comparison:**

| Method | Time per IV | Use case |
|--------|------------|----------|
| FDM (Brent) | ~15ms | Ground truth, validation, few queries |
| Interpolated (Newton) | ~3.5μs | Production, many queries |

The interpolated solver is ~5,000× faster, at the cost of pre-computation time and interpolation error (typically 10–60 bps depending on grid profile).

### Vega Pre-Check for Undefined IV

When an option's vega is near zero (deep OTM/ITM, or very short dated), implied volatility is effectively undefined — any tiny price error maps to huge IV swings.

The interpolated IV solver checks surface vega at three representative volatilities (10%, 25%, 50%) before starting the root search. If $\max(\nu) < \nu_\text{threshold}$ (default $10^{-4}$), it returns `VegaTooSmall` immediately (~600ns) instead of running a doomed Brent search.

This is more robust than a time-value/strike ratio heuristic, and costs only 3 surface evaluations.

---

## 12. Interpolated Greeks via Chain Rule

When American option prices are stored in an interpolated surface, Greeks are computed via the chain rule through the coordinate transform. This section covers the mathematics; see [INTERPOLATION_FRAMEWORK.md](INTERPOLATION_FRAMEWORK.md) for the software architecture.

### First-Order Greeks (Delta, Vega, Theta, Rho)

For a surface parameterized by coordinates $\mathbf{c} = T(S, K, \tau, \sigma, r)$, the chain rule gives:

$$\frac{\partial V}{\partial p} = \sum_{i=1}^{N} \frac{\partial V}{\partial c_i} \cdot \frac{\partial c_i}{\partial p}$$

where $p \in \{S, \sigma, \tau, r\}$ is the physical parameter and $N$ is the dimensionality of the interpolation space.

**StandardTransform4D.** Coordinates: $\mathbf{c} = (\ln(S/K),\; \tau,\; \sigma,\; r)$.

| Greek | Weights $\partial c_i / \partial p$ |
|-------|-----|
| Delta | $(1/S,\; 0,\; 0,\; 0)$ |
| Vega | $(0,\; 0,\; 1,\; 0)$ |
| Theta | $(0,\; -1,\; 0,\; 0)$ |
| Rho | $(0,\; 0,\; 0,\; 1)$ |

Each Greek requires exactly one interpolant partial evaluation (the single non-zero weight).

**DimensionlessTransform3D.** Coordinates: $\mathbf{c} = (\ln(S/K),\; \sigma^2\tau/2,\; \ln(2r/\sigma^2))$.

| Greek | Weights $\partial c_i / \partial p$ |
|-------|-----|
| Delta | $(1/S,\; 0,\; 0)$ |
| Vega | $(0,\; \sigma\tau,\; -2/\sigma)$ |
| Theta | $(0,\; -\sigma^2/2,\; 0)$ |
| Rho | $(0,\; 0,\; 1/r)$ |

The dimensionless transform's vega couples two axes ($\tau'$ and $\ln\kappa$), requiring two interpolant partial evaluations instead of one. This coupling is inherent in the dimensionless parameterization — $\sigma$ appears in both coordinate definitions.

### Gamma (Second-Order)

Gamma $= \partial^2 V / \partial S^2$ requires a second derivative. For $x = \ln(S/K)$:

$$\frac{\partial^2 V}{\partial S^2} = \frac{1}{S^2}\left(\frac{\partial^2 V}{\partial x^2} - \frac{\partial V}{\partial x}\right)$$

This follows from $\partial x / \partial S = 1/S$ and $\partial^2 x / \partial S^2 = -1/S^2$, applied to the chain rule for second derivatives.

**Analytical (B-spline interpolants).** B-spline interpolants provide `eval_second_partial(axis, coords)` for exact $\partial^2 V / \partial x^2$. The derivative of a cubic B-spline is a quadratic B-spline — computed analytically in $O(n)$ per evaluation.

**Finite difference (Chebyshev interpolants).** When analytical second partials are unavailable, central differences are used:

$$\frac{\partial^2 V}{\partial x^2} \approx \frac{V(x+h) - 2V(x) + V(x-h)}{h^2}$$

with $h = 10^{-4}$ in log-moneyness.

### EEP Decomposition for Greeks

All transform/chain-rule machinery operates on the EEP surface only. European Greeks are added at the final layer via exact Black-Scholes formulas:

$$\text{American Greek} = \text{EEP Greek} + \text{European Greek}$$

**Early exit.** When $\text{EEP} \leq 0$ (deep OTM), the American value equals the European value, so the American Greek equals the European Greek. The EEP layer detects this and returns the analytical European Greek directly, without evaluating any interpolation derivatives.

---

# Part III — Analysis

## 13. Convergence Analysis

### Overall Error Budget

The total pricing error has three independent contributions:

$$\varepsilon_\text{total} \sim C_x\Delta x^2 + C_t\Delta t^2 + \varepsilon_\text{obstacle}$$

- **Spatial**: $O(\Delta x^2)$ from centered differences
- **Temporal**: $O(\Delta t^2)$ from TR-BDF2
- **Obstacle**: the projection is non-expansive (doesn't amplify errors) and the free boundary is Lipschitz continuous (Kinderlehrer-Stampacchia)

The grid estimation strategy (section 6) balances these by coupling $\Delta t$ to $\Delta x_\min$, ensuring neither dominates.

### Grid Independence

To verify convergence, refine the grid until results stabilize:

1. Solve on coarse grid ($n = 100$, $\Delta t = 10^{-3}$)
2. Refine spatially ($n = 200$, $\Delta t = 10^{-3}$)
3. Refine temporally ($n = 200$, $\Delta t = 5 \times 10^{-4}$)
4. Check $|V_\text{refined} - V_\text{coarse}| < \varepsilon$

Typical behavior:
- ATM options: $10^{-3}$ price error at $n = 141$, $\Delta t = 10^{-3}$
- Deep ITM/OTM: require finer grids (steep exercise boundaries)
- Greeks: need 2–3× finer grids than prices (higher-order quantities are more sensitive)

### Interpolation Error (Price Tables)

For the B-spline price tables, the additional interpolation error is $O(h^4)$ per dimension. The separable fitting preserves this order. In practice, the dominant error source is the volatility dimension (highest curvature), which is why it receives 1.5× weight in the grid budget.

Adaptive refinement (section 10) provides a verified error bound by testing against fresh PDE solves at random parameter combinations.

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
