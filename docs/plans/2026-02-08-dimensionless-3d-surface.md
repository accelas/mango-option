# Dimensionless PDE Coordinates: 4D to 3D Surface Reduction

## Problem Statement

We have an American option pricing library that pre-computes prices on a 4D parameter grid and fits a tensor-product cubic B-spline for fast interpolation. The current axes are:

$$\bigl(x,\; \tau,\; \sigma,\; r\bigr)$$

where $x = \ln(S/K)$ is log-moneyness, $\tau$ is time to expiry, $\sigma$ is volatility, and $r$ is the risk-free rate. The question is whether a dimensionless coordinate change can collapse this to 3D, reducing build cost, query cost, and storage.

---

## 1. Current Architecture

### 1.1 The PDE

American option prices satisfy the Black-Scholes variational inequality in log-moneyness:

$$\frac{\partial u}{\partial \tau} = \frac{\sigma^2}{2}\frac{\partial^2 u}{\partial x^2} + \left(r - q - \frac{\sigma^2}{2}\right)\frac{\partial u}{\partial x} - r\,u, \qquad u \geq \psi$$

with initial condition $u(x, 0) = \psi(x) = \max(1 - e^x, 0)$ (put) or $\max(e^x - 1, 0)$ (call), and the complementarity condition enforced via the Projected Thomas algorithm (Brennan-Schwartz).

Here $q$ is the continuous dividend yield. The PDE is solved with TR-BDF2 time stepping and Rannacher startup.

### 1.2 The 4D Price Table

A single PDE solve produces a price surface over $(x, \tau)$ for fixed $(\sigma, r)$. To cover a range of volatilities and rates, we solve the PDE for every combination on a grid:

- **Outer loop:** $N_\sigma \times N_r$ PDE solves (e.g., $20 \times 10 = 200$)
- **Inner (free):** Each solve yields prices at all $(x_i, \tau_j)$ grid points via spatial grid snapshots

The results fill a 4D tensor $u[i,j,k,l]$ indexed by $(x, \tau, \sigma, r)$. A tensor-product cubic B-spline is fit via separable 1D fits (4 sequential passes). The fitted surface supports evaluation and analytic partial derivatives.

### 1.3 EEP Decomposition

Rather than storing raw American prices, we store the Early Exercise Premium:

$$\text{EEP}(x, \tau, \sigma, r) = \frac{P_\text{Am}(x, \tau, \sigma, r) - P_\text{Eu}(x, \tau, \sigma, r, q)}{K/K_\text{ref}}$$

At query time, the American price is reconstructed:

$$P_\text{Am} = \text{EEP}(x, \tau, \sigma, r) \cdot \frac{K}{K_\text{ref}} + P_\text{Eu}(S, K, \tau, \sigma, r, q)$$

where the European price $P_\text{Eu}$ is computed analytically via Black-Scholes. This decomposition exploits the fact that EEP is smooth and slowly varying, while the payoff kink and most parameter sensitivity live in the European component.

### 1.4 Query-Time Vega (the only Greek from the B-spline)

The interpolated IV solver uses Newton-Raphson, which requires vega $= \partial P_\text{Am}/\partial\sigma$. By the EEP decomposition:

$$\text{vega}_\text{Am} = \frac{K}{K_\text{ref}} \cdot \frac{\partial\,\text{EEP}}{\partial\sigma} + \text{vega}_\text{Eu}$$

where $\partial\text{EEP}/\partial\sigma$ is the analytic B-spline partial derivative along axis 2 (sigma). No other Greeks (delta, gamma, theta, rho) are computed from the B-spline surface. The European vega is computed in closed form.

### 1.5 Current Costs

| Metric | Value |
|--------|-------|
| PDE solves per surface | $N_\sigma \times N_r$ (e.g., 200) |
| B-spline coefficient lookups per query | $4^4 = 256$ |
| Query time | ~135 ns |
| Typical tensor size | $60 \times 20 \times 20 \times 10 = 240{,}000$ coefficients |

### 1.6 Three Usage Paths

The library has three paths that would be affected:

1. **Direct PDE solver.** Prices one option at a time with fixed scalar $(\sigma, r)$. No interpolation. Unaffected by this proposal except as a new "dimensionless PDE mode."

2. **Standard interpolation** (continuous dividend yield $q$). Builds one 4D B-spline surface. This is the primary target.

3. **Segmented interpolation** (discrete dividends). Partitions the maturity axis at dividend dates; builds a separate 4D B-spline per segment. Each segment solves with $q = 0$ because dividends are discrete. Segments are spliced at query time. Multiple reference strikes ($K_\text{ref}$) are interpolated via Catmull-Rom in $\ln K_\text{ref}$.

---

## 2. Proposed Dimensionless Coordinate Change

### 2.1 The Transform

Introduce dimensionless time $\tau'$ and dimensionless rate $\kappa$:

$$\tau' = \frac{\sigma^2 \tau}{2}, \qquad \kappa = \frac{2r}{\sigma^2}$$

The inverse:

$$\sigma = \sqrt{\frac{2\tau'}{\tau}}, \qquad r = \kappa \cdot \frac{\tau'}{\tau}$$

Or equivalently, given the original parameters $(S, K, \tau, \sigma, r)$:

$$x = \ln(S/K), \qquad \tau' = \sigma^2\tau/2, \qquad \kappa = 2r/\sigma^2$$

No new inputs are required at query time.

### 2.2 The Dimensionless PDE

Substituting into the Black-Scholes PDE and dividing by $\sigma^2/2$:

$$\frac{\partial u}{\partial \tau'} = \frac{\partial^2 u}{\partial x^2} + (\kappa - \delta - 1)\frac{\partial u}{\partial x} - \kappa\,u$$

where $\delta = 2q/\sigma^2$. The initial condition is unchanged:

$$u(x, 0) = \psi(x) = \max(1 - e^x, 0) \qquad \text{(put)}$$

The obstacle condition $u \geq \psi$ is unchanged.

**Key observation:** When $q = 0$, we have $\delta = 0$ and the PDE depends only on $(x, \tau', \kappa)$ — a 3D problem. The entire $(\sigma, \tau)$ plane collapses into $\tau'$, and $(\sigma, r)$ collapses into $\kappa$.

### 2.3 Expected Benefits (q = 0)

| Metric | Current (4D) | Proposed (3D) | Improvement |
|--------|-------------|--------------|-------------|
| PDE solves | $N_\sigma \times N_r$ (~200) | $N_\kappa$ (~30) | ~7x fewer |
| Coefficient lookups per query | $4^4 = 256$ | $4^3 = 64$ | 4x fewer |
| Estimated query time | ~135 ns | ~35 ns | ~4x faster |
| Typical tensor size | 240,000 | ~72,000 | ~3x smaller |

### 2.4 Applicability to the Three Paths

| Path | q value | 3D exact? | Notes |
|------|---------|-----------|-------|
| Standard, $q = 0$ | 0 | Yes | Full benefit |
| Standard, $q > 0$ | Fixed per surface | No | Needs one of the approaches in Section 3 |
| Segmented (discrete dividends) | 0 per segment | Yes | Full benefit; each segment uses $q = 0$ |

The segmented path is the cleanest case: discrete dividends are handled by the segment structure, and each segment's PDE uses $q = 0$.

---

## 3. The q > 0 Problem

When $q > 0$, the dimensionless PDE has an additional parameter $\delta = 2q/\sigma^2$. Since $q$ is fixed per surface but $\sigma$ varies across the grid, $\delta$ is not constant — it varies with the query point. This means the solution $u(x, \tau', \kappa, \delta)$ is technically 4D again.

However, the $\delta$ dependence may be weak, especially in the EEP. Below are four approaches ordered by complexity.

### 3.1 Option 1: Effective-Rate Approximation

Absorb $q$ into $\kappa$ by splitting the rate roles:

- **Drift:** use $\kappa_\text{eff} = 2(r - q)/\sigma^2$
- **Killing term:** use $\kappa = 2r/\sigma^2$

The PDE becomes:

$$\frac{\partial u}{\partial \tau'} = \frac{\partial^2 u}{\partial x^2} + (\kappa_\text{eff} - 1)\frac{\partial u}{\partial x} - \kappa\,u$$

This is not self-consistent: the drift uses $r - q$ but the killing term uses $r$, introducing an $O(q\tau)$ error. The surface remains 3D.

**Question for review:** Can this error be bounded tightly? For typical equity parameters ($q \sim 0.01\text{--}0.03$, $\tau \leq 2$), the error term $q\tau \leq 0.06$ — is this acceptable for the EEP?

### 3.2 Option 2: First-Order Perturbation in $\delta$

Solve the base problem at $\delta = 0$ to get $u_0(x, \tau', \kappa)$. Then solve the sensitivity PDE:

$$\frac{\partial v}{\partial \tau'} = \frac{\partial^2 v}{\partial x^2} + (\kappa - 1)\frac{\partial v}{\partial x} - \kappa\,v - \frac{\partial u_0}{\partial x}$$

where $v = \partial u / \partial\delta\big|_{\delta=0}$. The source term $-\partial u_0/\partial x$ is known from the base solve. At query time:

$$u(x, \tau', \kappa, \delta) \approx u_0(x, \tau', \kappa) + \delta \cdot v(x, \tau', \kappa)$$

This requires two 3D surfaces ($u_0$ and $v$), doubling storage and build time ($2 \times N_\kappa$ solves). Query cost is $2 \times 64 = 128$ lookups plus arithmetic.

**Questions for review:**

- The sensitivity PDE has a source term from $\partial u_0/\partial x$. For American options, $u_0$ is not smooth at the free boundary (exercise boundary). Does the kink in $\partial u_0/\partial x$ at the free boundary cause issues for the sensitivity PDE?
- Is first-order in $\delta$ sufficient? For $q = 0.03$, $\sigma \in [0.10, 0.50]$, we get $\delta \in [0.24, 6.0]$. The range is wide — is the linear approximation accurate at $\delta = 6$?
- The sensitivity PDE inherits the American constraint: how does the free boundary of the perturbed problem relate to the base problem's free boundary? Should $v$ satisfy $v \geq 0$ (since increasing $\delta$ should increase early exercise premium for puts)?

### 3.3 Option 3: $\delta$ as a Sparse 4th Axis

Build a 4D B-spline in $(x, \tau', \kappa, \delta)$ but with only 2--3 points along $\delta$. Since $q$ is fixed per surface, the range of $\delta$ is:

$$\delta \in \left[\frac{2q}{\sigma_\text{max}^2},\; \frac{2q}{\sigma_\text{min}^2}\right]$$

For $q = 0.02$, $\sigma \in [0.10, 0.50]$: $\delta \in [0.16, 4.0]$.

With 3 $\delta$ grid points, total solves $\approx 3 \times N_\kappa \approx 90$. Still much fewer than the current $N_\sigma \times N_r \approx 200$. Query cost is $4^4 = 256$ (same as current 4D), or potentially less if the sparse $\delta$ axis uses linear rather than cubic interpolation.

**Question for review:** Is the EEP surface smooth enough in $\delta$ that 2--3 grid points suffice? If so, would linear interpolation along $\delta$ (with cubic along $x, \tau', \kappa$) preserve adequate accuracy?

### 3.4 Option 4: Ignore $\delta$ for EEP

The EEP decomposition already separates the American price into:

$$P_\text{Am} = \text{EEP}(x, \tau', \kappa) \cdot \frac{K}{K_\text{ref}} + P_\text{Eu}(S, K, \tau, \sigma, r, q)$$

The European component $P_\text{Eu}$ is computed analytically and captures all $q$ sensitivity exactly. The hypothesis is that the EEP's residual dependence on $\delta$ is negligible:

$$\text{EEP}(x, \tau', \kappa, \delta) \approx \text{EEP}(x, \tau', \kappa, 0)$$

This would mean the 3D surface built with $q = 0$ can be used even when $q > 0$, as long as the European component is computed with the correct $q$.

**Questions for review:**

- For a put option, the early exercise boundary $S^*(\tau)$ depends on $q$ through the drift. Higher $q$ means lower effective drift, which shifts the exercise boundary. How much does this shift affect the EEP?
- The EEP for a put is $P_\text{Am} - P_\text{Eu}$. As $q$ increases, both the American and European prices decrease, but possibly at different rates. Is $\partial\text{EEP}/\partial q$ small relative to $\partial P_\text{Eu}/\partial q$ across the typical parameter range?
- Calls with $q > 0$ have non-trivial early exercise boundaries (unlike European calls). Is the EEP-vs-$\delta$ dependence stronger for calls than puts?
- What about deep ITM options where the early exercise boundary is most sensitive to $q$?

---

## 4. Vega Under the Coordinate Change

### 4.1 Current Vega

Vega is the only Greek computed from B-spline partials. Currently:

$$\text{vega}_\text{Am} = \frac{K}{K_\text{ref}} \cdot \frac{\partial\,\text{EEP}}{\partial\sigma}\bigg|_{\tau,r} + \text{vega}_\text{Eu}$$

where $\partial\text{EEP}/\partial\sigma$ is a single B-spline partial along axis 2 (the $\sigma$ axis), evaluated analytically from the B-spline derivative basis functions. Cost: 256 coefficient lookups.

### 4.2 Vega After the Transform

After the coordinate change, the surface is $\text{EEP}(x, \tau', \kappa)$. Since $\tau' = \sigma^2\tau/2$ and $\kappa = 2r/\sigma^2$, the chain rule gives:

$$\frac{\partial\,\text{EEP}}{\partial\sigma}\bigg|_{\tau,r} = \frac{\partial\,\text{EEP}}{\partial\tau'}\bigg|_{\kappa} \cdot \frac{\partial\tau'}{\partial\sigma}\bigg|_{\tau} + \frac{\partial\,\text{EEP}}{\partial\kappa}\bigg|_{\tau'} \cdot \frac{\partial\kappa}{\partial\sigma}\bigg|_{r}$$

The Jacobian entries:

$$\frac{\partial\tau'}{\partial\sigma} = \sigma\tau, \qquad \frac{\partial\kappa}{\partial\sigma} = -\frac{4r}{\sigma^3} = -\frac{2\kappa}{\sigma}$$

So:

$$\frac{\partial\,\text{EEP}}{\partial\sigma}\bigg|_{\tau,r} = \sigma\tau \cdot \frac{\partial\,\text{EEP}}{\partial\tau'} - \frac{2\kappa}{\sigma} \cdot \frac{\partial\,\text{EEP}}{\partial\kappa}$$

This requires **two** B-spline partial evaluations (axis 1 and axis 2 of the 3D surface) plus trivial arithmetic. Cost: $2 \times 64 = 128$ coefficient lookups. Net cost is lower than the current 256.

### 4.3 Other Greeks (Not Currently Used)

For completeness, the other Greeks under the transform:

| Greek | Formula | B-spline partials needed |
|-------|---------|------------------------|
| Delta-like | $\partial u/\partial x$ | 1 partial (axis 0), same as before |
| Theta | $\partial u/\partial\tau = (\sigma^2/2) \cdot \partial u/\partial\tau'$ | 1 partial (axis 1) |
| Rho | $\partial u/\partial r = (2/\sigma^2) \cdot \partial u/\partial\kappa$ | 1 partial (axis 2) |
| Vega | $\partial u/\partial\sigma = \sigma\tau \cdot \partial u/\partial\tau' - (2\kappa/\sigma) \cdot \partial u/\partial\kappa$ | 2 partials (axes 1, 2) |

Vega is the only Greek that requires more than one B-spline partial. This is because $\sigma$ enters both dimensionless coordinates.

---

## 5. Build-Time Changes

### 5.1 Dimensionless PDE Mode

The PDE solver needs a new mode that operates in the transformed coordinates. For each $\kappa_i$ on the grid:

1. Set up the dimensionless operator: $\mathcal{L}u = \partial^2 u/\partial x^2 + (\kappa_i - 1)\partial u/\partial x - \kappa_i u$ (for $q = 0$)
2. March in $\tau'$ from 0 to $\tau'_\text{max}$
3. Take snapshots at the $\tau'$ grid points
4. Apply the American obstacle constraint at each step

The spatial grid in $x$ is unchanged. The time grid is in $\tau'$ instead of $\tau$.

### 5.2 Grid Construction

The $\kappa$ grid must cover:

$$\kappa \in \left[\frac{2r_\text{min}}{\sigma_\text{max}^2},\; \frac{2r_\text{max}}{\sigma_\text{min}^2}\right]$$

For $r \in [0.01, 0.10]$, $\sigma \in [0.10, 0.50]$: $\kappa \in [0.08, 20.0]$. This is a wide range spanning ~2.5 orders of magnitude. Logarithmic spacing in $\kappa$ may be more appropriate than linear.

The $\tau'$ grid must cover:

$$\tau' \in \left[0,\; \frac{\sigma_\text{max}^2 \cdot \tau_\text{max}}{2}\right]$$

For $\sigma_\text{max} = 0.50$, $\tau_\text{max} = 2.0$: $\tau'_\text{max} = 0.25$.

**Question for review:** The $\kappa$ range is wide. Is the EEP smooth in $\kappa$, or does it have regions of high curvature (e.g., near $\kappa = 1$ where the drift changes sign)? How many $\kappa$ grid points are needed for the same interpolation accuracy as the current $N_\sigma \times N_r$ approach?

### 5.3 Segmented Surfaces

Each segment currently builds a 4D surface with a local $\tau$ grid. Under the transform, each segment would have a local $\tau'$ grid. The segment boundaries, currently at physical times $\tau_k$, would map to $\tau'_k = \sigma^2\tau_k/2$ — but since $\sigma$ varies, the boundary in $\tau'$ depends on the query. This means:

- Segment boundaries must remain in physical $\tau$ (as they are determined by dividend dates)
- The $\tau \to \tau'$ mapping happens per-query inside each segment
- Each segment's B-spline domain covers $\tau' \in [0, \sigma_\text{max}^2 \Delta\tau_k/2]$ where $\Delta\tau_k$ is the segment width

**Question for review:** Is there a cleaner way to handle the segment boundaries in dimensionless coordinates? Or is the per-query mapping the natural approach?

---

## 6. Adaptive Refinement Implications

The current adaptive refinement loop iterates over 4 dimensions and selects the worst dimension for targeted point insertion. Under the 3D transform:

- Refinement operates on $(x, \tau', \kappa)$ instead of $(x, \tau, \sigma, r)$
- Error attribution has 3 bins instead of 4
- Fewer total PDE solves per iteration (loop over $\kappa$ instead of $\sigma \times r$)
- Validation still uses Latin Hypercube Sampling in the 3D space, with coordinate transform to physical parameters for PDE reference solves

The validation step must map 3D sample points back to physical $(\sigma, r)$ for PDE reference solves. Multiple physical $(\sigma, \tau, r)$ combinations map to the same $(\tau', \kappa)$, so the validator should sample diverse physical parameters, not just diverse dimensionless ones.

**Question for review:** Does the coordinate transform affect the convergence rate of adaptive refinement? The EEP in $(x, \tau', \kappa)$ space may be smoother than in $(x, \tau, \sigma, r)$ space, which would mean faster convergence with fewer iterations.

---

## 7. Summary of Open Questions

Ordered by priority for the 3D reduction project:

1. **[Section 3.4]** Is the EEP's dependence on $\delta = 2q/\sigma^2$ negligible? Can we quantify $\partial\text{EEP}/\partial\delta$ relative to $\partial P_\text{Eu}/\partial q$ for typical equity parameters?

2. **[Section 3.2]** If Option 4 fails, is the first-order perturbation in $\delta$ accurate enough? The range $\delta \in [0.16, 6.0]$ for typical parameters is not small — does the linear approximation hold?

3. **[Section 5.2]** How many $\kappa$ grid points are needed for the same interpolation accuracy? Is the EEP smooth in $\kappa$, or are there curvature features (e.g., near $\kappa = 1$)?

4. **[Section 4.2]** Does the two-partial vega formula introduce numerical issues? The term $2\kappa/\sigma$ can be large when $\sigma$ is small and $r$ is moderate — does this amplify B-spline evaluation noise?

5. **[Section 5.3]** Is the per-query $\tau \to \tau'$ mapping for segmented surfaces the right approach, or is there a better formulation?

6. **[Section 6]** Does the dimensionless coordinate system improve or worsen adaptive refinement convergence?

7. **[Section 3.2]** For the perturbation approach: how does the free boundary kink in $\partial u_0/\partial x$ affect the sensitivity PDE? Does $v$ need its own American constraint?

---

## Appendix A: Notation Summary

| Symbol | Definition | Typical range |
|--------|-----------|---------------|
| $x$ | $\ln(S/K)$, log-moneyness | $[-1.5, 1.5]$ |
| $\tau$ | Time to expiry (years) | $[0, 2]$ |
| $\sigma$ | Volatility | $[0.05, 0.50]$ |
| $r$ | Risk-free rate | $[0.01, 0.10]$ |
| $q$ | Continuous dividend yield | $[0, 0.05]$ |
| $\tau'$ | $\sigma^2\tau/2$, dimensionless time | $[0, 0.25]$ |
| $\kappa$ | $2r/\sigma^2$, dimensionless rate | $[0.08, 20]$ |
| $\delta$ | $2q/\sigma^2$, dimensionless yield | $[0, 6]$ |
| $K_\text{ref}$ | Reference strike for normalization | Fixed per surface |
| EEP | Early Exercise Premium | $\geq 0$ |
| $\psi(x)$ | Payoff (normalized by $K$) | $\max(1 - e^x, 0)$ (put) |

## Appendix B: Jacobian for Coordinate Transform

The full Jacobian $(\sigma, r) \to (\tau', \kappa)$ at fixed $\tau$:

$$\frac{\partial(\tau', \kappa)}{\partial(\sigma, r)} = \begin{pmatrix} \sigma\tau & 0 \\ -4r/\sigma^3 & 2/\sigma^2 \end{pmatrix}$$

Determinant: $2\tau/\sigma$. Non-singular for $\sigma > 0$, $\tau > 0$.

Inverse (for mapping back):

$$\frac{\partial(\sigma, r)}{\partial(\tau', \kappa)} = \frac{1}{2\tau/\sigma}\begin{pmatrix} 2/\sigma^2 & 0 \\ 4r/\sigma^3 & \sigma\tau \end{pmatrix} = \begin{pmatrix} 1/(\sigma\tau) & 0 \\ 2r/(\sigma^2\tau) & \sigma/2 \end{pmatrix}$$

This is used in the chain rule for Greeks (Section 4).
