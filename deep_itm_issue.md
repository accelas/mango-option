# Deep ITM American Put Pricing Error - Expert Review Request

## Problem Summary

We have implemented two mathematically distinct obstacle-constrained PDE solvers for American option pricing:

1. **Projected Thomas Algorithm** (Brennan-Schwartz): Single-pass LCP solver that enforces `u ≥ ψ` during backward substitution
2. **Hybrid PDAS** (Hintermüller-Ito-Kunisch): Primal-dual active set with Newton warm-up and ramped θ

Both algorithms produce **identical incorrect values** for a deep in-the-money (ITM) American put test case, while correctly pricing at-the-money (ATM) and out-of-the-money (OTM) options. This suggests a fundamental issue in problem formulation rather than solver implementation.

## Test Case Details

**Option Parameters:**
- Type: American Put
- Spot (S): 0.25
- Strike (K): 100.0
- Maturity (T): 0.75 years
- Risk-free rate (r): 0.05
- Dividend yield (q): 0.0
- Volatility (σ): 0.2

**Moneyness:**
- Intrinsic value: K - S = 99.75
- Log-moneyness: ln(S/K) = ln(0.25/100) ≈ -5.99
- Deep ITM (spot is 1/400 of strike)

**Grid Configuration:**
- Log-space grid: x ∈ [-7.0, 2.0] with 301 points
- Grid spacing: sinh-spaced with concentration factor 2.0
- Time steps: 1500

**PDE Formulation:**
Backward pricing in log-moneyness (x = ln(S/K)):
```
∂V/∂τ = (σ²/2)·∂²V/∂x² + (r - q - σ²/2)·∂V/∂x - r·V
```
with obstacle constraint:
```
V(x,τ) ≥ ψ(x) = max(K - K·exp(x), 0) = K·max(1 - exp(x), 0)
```

**Time-stepping:** TR-BDF2 (L-stable composite method)
- Stage 1: Trapezoidal rule to t_n + γ·Δt (γ ≈ 0.586)
- Stage 2: BDF2 to t_n+1

## Observed Results

**Expected Behavior (Deep ITM Put):**
- Value at spot should equal intrinsic value: 99.75
- Option should be "locked" to immediate exercise boundary
- Time value should be negligible (< 0.50)

**Actual Results:**

| Solver | Value at Spot | Error | Status |
|--------|--------------|-------|--------|
| Heuristic (baseline) | 99.75 | 0.0 | ✅ PASS |
| Projected Thomas | 115.97 | +16.22 | ❌ FAIL |
| Hybrid PDAS | 115.97 | +16.22 | ❌ FAIL |

**Key Observations:**
1. Both new solvers produce **identical incorrect values** (115.97)
2. Error is consistently +16.22 points (~16% of intrinsic value)
3. Both solvers correctly price all ATM/OTM test cases
4. The original heuristic method (Newton + post-solve projection) gets deep ITM correct
5. Convergence is achieved (no numerical instability reported)

## Solver Implementation Details

### Projected Thomas (Brennan-Schwartz)

**Algorithm:**
```cpp
// Forward elimination (standard Thomas)
for i = 1 to n-1:
    c'[i] = c[i] / (b[i] - a[i]*c'[i-1])
    d'[i] = (d[i] - a[i]*d'[i-1]) / (b[i] - a[i]*c'[i-1])

// Projected backward substitution
u[n-1] = max(d'[n-1], ψ[n-1])
for i = n-2 down to 0:
    u[i] = max(d'[i] - c'[i]*u[i+1], ψ[i])  // Projection coupled with recursion
```

**Problem Transformation:**
- Solve: J·δu = -F(u) subject to (u + δu) ≥ ψ
- Equivalent to: δu ≥ (ψ - u)
- Pass `psi_delta[i] = psi[i] - u[i]` to projected Thomas

### Hybrid PDAS

**Algorithm:**
```cpp
// Phase 1: Newton warm-up (3 iterations)
for k = 1 to 3:
    Build Jacobian J
    Solve J·δu = -F(u)
    u ← u + δu
    u ← max(u, ψ)  // Post-solve projection

// Phase 2: PDAS with ramped θ
for k = 1 to max_iter:
    β = 0.2 + (0.9 - 0.2)·min(k/5, 1)  // Ramp from 0.2 to 0.9
    θ = β / L_max

    Update multipliers: λ[i] = max(0, -residual[i]) if active else 0
    Update active set: A = {i : (u[i] - ψ[i]) - θ·λ[i] ≤ 0}

    Enforce u[i] = ψ[i] for i ∈ A in linear system
    Solve modified system

    if active_set stable and complementarity < tol: break
```

## Why Both Solvers Fail Identically

The fact that two completely different algorithms produce the same incorrect value suggests:

1. **Not a solver bug**: The algorithms are implemented correctly but may violate M-matrix assumptions
2. **Possible boundary condition issue**: Deep ITM puts approach S=0 boundary
3. **Grid resolution**: At x ≈ -6, sinh-spacing may be too coarse
4. **Time-stepping artifact**: TR-BDF2 may accumulate error in extreme regions
5. **Obstacle evaluation**: ψ(x) = K·max(1 - exp(x), 0) may have numerical issues near x → -∞

## Questions for Expert Review

1. **M-Matrix Property**: For TR-BDF2 with the given time step (Δt ≈ T/1500 ≈ 0.0005), does the implicit Jacobian satisfy M-matrix conditions at x ≈ -6?
   - The Projected Thomas algorithm assumes M-matrix for convergence
   - Violations could cause the constraint to "over-project"

2. **Boundary Conditions**: We use Dirichlet BC at the left boundary (x = -7). For S ≈ 0 (x → -∞), should we instead use:
   - V(x_min, τ) = K·exp(-rτ) (discounted strike)?
   - Different BC formulation for deep ITM?

3. **Grid Spacing**: The sinh-spaced grid with 301 points spans [-7, 2]. At x = -6:
   - What is the local grid spacing Δx?
   - Is this sufficient to resolve the free boundary?
   - Could under-resolution cause the solvers to "miss" the exercise boundary?

4. **Obstacle Function**: For x < -5, exp(x) ≈ 0.0067, so ψ(x) ≈ K. Could floating-point errors in:
   ```
   psi[i] = K * max(1.0 - exp(x[i]), 0.0)
   ```
   cause the obstacle to be set incorrectly?

5. **Time-Step Size**: With 1500 steps over T=0.75, Δt ≈ 0.0005. For TR-BDF2 with deep ITM:
   - Is this small enough to maintain monotonicity?
   - Could accumulation of small errors over 1500 steps explain +16 point error?

6. **Initial Condition**: At maturity (τ=0), we have V(x,0) = ψ(x). For deep ITM (x ≈ -6):
   - Is the initial condition being set correctly?
   - Could there be an off-by-one error in time indexing?

## Heuristic Method (Works Correctly)

For reference, the heuristic method that passes this test uses:

```cpp
// Standard Newton iteration
for k = 1 to max_iter:
    Build Jacobian J
    Solve J·δu = -F(u)
    u ← u + δu

    // Empirical active set with tuned threshold
    if |u[i] - ψ[i]| < threshold:
        u[i] = ψ[i]

    if converged: break

// Post-iteration projection
u ← max(u, ψ)
```

This suggests the issue may be related to **when** the projection is applied (during solve vs after solve).

## Reproducible Test Case

```cpp
// Grid
auto grid_spec = GridSpec<double>::sinh_spaced(-7.0, 2.0, 301, 2.0);

// Workspace
auto workspace = AmericanSolverWorkspace::create(grid_spec.value(), 1500);

// Parameters
AmericanOptionParams params(
    0.25,   // spot
    100.0,  // strike
    0.75,   // maturity
    0.05,   // rate
    0.0,    // dividend
    OptionType::PUT,
    0.2     // volatility
);

// Solve (with Projected Thomas or Hybrid PDAS)
auto result = solver.solve();

// Expected: result.value_at(0.25) ≈ 99.75
// Actual:   result.value_at(0.25) = 115.97
```

## Request

Please review this formulation and help identify why both Projected Thomas and Hybrid PDAS produce values 16% above intrinsic for deep ITM American puts, while correctly pricing ATM/OTM options. The identical failure across different algorithms suggests a systematic issue in problem setup rather than solver implementation.
