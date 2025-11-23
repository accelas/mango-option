# Active Set Tolerance Calibration Challenge

## ✅ SOLVED - See Solution Document

**Solution**: Implemented primal-dual active set method with auto-scaling multiplier threshold. All 6 tests pass.

**Details**: See `primal_dual_active_set_solution.md`

---

## Original Problem Statement

We've successfully implemented an active set method to fix the deep ITM American put pricing issue (Issue #196), where values were inflated by ~$16 above intrinsic value. The active set method correctly locks nodes to the payoff in the exercise region by modifying the Jacobian.

**However**, we faced a tolerance calibration challenge where no single tolerance value passed all 6 tests.

## Test Results Summary

### Current Situation

**Deep ITM Test (PutImmediateExerciseAtBoundary)**:
- Parameters: S=0.25, K=100, T=0.75, σ=0.2, r=0.05
- Intrinsic value: $99.75
- Grid: sinh_spaced(-7.0, 2.0, 301, 2.0), 1500 time steps
- **Requires**: Tight tolerance (≤1e-10) to prevent +$16 drift
- **Result with tight tol**: ✅ PASS (value ≈ intrinsic)

**Moderate ITM Test (PutValueRespectsIntrinsicBound)**:
- Parameters: S=100, K=110, T=1.0, σ=0.25, r=0.03
- Intrinsic value: $10.00
- Expected: value ≥ $9.999999 (intrinsic - 1e-6)
- **Result with tight tol**: ❌ FAIL (value = $9.9936, locked too early)
- **Result with loose tol**: ✅ PASS

**ATM Call Test (CallValueIncreasesWithVolatility)**:
- Parameters: S=100, K=100, T=1.0, r=0.01, σ ∈ {0.15, 0.25, 0.4}
- Expected: Value increases with volatility
- **Result with tight tol**: ❌ FAIL (all values = $0, locked to zero payoff)
- **Result with loose tol**: ✅ PASS

## Approaches Tried

### 1. Fixed Absolute Tolerance
```cpp
constexpr double active_tol = 1e-10;
if (std::abs(u[i] - psi[i]) < active_tol || u[i] < psi[i]) {
    // Lock node
}
```
**Result**: Deep ITM passes, other tests fail (locks nodes with legitimate time value)

### 2. Adaptive Relative Tolerance
```cpp
const double rel_tol = 1e-8;
const double abs_tol = 1e-10;
const double tol = abs_tol + rel_tol * std::abs(psi[i]);
```
**Result**: Same failure pattern (tolerance still too tight for small payoffs)

### 3. Warmup Period (Delayed Activation)
```cpp
constexpr size_t warmup_iters = 1;  // or 2, 3
if (obstacle_ && iter >= warmup_iters) {
    // Apply active set
}
```
**Result**:
- warmup=3: Other tests pass, deep ITM fails (+$16 drift)
- warmup=2: Other tests pass, deep ITM fails (+$16 drift)
- warmup=1: Other tests pass, deep ITM fails (+$16 drift)

### 4. Residual-Based Criterion
```cpp
const bool near_obstacle = (u[i] <= psi[i] || (u[i] - psi[i]) < position_tol);
const bool pde_pulls_away = (residual_[i] < -1e-6);
if (near_obstacle && pde_pulls_away) {
    // Lock node
}
```
**Result**: Deep ITM passes, other tests fail (criterion still too aggressive)

## Root Cause Analysis

The fundamental challenge is distinguishing between two scenarios:

### Exercise Region (Should Lock)
- Deep ITM where early exercise is optimal
- PDE wants value above intrinsic (European-like ~$116)
- But American constraint requires u = ψ
- **Need**: Active set locks nodes to payoff

### Continuation Region (Should NOT Lock)
- Moderate ITM or ATM options
- Have legitimate time value above intrinsic (small premium)
- PDE correctly wants u > ψ
- **Need**: Nodes remain free to develop time value

**The Problem**: Both regions can have:
- Small distance from payoff (u - ψ < 1e-3)
- Negative residual (PDE wants to increase u)
- Similar Newton iteration behavior

## Alternative Approaches to Consider

### 1. Semi-Smooth Newton Method
Treat the complementarity condition directly:
```
min(u - ψ, L(u)) = 0
```
Use specialized Newton variant for non-smooth systems.

### 2. Penalty Method
Add penalty term to residual:
```cpp
residual[i] += penalty_coeff * min(0, u[i] - psi[i])^2
```
Gradually increase penalty coefficient across iterations.

### 3. Projected SOR (PSOR)
Replace Thomas solver with SOR iteration that projects after each update:
```cpp
u[i] = max(psi[i], SOR_update(u, i))
```

### 4. Primal-Dual Active Set
Track active set explicitly based on complementarity residual:
```cpp
active_set[i] = (u[i] - psi[i] < tol && constraint_residual > 0)
```

### 5. Grid-Dependent Tolerance
Use tighter tolerance on dense grid regions (where deep ITM lives):
```cpp
const double local_dx = grid[i+1] - grid[i];
const double tol = base_tol * sqrt(local_dx);
```

## Current Implementation

File: `src/pde/core/pde_solver.hpp:560-588`

The active set method is applied during Newton iteration by:
1. Saving original Jacobian before iterations
2. Each iteration: restore Jacobian, then lock active nodes
3. For locked nodes: set `jacobian_diag[i]=1`, zero off-diagonals, set `residual[i] = u[i] - psi[i]`

## Next Steps

Need to either:
1. Find a smarter tolerance criterion that distinguishes exercise vs continuation regions
2. Implement one of the alternative approaches listed above
3. Consult literature on obstacle problems for American option PDEs

## References

- User's explanation in previous session about incomplete complementarity enforcement
- Deep ITM issue documented in Issue #196
- Expert review: `docs/deep_itm_boundary_issue.md`
