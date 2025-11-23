# TR-BDF2 Stagnation Bug - RESOLVED ✅

## Summary

The TR-BDF2 time-stepping stagnation bug has been **completely fixed**. The solution evolved after consulting three AI experts (ChatGPT, Gemini, DeepSeek) who all identified the same root cause: incorrect Stage 2 coefficients.

## Test Results

| Status | Before | After |
|--------|--------|-------|
| **Tests Passing** | 5/7 | **6/7** ✅ |
| **pde_solver_test** | FAIL | **PASS** ✅ |
| **SteadyStateConvergence** | Error 0.097 | **PASS** ✅ |
| **american_option_test** | PASS | PASS |
| **Remaining Issues** | Mass conservation (Neumann BC) | Mass conservation (#6) |

## Root Cause Analysis

### Mathematical Problem

The original Stage 2 formulation was **mathematically inconsistent** with non-zero steady states:

At steady state where $u^{n+1} = u^* = u^n = u_s$ and $L(u_s) = 0$:

**Original (wrong) equation:**
$$(1 + 2\alpha) u_s - (1 + \alpha) u_s + \alpha u_s = 0$$
$$(2\alpha) u_s = 0$$

Since $\alpha = 1 - \gamma \approx 0.4142 \neq 0$, this **forces** $u_s = 0$.

But the PDE's true steady state is $u_\infty(0.5) \approx 0.1131$ (non-zero)!

### The Fix

Replaced incorrect formulation with **standard TR-BDF2** from Ascher, Ruuth, Wetton (1995):

**Incorrect Stage 2 (before):**
$$u^{n+1} - \frac{(1-\alpha)\Delta t}{1+2\alpha} L(u^{n+1}) = \frac{(1+\alpha)u^* - \alpha u^n}{1+2\alpha}$$

**Correct Stage 2 (after):**
$$u^{n+1} - \frac{(1-\gamma)\Delta t}{2-\gamma} L(u^{n+1}) = \frac{u^*}{\gamma(2-\gamma)} - \frac{(1-\gamma)^2 u^n}{\gamma(2-\gamma)}$$

### Coefficient Comparison

With $\gamma = 2 - \sqrt{2} \approx 0.5858$:

| Coefficient | Before | After | Purpose |
|-------------|--------|-------|---------|
| RHS coeff for $u^*$ | 0.773 | **1.207** | Weight of Stage 1 result |
| RHS coeff for $u^n$ | 0.227 | **0.207** | Weight of previous step |
| Time coeff | 0.003172 | 0.002929 | (Actually same!) |

### Numerical Impact

**Before (stagnated):**
- Step 1: $u(0.5) = 0.0077$
- Step 20: $u(0.5) = 0.0158$ (stuck)
- Step 150: $u(0.5) = 0.0158$ (still stuck)
- $L(u) = 0.935$ (large, indicating solution should evolve)

**After (converges):**
- Step 1: $u(0.5) = 0.0099$
- Step 20: $u(0.5) = 0.0996$ (advancing!)
- Step 150: $u(0.5) = 0.1131$ (converged!)
- $L(u) \approx 0.000$ (steady state achieved)

**Analytical steady state:** $u_\infty(0.5) = 1 - \cosh(0)/\cosh(0.5) \approx 0.1131$ ✓

## Expert Insights

### Gemini's Key Contribution
Proved the fundamental mathematical inconsistency by showing the original formulation forced $u_s = 0$ at steady state, creating an irreconcilable conflict with the PDE's true steady state.

### ChatGPT's Contribution
Identified that RHS = 0.01284 being smaller than both $u^n$ and $u^*$ was a red flag for coefficient error, and suggested checking the denominator.

### DeepSeek's Contribution
Provided the explicit correct formulation from literature with numerical coefficient values for verification.

## Implementation Changes

**File:** `src/pde_solver.c`

**Lines 387-401:** Replaced Stage 2 RHS computation
```c
// Before (WRONG)
const double alpha = 1.0 - gamma;
const double coeff = (1.0 - alpha) * dt / (1.0 + 2.0 * alpha);
solver->rhs[i] = ((1.0 + alpha) * solver->u_stage[i] - alpha * solver->u_current[i]) /
                (1.0 + 2.0 * alpha);

// After (CORRECT - Ascher, Ruuth, Wetton 1995)
const double one_minus_gamma = 1.0 - gamma;
const double two_minus_gamma = 2.0 - gamma;
const double denom = gamma * two_minus_gamma;
const double coeff = one_minus_gamma * dt / two_minus_gamma;
solver->rhs[i] = solver->u_stage[i] / denom -
                 one_minus_gamma * one_minus_gamma * solver->u_current[i] / denom;
```

**Additional improvements:**
- Pre-initialize `u_stage` with `u_current` (Stage 1 initial guess)
- Pre-initialize `u_next` with `u_stage` (Stage 2 initial guess)

## Lessons Learned

1. **Verify schemes against literature:** Always cross-check numerical method implementations against canonical references, not just derived formulas.

2. **Steady-state analysis is powerful:** Testing whether the numerical scheme reproduces the correct steady state can reveal fundamental consistency issues.

3. **RHS sanity checks matter:** If RHS values fall outside the range of the inputs for an advancing scheme, investigate coefficient errors.

4. **Newton convergence ≠ correctness:** The Newton solver can converge perfectly while solving the wrong equation if coefficients are incorrect.

5. **AI experts are valuable:** Consulting multiple AI models with different training led to quick identification of a subtle mathematical bug.

## References

- Ascher, U. M., Ruuth, S. J., & Wetton, B. T. (1995). *Implicit-explicit methods for time-dependent partial differential equations.* SIAM Journal on Numerical Analysis, 32(3), 797-823.

- Bank, R. E., Coughran Jr, W. M., Fichtner, W., Grosse, E. H., Rose, D. J., & Smith, R. K. (1985). *Transient simulation of silicon devices and circuits.* IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 4(4), 436-451.

## Commit

```
commit 1538c1b
Fix TR-BDF2 Stage 2 coefficients to use standard formulation

Replaced incorrect coefficients with standard TR-BDF2 formulation
from Ascher, Ruuth, and Wetton (1995).

Fixes #7
```

## Next Steps

Only one test remains failing:
- **Issue #6:** Neumann boundary conditions violate mass conservation
  - This is a separate issue with ghost point implementation
  - Does not affect Dirichlet BC problems
  - Documented and tracked

## Acknowledgments

Special thanks to:
- **Gemini** for the mathematical steady-state consistency proof
- **ChatGPT** for the detailed debugging checklist and RHS sanity check
- **DeepSeek** for providing the exact reference formulation with coefficients
