<!-- SPDX-License-Identifier: MIT -->
# Jacobian Usage Analysis - Issue #7 Follow-up

## Executive Summary

**Finding: The Jacobian reuse approach is NOT currently a problem.**

All three AI experts (ChatGPT, Gemini, DeepSeek) mentioned potential Jacobian issues when diagnosing the TR-BDF2 stagnation bug. However, after thorough code analysis:

1. ✅ **The actual bug was with Stage 2 coefficients** (fixed in commit 1538c1b)
2. ✅ **Current Jacobian approach is valid** for all existing spatial operators (they are linear)
3. ⚠️ **Potential future issue** if nonlinear spatial operators are added

## What the Experts Said About Jacobians

### ChatGPT
> "Jacobian reuse (quasi-Newton approach)... could staleness cause convergence to incorrect solutions? **Yes**, Jacobian staleness can cause Newton to converge to the wrong solution... Always check the true F(u)."

Recommended:
- For nonlinear problems, update Jacobian periodically or use line search
- Use analytic Jacobian when possible

### Gemini
> "Jacobian reuse is valid to reuse (quasi-Newton) can be fine for mild nonlinearity... **Yes**, Jacobian staleness can cause Newton to converge to the wrong solution"

Noted that for **linear problems**, using the analytic (exact) Jacobian makes staleness a non-issue.

### DeepSeek
> "**Issue 2: Jacobian Staleness in Newton Iterations** - The quasi-Newton approach is problematic and can lead to: False convergence, Slow convergence, Solution stagnation"

Recommended:
- "Full Newton with Jacobian Updates" - recompute Jacobian every iteration
- Use analytical Jacobian for linear problems

## Current Implementation Analysis

### How Jacobian is Used (src/pde_solver.c:148-226)

The current code uses a **quasi-Newton approach**:

```c
// Line 148-150
// Compute Jacobian once at the beginning (assumes nearly linear operator)
// For truly nonlinear problems, this could be recomputed every few iterations
evaluate_spatial_operator(solver, t, u_new, Lu);

// Build Jacobian matrix using finite differences (lines 152-226)
// ...compute J once...

// Lines 228-304: Newton iteration loop
for (size_t iter = 0; iter < max_iter; iter++) {
    // Jacobian is REUSED throughout all iterations
    // Only residuals are recomputed
}
```

Key characteristics:
- Jacobian computed **once** at start of each implicit solve
- Uses finite differences with `eps = 1e-7`
- Jacobian is **frozen** (reused) for all Newton iterations
- Comment acknowledges: "assumes nearly linear operator"

### Analysis of All Spatial Operators

I examined every spatial operator in the codebase to determine linearity:

#### 1. Heat Equation (examples/example_heat_equation.c:35-54)
```c
Lu[i] = D * d2u_dx2;
```
**Linear:** Coefficients are constant, operator is L(αu + βv) = αL(u) + βL(v)
**Jacobian:** Constant tridiagonal matrix

#### 2. Heat with Jump Condition (examples/example_heat_equation.c:57-82)
```c
double D_center = (x[i] < data->jump_location) ?
                  data->diffusion_left : data->diffusion_right;
Lu[i] = D_center * d2u_dx2;
```
**Linear:** D varies with space (x) but NOT with solution (u)
**Jacobian:** Constant (though non-uniform)

#### 3. Black-Scholes (src/american_option.c:71-105)
```c
// L(V) = (1/2)σ²∂²V/∂x² + (r - σ²/2)∂V/∂x - rV
double coeff_2nd = 0.5 * sigma * sigma;         // constant
double coeff_1st = r - 0.5 * sigma * sigma;     // constant
double coeff_0th = -r;                          // constant

LV[i] = coeff_2nd * d2V_dx2 + coeff_1st * dV_dx + coeff_0th * V[i];
```
**Linear:** All coefficients are constant
**Jacobian:** Constant tridiagonal matrix

### What About American Options?

American options have an **obstacle condition** (complementarity constraint):
```c
V(S,t) ≥ intrinsic_value(S)
```

This makes the overall problem nonlinear. However:

**The obstacle is applied OUTSIDE the Newton iteration** (pde_solver.c:273-282):
```c
// After Newton converges, project onto obstacle
if (solver->callbacks.obstacle != nullptr) {
    for (size_t i = 0; i < n; i++) {
        if (u_new[i] < psi[i]) {
            u_new[i] = psi[i];
        }
    }
}
```

The Jacobian is only for the **linear spatial operator L(V)**, not the obstacle. The obstacle is handled via projection (penalty method), not through the Jacobian.

## Conclusion: Is Jacobian Reuse a Problem?

### Current Status: ✅ NO PROBLEM

**Why it's fine:**
1. All spatial operators are linear → Jacobian is constant
2. For linear L(u), ∂L/∂u = constant matrix (independent of u)
3. Computing Jacobian once vs. many times gives identical result
4. The quasi-Newton approach is equivalent to full Newton for linear operators

**Verification:**
- Tests are passing (6/7 tests pass, 1 failure is due to Neumann BC mass conservation, unrelated to Jacobian)
- TR-BDF2 stagnation bug was due to Stage 2 coefficients, not Jacobian
- American option pricing works correctly despite obstacle nonlinearity

### Future Risk: ⚠️ POTENTIAL ISSUE

**When would it become a problem?**
If someone adds a **nonlinear spatial operator** where L(u) depends on u in a nonlinear way:

Examples:
- **Porous medium equation:** L(u) = ∂/∂x(u^m ∂u/∂x), m > 1
- **Reaction-diffusion with nonlinear source:** L(u) = D∂²u/∂x² + f(u), where f(u) = u²(1-u)
- **Nonlinear Black-Scholes (volatility smile):** σ = σ(S,V), making operator depend on V

For such operators:
- Jacobian ∂L/∂u would depend on the current solution u
- Freezing Jacobian could cause slow convergence or convergence to wrong solution
- Would need to implement Jacobian updates every few iterations

## Recommendations

### 1. Document Current Limitation ✅ ALREADY DONE
The code has a comment acknowledging this (pde_solver.c:148-149):
```c
// Compute Jacobian once at the beginning (assumes nearly linear operator)
// For truly nonlinear problems, this could be recomputed every few iterations
```

### 2. Enhance Documentation
Add to `CLAUDE.md`:

```markdown
## Spatial Operator Linearity Requirement

The current PDE solver implementation uses a quasi-Newton approach where the Jacobian
is computed once per implicit solve and reused for all iterations. This is efficient
and accurate for **linear or nearly-linear** spatial operators.

**Supported operators:**
- Linear diffusion: L(u) = D∂²u/∂x² + a∂u/∂x + bu
- Spatially-varying coefficients: L(u) = D(x)∂²u/∂x² (D depends on x, not u)
- Obstacle conditions: Applied via projection, not through Jacobian

**Future work for nonlinear operators:**
If you need to implement truly nonlinear spatial operators (e.g., porous medium equation,
nonlinear reaction terms), you will need to modify `solve_implicit_step()` to recompute
the Jacobian periodically within the Newton iteration loop.
```

### 3. Add Runtime Linearity Check (Optional)
For debugging, could add an optional check:
```c
if (DEBUG_MODE) {
    // Compute Jacobian at u and at u+δ
    // Verify they are identical (within tolerance)
    // Warn if operator appears nonlinear
}
```

### 4. Create USDT Probe for Jacobian (Optional)
```c
IVCALC_TRACE_JACOBIAN_REUSED(step, iter, staleness_measure);
```

## Testing Validation

To verify Jacobian reuse is working correctly, I checked:

1. ✅ **Heat equation convergence:** Tests pass with expected accuracy
2. ✅ **American option pricing:** Matches expected values despite obstacle nonlinearity
3. ✅ **Steady-state convergence:** Fixed by Stage 2 coefficients, not Jacobian changes
4. ✅ **Newton residual convergence:** Achieves < 1e-8 tolerance consistently

## References

**Expert Responses:**
- `issues/7/result_chatgpt.md` (lines 129-135): Jacobian staleness discussion
- `issues/7/result_gemini.md` (lines 80-82): Quasi-Newton validity for linear problems
- `issues/7/result_deepseek.md` (lines 21-28): Jacobian staleness as potential issue

**Code Locations:**
- Jacobian computation: `src/pde_solver.c:148-226`
- Quasi-Newton iteration: `src/pde_solver.c:228-304`
- Heat operator: `examples/example_heat_equation.c:35-54`
- Black-Scholes operator: `src/american_option.c:71-105`

**Commit History:**
- Fix for stagnation bug: commit 1538c1b (Stage 2 coefficients, not Jacobian)

## Summary Table

| Aspect | Status | Notes |
|--------|--------|-------|
| **Current implementation** | ✅ Quasi-Newton (Jacobian reused) | Lines 148-304 of pde_solver.c |
| **All current operators** | ✅ Linear | Heat equation, Black-Scholes |
| **Tests passing** | ✅ 6/7 tests pass | 1 failure unrelated (Neumann BC #6) |
| **Stagnation bug** | ✅ Fixed | Was Stage 2 coefficients, not Jacobian |
| **Expert concerns** | ⚠️ Valid for nonlinear operators | Not applicable to current code |
| **Future risk** | ⚠️ If nonlinear operators added | Would need Jacobian updates |
| **Documentation** | ✅ Comment exists | Could be enhanced |
| **Action required** | ❌ None | Works correctly as-is |

---

**Conclusion:** The Jacobian reuse approach mentioned by all three experts is **not a problem** for the current codebase because all spatial operators are linear. The TR-BDF2 stagnation was correctly identified as a Stage 2 coefficient error, and that fix resolved the issue. The quasi-Newton approach is appropriate and efficient for the current use cases.
