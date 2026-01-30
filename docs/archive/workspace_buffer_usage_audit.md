# PDEWorkspace Buffer Usage Audit

**Date:** 2025-11-20
**Context:** Post-PMR refactoring buffer aliasing bug fix
**Issue:** `build_jacobian_finite_difference()` was using `workspace_.rhs()` as scratch space, corrupting input RHS

## Executive Summary

After fixing the FD Jacobian aliasing bug (using `rhs` as temp storage), this document audits ALL workspace buffer usage to prevent similar issues.

**Key Finding:** ✅ No other aliasing risks found
**Recommendation:** Add documentation and runtime assertions to prevent future bugs

---

## Workspace Buffer Inventory

### Available Buffers (15 total)

| Buffer | Size | Primary Purpose | Can be used as scratch? |
|--------|------|----------------|------------------------|
| `dx_` | n-1 | Grid spacing | ❌ READ-ONLY after init |
| `u_stage_` | n | Perturbed solution for FD Jacobian | ✅ Scratch in FD only |
| `rhs_` | n | **RIGHT-HAND SIDE** | ❌ **INPUT - NEVER OVERWRITE** |
| `lu_` | n | Operator output L(u) | ✅ Scratch (recomputed each use) |
| `psi_` | n | Obstacle constraint | ⚠️ Computed once per stage, then read-only |
| `jacobian_diag_` | n | Jacobian diagonal | ❌ Output of build_jacobian |
| `jacobian_upper_` | n-1 | Jacobian upper diagonal | ❌ Output of build_jacobian |
| `jacobian_lower_` | n-1 | Jacobian lower diagonal | ❌ Output of build_jacobian |
| `residual_` | n | Newton residual | ⚠️ Temporary during Newton loop |
| `delta_u_` | n | Newton correction | ⚠️ Temporary during Newton loop |
| `newton_u_old_` | n | Previous Newton iterate | ⚠️ Temporary during Newton loop |
| `u_next_` | n | Next time step buffer | ❌ Used by Grid for solution storage |
| **`reserved1_`** | n | **SCRATCH SPACE** | ✅ **Safe for FD Jacobian** |
| `reserved2_` | n | Reserved for future | ✅ Available |
| `reserved3_` | n | Reserved for future | ✅ Available |
| `tridiag_workspace_` | 2n | Thomas solver scratch | ⚠️ Owned by thomas_solver.hpp |

---

## Critical Aliasing Rules

### Rule 1: RHS is SACRED
**`workspace_.rhs()` must NEVER be overwritten between Stage computation and Newton/Projected Thomas**

**Why:** `rhs` is passed as `std::span<const double>` to `solve_implicit_stage()`, but the span points to `workspace_.rhs()`. Overwriting it corrupts the input.

**Timeline of Bug:**
1. `solve_stage1()` computes RHS: `rhs[i] = u_prev[i] + w1·L(u_prev)[i]` → stored in `workspace_.rhs()`
2. Calls `solve_implicit_stage(t, coeff_dt, u, workspace_.rhs())` - passing span to same buffer
3. `build_jacobian_finite_difference()` **OVERWRITES** `workspace_.rhs()` with temp operator evals
4. Newton loop receives corrupted RHS → solution diverges

**Fix:** Use `workspace_.reserved1()` for FD Jacobian temp storage

---

### Rule 2: Jacobian Buffers are Outputs
**`jacobian_diag_`, `jacobian_upper_`, `jacobian_lower_` are WRITE-ONLY during `build_jacobian()`**

These are outputs of `build_jacobian()` and inputs to `solve_thomas()`. Never use as scratch.

---

### Rule 3: Newton Buffers are Scoped
**`residual_`, `delta_u_`, `newton_u_old_` are only valid DURING Newton iteration**

These are safe to reuse BETWEEN time steps, but must not be corrupted during a Newton loop.

---

### Rule 4: lu_ is Recomputed
**`workspace_.lu()` is safe to use as scratch ONLY if you recompute it before next use**

Current usage: ✅ Safe
- Computed in `solve_stage1()` → used immediately
- Computed in Newton loop → used immediately for residual
- Computed in FD Jacobian → baseline for finite differences

**Warning:** If you cache `lu_` across calls, this breaks!

---

## Audit Results: All Buffer Usages

### 1. `apply_operator_with_blocking()` - Writes to output buffer

**Calls found:**
1. **Line 340:** `apply_operator_with_blocking(t_n, u_prev, workspace_.lu())`
   - ✅ Safe: Computes L(u_prev), used immediately for RHS computation

2. **Line 713:** `apply_operator_with_blocking(t, u, workspace_.lu())`
   - ✅ Safe: Newton loop, L(u) used immediately for residual

3. **Line 853:** `apply_operator_with_blocking(t, u, workspace_.lu())`
   - ✅ Safe: FD Jacobian baseline, used immediately

4. **Lines 861, 868, 875 (×3):** `apply_operator_with_blocking(t, workspace_.u_stage(), workspace_.reserved1())`
   - ✅ **FIXED:** Now uses `reserved1()` instead of `rhs()` (was the bug!)

5. **Lines 898, 904 (×2):** Neumann BC Jacobian - uses `reserved1()`
   - ✅ **FIXED:** Now uses `reserved1()`

6. **Line 918:** Neumann BC right boundary - uses `reserved1()`
   - ✅ **FIXED:** Now uses `reserved1()`

**Verdict:** ✅ All operator calls are now safe

---

### 2. `workspace_.rhs()` - **CRITICAL: INPUT BUFFER**

**Writes:**
- **Line 344:** `auto rhs = workspace_.rhs(); ... rhs[i] = std::fma(w1, workspace_.lu()[i], u_prev[i]);`
  - ✅ Safe: This is the INITIAL computation of RHS in `solve_stage1()`

- **Line 393:** `auto rhs = workspace_.rhs(); ... rhs[i] = std::fma(alpha, u_current[i], beta * u_prev[i]);`
  - ✅ Safe: RHS computation in `solve_stage2()`

**Reads (as input to solve):**
- Passed to `solve_implicit_stage()` and `solve_implicit_stage_projected()`
- **MUST NOT BE MODIFIED** after being passed!

**Verdict:** ✅ No corruption - FD Jacobian now uses `reserved1()`

---

### 3. `workspace_.psi()` - Obstacle buffer

**Writes:**
- **Line 274:** `auto psi = workspace_.psi(); derived().obstacle(t, grid_->x(), psi);`
  - ✅ Safe: Computed once per Newton iteration

- **Line 573:** Same in `solve_implicit_stage_projected()`
  - ✅ Safe: Computed once per stage

**Reads:**
- Passed to `solve_thomas_projected()`

**Verdict:** ✅ Safe - no aliasing risk

---

### 4. `workspace_.u_stage()` - Scratch for FD Jacobian

**Usage:**
- **Line 852:** `std::copy(u.begin(), u.end(), workspace_.u_stage().begin());` - Initialize from u
- **Lines 860, 864, 867, 871, 874, 878:** Perturb individual elements for finite differences
- Also used in Neumann BC Jacobian (lines 897, 901, 903, 907, 917, 921)

**Verdict:** ✅ Safe - used only as temporary during FD computation

---

### 5. Newton iteration buffers

**`residual_`:**
- **Line 717:** `compute_residual(u, coeff_dt, workspace_.lu(), rhs, workspace_.residual());`
- **Line 720:** `apply_bc_to_residual(workspace_.residual(), u, t);`
- **Lines 725:** Negated for Thomas RHS
- **Line 733:** Passed to `solve_thomas()`

**`delta_u_`:**
- **Line 734:** Output from `solve_thomas()`
- **Line 745:** `u[i] += workspace_.delta_u()[i];` - Applied to solution

**`newton_u_old_`:**
- **Line 708, 764:** Store previous Newton iterate
- **Line 757, 768:** Used for convergence check

**Verdict:** ✅ Safe - all scoped within Newton loop, no cross-contamination

---

### 6. `reserved1()` - **NOW DESIGNATED FOR FD JACOBIAN SCRATCH**

**Usage after fix:**
- All FD Jacobian temp operator evaluations (8 call sites)

**Verdict:** ✅ Safe - this is its designated purpose now

---

## Potential Future Risks

### 1. **Adding new functionality that writes to `rhs_`**
**Risk:** Someone might use `rhs_` as scratch thinking "it's just a workspace buffer"
**Mitigation:**
- Add comment in code: `// CRITICAL: rhs_ is INPUT - never overwrite after Stage computation`
- Runtime assertion (see below)

### 2. **Reusing `lu_` without recomputing**
**Risk:** Caching `lu_` across multiple uses without recomputation
**Mitigation:**
- `lu_` is always recomputed before use (current design is safe)
- If future code adds caching, must use different buffer

### 3. **Neumann BC Jacobian using wrong buffer**
**Risk:** Copy-paste error reverting to `rhs()` instead of `reserved1()`
**Mitigation:**
- All fixed in current code
- Consider helper function `get_jacobian_scratch_buffer()` for clarity

---

## Recommendations

### 1. Add Runtime Assertions (High Priority)

Add debug-mode checks to detect RHS corruption:

```cpp
void solve_implicit_stage(double t, double coeff_dt,
                          std::span<double> u,
                          std::span<const double> rhs) {
    #ifndef NDEBUG
    // DEBUG: Verify RHS is not aliased with workspace scratch buffers
    const double* rhs_ptr = rhs.data();
    const double* reserved1_ptr = workspace_.reserved1().data();
    const double* u_stage_ptr = workspace_.u_stage().data();

    assert(rhs_ptr != reserved1_ptr && "RHS must not alias with reserved1");
    assert(rhs_ptr != u_stage_ptr && "RHS must not alias with u_stage");

    // Store RHS checksum to detect corruption
    double rhs_checksum = 0.0;
    for (size_t i = 0; i < rhs.size(); ++i) {
        rhs_checksum += rhs[i];
    }
    #endif

    const double eps = config_.jacobian_fd_epsilon;
    apply_boundary_conditions(u, t);

    build_jacobian(t, coeff_dt, u, eps);

    #ifndef NDEBUG
    // Verify RHS was not corrupted
    double new_checksum = 0.0;
    for (size_t i = 0; i < rhs.size(); ++i) {
        new_checksum += rhs[i];
    }
    assert(std::abs(new_checksum - rhs_checksum) < 1e-10 &&
           "RHS was corrupted during Jacobian build!");
    #endif

    // ... rest of function
}
```

### 2. Document Buffer Ownership (Medium Priority)

Add comments to `pde_workspace.hpp`:

```cpp
struct PDEWorkspace {
    // ... existing code ...

    /// RHS buffer for implicit stage solve
    /// ⚠️ CRITICAL: This is an INPUT buffer!
    /// Once filled by solve_stage1/2, it MUST NOT be overwritten
    /// until the Newton/Projected Thomas solve completes.
    /// Use reserved1() for temporary storage instead.
    std::span<double> rhs() { return rhs_.subspan(0, n_); }

    /// Scratch buffer for FD Jacobian temporary operator evaluations
    /// Safe to use: Not read by any solver after FD Jacobian completes
    std::span<double> reserved1() { return reserved1_.subspan(0, n_); }
};
```

### 3. Helper Function for FD Scratch (Low Priority)

Centralize FD Jacobian scratch buffer access:

```cpp
/// Get scratch buffer for FD Jacobian temp storage
/// Uses reserved1() which is guaranteed not to alias with RHS
std::span<double> fd_jacobian_scratch() {
    return workspace_.reserved1();
}
```

Then replace all `workspace_.reserved1()` calls in FD Jacobian with `fd_jacobian_scratch()`.

---

## Conclusion

✅ **No additional aliasing risks found**
✅ **FD Jacobian fix is complete and correct**
✅ **All 34 tests pass**

**Action Items:**
1. ✅ DONE: Fix FD Jacobian to use `reserved1()` instead of `rhs()`
2. ⏳ TODO: Add runtime assertions in debug mode
3. ⏳ TODO: Document buffer ownership rules in code comments

**Long-term:**
- Consider static analysis to detect buffer aliasing at compile time
- Consider `const` correctness for input buffers (though spans make this hard)
