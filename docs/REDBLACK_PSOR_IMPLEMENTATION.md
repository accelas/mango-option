# Red-Black PSOR Implementation

## Overview

This document describes the Red-Black Projected Successive Over-Relaxation (PSOR) solver implementation for the PDE solver, based on optimizations identified in the FastVol analysis (Phase 2).

## Motivation

From the FastVol analysis, Red-Black PSOR provides:
- **2-3x speedup** over standard implicit solvers
- **Full SIMD vectorization** through elimination of loop-carried dependencies
- **Better cache locality** through sequential memory access patterns
- **"As few iterations as PSOR, as fast as Jacobi"**

## Key Changes

### 1. Data Structure Modifications

Added to `PDESolver` struct:
```c
// Red-Black arrays for split storage
double *u_red;        // Even indices (0, 2, 4, ...)
double *u_black;      // Odd indices (1, 3, 5, ...)
size_t n_red;         // Number of red points
size_t n_black;       // Number of black points

// Adaptive relaxation
double omega;         // Relaxation parameter (typically ~1.8)
size_t last_iter_count;  // For adaptive ω tuning
```

### 2. Memory Layout

**Workspace allocation updated:**
- Original: 10 arrays × n points
- New: 10 arrays × n points + 2 split arrays (n_red + n_black)
- All arrays 64-byte aligned for AVX-512 SIMD
- Total additional memory: ~n doubles (~8KB for typical n=1000)

**Memory layout benefits:**
- Red points stored contiguously → better cache locality
- Black points stored contiguously → better cache locality
- Sequential access enables compiler vectorization

### 3. Algorithm

**Red-Black Gauss-Seidel with SOR:**

```
For each iteration:
  1. Split u into red (even) and black (odd) arrays
  2. Update ALL red points using black neighbors (fully vectorized)
  3. Update ALL black points using updated red neighbors (fully vectorized)
  4. Merge red-black back to standard representation
  5. Apply boundary and obstacle conditions
  6. Check convergence
```

**Key insight:** Red points only depend on black neighbors and vice versa, so:
- All red updates are independent → can be vectorized
- All black updates are independent → can be vectorized
- No loop-carried dependencies within each color

### 4. Implementation Details

**Split function:**
```c
// Splits u[0..n-1] into:
//   u_red[0..n_red-1]  = u[0], u[2], u[4], ...
//   u_black[0..n_black-1] = u[1], u[3], u[5], ...
```

**Merge function:**
```c
// Merges red-black arrays back to standard grid ordering
```

**PSOR update (for diffusion):**
```c
// Implicit equation: u_i - coeff_dt * L(u)_i = rhs_i
// For diffusion: L(u)_i = (u_{i-1} - 2u_i + u_{i+1}) / dx²
// Gauss-Seidel: u_new_i = (rhs_i + coeff*(u_{i-1} + u_{i+1})) / (1 + 2*coeff)
// SOR: u_i = ω * u_new_i + (1-ω) * u_old_i
```

**Relaxation parameter:**
- Initial: ω = 1.8 (near-optimal for Laplacian)
- Future: Adaptive tuning based on convergence behavior (Phase 2.2)

## Performance Expectations

Based on FastVol analysis and our implementation:

| Optimization | Expected Speedup | Notes |
|--------------|------------------|-------|
| Even-odd split | +20-30% | Better cache locality |
| SIMD vectorization | +2-4x | Compiler can vectorize both loops |
| Fewer iterations | +20-40% | Red-Black converges faster than Jacobi |
| **Combined** | **2-3x total** | Realistic expectation |

## Limitations

**Current implementation assumes:**
1. **Diffusion-dominated operator**: The stencil assumes a 3-point diffusion pattern
2. **Constant coefficients**: Works best when PDE coefficients don't vary significantly
3. **1D spatial domain**: Extension to 2D would use different coloring schemes

**Not suitable for:**
- Strongly advection-dominated problems (may not converge)
- Highly nonlinear operators (would need Newton iteration)
- Non-standard stencils (would need generalization)

## Comparison with Old Solver

### Old: Newton Iteration with Tridiagonal Solver
- **Algorithm**: Linearize → build Jacobian → solve tridiagonal system
- **Pros**: Very general, handles nonlinear operators, robust
- **Cons**: Jacobian computation expensive, tridiagonal solver not vectorizable

### New: Red-Black PSOR
- **Algorithm**: Split → update red → update black → merge → repeat
- **Pros**: Fully vectorizable, simple, fewer operations per iteration
- **Cons**: Assumes specific operator structure, may need more iterations for nonlinear problems

**When to use each:**
- Red-Black PSOR: Diffusion PDEs (heat equation, Black-Scholes, etc.)
- Newton iteration: Nonlinear PDEs, complex operators, when robustness > speed

## Testing Strategy

The implementation should be validated with:

1. **Heat equation** (pure diffusion): Should converge quickly
2. **Black-Scholes PDE** (American options): Primary use case
3. **Convergence rate**: Compare iteration counts vs old solver
4. **Accuracy**: Verify solutions match to within tolerance
5. **Performance**: Benchmark against old solver and QuantLib

## Future Improvements (Phase 2.2)

### Adaptive Relaxation Parameter
```c
// Track convergence behavior
if (iterations_increased) {
    omega = omega * 0.95;  // Decrease ω
} else if (iterations_decreased) {
    omega = omega * 1.05;  // Increase ω (up to ~1.95)
}
```

**Expected benefit:** +20-40% additional speedup by optimizing ω per timestep

### Precomputed Coefficients
For constant-coefficient PDEs, precompute:
```c
double *diag_inv;  // 1 / (1 + 2*coeff_dt/dx²)
```

**Expected benefit:** +5-10% by eliminating divisions

## Integration with TR-BDF2

Red-Black PSOR now used in both TR-BDF2 stages:
- **Stage 1**: Trapezoidal rule → implicit solve with Red-Black PSOR
- **Stage 2**: BDF2 → implicit solve with Red-Black PSOR

Both stages benefit from the vectorization and cache locality improvements.

## References

1. FastVol analysis: `docs/FASTVOL_ANALYSIS_AND_PLAN.md`
2. Original paper: Ascher, Ruuth, Wetton (1995) - TR-BDF2 method
3. Red-Black SOR: Classical iterative methods literature

## Code Locations

- Header: `src/pde_solver.h` (lines 132-143)
- Implementation: `src/pde_solver.c`
  - `split_red_black()` (lines 115-129)
  - `merge_red_black()` (lines 132-148)
  - `solve_redblack_psor()` (lines 150-262)
- Usage: `pde_solver_step_internal()` (lines 626-654)
