<!-- SPDX-License-Identifier: MIT -->
# Current Implementation Analysis

**Date**: 2025-01-14
**File**: `src/interpolation/bspline_fitter_4d.hpp`
**Class**: `BSplineCollocation1D`

## Matrix Storage

### Compact Banded Storage
- **Band values**: `band_values_[i * 4 + k]` where k ∈ [0,3]
  - Type: `std::vector<double>` with size `n × 4` (row-major layout)
  - Each row stores exactly 4 non-zero entries from the collocation matrix
  - Location: Line 185

- **Column starts**: `band_col_start_[i]` indicates first non-zero column for row i
  - Type: `std::vector<int>` with size `n`
  - Computed as `max(0, span - 3)` where `span` is the knot span
  - Location: Line 186, set at Line 201

### Matrix Type
**4-diagonal banded matrix** (confirmed)
- Cubic B-splines have degree 3, which means each basis function has compact support spanning 4 knot intervals
- At any grid point x_i, exactly 4 basis functions are non-zero: N_{j-3}, N_{j-2}, N_{j-1}, N_j
- This creates a banded structure where row i has non-zero entries in columns [col_start, col_start+3]

### Matrix Construction
- **Method**: `build_collocation_matrix()` at lines 189-214
- **Basis evaluation**: Uses `cubic_basis_nonuniform()` from `bspline_utils.hpp`
- **Knot vector**: Clamped cubic knots created by `clamped_knots_cubic(grid_)`
- **Band filling**: Loop over 4 basis functions, store in compact format

## Boundary Conditions

### Implicit Natural Boundary Conditions
- Clamped knot vector with multiplicity 4 at endpoints
- Knots created by `clamped_knots_cubic()` function
- This enforces interpolation at boundaries (not natural splines)
- No explicit boundary condition application in solver

### Knot Vector Structure
For grid with n points:
- Total knots: n + 4 (degree 3 + 1)
- Left clamp: knots[0:4] all equal to grid[0]
- Interior: knots[4:n] match interior grid points
- Right clamp: knots[n:n+4] all equal to grid[n-1]

## Current Solver Implementation

### Bottleneck Confirmed ✅

**Dense allocation**: Line 221
```cpp
std::vector<double> A(n_ * n_, 0.0);
```
- Allocates full n×n matrix despite only 4n non-zero entries
- Memory waste: O(n²) vs optimal O(n)

**Expansion to dense**: Lines 224-232
```cpp
for (size_t i = 0; i < n_; ++i) {
    int j_start = band_col_start_[i];
    int j_end = std::min(j_start + 4, static_cast<int>(n_));

    for (int j = j_start; j < j_end; ++j) {
        int band_idx = j - j_start;
        A[i * n_ + j] = band_values_[i * 4 + band_idx];
    }
}
```

**Solver**: Lines 234-288 - Gaussian elimination with partial pivoting
- Algorithm: Full Gaussian elimination with partial pivoting
- Time complexity: O(n³) due to dense matrix operations
- Pivoting: Lines 237-261 - searches all rows below current pivot
- Elimination: Lines 263-274 - updates entire matrix rows
- Back substitution: Lines 276-285

### Why This Is Inefficient

1. **Memory overhead**: Stores n² doubles instead of 4n
   - For n=100: 10,000 doubles (80 KB) vs 400 doubles (3.2 KB) - 25× waste
   - For n=500: 250,000 doubles (2 MB) vs 2,000 doubles (16 KB) - 125× waste

2. **Computational waste**:
   - Pivoting searches all rows: O(n²) comparisons total
   - Elimination updates full rows: O(n³) operations
   - Most operations on zeros: ~96% of matrix is zero for n=100

3. **Cache inefficiency**:
   - Row-major dense storage means column operations (pivoting) access non-contiguous memory
   - Poor cache locality during elimination phase

## Residual Computation (Already Efficient)

**Method**: `compute_residual()` at lines 291-312
- **Correctly uses banded storage**: Only loops over 4 non-zero entries per row
- Time complexity: O(n) ✅
- Uses `std::fma()` for accuracy
- This shows the team knows how to exploit band structure!

## Verification: 4-Diagonal Structure

### From Cubic Basis Functions
Examined `cubic_basis_nonuniform()` in `bspline_utils.hpp`:
- Evaluates exactly 4 basis functions: N[0], N[1], N[2], N[3]
- Uses Cox-de Boor recursion: degree 0 → 1 → 2 → 3
- Compact support: only 4 consecutive basis functions overlap at any x

### Matrix Bandwidth Calculation
- Cubic B-spline (degree p=3) has support span of p+1 = 4 intervals
- Each row i of collocation matrix B has non-zero entries B[i, j] for j ∈ [j_start, j_start+3]
- Lower bandwidth: 3 (entries below diagonal)
- Upper bandwidth: 0 to 3 (varies by row, depends on span)
- **Total diagonals**: 4 per row (main + 3 below, or shifted pattern)

## Opportunity Summary

### What Can Be Improved

1. **Replace dense expansion** (Line 221):
   - Keep band_values_ in compact form
   - Operate directly on 4-diagonal structure

2. **Use banded LU solver**:
   - Time: O(n³) → O(n) for fixed bandwidth
   - Space: O(n²) → O(n)
   - No pivoting needed for SPD matrices from B-spline collocation

3. **Expected speedup**:
   - Per-solve: 5-10× for n=100 (micro-benchmark)
   - End-to-end: 1.47× for 50×30×20×10 grid (design doc estimate)

### What Works Well

1. **Compact storage**: band_values_ already stores only 4n entries ✅
2. **Residual computation**: Already uses banded structure ✅
3. **Matrix construction**: Efficient basis evaluation ✅

## Implementation Strategy

### Phase 0 Plan (from design doc)
1. Create `BandedMatrixStorage` class for clean API
2. Implement `banded_lu_solve()` using Doolittle algorithm
3. Replace `solve_banded_system()` to use banded solver
4. Regression test: verify identical results to dense solver (FP precision)
5. Benchmark: confirm target speedup

### Risk Assessment
- **Low risk**: Banded structure is already validated (residual computation proves it)
- **Easy regression testing**: Can keep dense solver temporarily for comparison
- **No API changes**: Internal optimization only

## Files to Modify

1. **Primary**: `src/interpolation/bspline_fitter_4d.hpp`
   - Add `BandedMatrixStorage` class (before BSplineCollocation1D)
   - Add `banded_lu_solve()` function
   - Modify `BSplineCollocation1D::solve_banded_system()` implementation

2. **Tests**: Create `tests/bspline_banded_solver_test.cc`
   - Unit tests for banded storage
   - Unit tests for banded LU solve
   - Regression test vs dense solver
   - Integration test with BSplineCollocation1D

3. **Benchmarks**: Create `benchmarks/bspline_banded_solver_benchmark.cc`
   - Compare banded vs dense for n ∈ {50, 100, 200, 500}
   - Measure end-to-end speedup on realistic 4D grid

## Next Steps

Proceed to **Task 2**: Write banded solver test baseline
