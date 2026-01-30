<!-- SPDX-License-Identifier: MIT -->
# Validation Framework Bug: 3-Point "Ground Truth"

## Issue #61 Root Cause

The adaptive refinement tests were failing with constant 1134bp errors despite:
- ✅ Interpolation working perfectly (diagnostic test shows ~1e-4 error)
- ✅ Grid expansion working correctly
- ✅ Context recreation working correctly

## The Smoking Gun

`src/validation.c:199-206`:

```c
// Create fine moneyness grid around sample point
double m_grid[3] = {
    s->moneyness * 0.95,
    s->moneyness,
    s->moneyness * 1.05
};

AmericanOptionResult fdm_result = american_option_solve(
    &option_data, m_grid, 3, grid_params->dt, grid_params->n_steps);
```

**Validation solves FDM on only 3 spatial points to generate "ground truth"!**

## Impact

`american_option_solve()` line 386-392:
```c
SpatialGrid grid = {
    .x_min = x_grid[0],
    .x_max = x_grid[n_m - 1],
    .n_points = n_m,  // <--- THIS IS 3!!!
    .dx = (x_grid[n_m - 1] - x_grid[0]) / (n_m - 1),
    .x = x_grid
};
```

**Comparison**:
- Price table precomputation: 101 spatial points → accurate FDM
- Validation "ground truth": **3 spatial points** → wildly inaccurate FDM

## Why This Explains Everything

1. **Constant errors (1134.7 ± 0.1 bp)**:
   - 3-point FDM is systematically wrong by ~11% IV
   - Error is constant because grid spacing is proportional to moneyness

2. **Grid refinement doesn't help**:
   - Interpolation quality irrelevant when comparing against garbage reference
   - No matter how good interpolation gets, it's compared to wrong answer

3. **Errors enormous (11% IV difference)**:
   - 3 spatial points cannot capture option pricing curvature
   - Second-order PDE on 3 points → terrible discretization error

## Original Design Intent (Likely)

The comment says "Create fine moneyness grid" suggesting intent was:
- Create **dense grid around the sample point**
- Use `grid_params` to specify **resolution of that grid**

But implementation only uses 3 points regardless of `grid_params->n_points`.

## Fix Options

### Option 1: Use grid_params->n_points for validation FDM

```c
// Create fine moneyness grid around sample point
size_t n_validation = grid_params->n_points;  // Use same resolution as table!
double *m_grid = malloc(n_validation * sizeof(double));

double m_min = s->moneyness * 0.9;  // Wider range
double m_max = s->moneyness * 1.1;

for (size_t i = 0; i < n_validation; i++) {
    double frac = (double)i / (n_validation - 1);
    m_grid[i] = m_min + frac * (m_max - m_min);
}

AmericanOptionResult fdm_result = american_option_solve(
    &option_data, m_grid, n_validation, grid_params->dt, grid_params->n_steps);

// Extract price at center via interpolation
double price_fdm = pde_solver_interpolate(fdm_result.solver, s->moneyness);
```

**Pros**: Accurate ground truth
**Cons**: **VERY SLOW** - solving 101-point FDM for each validation sample (100 samples × 4 minutes each = 400 minutes!)

### Option 2: Precompute reference table at higher resolution

```c
// During validation setup:
// 1. Create reference table with 2× denser grid than test table
// 2. Precompute all prices on reference table
// 3. During validation: compare test table interpolation to reference table interpolation

// This tests interpolation quality, not FDM accuracy
```

**Pros**: Fast, tests what we care about (interpolation)
**Cons**: Doesn't test absolute FDM accuracy

### Option 3: Hybrid approach

```c
// For adaptive refinement:
// - Use Option 2 (fast, tests interpolation quality)
//
// For absolute accuracy testing:
// - Use Option 1 on small sample (10 points, not 100)
// - Document as "slow accuracy test"
```

## Recommended Solution

**Immediate**: Use Option 2 for adaptive refinement tests
- Fast enough to run in CI
- Tests interpolation quality (the actual goal)
- Reference table can be 2-3× denser

**Later**: Add Option 3 for comprehensive accuracy validation
- Separate test suite
- Small sample size
- Compare against QuantLib as well

## Test Results After Fix

Expected behavior with Option 2:
- Iteration 1 (10 points): P95 ~50-200 bp (coarse grid)
- Iteration 2 (16 points): P95 ~10-50 bp (improving)
- Iteration 3 (28 points): P95 ~1-10 bp (near target)

**Monotonic improvement** proving adaptive refinement works.
