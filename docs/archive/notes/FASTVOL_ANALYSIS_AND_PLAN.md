# FastVol Analysis and Optimization Plan for mango-iv

## Executive Summary

After analyzing the fastvol repository (https://github.com/vgalanti/fastvol), I've identified several CPU optimization techniques that could significantly improve mango-iv's performance. Fastvol achieves impressive speeds through memory optimization, algorithmic improvements, and careful SIMD utilization.

---

## FastVol's Key CPU Optimization Techniques

### 1. Memory Layout and Alignment

**What They Do:**
- **64-byte alignment** for all arrays: `alignas(64)` for AVX-512 compatibility
- **Stack allocation first**: Up to 2048 steps on stack, heap only for larger grids
- **Dual-array buffers**: Eliminates full tree storage (O(n²) → O(n))
- **Even-odd splitting**: Separate `pe[]` and `po[]` arrays for SIMD-friendly sequential access
- **Explicit alignment hints**: `__builtin_assume_aligned` for compiler optimization

**Performance Impact:**
- Avoids malloc overhead for typical cases
- Enables automatic vectorization (2-8x speedup on inner loops)
- Better cache locality

### 2. OpenMP SIMD Pragmas

**What They Do:**
```cpp
#pragma omp simd
for (size_t i = 1; i < n - 1; i++) {
    // vectorizable operations
}
```

**Applied to:**
- Payoff initialization
- Backward induction loops
- Boundary condition evaluation
- Root-finding bound generation

**Performance Impact:**
- 2-8x parallel execution per iteration (4-16x for fp32)
- Low register pressure enables CPU interleaving

### 3. Fused Multiply-Add (FMA)

**What They Do:**
```cpp
fm::fma(a, b, c)  // computes a*b + c in single instruction
```

**Benefits:**
- Single CPU instruction instead of two
- Higher accuracy (no intermediate rounding)
- Essential for vectorization efficiency

### 4. Restrict Pointers

**What They Do:**
```cpp
void compute(T* __restrict__ output, const T* __restrict__ input)
```

**Benefits:**
- Tells compiler pointers don't alias
- Enables aggressive loop optimization
- Allows more vectorization opportunities

### 5. Precomputation and Lookup Tables

**What They Do:**
- **Tree method**: Precompute all payoff values (2n+1 entries)
- **PDE method**: Precompute constant matrix coefficients
- **Risk-neutral probability**: Computed once outside loops

**Performance Impact:**
- 80%+ reduction in inner loop computation
- Eliminates redundant `pow()` calls (expensive)

### 6. Red-Black Grid Splitting (PDE Solver)

**What They Do:**
- Split spatial grid into "red" (even indices) and "black" (odd indices)
- Update red points using black values, then vice versa
- Separate buffers for red/black instead of strided access

**Performance Impact:**
- "As few iterations as PSOR, as fast as Jacobi"
- Full 2x speedup over naive PSOR
- Eliminates loop-carried dependencies for vectorization

### 7. Adaptive Relaxation Parameter (ω)

**What They Do:**
- Start with theoretical optimal: `ω_opt = 2 / (1 + √(1 - ρ(B)²))`
- Adjust between timesteps based on convergence behavior
- Reverse direction when iterations increase
- Narrow search range when improving

**Performance Impact:**
- Converges "within the 30 or so first timesteps"
- Prevents "10x runtime balloon" from poor ω selection
- Critical for PDE solver efficiency

### 8. Log-Space Transformation (PDE)

**What They Do:**
- Transform `S → x = log(S)` in Black-Scholes PDE
- Results in **constant matrix coefficients**
- No need to recompute matrix for different strikes/spots

**Performance Impact:**
- Enables pre-computation of tridiagonal matrix
- Better memory access patterns
- More SIMD-friendly operations

### 9. Batch Processing with OpenMP

**What They Do:**
```cpp
#pragma omp parallel for
for (size_t i = 0; i < batch_size; i++) {
    // price option i
}
```

**Performance Impact:**
- Multi-core utilization
- Wall time dramatically reduced (e.g., 2.5ms scalar → 40.8µs wall for 4096 steps)

### 10. Implied Volatility Optimization

**What They Do:**
- **Two-stage approach**: Coarse approximation + refined root-finding
- **Adaptive bracket expansion**: Multipliers 0.8/1.2 within bounds [1e-4, 20.0]
- **Dual convergence criteria**: Price-space AND volatility-space tolerance
- **Templated solvers**: Newton, Brent, Bisection for different scenarios

**Performance Impact:**
- European IV: 252.8 ns (batch, fp64)
- Robust convergence with minimal iterations

---

## Performance Comparison: FastVol vs mango-iv

### FastVol Benchmarks (CPU, fp64)

**American Options (BOPM):**
- 256 steps: 199.7 ns (batch)
- 1024 steps: not listed, interpolated ~5-10 µs
- 4096 steps: 40.8 µs (batch)

**American Options (PSOR):**
- 128x256 grid: 5.4 µs (batch)
- 512x1024 grid: 139.7 µs (batch)

### Our Implementation (from benchmark)

**American Options (PDE, 1000 steps, 400 spatial points):**
- Put: 21.7 ms
- Call: 22.5 ms

### Speed Ratio Analysis

FastVol's 512x1024 PSOR (comparable resolution):
- **139.7 µs vs our 21.7 ms**
- **~155x faster** 🚨

Even accounting for batch vs scalar differences:
- Our scalar time would be comparable to their scalar
- But they achieve **60x faster batch processing through OpenMP parallel for**

---

## Optimization Plan for mango-iv

Based on fastvol analysis, here's a prioritized plan with estimated impact:

### Phase 1: Memory Optimization (High Impact, Low Risk)

**1.1 Add 64-byte Alignment**
- Change: `alignas(64)` for all workspace arrays
- Add: `__builtin_assume_aligned` hints
- **Estimated Impact**: +10-20% speedup
- **Effort**: Low (1 day)
- **Risk**: None

**1.2 Split Even-Odd Arrays**
- Change: Current arrays access with stride-2 pattern
- New: Separate even/odd buffers for sequential access
- **Estimated Impact**: +20-30% speedup (enables vectorization)
- **Effort**: Medium (2-3 days)
- **Risk**: Low (similar to BC refactoring, but affects hot path)

**1.3 Stack vs Heap Allocation**
- Change: Add stack-based path for typical grid sizes
- Threshold: n_points ≤ 1024 (uses ~80KB stack)
- **Estimated Impact**: +5-10% for typical cases
- **Effort**: Low (1 day)
- **Risk**: Low

### Phase 2: Algorithmic Improvements (High Impact, Medium Risk)

**2.1 Red-Black PSOR Method**
- Change: Replace current implicit solver with Red-Black PSOR
- Benefits: Better vectorization, fewer iterations
- **Estimated Impact**: +50-100% speedup (2-3x faster)
- **Effort**: High (1-2 weeks)
- **Risk**: Medium (requires thorough testing)

**2.2 Adaptive Relaxation Parameter**
- Change: Current fixed relaxation → adaptive ω tuning
- Benefits: Faster convergence, fewer iterations
- **Estimated Impact**: +20-40% speedup
- **Effort**: Medium (3-5 days)
- **Risk**: Low (can fall back to fixed ω)

**2.3 Log-Space Transformation**
- Change: Transform to x = log(S) for constant coefficients
- Benefits: Precompute matrix, better memory access
- **Estimated Impact**: +15-25% speedup
- **Effort**: High (1-2 weeks)
- **Risk**: High (changes fundamental discretization)

### Phase 3: FMA and Restrict (Medium Impact, Low Risk)

**3.1 Fused Multiply-Add**
- Change: Replace `a * b + c` with `fma(a, b, c)`
- Apply to: All hot loop arithmetic
- **Estimated Impact**: +5-10% speedup + accuracy
- **Effort**: Low (1-2 days)
- **Risk**: None

**3.2 Restrict Pointers**
- Change: Add `__restrict__` to function parameters
- Benefits: Better compiler optimization
- **Estimated Impact**: +5-10% speedup
- **Effort**: Low (1 day)
- **Risk**: None (assuming no actual aliasing)

### Phase 4: Batch Processing (High Impact, Low Risk)

**4.1 OpenMP Batch API**
- Add: `american_option_price_batch()` function
- Use: `#pragma omp parallel for` for multiple options
- **Estimated Impact**: ~60x speedup for batch workloads
- **Effort**: Low (2-3 days)
- **Risk**: None (additive feature)

**4.2 Vectorized IV Calculation**
- Change: Process multiple IV calculations in parallel
- Use: OpenMP + SIMD for batch IV solving
- **Estimated Impact**: 10-50x for batch IV
- **Effort**: Medium (1 week)
- **Risk**: Low

### Phase 5: Advanced Optimizations (Medium Impact, High Effort)

**5.1 SLEEF Integration**
- Add: SLEEF library for vectorized math functions
- Replace: std::exp, std::log, etc.
- **Estimated Impact**: +10-20% speedup
- **Effort**: Medium (1 week + dependency management)
- **Risk**: Medium (new dependency)

**5.2 Profile-Guided Optimization (PGO)**
- Use: GCC/Clang PGO for hot path optimization
- Benefits: Better inlining, branch prediction
- **Estimated Impact**: +10-15% speedup
- **Effort**: Low (build system changes)
- **Risk**: Low

**5.3 Alternative Pricing Methods**
- Add: BOPM (tree method) as alternative to PDE
- Benefits: Different convergence characteristics
- **Estimated Impact**: Complementary (user choice)
- **Effort**: High (2-3 weeks)
- **Risk**: Low (additive feature)

---

## Recommended Implementation Roadmap

### Milestone 1: Quick Wins (1-2 weeks)
**Target: 1.5-2x speedup**

1. ✅ Add 64-byte alignment and hints
2. ✅ Add FMA operations
3. ✅ Add restrict pointers
4. ✅ Stack allocation for small grids
5. ✅ Add batch processing API

**Expected Result**: 10-15ms per option (from 21.7ms)

### Milestone 2: Memory Layout Refactoring (2-3 weeks)
**Target: Additional 1.3-1.5x speedup**

1. ✅ Split even-odd arrays
2. ✅ Optimize memory access patterns
3. ✅ Benchmark and validate

**Expected Result**: 7-10ms per option

### Milestone 3: Solver Improvements (3-4 weeks)
**Target: Additional 2-3x speedup**

1. ✅ Implement Red-Black PSOR
2. ✅ Add adaptive relaxation parameter
3. ✅ Extensive testing and validation

**Expected Result**: 3-5ms per option

### Milestone 4: Advanced Features (4-6 weeks)
**Target: Batch speedup and alternative methods**

1. ✅ Optimize batch processing
2. ✅ Consider log-space transformation
3. ✅ Evaluate BOPM implementation
4. ✅ PGO and SLEEF integration

**Expected Result**: <200µs per option in batch mode

---

## Risk Assessment

### Low Risk, High Reward
- ✅ 64-byte alignment
- ✅ FMA operations
- ✅ Restrict pointers
- ✅ Batch API (additive)
- ✅ Stack allocation

**Recommendation**: Start here immediately

### Medium Risk, High Reward
- ⚠️ Even-odd array splitting
- ⚠️ Adaptive relaxation
- ⚠️ Red-Black PSOR

**Recommendation**: Implement with comprehensive testing and benchmarking

### High Risk, High Reward
- ⚠️⚠️ Log-space transformation
- ⚠️⚠️ Complete solver replacement

**Recommendation**: Evaluate in separate experimental branches (like BC refactoring)

---

## Key Lessons from FastVol

### 1. Memory Layout Matters More Than Algorithms
FastVol's biggest wins come from:
- Sequential memory access (even-odd split)
- Alignment for SIMD
- Precomputation to reduce memory traffic

**Not** from fancy algorithms - they use standard methods (BOPM, PSOR, Brent's).

### 2. Batch Processing is Essential
Their wall-time benchmarks show **60x+ speedup** from OpenMP parallel for.
- Scalar: 2.5 ms
- Wall (batch): 40.8 µs

**Takeaway**: Single-option pricing is not the right benchmark. Real workloads need batches.

### 3. SIMD Requires Careful Data Layout
You can't just add `#pragma omp simd` and expect speedup. You need:
- Sequential memory access
- Low register pressure
- No loop-carried dependencies
- Aligned data

**Takeaway**: Our even-odd split or Red-Black approach enables this.

### 4. Convergence Optimization > Raw Speed
FastVol's adaptive ω prevents "10x runtime balloon".
Their Red-Black PSOR has "as few iterations as PSOR".

**Takeaway**: Algorithmic convergence improvements compound with per-iteration speedups.

### 5. Stack Allocation for Common Cases
They avoid malloc for grids up to 2048 steps.

**Takeaway**: 1000 timesteps × 400 spatial points = 400K doubles = 3.2MB. Too big for stack, but could stack-allocate temporaries.

---

## Specific Code Recommendations

### Immediate Changes (This Week)

**1. Add to pde_solver.c:**
```c
#include <fenv.h>
#pragma STDC FENV_ACCESS ON

// Add to workspace allocation
posix_memalign((void**)&solver->workspace, 64, total_size * sizeof(double));

// Add to array usage
double *u = __builtin_assume_aligned(solver->u_current, 64);
```

**2. Replace operations:**
```c
// Before:
Lu[i] = D * (u[i-1] - 2.0*u[i] + u[i+1]) / (dx*dx);

// After:
const double coeff = D / (dx * dx);
Lu[i] = fma(u[i-1] + u[i+1], coeff, -2.0 * coeff * u[i]);
```

**3. Add restrict:**
```c
void evaluate_spatial_operator(PDESolver *solver, double t,
                               const double * __restrict__ u,
                               double * __restrict__ Lu);
```

### Medium-Term Changes (Next Month)

**4. Implement Red-Black PSOR** based on fastvol/american/psor.hpp:
- Split grid into red/black
- Update red using black, then black using red
- Eliminates stride-2 access pattern
- Enables full vectorization

**5. Add batch API:**
```c
int pde_solver_solve_batch(PDESolver **solvers, size_t n_solvers);
```

**6. Adaptive relaxation:**
- Track iterations per timestep
- Adjust ω when iterations increase
- Narrow range when improving

---

## Expected Final Performance

### Conservative Estimates

| Milestone | Speedup | Time per Option | vs Original |
|-----------|---------|-----------------|-------------|
| Current | 1.0x | 21.7 ms | baseline |
| M1 (Quick wins) | 1.5-2x | 11-14 ms | 1.5-2x |
| M2 (Memory) | 2-3x | 7-11 ms | 2-3x |
| M3 (Solver) | 4-7x | 3-5 ms | 4-7x |
| M4 (Batch) | 100-200x | <200 µs | **100-200x** |

### Aggressive Estimates (All Optimizations)

**Single option**: 2-3 ms (~10x faster than current)
**Batch (64 options)**: <200 µs wall time (~100x faster than current)

This would bring us much closer to fastvol's performance levels (~140 µs for comparable resolution).

---

## Comparison to QuantLib

Current benchmark:
- QuantLib: 10.4 ms
- mango-iv: 21.7 ms
- Ratio: 2.1x slower

After optimizations:
- QuantLib: 10.4 ms
- mango-iv (optimized): 2-3 ms
- Ratio: **3-5x FASTER than QuantLib** 🎯

---

## Conclusion

FastVol demonstrates that **100-200x speedup is achievable** through:
1. Memory layout optimization (enabling SIMD)
2. Batch processing (OpenMP parallelism)
3. Algorithmic improvements (Red-Black PSOR, adaptive ω)
4. Careful precomputation and cache-friendly access

The most impactful changes are:
1. **Batch API** (60x+ speedup) - Low effort, high reward
2. **Red-Black PSOR** (2-3x) - Medium effort, high reward
3. **Memory alignment + FMA** (1.5-2x) - Low effort, high reward

**Recommendation**: Implement in phases, benchmarking after each milestone to validate improvements and catch regressions early.

The BC refactoring experiment taught us to measure performance impact. Let's apply that discipline to these optimizations, starting with the low-risk, high-reward changes.
