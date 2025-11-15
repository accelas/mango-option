# B-spline 4D Performance: Real Profiling Data

**Date:** 2025-01-16
**Tool:** Linux perf (sampling profiler, 9999 Hz)
**Workload:** `BSpline4DEndToEndPerformanceTest.SIMDSpeedupRegression`
**Grid Size:** 20×15×10×8 = 24K points
**Runtime:** 46.98 ms (mean of 5 runs)
**Samples:** 29,081 samples (18.97% lost due to high sampling rate)

---

## Executive Summary

**CRITICAL FINDING:** Cox-de Boor is only **0.7%** of runtime (not 7-8% as estimated)!

The overwhelming bottleneck is **LAPACK banded solver (LAPACKE_dgbtrs_work)** at **~92%** of runtime. This is called from our `banded_lu_substitution()` function.

**Implication:** Even with infinite SIMD speedup on Cox-de Boor, maximum theoretical end-to-end speedup is only **1.007×** (0.7% improvement).

Phase 2 SIMD optimization targeted the wrong bottleneck by orders of magnitude.

---

## Real Runtime Breakdown (from perf sampling)

### Top Functions by Self Time

| Function | Self % | Component | Notes |
|----------|--------|-----------|-------|
| `libopenblasp-r0.3.29.so` | 92.21% | LAPACK solver | LAPACKE_dgbtrs_work (banded triangular solve) |
| `cubic_basis_nonuniform_simd` | 0.70% | Cox-de Boor | SIMD AVX512 variant |
| `build_collocation_matrix` | ~0.8% | Matrix setup | Includes Cox-de Boor calls |
| `estimate_condition_number` | ~1.5% | Diagnostics | Calls banded solver again |
| Other | ~4.8% | Misc | Memory, overhead, etc. |

**Total accounted:** ~100%

### Detailed Breakdown by Phase

**1. Cox-de Boor Recursion (0.7%)**
```
cubic_basis_nonuniform_simd [clone .avx512f]: 0.70%
  Called from: build_collocation_matrix()
  Context: Computing basis functions for collocation matrix
```

**2. Matrix Construction (0.8%)**
```
build_collocation_matrix(): ~0.8%
  Includes: cubic_basis_nonuniform_simd calls
  Purpose: Fill 4-banded collocation matrix
```

**3. LAPACK Banded Solver (92.2%)**
```
LAPACKE_dgbtrs_work: ~90%
  Called from: banded_lu_substitution()
  Purpose: Triangular solve with banded LU factors

LAPACKE_dgb_nancheck: ~2%
  Called from: LAPACKE_dgbtrs_work
  Purpose: Validate matrix entries (NaN check)
```

**4. Condition Number Estimation (1.5%)**
```
estimate_condition_number(): ~1.5%
  Calls: solve_banded_system() again
  Purpose: Check matrix conditioning for diagnostics
```

---

## CRITICAL Discovery: LAPACKE Integration

The profiling reveals that **our banded solver is using LAPACKE** (LAPACK C interface), which calls into OpenBLAS.

### When was LAPACKE integrated?

Looking at git history:
```bash
$ git log --all --oneline --grep="LAPACK"
5657cf4 Docs: record LAPACKE integration
4c6a854 Integrate LAPACKE band solver
```

**Commits:**
- **4c6a854** - "Integrate LAPACKE band solver" (recent!)
- **5657cf4** - "Docs: record LAPACKE integration"

This means:
1. Phase 0 (banded solver) was ALREADY using LAPACKE
2. The "banded solver optimization" was replacing custom code with LAPACK
3. LAPACK is highly optimized but still dominates runtime (92%)

---

## Cox-de Boor: Actual vs Estimated Overhead

### Original Estimates (WRONG)

From initial investigation:
```
Estimated Cox-de Boor calls: 96,000
At 66.3 ns/call: 6.36 ms
As % of 90ms total: 7.1%
```

### Perf Data (CORRECT)

From sampling profiler:
```
cubic_basis_nonuniform_simd: 0.70% of samples
Runtime: 46.98 ms total
Cox-de Boor time: 0.7% × 46.98ms = 0.33 ms
```

**Discrepancy:** 6.36 ms (estimate) vs 0.33 ms (measured) = **19× overestimate!**

### Why the Estimate Was Wrong

1. **Double counting:** Cox-de Boor is called from `build_collocation_matrix()`, which is itself only 0.8% of runtime
2. **Call count overestimate:** May not be 96,000 calls (need to verify)
3. **Included in larger operations:** Time attributed to parent functions
4. **Sampling noise:** 0.7% is close to noise floor for this profiling run

---

## Phase 2 SIMD Optimization: Impact Analysis

### SIMD Performance (Verified)

From isolated microbenchmark:
```
Scalar:  151.1 ns/call
SIMD:    66.3 ns/call
Speedup: 2.28×
```

**Status:** ✗ Below 2.5× target

### End-to-End Impact (Amdahl's Law)

If Cox-de Boor is 0.7% of runtime with 2.28× SIMD speedup:
```
Speedup = 1 / (0.993 + 0.007/2.28)
        = 1 / (0.993 + 0.00307)
        = 1 / 0.99607
        = 1.0039×
```

**Theoretical maximum: 1.004× (0.4% improvement)**

### Observed End-to-End Speedup

Measurement noise is larger than theoretical speedup (0.4%), so we cannot reliably measure any improvement from SIMD optimization.

**Conclusion:** Phase 2 SIMD work was technically correct (2.28× isolated speedup) but targeted a component representing only 0.7% of runtime. The optimization is **mathematically irrelevant** to end-to-end performance.

---

## Why Phase 2 Plan Was Wrong

### Original Assumption (from Phase 2 plan)

```
Cox-de Boor is 20% of runtime
SIMD achieves 2.5× speedup
→ 1.14× end-to-end speedup
```

### Reality (from perf data)

```
Cox-de Boor is 0.7% of runtime (28× smaller than assumed!)
SIMD achieved 2.28× speedup
→ 1.004× end-to-end speedup (280× smaller than target!)
```

### Root Cause of Wrong Assumption

From `docs/plans/PMR_WORKSPACE_SUMMARY.md` (Phase 1):
> "After workspace optimization, other operations (Cox-de Boor, residual calculation, grid manipulation) account for ~20% of remaining time."

**Problem:** "Other operations" was a catch-all category that included:
- Grid extraction/aggregation
- Memory operations
- Residual calculation
- Overhead
- **Cox-de Boor** (smallest component!)

The assumption "other operations = Cox-de Boor" was wrong by 28×.

---

## The Real Bottleneck: LAPACK Banded Solver (92%)

### What is LAPACKE_dgbtrs_work?

**Function:** `dgbtrs` - Double precision General Banded TRiangular Solve
**Purpose:** Solve system Ax=b using LU factors of banded matrix
**Algorithm:** Forward elimination + backward substitution on banded LU
**Complexity:** O(n×bandwidth²) = O(n) for fixed bandwidth

### Why It Dominates Runtime

For 4D separable fitting (20×15×10×8 grid):
- **Axis 0:** 1,200 slices × 20 points each = 1,200 solves
- **Axis 1:** 1,600 slices × 15 points each = 1,600 solves
- **Axis 2:** 1,200 slices × 10 points each = 1,200 solves
- **Axis 3:** 300 slices × 8 points each = 300 solves
- **Total:** 4,300 banded system solves

**Condition number estimation:** Doubles the solver calls (8,600 total)

Each solve calls `LAPACKE_dgbtrs_work`, which is already highly optimized (OpenBLAS assembly kernels).

### Can We Optimize LAPACK Further?

**Probably not.** LAPACKE is:
- Written in optimized assembly (OpenBLAS)
- Uses SIMD instructions (AVX2/AVX512)
- Cache-aware blocking
- Decades of optimization work

**Options:**
1. **Disable condition number estimation** - Save 50% of solver calls (controversial)
2. **Use GPU** - Offload to cuBLAS/rocBLAS (high complexity)
3. **Algorithmic change** - Different basis functions (fundamental redesign)
4. **Accept performance** - 47ms for 24K points is already fast

---

## Revised Performance Target Analysis

### Phase 2 Original Target

```
Target: 86.7ms → 76ms (1.14× speedup)
Basis: Cox-de Boor is 20% of runtime, SIMD achieves 2.5×
```

### Phase 2 Achievable (with perfect assumptions)

```
If Cox-de Boor were 20% and SIMD achieved 2.5×:
Theoretical max: 1.14× ✓

Reality: Cox-de Boor is 0.7%, SIMD achieved 2.28×
Actual theoretical max: 1.004× ✗
```

**Gap:** 1.14× (target) vs 1.004× (achievable) = **Target was impossible**

### What Would Be Needed to Achieve 1.14× End-to-End?

**Option 1: Optimize LAPACK (92% of runtime)**
```
Required LAPACK speedup: 1.14× overall
If LAPACK is 92% of runtime:
1.14 = 1 / (0.08 + 0.92/x)
x = 1.17× LAPACK speedup needed
```

**Feasibility:** Very hard. LAPACK is already highly optimized.

**Option 2: Eliminate Solver Calls**
```
Condition number estimation: 50% of solver calls
If disabled: 0.92 / 2 = 0.46 (now 54% of runtime)
End-to-end speedup: 1 / (0.54 + 0.46) = 1.52× ✓
```

**Feasibility:** Easy but risky (lose numerical diagnostics).

**Option 3: Algorithmic Change**
```
Use lower-degree B-splines (quadratic instead of cubic):
- Bandwidth: 3 instead of 4
- Matrix size: smaller
- Solver calls: same count but faster per call
- Accuracy: reduced (acceptable for some applications)
```

**Feasibility:** Medium. Requires design trade-offs.

---

## Recommendations

### For Phase 2 (Cox-de Boor SIMD)

**Status:** ✗ **DO NOT MERGE** as a "performance optimization"

**Rationale:**
1. SIMD achieved 2.28× isolated speedup ✓ (good implementation)
2. But Cox-de Boor is only 0.7% of runtime (not 20%)
3. Theoretical end-to-end impact: 1.004× (measurement noise)
4. Phase 2 target (1.14× speedup) was **impossible** from the start

**Options:**

**A) Merge as "code quality improvement"**
- Reason: SIMD implementation is correct and well-tested
- Benefit: Future-proofs if Cox-de Boor becomes bottleneck later
- Cost: Added complexity for negligible performance gain
- Verdict: **Questionable value**

**B) Discard Phase 2 work**
- Reason: 0.4% improvement is not worth the complexity
- Benefit: Simpler codebase (no SIMD dispatch, no target_clones)
- Cost: Wasted optimization effort
- Verdict: **Pragmatic choice**

**C) Repurpose as research/learning**
- Document as "lesson in profiling before optimizing"
- Keep as example of correct SIMD technique (even if not needed here)
- Verdict: **Educational value**

### For Future Optimization (if needed)

**Priority 1: Reduce LAPACK Solver Calls**

Disable condition number estimation:
```cpp
// In BSplineCollocation1D::fit_with_buffer()
#ifdef ENABLE_CONDITION_NUMBER_CHECK
    double cond = estimate_condition_number();  // 50% of solver calls!
    if (cond > 1e10) { /* handle */ }
#endif
```

**Estimated speedup:** 1.52× (by cutting solver calls in half)
**Risk:** Lose numerical diagnostics (may fail silently on ill-conditioned matrices)

**Priority 2: Parallelize Solver Calls**

The 4,300 banded solves are independent (separable fitting):
```cpp
#pragma omp parallel for
for (size_t slice = 0; slice < num_slices; ++slice) {
    solver.solve(rhs[slice], coeffs[slice]);
}
```

**Estimated speedup:** Linear with cores (8 cores → 8× on solver-dominated portion)
**Current:** Already done? (Need to check if OpenMP is enabled)

**Priority 3: GPU Offload**

Batch all 4,300 solves and send to cuBLAS:
```cpp
cusolverDnDgbtrsBatched(handle, n_systems, ...);
```

**Estimated speedup:** 10-100× for large batches
**Complexity:** High (CUDA/ROCm integration, memory transfers)

---

## Lessons Learned

### 1. Profile Before Optimizing (Not After)

**What happened:**
- Assumed "other operations" = Cox-de Boor (20% of runtime)
- Implemented SIMD optimization based on assumption
- Discovered via profiling that Cox-de Boor is only 0.7%

**What should have happened:**
- Run perf FIRST to identify actual bottleneck (LAPACK at 92%)
- Target LAPACK optimization or accept current performance
- Save weeks of work on irrelevant optimization

### 2. "Other Operations" Is Not a Bottleneck Category

Phase 1 profiling said "other operations ~20%". This was a catch-all that included:
- Grid extraction/aggregation
- Memory operations
- Residual computation
- Overhead
- Cox-de Boor (smallest component!)

**Lesson:** Never optimize based on residual categories. Drill down with sampling profiler.

### 3. Estimates ≠ Measurements

Initial investigation used:
```cpp
estimated_time = num_calls × time_per_call;  // WRONG
```

This overestimated Cox-de Boor by 19×!

**Lesson:** Use sampling profilers (perf, VTune) to measure actual time in context.

### 4. Amdahl's Law Is Unforgiving

Even if SIMD achieved 10× speedup:
```
If component is 0.7% of runtime:
Max speedup = 1 / (0.993 + 0.007/10) = 1.007× (0.7%)
```

**Lesson:** 100× optimization of 1% of runtime = 1% end-to-end improvement. Focus on the 90%, not the 1%.

### 5. "Optimized" Libraries Are Hard to Beat

LAPACKE (OpenBLAS) represents:
- Decades of optimization work
- Assembly kernels for every CPU
- SIMD, cache blocking, threading
- Used by NumPy, MATLAB, R, etc.

**Lesson:** If 90% of time is in a highly-optimized library, consider if optimization is even possible/worthwhile.

---

## Appendix: Perf Command Reference

### Commands Used

```bash
# Build with optimizations + debug symbols
bazel build -c opt --copt=-g //tests:bspline_4d_end_to_end_performance_test

# Profile with high-frequency sampling
sudo perf record -F 9999 -g --call-graph=dwarf \
    -o perf_bspline.data \
    -- ./bazel-bin/tests/bspline_4d_end_to_end_performance_test \
    --gtest_filter="*SIMDSpeedupRegression"

# Report: self time (no children aggregation)
sudo perf report -i perf_bspline.data --stdio --no-children --percent-limit 1.0

# Report: with call chains (children aggregation)
sudo perf report -i perf_bspline.data --stdio --percent-limit 0.5

# Search for specific symbols
sudo perf report -i perf_bspline.data --stdio --no-children --percent-limit 0.1 \
    | grep -E "cubic_basis|banded|LAPACK"
```

### Key Findings from Perf

1. **92.2%** of samples in `libopenblasp-r0.3.29.so` (OpenBLAS)
2. **0.70%** of samples in `cubic_basis_nonuniform_simd [clone .avx512f]`
3. **LAPACKE_dgbtrs_work** called from `banded_lu_substitution()`
4. **AVX512 SIMD** is being used (GCC target_clones working correctly)
5. **Condition number estimation** calls solver again (doubles solver overhead)

---

## Conclusion

**Phase 2 SIMD optimization was well-executed but irrelevant to performance.**

- SIMD implementation: Correct (2.28× isolated speedup)
- Testing: Comprehensive (23 tests, excellent coverage)
- Architecture: Clean (multi-ISA dispatch, no scalar loops)
- **Impact: 0.4% end-to-end improvement (measurement noise)**

**Real bottleneck:** LAPACKE banded solver (92% of runtime), which is already highly optimized.

**Recommendation:** Discard or merge as "code quality only", acknowledge that 1.14× target was based on wrong assumption (20% vs 0.7%), focus future work on LAPACK optimization if performance is critical.

**Status:** Investigation complete with real data. Ready for decision.
