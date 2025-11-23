# Cox-de Boor SIMD Removal Rationale

**Date:** 2025-01-16
**Branch:** feature/dgbcon-condition-estimation
**Decision:** Remove SIMD implementation of Cox-de Boor basis functions

---

## Summary

SIMD vectorization of Cox-de Boor recursion was attempted in Phase 2 (PR #169) but subsequently removed to reduce maintenance burden. While the SIMD implementation was technically correct and achieved 2.28× isolated speedup, **perf profiling revealed Cox-de Boor represents only 0.7% of total runtime**, making the optimization irrelevant to end-to-end performance.

---

## SIMD Implementation Details

### What Was Implemented

**Files Created:**
- `src/interpolation/bspline_utils.cpp` - SIMD implementations
- `tests/bspline_simd_test.cc` - Comprehensive correctness tests
- `tests/bspline_simd_smoke_test.cc` - Smoke tests

**SIMD Functions:**
```cpp
[[gnu::target_clones("default","avx2","avx512f")]]
void cubic_basis_nonuniform_simd(
    const std::vector<double>& t,
    int i,
    double x,
    double N[4]);

[[gnu::target_clones("default","avx2","avx512f")]]
simd4d cubic_basis_degree0_simd(
    const std::vector<double>& t,
    int i,
    double x);
```

**Technology Stack:**
- `std::experimental::simd` (C++17 parallelism TS)
- `[[gnu::target_clones]]` for multi-ISA dispatch (default, AVX2, AVX512)
- `fixed_size_simd<double, 4>` for 4-way vectorization

### Performance Achieved

**Isolated Microbenchmark:**
- Scalar: 151.1 ns/call
- SIMD: 66.3 ns/call
- **Speedup: 2.28×** ✓

**Test Coverage:**
- 23 comprehensive tests
- Scalar-SIMD equivalence validation (< 1e-14 error)
- Edge cases: boundaries, repeated knots, uniform/non-uniform spacing
- Partition of unity verification

### Quality

The SIMD implementation was:
- ✓ **Correct** - Passed all 23 tests with < 1e-14 error vs scalar
- ✓ **Well-tested** - Comprehensive test suite
- ✓ **Multi-ISA** - Automatic AVX2/AVX512 dispatch
- ✓ **Production-ready** - Used in collocation matrix building

---

## Why It Was Removed

### Perf Profiling Results

**Real bottleneck distribution** (from Linux perf sampling profiler):

| Component | Self % | Notes |
|-----------|--------|-------|
| LAPACKE_dgbtrs_work | 92.21% | Banded triangular solve (LAPACK) |
| cubic_basis_nonuniform_simd | **0.70%** | Cox-de Boor SIMD version |
| build_collocation_matrix | 0.8% | Matrix setup (includes Cox-de Boor) |
| estimate_condition_number | 1.5% | Diagnostics |
| Other | 4.8% | Memory, overhead |

**CRITICAL FINDING:** Cox-de Boor is only **0.7%** of runtime (not 20% as estimated).

### Amdahl's Law Calculation

**Theoretical maximum end-to-end speedup:**
```
With 2.28× SIMD speedup on 0.7% of runtime:
Speedup = 1 / (0.993 + 0.007/2.28) = 1.0039×
```

**Result:** Even with perfect SIMD (∞× speedup), maximum improvement is **1.007×** (0.7%).

### Wrong Bottleneck Assumption

**Phase 2 plan assumed:**
- Cox-de Boor: 20% of runtime
- SIMD: 2.5× speedup
- Predicted: 1.14× end-to-end improvement

**Reality (from perf):**
- Cox-de Boor: **0.7%** of runtime (28× smaller)
- SIMD: 2.28× speedup (slightly below target)
- Actual: **1.004×** end-to-end improvement (280× smaller than predicted)

**Root cause:** "Other operations" in Phase 1 profiling was a catch-all including grid manipulation, memory ops, and Cox-de Boor. The assumption "other operations = Cox-de Boor" was wrong by **28×**.

### Maintenance Burden vs Benefit

**Complexity added:**
- 170 lines SIMD implementation
- 2 test files (400+ lines)
- Separate compilation unit (.cpp) for IFUNC resolver
- Multi-ISA dispatch overhead
- `std::experimental::simd` dependency

**Benefit:**
- 0.4% end-to-end improvement (within measurement noise)

**Conclusion:** Complexity not justified for negligible benefit.

---

## Lessons Learned

### 1. Profile Before Optimizing (Not After)

**What happened:**
- Estimated Cox-de Boor at 20% based on "other operations" residual
- Implemented SIMD optimization
- Profiled with perf → discovered Cox-de Boor is 0.7%

**What should have happened:**
- Profile with perf FIRST
- Identify LAPACKE as 92% bottleneck
- Target LAPACKE or accept current performance
- Save weeks optimizing irrelevant 0.7%

### 2. "Other Operations" Is Not a Bottleneck

Phase 1 profiling: "other operations ~20%" was a catch-all:
- Grid extraction/aggregation
- Memory operations
- Residual computation
- Overhead
- **Cox-de Boor (smallest component!)**

**Lesson:** Never optimize residual categories. Use sampling profiler to drill down.

### 3. Estimates ≠ Measurements

Initial estimate: `96,000 calls × 66ns = 6.36ms (7.1% of 90ms)`
Perf measurement: `0.7% of 47ms = 0.33ms`

**Overestimated by 19×!**

**Lesson:** Synthetic estimates from call counts are unreliable. Use perf/VTune.

### 4. Amdahl's Law Is Unforgiving

```
Even with 10× speedup on 0.7% component:
Max speedup = 1 / (0.993 + 0.007/10) = 1.007× (0.7% improvement)
```

**Lesson:** 100× optimization of 1% = 1% end-to-end improvement. Focus on the 90%.

---

## The Real Optimization: dgbcon

**This PR's actual value:** LAPACKE_dgbcon condition number estimation

**Performance impact:**
- Before: 46.98 ms (with n solver calls for condition estimation)
- After: 30.88 ms (with dgbcon)
- **Speedup: 1.52×** ✓

**Why it works:**
- Targets 92% bottleneck (LAPACKE), not 0.7% (Cox-de Boor)
- Replaces n expensive LAPACKE_dgbtrs calls with single dgbcon
- Preserves numerical diagnostics
- Simple, maintainable

---

## What Was Kept

**Scalar inline implementation remains:**
```cpp
inline void cubic_basis_nonuniform(
    const std::vector<double>& t,
    int i,
    double x,
    double N[4]);
```

**Why:**
- Header-only (no linking required)
- Compiler auto-vectorization with `#pragma omp simd`
- Used in runtime evaluation (different usage pattern than collocation)
- Simpler, no multi-ISA dispatch overhead
- Adequate performance for 0.7% of runtime

---

## References

- **Phase 2 Plan:** `docs/plans/2025-01-16-cox-de-boor-simd-plan.md`
- **Perf Profiling:** `docs/debugging/PERF_ANALYSIS.md`
- **Investigation Status:** `docs/debugging/INVESTIGATION_STATUS.md`
- **Closed PR:** #169 (Cox-de Boor SIMD)
- **This PR:** #170 (dgbcon optimization - 1.52× speedup)

---

## Conclusion

SIMD optimization was well-executed but **targeted the wrong bottleneck by 2 orders of magnitude**. The real value of optimization comes from targeting the dominant cost (LAPACKE at 92%), not minor costs (Cox-de Boor at 0.7%).

**Key takeaway:** Always profile with sampling tools (perf, VTune) before optimizing. Estimates from call counts and residual categories are unreliable and can lead to weeks of wasted optimization effort.
