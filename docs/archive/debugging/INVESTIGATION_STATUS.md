<!-- SPDX-License-Identifier: MIT -->
# Cox-de Boor SIMD Investigation Status

**Date:** 2025-01-16
**Branch:** feature/cox-de-boor-simd
**Status:** ⚠️ INCOMPLETE - Methodology issues identified

---

## Summary of Codex Reviews

After two rounds of peer review by Codex subagent, the investigation has **critical methodology flaws** that must be addressed before any conclusions can be drawn.

---

## What We Know (Verified)

### ✅ Isolated SIMD Performance (Measured)

From `tests/cox_de_boor_profile_test.cc`:
```
SIMD per evaluation:   66.3 ns
Scalar per evaluation: 151.1 ns
SIMD speedup:          2.28×
```

**Status:** ✗ **BELOW 2.5× target** (Phase 2 goal not met)

---

## What We DON'T Know (Needs Direct Measurement)

### ❌ Cox-de Boor Percentage of Total Runtime

- **Original claim:** 7.5% of runtime
- **Problem:** Derived from estimates (96,000 calls × 66ns ÷ 90ms), not measured
- **Need:** Direct instrumentation or profiling of actual pipeline

### ❌ Actual Runtime Breakdown

- **Original claim:** Banded solver is 92.5% of runtime
- **Problem:** This is just "remaining time" after subtracting estimate, not measured
- **Need:** Scoped timers or sampling profiler (perf, VTune) on real workload

### ❌ End-to-End Speedup with SIMD

- **Original claim:** ~1.0× end-to-end speedup
- **Problem:** Measurements vary (81ms, 90ms) without clear SIMD on/off comparison
- **Need:** Controlled A/B test with SIMD enabled vs disabled

---

## Codex Review Findings

### Round 1: Initial Investigation (Commit 10e83bf)

**Issues Identified:**
1. Speedup claim (2.69×) inconsistent with test data (2.33×)
2. Cox-de Boor percentage (7.5%) derived, not measured
3. No instrumentation of matrix construction or banded solver
4. Tests use cout, not EXPECT assertions (can't catch regressions)
5. Amdahl's law calculations based on unverified inputs

**Verdict:** CHANGES REQUESTED

### Round 2: Corrected Investigation (Commit 11a4c9d)

**Issues Identified:**
1. Still using derived estimates in tests (no direct measurement)
2. Per-call instrumentation has 110ns overhead vs 70ns work (invalid)
3. Created instrumented tests but they measure NOTHING from real pipeline
4. Tests still have no EXPECT assertions or regression guards
5. Documentation says "per-call timing is invalid" but code still uses it

**Verdict:** CHANGES STILL REQUESTED

**Quote from Codex:**
> "Because the instrumentation wrapper still records per-call durations with `high_resolution_clock`, every measurement and assertion in that file is dominated by measurement overhead—the test now enforces impossible thresholds like `<100 ns` while the instrumentation itself costs more than that."

---

## Methodology Problems

### Problem 1: Synthetic Estimates vs Real Measurements

**What I did:**
```cpp
// Estimate Cox-de Boor time
double cox_de_boor_ms = 96000 * 66.3e-9 * 1000;  // CALCULATION
double percentage = cox_de_boor_ms / 90.0 * 100;  // DERIVED
```

**What Codex wants:**
```cpp
// Measure Cox-de Boor time WITHIN actual pipeline
auto start = high_resolution_clock::now();
for (/* thousands of calls in batch */) {
    cubic_basis_nonuniform_simd(...);
}
auto end = high_resolution_clock::now();
// Actual measurement, not calculation
```

### Problem 2: Per-Call Timing Overhead

**What I did:**
```cpp
auto start = high_resolution_clock::now();  // ~50ns
cubic_basis_nonuniform_simd(...);            // ~70ns work
auto end = high_resolution_clock::now();    // ~50ns
// Overhead (100ns) > Work (70ns) = INVALID
```

**What Codex wants:**
- Batch timing (1000+ calls, divide by count)
- Sampling profiler (perf, VTune)
- Zero-overhead USDT probes (already in codebase)

### Problem 3: No Machine-Checkable Assertions

**What I did:**
```cpp
std::cout << "Cox-de Boor: " << percentage << "%\n";  // No assertion
```

**What Codex wants:**
```cpp
EXPECT_LT(percentage, 15.0) << "Cox-de Boor should be < 15% of total";
// CI will catch regressions
```

---

## Why This Matters

### Phase 2 Target: 1.14× End-to-End Speedup

**Based on assumptions:**
- Cox-de Boor is 20% of runtime
- SIMD achieves 2.5× speedup
- Amdahl: 1 / (0.80 + 0.20/2.5) = 1.14×

**Actual (so far):**
- SIMD achieved 2.28× speedup ✗ (below 2.5× target)
- Cox-de Boor percentage unknown (needs measurement)
- End-to-end speedup ~1.0× (approximate)

**Implication:**
- Even if Cox-de Boor is 20% of runtime
- With 2.28× SIMD speedup
- Theoretical max: 1 / (0.80 + 0.20/2.28) = **1.11×** (below 1.14× target)

**Phase 2 did NOT meet its performance target.**

---

## Recommended Next Steps

### Step 1: Use Proper Profiling Tools

**Option A: Sampling Profiler**
```bash
# Build optimized test
bazel build -c opt //tests:bspline_4d_end_to_end_performance_test

# Profile with perf
sudo perf record -g --call-graph=dwarf \
    ./bazel-bin/tests/bspline_4d_end_to_end_performance_test \
    --gtest_filter="*SIMDSpeedupRegression"

# Get report
sudo perf report --stdio | grep -E "cubic_basis|banded_lu"
```

**Benefits:**
- Zero overhead (sampling)
- Shows actual time distribution
- Identifies real bottlenecks

**Option B: USDT Probes**
```cpp
// Already in codebase! Use existing MANGO_TRACE macros
MANGO_TRACE_ALGO_START(MODULE_BSPLINE, ...);
// ... work ...
MANGO_TRACE_ALGO_COMPLETE(MODULE_BSPLINE, duration_ns);
```

**Benefits:**
- Zero overhead when not tracing
- Can enable in production
- Already integrated

**Option C: Scoped Timers in Production Code**
```cpp
// Add to BSplineFitter4DSeparable::fit_axis()
#ifdef ENABLE_PROFILING
    auto cox_de_boor_timer = ScopedTimer("cox_de_boor");
#endif
```

**Benefits:**
- Direct measurement in context
- Can be compile-time disabled
- Accurate batched timing

### Step 2: Understand Why SIMD is Only 2.28×

**Investigate:**
- Why below 2.5× target?
- Is std::experimental::simd overhead the issue?
- Would hand-written AVX2 intrinsics be faster?
- Is [[gnu::target_clones]] dispatch overhead significant?

**Method:**
- Check assembly output (`objdump -d`)
- Profile with perf (cache misses, branch mispredicts)
- Compare to theoretical peak (4 doubles × SIMD width)

### Step 3: Make Evidence-Based Decision

**Once we have real data:**

**Scenario A: Cox-de Boor is < 10% of runtime**
- Accept 2.28× SIMD speedup (theoretical max ~1.05× end-to-end)
- Document why 1.14× target was unreachable (wrong bottleneck assumption)
- Recommend Phase 2 complete despite target miss

**Scenario B: Cox-de Boor is 15-20% of runtime**
- Investigate why SIMD only 2.28× (not 2.5×)
- If fixable: improve SIMD implementation
- If not fixable: accept 2.28× and update targets

**Scenario C: Banded solver dominates > 85%**
- Phase 2 optimization was correct implementation, wrong target
- Future work: optimize banded solver (if ROI justifies it)
- Accept Phase 2, update documentation with lessons learned

---

## Current Recommendation

**DO NOT MERGE Phase 2 yet.**

**Reasons:**
1. Performance target (2.5× SIMD speedup) not met (achieved 2.28×)
2. No verified measurements of Cox-de Boor overhead in pipeline
3. No evidence-based explanation for why target was missed
4. Investigation methodology has critical flaws (per Codex reviews)

**Next Actions:**
1. Profile with perf/USDT to get real runtime breakdown
2. Understand why SIMD is 2.28× instead of 2.5×
3. Recalculate Amdahl's law with measured (not derived) inputs
4. Make evidence-based decision on accepting vs improving

---

## Lessons for Future Optimization Work

1. **Profile BEFORE optimizing** - Don't assume bottlenecks
2. **Measure, don't calculate** - Direct instrumentation, not math
3. **Verify targets before claiming success** - 2.28× ≠ 2.5×
4. **Use proper profiling tools** - perf/VTune, not per-call timing
5. **Peer review methodology** - Not just results
6. **Machine-checkable tests** - EXPECT assertions, not cout

---

## Files Status

### Tests Created (Methodology Flawed)
- ❌ `tests/cox_de_boor_profile_test.cc` - Uses estimates, no assertions
- ❌ `tests/bspline_runtime_breakdown_test.cc` - Derived percentages
- ❌ `tests/bspline_instrumented_profile_test.cc` - Invalid per-call timing

### Documentation (Conclusions Premature)
- ❌ `docs/debugging/cox-de-boor-simd-investigation.md` - Based on flawed data
- ✓ `docs/debugging/cox-de-boor-simd-investigation-corrected.md` - Acknowledges issues
- ✓ `docs/debugging/INVESTIGATION_STATUS.md` - This file

### Commits
- 10e83bf - Initial investigation (Codex: CHANGES REQUESTED)
- 11a4c9d - Corrected measurements (Codex: CHANGES STILL REQUESTED)

---

## Conclusion

**Investigation is INCOMPLETE.**

Codex correctly identified that I'm still using synthetic estimates instead of real measurements. The entire bottleneck analysis rests on calculations (96,000 calls × 66ns ÷ 90ms = 7.5%), not direct instrumentation of the actual pipeline.

**Before proceeding:**
- Profile with perf to get real runtime breakdown
- Measure Cox-de Boor overhead IN CONTEXT (not isolated)
- Understand why SIMD is 2.28× (below 2.5× target)
- Make decision based on data, not assumptions

**Status:** Awaiting proper profiling and measurement.
