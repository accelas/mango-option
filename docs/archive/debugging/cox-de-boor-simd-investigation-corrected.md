# Cox-de Boor SIMD Performance Investigation (Corrected)

**Date:** 2025-01-16
**Investigator:** Claude Code
**Method:** Systematic Debugging with Codex peer review
**Issue:** Phase 2 SIMD optimization delivered 1.0× speedup instead of 1.14× target

---

## Executive Summary

**Root Cause (Corrected after Codex review):**

1. **SIMD achieved 2.28× speedup** (BELOW 2.5× target, not above as initially claimed)
2. **Cox-de Boor percentage unclear** - needs actual pipeline instrumentation to measure
3. **Initial investigation had methodology flaws** - used derived estimates instead of direct measurements

**Status:** Investigation incomplete. Need to:
- Determine why SIMD only achieved 2.28× (not 2.5× target)
- Measure actual Cox-de Boor overhead in pipeline (not estimate)
- Use proper profiling tools (not per-call timing with overhead)

---

## Codex Review Feedback

### Critical Issues Identified

1. **Inconsistent speedup claims**
   - Initial report claimed 2.69× speedup
   - Test data shows only 2.28× speedup
   - **Below 2.5× target**, not above

2. **Derived vs measured percentages**
   - 7.5% figure was calculated from estimates, not measured
   - Hard-coded assumptions (6.78ms, 50ns/eval)
   - No direct measurement of Cox-de Boor time within pipeline

3. **Unverified bottleneck analysis**
   - 92.5% "other operations" is just remainder
   - No instrumentation of matrix construction or banded solver
   - Speculative breakdown without data

4. **No machine-checkable assertions**
   - Tests use cout, not EXPECT/ASSERT
   - Can't catch performance regressions in CI

5. **Amdahl's law depends on unverified inputs**
   - If speedup is 2.28× (not 2.69×)
   - If percentage is different than 7.5%
   - Conclusions change significantly

---

## Corrected Measurements

### Test 1: Isolated SIMD Performance (Verified)

```
SIMD per evaluation:   66.3 ns
Scalar per evaluation: 151.1 ns
SIMD speedup:          2.28×
```

**Status:** ✗ **BELOW 2.5× target**

### Test 2: Pipeline Overhead (Needs Direct Measurement)

Current approach uses estimates:
- Assumes 96,000 Cox-de Boor calls (not counted)
- Multiplies by measured 66.3ns per call
- Compares to total 90ms runtime

**Problem:** This is a calculation, not a measurement. We don't know:
- Actual number of Cox-de Boor calls in pipeline
- Overhead from grid extraction/aggregation
- Time spent in matrix construction vs solver

**Needed:** Direct instrumentation with scoped timers or profiling tools.

---

## Why Instrumentation Failed

### Attempt: Per-Call Timing

```cpp
auto start = std::chrono::high_resolution_clock::now();
cubic_basis_nonuniform_simd(t, i, x, N);  // ~70ns work
auto end = std::chrono::high_resolution_clock::now();  // ~100ns overhead!
```

**Result:** Timing overhead dominates the work being timed.
- SIMD appeared SLOWER (0.94×) due to instrumentation
- Cannot use per-call timing for nanosecond-scale operations

### Better Approach Needed

1. **Scoped timers around large batches**
   - Time 1,000+ calls together, divide by count
   - Reduce overhead percentage

2. **Sampling profiler** (perf, VTune)
   - See actual time distribution
   - No instrumentation overhead

3. **USDT probes** (already in codebase)
   - Zero overhead when not tracing
   - Enable with bpftrace/systemtap

---

## Phase 2 Target Analysis

### Original Target

From `docs/plans/2025-01-16-cox-de-boor-simd-plan.md`:
```
Cox-de Boor optimization alone → 1.14× incremental speedup
Target: 86.7ms → 76ms
```

### Assumptions Made

1. Cox-de Boor is 20% of runtime
2. SIMD will achieve 2.5× speedup
3. Amdahl's law: 1 / (0.80 + 0.20/2.5) = 1.14×

### What We Know Now

1. **SIMD achieved 2.28× speedup** ✗ (below 2.5× target)
2. **Cox-de Boor percentage unknown** (need direct measurement)
3. **If it's 20%:** Theoretical max = 1 / (0.80 + 0.20/2.28) = **1.11×**
4. **Observed:** ~1.0× end-to-end

### Gap Analysis

Even if Cox-de Boor is 20% of runtime:
- Theoretical speedup with 2.28× SIMD: 1.11×
- Target speedup: 1.14×
- **Gap: 0.03× (3% difference)**

This suggests:
- Either Cox-de Boor is < 20% of runtime
- Or there's overhead from SIMD (cache effects, alignment, dispatch)

---

## Next Steps (Corrected Investigation Plan)

### 1. Understand Why SIMD is Only 2.28× (Below Target)

**Hypothesis:**
- std::experimental::simd has overhead vs intrinsics
- [[gnu::target_clones]] dispatch overhead
- Memory alignment issues
- Cache effects from gather/scatter operations

**Investigation:**
- Profile SIMD implementation with perf
- Compare to hand-written AVX2 intrinsics
- Check assembly output for vectorization quality
- Measure dispatch overhead

### 2. Measure Actual Cox-de Boor Overhead in Pipeline

**Method:**
- Add scoped timers to BSplineFitter4DSeparable::fit_axis()
- Accumulate time in Cox-de Boor calls (batch of 1000s)
- Compare to total fit time

**Alternative:**
- Use perf record/report to sample execution
- No instrumentation overhead
- Shows actual time distribution

### 3. Profile Matrix Construction and Banded Solver

**Method:**
- Time matrix construction separately from solver
- Time LU decomposition separately from forward/backward solve
- Identify actual dominant operation

### 4. Recalculate Amdahl's Law with Measured Data

Once we have:
- Actual SIMD speedup (2.28×) ✓
- Measured Cox-de Boor percentage (TBD)
- Measured other operation times (TBD)

Then calculate realistic theoretical maximum speedup.

---

## Lessons Learned (Updated)

### What Went Wrong

1. **Made performance claims without verified measurements**
   - Claimed 2.69× speedup, actually 2.28×
   - Claimed Cox-de Boor is 7.5%, not measured directly
   - Used derived estimates instead of instrumentation

2. **Relied on calculations instead of profiling**
   - Multiplied assumptions to get percentages
   - Didn't use existing profiling tools (perf, USDT)
   - Created flawed instrumentation (per-call timing)

3. **Rushed to conclusions**
   - Recommended accepting Phase 2 as complete
   - Before verifying target was met (2.5× SIMD speedup)
   - Before measuring actual bottlenecks

### How to Fix

1. **Measure first, calculate second**
   - Use profiling tools, not math
   - Direct instrumentation, not estimates
   - Verify every claim with data

2. **Check targets before claiming success**
   - Phase 2 target: 2.5× SIMD speedup
   - Actual: 2.28× speedup
   - This is a **miss**, not a success

3. **Peer review methodology, not just results**
   - Codex caught methodology flaws
   - Should have validated approach before executing

---

## Current Status

**Investigation incomplete. Corrected findings:**

1. ✗ SIMD achieved 2.28× speedup (BELOW 2.5× target)
2. ? Cox-de Boor percentage unknown (need measurement)
3. ? Amdahl's law calculation pending verified inputs
4. ✗ Phase 2 did NOT meet its SIMD performance target

**Cannot recommend accepting Phase 2 until:**
- SIMD speedup reaches 2.5× target, OR
- Root cause for 2.28× speedup is understood and accepted
- Actual pipeline bottlenecks are measured (not estimated)

---

## Appendix: Test Results

### Corrected Isolated SIMD Performance

```
=== Cox-de Boor Profiling ===
Evaluations: 100000

SIMD Cox-de Boor:
  Total time (5 runs): 6627.6 µs
  Per evaluation: 66.276 ns

Scalar Cox-de Boor:
  Total time (5 runs): 15112.4 µs
  Per evaluation: 151.124 ns

SIMD Speedup: 2.28× ✗ (target: 2.5×)
```

### Why Per-Call Instrumentation Failed

```
=== With Instrumentation Overhead ===
SIMD:   177.51 ns/eval  (instrumented)
Scalar: 166.29 ns/eval  (instrumented)
Speedup: 0.94×  (timing overhead dominates!)
```

Instrumentation overhead (~110ns per call) is larger than the operation being measured (~70ns), making results invalid.
