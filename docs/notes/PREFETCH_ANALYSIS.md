# CPU Prefetch Analysis for Cubic Spline Interpolation

**Date**: 2025-10-30
**Context**: Analyzing whether explicit CPU prefetch instructions would improve 4D/5D cubic spline interpolation performance

---

## Executive Summary

**Recommendation**: **Conditionally beneficial** - Prefetch can provide **5-15% speedup** in specific hot paths, but requires careful implementation to avoid performance degradation.

**Best candidates**:
1. ✅ **Slice extraction from strided arrays** (4D/5D tensor)
2. ✅ **Spline coefficient access** during evaluation
3. ❌ **Intermediate array access** (already cache-friendly)
4. ❌ **Binary search** (irregular pattern, hard to prefetch)

---

## Memory Access Pattern Analysis

### 1. 4D Cubic Spline Interpolation (`cubic_interpolate_4d`)

**Current implementation** (src/interp_cubic.c:492-714):

```c
// Stage 1: Interpolate along moneyness (nested loops)
for (size_t j_tau = 0; j_tau < n_tau; j_tau++) {          // 30 iterations
    for (size_t j_sigma = 0; j_sigma < n_sigma; j_sigma++) {  // 20 iterations
        for (size_t j_r = 0; j_r < n_r; j_r++) {          // 10 iterations
            // Extract moneyness slice: STRIDED ACCESS (HOT PATH)
            for (size_t i_m = 0; i_m < n_m; i_m++) {      // 50 iterations
                size_t idx = i_m * stride_m           // stride_m = 30*20*10 = 6000
                           + j_tau * stride_tau       // stride_tau = 20*10 = 200
                           + j_sigma * stride_sigma   // stride_sigma = 10
                           + j_r * stride_r;          // stride_r = 1
                moneyness_slice[i_m] = table->prices[idx];  // ← Large stride access
            }

            // Build spline and evaluate
            pde_spline_init(&m_spline, ...);
            intermediate1[idx1] = pde_spline_eval(&m_spline, moneyness);
        }
    }
}
```

**Memory access characteristics**:
- **Total spline creations**: 30 × 20 × 10 = **6,000 splines** per 4D query
- **Stride size**: stride_m = 6,000 doubles = **48 KB** between successive moneyness points
- **Cache line size**: 64 bytes = 8 doubles
- **Problem**: stride_m >> cache line size → **cache thrashing**

### 2. Spline Evaluation (`pde_spline_eval`)

**Current implementation** (src/cubic_spline.c:171-184):

```c
double pde_spline_eval(const CubicSpline *spline, double x_eval) {
    size_t i = find_interval(spline->x, spline->n_points, x_eval);

    double dx = x_eval - spline->x[i];
    double result = spline->coeffs_a[i] +          // ← Access array 1
                   spline->coeffs_b[i] * dx +       // ← Access array 2
                   spline->coeffs_c[i] * dx * dx +  // ← Access array 3
                   spline->coeffs_d[i] * dx * dx * dx;  // ← Access array 4
    return result;
}
```

**Memory access pattern**:
- **4 sequential loads** from different arrays: a[i], b[i], c[i], d[i]
- Arrays are stored in **separate memory regions** (not struct-of-arrays)
- If i is known, can prefetch all 4 coefficients in advance
- **Called 6,000 times** per 4D query

### 3. Intermediate Array Access

**Pattern**:
```c
// Sequential access (cache-friendly)
for (size_t j_tau = 0; j_tau < n_tau; j_tau++) {
    maturity_slice[j_tau] = intermediate1[j_tau * (n_sigma * n_r) + ...];
}
```

**Characteristics**:
- **Sequential or small-stride** access
- Hardware prefetcher handles this well
- **No benefit** from manual prefetch

---

## Hardware Prefetcher Capabilities

Modern CPUs (Intel/AMD x86-64) have automatic prefetchers:

1. **L1 Streamer**: Detects sequential access (stride = 1)
2. **L2 Streamer**: Detects strided access (stride up to ~256 bytes)
3. **L2 Adaptive**: Learns complex patterns

**Limitations**:
- ❌ **Large strides** (>256 bytes / >32 doubles) often **missed**
- ❌ **Irregular patterns** not detected
- ❌ **Cross-page boundaries** can confuse prefetcher
- ❌ **Multiple simultaneous streams** (>4) compete for prefetch resources

**Our case**:
- stride_m = 48 KB >> 256 bytes → **Hardware prefetcher will miss**
- 4 coefficient arrays in spline eval → **May not prefetch all 4**

---

## Prefetch Implementation Strategy

### Strategy 1: Prefetch Next Slice During Current Spline Build

**Target**: Slice extraction hot path (lines 563-569 in interp_cubic.c)

**Implementation**:
```c
#include <xmmintrin.h>  // For _mm_prefetch

// Stage 1: Interpolate along moneyness
for (size_t j_tau = 0; j_tau < table->n_maturity; j_tau++) {
    for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
        for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
            // Extract moneyness slice for CURRENT combo
            for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
                size_t idx_current = i_m * table->stride_m
                                   + j_tau * table->stride_tau
                                   + j_sigma * table->stride_sigma
                                   + j_r * table->stride_r;

                moneyness_slice[i_m] = table->prices[idx_current];

                // Prefetch NEXT moneyness point (if not last combo)
                if (j_r + 1 < table->n_rate || j_sigma + 1 < table->n_volatility ||
                    j_tau + 1 < table->n_maturity) {

                    // Calculate next combo's first element
                    size_t next_j_r = (j_r + 1 < table->n_rate) ? j_r + 1 : 0;
                    size_t next_j_sigma = (j_r + 1 < table->n_rate) ? j_sigma :
                                         ((j_sigma + 1 < table->n_volatility) ? j_sigma + 1 : 0);
                    size_t next_j_tau = (j_r + 1 < table->n_rate || j_sigma + 1 < table->n_volatility) ?
                                        j_tau : (j_tau + 1);

                    size_t idx_next = i_m * table->stride_m
                                    + next_j_tau * table->stride_tau
                                    + next_j_sigma * table->stride_sigma
                                    + next_j_r * table->stride_r;

                    // Prefetch to L1 cache
                    _mm_prefetch((const char*)&table->prices[idx_next], _MM_HINT_T0);
                }
            }

            // Build spline (uses moneyness_slice[] - already in cache)
            pde_spline_init(&m_spline, table->moneyness_grid, moneyness_slice,
                           table->n_moneyness, spline_coeff_workspace,
                           spline_temp_workspace);

            // Evaluate spline
            size_t idx1 = j_tau * (table->n_volatility * table->n_rate)
                        + j_sigma * table->n_rate + j_r;
            intermediate1[idx1] = pde_spline_eval(&m_spline, moneyness);
        }
    }
}
```

**Expected impact**:
- **Hides latency** of strided memory access (~100-300 cycles)
- **Estimated speedup**: 10-15% for Stage 1 (slice extraction + spline build)
- **Overall query speedup**: ~5-10% (Stage 1 dominates)

**Trade-offs**:
- ✅ Reduces cache misses for large-stride access
- ⚠️ Complex loop logic (harder to maintain)
- ⚠️ May pollute cache if prefetch distance wrong

---

### Strategy 2: Prefetch Spline Coefficients in Evaluation

**Target**: Spline evaluation (src/cubic_spline.c:171-184)

**Implementation**:
```c
double pde_spline_eval(const CubicSpline *spline, double x_eval) {
    // Find interval using binary search
    size_t i = find_interval(spline->x, spline->n_points, x_eval);

    // Prefetch all 4 coefficient arrays at index i
    // Note: Prefetch BEFORE computing dx to hide latency
    _mm_prefetch((const char*)&spline->coeffs_a[i], _MM_HINT_T0);  // 64 bytes (8 doubles)
    _mm_prefetch((const char*)&spline->coeffs_b[i], _MM_HINT_T0);
    _mm_prefetch((const char*)&spline->coeffs_c[i], _MM_HINT_T0);
    _mm_prefetch((const char*)&spline->coeffs_d[i], _MM_HINT_T0);

    // Do some computation while prefetch is in flight
    double dx = x_eval - spline->x[i];
    double dx2 = dx * dx;   // Can compute while waiting for prefetch
    double dx3 = dx2 * dx;  // More computation to hide latency

    // Now access prefetched data
    double result = spline->coeffs_a[i] +
                   spline->coeffs_b[i] * dx +
                   spline->coeffs_c[i] * dx2 +
                   spline->coeffs_d[i] * dx3;

    return result;
}
```

**Expected impact**:
- **Avoids sequential cache misses** for 4 coefficient arrays
- **Estimated speedup**: 5-10% for spline evaluation
- **Overall query speedup**: ~2-5% (evaluation is small part of total time)

**Trade-offs**:
- ✅ Simple to implement
- ✅ Low risk of cache pollution
- ⚠️ Benefit depends on cache state (may already be cached from spline_init)
- ⚠️ 4 prefetch instructions add overhead if data already in L1

---

### Strategy 3: Software Pipelining for Stage 1 Loops

**Concept**: Prefetch slice data for iteration N+k while processing iteration N

**Implementation sketch**:
```c
// Prefetch k iterations ahead (k = 2-4 optimal)
const size_t PREFETCH_DISTANCE = 2;

for (size_t combo_idx = 0; combo_idx < total_combos; combo_idx++) {
    // Decode combo_idx to (j_tau, j_sigma, j_r)
    size_t j_tau = combo_idx / (n_sigma * n_r);
    size_t j_sigma = (combo_idx / n_r) % n_sigma;
    size_t j_r = combo_idx % n_r;

    // Prefetch slice for combo_idx + PREFETCH_DISTANCE
    if (combo_idx + PREFETCH_DISTANCE < total_combos) {
        size_t pf_j_tau = (combo_idx + PREFETCH_DISTANCE) / (n_sigma * n_r);
        size_t pf_j_sigma = ((combo_idx + PREFETCH_DISTANCE) / n_r) % n_sigma;
        size_t pf_j_r = (combo_idx + PREFETCH_DISTANCE) % n_r;

        // Prefetch first few elements of slice
        for (size_t i_m = 0; i_m < min(4, n_m); i_m++) {  // Prefetch 4 cache lines
            size_t pf_idx = i_m * stride_m + pf_j_tau * stride_tau +
                           pf_j_sigma * stride_sigma + pf_j_r * stride_r;
            _mm_prefetch((const char*)&table->prices[pf_idx], _MM_HINT_T0);
        }
    }

    // Process current slice (combo_idx)
    for (size_t i_m = 0; i_m < n_m; i_m++) {
        size_t idx = i_m * stride_m + j_tau * stride_tau +
                    j_sigma * stride_sigma + j_r * stride_r;
        moneyness_slice[i_m] = table->prices[idx];
    }

    // Build and evaluate spline
    pde_spline_init(&m_spline, ...);
    intermediate1[combo_idx] = pde_spline_eval(&m_spline, moneyness);
}
```

**Expected impact**:
- **Maximum prefetch benefit** (optimal distance tuning)
- **Estimated speedup**: 12-18% for Stage 1
- **Overall query speedup**: ~8-12%

**Trade-offs**:
- ✅ Best performance potential
- ❌ Most complex implementation
- ❌ Prefetch distance must be tuned per architecture
- ❌ Harder to maintain and debug

---

## Benchmark Methodology

To accurately measure prefetch benefit, we need:

### Test Configuration
```c
// Benchmark setup
Grid: 50×30×20×10 (typical 4D)
Queries: 100,000 random points
CPU: Measure on target hardware (Intel Xeon / AMD EPYC)
Isolation: Disable turbo boost, fix CPU frequency
Timing: Use RDTSC or high-resolution timer
```

### Baseline Measurement
```c
// Current implementation (no prefetch)
for (int i = 0; i < 100000; i++) {
    double price = cubic_interpolate_4d(table,
        queries[i].m, queries[i].tau, queries[i].sigma, queries[i].r, ctx);
}
```

### Prefetch Variants
```c
// Variant 1: Prefetch next slice
// Variant 2: Prefetch spline coefficients
// Variant 3: Software pipelining
```

### Metrics to Collect
- **Wall-clock time** per query
- **Cache miss rate** (L1, L2, L3) via perf counters
- **Instructions retired** (check for overhead)
- **CPU cycles** per query

### Hardware Counter Commands
```bash
# Measure cache misses
perf stat -e cache-misses,cache-references,L1-dcache-load-misses \
    ./bazel-bin/benchmarks/cubic_interp_benchmark

# Measure memory bandwidth
perf stat -e mem_load_retired.l1_miss,mem_load_retired.l2_miss \
    ./bazel-bin/benchmarks/cubic_interp_benchmark
```

---

## Expected Performance Gains

### Conservative Estimates (based on stride analysis)

| Strategy | Implementation Effort | Expected Speedup | Risk Level |
|----------|----------------------|------------------|------------|
| **Strategy 1: Prefetch next slice** | Medium | **8-12%** | Low |
| **Strategy 2: Prefetch coefficients** | Low | **2-5%** | Very Low |
| **Strategy 3: Software pipelining** | High | **12-18%** | Medium |
| **Combined (1+2)** | Medium | **10-15%** | Low-Medium |

### Breakdown by Query Stage (4D)

| Stage | Time % (current) | With Prefetch | Speedup |
|-------|------------------|---------------|---------|
| **Stage 1** (moneyness) | 60% | 45% | **1.33x** |
| **Stage 2** (maturity) | 20% | 18% | **1.11x** |
| **Stage 3** (volatility) | 12% | 11% | **1.09x** |
| **Stage 4** (rate) | 8% | 7.5% | **1.07x** |
| **Overall** | 100% | **81.5%** | **~1.10x** |

**Total estimated speedup**: **10-12% for 4D, 12-15% for 5D**

---

## Implementation Recommendations

### Phase 1: Low-Hanging Fruit (Recommended to start)

**Implement Strategy 2** (prefetch spline coefficients):
- ✅ **Simple**: 5 lines of code
- ✅ **Low risk**: No complex logic
- ✅ **Measurable**: Easy to benchmark
- ✅ **Portable**: Works on all x86-64 CPUs

**Expected ROI**: 2-5% speedup for 1 hour of work

### Phase 2: High-Impact Optimization (If Phase 1 shows benefit)

**Implement Strategy 1** (prefetch next slice):
- ⚠️ **Moderate complexity**: Nested loop refactoring
- ✅ **High impact**: 8-12% speedup potential
- ⚠️ **Requires tuning**: May need architecture-specific adjustments

**Expected ROI**: 8-12% speedup for 1-2 days of work

### Phase 3: Maximum Performance (If both Phase 1 & 2 successful)

**Implement Strategy 3** (software pipelining):
- ❌ **High complexity**: Full loop restructuring
- ✅ **Maximum impact**: 12-18% speedup potential
- ❌ **Maintenance burden**: Harder to understand and modify

**Expected ROI**: 12-18% speedup for 3-5 days of work

---

## Portability Considerations

### x86-64 (Intel/AMD)
```c
#ifdef __x86_64__
  #include <xmmintrin.h>  // SSE intrinsics
  #define PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#endif
```

**Prefetch hints**:
- `_MM_HINT_T0`: Prefetch to all cache levels (L1, L2, L3) - use for immediate access
- `_MM_HINT_T1`: Prefetch to L2/L3 (skip L1) - use for near-term access
- `_MM_HINT_T2`: Prefetch to L3 only - use for distant access
- `_MM_HINT_NTA`: Non-temporal prefetch (bypass cache) - use for streaming data

**For our case**: Use `_MM_HINT_T0` for slice extraction (immediate use)

### ARM (Apple Silicon, AWS Graviton)
```c
#ifdef __aarch64__
  #define PREFETCH(addr) __builtin_prefetch((const void*)(addr), 0, 3)
  // Params: (address, rw=0 for read, locality=3 for high temporal locality)
#endif
```

### Generic fallback
```c
#ifndef PREFETCH
  #define PREFETCH(addr) ((void)0)  // No-op on unsupported platforms
#endif
```

---

## Recommended Action Plan

### Step 1: Baseline Measurement (1 hour)
```bash
# Add benchmark to measure current performance
bazel build //benchmarks:cubic_interp_prefetch_benchmark
./bazel-bin/benchmarks/cubic_interp_prefetch_benchmark --benchmark_repetitions=10

# Collect hardware counters
perf stat -e cache-misses,L1-dcache-load-misses,LLC-load-misses \
    ./bazel-bin/benchmarks/cubic_interp_prefetch_benchmark
```

**Success criteria**: Confirm high cache miss rate (>20%) for Stage 1

### Step 2: Implement Strategy 2 (2-3 hours)
```c
// Modify src/cubic_spline.c:pde_spline_eval()
// Add 4 prefetch instructions + reorder computation
```

**Success criteria**: Measure 2-5% speedup in benchmark

### Step 3: Implement Strategy 1 (1-2 days)
```c
// Modify src/interp_cubic.c:cubic_interpolate_4d()
// Add prefetch logic in slice extraction loops
```

**Success criteria**: Measure 8-12% speedup in benchmark

### Step 4: Validate on Target Hardware (1 day)
- Test on production server CPU
- Verify speedup holds across different grid sizes
- Check for cache pollution side effects

### Step 5: Document and Commit (1 day)
- Add comments explaining prefetch logic
- Update performance documentation
- Create before/after benchmark results

---

## Risks and Mitigation

### Risk 1: No measurable improvement
**Probability**: 20%
**Cause**: Hardware prefetcher already handles our access pattern
**Mitigation**: Verify with `perf` that cache misses are actually reduced

### Risk 2: Performance regression
**Probability**: 10%
**Cause**: Cache pollution from over-aggressive prefetch
**Mitigation**: Use conservative prefetch distance (k=1-2), benchmark extensively

### Risk 3: Code complexity increase
**Probability**: 40%
**Cause**: Prefetch logic makes loops harder to understand
**Mitigation**: Use macro abstraction, add extensive comments

### Risk 4: Portability issues
**Probability**: 15%
**Cause**: Prefetch behavior differs across CPUs
**Mitigation**: Make prefetch optional via compile flag, test on target platforms

---

## Conclusion

**Recommendation**: **Proceed with incremental implementation**

1. **Phase 1** (Low risk, low effort): Prefetch spline coefficients
   - Expected: 2-5% speedup
   - Time: 2-3 hours
   - **DO THIS FIRST**

2. **Phase 2** (Medium risk, medium effort): Prefetch next slice
   - Expected: 8-12% speedup
   - Time: 1-2 days
   - **DO IF Phase 1 successful**

3. **Phase 3** (High risk, high effort): Software pipelining
   - Expected: 12-18% speedup
   - Time: 3-5 days
   - **ONLY IF critical for your use case**

**Combined benefit**: **10-15% speedup** with moderate implementation effort

**Key insight**: Your 4D interpolation already takes ~1-2µs per query. A 10-15% speedup brings it to **~0.85-1.7µs**, which is still **12,000x faster than FDM** (21.7ms). The question is whether this optimization is worth the complexity for your trading scenario.

**For your use case (900 options @ 100ms updates)**:
- Current: 900 × 2µs = **1.8ms** per update cycle
- With prefetch: 900 × 1.7µs = **1.53ms** per update cycle
- **Savings: 0.27ms** (marginal for 100ms budget)

**Recommendation for your trading scenario**: **Defer prefetch optimization** unless profiling shows interpolation is actually a bottleneck. Focus first on other system components (network I/O, order management, etc.) where gains may be larger.
