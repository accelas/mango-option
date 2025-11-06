# C++20 Benchmark Results

**Date:** 2025-11-05
**Build:** Release (`-c opt`) with `-march=native`
**CPU:** 32 cores @ 5058 MHz (AMD Ryzen with AVX-512 support)
**OpenMP:** Enabled but not yet used (no `#pragma omp parallel for` in source)

---

## Executive Summary

### ✅ Strengths
- **Excellent numerical accuracy:** <2% error vs QuantLib across all scenarios
- **Predictable convergence:** Error decreases consistently with finer grids
- **Stable implementation:** Handles ATM, OTM, ITM, high/low vol consistently

### ⚠️ Performance Concerns
- **4-15x slower than QuantLib** on single thread
- **Performance gap widens** with larger grids (4x at 101x1000 → 15x at 501x5000)
- **Minimal optimization benefit:** Release build only 8% faster than debug on large grids
- **Single-threaded only:** No OpenMP parallel regions yet

---

## 1. Component Performance

### American Option Pricing
| Grid Size | Time per Option | Throughput |
|-----------|----------------|------------|
| 51x500 | 3.3ms | 300 opts/sec |
| 101x1000 | 13.8ms | 72 opts/sec |
| 201x2000 | 73ms | 14 opts/sec |
| 501x5000 | 1035ms | 1 opt/sec |

### Implied Volatility Solver
| Scenario | Time per IV |
|----------|-------------|
| ATM Put (1Y) | 130ms |
| OTM Put (3M) | 157ms |
| ITM Put (2Y) | 104ms |

**Average throughput:** ~7 IV calculations/second

---

## 2. Performance vs QuantLib

### Direct Comparison (101x1000 grid)
| Scenario | mango-iv | QuantLib | Slowdown |
|----------|----------|----------|----------|
| ATM Put | 13.0ms | 2.9ms | **4.5x** |
| OTM Put | 13.3ms | 3.0ms | **4.4x** |
| ITM Put | 13.3ms | 3.0ms | **4.5x** |

### Grid Resolution Scaling
| Grid Size | mango-iv | QuantLib | Slowdown |
|-----------|----------|----------|----------|
| 101x1000 | 13.7ms | 3.0ms | **4.6x** |
| 201x2000 | 80ms | 11.1ms | **7.2x** |
| 501x5000 | **1037ms** | 63.3ms | **16.4x** ⚠️ |

**Critical Finding:** The performance gap **widens dramatically** with larger grids, suggesting algorithmic inefficiency rather than just slower execution.

---

## 3. Accuracy vs QuantLib

### Price Accuracy (201x2000 grid)
| Scenario | mango-iv | QuantLib | Error | Rel Error |
|----------|----------|----------|-------|-----------|
| ATM Put 1Y | 6.637 | 6.660 | 0.023 | **0.35%** ✅ |
| OTM Put 3M | 2.289 | 2.292 | 0.004 | **0.17%** ✅ |
| ITM Put 2Y | 15.854 | 15.856 | 0.002 | **0.01%** ⭐ |
| Deep ITM Put 6M | 20.125 | 20.125 | 0.001 | **0.003%** ⭐ |
| High Vol Put 1Y | 18.036 | 18.045 | 0.009 | **0.05%** ✅ |
| Low Vol Put 1Y | 2.848 | 2.895 | 0.047 | **1.62%** ⚠️ |
| Long Maturity 5Y | 11.637 | 11.644 | 0.007 | **0.06%** ✅ |

**All scenarios < 2% error** - excellent agreement with QuantLib reference.

### Greeks Accuracy (201x2000 grid, ATM Put 1Y)
| Greek | mango-iv | QuantLib | Rel Error |
|-------|----------|----------|-----------|
| Delta | -0.423 | -0.423 | **0.004%** ⭐ |
| Gamma | 0.02154 | 0.02148 | **0.29%** ✅ |
| Theta | 0.00 | -2.70 | **100%** 🐛 |

**Bug Identified:** Theta computation returns 0 instead of proper value.

### Convergence Study (ATM Put 1Y)
| Grid | Price | Error vs Ref | Rel Error |
|------|-------|--------------|-----------|
| 51x500 | 6.265 | 0.396 | **5.94%** |
| 101x1000 | 6.565 | 0.095 | **1.43%** |
| 201x2000 | 6.637 | 0.024 | **0.36%** ✅ |
| 501x5000 | 6.657 | 0.004 | **0.06%** ⭐ |

Reference (QuantLib 1001x10000): 6.661

**Convergence rate:** ~4x error reduction per 2x grid refinement (expected for second-order method).

---

## Analysis

### Why Is mango-iv Slower?

**Scaling Analysis:**
- 101 → 201 points (2x): **5.8x slower** (13.7ms → 80ms)
- 201 → 501 points (2.5x): **13x slower** (80ms → 1037ms)

**Expected:** O(n) or O(n log n) for well-optimized FDM
**Observed:** Worse than O(n²) behavior

**Likely Causes:**
1. **Cache inefficiency:** Poor memory access patterns for large grids
2. **Algorithmic complexity:** Hidden O(n²) loops somewhere
3. **Workspace allocation overhead:** Repeated allocations during iterations
4. **No parallelization:** Single-threaded vs QuantLib's optimized loops

### Optimization Impact

**Debug → Release build improvement:**
- Small grids (101x1000): **No change** (13-14ms)
- Large grids (501x5000): **Only 8%** (1048ms → 956ms)

**Interpretation:** Bottleneck is NOT in computation-heavy loops that benefit from optimization, but likely in:
- Memory access patterns (cache misses)
- Algorithmic complexity (nested loops)
- Sequential dependencies (can't vectorize)

---

## Recommendations

### Immediate (High Impact)
1. **Profile with perf/vtune** to identify hotspots
   ```bash
   perf record -g bazel-bin/benchmarks/component_performance
   perf report
   ```

2. **Add OpenMP parallelization** to spatial loops
   - Target: 16-32 thread speedup on 32-core machine
   - Expected: ~4x faster with good scaling

3. **Fix theta computation bug**

### Medium Term
4. **Optimize memory access patterns**
   - Check for cache-inefficient stride patterns
   - Consider data layout changes (AoS vs SoA)

5. **Algorithm audit**
   - Look for O(n²) loops that could be O(n)
   - Consider more efficient linear solvers

6. **Compare with QuantLib implementation**
   - Study their FDM approach
   - Identify algorithmic differences

### Long Term
7. **Consider GPU acceleration** for large grids
8. **Implement adaptive mesh refinement**
9. **Add batch processing optimizations**

---

## Conclusion

The C++20 implementation is **numerically excellent** but **performance-limited** by:
1. Single-threaded execution (biggest opportunity)
2. Poor scaling with grid size (algorithmic issue)
3. Suboptimal memory access patterns

**Quick win:** Add OpenMP `#pragma omp parallel for` to spatial loops → expect 4-8x speedup.

**Fundamental fix:** Profile and optimize the O(n²) behavior in grid scaling.
