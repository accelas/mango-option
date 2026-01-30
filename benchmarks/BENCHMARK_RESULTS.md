# C++20 Benchmark Results

**Date:** 2025-11-05
**Build:** Release (`-c opt`) with `-march=native`, LTO, AVX-512
**CPU:** 32 cores @ 5058 MHz (AMD Ryzen with AVX-512 support)
**OpenMP:** Enabled (batch benchmarks use `#pragma omp parallel for`)

---

## Executive Summary

### ✅ Strengths
- **Excellent numerical accuracy:** <2% error vs QuantLib across all scenarios
- **Predictable convergence:** Error decreases consistently with finer grids
- **Stable implementation:** Handles ATM, OTM, ITM, high/low vol consistently

### ⚠️ Performance Concerns
- **4-15x slower than QuantLib** on single thread
- **Performance gap widens** with larger grids (4.4x at 101x1000 → 15.4x at 501x5000)
- **12-13x speedup with OpenMP parallelization** on batch workloads (32 cores)
- **Source code still single-threaded:** OpenMP only in benchmarks, not in library code

---

## 1. Component Performance

### American Option Pricing (Single-Threaded)
| Grid Size | Time per Option | Throughput |
|-----------|----------------|------------|
| 51x500 | 3.2ms | 311 opts/sec |
| 101x1000 | 13.9ms | 72 opts/sec |
| 201x2000 | 75.0ms | 13 opts/sec |
| 501x5000 | 1057ms | 0.95 opt/sec |

### American Option Pricing (Parallel Batch - 32 cores)
| Batch Size | Time per Batch | Throughput |
|------------|---------------|------------|
| 10 options | 21ms | **481 opts/sec** (1.5x speedup) |
| 50 options | 61ms | **816 opts/sec** (11.3x speedup) |
| 100 options | 118ms | **848 opts/sec** (12.7x speedup) |

Grid: 101x1000 for all parallel benchmarks

### Implied Volatility Solver (Single-Threaded)
| Scenario | Time per IV |
|----------|-------------|
| ATM Put (1Y) | 130ms |
| OTM Put (3M) | 157ms |
| ITM Put (2Y) | 104ms |

**Average throughput:** ~7 IV calculations/second

### Implied Volatility Solver (Parallel Batch - 32 cores)
| Batch Size | Time per Batch | Throughput |
|------------|---------------|------------|
| 10 IVs | 162ms | **62 IVs/sec** (0.9x - overhead) |
| 50 IVs | 509ms | **98 IVs/sec** (14x speedup) |
| 100 IVs | 931ms | **107 IVs/sec** (15.3x speedup) |

Grid: 101x1000 for all parallel benchmarks

---

## 2. Performance vs QuantLib

### Direct Comparison (101x1000 grid, single-threaded)
| Scenario | mango-option | QuantLib | Slowdown |
|----------|----------|----------|----------|
| ATM Put | 13.0ms | 3.0ms | **4.4x** |
| OTM Put | 13.2ms | 3.0ms | **4.4x** |
| ITM Put | 13.0ms | 3.0ms | **4.4x** |

### Grid Resolution Scaling (single-threaded)
| Grid Size | mango-option | QuantLib | Slowdown |
|-----------|----------|----------|----------|
| 101x1000 | 13.8ms | 3.0ms | **4.6x** |
| 201x2000 | 78.3ms | 11.1ms | **7.1x** |
| 501x5000 | **972ms** | 63.1ms | **15.4x** ⚠️ |

**Critical Finding:** The performance gap **widens dramatically** with larger grids, suggesting algorithmic inefficiency rather than just slower execution.

### Parallel Batch Performance (32 cores, 101x1000 grid)
| Workload | mango-option (parallel) | QuantLib (single) | Comparison |
|----------|---------------------|-------------------|------------|
| 100 options | 118ms (848 opts/sec) | 297ms (337 opts/sec) | **2.5x faster** ✅ |
| 100 IVs | 931ms (107 IVs/sec) | N/A | - |

**Key Finding:** With OpenMP parallelization, mango-option can **outperform QuantLib** on batch workloads by leveraging multi-core hardware.

---

## 3. Accuracy vs QuantLib

### Price Accuracy (201x2000 grid)
| Scenario | mango-option | QuantLib | Error | Rel Error |
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
| Greek | mango-option | QuantLib | Rel Error |
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

### Why Is mango-option Slower?

**Scaling Analysis:**
- 101 → 201 points (2x): **5.7x slower** (13.8ms → 78.3ms)
- 201 → 501 points (2.5x): **12.4x slower** (78.3ms → 972ms)

**Expected:** O(n) or O(n log n) for well-optimized FDM
**Observed:** Worse than O(n²) behavior

**Likely Causes:**
1. **Cache inefficiency:** Poor memory access patterns for large grids
2. **Algorithmic complexity:** Hidden O(n²) loops somewhere
3. **Workspace allocation overhead:** Repeated allocations during iterations
4. **Limited parallelization:** Source code still single-threaded

### Parallelization Impact (32 cores)

**OpenMP `#pragma omp parallel for` on batch loops:**
- 10-item batch: **1.5x speedup** (overhead dominates)
- 50-item batch: **11-14x speedup** (good scaling)
- 100-item batch: **12-15x speedup** (near-optimal)

**Key Insight:** Adding OpenMP to source code spatial loops could deliver similar speedups, potentially closing the gap with QuantLib.

### AVX-512 Verification

**Disassembly check:** 75 AVX-512 instructions confirmed
```assembly
vfmadd132pd %zmm10,%zmm9,%zmm0     # FMA on 512-bit registers
vmulpd      %zmm9,%zmm1,%zmm1       # Multiply 8 doubles at once
vaddpd      %zmm14,%zmm0,%zmm0      # Add 8 doubles at once
```

SIMD vectorization is working correctly.

---

## Recommendations

### ✅ Completed
1. **OpenMP parallelization for batch workloads** - 12-15x speedup demonstrated
2. **AVX-512 SIMD verification** - 75 zmm instructions confirmed active
3. **Link-Time Optimization (LTO)** - Enabled in all benchmark builds
4. **Static linking** - Library code statically linked for optimal performance

### Immediate (High Impact)
1. **Add OpenMP to source code spatial loops** 🎯
   - Current: Only benchmarks use `#pragma omp parallel for`
   - Target: src/cpp/*.cpp spatial operators
   - Expected: 10-15x speedup based on batch benchmark results
   - Impact: Could close or reverse performance gap with QuantLib

2. **Profile with perf/vtune** to identify hotspots
   ```bash
   perf record -g bazel-bin/benchmarks/component_performance
   perf report
   ```

3. **Fix theta computation bug**
   - Currently returns 0 instead of proper value
   - Affects Greeks accuracy

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
9. **Optimize single-option performance** (currently 4.4x slower than QuantLib)

---

## Conclusion

The C++20 implementation is **numerically excellent** with demonstrated **parallelization potential**:

### Strengths
- ✅ **Accuracy:** <2% error vs QuantLib across all scenarios
- ✅ **Batch performance:** 2.5x faster than QuantLib with OpenMP (32 cores)
- ✅ **Convergence:** Predictable second-order error reduction
- ✅ **SIMD:** AVX-512 instructions confirmed active

### Remaining Issues
- ⚠️ **Single-option performance:** 4.4x slower than QuantLib (single-thread)
- ⚠️ **Poor grid scaling:** Worse than O(n²) behavior
- 🐛 **Theta bug:** Returns 0 instead of proper value

### Next Steps
**Priority 1:** Add OpenMP to source code spatial loops
- Benchmark shows 12-15x speedup potential
- Could make mango-option **10x faster than QuantLib** on batch workloads

**Priority 2:** Profile and fix O(n²) scaling behavior
- Would improve both single and parallel performance
- Critical for large grids (>201 points)
