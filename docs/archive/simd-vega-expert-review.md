# Expert Review: SIMD Vega Optimization Analysis

**Date:** 2025-01-13
**Reviewer:** Codex AI (via MCP)
**Context:** B-spline 4D vega computation optimization

## Summary

Expert review confirms **scalar triple is the optimal production approach** for single-query vega computation. Vertical SIMD's 337ns regression is a fundamental structural limitation, not a fixable implementation detail.

## Key Findings

### 1. Vertical SIMD: Structural Limitation (Not Fixable)

**Root cause:** Modern CPUs (Zen 3+, Skylake+) have ≥3 independent FMA execution units. The scalar triple exploits this by running 3 independent accumulator chains simultaneously:

```cpp
// Scalar: 3 chains execute in parallel on separate FMA units
price_down = std::fma(coeff, w_down * w_r, price_down);  // FMA unit 0
price_base = std::fma(coeff, w_base * w_r, price_base);  // FMA unit 1
price_up = std::fma(coeff, w_up * w_r, price_up);        // FMA unit 2
```

**SIMD collapses ILP:** Packing 3 lanes into one SIMD register creates a single serialized dependency chain, forcing all FMAs through one execution port despite having multiple ports available.

**Quote:**
> "As long as you have just three meaningful lanes and the hardware already has ≥3 FMA pipes, vertical SIMD will stay behind."

**Verdict:** Even perfect implementation (hand-written intrinsics, zero overhead) cannot overcome the ILP advantage. The bottleneck is architectural, not code-level.

### 2. Pack/Broadcast Overhead: Symptom, Not Cause

While we measured 100-150ns broadcast overhead and 50-100ns stack packing overhead, these are **symptoms** of forcing scalar-friendly data through SIMD machinery.

**Potential micro-optimizations:**
- Hand-write intrinsics: `_mm256_set_pd` with immediates to keep sigma weights in registers
- Fully unroll rate loop with `vfma231pd` memory operands
- Skip `simd_t(coeff*w_r)` temporaries

**Expected gain:** "Few dozen nanoseconds" - chips away at symptoms but doesn't change structural disadvantage.

**Cost/benefit:** Not worth engineering complexity for <10% improvement that still loses to scalar.

### 3. Dual-Accumulator: Confirms Hypothesis But Doesn't Fix Root Cause

Our dual-accumulator experiment (612ns → 470ns, 23% improvement) validates that dependency chain serialization was a major factor.

**Expert analysis:**
> "Even with two accumulators you still pay the same pack/broadcast cost plus the inevitable dependency across lanes."

The 142ns improvement (23%) proves ILP loss was real, but remaining 197ns gap (vs scalar 273ns) is the irreducible pack/broadcast tax for this access pattern.

### 4. Horizontal SIMD: Feasible But Engineering-Heavy

**Current naive implementation:** Just demonstrates API, processes queries in scalar loop. The 5% overhead (1113ns vs 1054ns) is from unused SIMD register shuffling.

**Full optimization requires:**

1. **Vectorized span search:** Binary search on 4 queries in parallel using masked compares
2. **Vectorized cubic basis:** Evaluate basis functions for 4 different spans simultaneously
3. **Coefficient traversal:** Either:
   - Gather operations (higher latency)
   - Lane-major coefficient layout (memory reorganization)

**Challenge:** Spans differ per query → cannot reuse contiguous coefficient blocks → must restructure storage or accept gather latency.

**Expected gains:**

| Scenario | Speedup | Notes |
|----------|---------|-------|
| **Micro-batched queries** (shared spans) | 2.0–2.5× | Queries in same (m,t,v,r) cell |
| **Random queries** (diverse spans) | 1.2–1.5× | Gather latency + control divergence |

**Quote:**
> "Weigh the engineering lift against that ceiling."

**Verdict:** Only worth implementing if production workloads provide sustained 4-8 query batches with spatial locality.

### 5. Analytic Vega: The Best Optimization (Not Yet Tried!)

**Key insight from expert:** Instead of finite-difference triple evaluation, compute **analytic derivative** using B-spline derivative basis:

```cpp
// Current FD approach: eval(σ-ε, σ, σ+ε) → vega ≈ (V_up - V_down) / (2ε)
// Analytic approach: eval once with B'_k(σ) for volatility dimension
```

**Advantages:**
- Single tensor contraction (vs 3 evaluations)
- Light derivative basis evaluation
- Eliminates ±ε passes entirely
- Side-steps entire SIMD discussion

**B-spline derivatives are straightforward:** Cox-de Boor recursion has well-known derivative formulas.

**Expected performance:** Significantly faster than even scalar triple (273ns) - potentially <150ns.

**Priority:** HIGH - Prototype this first before further SIMD work.

### 6. Other Optimization Opportunities

#### Hoist Invariants in Scalar Triple
- `d_min/d_max` depend only on `lr` (rate span index)
- Unroll at-most-4 rate iterations completely
- Add `__restrict__` on `coeff_block`
- Expected gain: ~5% tighter scheduling

#### Compiler Flags
```bash
-ffast-math -fno-trapping-math  # If numerically acceptable
```
Gives compiler more freedom to keep basis arrays in registers.

**Caution:** Measure carefully - can affect numerical stability.

#### GPU/Accelerator Offload
For throughput workloads, bottleneck is memory bandwidth, not FMA count. Consider:
- GPU kernel for large batches
- Many-core CPU accelerator
- Will dominate any single-core SIMD tweak

### 7. Analysis Tooling Recommendations

#### llvm-mca: Quantify ILP and Register Pressure
```bash
llvm-mca -mtriple=x86_64 -mcpu=znver3 -resource-pressure \
  <extracted_loop.s>
```
Extract scalar and SIMD loops from assembly, compare:
- ILP (instructions per cycle)
- Issue ports (which FMA units saturate)
- Register pressure (actual spills)

#### perf: Confirm Hardware Utilization
```bash
perf stat -d ./iv_interpolation_profile --benchmark_filter=VegaTriple
perf stat -d ./iv_interpolation_profile --benchmark_filter=VegaSIMD

perf record -e cycles,uops_executed.port_5 ./iv_interpolation_profile
perf report
```
- Scalar should saturate multiple FMA ports
- SIMD should stall on `uops_executed.port_5` (broadcast port)

#### Assembly Inspection
```bash
objdump -dr bazel-bin/benchmarks/iv_interpolation_profile | less
```
Look for:
- Actual register spills in SIMD versions
- Broadcast instructions in hot path
- Stack store/load patterns

#### Advanced Profiling
- **VTune:** Deep microarchitecture analysis
- **uiCA:** Cycle-accurate simulation
- **likwid-perfctr:** Memory-level parallelism

## Production Recommendations

### Immediate Actions

1. ✅ **Use scalar triple for production** - Already done, integrated into IV solver
2. **Prototype analytic vega derivative** - Highest priority next step
3. **Run llvm-mca/perf** - Document ILP and register usage differences
4. **Keep SIMD variants as testbeds** - Gate behind runtime flag for future retest

### Decision Matrix for Horizontal SIMD

| Workload Characteristic | Recommendation |
|------------------------|----------------|
| Single queries | Use scalar triple (273ns) |
| 4-8 query batches, random | Use sequential scalar (1054ns) |
| 4-8 query batches, spatially clustered | Consider full horizontal SIMD (2.0-2.5× potential) |
| Massive throughput (1000s) | GPU/accelerator offload |

### Future Work

1. **Analytic derivative implementation**
   - Prototype B-spline derivative basis (`B'_k(σ)`)
   - Benchmark against scalar triple
   - Expected: <150ns per query

2. **Microarchitecture documentation**
   - `llvm-mca` analysis of ILP
   - `perf` confirmation of FMA port saturation
   - Assembly inspection for register usage

3. **Horizontal SIMD decision**
   - Profile production workloads for batch locality
   - If sustained 4-8 way batches with shared spans exist → implement
   - Otherwise → sequential scalar is optimal

## Conclusion

**Expert verdict:** Scalar triple is correct production choice. Vertical SIMD's failure is fundamental (ILP > narrow SIMD width), not fixable. Horizontal SIMD only worthwhile for specific batch workloads.

**Highest-impact next step:** Analytic vega derivative (single evaluation with `B'_k(σ)`) - side-steps entire SIMD discussion and should significantly outperform even scalar triple.

**Quote:**
> "The scalar triple is the right production path for single queries. Keep the SIMD variants around as testbeds but gate them behind a runtime flag so you can retest when AVX-512-wide cores or future compilers change the equation."
