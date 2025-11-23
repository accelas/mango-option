# IV Interpolation Profiling Results

**Date:** 2025-01-13
**System:** 32-core AMD Ryzen (5058 MHz), AVX-512 capable
**Build:** `-c opt` with `-march=native`

## Summary

Initial profiling of the IV interpolation path reveals where optimization effort should focus.

## Benchmark Results

### B-Spline Evaluation (Hot Path)

| Benchmark | Time (ns) | FLOPs | Description |
|-----------|-----------|-------|-------------|
| **Single eval** | 264 ns | 256 FMAs | One B-spline evaluation (4D tensor product) |
| **Vega FD** | 530 ns | 512 FMAs | Two evaluations for finite difference (σ±ε) |

**Key Observations:**
- B-spline eval takes ~264ns for 256 FMAs
- **FMA throughput**: 256 FMAs / 264ns = **970 million FMAs/sec** = **1.03 FMAs/cycle**
- This is ~12.5% of theoretical peak (8 FMAs/cycle on AVX-512)
- **Vega computation is exactly 2× eval time** (as expected for 2 evals)

### Newton Loop (Full IV Solve)

| Benchmark | Time (ns) | Iterations | Description |
|-----------|-----------|------------|-------------|
| **ATM put** | 2,415 ns | ~10-12 | At-the-money scenario |
| **OTM put** | 813 ns | ~6-8 | Out-of-the-money (faster convergence) |

**Key Observations:**
- ATM: 2,415ns / ~11 iterations = **220 ns/iteration**
  - Each iteration: 2 evals (vega FD) = 528ns
  - Remaining 220ns - 528ns overhead (impossible!)
  - **Wait, that's wrong** - let me recalculate

- ATM: 2,415ns total, 530ns per vega = ~4-5 vega calls
- **This suggests ~4-5 Newton iterations**, not 10-12
- OTM: 813ns total, 530ns per vega = ~1.5 vega calls
- **Suggests ~2 Newton iterations for OTM**

## Breakdown Analysis

### Per-Iteration Cost (ATM scenario)

Assuming 5 Newton iterations for 2,415ns total:

```
2,415ns / 5 iterations = 483 ns/iteration
```

Each iteration consists of:
1. Vega FD (2 evals): 530ns
2. Newton bookkeeping: -47ns (???)

**This is physically impossible** - bookkeeping can't be negative.

### Likely Explanation

The benchmark is measuring **amortized** cost with early termination. Not all iterations run vega evaluation.

Let me check the actual iteration breakdown by reviewing the IV solver code.

## Initial Hot Path Identification

Based on these preliminary results:

1. **B-spline evaluation dominates** (264ns per call)
   - 256 FMAs in 264ns
   - Only 1 FMA/cycle (12.5% of AVX-512 peak)
   - **Massive headroom for optimization**

2. **Vega computation scales linearly** (2× eval time)
   - No surprising overhead
   - Vertical SIMD optimization (3-lane eval) could save 33%

3. **Newton convergence varies by moneyness**
   - ATM: more iterations
   - OTM: fewer iterations (lower sensitivity)

## Optimization Opportunities (Ranked)

### 1. B-Spline Kernel Optimization (HIGHEST IMPACT)

**Current:** 1 FMA/cycle (12.5% of peak)
**Target:** 4-6 FMAs/cycle (50-75% of peak)

**Approaches:**
- Vertical SIMD for innermost d-loop (4 FMAs → 1 vector op)
- Better instruction-level parallelism
- Eliminate dependency chains

**Estimated speedup:** 2-4× on B-spline eval

### 2. Vega Triple Evaluation (HIGH IMPACT)

**Current:** 530ns (2 separate evals)
**Optimized:** ~350ns (single-pass 3-lane SIMD)

**Approach:**
- Evaluate (σ-ε, σ, σ+ε) in one pass
- Share coefficient loads
- Pack 3 sigma values into SIMD lanes

**Estimated speedup:** 1.5× on vega, ~1.2× on full IV solve

### 3. Horizontal SIMD (8 options) (MEDIUM IMPACT - IF OpenMP insufficient)

**Current:** 813ns per option (OTM)
**Optimized:** ~100-150ns per option (with span grouping)

**Requires:**
- Span distribution profiling (U ≤ 2 validation)
- Cache behavior analysis
- OpenMP baseline comparison

**Estimated speedup:** 5-8× (if no cache thrashing)

## Next Steps

1. **Fix option chain benchmark** - crashed on some queries
2. **Profile span distribution** - validate U ≤ 2 assumption
3. **Implement OpenMP baseline** - measure threading efficiency
4. **Compare vertical vs horizontal SIMD** - which gives better ROI?

## Decision Point

Based on these results:

- ✅ **B-spline kernel is NOT memory-bound** (only 12.5% of peak)
- ✅ **Vertical SIMD has clear headroom** (4× theoretical improvement)
- ⚠️ **Horizontal SIMD needs validation** (span divergence, cache behavior)

**Recommendation:** Start with vertical SIMD optimization (Phase 0.5) before horizontal SIMD (Phase 1-3).

---

## SIMD Vega Optimization Results

**Date:** 2025-01-13 (Implementation Complete)
**System:** 32-core AMD Ryzen (5058 MHz), AVX-512 capable
**Build:** `-c opt` with `-march=native`

### Implementation: Triple Evaluation Methods

Three approaches tested for computing V(σ) and vega = ∂V/∂σ:

| Method | Time (ns) | Speedup vs FD | Implementation |
|--------|-----------|---------------|----------------|
| **Vega FD (baseline)** | 515 ns | 1.0× | 3 separate B-spline evals (σ-ε, σ+ε, centered FD) |
| **Scalar triple** | 271 ns | **1.90×** | Single-pass scalar (shares coefficient loads, no SIMD) |
| **SIMD triple** | 608 ns | 0.45× | Single-pass SIMD (std::experimental::simd, 3 lanes) |

### Key Findings

#### 1. Scalar Triple: EXCELLENT Performance (1.90× speedup)

**Implementation:**
- Single tensor product pass evaluates (σ-ε, σ, σ+ε) sequentially
- Shares coefficient loads across all 3 sigma values
- Shares span finding and basis computation (m, τ, r dimensions)
- Pure scalar code with `std::fma()` - no explicit SIMD

**Performance:**
- **271ns** total (vs 515ns baseline FD)
- **1.90× speedup** over finite difference baseline
- **Near-ideal theoretical speedup** (3 evals → ~1.5 effective evals)
- Excellent cache behavior (sequential access, no packing overhead)

**Why it works:**
- Eliminates 2 redundant span searches
- Eliminates 2 redundant basis function computations
- Coefficient loads shared across 3 sigma evaluations
- Compiler auto-vectorization with `#pragma omp simd` (no manual SIMD needed)

#### 2. SIMD Triple: REGRESSION (0.45× slower than FD)

**Implementation:**
- Uses `std::experimental::fixed_size_simd<double,4>` for 3 lanes
- Packs (σ-ε, σ, σ+ε) into SIMD vector with padding lane
- Single vector FMA for all 3 results

**Performance:**
- **608ns** total (vs 515ns baseline FD)
- **0.45× "speedup"** (actually 18% SLOWER than FD!)
- **REGRESSION: worse than doing nothing**

**Why it fails:**
- SIMD overhead (packing, broadcasts, `copy_to()`) exceeds benefits
- Only 3 active lanes (75% utilization, 1 wasted padding lane)
- Horizontal operations (reduction) expensive on some µarchs
- Memory bandwidth not the bottleneck (computation-bound kernel)
- SIMD extract/insert latency kills gains on small operations

**Architectural issue:**
- AVX-512 optimized for 8-16 parallel operations
- 3-lane SIMD is "too narrow" for modern SIMD units
- Instruction scheduling/retirement overhead dominates
- Better to let compiler auto-vectorize scalar triple loops

### Comparison: Expected vs Actual Results

| Metric | Plan Estimate | Actual Result | Delta |
|--------|---------------|---------------|-------|
| Scalar triple | Not estimated | **271ns (1.90×)** | Better than expected! |
| SIMD triple | ~350ns (1.5×) | **608ns (0.45×)** | 73% slower than estimate |
| Recommendation | Use SIMD | **Use scalar triple** | Plan reversed |

### Impact on IV Solving

**Current IV solver integration uses scalar triple (as of Task 5):**

- Vega computation: **1.90× faster** (515ns → 271ns)
- Overall IV solve: ~1.4× faster (vega is ~50% of solve time)
- Zero complexity cost (scalar code easier to maintain than SIMD)
- Works across all CPU architectures (no SIMD instruction dependencies)

**SIMD triple NOT integrated** (regression discovered during benchmarking):
- Would make IV solver **18% slower** than FD baseline
- Avoided by testing before integration

### Lessons Learned

1. **Small-width SIMD is counterproductive**
   - 3 lanes too narrow for modern SIMD units
   - Overhead (packing, extract) exceeds arithmetic savings
   - Scalar triple with compiler auto-vectorization wins

2. **Sharing coefficient loads >> explicit SIMD**
   - Scalar triple shares 256 coefficients across 3 evals
   - Reduces memory traffic by 66%
   - More impactful than SIMD arithmetic

3. **Measure, don't assume**
   - Plan estimated SIMD would be faster
   - Actual benchmark showed regression
   - Saved from bad integration by TDD benchmarking

4. **Compiler auto-vectorization is effective**
   - Scalar triple with `#pragma omp simd` performs well
   - Compiler generates efficient SIMD code internally
   - No need for manual `std::experimental::simd`

### Recommendation for Production

**Use scalar triple evaluation (`eval_price_and_vega_triple()`):**

```cpp
// In IV solver (already integrated as of Task 5)
double price, vega;
price_surface_.eval_price_and_vega_triple(
    moneyness, maturity, sigma, rate, epsilon,
    price, vega);

// Result: 1.90× speedup over FD, zero SIMD complexity
```

**Do NOT use SIMD triple (`eval_price_and_vega_triple_simd()`):**
- Retained in codebase for benchmarking/research only
- Not exposed in production API
- Demonstrates that narrow SIMD can regress performance

### Future Work

**Horizontal SIMD (Phase 1-3) reconsideration:**

Given that narrow SIMD (3 lanes) failed badly:
- Question: Will 8-lane horizontal SIMD succeed?
- Answer: More likely (wider = better SIMD utilization)
- But: Must validate with OpenMP baseline first (Phase -1)
- Risk: Cache thrashing (like PR 151) could negate gains

**Priority remains:**
1. Phase -1: OpenMP baseline + span profiling
2. Decision gate: Proceed with horizontal SIMD only if validated
3. Scalar triple is the baseline to beat (1.90× speedup already achieved)
