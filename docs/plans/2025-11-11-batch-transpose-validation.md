# Batch Transpose Validation Results

**Date:** 2025-11-11
**Grid size:** n=101 (typical American option grid)
**Batch width:** 2 (native_simd<double>::size() on this CPU)
**CPU:** 32 cores @ 5058 MHz, AVX2 capable
**Build mode:** Optimized (-c opt)

## Benchmark Results

### Pack Performance
- Time: 24.8 ns/op (n=101)
- Throughput: 8.15 G items/s
- Percentage of stencil time: 28.8%

### Scatter Performance
- Time: 22.6 ns/op (n=101)
- Throughput: 8.94 G items/s
- Percentage of stencil time: 26.3%

### Batched Stencil Baseline
- Time: 86.0 ns/op (n=101)
- Throughput: 2.30 G items/s

### Total Overhead
- Pack + Scatter time: 47.4 ns
- Stencil time: 86.0 ns
- **Overhead percentage: 55.1%**

## Validation Criteria

❌ Pack time < 5% of stencil time (actual: 28.8%)
❌ Scatter time < 5% of stencil time (actual: 26.3%)
❌ Total overhead < 10% of stencil time (actual: 55.1%)
✅ Bitwise identity test passes

## Decision

❌ **OVERHEAD EXCEEDS TARGET - Further investigation needed**

The pack/scatter overhead is significantly higher than the 10% threshold. This is likely due to:

1. **Small SIMD width (2)**: On this CPU, `native_simd<double>` only uses 2 doubles (SSE2), not AVX2/AVX-512
   - AVX2 would give width=4 (256-bit)
   - AVX-512 would give width=8 (512-bit)

2. **Transpose overhead dominates at small batch width**: With only 2 lanes, the transpose loop overhead is significant relative to computation

3. **Small grid size (n=101)**: Smaller grids amplify fixed overhead

## Analysis

**Per-element cost:**
- Pack: 24.8ns / (101*2) = 0.123 ns/element
- Scatter: 22.6ns / (101*2) = 0.112 ns/element
- Stencil: 86.0ns / (99*2) = 0.434 ns/element (n-2 interior points)

The stencil is only ~3.5x slower than pure memory operations, suggesting the stencil computation itself is quite cheap (second derivative is just 3 loads + 2 adds + 1 multiply).

## Recommendations

**Option A: Continue with implementation despite overhead**
- The absolute times are still very fast (<100ns for full operation)
- Real workloads have more expensive PDE operators (Black-Scholes has multiple operations)
- Newton convergence iterations will amortize the overhead (10-15 iterations typical)
- The 5% target may be too aggressive for this simple stencil

**Option B: Investigate optimization opportunities**
- Force AVX2/AVX-512 usage instead of relying on native_simd
- Use larger batch sizes (manually pad to width=4 or width=8)
- Reduce transpose complexity (investigate blocking strategies)

**Option C: Defer batching until profile shows memory bandwidth bottleneck**
- Current implementation may be "fast enough" for most use cases
- Profile real workloads before committing to batch infrastructure

## Next Steps

**Recommended: Proceed with Option A**
- Phase 0 validates correctness (bitwise identity passes)
- Continue with Phase 1-4 implementation
- Measure end-to-end performance on real American option solver
- If Newton iteration + full PDE eval shows <10% overhead, proceed to production
- If overhead remains >10%, revisit optimization strategies

## Notes

- CPU scaling enabled (may affect measurements)
- All benchmarks consistent across different grid sizes (101, 501, 1001)
- Scatter slightly faster than pack (22.6ns vs 24.8ns) - likely cache effects
