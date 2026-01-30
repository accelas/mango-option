<!-- SPDX-License-Identifier: MIT -->
# Compiler Performance Comparison: Clang vs GCC

**Configuration:** Both with Scalar backend only (SIMD disabled)
- **Clang:** 19.1.7 (Debian)
- **GCC:** 14.2.0 (Debian)

## Benchmark Results

| Benchmark | Clang 19.1.7 | GCC 14.2.0 | Speedup |
|-----------|-------------|------------|---------|
| American single (101x498) | 1.11 ms | 1.28 ms | **1.15x faster** |
| American sequential (64) | 69.77 ms | 81.17 ms | **1.16x faster** |
| American parallel batch (64) | 3.96 ms | 5.90 ms | **1.49x faster** |
| American IV (FDM, 101x1k) | 15.70 ms | 17.99 ms | **1.15x faster** |
| American IV (FDM, 201x2k) | 15.65 ms | 18.07 ms | **1.15x faster** |
| American IV (B-spline) | 2.68 µs | 2.73 µs | 1.02x faster |
| Price table interpolation | 0.25 µs | 0.26 µs | 1.04x faster |
| Greeks (vega, gamma) | 1.26 µs | 1.28 µs | 1.02x faster |
| Option chain (5×3) | 1.12 ms | 1.28 ms | **1.14x faster** |

## Summary

**Clang is consistently 15-49% faster than GCC across all major benchmarks!**

### Key Findings:
- **PDE-heavy workloads:** 15-16% faster (American option pricing, IV calculations)
- **Parallel batch processing:** 49% faster (best improvement)
- **Light workloads:** Negligible difference (interpolation, Greeks)
- **Average improvement:** ~15% across core pricing operations

### Notes:
- Both tested with Scalar backend only (no SIMD vectorization)
- SIMD disabled due to Clang linking issues with std::experimental::simd
- Real-world advantage may be smaller with SIMD enabled on GCC
- However, Clang's better scalar code generation is a clear win

### Recommendation:
**Switch to Clang 19 as default compiler** if we can resolve the SIMD linking issues or accept Scalar-only performance (which is still 15% better than GCC).
