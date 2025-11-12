# Contract Chain Performance Benchmarks

## Overview

This benchmark measures the performance of **cross-contract SIMD vectorization** (batch mode) versus traditional single-contract solving for American option pricing.

**Contract chains** are sequences of option contracts with varying parameters:
- Different strikes (e.g., 80-120, ±20% from ATM)
- Different volatilities (e.g., 15%-30%)
- Different maturities (e.g., 0.25-2.0 years)

The benchmark compares two solving strategies:
1. **Single-contract mode**: Solve each contract individually (sequential)
2. **Batch mode**: Solve multiple contracts simultaneously using SIMD vectorization

## Expected Speedup

Batch mode achieves speedup through cross-contract SIMD vectorization:
- **AVX2 (SIMD width = 4)**: ~3-3.5x speedup
- **AVX-512 (SIMD width = 8)**: ~6-7x speedup

The speedup comes from:
- Vectorized spatial operator evaluation (finite difference stencils)
- Vectorized Newton-Raphson iterations
- Vectorized boundary condition application
- Reduced loop overhead and better cache utilization

## Running the Benchmark

### Quick Test (10 contracts only)
```bash
bazel run //tests:contract_chain_benchmark -- --benchmark_filter=10
```

### Full Benchmark Suite (10, 20, 30, 50 contracts)
```bash
bazel run //tests:contract_chain_benchmark
```

### Custom Benchmark Options
```bash
# Run with specific minimum time per benchmark
bazel run //tests:contract_chain_benchmark -- --benchmark_min_time=10.0

# Run only batch mode benchmarks
bazel run //tests:contract_chain_benchmark -- --benchmark_filter=BatchMode

# Generate CSV output
bazel run //tests:contract_chain_benchmark -- --benchmark_format=csv > results.csv
```

## Interpreting Results

### Key Metrics

1. **Time**: Wall-clock time per iteration (milliseconds)
2. **throughput_contracts_per_sec**: Number of contracts solved per second
3. **simd_width**: SIMD vector width being used (4 for AVX2, 8 for AVX-512)
4. **contracts**: Number of contracts in the chain

### Sample Output

```
-------------------------------------------------------------------------------
Benchmark                               Time       CPU   Iterations   Metrics
-------------------------------------------------------------------------------
BM_ContractChain_SingleContract/10   596 ms   596 ms            1
  throughput_contracts_per_sec=16.8/s simd_width=4 contracts=10

BM_ContractChain_BatchMode/10        178 ms   178 ms            4
  throughput_contracts_per_sec=56.2/s simd_width=4 contracts=10
```

**Speedup calculation**: 596ms / 178ms = **3.35x faster** (batch vs single-contract)

### Understanding Performance Characteristics

**Good speedup indicators:**
- Batch throughput >> single-contract throughput
- Speedup ratio close to SIMD width (e.g., 4x for AVX2, 8x for AVX-512)
- Consistent speedup across different contract chain sizes

**Factors affecting speedup:**
- **SIMD width**: Larger SIMD width → better theoretical speedup
- **Chain size**: Larger chains amortize batch setup overhead
- **Grid size**: Larger grids have more vectorizable work
- **Convergence iterations**: More Newton iterations → more vectorizable work
- **Build mode**: DEBUG builds are slower, use optimized builds for production

## Benchmark Configuration

### Grid Parameters
- **n_space**: 101 grid points (realistic spatial discretization)
- **n_time**: 1000 time steps (realistic temporal discretization)
- **x_min, x_max**: -1.0 to 1.0 (log-moneyness space)

### Contract Parameters
- **Strikes**: 80-120 (±20% from ATM)
- **Volatilities**: 15%-30% (realistic market range)
- **Maturities**: 0.25-2.0 years (quarterly to 2-year options)
- **Rate**: 5% (constant risk-free rate)
- **Dividend**: 2% (constant dividend yield)

### Solver Configuration
- **TR-BDF2**: L-stable implicit time-stepping
- **Newton-Raphson**: Implicit solve with Jacobian
- **Tolerance**: 1e-6 (production-quality convergence)
- **Max iterations**: 100 (sufficient for convergence)

## Performance Baseline

Expected timing (on modern CPU with AVX-512):

| Contracts | Single-Contract Time | Batch Mode Time | Expected Speedup |
|-----------|---------------------|-----------------|------------------|
| 10        | ~600ms              | ~90ms           | ~6.7x            |
| 20        | ~1200ms             | ~180ms          | ~6.7x            |
| 30        | ~1800ms             | ~270ms          | ~6.7x            |
| 50        | ~3000ms             | ~450ms          | ~6.7x            |

**Note**: Actual timings depend on CPU, memory bandwidth, and build optimization level.

## Build Optimization

For accurate benchmarks, use optimized builds:

```bash
# Build with optimizations
bazel build -c opt //tests:contract_chain_benchmark

# Run optimized benchmark
bazel run -c opt //tests:contract_chain_benchmark
```

## Related Tests

- `pde_solver_batch_test.cc`: Correctness tests for batch mode
- `batch_convergence_test.cc`: Convergence validation tests
- `batch_transpose_benchmark.cc`: Low-level transpose performance
- `spatial_operator_batch_test.cc`: Spatial operator batch correctness

## Technical Details

### Cross-Contract Vectorization

Batch mode uses **Structure-of-Arrays-of-Structures** (SoAoS) layout:
- Interior grid points: Array-of-Structures (AoS) for SIMD vectorization
- Contract-level data: Structure-of-Arrays (SoA) for per-lane access

**Key operations vectorized:**
- Finite difference stencils (∂²u/∂x², ∂u/∂x)
- Newton-Raphson Jacobian assembly
- Tridiagonal solve (Thomas algorithm)
- Boundary condition application
- Obstacle condition projection

### Memory Layout

```
Single-contract:  [u₀, u₁, u₂, ..., uₙ]
Batch mode (AoS): [u₀⁽⁰⁾, u₀⁽¹⁾, u₀⁽²⁾, u₀⁽³⁾,  ← SIMD vector (4 contracts)
                   u₁⁽⁰⁾, u₁⁽¹⁾, u₁⁽²⁾, u₁⁽³⁾,  ← SIMD vector
                   ...]
```

This layout enables efficient SIMD operations on corresponding grid points across multiple contracts.

## Future Work

- Measure memory bandwidth utilization
- Profile cache miss rates
- Benchmark with different grid sizes
- Test on ARM Neon (SIMD width = 2)
- Add batch mode with homogeneous parameters (simpler, potentially faster)
