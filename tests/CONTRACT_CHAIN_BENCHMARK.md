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

## ⚠️ Build Mode Matters

**WARNING: This benchmark MUST be run with optimized builds to see meaningful speedup results.**

```bash
# CORRECT: Use optimized build
bazel run -c opt //tests:contract_chain_benchmark

# WRONG: Debug build shows little/no speedup
bazel run //tests:contract_chain_benchmark  # Don't do this!
```

**Why this matters:**
- **Debug builds (-c dbg)**: No SIMD vectorization, no inlining, bounds checking enabled
  - Batch mode speedup: ~1.1-1.3x (minimal, not representative)
  - Single-contract performance: ~10x slower than optimized
- **Optimized builds (-c opt)**: Full SIMD vectorization, aggressive inlining, loop unrolling
  - Batch mode speedup: ~3-7x (actual production performance)
  - This is what you want to measure!

If your benchmark shows speedup less than 2x, check your build mode first!

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

## Limitations of Benchmark Implementation

**This benchmark uses a simplified batch mode implementation with the following restrictions:**

### Homogeneous Maturity Per Batch
All contracts in a batch must share the **same maturity** (time to expiration). This simplification allows:
- Single time-stepping loop for all contracts
- Uniform convergence across the batch
- Simplified time grid construction

**Example:** A batch can contain contracts with maturities [1.0, 1.0, 1.0, 1.0] but NOT [0.5, 1.0, 1.5, 2.0].

### Homogeneous Strike (Payoff) Per Batch
All contracts in a batch use the **same strike price** (payoff structure). This simplification allows:
- Uniform obstacle condition evaluation
- Shared payoff computation
- Single early exercise boundary per time step

**Example:** A batch can contain contracts with strikes [100, 100, 100, 100] but NOT [80, 90, 100, 110].

### Variable PDE Coefficients
**What DOES vary within a batch:**
- Volatility (σ): Different implied volatilities per contract
- Interest rate (r): Different risk-free rates per contract
- Dividend yield (q): Different dividend yields per contract

This allows realistic batch processing scenarios like:
- Pricing the same option across different vol surfaces
- Pricing the same option across different rate curves
- Parameter sensitivity analysis (vega, rho, dividend sensitivity)

### Why These Limitations Exist
**This is a benchmark simplification, NOT a fundamental limitation of batch mode:**

The batch vectorization infrastructure itself can handle:
- Different maturities (with more complex time-stepping logic)
- Different strikes (with per-contract payoff evaluation)
- Fully heterogeneous contract chains

However, for this benchmark we prioritize:
1. **Clear performance measurement**: Isolate SIMD vectorization gains
2. **Implementation simplicity**: Avoid complexity that obscures the core algorithm
3. **Realistic use cases**: Most production scenarios batch similar contracts

**Future work:** Extend batch mode to support fully heterogeneous contract chains (different maturities and strikes).

## Future Work

- Measure memory bandwidth utilization
- Profile cache miss rates
- Benchmark with different grid sizes
- Test on ARM Neon (SIMD width = 2)
- Add support for heterogeneous maturities and strikes in batch mode
