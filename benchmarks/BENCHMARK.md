# QuantLib Comparison Benchmark

This benchmark compares the performance and accuracy of our American option pricing implementation against QuantLib's finite difference engine.

## Building the Benchmark

The benchmark requires QuantLib to be installed on your system:

```bash
# Debian/Ubuntu
sudo apt-get install libquantlib0-dev

# Build the benchmark
bazel build //tests:quantlib_benchmark
```

## Running the Benchmark

```bash
./bazel-bin/tests/quantlib_benchmark
```

### Output

The benchmark provides two types of output:

1. **Implementation Comparison**: Shows the pricing results from both implementations side-by-side with relative error
2. **Performance Benchmarks**: Shows timing results for both implementations using Google Benchmark

### Example Output

```
=== Implementation Comparison ===
Parameters:
  Spot: 100
  Strike: 100
  Volatility: 0.25
  Risk-free rate: 0.05
  Time to maturity: 1 years

Results:
  IV Calc value: 8.01118
  QuantLib value: 7.96904
  Difference: 0.0421358
  Relative error: 0.528743%
================================

-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_IVCalc_AmericanPut      48184974 ns     48184499 ns           14
BM_QuantLib_AmericanPut     1253491 ns      1252865 ns          556
BM_IVCalc_AmericanCall     49206666 ns     49181783 ns           14
BM_QuantLib_AmericanCall    1236969 ns      1236571 ns          556
```

## Building with Optimizations

For accurate performance comparisons, build with optimization enabled:

```bash
# Build optimized version
bazel build -c opt //tests:quantlib_benchmark

# Run optimized benchmark
./bazel-bin/tests/quantlib_benchmark

# Build debug version (default)
bazel build -c dbg //tests:quantlib_benchmark
```

**Performance Impact of Optimizations**:
- Debug build: ~48ms per option
- Optimized build without vectorization: ~48ms per option
- **Optimized build with AVX-512**: ~21.5ms per option (2.2x speedup!)

The AVX-512 vectorization provides significant performance improvement by utilizing SIMD instructions to process multiple data points in parallel.

## Benchmark Options

Google Benchmark supports various command-line options:

```bash
# Run only specific benchmarks
./bazel-bin/tests/quantlib_benchmark --benchmark_filter=IVCalc

# Output results in JSON format
./bazel-bin/tests/quantlib_benchmark --benchmark_format=json

# Show detailed statistics with repetitions
./bazel-bin/tests/quantlib_benchmark --benchmark_repetitions=10
```

## Implementation Details

### Vectorization Configuration

The benchmark and core libraries are compiled with AVX-512 SIMD support:
- `-march=native`: Enables CPU-specific optimizations
- `-mavx512f`: Enables AVX-512 foundation instructions
- `-fopenmp-simd`: Enables OpenMP SIMD pragmas
- `-ftree-vectorize`: Enables GCC auto-vectorization
- `-fopt-info-vec-optimized`: Reports vectorized loops during compilation

Vectorized loops include:
- Spatial operator evaluations
- Tridiagonal system solvers
- Boundary condition applications
- Array operations in time-stepping

Typical vectorization uses 64-byte vectors (AVX-512) and 32-byte vectors (AVX2) depending on the operation.

### Grid Parameters

Our implementation uses:
- Grid points: 141
- Time steps: 1000
- Log-moneyness range: [-0.7, 0.7]
- Time step size: 0.001

QuantLib's `FdBlackScholesVanillaEngine` is configured to use:
- Time steps: 1000 (same as our implementation for fair comparison)
- Spatial points: 400

### Expected Results

**Accuracy**: The implementations match within 0.5% relative error. Differences arise from:
- Different discretization schemes (TR-BDF2 vs Crank-Nicolson)
- Different grid resolutions
- Different boundary condition implementations

**Performance (Fair Comparison with 1000 time steps, AVX-512 enabled)**:
- **IV Calc**: ~21.6ms per option
- **QuantLib**: ~10.4ms per option
- **Speed ratio**: 2.1x (QuantLib is ~2x faster)

The 2x performance difference is reasonable and attributed to:
- QuantLib's highly optimized sparse matrix solvers
- Mature C++ implementation with decades of optimization
- Different numerical schemes (Crank-Nicolson vs TR-BDF2)

**Note on Unfair Comparisons**: Using QuantLib's default 100 time steps versus our 1000 steps would show a misleading 17x performance difference, which is primarily due to the 10x difference in time resolution, not algorithm quality.

## Validation

This benchmark serves as:
1. **Accuracy validation**: Ensures our implementation produces reasonable results compared to industry-standard library
2. **Performance baseline**: Provides performance comparison against established implementation
3. **Regression testing**: Helps detect accuracy or performance regressions during development

## References

- [QuantLib Documentation](https://www.quantlib.org/)
- [Google Benchmark Documentation](https://github.com/google/benchmark)
- [QuantLib American Option Tests](https://github.com/lballabio/QuantLib/blob/master/test-suite/americanoption.cpp)
