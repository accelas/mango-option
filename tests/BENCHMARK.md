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

## Benchmark Options

Google Benchmark supports various command-line options:

```bash
# Run only specific benchmarks
./bazel-bin/tests/quantlib_benchmark --benchmark_filter=IVCalc

# Output results in JSON format
./bazel-bin/tests/quantlib_benchmark --benchmark_format=json

# Control the minimum benchmark time
./bazel-bin/tests/quantlib_benchmark --benchmark_min_time=2.0

# Show more detailed statistics
./bazel-bin/tests/quantlib_benchmark --benchmark_repetitions=10
```

## Implementation Details

### Grid Parameters

Our implementation uses:
- Grid points: 141
- Time steps: 1000
- Log-moneyness range: [-0.7, 0.7]
- Time step size: 0.001

QuantLib's `FdBlackScholesVanillaEngine` uses:
- Time steps: 100
- Spatial points: 400

### Expected Results

**Accuracy**: The implementations typically match within 0.5-1% relative error. Differences arise from:
- Different discretization schemes (TR-BDF2 vs Crank-Nicolson)
- Different grid resolutions
- Different boundary condition implementations

**Performance**: QuantLib is typically 30-40x faster in debug builds because:
- Highly optimized C++ implementation
- Efficient sparse matrix solvers
- Optimized memory layout

In release builds with optimizations enabled, the performance gap should narrow significantly.

## Validation

This benchmark serves as:
1. **Accuracy validation**: Ensures our implementation produces reasonable results compared to industry-standard library
2. **Performance baseline**: Provides performance comparison against established implementation
3. **Regression testing**: Helps detect accuracy or performance regressions during development

## References

- [QuantLib Documentation](https://www.quantlib.org/)
- [Google Benchmark Documentation](https://github.com/google/benchmark)
- [QuantLib American Option Tests](https://github.com/lballabio/QuantLib/blob/master/test-suite/americanoption.cpp)
