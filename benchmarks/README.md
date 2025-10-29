# Performance Benchmarks

This directory contains performance benchmarks for the iv_calc library. **Benchmarks are NOT run in CI** to keep CI fast and focused on correctness testing.

## Available Benchmarks

### 1. `batch_benchmark`
Comprehensive benchmarking suite for American option batch processing.

**Includes:**
- Sequential vs Batch comparison
- Thread scalability (1-32 threads)
- Thread efficiency analysis
- Batch scaling (5-200 options)
- Large batch throughput (500-2000 options)
- Grid resolution impact
- Time step impact
- Implied volatility sequential performance

### 2. `quantlib_benchmark`
Comparison benchmarks against QuantLib (industry-standard library).

**Requires:** `libquantlib0-dev` installed

## Running Benchmarks

### Build All Benchmarks

```bash
bazel build //benchmarks:all
```

### Run Specific Benchmark

```bash
# Run batch benchmark (all tests)
./bazel-bin/benchmarks/batch_benchmark

# Run with filters
./bazel-bin/benchmarks/batch_benchmark --benchmark_filter="ThreadScaling"

# Run with minimum time per test
./bazel-bin/benchmarks/batch_benchmark --benchmark_min_time=2s

# Save results to file
./bazel-bin/benchmarks/batch_benchmark | tee results.txt
```

### Run QuantLib Comparison

```bash
# Install QuantLib first
sudo apt-get install libquantlib0-dev

# Build and run
bazel build //benchmarks:quantlib_benchmark
./bazel-bin/benchmarks/quantlib_benchmark
```

## Benchmark Results

### Thread Scalability (100 options)

| Threads | Time   | Speedup | Throughput    | Efficiency |
|---------|--------|---------|---------------|------------|
| 1       | 706ms  | 1.0x    | 142 opts/s    | 100%       |
| 2       | 358ms  | 2.0x    | 279 opts/s    | 99%        |
| 4       | 179ms  | 3.9x    | 558 opts/s    | 98%        |
| 8       | 97ms   | 7.3x    | 1,031 opts/s  | 91%        |
| 16      | 62ms   | 11.4x   | 1,612 opts/s  | 71%        |
| 32      | 61ms   | 11.6x   | 1,649 opts/s  | 36%        |

### Sequential vs Batch (Default OpenMP)

| Options | Sequential | Batch      | Speedup  |
|---------|-----------|------------|----------|
| 10      | 67ms      | 15ms       | 4.5x     |
| 25      | 178ms     | 22ms       | 8.1x     |
| 50      | 367ms     | 35ms       | 10.5x    |
| 100     | 748ms     | 64ms       | 11.7x    |

### Large Batch Throughput (32 cores)

| Options | Time   | Throughput    |
|---------|--------|---------------|
| 500     | 253ms  | 1,980 opts/s  |
| 1000    | 497ms  | 2,012 opts/s  |
| 2000    | 991ms  | 2,019 opts/s  |

## Interpreting Results

### CPU Time vs Real Time
- **CPU time**: Total CPU time across all threads
- **Real time**: Wall-clock time (what users experience)
- Use `--UseRealTime()` for parallel benchmarks

### Items Per Second
- Higher is better
- Measures throughput (options priced per second)

### Speedup
- Linear speedup: Speedup = N threads
- Superlinear: Speedup > N (rare, usually cache effects)
- Sublinear: Speedup < N (common due to overhead)

### Efficiency
- Efficiency = Speedup / N threads
- 100% = perfect scaling
- >70% = good
- <50% = significant overhead

## Tags

All benchmarks are tagged with:
- `benchmark`: Identifies performance benchmarks
- `manual`: Prevents automatic execution in CI

This means:
- `bazel build //...` will NOT build benchmarks (due to `--build_tag_filters=-benchmark`)
- `bazel test //...` will NOT run benchmarks (cc_binary, not cc_test)
- Must explicitly request: `bazel build //benchmarks:all`

## Why Separate from Tests?

1. **CI Speed**: Benchmarks take minutes; tests take seconds
2. **Noise Sensitivity**: Benchmarks need consistent environment
3. **Dependencies**: QuantLib may not be available in CI
4. **Purpose**: Tests verify correctness; benchmarks measure performance
5. **Frequency**: Tests run on every commit; benchmarks run periodically

## When to Run Benchmarks

- After performance optimizations
- Before/after major refactoring
- When evaluating algorithm changes
- For release performance metrics
- To investigate performance regressions

## See Also

- [BENCHMARK.md](BENCHMARK.md) - Detailed benchmark results and methodology
- [../tests/](../tests/) - Unit and integration tests (run in CI)
- [../CLAUDE.md](../CLAUDE.md) - Project overview and build instructions
