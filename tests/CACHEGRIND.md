# Cachegrind Testing for Cache-Blocking Verification

This directory contains tools for verifying that cache-blocking optimization provides measurable L1 cache performance improvements.

## Overview

The cache-blocking optimization splits large grids into cache-friendly blocks to improve memory locality. Cachegrind (part of Valgrind) provides detailed cache simulation to measure the actual impact on L1/L2/LL cache performance.

## Components

### 1. Cachegrind Harness (`cachegrind_harness.cc`)

Standalone binary that runs the PDE solver in two modes:
- `--with-blocking`: Cache blocking enabled (threshold = ~5461)
- `--without-blocking`: Cache blocking disabled (threshold = 100000)

**Configuration:**
- Grid size: n = 10,000 (above blocking threshold)
- Time steps: 50 (sufficient for cache pattern measurement)
- Problem: Heat equation with Gaussian initial condition
- Operator: Laplacian (pure diffusion)

### 2. Test Runner Script (`scripts/run_cachegrind_test.sh`)

Automated script that:
1. Builds the cachegrind_harness binary
2. Runs it twice through Valgrind/Cachegrind (with and without blocking)
3. Parses cachegrind output files
4. Compares L1 data cache statistics
5. Reports improvement (or lack thereof)

## Usage

### Quick Start

From the repository root:

```bash
./scripts/run_cachegrind_test.sh
```

### Manual Usage

Build the harness:
```bash
bazel build //tests:cachegrind_harness
```

Run with cachegrind:
```bash
# Without blocking
valgrind --tool=cachegrind \
    --cachegrind-out-file=cachegrind.out.no-block \
    ./bazel-bin/tests/cachegrind_harness --without-blocking

# With blocking
valgrind --tool=cachegrind \
    --cachegrind-out-file=cachegrind.out.with-block \
    ./bazel-bin/tests/cachegrind_harness --with-blocking
```

Analyze results:
```bash
cg_annotate cachegrind.out.no-block
cg_annotate cachegrind.out.with-block
```

## Metrics

The test focuses on **L1 data cache (D1)** performance:

| Metric | Description |
|--------|-------------|
| **D refs** | Total data memory references |
| **D1 misses** | L1 data cache misses |
| **LLd misses** | Last-level data cache misses |
| **D1 miss rate** | Percentage: (D1 misses / D refs) × 100 |

### Expected Results

On typical hardware (32KB L1 cache):
- **L1 miss reduction:** 10-30% with cache blocking
- **LL miss reduction:** 5-15% with cache blocking

Results are **hardware-dependent**:
- Large L1 cache (128KB+): Smaller improvement
- Small L1 cache (16KB): Larger improvement
- NUMA systems: Variable results depending on memory locality

## Interpreting Results

### Success Criteria

✓ **SUCCESS:** L1 miss reduction ≥ 5%
- Cache blocking provides measurable benefit
- Grid/block sizes are appropriate for hardware

⚠ **MARGINAL:** 0% < L1 miss reduction < 5%
- Modest improvement (within measurement noise)
- May indicate large L1 cache or other bottlenecks

✗ **FAILURE:** L1 miss reduction ≤ 0%
- No cache benefit (or performance regression)
- Possible causes:
  - Hardware cache larger than expected
  - Cache blocking overhead exceeds benefit
  - Incorrect block size calculation

### Example Output

```
┌─────────────────────────────────────────────────────────────┐
│                    L1 Data Cache Statistics                 │
├─────────────────────────────────────────────────────────────┤
│ Metric                    │      No Blocking │  With Blocking │
├─────────────────────────────────────────────────────────────┤
│ D refs (total)            │    1,234,567,890 │  1,234,567,890 │
│ D1 misses                 │       12,345,678 │     10,123,456 │
│ LLd misses                │        1,234,567 │      1,111,111 │
│ D1 miss rate              │            1.00% │          0.82% │
└─────────────────────────────────────────────────────────────┘

Improvement Summary:
  L1 miss reduction: 18.00%
  LL miss reduction: 10.00%

✓ SUCCESS: Cache blocking reduces L1 misses by 18.00%
```

## Troubleshooting

### Valgrind Not Found

Install valgrind:
```bash
# Ubuntu/Debian
sudo apt-get install valgrind

# macOS (limited support)
brew install valgrind
```

### Bazel Not Found

Install bazelisk:
```bash
# Using npm
npm install -g @bazel/bazelisk

# Or download from GitHub
# https://github.com/bazelbuild/bazelisk/releases
```

### Test Takes Too Long

Cachegrind simulation adds ~10-50× overhead. Expected runtime:
- Native execution: ~2 seconds
- Under cachegrind: ~30-60 seconds per run

To reduce runtime, edit `cachegrind_harness.cc`:
- Reduce grid size: `const size_t n = 5000;` (line 39)
- Reduce time steps: `mango::TimeDomain time(0.0, 0.01, 0.001);` (line 42)

## Technical Details

### Cache Configuration

Cachegrind uses the following default cache model:
- **L1 I-cache:** 32KB, 8-way, 64B line
- **L1 D-cache:** 32KB, 8-way, 64B line
- **LL cache:** 8MB, 16-way, 64B line

Override with flags:
```bash
valgrind --tool=cachegrind \
    --I1=65536,8,64 \
    --D1=65536,8,64 \
    --LL=16777216,16,64 \
    ./cachegrind_harness --with-blocking
```

### Block Size Calculation

Cache blocking targets **L1 D-cache** (32KB typical):
- Working set per point: 24 bytes (u_current, u_next, workspace)
- L1 optimal block: 32KB / 24B ≈ 1365 points
- Threshold for activation: 4× optimal ≈ 5461 points

See `src/cpp/cache_config.hpp` for implementation.

## References

- [Valgrind Cachegrind Documentation](https://valgrind.org/docs/manual/cg-manual.html)
- [Cache-Oblivious Algorithms (Frigo et al.)](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm)
- Project documentation: `CLAUDE.md` (Cache-Blocking Optimization section)

## See Also

- `tests/cache_blocking_benchmark.cc` - Wall-clock time comparison
- `src/cpp/cache_config.hpp` - Cache configuration implementation
- `src/cpp/workspace.hpp` - Workspace with cache-blocking support
