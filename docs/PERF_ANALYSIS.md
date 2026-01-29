# Instruction-Level Performance Analysis

Microarchitectural profiling of the American option PDE solver using `perf stat` on real market data with auto-estimated grids.

## Methodology

### Hardware

- CPU: 32-core x86_64 @ 5058 MHz
- L1D: 48 KiB per core, L2: 1024 KiB per core, L3: 32 MiB shared
- Measured on Linux 6.12

### Workload

Two benchmarks from `//benchmarks:real_data_benchmark`, both using `estimate_grid_for_option()` for automatic sinh-spaced grid estimation:

| Benchmark | Description | Options | Grid per option |
|-----------|-------------|---------|-----------------|
| `BM_RealData_AmericanSingle` | Single ATM put (K=675, T=0.090) | 1 | 101 x 150 |
| `BM_RealData_AmericanSequential` | 64 real SPY puts, sequential | 64 | ~101 x 150 |

Grid dimensions are auto-estimated from option parameters (volatility, maturity, moneyness). The default `GridAccuracyParams` produces 101 sinh-spaced spatial points and 150 time steps for short-dated SPY options, yielding 15,150 grid elements per solve.

Each time step performs a TR-BDF2 composite solve: a trapezoidal stage followed by a BDF2 stage, each requiring a tridiagonal system solve across the spatial grid.

### Counters

```bash
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,\
cache-references,cache-misses,stalled-cycles-frontend \
  bazel-bin/benchmarks/real_data_benchmark \
  --benchmark_filter='BM_RealData_AmericanSingle' \
  --benchmark_repetitions=3 --benchmark_min_time=2s
```

Built with `bazel build -c opt` (O3, native).

### Derived metrics

- **Cycles/element** = total cycles / (solves x n_space x n_time)
- **IPC** = instructions / cycles
- **Op intensity** = instructions / L1 data loads (proxy for arithmetic-to-memory ratio; higher means more compute per byte accessed)

## Results

### Raw perf counters

**Single option (6,763 iterations x 3 reps, 8.88s wall):**

| Counter | Value |
|---------|-------|
| Cycles | 44,686,901,251 |
| Instructions | 85,896,059,833 |
| L1-dcache-loads | 41,032,959,107 |
| L1-dcache-load-misses | 2,141,162 |
| LLC references | 8,305,151 |
| LLC misses | 327,595 |
| Stalled-cycles-frontend | 1,760,678,737 |

**Sequential 64-option batch (190 iterations x 3 reps, 10.53s wall):**

| Counter | Value |
|---------|-------|
| Cycles | 52,889,294,167 |
| Instructions | 102,839,667,045 |
| L1-dcache-loads | 48,958,724,505 |
| L1-dcache-load-misses | 7,342,879 |
| LLC references | 30,591,722 |
| LLC misses | 571,363 |
| Stalled-cycles-frontend | 1,486,541,338 |

### Per-element metrics

| Metric | Single (ATM) | Sequential (64 puts) |
|--------|:------------:|:--------------------:|
| **IPC** | 1.92 | 1.94 |
| **Cycles/element** | 137.5 | 76.7 |
| **Insn/element** | 264.3 | 149.0 |
| **L1 loads/element** | 126.3 | 71.0 |
| **L1 miss rate** | 0.005% | 0.015% |
| **LLC miss rate** | 3.94% | 1.87% |
| **Op intensity (insn/load)** | 2.09 | 2.10 |
| **Frontend stall** | 3.94% | 2.81% |

### Solve timing

| Benchmark | Time per solve | Throughput |
|-----------|:--------------:|:----------:|
| Single ATM put | 414 us | 2,415/s |
| Sequential (per option) | 231 us | 4,325/s |

## Interpretation

**IPC (1.92-1.94).** The pipeline is well-utilized for a tridiagonal solver workload. TR-BDF2 involves two sequential sweeps per time step (forward and backward Thomas algorithm), which limits instruction-level parallelism. An IPC near 2.0 on a superscalar core is close to the ceiling for this access pattern.

**Op intensity (2.1 insn/load).** Each L1 load supports about 2 instructions of arithmetic. This is consistent with a banded linear solver sweeping contiguous double arrays: each element requires a load, a multiply-add, and a store. The workload is balanced between compute and memory — not purely memory-bound, not purely compute-bound.

**L1 miss rate (0.005-0.015%).** The working set for a 101-point tridiagonal system is approximately 800 bytes per vector (101 doubles). With 5-6 vectors active during a Thomas sweep (~5 KiB), the data fits comfortably in L1 (48 KiB). Cache pressure is negligible.

**Sequential is 1.8x cheaper per element.** The single-option benchmark pays proportionally more for grid estimation, PMR pool allocation, and `GridSpec` construction. In the sequential loop these costs amortize: the hot solve path dominates, and the solver reuses warm cache lines from the previous option's similar-sized grid.

**LLC miss rate (1.87-3.94%).** The few LLC misses that occur come from cold-start allocation and benchmark framework overhead, not from the solver hot path. The entire solver working set lives in L1/L2.

## Reproducing

```bash
# Build optimized
bazel build -c opt //benchmarks:real_data_benchmark

# Single option with grid counters
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,\
cache-references,cache-misses,stalled-cycles-frontend \
  bazel-bin/benchmarks/real_data_benchmark \
  --benchmark_filter='BM_RealData_AmericanSingle' \
  --benchmark_repetitions=3 --benchmark_min_time=2s

# Sequential 64-option batch
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,\
cache-references,cache-misses,stalled-cycles-frontend \
  bazel-bin/benchmarks/real_data_benchmark \
  --benchmark_filter='BM_RealData_AmericanSequential' \
  --benchmark_repetitions=3 --benchmark_min_time=2s
```

For stable results, disable CPU frequency scaling and ASLR:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
```
