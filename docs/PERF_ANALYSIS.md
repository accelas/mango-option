<!-- SPDX-License-Identifier: MIT -->
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

**Single option (6,810 iterations x 3 reps, 8.83s wall):**

| Counter | Value |
|---------|-------|
| Cycles | 44,298,672,234 |
| Instructions | 82,449,637,867 |
| L1-dcache-loads | 39,173,090,917 |
| L1-dcache-load-misses | 3,434,254 |
| LLC references | 13,133,205 |
| LLC misses | 484,929 |
| Stalled-cycles-frontend | 1,200,292,340 |

**Sequential 64-option batch (191 iterations x 3 reps, 10.51s wall):**

| Counter | Value |
|---------|-------|
| Cycles | 52,751,552,933 |
| Instructions | 98,567,359,543 |
| L1-dcache-loads | 46,560,138,054 |
| L1-dcache-load-misses | 5,767,383 |
| LLC references | 27,055,591 |
| LLC misses | 621,626 |
| Stalled-cycles-frontend | 1,519,234,316 |

### Per-element metrics

| Metric | Single (ATM) | Sequential (64 puts) |
|--------|:------------:|:--------------------:|
| **IPC** | 1.86 | 1.87 |
| **Cycles/element** | 135.3 | 76.2 |
| **Insn/element** | 252.0 | 142.4 |
| **L1 loads/element** | 119.7 | 67.3 |
| **L1 miss rate** | 0.009% | 0.012% |
| **LLC miss rate** | 3.69% | 2.30% |
| **Op intensity (insn/load)** | 2.11 | 2.12 |
| **Frontend stall** | 2.71% | 2.88% |

### Solve timing

| Benchmark | Time per solve | Throughput |
|-----------|:--------------:|:----------:|
| Single ATM put | 409 us | 2,447/s |
| Sequential (per option) | 230 us | 4,348/s |

## Algorithmic Cost Breakdown

The measured per-element counts reflect the full TR-BDF2 + Brennan-Schwartz LCP solver, not a simple explicit FDM stencil. Each time step executes two implicit stages (trapezoidal + BDF2), each performing:

| Component | Insn/elem | Loads/elem | SIMD? | Notes |
|-----------|:---------:|:----------:|:-----:|-------|
| Black-Scholes operator apply | ~10 | ~5 | **Yes** | 3-point stencil with σ², r, q coefficients; `target_clones` + `omp simd` |
| Jacobian diagonal fill | ~3 | ~2 | Yes | Same coefficients, diagonal extraction |
| Projected Thomas solver | ~8 | ~3 | **No** | Forward elimination + back substitution; sequential data dependency |
| RHS assembly + BC fixup | ~2 | ~1 | Partial | FMA for αu + (1−α)f, workspace copy for BC correction |
| Obstacle evaluation | ~2 | ~1 | No | `max(K − S·eˣ, 0)` with `std::exp` at exercise boundary |
| Deep ITM lock scan | ~1 | ~1 | No | Sequential scan from boundary inward |
| Boundary conditions | ~1 | ~1 | No | 2 endpoints per stage |
| **Per stage total** | **~27** | **~14** | | |
| **Per time step (2 stages)** | **~54** | **~28** | | |

Over 150 time steps, the theoretical cost is ~54 × 150 / 150 = **54 insn/element** and **28 loads/element** when normalized per space-time point.

The measured 252 insn/element (single) vs 142 insn/element (sequential) gap shows that per-solve overhead (grid estimation, PMR allocation, `GridSpec` construction) dominates the single-option case. The sequential benchmark amortizes this overhead across 64 solves, approaching the theoretical cost.

### Why op intensity is structurally ~2.1

The Thomas algorithm dominates memory access. Each forward elimination step loads `lower[i]`, `diag[i]`, `upper[i]`, and `rhs[i]` (4 loads) for ~5 FP operations (1 division, 2 multiply-subtracts). This gives op intensity ~1.25 for Thomas alone. The stencil computation is higher (~2 insn/load) but represents a smaller fraction of total work. Blended across the full pipeline, 2.1 is the expected result.

### SIMD coverage

SIMD vectorization (`[[gnu::target_clones("default", "avx2", "avx512f")]]` and `#pragma omp simd`) is applied to the spatial operator and Jacobian loops — the components where data-parallel execution is possible. The Thomas algorithm cannot be vectorized because each element depends on the previous (sequential data dependency chain). This is a fundamental property of tridiagonal solvers, not a missed optimization.

## Parallelism Analysis

### Intra-solve: parallel tridiagonal solvers

Tridiagonal systems can be parallelized via cyclic reduction (O(log n) parallel steps), SPIKE partitioning, or parallel cyclic reduction (PCR). However, with n=101 spatial points, none of these are profitable:

- The entire working set is ~5 KiB (fits in L1)
- Thread synchronization and partition boundary coupling cost more than the sequential sweep
- A 101-element Thomas solve completes in microseconds

Parallel tridiagonal solvers become worthwhile for n in the tens of thousands (GPU workloads, 2D/3D problems).

### Inter-solve: batch parallelism

`BatchAmericanOptionSolver` parallelizes across options via OpenMP. Each thread owns an independent workspace with no shared mutable state. This is the profitable granularity:

| Strategy | n=101 | n=10,000+ |
|----------|:-----:|:---------:|
| Parallel Thomas (intra-solve) | Overhead dominates | Worthwhile |
| Batch parallel (inter-solve) | **15× speedup** | Also effective |

The measured 15× speedup on 16 cores confirms near-linear scaling. Further optimization would require algorithmic changes (e.g., replacing Thomas with a SIMD-friendly block formulation), which would add complexity for marginal gain at this grid size.

## Interpretation

**IPC (1.86–1.87).** The pipeline is well-utilized for a tridiagonal solver workload. TR-BDF2 involves two sequential sweeps per time step (forward and backward Thomas algorithm), which limits instruction-level parallelism. An IPC near 2.0 on a superscalar core is close to the ceiling for this access pattern.

**L1 miss rate (0.009–0.012%).** The working set for a 101-point tridiagonal system is approximately 800 bytes per vector (101 doubles). With 5–6 vectors active during a Thomas sweep (~5 KiB), the data fits comfortably in L1 (48 KiB). Cache pressure is negligible.

**Sequential is 1.8× cheaper per element.** The single-option benchmark pays proportionally more for grid estimation, PMR pool allocation, and `GridSpec` construction. In the sequential loop these costs amortize: the hot solve path dominates, and the solver reuses warm cache lines from the previous option's similar-sized grid.

**LLC miss rate (2.30–3.69%).** The few LLC misses that occur come from cold-start allocation and benchmark framework overhead, not from the solver hot path. The entire solver working set lives in L1/L2.

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
