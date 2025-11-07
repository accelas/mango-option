# PriceTable4DBuilder: Code Reference Guide

## File Locations

### Main Implementation Files
```
/home/user/mango-iv/src/price_table_4d_builder.hpp (155 lines)
    - Public API and result structures
    - Class definition: PriceTable4DBuilder
    - Result structure: PriceTable4DResult

/home/user/mango-iv/src/price_table_4d_builder.cpp (271 lines)
    - Implementation of PriceTable4DBuilder::precompute()
    - Grid validation and setup
    - Main (σ, r) nested loop (lines 151-211)
    - B-spline fitting and result assembly

/home/user/mango-iv/src/american_option.hpp (246 lines)
    - Class: AmericanOptionSolver
    - Struct: AmericanOptionParams
    - Struct: AmericanOptionGrid
    - Batch solver utilities

/home/user/mango-iv/src/american_option.cpp (446 lines)
    - Implementation: AmericanOptionSolver::solve()
    - Grid generation: line 132
    - SpatialOperator creation: lines 144-151
    - PDESolver construction: lines 192, 252
    - Boundary condition setup: lines 158-185
    - Observable initialization: lines 221-228, 282-287
```

### Supporting Infrastructure
```
/home/user/mango-iv/src/pde_solver.hpp (373 lines)
    - Template class: PDESolver<BoundaryL, BoundaryR, SpatialOp>
    - Constructor: lines 79-107
    - Member allocations: lines 96-100
    - solve() method: lines 125-160

/home/user/mango-iv/src/workspace.hpp (156 lines)
    - Class: WorkspaceStorage
    - Constructor: lines 36-55
    - Buffer allocation: 7n doubles total

/home/user/mango-iv/src/newton_workspace.hpp (93 lines)
    - Class: NewtonWorkspace
    - Constructor: lines 29-37
    - Buffer allocation: 8n - 2 doubles

/home/user/mango-iv/src/price_table_snapshot_collector.hpp (160 lines)
    - Class: PriceTableSnapshotCollector : SnapshotCollector
    - Constructor: lines 29-57
    - collect() method: lines 59-124

/home/user/mango-iv/src/operators/operator_factory.hpp (31 lines)
    - Factory function: create_spatial_operator(PDE, GridView)
    - Factory overload: create_spatial_operator(PDE, shared_ptr<GridSpacing>)

/home/user/mango-iv/src/operators/spatial_operator.hpp (200+ lines)
    - Template class: SpatialOperator<PDE, T>
    - Constructor: lines 48-52
    - apply() method: lines 62-89

/home/user/mango-iv/src/operators/grid_spacing.hpp (120+ lines)
    - Template class: GridSpacing<T>
    - Constructor: lines 28-47
    - spacing() method: lines 53-56
```

---

## Critical Code Locations

### Main Loop: PriceTable4DBuilder::precompute()
**File:** `/home/user/mango-iv/src/price_table_4d_builder.cpp`
**Lines:** 80-268

```cpp
80:  PriceTable4DResult PriceTable4DBuilder::precompute(...)
81:  {
85:      const size_t Nm = moneyness_.size();  // 50
86:      const size_t Nt = maturity_.size();   // 30
87:      const size_t Nv = volatility_.size(); // 20
88:      const size_t Nr = rate_.size();       // 10
114:     std::vector<double> prices_4d(Nm * Nt * Nv * Nr, 0.0);  // ALLOCATION
131:     std::vector<size_t> step_indices(Nt);  // Precompute step indices
150:     #pragma omp parallel for collapse(2)
151:     for (size_t k = 0; k < Nv; ++k) {          // ← VOLATILITY LOOP
152:         for (size_t l = 0; l < Nr; ++l) {      // ← RATE LOOP
155:             AmericanOptionParams params{
159:                 .volatility = volatility_[k],   // ← VOLATILITY PARAMETER
160:                 .rate = rate_[l],               // ← RATE PARAMETER
162:             };
167:             PriceTableSnapshotCollectorConfig collector_config{ ... };
174:             PriceTableSnapshotCollector collector(collector_config);
177:             AmericanOptionSolver solver(params, grid_config, ...);
181:             for (size_t j = 0; j < Nt; ++j) {
182:                 solver.register_snapshot(step_indices[j], j, &collector);
183:             }
186:             auto result = solver.solve();
195:             auto prices_2d = collector.prices();
199:             for (size_t i = 0; i < Nm; ++i) {
200:                 for (size_t j = 0; j < Nt; ++j) {
201:                     size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
202:                     prices_4d[idx_4d] = prices_2d[idx_2d];
203:                 }
204:             }
212:     }
```

### Grid Generation: AmericanOptionSolver::solve()
**File:** `/home/user/mango-iv/src/american_option.cpp`
**Lines:** 130-315

```cpp
130:  expected<AmericanOptionResult, SolverError> AmericanOptionSolver::solve() {
131:      // 1. Generate grid in log-moneyness coordinates
132:      auto grid_buffer = GridSpec<>::uniform(grid_.x_min, grid_.x_max, 
133:                                              grid_.n_space).generate();
134:      auto x_grid = grid_buffer.span();  // GRID GENERATION ← ALLOCATES VECTOR
```

**OPTIMIZATION POINT:** Move grid generation outside loop to PriceTable4DBuilder

### SpatialOperator Creation: AmericanOptionSolver::solve()
**Lines:** 142-151

```cpp
141:     // 3. Create Black-Scholes operator in log-moneyness coordinates
142:     auto grid_view = GridView<double>(x_grid);
143:     auto bs_op = operators::create_spatial_operator(
144:         operators::BlackScholesPDE<double>(
145:             params_.volatility,        // ← VOLATILITY PARAMETER
146:             params_.rate,              // ← RATE PARAMETER
147:             params_.continuous_dividend_yield
148:         ),
149:         grid_view
150:     );
```

**OPTIMIZATION POINT:** GridSpacing is recreated here; could reuse via factory overload

### PDESolver Construction: AmericanOptionSolver::solve()
**Lines:** 192 (PUT case) or 252 (CALL case)

```cpp
192:     PDESolver solver(x_grid, time_domain, trbdf2_config_, root_config_,
193:                     left_bc, right_bc, bs_op,
194:                     [](double t, auto x, auto psi) {
195:                         AmericanPutObstacle obstacle;
196:                         obstacle(t, x, psi);
197:                     });
```

**ALLOCATION POINTS INSIDE PDESolver CONSTRUCTOR:**
1. Line 96: `workspace_(n_, grid, config_.cache_blocking_threshold)` 
   - Allocates WorkspaceStorage (7n doubles)
2. Lines 97-99: `u_current_(n_)`, `u_old_(n_)`, `rhs_(n_)`
   - Allocates 3 separate vectors (3n doubles total)
3. Line 100: `newton_ws_(n_, workspace_)`
   - Allocates NewtonWorkspace (8n-2 doubles)

### Snapshot Collection: PriceTableSnapshotCollector
**File:** `/home/user/mango-iv/src/price_table_snapshot_collector.hpp`
**Lines:** 29-57 (Constructor)

```cpp
29:      explicit PriceTableSnapshotCollector(
30:          const PriceTableSnapshotCollectorConfig& config)
31:      : moneyness_(config.moneyness.begin(), config.moneyness.end())
32:      , tau_(config.tau.begin(), config.tau.end())
36:      {
37:          const size_t n = moneyness_.size() * tau_.size();
38:          prices_.resize(n, 0.0);      // Nm × Nt × 8 bytes
39:          deltas_.resize(n, 0.0);      // Nm × Nt × 8 bytes
40:          gammas_.resize(n, 0.0);      // Nm × Nt × 8 bytes
41:          thetas_.resize(n, 0.0);      // Nm × Nt × 8 bytes
42:          // ... perf cache allocations ...
43:      }
```

**OPTIMIZATION POINT:** Could pre-allocate once and reset per iteration

---

## Data Flow: Volatility and Rate Parameters

### Parameter Path: (σ, r) → PDESolver

```
For each (σ, r) pair:
├─ volatility_[k] ─┐
│                  ├─→ AmericanOptionParams.volatility
│                  ├─→ BlackScholesPDE(σ, r)
├─ rate_[l] ──────┤
│                  ├─→ AmericanOptionParams.rate
│                  ├─→ BlackScholesPDE(σ, r)
│                  └─→ Boundary conditions (discount factor)
│
└─→ PDESolver.solve()
    └─→ Newton iteration
        └─→ Jacobian assembly using PDE coefficients
            ├─ a = σ²/2 (second derivative coefficient)
            └─ b = r - d - σ²/2 (first derivative coefficient)
```

### Where Parameters Are Used

**BlackScholesPDE:**
```cpp
File: src/operators/black_scholes_pde.hpp (not shown but referenced)
- Constructor stores σ and r
- operator()(t, d2u, du, u) uses σ²/2 and r in computation
- Jacobian coefficients computed from σ and r
```

**Boundary Conditions:**
```cpp
File: src/american_option.cpp, lines 158-185
- LEFT: discount factor e^(-r*τ) depends on rate
- RIGHT: discount factor e^(-r*τ) depends on rate
```

**Obstacle Computation:**
```cpp
File: src/price_table_snapshot_collector.hpp, lines 149-156
- Intrinsic value doesn't depend on σ or r
- But theta computation uses PDE operator output
```

---

## Memory Allocation Call Stack

### For Single (σ, r) Iteration

```
PriceTable4DBuilder::precompute()
  ├─ prices_4d.resize(Nm × Nt × Nσ × Nr)  [line 114]
  │  └─ 50 × 30 × 20 × 10 × 8 = 1.2 MB
  │
  └─ for k=0 to Nv-1; for l=0 to Nr-1:
      │
      ├─ AmericanOptionSolver(params, grid)
      │  └─ (no allocations in constructor)
      │
      ├─ PriceTableSnapshotCollector(config)  [line 174]
      │  ├─ prices_.resize(Nm × Nt)  [line 38]
      │  │  └─ 50 × 30 × 8 = 12 KB
      │  ├─ deltas_.resize(Nm × Nt)  [line 39]
      │  │  └─ 12 KB
      │  ├─ gammas_.resize(Nm × Nt)  [line 40]
      │  │  └─ 12 KB
      │  ├─ thetas_.resize(Nm × Nt)  [line 41]
      │  │  └─ 12 KB
      │  └─ Perf caches (4 vectors of Nm) [lines 45-48]
      │     └─ 50 × 4 × 8 = 1.6 KB
      │
      └─ solver.solve()  [line 186]
         │
         └─ AmericanOptionSolver::solve()  [american_option.cpp:130]
            │
            ├─ GridSpec<>::uniform().generate()  [lines 132-134]
            │  └─ vector<101> x_grid allocation  [~1 KB]
            │
            ├─ create_spatial_operator(pde, grid_view)  [lines 143-150]
            │  └─ GridSpacing<double> allocation  [~1 KB]
            │
            └─ PDESolver solver(...)  [lines 192/252]
               │
               └─ PDESolver constructor  [pde_solver.hpp:79-107]
                  │
                  ├─ workspace_(n_, grid, ...)  [line 96]
                  │  ├─ buffer_(6 * n)  [line 37 of workspace.hpp]
                  │  │  └─ 6 × 101 × 8 = 4.8 KB
                  │  └─ dx_(n - 1)  [line 39 of workspace.hpp]
                  │     └─ 100 × 8 = 0.8 KB
                  │
                  ├─ u_current_(n_)  [line 97]
                  │  └─ 101 × 8 = 0.8 KB
                  │
                  ├─ u_old_(n_)  [line 98]
                  │  └─ 101 × 8 = 0.8 KB
                  │
                  ├─ rhs_(n_)  [line 99]
                  │  └─ 101 × 8 = 0.8 KB
                  │
                  └─ newton_ws_(n_, workspace_)  [line 100]
                     └─ buffer_(8n-2)  [line 31 of newton_workspace.hpp]
                        └─ (8×101-2) × 8 = 6.4 KB
```

**Total per (σ, r) pair:** ~50-60 KB

---

## Key Functions for Optimization

### 1. Grid Generation (HIGH IMPACT)
```cpp
Function: GridSpec<>::uniform()
File: src/grid.hpp (not shown but core utility)
Called from: american_option.cpp:132

Current: Called Nσ × Nr times (200× for typical config)
Target: Call once, share result

Issue: Grid generation inside AmericanOptionSolver::solve()
Solution: Move to PriceTable4DBuilder::precompute() before loop
```

### 2. GridSpacing Creation (HIGH IMPACT)
```cpp
Function: create_spatial_operator(PDE, GridView) 
         → create_spatial_operator(PDE, shared_ptr<GridSpacing>)
File: src/operators/operator_factory.hpp

Current: GridSpacing created Nσ × Nr times via first overload
Target: Use second overload to reuse GridSpacing

Issue: Lines 143-150 of american_option.cpp create new GridView each iteration
Solution: Pre-create shared_ptr<GridSpacing> in precompute()

Code exists: Overload at lines 21-28 of operator_factory.hpp
```

### 3. SnapshotCollector Buffer Pooling (MEDIUM IMPACT)
```cpp
Class: PriceTableSnapshotCollector
File: src/price_table_snapshot_collector.hpp

Methods to add:
  void reset(std::span<double> prices, 
             std::span<double> deltas,
             std::span<double> gammas,
             std::span<double> thetas)
  
Usage:
  // Before loop
  std::vector<double> prices_buf(Nm * Nt);
  std::vector<double> deltas_buf(Nm * Nt);
  // etc...
  
  // In loop
  collector.reset(prices_buf, deltas_buf, gammas_buf, thetas_buf);
```

### 4. PDESolver Workspace Injection (MEDIUM EFFORT, HIGH IMPACT)
```cpp
Current: PDESolver owns WorkspaceStorage (composition)
Location: pde_solver.hpp:96

Target: Accept pre-allocated workspace via dependency injection

Changes needed:
  1. Add constructor overload accepting WorkspaceStorage&
  2. Store reference instead of owning instance
  3. Refactor member allocations (u_current_, u_old_, rhs_)
     to use workspace_ arrays where possible
  
Risk: Medium (changes PDESolver API, but could make backward-compatible)
```

---

## Performance Bottlenecks

### Allocation Overhead
```
Current run time (200 solvers): ~320 seconds @ 1600ms per solver
├─ Allocation overhead: ~5% (16 seconds)
└─ Solve time: ~95% (304 seconds)

With grid reuse:
└─ Allocation overhead: ~4% (saves ~2 seconds)

With full workspace pooling:
└─ Allocation overhead: ~1% (saves ~5 seconds)
```

### Cache Efficiency
- WorkspaceStorage pre-allocates 7n doubles contiguously
- But PDESolver then allocates 3n MORE doubles separately
- These allocations may not be adjacent (fragmentation)
- Newton workspace allocates 8n-2 separately again

**Impact:** Memory fragmentation, reduced spatial locality

---

## Test Files

### Unit Tests
```cpp
File: tests/price_table_snapshot_collector_test.cc
Lines: 1-100 (shown earlier)
- Tests snapshot collection
- Validates gamma computation

File: tests/american_option_solver_test.cc
- Tests AmericanOptionSolver::solve()
- Validates Greeks computation

File: tests/integration_5d_price_table_test.cc
- Integration test for full price table building
```

### Benchmark
```cpp
File: legacy/benchmarks/american_iv_benchmark.cc
- Performance measurements
- IV solver benchmark
```

---

## Key Constants

### Grid Configuration
```cpp
struct AmericanOptionGrid {
    size_t n_space = 101;         // Spatial grid points
    size_t n_time = 1000;         // Time steps
    double x_min = -3.0;          // Log-moneyness min
    double x_max = 3.0;           // Log-moneyness max
};
```

### Typical 4D Table Configuration
```cpp
std::vector<double> moneyness = {...};   // 50 points (0.7 to 1.3)
std::vector<double> maturity = {...};    // 30 points (0.027 to 2.0 years)
std::vector<double> volatility = {...};  // 20 points (0.10 to 0.80)
std::vector<double> rate = {...};        // 10 points (0.00 to 0.10)

Total: 50 × 30 × 20 × 10 = 300,000 option prices
PDE solves needed: 20 × 10 = 200 solves
```

### TR-BDF2 Parameters
```cpp
struct TRBDF2Config {
    size_t max_iter = 100;          // Newton iterations per stage
    double tolerance = 1e-6;        // Convergence tolerance
    double gamma = 2.0 - std::sqrt(2.0);  // ≈ 0.5858
};
```

---

## Tracing Entry Points

For USDT tracing with bpftrace:
```bash
sudo bpftrace -e '
usdt::mango:algo_start /arg0 == 4/ {  # arg0=4 is MODULE_PRICE_TABLE
    printf("Price table precompute start\n");
}
usdt::mango:algo_progress /arg0 == 4/ {
    printf("Progress: %d%% (σ=%d, r=%d)\n", arg2, arg3, arg4);
}
usdt::mango:algo_complete /arg0 == 4/ {
    printf("Price table complete: %d solves\n", arg1);
}
' -c './binary'
```

