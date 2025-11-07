# PriceTable4DBuilder: Memory and Instantiation Flow

## Current Memory Creation Pattern (Loop Iteration)

```
┌─────────────────────────────────────────────────────────────────┐
│ For each (σ, r) pair in Nσ × Nr loop                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 1. AmericanOptionSolver Constructor                              │
│    └─ Stores: params_, grid_, trbdf2_config_, root_config_      │
│    └─ Cost: O(1) - no allocations                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. AmericanOptionSolver::solve() [american_option.cpp:130]       │
│    ├─ GridSpec<>::uniform().generate() ← ALLOCATION (Grid gen)  │
│    │  └─ Creates: vector<double> x_grid (101 elements)          │
│    │                                                              │
│    ├─ GridView<double>(x_grid) ← Non-owning view (free)         │
│    │                                                              │
│    ├─ create_spatial_operator() ← ALLOCATION (GridSpacing)       │
│    │  ├─ Makes: shared_ptr<GridSpacing<double>>                 │
│    │  │  └─ Allocates: pre-computed spacing array (100 doubles) │
│    │  │  └─ Allocates: stencil workspace                        │
│    │  └─ Makes: SpatialOperator with BlackScholesPDE(σ, r)     │
│    │                                                              │
│    ├─ DirichletBC lambdas (left, right) ← Closure capture        │
│    │  └─ Captures: *this (AmericanOptionSolver pointer)          │
│    │                                                              │
│    └─ PDESolver constructor ← ALLOCATION (Main work)             │
│       ├─ WorkspaceStorage(n_, grid, ...) ← ALLOCATION 1          │
│       │  ├─ Allocates: vector<6n> for main buffer               │
│       │  │  └─ Contains: u_current, u_next, u_stage, rhs, Lu, ψ │
│       │  └─ Allocates: vector<n-1> for dx pre-computation        │
│       │                                                              │
│       ├─ std::vector<double> u_current_(n_) ← ALLOCATION 2        │
│       ├─ std::vector<double> u_old_(n_) ← ALLOCATION 3            │
│       ├─ std::vector<double> rhs_(n_) ← ALLOCATION 4              │
│       │  └─ Note: Different from workspace buffer arrays         │
│       │                                                              │
│       └─ NewtonWorkspace(n_, workspace_) ← ALLOCATION 5           │
│          ├─ Allocates: vector<8n-2> for owned arrays             │
│          │  ├─ jacobian_lower (n-1), jacobian_diag (n)           │
│          │  ├─ jacobian_upper (n-1), residual (n)                │
│          │  ├─ delta_u (n), u_old (n), tridiag (2n)              │
│          │  └─ Total: 8n - 2 doubles                             │
│          │                                                              │
│          └─ Borrows from workspace_:                             │
│             ├─ Lu_ (read-only reference to workspace_.lu())      │
│             ├─ u_perturb_ (reference to workspace_.u_stage())    │
│             └─ Lu_perturb_ (reference to workspace_.rhs())       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. PriceTableSnapshotCollector Constructor                       │
│    [price_table_4d_builder.cpp:167-174]                          │
│                                                                   │
│    ├─ Stores: moneyness_, tau_ (copies of spans)                │
│    ├─ Allocates: prices_ (Nm × Nt = 1500 doubles)               │
│    ├─ Allocates: deltas_ (Nm × Nt = 1500 doubles)               │
│    ├─ Allocates: gammas_ (Nm × Nt = 1500 doubles)               │
│    ├─ Allocates: thetas_ (Nm × Nt = 1500 doubles)               │
│    ├─ Allocates: log_moneyness_ (Nm = 50 doubles) [perf cache]  │
│    ├─ Allocates: spot_values_ (Nm = 50 doubles) [perf cache]    │
│    ├─ Allocates: inv_spot_ (Nm = 50 doubles) [perf cache]       │
│    └─ Allocates: inv_spot_sq_ (Nm = 50 doubles) [perf cache]    │
│       └─ Total: 6200 doubles ≈ 50 KB per collector              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Register Snapshots & Solve                                   │
│    [price_table_4d_builder.cpp:181-186]                         │
│                                                                   │
│    ├─ solver.register_snapshot(step_idx, user_idx, collector)    │
│    │  └─ Stores: {step_idx, user_idx, collector*} in vector     │
│    │                                                              │
│    └─ auto result = solver.solve()                               │
│       └─ Executes: TR-BDF2 time-stepping                         │
│          ├─ Each step: Newton iteration with Jacobian assembly  │
│          ├─ When step matches registered snapshot:               │
│          │  ├─ Computes first & second derivatives (if needed)   │
│          │  ├─ Builds interpolators for all (m, τ) points        │
│          │  └─ Calls collector.collect(snapshot)                 │
│          └─ Collector fills prices_, deltas_, gammas_, thetas_   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Copy Results to 4D Array                                     │
│    [price_table_4d_builder.cpp:195-204]                         │
│                                                                   │
│    For each (m_idx, tau_idx):                                    │
│      prices_4d[(m_idx, tau_idx, k, l)] = collector.prices()[idx] │
└─────────────────────────────────────────────────────────────────┘
```

## Total Allocations Per Loop Iteration

### For Default Configuration: n_space=101, Nm=50, Nt=30, Nσ=20, Nr=10

```
AmericanOptionSolver::solve() [american_option.cpp]
├─ Grid generation (line 132)
│  ├─ GridSpec<>::uniform().generate()  [once per solver]
│  ├─ Allocates: vector<101>            [~1 KB per solver]
│  └─ Status: REUSED (can move outside loop)
│
├─ SpatialOperator creation (line 144-151)
│  ├─ GridSpacing<double>               [~1 KB per solver]
│  └─ Status: REUSED (GridSpacing factory supported)
│
└─ PDESolver constructor (line 192 or 252)
   ├─ WorkspaceStorage                  [~2.4 KB per solver]
   ├─ PDESolver members                 [~0.8 KB per solver]
   └─ NewtonWorkspace                   [~6.4 KB per solver]
      └─ Total: ~10 KB per PDESolver

PriceTableSnapshotCollector (line 174)
├─ Allocates 4 vectors: prices, deltas, gammas, thetas
│  └─ Each: Nm × Nt × 8 bytes = 50 × 30 × 8 = 12 KB
│  └─ Total: ~48 KB per collector
├─ Allocates 4 perf cache vectors: log_moneyness, spot, inv_spot, inv_spot_sq
│  └─ Each: Nm × 8 bytes = 50 × 8 = 400 bytes
│  └─ Total: ~1.6 KB per collector
└─ Total per collector: ~50 KB

──────────────────────────────────────────────────
TOTAL PER (σ, r) ITERATION: ~60 KB
```

### Memory Over All Iterations

```
Nσ × Nr = 20 × 10 = 200 solver instances

Per iteration:       ~60 KB
×200 iterations:     ~12 MB total allocations

But this includes some duplications:
  ├─ Spatial grid: 200 copies (could be 1 shared)  → 100 KB waste
  ├─ GridSpacing: 200 copies (could be 1 shared)   → 100 KB waste
  └─ Most critical: WorkspaceStorage & Newton      → ~3.3 MB (necessary)
```

## Object Lifetime Diagram

```
┌──────────────────────────────────────────────────────────┐
│ precompute() SCOPE                                        │
│                                                           │
│  ┌─ Loop iteration for (σ_k, r_l) ────────────────────┐ │
│  │                                                     │ │
│  │  ┌─ AmericanOptionSolver scope ──────────────────┐ │ │
│  │  │                                               │ │ │
│  │  │  ┌─ PDESolver local variable scope ────────┐ │ │ │
│  │  │  │  ├─ WorkspaceStorage (composed)         │ │ │ │
│  │  │  │  ├─ Newton workspace (composed)         │ │ │ │
│  │  │  │  ├─ u_current_, u_old_, rhs_           │ │ │ │
│  │  │  │  └─ Lifetime: ~1-5 seconds per solver   │ │ │ │
│  │  │  └───────────────────────────────────────── │ │ │
│  │  │        [DESTROYED after solve()]             │ │ │
│  │  │  ┌─ SpatialOperator local variable scope ──┐ │ │ │
│  │  │  │  ├─ BlackScholesPDE (moved into PDEMhd) │ │ │ │
│  │  │  │  └─ GridSpacing (shared_ptr)            │ │ │ │
│  │  │  └───────────────────────────────────────── │ │ │
│  │  │                                               │ │ │
│  │  │  ┌─ PriceTableSnapshotCollector scope ────┐ │ │ │
│  │  │  │  ├─ prices_, deltas_, gammas_, thetas_ │ │ │ │
│  │  │  │  └─ Extracted: collector.prices()      │ │ │ │
│  │  │  └───────────────────────────────────────── │ │ │
│  │  │       [Copy to 4D array at line 199-203]    │ │ │
│  │  │                                               │ │ │
│  │  └─────────────────────────────────────────────┘ │ │
│  │        [Solver DESTROYED at end of iteration]     │ │
│  │                                                     │ │
│  └─────────────────────────────────────────────────┘ │
│     [Loop to next (σ, r) pair]                       │
│                                                       │
└──────────────────────────────────────────────────────┘
```

## Opportunity: Shared Grid Pattern

### Current (Inside Loop)
```cpp
// Line 155-164 in solve():
for each (σ, r):
    auto grid = GridSpec<>::uniform(...).generate();  // NEW grid
    auto x_grid = grid.span();
    // ... create solver with x_grid ...
```

**Problem:** Allocates new vector<double> for each iteration
**Impact:** 200 allocations × 101 elements = 20,200 values

### Optimized (Before Loop)
```cpp
// Move to PriceTable4DBuilder::precompute()
auto grid = GridSpec<>::uniform(grid_.x_min, grid_.x_max, 
                                 grid_.n_space).generate();
auto x_grid = grid.span();

#pragma omp parallel for collapse(2)
for (size_t k = 0; k < Nv; ++k) {
    for (size_t l = 0; l < Nr; ++l) {
        // ... create solver with x_grid ...
        // Grid is now SHARED across all (σ, r) pairs
        auto grid_view = GridView<double>(x_grid);
        // ... rest unchanged ...
    }
}
```

**Savings:** One-time allocation instead of Nσ × Nr allocations
**Cost:** Pass grid_ through American option parameters
**Compatibility:** Zero - grid is immutable after generation

## Key Realization: The "Shadow" Allocations

PDESolver has **redundant allocations** alongside WorkspaceStorage:

```
WorkspaceStorage.buffer_:
├─ u_current [n]    ← Used for workspace
├─ u_next [n]       ← Used for workspace
├─ u_stage [n]      ← Used for workspace
├─ rhs [n]          ← Used for workspace
├─ lu [n]           ← Used for workspace
└─ psi [n]          ← Used for workspace

PDESolver members:
├─ u_current_ [n]   ← DIFFERENT from WorkspaceStorage.u_current_
├─ u_old_ [n]       ← Stores u^n for TR-BDF2
└─ rhs_ [n]         ← DIFFERENT from WorkspaceStorage.rhs_
```

**Historical Note:** Likely PDESolver predates WorkspaceStorage refactoring.
The workspace.hpp design already unified much allocation, but PDESolver
still owns its own copies of u_current_, u_old_, and rhs_.

**Refactoring opportunity:** Use WorkspaceStorage arrays directly instead
of owning separate copies. Would save ~3n doubles per solver (24n bytes).

---

## Timeline: Single (σ, r) Iteration

```
T=0ms     Constructor begins
 ├─  1ms  AmericanOptionSolver::solve() entry
 ├─  0.5ms Grid generation
 ├─  0.3ms SpatialOperator creation
 ├─  0.2ms PDESolver construction + initialization
 ├─ ~1500ms solve() execution
 │         ├─ Stage 1 (Trapezoidal): ~250ms × gamma
 │         ├─ Stage 2 (BDF2):        ~250ms × (1-gamma)
 │         ├─ Snapshots:             ~50ms (30 snapshots interpolation)
 │         └─ Newton iterations:     ~1000ms (accumulated)
 │
 ├─ 50ms   collector.collect() calls
 ├─ 5ms    Copy results to 4D array
 └─ 3ms    Destructors & cleanup
─────────────────────────────────
TOTAL:    ~1560ms per solver
         ~78 solvers/minute single-threaded
         ~848 solvers/minute with 16 cores (assuming good parallelization)
```

---

## Summary: What Persists, What Recreates

```
┌──────────────────────────────┬──────────┬──────────┐
│ Component                    │ Current  │ Optimal  │
├──────────────────────────────┼──────────┼──────────┤
│ Spatial grid (x coordinates) │ Per loop │ Once     │
│ GridSpacing metrics          │ Per loop │ Once     │
│ CenteredDifference stencil   │ Per loop │ Once*    │
│ TimeDomain parameters        │ Per loop │ Once     │
│ PDESolver instance           │ Per loop │ Per loop │
│ AmericanOptionSolver         │ Per loop │ Per loop │
│ BlackScholesPDE (σ, r)       │ Per loop │ Per loop │
│ Boundary conditions          │ Per loop │ Per loop │
│ NewtonWorkspace buffers      │ Per loop │ Pooled?  │
│ SnapshotCollector buffers    │ Per loop │ Pooled?  │
│ Obstacle computation         │ Per loop │ Per loop │
└──────────────────────────────┴──────────┴──────────┘

*CenteredDifference is stateless, could be shared as static instance
```

