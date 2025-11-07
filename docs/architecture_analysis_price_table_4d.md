# PriceTable4DBuilder Architecture Analysis

## Executive Summary

The current architecture creates **separate AmericanOptionSolver instances for each (σ, r) pair** in the nested loop (lines 151-211 of `price_table_4d_builder.cpp`). Each instance has its own complete PDESolver with dedicated memory allocations. This analysis identifies what can be reused across multiple solvers when only volatility and rate change.

---

## Current Architecture Flow

### Entry Point: PriceTable4DBuilder::precompute()
```
PriceTable4DBuilder::precompute()
  ├─ Create 4D price array (Nm × Nt × Nσ × Nr)
  └─ For each (σ, r) pair [k, l] in parallel:
       ├─ Create AmericanOptionParams with volatility_[k], rate_[l]
       ├─ Create PriceTableSnapshotCollector
       ├─ Create AmericanOptionSolver
       ├─ Register N_t snapshots
       └─ Call solver.solve()
```

**Key insight:** Each iteration creates an entirely NEW solver instance with fresh allocations.

---

## Memory Allocations Per Solver Creation

### 1. AmericanOptionSolver Constructor (american_option.cpp:115-128)
Simply stores parameters and validates them. **No allocations here.**

### 2. PDESolver Constructor (pde_solver.hpp:79-107)

The real allocations happen in PDESolver's initialization:

```cpp
PDESolver(...) 
  : grid_(grid)                              // Non-owning span
  , workspace_(n_, grid, config_.cache_blocking_threshold)  // ALLOCATION 1
  , u_current_(n_)                           // ALLOCATION 2
  , u_old_(n_)                               // ALLOCATION 3
  , rhs_(n_)                                 // ALLOCATION 4
  , newton_ws_(n_, workspace_)               // Uses workspace + ALLOCATION 5
```

#### Allocation 1: WorkspaceStorage (workspace.hpp:36-55)
```cpp
buffer_(6 * n)              // 6n doubles for u_current, u_next, u_stage, rhs, Lu, psi
dx_(n - 1)                  // Pre-computed grid spacing: (n-1) doubles
```
**Memory:** 6n + (n-1) ≈ 7n doubles ≈ 56n bytes (for n=101: ~5.6 KB)

#### Allocation 2-4: PDESolver Member Vectors
```cpp
u_current_(n_)              // n doubles = 8n bytes
u_old_(n_)                  // n doubles = 8n bytes
rhs_(n_)                    // n doubles = 8n bytes
```
**Memory:** 3n doubles = 24n bytes (for n=101: ~2.4 KB)

#### Allocation 5: NewtonWorkspace (newton_workspace.hpp:29-37)
```cpp
buffer_(compute_buffer_size(n))  // 8n - 2 doubles
  ├─ jacobian_lower: n-1
  ├─ jacobian_diag: n
  ├─ jacobian_upper: n-1
  ├─ residual: n
  ├─ delta_u: n
  ├─ u_old: n
  └─ tridiag_workspace: 2n
  
+ Borrowed from WorkspaceStorage:
  ├─ Lu_ (read-only during Newton)
  ├─ u_perturb_ (from u_stage)
  └─ Lu_perturb_ (from rhs)
```
**Memory:** (8n - 2) doubles ≈ 64n bytes (for n=101: ~6.4 KB)

#### GridSpacing (operator_factory.hpp → spatial_operator.hpp)
```cpp
if (is_uniform_) {
    // Pre-computed constants: O(1)
} else {
    dx_array_.reserve(n - 1)  // Could allocate for non-uniform
}
```
**Memory:** Negligible for uniform grids (our case), ~8n bytes for non-uniform

#### SpatialOperator (operators/spatial_operator.hpp:48-51)
```cpp
pde_(...)           // Copy of BlackScholesPDE (small, ~100 bytes)
spacing_(...)       // shared_ptr to GridSpacing (~16 bytes)
stencil_(*spacing_) // CenteredDifference (small)
```
**Memory:** ~200 bytes (stack-allocated, negligible)

### Total Memory Per Solver Instance

For n=101 (typical grid size):
```
WorkspaceStorage:        7n ≈ 700 bytes
PDESolver members:       3n ≈ 300 bytes
NewtonWorkspace:         8n ≈ 800 bytes
GridSpacing:             ~0 (uniform) or ~800 bytes (non-uniform)
SpatialOperator:         ~200 bytes
─────────────────────────────────
Total per solver:      ~1.8 KB to 2.6 KB

For Nσ × Nr = 20 × 10 = 200 solvers:
Total allocation:      ~360 KB to 520 KB
```

**BUT** this analysis misses the construction overhead: instantiation, validation, lambda captures, etc.

---

## Parameter-Dependent vs. Reusable Components

### What Changes with (σ, r)?

**Directly Parameter-Dependent:**
1. **BlackScholesPDE** - Contains σ and r as member variables
2. **Boundary conditions** - Time-dependent, use σ and r
3. **AmericanOptionSolver parameters** - Stored in `params_` (σ, r, etc.)
4. **Obstacle computation** - Re-evaluated at each snapshot (strike-dependent but constant)
5. **RHS vector** - Recomputed in each TR-BDF2 stage using σ and r
6. **Jacobian** - Recomputed using PDE coefficients (σ, r)
7. **NewtonWorkspace** - Workspace itself is generic, but used differently

**Grid-Only Dependent (Can Be Reused):**
1. **SpatialGrid** - Same grid for all (σ, r) pairs
2. **GridSpacing** - Derived from spatial grid only
3. **CenteredDifference stencil** - Depends only on grid spacing
4. **WorkspaceStorage buffer layout** - Fixed size n
5. **Cache-blocking configuration** - Same for all solvers
6. **TimeDomain** - Same (T_max, n_time, dt)
7. **PriceTableSnapshotCollector structure** - Same moneyness/tau grid

**Reconfigurable (Could be pooled):**
1. **PDESolver instance** - Needs new instance, but could pre-allocate buffers
2. **NewtonWorkspace** - Buffers could be pre-allocated and reused
3. **WorkspaceStorage** - Buffers could be pre-allocated and reset

---

## Current Inefficiencies

### 1. Grid Construction Duplication (american_option.cpp:132)
```cpp
auto grid_buffer = GridSpec<>::uniform(grid_.x_min, grid_.x_max, grid_.n_space).generate();
auto x_grid = grid_buffer.span();
```
**Happens:** Once per solver
**Could happen:** Once globally, reused for all solvers
**Impact:** O(n) allocations per solver (grid generation is cheap but repeated)

### 2. SpatialOperator Recreation (american_option.cpp:144-151)
```cpp
auto grid_view = GridView<double>(x_grid);
auto bs_op = operators::create_spatial_operator(
    operators::BlackScholesPDE<double>(...),
    grid_view
);
```
**Happens:** Once per solver
**Could happen:** Template could be pre-created, only σ/r injected
**Impact:** Creates GridSpacing, CenteredDifference stencil, and BlackScholesPDE each time

### 3. Boundary Condition Capture (american_option.cpp:158-185)
```cpp
auto left_bc = DirichletBC([this](double t, double x) { ... });
auto right_bc = DirichletBC([this](double t, double x) { ... });
```
**Happens:** Once per solver
**Could happen:** Pre-create with σ/r as function parameters
**Impact:** Lambda capture overhead (small but repeated)

### 4. PDESolver Allocation (american_option.cpp:192 or 252)
```cpp
PDESolver solver(x_grid, time_domain, trbdf2_config_, root_config_,
                left_bc, right_bc, bs_op,
                [](double t, auto x, auto psi) { ... });
```
**Happens:** Once per solver
**Must happen:** New instance required (state is mutable during solve)
**But:** Could use arena allocator or pre-allocated buffer pool

### 5. Snapshot Collector Recreation (price_table_4d_builder.cpp:167-174)
```cpp
PriceTableSnapshotCollector collector(collector_config);
for (size_t j = 0; j < Nt; ++j) {
    collector.prices_.resize(Nm * Nt);  // Allocates here
    ...
}
```
**Happens:** Once per (σ, r) pair
**Could happen:** Pre-allocate once, reset per iteration
**Impact:** Per-solver allocation (~800 bytes for Nm × Nt)

---

## Reusability Opportunities (Ranked by Impact)

### HIGH IMPACT: Pre-allocate Spatial Grid

**Current:**
```cpp
for each (σ, r):
    auto grid = GridSpec<>::uniform(...).generate()  // NEW allocation
```

**Optimized:**
```cpp
// Outside loop
auto grid = GridSpec<>::uniform(grid_.x_min, grid_.x_max, grid_.n_space).generate();
auto x_grid = grid.span();

// Inside loop
auto grid_view = GridView<double>(x_grid);  // Non-owning, cheap
```

**Savings:** O(Nσ × Nr × n) = 20×10×101 = 20,200 unnecessary allocations

---

### HIGH IMPACT: Reuse GridSpacing

**Current:**
```cpp
for each (σ, r):
    auto bs_op = operators::create_spatial_operator(pde, GridView(x_grid));
                                                      // NEW GridSpacing created
```

**Optimized:**
```cpp
// Outside loop
auto spacing = std::make_shared<GridSpacing<double>>(GridView(x_grid));

// Inside loop (use operator factory overload)
auto bs_op = operators::create_spatial_operator(pde, spacing);  // Reuse
```

**Savings:** O(Nσ × Nr × (GridSpacing allocation + stencil setup))
**Code:** Already has operator factory overload on line 21-28 of operator_factory.hpp!

---

### MEDIUM IMPACT: Pre-allocate WorkspaceStorage

**Current:**
```cpp
for each (σ, r):
    PDESolver(...) → WorkspaceStorage(n_, grid, ...)  // NEW allocation
```

**Optimized:**
```cpp
// Outside loop (per-thread in OpenMP)
WorkspaceStorage workspace(n, x_grid);

// Inside loop
PDESolver(..., workspace) → Reuse existing buffers
```

**Limitation:** PDESolver doesn't currently accept workspace (owns via composition)
**Refactoring needed:** Extract workspace initialization

**Savings:** O(Nσ × Nr × 7n doubles) = 20×10×7×101×8 bytes = ~113 KB for single-threaded
**Scaling:** Per-thread workspace needed for OpenMP (still beneficial)

---

### MEDIUM IMPACT: Pre-allocate SnapshotCollector

**Current:**
```cpp
for each (σ, r):
    PriceTableSnapshotCollector collector(config);
    collector.prices_.resize(Nm × Nt);  // Allocation
    collector.deltas_.resize(Nm × Nt);
    collector.gammas_.resize(Nm × Nt);
    collector.thetas_.resize(Nm × Nt);
```

**Optimized:**
```cpp
// Outside loop
std::vector<double> prices_buffer(Nm * Nt);
std::vector<double> deltas_buffer(Nm * Nt);
std::vector<double> gammas_buffer(Nm * Nt);
std::vector<double> thetas_buffer(Nm * Nt);

// Inside loop
collector.reset(prices_buffer, deltas_buffer, ...);  // Reuse buffers
```

**Savings:** O(Nσ × Nr × 4 × Nm × Nt × 8 bytes) = 20×10×4×50×30×8 = ~1.2 MB

---

### LOW IMPACT: PDESolver Instance Pooling

**Current:** Each (σ, r) pair creates new PDESolver

**Challenge:** PDESolver state is mutable during solve():
- `u_current_`, `u_old_`, `rhs_` modified
- `newton_ws_` state changes
- Snapshot requests stored

**Option 1: Reset Pattern**
```cpp
class PDESolver {
    void reset(const SpatialOp& new_op) {
        spatial_op_ = new_op;
        std::fill(u_current_.begin(), u_current_.end(), 0.0);
        std::fill(u_old_.begin(), u_old_.end(), 0.0);
        std::fill(rhs_.begin(), rhs_.end(), 0.0);
        snapshot_requests_.clear();
    }
};
```

**Savings:** Avoids PDESolver constructor/destructor overhead per (σ, r) pair
**Cost:** Adds reset() call overhead
**Net benefit:** Likely negative (constructor is already cheap)

**Option 2: WorkspacePool Pattern**
```cpp
class WorkspacePool {
    std::vector<std::unique_ptr<WorkspaceStorage>> spaces_;
    
    WorkspaceStorage& acquire() {
        if (available_.empty()) {
            return *spaces_.emplace_back(...);
        }
        auto ws = available_.back();
        available_.pop_back();
        return *ws;
    }
};
```

**Feasibility:** High (WorkspaceStorage is self-contained)
**Savings:** Reduced allocation overhead for large Nσ × Nr
**OpenMP compatibility:** Require per-thread pools

---

## Allocation Summary Table

| Component | Size Per Solver | Count | Total (200 solvers) |
|-----------|-----------------|-------|-------------------|
| WorkspaceStorage.buffer | 6n | 200 | ~1.2 MB |
| WorkspaceStorage.dx | n-1 | 200 | ~161 KB |
| PDESolver.u_current | n | 200 | ~161 KB |
| PDESolver.u_old | n | 200 | ~161 KB |
| PDESolver.rhs | n | 200 | ~161 KB |
| NewtonWorkspace.buffer | 8n-2 | 200 | ~1.3 MB |
| **Total allocations** | - | - | **~3.3 MB** |
| **Unnecessary (if reused)** | - | - | **~2.0 MB** (grid, spacing) |

---

## Recommended Optimization Priority

### Phase 1: Immediate Wins (Low Risk)
1. **Pre-allocate spatial grid** (Lines 132 before loop)
   - Savings: ~100 KB per run
   - Risk: Minimal
   - Effort: 5 lines
   
2. **Reuse GridSpacing** (Already supported by factory)
   - Savings: ~50 KB per run + GC pressure
   - Risk: Minimal (factory overload exists)
   - Effort: 3 lines

### Phase 2: Medium Effort
1. **SnapshotCollector buffer pooling**
   - Savings: ~1.2 MB per run (20% of allocations)
   - Risk: Low (collector is stateless between resets)
   - Effort: Add reset() method, refactor loop

2. **WorkspaceStorage pre-allocation** (requires PDESolver refactoring)
   - Savings: ~1.4 MB per run (42% of allocations)
   - Risk: Medium (PDESolver composition needs change)
   - Effort: Extract workspace dependency injection

### Phase 3: Diminishing Returns
1. **PDESolver pooling** - Net benefit unclear
2. **SpatialOperator caching** - Minimal benefit
3. **Boundary condition pre-capture** - Negligible benefit

---

## Key Architectural Insights

### What PDESolver Owns
- Solution state (`u_current_`, `u_old_`)
- Implicit iteration workspace (`newton_ws_`)
- Snapshots and temporal events (request vectors)

### What Could Be Shared
- Grid coordinates (truly immutable)
- Grid spacing (immutable once computed)
- Stencil structure (depends only on grid)
- Time domain configuration (same for all)
- Cache blocking config (same for all)

### Why Reuse Is Hard
1. **PDESolver is mutable** - Solve modifies all internal state
2. **Composition pattern** - Doesn't support dependency injection of workspace
3. **Snapshot state** - Requests are stored per solver
4. **OpenMP thread safety** - Shared objects need synchronization

---

## Implementation Notes for Optimization

### Quick Win: Grid Reuse
```cpp
// In PriceTable4DBuilder::precompute()
auto grid_buffer = GridSpec<>::uniform(x_min, x_max, n_space).generate();
auto x_grid = grid_buffer.span();

#pragma omp parallel for collapse(2)
for (size_t k = 0; k < Nv; ++k) {
    for (size_t l = 0; l < Nr; ++l) {
        auto grid_view = GridView<double>(x_grid);  // Reuse x_grid
        auto spacing = std::make_shared<GridSpacing<double>>(grid_view);
        auto bs_op = operators::create_spatial_operator(
            operators::BlackScholesPDE<double>(...),
            spacing  // Reuse spacing via factory overload
        );
        // ... rest of loop
    }
}
```

### Full Refactoring: Workspace Pooling
Would require:
1. Modify PDESolver to accept WorkspaceStorage dependency
2. Create WorkspacePool managing pre-allocated buffers
3. Distribute pools per OpenMP thread
4. Reset buffers between iterations

**Estimated effort:** 2-3 hours
**Expected speedup:** ~5-10% (allocation overhead reduction)
**Risk:** Medium (changes PDESolver API)

