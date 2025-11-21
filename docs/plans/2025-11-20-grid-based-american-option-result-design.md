# Grid-Based American Option Result Design

**Date:** 2025-11-20
**Status:** Design Complete, Ready for Implementation

## Motivation

The current `AmericanOptionResult` duplicates storage and logic that belongs in the generic PDE framework. This refactoring addresses four goals:

1. **Consistency** - American option solver should return the same Grid type PDESolver uses
2. **Simplification** - Eliminate duplicate storage of grid data and solution arrays
3. **API Evolution** - Better fit with overall PDE framework architecture
4. **Performance** - Reduce memory copies by directly exposing internal Grid

Additionally, the current `AmericanSolverWorkspace` mixes concerns (Grid ownership + PMR buffers), violating the principle that Grid should not be shared across solves while workspace buffers should be reusable.

## Architecture: Three-Layer Separation

### Terminology: "Reusable" vs "Shareable"

**Reusable across solves:** Object can be used for multiple `solve()` calls without recreation
**Not reusable:** Each `solve()` requires a fresh instance (previous data would be overwritten)

Note: `std::shared_ptr` is used for lifetime management and result ownership, NOT for reuse across solves.

### Layer 1: Grid (NOT Reusable Across Solves)
**Purpose:** Grid specification + time domain + solution storage + optional snapshots

**Ownership:** Created fresh per `solve()`, returned to user via `std::shared_ptr` for result lifetime management

**Why not reusable:** Each solve overwrites solution data, so Grid cannot be reused for subsequent solves

**Contents:**
- `GridSpec<T>` (spatial grid specification)
- `TimeDomain` (temporal grid specification)
- `GridBuffer<T>` (spatial grid points)
- `GridSpacing<T>` (spacing information for operators)
- `std::vector<T>` (solution storage: current + previous)
- `std::optional<std::vector<T>>` (snapshot storage: optional)

### Layer 2: PDEWorkspace (Reusable Across Solves)
**Purpose:** Named spans to caller-managed PMR buffers (temporary scratch space)

**Ownership:** Caller allocates PMR buffer, creates PDEWorkspace spans, manages lifetime

**Why reusable:** Just temporary scratch buffers - content doesn't matter between solves, safe to reuse

**Lifetime requirement:** PMR buffer must outlive all `solve()` calls using the workspace

**Contents:** Just spans (no ownership):
- `dx`, `u_stage`, `rhs`, `lu`, `psi`
- Jacobian arrays: `jacobian_diag`, `jacobian_upper`, `jacobian_lower`
- Newton arrays: `residual`, `delta_u`, `newton_u_old`
- `tridiag_workspace`

### Layer 3: AmericanOptionSolver
**Purpose:** Domain-specific solver that creates Grid and uses PDEWorkspace

**Ownership:** Creates Grid during solve(), receives PDEWorkspace from caller

**Flow:**
1. Receives: `AmericanOptionParams`, `PDEWorkspace`, optional snapshot times
2. Creates: Fresh Grid with GridSpec + TimeDomain computed from params
3. Solves: PDE using Grid + PDEWorkspace
4. Returns: `AmericanOptionResult` wrapping Grid + params

## Design Details

### 1. Grid Snapshot System

**API:**

```cpp
class Grid {
public:
    // Factory with snapshot times (converted to indices internally)
    static std::expected<std::shared_ptr<Grid<T>>, std::string>
    create(const GridSpec<T>& grid_spec,
           const TimeDomain& time_domain,
           std::span<const double> snapshot_times = {});

    // Query snapshots
    bool has_snapshots() const;
    std::span<const T> at(size_t snapshot_idx) const;  // Returns spatial solution at snapshot
    size_t num_snapshots() const;  // Number of recorded snapshots
    std::span<const double> snapshot_times() const;  // Times corresponding to each snapshot (after snapping)

    // For PDESolver (internal use only)
    bool should_record(size_t time_step_idx) const;  // Check if time step should be recorded
    void record(size_t time_step_idx, std::span<const double> sol);  // Record spatial solution

private:
    std::vector<size_t> snapshot_indices_;  // Sorted time step indices to record
    std::vector<double> snapshot_times_;  // Actual times after snapping (for query)
    std::optional<std::vector<T>> surface_history_;  // 2D: num_snapshots × n_space (row-major)

    // Helper: Map time step index → snapshot index (or nullopt if not recorded)
    std::optional<size_t> find_snapshot_index(size_t time_step_idx) const;
};
```

**Storage layout:**

Snapshots are stored in a flat 1D vector with row-major ordering:
```
surface_history_[snapshot_idx * n_space + space_idx] = solution[space_idx]
```

Example: 3 snapshots, 100 spatial points → 300 element vector
- Snapshot 0: elements [0, 99]
- Snapshot 1: elements [100, 199]
- Snapshot 2: elements [200, 299]

**Time-to-index conversion:**

```cpp
std::expected<std::vector<size_t>, std::string> convert_times_to_indices(
    std::span<const double> times,
    const TimeDomain& time_domain)
{
    const double dt = time_domain.dt();
    const double t_max = time_domain.t_end();
    const size_t n_steps = time_domain.n_steps();

    std::vector<size_t> indices;
    indices.reserve(times.size());

    for (double t : times) {
        // Validate time is in valid range
        if (t < 0.0 || t > t_max) {
            return std::unexpected(std::format(
                "Snapshot time {} out of range [0, {}]", t, t_max));
        }

        // Convert to nearest time step (snap to grid)
        // Use floor + 0.5 to round to nearest, not llround (which can overshoot)
        double step_exact = t / dt;
        size_t step_idx = static_cast<size_t>(std::floor(step_exact + 0.5));

        // Clamp to valid range (handles floating point rounding at boundaries)
        step_idx = std::min(step_idx, n_steps - 1);

        indices.push_back(step_idx);
    }

    // Sort and deduplicate
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

    return indices;
}
```

**Key differences from clamping approach:**
1. **Validates** out-of-range times instead of silently clamping (catches user errors)
2. **Snaps to nearest** time step using floor(t/dt + 0.5) instead of llround (more predictable)
3. **Returns expected** to propagate validation errors to caller

**PDESolver integration:**

```cpp
// In PDESolver::solve() time-stepping loop:

// Record initial condition at t=0 if requested
if (grid_->should_record(0)) {
    grid_->record(0, u_current);
}

for (size_t step = 0; step < time.n_steps(); ++step) {
    // ... TR-BDF2 stages ...

    // Process temporal events (discrete dividends, etc.)
    process_temporal_events(step + 1, u_current);

    // Record snapshot AFTER events (captures true PDE state at t_{n+1})
    if (grid_->should_record(step + 1)) {
        grid_->record(step + 1, u_current);
    }
}
```

**Snapshot Timing Details:**

The snapshot system uses **state indices** (not time step indices) in the range `[0, n_steps]`:
- State 0: Initial condition at `t = 0` (maturity in backward PDE)
- State n: Final state at `t = n * dt`

When snapshots are recorded **after** TR-BDF2 stages and temporal events:
- Snapshot at state index `k` captures the PDE state **after** the k-th time step completes
- Initial condition (state 0) must be recorded explicitly **before** the time-stepping loop
- All other states are recorded **after** events to capture the true PDE state

The time-to-index conversion maps user-specified times to state indices:
```cpp
// User requests snapshot at t=0.5 (maturity in backward PDE)
// Convert to state index: floor(0.5 / dt + 0.5)
// If dt=0.1, state_idx = 5 → captures state after 5 time steps
```

### 2. AmericanOptionSolver Changes

**Constructor:**

```cpp
class AmericanOptionSolver {
public:
    AmericanOptionSolver(
        const AmericanOptionParams& params,
        PDEWorkspace workspace,  // Just spans, caller manages buffer
        std::optional<std::span<const double>> snapshot_times = std::nullopt);

    std::expected<AmericanOptionResult, SolverError> solve();

private:
    AmericanOptionParams params_;
    PDEWorkspace workspace_;
    std::vector<double> snapshot_times_;  // Copied from span
};
```

**solve() implementation:**

```cpp
std::expected<AmericanOptionResult, SolverError>
AmericanOptionSolver::solve() {
    // Compute grid configuration from params
    auto [grid_spec, n_time] = estimate_grid_for_option(params_);
    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, params_.maturity, n_time);

    // Validate workspace size matches grid
    if (workspace_.size() != grid_spec.n_points()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = std::format(
                "Workspace size mismatch: workspace has {} points, grid requires {}",
                workspace_.size(), grid_spec.n_points())
        });
    }

    // Create Grid with optional snapshots (NOT REUSABLE)
    auto grid_result = Grid<double>::create(grid_spec, time_domain, snapshot_times_);

    if (!grid_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::GridCreationFailed,
            .message = grid_result.error()
        });
    }

    auto grid = grid_result.value();

    // Create PDESolver with Grid + Workspace
    // PDESolver is created fresh each solve, takes Grid by shared_ptr
    AmericanPutSolver pde_solver(grid, workspace_);

    // Initialize PDE with payoff condition
    pde_solver.initialize([](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(1.0 - std::exp(x[i]), 0.0);  // Put payoff in log-moneyness
        }
    });

    // Solve PDE (modifies Grid in-place, records snapshots if configured)
    auto solve_result = pde_solver.solve();

    if (!solve_result.has_value()) {
        return std::unexpected(solve_result.error());
    }

    // Wrap Grid + params → AmericanOptionResult (explicit, no metaprogramming)
    return AmericanOptionResult(grid, params_);
}
```

**PDESolver Construction Pattern:**

PDESolver is constructed fresh for each solve, taking:
1. `std::shared_ptr<Grid>` - Grid to solve on (created above)
2. `PDEWorkspace` - Temporary workspace buffers (reused from caller)

PDESolver stores a reference/pointer to Grid and modifies it in-place during `solve()`.

### 3. AmericanOptionResult Wrapper

**API:**

```cpp
class AmericanOptionResult {
public:
    // Constructor (called by AmericanOptionSolver)
    AmericanOptionResult(std::shared_ptr<Grid<double>> grid,
                         const AmericanOptionParams& params);

    // Convenience: value at current spot
    double value() const { return value_at(params_.spot); }

    // Interpolation to arbitrary spot price
    double value_at(double spot) const;

    // Greeks (lazy-computed, cached)
    double delta() const;
    double gamma() const;
    double theta() const;

    // Snapshot access (delegates to grid)
    bool has_snapshots() const { return grid_->has_snapshots(); }
    std::span<const double> at_time(size_t snapshot_idx) const {
        return grid_->at(snapshot_idx);
    }
    size_t num_snapshots() const { return grid_->num_snapshots(); }
    std::span<const double> snapshot_times() const { return grid_->snapshot_times(); }

    // Direct grid access (for advanced users)
    const Grid<double>& grid() const { return *grid_; }
    std::shared_ptr<Grid<double>> grid_ptr() const { return grid_; }

private:
    std::shared_ptr<Grid<double>> grid_;
    AmericanOptionParams params_;

    // Lazy initialization for Greeks
    mutable std::unique_ptr<GridSpacing<double>> grid_spacing_;
    mutable std::unique_ptr<operators::CenteredDifference<double>> diff_op_;

    // Helper: interpolate solution at log-moneyness
    double interpolate_solution(double log_moneyness) const;
};
```

**Key methods:**
- `value()`: Shortcut for `value_at(params_.spot)`
- `value_at(spot)`: Interpolates to arbitrary spot, denormalizes using `params_.strike`
- Greeks: Use existing `CenteredDifference` infrastructure (lazy-computed, mutable cache)
- `at_time(snapshot_idx)`: Returns spatial solution at snapshot index (NOT time step index)

**Thread Safety:**
- **AmericanOptionResult is NOT thread-safe for concurrent reads** due to mutable Greeks cache
- Safe usage: Compute Greeks once in single thread, then read-only access is safe
- Concurrent solves: Each thread gets its own AmericanOptionResult (Grid is per-solve)
- Alternative: Eager Greeks computation in constructor (no cache, fully thread-safe reads)

### 4. AmericanSolverWorkspace Removal

**Before (problematic):**
```cpp
class AmericanSolverWorkspace {
    std::shared_ptr<Grid<double>> grid_;  // WRONG: Grid shouldn't be shared
    std::pmr::vector<double> pmr_buffer_;
    PDEWorkspace workspace_spans_;
};
```

**After (clean separation):**
```cpp
// Caller manages PMR buffer directly
std::pmr::synchronized_pool_resource pool;
std::pmr::vector<double> buffer(PDEWorkspace::required_size(n_space), 0.0, &pool);

// Create workspace spans (reusable)
auto workspace = PDEWorkspace::from_buffer(buffer, n_space).value();

// Solver creates Grid internally (NOT reusable)
AmericanOptionSolver solver(params, workspace, snapshot_times);
auto result = solver.solve();  // Creates fresh Grid
```

### 5. PDESolver Return Type

**Change PDESolver to return Grid via std::expected:**

```cpp
class PDESolver {
public:
    // OLD: bool solve()
    // NEW: returns Grid or error
    std::expected<void, SolverError> solve();
};
```

Note: PDESolver modifies Grid in-place (Grid passed to constructor), so it returns `void` on success. The Grid is already owned by the caller.

## Usage Examples

### Example 1: Basic Usage (No Snapshots)

```cpp
#include "src/option/american_option.hpp"
#include "src/pde/core/pde_workspace.hpp"

// Define params
AmericanOptionParams params{
    .spot = 100.0,
    .strike = 100.0,
    .maturity = 1.0,
    .volatility = 0.20,
    .rate = 0.05,
    .dividend_yield = 0.02,
    .type = OptionType::PUT
};

// Estimate grid size
auto [grid_spec, n_time] = estimate_grid_for_option(params);
size_t n_space = grid_spec.n_points();

// Allocate workspace buffer (reusable)
std::pmr::synchronized_pool_resource pool;
std::pmr::vector<double> buffer(PDEWorkspace::required_size(n_space), 0.0, &pool);
auto workspace = PDEWorkspace::from_buffer(buffer, n_space).value();

// Solve (no snapshots)
AmericanOptionSolver solver(params, workspace);
auto result = solver.solve();

if (result.has_value()) {
    std::cout << "Price: " << result->value() << "\n";
    std::cout << "Delta: " << result->delta() << "\n";
    std::cout << "Gamma: " << result->gamma() << "\n";
}
```

### Example 2: With Snapshots (Price Table)

```cpp
// Same setup as Example 1...

// Solve with specific snapshot times
std::vector<double> snapshot_times = {0.25, 0.5, 1.0};
AmericanOptionSolver solver(params, workspace, snapshot_times);
auto result = solver.solve();

if (result.has_value()) {
    // Access snapshots
    if (result->has_snapshots()) {
        for (size_t i = 0; i < result->num_snapshots(); ++i) {
            auto snapshot = result->at_time(i);
            // Process snapshot...
        }
    }
}
```

### Example 3: Workspace Reuse

```cpp
// Allocate workspace once for maximum grid size
std::pmr::synchronized_pool_resource pool;

// IMPORTANT: Workspace must be sized for largest grid in batch
// Compute max n_space across all options
size_t max_n_space = 0;
for (const auto& params : option_batch) {
    auto [grid_spec, n_time] = estimate_grid_for_option(params);
    max_n_space = std::max(max_n_space, grid_spec.n_points());
}

// Allocate workspace for maximum size
std::pmr::vector<double> buffer(PDEWorkspace::required_size(max_n_space), 0.0, &pool);

// Solve multiple options (reuse workspace)
for (const auto& params : option_batch) {
    auto [grid_spec, n_time] = estimate_grid_for_option(params);
    size_t n_space = grid_spec.n_points();

    // Create workspace spans for this grid size
    auto workspace = PDEWorkspace::from_buffer(buffer, n_space).value();

    AmericanOptionSolver solver(params, workspace);
    auto result = solver.solve();
    // Each solve creates fresh Grid, reuses workspace buffer
}
```

**Important:** Workspace size must match grid size. The solver validates this at the start of `solve()` and returns an error on mismatch. Either:
1. Allocate for maximum grid size and create appropriately-sized spans per solve (shown above)
2. Recreate workspace for each option if grid sizes vary significantly

**Workspace Size Validation:**
```cpp
// AmericanOptionSolver::solve() checks workspace size
if (workspace_.size() != grid_spec.n_points()) {
    return std::unexpected(SolverError{
        .code = SolverErrorCode::InvalidConfiguration,
        .message = "Workspace size mismatch: ..."
    });
}
```

This prevents silent out-of-bounds writes if a caller reuses a workspace sized for a different grid.

## Migration Path

### Step 1: Add Snapshot Support to Grid
- Add `snapshot_indices_` and `surface_history_` members
- Add `should_record()`, `record()`, `at()` methods
- Add `snapshot_times` parameter to `Grid::create()`
- Implement time-to-index conversion

### Step 2: Update PDESolver
- Change return type to `std::expected<void, SolverError>`
- Add snapshot recording in time-stepping loop

### Step 3: Refactor AmericanOptionResult
- Change to wrapper class with `Grid` + `params`
- Implement `value_at()`, Greeks, `at_time()` delegation
- Remove duplicate storage

### Step 4: Update AmericanOptionSolver
- Remove `AmericanSolverWorkspace` dependency
- Accept `PDEWorkspace` directly in constructor
- Create Grid during `solve()` from params
- Return wrapped `AmericanOptionResult`

### Step 5: Remove AmericanSolverWorkspace
- Delete `AmericanSolverWorkspace` class
- Update all call sites to use `PDEWorkspace` directly
- Update documentation and examples

### Step 6: Update Tests
- Update `american_option_test.cc` for new API
- Add tests for Grid snapshot functionality
- Add tests for workspace reuse patterns

## Benefits

1. **Consistency**: Grid is the single source of truth for PDE solutions
2. **Memory Efficiency**: No duplicate storage, snapshots opt-in
3. **Reusability**: PDEWorkspace buffers can be reused across solves
4. **Clarity**: Clean separation of concerns (Grid vs Workspace vs Solver)
5. **Flexibility**: Users control PMR allocation strategy
6. **Performance**: Reduced allocations and copies

## Trade-offs

1. **API Change**: Breaking change for existing users (requires migration)
2. **More Verbose**: Users must allocate workspace buffers explicitly
3. **Complexity**: Three-layer architecture requires understanding ownership model

The benefits outweigh the costs, and the new API is more honest about ownership and lifetime management.

## Resolved Design Questions

### 1. Snapshot Indexing
**Question:** Is `record(idx)` parameter a time step or snapshot sequence index?

**Answer:** `record(time_step_idx)` takes time step index. Grid internally maps to snapshot sequence index using `find_snapshot_index()`. Clear separation between:
- Time step index: 0 to n_steps-1 (PDE loop counter)
- Snapshot index: 0 to num_snapshots-1 (storage array index)

### 2. Interpolation and Boundary Handling
**Question:** How are boundaries handled in `value_at()` and Greeks?

**Answer:**
- `value_at(spot)` converts spot → log-moneyness → interpolates using cubic spline
- Boundary clamping: If spot outside grid domain, clamp to boundary value
- Greeks use `CenteredDifference` with grid boundaries (no extrapolation)

### 3. Grid Estimation Error Handling
**Question:** How are failures in `estimate_grid_for_option()` handled?

**Answer:** Function always succeeds (clamps to min/max grid sizes). Extreme params result in boundary grids (min_spatial_points or max_spatial_points), which may be suboptimal but are valid.

### 4. Out-of-Range Snapshot Times
**Question:** Should times beyond maturity or negative be rejected?

**Answer:** Yes - `convert_times_to_indices()` validates and returns error for out-of-range times instead of silently clamping. This catches user mistakes early.

### 5. Non-Uniform Time Steps
**Question:** Does time-to-index conversion work with non-uniform `TimeDomain`?

**Answer:** Current design assumes uniform dt. If non-uniform time steps are needed, `TimeDomain` would need to expose the schedule, and conversion would search the schedule instead of using `t/dt`. Document this assumption.

### 6. Greeks Thread Safety Decision
**Question:** Should AmericanOptionResult be thread-safe?

**Answer:** No - documented as NOT thread-safe for concurrent reads due to lazy Greeks cache. Each solve gets its own result. If thread-safe reads are needed later, can add eager Greeks computation option.

### 7. Snapshot Return Type Consistency
**Question:** Should `Grid::at()` return `std::span<const T>` or `std::span<const double>`?

**Answer:** Must return `std::span<const T>` to be consistent with Grid template parameter. Using hardcoded `double` breaks non-double instantiations.

### 8. Snapshot Time Metadata
**Question:** How do callers know which time each snapshot corresponds to after snapping/deduplication?

**Answer:** Grid exposes `snapshot_times()` method returning the actual recorded times (after snapping). This allows callers to correlate snapshot indices with times reliably.

### 9. Initial Condition and Temporal Event Ordering
**Question:** When should snapshots be recorded relative to initial condition and temporal events?

**Answer:**
- Initial condition (state 0) is recorded **before** time-stepping loop
- All other snapshots are recorded **after** TR-BDF2 stages AND temporal events
- This ensures stored states reflect the true PDE state including discrete dividends
- Snapshot system uses state indices `[0, n_steps]`, not time step indices `[0, n_steps-1]`

### 10. Workspace Size Validation
**Question:** How to prevent out-of-bounds writes if workspace size doesn't match grid size?

**Answer:** AmericanOptionSolver validates `workspace_.size() == grid_spec.n_points()` at start of `solve()` and returns `SolverError` on mismatch. This catches reuse errors early before PDESolver iterations.

## Open Questions

### 1. Serialization/Export
Should Grid provide serialization hooks for saving/loading snapshot data? Defer to future work if needed.

### 2. Partial Snapshots
Should we support strided/partial spatial snapshots to reduce memory? Current design records full `n_space` per snapshot. Defer to future work if memory becomes an issue.
