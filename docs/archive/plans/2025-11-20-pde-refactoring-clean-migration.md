# PDE Workspace Refactoring - Clean Migration (No Dual Paths)

**Date:** 2025-11-20
**Status:** Final approved design
**Approach:** Direct replacement (no coexistence, no dead code)

---

## Principle: Clean Cut, No Technical Debt

**Old approach (v3):** Dual workspace_ptr_ + workspace_spans_ with get_*() helpers
**New approach:** Direct replacement, delete old code immediately

**Why:**
- Simpler implementation (no branching logic)
- No dead code accumulation
- Cleaner mental model
- Easier to maintain long-term

---

## Migration Strategy: Big Bang Replacement

### Phase 1: Add New Infrastructure (Additive Only)

**Step 1.1:** Create GridWithSolution
**Step 1.2:** Create PDEWorkspaceSpans (complete, 15 arrays + tridiag)
**Step 1.3:** Tests for new classes

**No PDESolver changes yet.**

### Phase 2: Replace PDESolver Constructor (Breaking Change)

**Step 2.1:** Replace old constructor signature

```cpp
// DELETE OLD:
PDESolver(std::span<const double> grid,
          const TimeDomain& time,
          TRBDF2Config config,
          std::optional<ObstacleCallback> obstacle,
          PDEWorkspace* workspace,
          std::span<double> output_buffer = {});

// ADD NEW:
PDESolver(std::shared_ptr<GridWithSolution<double>> grid,
          PDEWorkspaceSpans workspace);
```

**Step 2.2:** Update all member variables

```cpp
// DELETE OLD:
PDEWorkspace* workspace_;
std::span<const double> grid_;
std::span<double> output_buffer_;

// ADD NEW:
std::shared_ptr<GridWithSolution<double>> grid_;
PDEWorkspaceSpans workspace_;
```

**Step 2.3:** Replace all workspace accesses directly

```cpp
// OLD: workspace_->rhs()
// NEW: workspace_.rhs()

// OLD: workspace_->lu()
// NEW: workspace_.lu()

// OLD: workspace_->u_stage()
// NEW: workspace_.u_stage()
```

**No get_*() helpers needed - direct member access!**

### Phase 3: Update All Derived Solvers Simultaneously

**Update in single commit:**
- AmericanPutSolver constructor
- AmericanCallSolver constructor
- Any other PDESolver subclasses

**Step 3.1:** AmericanPutSolver

```cpp
// OLD:
AmericanPutSolver(const PricingParams& params,
                  std::shared_ptr<AmericanSolverWorkspace> workspace,
                  std::span<double> output_buffer = {});

// NEW:
AmericanPutSolver(const PricingParams& params,
                  std::shared_ptr<GridWithSolution<double>> grid,
                  PDEWorkspaceSpans workspace);
```

**Step 3.2:** Update constructor implementation

```cpp
AmericanPutSolver(const PricingParams& params,
                  std::shared_ptr<GridWithSolution<double>> grid,
                  PDEWorkspaceSpans workspace)
    : PDESolver<AmericanPutSolver>(grid, workspace)
    , params_(params)
    , left_bc_(create_left_bc(params))
    , right_bc_(create_right_bc(params))
    , spatial_op_(create_spatial_op(params, grid->spacing()))
{
    // Set obstacle callback
    this->set_obstacle(create_obstacle_callback(params.strike, params.type));
}
```

### Phase 4: Delete Old Infrastructure

**Step 4.1:** Delete AmericanSolverWorkspace class

```bash
git rm src/option/american_solver_workspace.hpp
git rm src/option/american_solver_workspace.cpp
git rm tests/american_solver_workspace_test.cc
```

**Step 4.2:** Keep PDEWorkspace (still used elsewhere)

PDEWorkspace is used by other code, so keep it for now. Can be migrated separately later.

### Phase 5: Update All Tests Simultaneously

**Single commit updates:**
- All American option tests
- All PDE solver tests
- Integration tests

**Test migration pattern:**

```cpp
// OLD:
auto workspace = AmericanSolverWorkspace::create(grid_spec, n_time, &pool).value();
AmericanPutSolver solver(params, workspace, output_buffer);

// NEW:
auto grid = GridWithSolution<double>::create(grid_spec, time).value();
auto workspace_buffer = create_workspace_buffer(n_space, &pool);
auto workspace = PDEWorkspaceSpans::from_buffer_and_grid(
    workspace_buffer, grid->x(), n_space).value();
AmericanPutSolver solver(params, grid, workspace);
```

---

## PDEWorkspaceSpans Complete Implementation

### Helper: Create workspace buffer with grid

```cpp
struct PDEWorkspaceSpans {
    /// Create from buffer and grid (computes dx automatically)
    static std::expected<PDEWorkspaceSpans, std::string>
    from_buffer_and_grid(std::span<double> buffer,
                        std::span<const double> grid,
                        size_t n) {
        auto workspace_result = from_buffer(buffer, n);
        if (!workspace_result.has_value()) {
            return std::unexpected(workspace_result.error());
        }

        auto workspace = workspace_result.value();

        // Compute dx from grid (was done by PDEWorkspace ctor)
        auto dx_span = workspace.dx();
        for (size_t i = 0; i < n - 1; ++i) {
            dx_span[i] = grid[i + 1] - grid[i];
        }

        return workspace;
    }

    /// Calculate required buffer size
    static size_t required_size(size_t n) {
        size_t n_padded = ((n + 7) / 8) * 8;

        // 15 arrays @ n each:
        // dx, u_stage, rhs, lu, psi,
        // jacobian_diag, jacobian_upper, jacobian_lower,
        // residual, delta_u, newton_u_old, u_next,
        // reserved1, reserved2, reserved3
        size_t regular = 15 * n_padded;

        // tridiag_workspace @ 2n
        size_t tridiag = ((2 * n + 7) / 8) * 8;

        return regular + tridiag;
    }

    /// Create from buffer (no grid, dx uninitialized)
    static std::expected<PDEWorkspaceSpans, std::string>
    from_buffer(std::span<double> buffer, size_t n) {
        size_t required = required_size(n);

        if (buffer.size() < required) {
            return std::unexpected(std::format(
                "Workspace buffer too small: {} < {} required for n={}",
                buffer.size(), required, n));
        }

        size_t n_padded = ((n + 7) / 8) * 8;
        PDEWorkspaceSpans workspace;
        workspace.n_ = n;

        size_t offset = 0;

        workspace.dx_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.u_stage_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.rhs_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.lu_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.psi_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.jacobian_diag_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.jacobian_upper_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.jacobian_lower_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.residual_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.delta_u_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.newton_u_old_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.u_next_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.reserved1_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.reserved2_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.reserved3_ = buffer.subspan(offset, n);
        offset += n_padded;

        size_t tridiag_padded = ((2 * n + 7) / 8) * 8;
        workspace.tridiag_workspace_ = buffer.subspan(offset, 2 * n);

        return workspace;
    }

    // Accessors (same as before)
    std::span<double> dx() { return dx_.subspan(0, n_ - 1); }
    std::span<const double> dx() const { return dx_.subspan(0, n_ - 1); }

    std::span<double> u_stage() { return u_stage_; }
    std::span<const double> u_stage() const { return u_stage_; }

    std::span<double> rhs() { return rhs_; }
    std::span<const double> rhs() const { return rhs_; }

    std::span<double> lu() { return lu_; }
    std::span<const double> lu() const { return lu_; }

    std::span<double> psi() { return psi_; }
    std::span<const double> psi() const { return psi_; }

    std::span<double> jacobian_diag() { return jacobian_diag_; }
    std::span<const double> jacobian_diag() const { return jacobian_diag_; }

    std::span<double> jacobian_upper() { return jacobian_upper_.subspan(0, n_ - 1); }
    std::span<const double> jacobian_upper() const { return jacobian_upper_.subspan(0, n_ - 1); }

    std::span<double> jacobian_lower() { return jacobian_lower_.subspan(0, n_ - 1); }
    std::span<const double> jacobian_lower() const { return jacobian_lower_.subspan(0, n_ - 1); }

    std::span<double> residual() { return residual_; }
    std::span<const double> residual() const { return residual_; }

    std::span<double> delta_u() { return delta_u_; }
    std::span<const double> delta_u() const { return delta_u_; }

    std::span<double> newton_u_old() { return newton_u_old_; }
    std::span<const double> newton_u_old() const { return newton_u_old_; }

    std::span<double> u_next() { return u_next_; }
    std::span<const double> u_next() const { return u_next_; }

    std::span<double> tridiag_workspace() { return tridiag_workspace_; }
    std::span<const double> tridiag_workspace() const { return tridiag_workspace_; }

    size_t size() const { return n_; }

private:
    size_t n_;
    std::span<double> dx_;
    std::span<double> u_stage_;
    std::span<double> rhs_;
    std::span<double> lu_;
    std::span<double> psi_;
    std::span<double> jacobian_diag_;
    std::span<double> jacobian_upper_;
    std::span<double> jacobian_lower_;
    std::span<double> residual_;
    std::span<double> delta_u_;
    std::span<double> newton_u_old_;
    std::span<double> u_next_;
    std::span<double> tridiag_workspace_;
    std::span<double> reserved1_;
    std::span<double> reserved2_;
    std::span<double> reserved3_;
};
```

---

## PDESolver Simplified Implementation

### Clean Constructor (No Dual Paths)

```cpp
template<typename Derived>
class PDESolver {
public:
    /// NEW: Only constructor (replaces old)
    PDESolver(std::shared_ptr<GridWithSolution<double>> grid,
              PDEWorkspaceSpans workspace)
        : grid_(grid)
        , workspace_(workspace)
        , time_(grid->time())
        , config_()
        , obstacle_()
        , n_(grid->n_space())
    {
        // Allocate solution storage
        solution_storage_.resize(2 * n_);
        u_current_ = std::span{solution_storage_}.subspan(0, n_);
        u_old_ = std::span{solution_storage_}.subspan(n_, n_);
    }

    // Set obstacle (for American options)
    void set_obstacle(ObstacleCallback callback) {
        obstacle_ = std::move(callback);
    }

    // Solve (updated to use workspace_ member directly)
    std::expected<void, SolverError> solve() {
        double t = time_.t_start();
        const double dt = time_.dt();

        for (size_t step = 0; step < time_.n_steps(); ++step) {
            double t_old = t;

            // Swap buffers
            std::swap(u_current_, u_old_);

            // Stage 1: Trapezoidal rule
            double t_stage1 = t + config_.gamma * dt;
            auto stage1_ok = solve_stage1(t, t_stage1, dt);
            if (!stage1_ok) {
                return std::unexpected(stage1_ok.error());
            }

            // Stage 2: BDF2
            double t_next = t + dt;
            auto stage2_ok = solve_stage2(t_stage1, t_next, dt);
            if (!stage2_ok) {
                return std::unexpected(stage2_ok.error());
            }

            t = t_next;

            process_temporal_events(t_old, t_next, step);
        }

        // Write final 2 steps to Grid (for theta)
        auto grid_current = grid_->solution();
        auto grid_prev = grid_->solution_prev();
        std::copy(u_current_.begin(), u_current_.end(), grid_current.begin());
        std::copy(u_old_.begin(), u_old_.end(), grid_prev.begin());

        return {};
    }

    std::span<const double> solution() const {
        return u_current_;
    }

    std::shared_ptr<GridWithSolution<double>> grid() const {
        return grid_;
    }

private:
    // Simplified members (no dual paths!)
    std::shared_ptr<GridWithSolution<double>> grid_;
    PDEWorkspaceSpans workspace_;
    TimeDomain time_;
    TRBDF2Config config_;
    std::optional<ObstacleCallback> obstacle_;
    size_t n_;

    // Solution storage
    std::vector<double> solution_storage_;
    std::span<double> u_current_;
    std::span<double> u_old_;

    // Stage solving methods use workspace_ directly
    std::expected<void, SolverError> solve_stage1(double t_start, double t_end, double dt) {
        // Uses workspace_.rhs(), workspace_.lu(), etc. directly
        // No get_*() indirection!
    }

    // Build Jacobian (uses workspace_ directly)
    void build_jacobian(double t, double coeff_dt, std::span<const double> u, double eps) {
        // OLD: workspace_->u_stage()
        // NEW: workspace_.u_stage()

        auto u_stage = workspace_.u_stage();
        auto rhs = workspace_.rhs();
        // ... etc
    }

    // Other helper methods...
};
```

**Key simplifications:**
- No `workspace_ptr_` / `workspace_spans_` branching
- No `get_*()` helper methods
- Direct member access: `workspace_.rhs()` instead of `workspace_->rhs()`
- Cleaner, easier to read and maintain

---

## Commit Strategy: Atomic Big-Bang

### Commit 1: Add New Infrastructure
- Add `GridWithSolution` class
- Add `PDEWorkspaceSpans` struct
- Add tests for both
- **Zero changes to existing code**

### Commit 2: Update PDESolver + All Derived Solvers
- Replace PDESolver constructor
- Update all workspace_-> to workspace_.
- Update AmericanPutSolver constructor
- Update AmericanCallSolver constructor
- Update all tests
- Delete AmericanSolverWorkspace
- **Single atomic commit - everything or nothing**

### Commit 3: Enable Theta
- Implement `compute_theta()` using Grid storage
- Add tests

**Benefits of atomic commit:**
- Bisectable (Commit 1 = safe, Commit 2 = breaking but complete)
- No dead code accumulation
- Clear git history
- Easy to review (all changes in one place)

---

## Breaking Change Migration Guide

### For Users of AmericanOptionSolver

**OLD API:**
```cpp
auto workspace = AmericanSolverWorkspace::create(grid_spec, n_time, &pool).value();
AmericanPutSolver solver(params, workspace);
auto result = solver.solve();
```

**NEW API:**
```cpp
// Create grid
auto grid = GridWithSolution<double>::create(grid_spec, time).value();

// Create workspace buffer
std::pmr::synchronized_pool_resource pool;
size_t buffer_size = PDEWorkspaceSpans::required_size(grid->n_space());
std::pmr::vector<double> buffer(buffer_size, &pool);

// Create workspace spans
auto workspace = PDEWorkspaceSpans::from_buffer_and_grid(
    buffer, grid->x(), grid->n_space()).value();

// Create solver
AmericanPutSolver solver(params, grid, workspace);
auto result = solver.solve();

// Access solution from grid
auto solution = grid->solution();

// Compute theta
auto greeks = solver.compute_greeks();
double theta = greeks.value().theta;  // Now works!
```

---

## Testing Strategy

### Before Commit 2 (Preparation)
- Write new tests using new API
- Keep them disabled (DISABLED_ prefix)
- Verify new infrastructure works in isolation

### Commit 2 (Atomic Switch)
- Enable new tests
- Update all existing tests
- Remove old tests for AmericanSolverWorkspace
- **All tests must pass in single commit**

### After Commit 2 (Verification)
- Run full test suite
- Run benchmarks (verify no regression)
- Run deep ITM tests (verify Projected-Thomas still works)

---

## Timeline

- **Commit 1:** 2 hours (new infrastructure + tests)
- **Commit 2:** 4 hours (atomic update of PDESolver + all derived + all tests)
- **Commit 3:** 30 minutes (theta)
- **Total:** ~6.5 hours

**Reduced from v3:** ~6.5 hours (vs 6-7 hours with dual paths)
**Simpler code:** No get_*() helpers, no branching, no dead code

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Breaking change | High | Atomic commit strategy |
| Test updates | Medium | Update all in one commit |
| Missing edge case | Low | Comprehensive test coverage |
| Regression | Low | Benchmark suite |

---

## Summary

**Approach:** Clean cut, no technical debt

**Advantages over dual-path:**
- Simpler implementation (no get_*() helpers)
- No dead code
- Clearer mental model
- Easier long-term maintenance

**Disadvantages:**
- Breaking change (requires updating all call sites)
- Larger atomic commit

**Decision:** Worth it. Clean code > incremental compatibility.

---

**Status:** ✅ Final design - ready for implementation

**Next:** Begin Commit 1 (add infrastructure)
