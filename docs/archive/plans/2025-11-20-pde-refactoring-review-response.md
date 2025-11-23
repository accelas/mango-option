# PDE Workspace Refactoring - Codex Review Response

**Date:** 2025-11-20
**Reviewer:** Codex Subagent
**Status:** BLOCKED - Needs revision before implementation

## Codex Assessment: "Needs revision"

The review identified several critical issues that would prevent the plan from compiling or cause correctness regressions.

---

## Critical Issues Identified

### 1. GridSpacing API Mismatch (P0 - Won't compile)

**Issue:** Plan uses `GridSpacing<T>::create(grid_buffer.span(), ...)` but actual API requires `GridView` constructor.

**Location:** `docs/plans/2025-11-20-pde-workspace-refactoring.md:186-190`

**Current API (from src/pde/core/grid.hpp):**
```cpp
template<typename T>
class GridSpacing {
    // Constructor takes GridView, not span
    explicit GridSpacing(const GridView<T>& grid_view);
};
```

**Fix Required:**
```cpp
// GridWithSolution::create() must:
auto grid_buffer = grid_spec.generate();
auto grid_view = grid_buffer.view();  // Get GridView

auto spacing_result = GridSpacing<T>::create(grid_view);  // Use GridView
```

**Action:** Update GridWithSolution implementation to use GridView pattern correctly.

---

### 2. Missing Workspace Arrays (P0 - Will break solver)

**Issue:** `PDEWorkspaceSpans` only includes 7 arrays, missing several that PDESolver currently uses:
- `dx` (used in boundary conditions: `src/pde/core/pde_solver.hpp:292-304`)
- `lu` (tridiagonal solver workspace)
- `newton_u_old` (Newton iteration scratch)
- `tridiag_workspace` (Thomas solver)
- Stage buffers (`u_stage`)

**Location:** `docs/plans/2025-11-20-pde-workspace-refactoring.md:532-580`

**Current workspace usage:**
```cpp
// Boundary conditions need dx
workspace_->dx()  // Called in apply_boundary_conditions

// Tridiagonal solve needs lu
workspace_->lu()  // Called in thomas_solve

// Newton needs scratch
workspace_->newton_u_old()  // Called in newton_iteration
```

**Fix Required:**
Expand `PDEWorkspaceSpans` to include **all** arrays PDESolver needs:

```cpp
struct PDEWorkspaceSpans {
    static size_t required_size(size_t n) {
        size_t n_padded = ((n + 7) / 8) * 8;
        // 13 arrays total (not 7):
        // rhs, jacobian_diag, jacobian_upper, jacobian_lower,
        // residual, delta_u, psi,
        // dx, lu, newton_u_old, tridiag_workspace, u_stage, u_next
        return 13 * n_padded;
    }

    // Add missing accessors
    std::span<double> dx();
    std::span<double> lu();
    std::span<double> newton_u_old();
    std::span<double> tridiag_workspace();
    std::span<double> u_stage();
    std::span<double> u_next();
};
```

**Action:** Update PDEWorkspaceSpans layout to include all 13 arrays.

---

### 3. Obstacle CRTP Breaks LCP Solver (P0 - Correctness regression)

**Issue:** Proposed obstacle integration uses simple `max(u, psi)` projection, bypassing existing Projected-Thomas LCP solver that prevents "lift above intrinsic" bug.

**Location:** `docs/plans/2025-11-20-pde-workspace-refactoring.md:1483-1528`

**Current implementation:** `src/pde/core/pde_solver.hpp:569-776`
- Uses `solve_implicit_stage_projected` with Dirichlet RHS fixes
- Deep ITM locking via Brennan-Schwartz method
- Correct convergence characteristics

**Proposed implementation:** (Incorrect)
```cpp
// WRONG - simple projection will reintroduce bug
obstacle_.apply(t, x, psi);
for (size_t i = 0; i < n; ++i) {
    u_span[i] = std::max(u_span[i], psi_span[i]);
}
```

**Fix Required:**
Keep existing `solve_implicit_stage_projected` path, integrate obstacle policy into it:

```cpp
if constexpr (!std::is_same_v<typename ObstaclePolicy::tag, obstacles::no_obstacle_tag>) {
    // Use existing Projected-Thomas solver
    auto result = solve_implicit_stage_projected(
        t_start, t_end, dt,
        [this](double t, auto x, auto psi) {
            obstacle_.apply(t, x, psi);  // CRTP obstacle fills psi
        }
    );
} else {
    // Standard implicit solve (no obstacle)
    auto result = solve_implicit_stage(t_start, t_end, dt);
}
```

**Action:** Integrate obstacle CRTP with existing LCP solver, don't replace it.

---

### 4. PDESolver State Duplication (P0 - Won't compile)

**Issue:** New constructor introduces `grid_with_solution_`, `workspace_spans_`, `vector<double> u_current_/u_old_`, but existing class has `span<double> u_current_/u_old_`, `grid_`, `workspace_` used throughout.

**Location:** `docs/plans/2025-11-20-pde-workspace-refactoring.md:713-746`

**Current PDESolver members:**
```cpp
template<typename Derived>
class PDESolver {
private:
    std::span<const double> grid_;
    PDEWorkspace* workspace_;
    std::span<double> u_current_;  // Points into output_buffer or solution_storage_
    std::span<double> u_old_;
};
```

**Proposed (conflicts):**
```cpp
private:
    std::shared_ptr<GridWithSolution<double>> grid_with_solution_;
    PDEWorkspaceSpans workspace_spans_;
    std::vector<double> u_current_;  // CONFLICT - already exists as span
    std::vector<double> u_old_;
```

**Fix Required:**
Either:
- **Option A:** Replace existing members entirely (requires updating all call sites)
- **Option B:** Adapt existing members to point into new owners via spans

**Recommended: Option B** (less invasive)
```cpp
PDESolver(std::shared_ptr<GridWithSolution<double>> grid,
          PDEWorkspaceSpans workspace,
          ObstaclePolicy obstacle = {})
    : grid_with_solution_(grid)
    , workspace_spans_(workspace)
    , obstacle_(std::move(obstacle))
    , grid_(grid->x())  // Adapt existing span to new owner
    , time_(grid->time())
    , n_(grid->n_space())
{
    // Allocate internal u_current_/u_old_ backing storage
    solution_storage_.resize(2 * n_);
    u_current_ = std::span{solution_storage_}.subspan(0, n_);
    u_old_ = std::span{solution_storage_}.subspan(n_, n_);

    // Create PDEWorkspace adapter for legacy code
    workspace_ = create_workspace_adapter(workspace_spans_);
}
```

**Action:** Design migration strategy that preserves existing member names/types during transition.

---

### 5. Buffer Size Validation Missing (P1 - UB risk)

**Issue:** `PDEWorkspaceSpans::from_buffer()` doesn't validate buffer size, risking out-of-bounds subspans.

**Location:** `docs/plans/2025-11-20-pde-workspace-refactoring.md:558-575`

**Current code:** (Unsafe)
```cpp
static PDEWorkspaceSpans from_buffer(std::span<double> buffer, size_t n) {
    // No check if buffer.size() >= required_size(n)
    size_t offset = 0;
    workspace.rhs_ = buffer.subspan(offset, n);  // UB if buffer too small
    // ...
}
```

**Fix Required:**
```cpp
static std::expected<PDEWorkspaceSpans, std::string>
from_buffer(std::span<double> buffer, size_t n) {
    size_t required = required_size(n);
    if (buffer.size() < required) {
        return std::unexpected(std::format(
            "Buffer too small: {} < {} required", buffer.size(), required));
    }

    // Safe to slice
    PDEWorkspaceSpans workspace;
    // ...
    return workspace;
}
```

**Action:** Add buffer size validation with std::expected return type.

---

### 6. Output Buffer Feature Regression (P1 - Silent data loss)

**Issue:** Writing only last 2 steps to Grid ignores existing `output_buffer_` feature and snapshot callbacks.

**Location:** `docs/plans/2025-11-20-pde-workspace-refactoring.md:850-879`

**Current behavior:** Consumers can provide `output_buffer` to collect all time steps or use snapshot callbacks for sparse collection.

**Proposed:** Writes only final 2 steps to Grid, loses per-step output.

**Fix Required:**
Keep existing output_buffer logic, add Grid storage for final 2 steps:

```cpp
std::expected<void, SolverError> solve() {
    // ... time stepping loop ...

    // Existing output_buffer logic (preserve!)
    if (!output_buffer_.empty() && step + 1 < time_.n_steps()) {
        u_old_ = u_current_;
        u_current_ = output_buffer_.subspan((step + 1) * n_, n_);
    }

    // Existing snapshot callback (preserve!)
    if (snapshot_callback_) {
        auto dest = (*snapshot_callback_)(step + 1, t_next);
        if (!dest.empty()) {
            std::copy(u_current_.begin(), u_current_.end(), dest.begin());
        }
    }

    // NEW: Also write final 2 steps to Grid (for theta)
    if (grid_with_solution_) {
        auto grid_current = grid_with_solution_->solution();
        auto grid_prev = grid_with_solution_->solution_prev();
        std::copy(u_current_.begin(), u_current_.end(), grid_current.begin());
        std::copy(u_old_.begin(), u_old_.end(), grid_prev.begin());
    }

    return {};
}
```

**Action:** Preserve all existing output mechanisms, add Grid storage as supplementary.

---

## Missing Test Coverage

### 1. Negative Tests (P1)
- Invalid grid specs (empty, negative range)
- Undersized workspace buffers
- Null pointers in constructors

### 2. Solver Integration Tests (P0)
- Temporal events (dividend adjustments)
- External `output_buffer` usage
- Projected-Thomas LCP path with obstacles
- Non-uniform grids with obstacles (exercises `dx()` access)

### 3. Performance Tests (P2)
- Regression: current design vs new design
- Memory usage: verify 500× claim
- PMR reuse: malloc count in batch operations

---

## Revised Implementation Strategy

### Phase 0: Investigation (REQUIRED BEFORE CODING)

**Task 0.1:** Map current PDESolver dependencies
- Grep for all `workspace_->` calls in `src/pde/core/pde_solver.hpp`
- List all arrays accessed
- Document usage context (boundary, LCP, Newton, tridiag)

**Task 0.2:** Study Projected-Thomas implementation
- Read `solve_implicit_stage_projected` in detail (`src/pde/core/pde_solver.hpp:569-776`)
- Understand Dirichlet RHS fix mechanism
- Document how obstacle callback integrates

**Task 0.3:** Design workspace migration strategy
- How to evolve from current PDEWorkspace pointer to PDEWorkspaceSpans?
- Can both coexist during migration?
- What's the adapter pattern to minimize call-site changes?

**Deliverable:** Updated plan with complete workspace array list and LCP integration design.

### Phase 1-7: Proceed with revised plan

Only begin implementation after Phase 0 investigation is complete and plan is updated.

---

## Decision Required

**Options:**

1. **Pause and investigate** (RECOMMENDED)
   - Complete Phase 0 investigation tasks
   - Update plan with complete array list and LCP integration
   - Re-submit to Codex for review
   - Then proceed with implementation

2. **Proceed with current plan** (NOT RECOMMENDED)
   - High risk of correctness regressions
   - Will require significant rework mid-implementation
   - May introduce subtle bugs (deep ITM pricing)

3. **Simplify scope**
   - Focus only on Grid class (Phase 1)
   - Defer workspace refactoring until Grid is stable
   - Incremental migration reduces risk

---

## Recommendation

**PAUSE implementation. Complete Phase 0 investigation first.**

The Codex review identified fundamental gaps in understanding current solver internals. Attempting implementation without this knowledge will:
- Introduce correctness bugs (LCP solver regression)
- Require extensive rework (missing workspace arrays)
- Risk undefined behavior (buffer overruns)

**Next steps:**
1. Complete Phase 0 investigation (2-3 hours)
2. Update plan with findings
3. Re-submit for Codex review
4. Proceed with implementation only after approval

**Timeline impact:** +1 day for investigation, but saves 2-3 days of debugging and rework.

---

**Status:** Awaiting decision on how to proceed.
