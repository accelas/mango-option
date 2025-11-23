# PDE Workspace Refactoring - Progress Update

**Date:** 2025-11-20
**Session:** Continued implementation
**Branch:** `wip/pde-workspace-refactoring`

---

## Completed in This Session

### ✅ Task 2.1: Updated PDESolver member variables
- Removed all old members (workspace_*, u_current_, u_old_, rhs_, jacobian_*, output_buffer_, solution_storage_)
- Added: grid_with_solution_, workspace_ (PDEWorkspaceSpans)
- Removed: acquire_workspace() helper

### ✅ Task 2.2: Replaced PDESolver constructor
- New signature: `PDESolver(shared_ptr<GridWithSolution>, PDEWorkspaceSpans, obstacle)`
- Updated member initialization
- Updated `initialize()` to use Grid's solution buffers
- Updated `solve()` to use Grid's solution buffers
- Updated `solution()` accessor to return from Grid

### ✅ Task 2.3: Update workspace access (ALL 33 instances)
- Automated replacement: `workspace_->` → `workspace_.`
- Using sed: `sed -i 's/workspace_->/workspace_./g'`
- Verified: 0 remaining `workspace_->` references

---

## Remaining Work for Commit 2

### ❌ Task 2.4: Update u_current_/u_old_ references

**Status:** PARTIAL - solve() and initialize() updated, but stage methods NOT updated

**Problem:** The stage methods (solve_stage1, solve_stage2) and many helper methods still reference:
- `u_current_` member variable (no longer exists)
- `u_old_` member variable (no longer exists)
- `rhs_` member variable (no longer exists, should use workspace_.rhs())
- `time_` member variable (no longer exists, should use grid_with_solution_->time())

**Required changes:**

1. **Update solve_stage1() signature:**
   ```cpp
   // OLD:
   std::expected<void, SolverError> solve_stage1(double t_n, double t_stage, double dt)

   // NEW:
   std::expected<void, SolverError> solve_stage1(double t_n, double t_stage, double dt,
                                                  std::span<double> u_current,
                                                  std::span<const double> u_prev)
   ```

2. **Update solve_stage2() signature:**
   ```cpp
   // OLD:
   std::expected<void, SolverError> solve_stage2(double t_stage, double t_next, double dt)

   // NEW:
   std::expected<void, SolverError> solve_stage2(double t_stage, double t_next, double dt,
                                                  std::span<double> u_current,
                                                  std::span<const double> u_prev)
   ```

3. **Update all uses of rhs_ to workspace_.rhs():**
   - Lines 339, 340, 349, 387, 388, 397

4. **Update process_temporal_events():**
   - Line 252: `u_current_` → pass u_current as parameter
   - Line 263: `u_current_` → pass u_current as parameter

5. **Update all Newton solver methods:**
   - solve_implicit_stage_dispatch()
   - solve_implicit_stage_newton()
   - solve_implicit_stage_projected()
   - build_jacobian()
   - All currently use member variables that no longer exist

**Estimated scope:** ~50 additional lines need updating

### ❌ Task 2.5: Already done in solve()
The final solution is already in grid_with_solution_->solution() at the end of solve().

### ❌ Task 2.6-2.7: Update American solvers
Not started - depends on completing Task 2.4

### ❌ Task 2.8: Update all tests
Not started - depends on completing Tasks 2.4-2.7

### ❌ Task 2.9: Delete AmericanSolverWorkspace
Not started - depends on completing Task 2.8

---

## Current Compilation Status

**⚠️ Code does NOT compile** - Expected for atomic migration

**Compilation errors expected:**
1. `u_current_` undeclared (member doesn't exist) - ~20 references remain
2. `u_old_` undeclared (member doesn't exist) - ~15 references remain
3. `rhs_` undeclared (member doesn't exist) - ~6 references remain
4. `time_` undeclared (member doesn't exist) - needs grid_with_solution_->time()
5. solve_stage1/solve_stage2 called with wrong number of arguments

---

## Strategy for Completion

### Option A: Manual Method Signature Updates
Continue updating each method to accept u_current/u_prev as parameters:
- Update solve_stage1() and solve_stage2() signatures
- Update all Newton solver methods
- Update process_temporal_events()
- Estimated time: 1-2 hours

### Option B: Automated with Verification
Write a script to:
1. Find all uses of u_current_, u_old_, rhs_, time_
2. Generate replacement patterns
3. Apply and verify compilation

### Recommended: Option A (Manual)
Given the context-sensitive nature of parameter passing, manual is safer.

---

## Next Steps

1. **Update solve_stage1() and solve_stage2():**
   - Add u_current, u_prev parameters
   - Replace u_current_, u_old_ with parameters
   - Replace rhs_ with workspace_.rhs()

2. **Update Newton solver methods:**
   - Pass u_current/u_prev through call chain
   - Replace jacobian_*, residual_, delta_u_, newton_u_old_ with workspace_.*

3. **Update process_temporal_events():**
   - Accept u_current as parameter from solve()

4. **Verify compilation:**
   ```bash
   bazel build //src/pde/core:pde_solver
   ```

5. **Continue with Tasks 2.6-2.9**

---

## Files Modified So Far

- `src/pde/core/pde_solver.hpp` - PARTIAL (constructor done, methods IN PROGRESS)
- `src/pde/core/BUILD.bazel` - DONE (dependencies updated)

**Status:** ~70% complete for Task 2.4, ~40% complete for entire Commit 2

---

## Rollback to Last Known Good State

If needed to restart:
```bash
git checkout main  # Commit 1 infrastructure (APPROVED, tests pass)
git branch -D wip/pde-workspace-refactoring  # Delete WIP
```

Infrastructure (GridWithSolution + PDEWorkspaceSpans) is safe on main and ready for future use.
