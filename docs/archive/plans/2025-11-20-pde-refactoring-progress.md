<!-- SPDX-License-Identifier: MIT -->
# PDE Workspace Refactoring - Implementation Progress

**Date:** 2025-11-20
**Session:** Implementation in progress

---

## Completed Work

### ✅ Commit 1: Infrastructure (COMMITTED - SHA: 17771f0)

**Files created:**
- `src/pde/core/grid_with_solution.hpp` - Grid with persistent solution storage
- `src/pde/core/pde_workspace_spans.hpp` - Named spans to PMR buffers
- `tests/grid_with_solution_test.cc` - 4 passing tests
- `tests/pde_workspace_spans_test.cc` - 8 passing tests

**Build targets added:**
- `//src/pde/core:grid_with_solution`
- `//src/pde/core:pde_workspace_spans`
- `//tests:grid_with_solution_test` (✅ ALL PASS)
- `//tests:pde_workspace_spans_test` (✅ ALL PASS)

**Verification:**
```bash
bazel test //tests:grid_with_solution_test --test_output=all  # ✅ 4/4 PASS
bazel test //tests:pde_workspace_spans_test --test_output=all # ✅ 8/8 PASS
```

---

## In Progress Work

### 🔄 Commit 2: Atomic Update (PARTIALLY STARTED - NOT COMPILABLE)

**Status:** Constructor updated, member variables updated, but method bodies NOT yet updated.

**Warning:** ⚠️ Code does NOT compile in current state. This is expected for atomic migration.

**Completed substeps:**

1. ✅ **Task 2.1:** Updated PDESolver member variables
   - File: `src/pde/core/pde_solver.hpp` (lines 245-268)
   - Removed: `workspace_owner_`, `workspace_*`, `output_buffer_`, `solution_storage_`, `u_current_`, `u_old_`, internal arrays (rhs_, jacobian_*, etc.)
   - Added: `grid_with_solution_`, `workspace_` (PDEWorkspaceSpans)
   - Removed: `acquire_workspace()` helper

2. 🔄 **Task 2.2:** Replaced PDESolver constructor (PARTIAL)
   - ✅ Updated constructor signature (lines 84-97)
   - ✅ Updated member initialization list
   - ❌ NOT updated: initialize(), solve(), and 50+ other method calls

3. ✅ Updated BUILD.bazel dependencies
   - Added: `:grid_with_solution`, `:pde_workspace_spans`
   - Removed: `:pde_workspace` dependency

**Remaining substeps for Commit 2:**

4. ❌ **Task 2.3:** Update workspace access (batch 1)
   - Replace 33 instances of `workspace_->rhs()` → `workspace_.rhs()`
   - Files: `src/pde/core/pde_solver.hpp`

5. ❌ **Task 2.4:** Update solution buffer access
   - Replace all `u_current_` → `grid_with_solution_->solution()`
   - Replace all `u_old_` → `grid_with_solution_->solution_prev()`
   - Remove `output_buffer_` logic from solve()
   - Files: `src/pde/core/pde_solver.hpp`

6. ❌ **Task 2.5:** Update solve() to write to Grid
   - After final time step, copy solution to Grid storage
   - Files: `src/pde/core/pde_solver.hpp`

7. ❌ **Task 2.6:** Update AmericanPutSolver
   - New constructor: `(AmericanOptionParams, shared_ptr<Grid>, PDEWorkspaceSpans)`
   - Files: `src/option/american_put_solver.hpp`, `src/option/american_put_solver.cpp`

8. ❌ **Task 2.7:** Update AmericanCallSolver
   - New constructor: `(AmericanOptionParams, shared_ptr<Grid>, PDEWorkspaceSpans)`
   - Files: `src/option/american_call_solver.hpp`, `src/option/american_call_solver.cpp`

9. ❌ **Task 2.8:** Update all tests
   - Files: `tests/pde_solver_test.cc`, `tests/american_option_test.cc`, and ~10 others
   - Each test needs: Create Grid, allocate PMR buffer, create PDEWorkspaceSpans

10. ❌ **Task 2.9:** Delete AmericanSolverWorkspace
    - Files: `src/option/american_solver_workspace.hpp`, `src/option/american_solver_workspace.cpp`
    - Remove BUILD targets

---

## Key Design Decisions (from approved plan)

1. **Grid owns solution** (2 × n_space for current + previous)
2. **Workspace provides temporary arrays** (15 arrays + tridiag @ 2n, PMR-backed)
3. **No dual constructors** - Clean cut, delete old API immediately
4. **Atomic migration** - Big-bang commit, no incremental path
5. **PDESolver becomes pure compute** - No memory management

---

## Automated Script Approach (Recommended)

Due to the large number of mechanical changes (100+ lines across 15+ files), consider using automated refactoring:

```bash
# Replace workspace_-> with workspace_.
find src/pde/core/pde_solver.hpp -type f -exec sed -i 's/workspace_->/workspace_./g' {} \;

# Then manually update u_current_/u_old_ references (context-dependent)
```

---

## Next Steps

**Option A: Continue manually**
- Complete Tasks 2.3-2.9 (estimate: 2-3 hours)
- High risk of errors due to volume

**Option B: Use task automation**
- Write script to perform mechanical replacements
- Manually verify context-sensitive changes
- Lower error rate, faster completion

**Option C: Defer to fresh session**
- Current context: 90K/200K tokens used
- Remaining work requires careful attention
- Recommend checkpoint and resume

---

## Verification Commands

**After Commit 2 completes:**

```bash
# Verify compilation
bazel build //src/pde/core:pde_solver
bazel build //src/option:american_option

# Run tests
bazel test //tests:pde_solver_test --test_output=all
bazel test //tests:american_option_test --test_output=all
bazel test //... --test_output=errors
```

**Expected:** All tests pass with new Grid + Workspace API

---

## Rollback Plan

If Commit 2 fails:

```bash
git reset --hard 17771f0  # Rollback to Commit 1
# Commit 1 infrastructure still available for future use
```

---

**Status:** ⚠️ Paused at Task 2.2 (constructor updated, methods NOT updated, code NOT compilable)

**Recommendation:** Resume in fresh session OR use automated refactoring script
