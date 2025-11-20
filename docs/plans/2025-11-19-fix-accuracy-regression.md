# Fix American Option Accuracy Regression Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 40-100x accuracy regression (0.35% → 14.5% error) introduced in commit 2809acd by reverting the problematic CRTP refactoring and re-implementing incrementally with proper testing.

**Architecture:** Revert commit 2809acd entirely, then re-apply its beneficial changes (PMR, API improvements) incrementally with validation at each step. The regression manifests as convergence getting WORSE with finer grids, indicating a fundamental numerical bug in the CRTP refactoring.

**Tech Stack:** C++23, Bazel, GoogleTest, QuantLib (for validation)

---

## Background

**Regression Evidence:**
- CHECKPOINT_1 (commit 5f863f9): 0.35% error ✓
- After 2809acd: 14.5% error ✗
- Convergence anomaly: Finer grids produce WORSE accuracy
  - 51×500: 7.98% error
  - 201×2000: 14.32% error
  - 501×5000: 14.65% error

**Commit 2809acd changes:**
- 32 files changed, 3224 insertions, 1002 deletions
- Refactored PDESolver to use CRTP pattern
- Created american_pde_solver.hpp with AmericanPutSolver/AmericanCallSolver
- Refactored PDEWorkspace to use PMR
- Changed workspace API
- Modified boundary condition application

**Hypothesis:** The CRTP refactoring introduced a subtle bug in PDE solving logic. Reverting completely and re-applying incrementally will identify the specific change that broke accuracy.

---

## Task 1: Establish Baseline and Validate Revert Strategy

**Files:**
- No files modified (diagnostic only)

**Step 1: Run accuracy benchmark at current main**

Run: `cd /home/kai/work/iv_calc/.worktrees/debug-accuracy-regression && bazel run //benchmarks:quantlib_accuracy 2>&1 | head -20`

Expected output: ATM Put 1Y showing ~14.5% error

**Step 2: Checkout CHECKPOINT_1 and validate baseline**

```bash
cd /home/kai/work/iv_calc/.worktrees/debug-accuracy-regression
git checkout CHECKPOINT_1
bazel build //benchmarks:quantlib_accuracy
./bazel-bin/benchmarks/quantlib_accuracy 2>&1 | head -20
```

Expected output: ATM Put 1Y showing ~0.35% error

**Step 3: Return to main**

```bash
git checkout main
```

**Step 4: Document findings**

Create verification log showing both results confirm the regression.

---

## Task 2: Revert Commit 2809acd

**Files:**
- All files affected by commit 2809acd will be reverted

**Step 1: Create revert branch**

```bash
cd /home/kai/work/iv_calc/.worktrees/debug-accuracy-regression
git checkout -b fix/revert-crtp-refactoring
```

**Step 2: Revert the problematic commit**

```bash
git revert 2809acd --no-edit
```

Expected: Git may report conflicts due to subsequent commits building on 2809acd

**Step 3: Handle conflicts if present**

If conflicts occur, analyze each conflict and resolve by:
- Keeping the PRE-2809acd version for core PDE solver logic
- Keeping POST-2809acd version only for unrelated improvements

Document resolution strategy for each conflict.

**Step 4: Build after revert**

```bash
bazel build //benchmarks:quantlib_accuracy
```

Expected: Build may fail due to API incompatibilities with later commits

**Step 5: Document build failures**

List all build errors and categorize:
- API mismatches (need compatibility shims)
- Missing files (need to restore)
- Type errors (need to fix)

---

## Task 3: Fix Build Errors from Revert

**Files:**
- `benchmarks/quantlib_accuracy.cc` (if API changed)
- `src/option/american_option.cpp` (if API reverted)
- Any other files with build errors

**Step 1: Identify minimum API compatibility fixes**

For each build error:
- Check if it's due to workspace API changes
- Check if it's due to solver constructor changes
- Check if it's due to boundary condition changes

**Step 2: Add compatibility shims if needed**

If benchmarks reference new API that was removed:

```cpp
// In american_option.hpp or workspace header
// Add overload that forwards to old API
static std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string>
create(double x_min, double x_max, size_t n_space, size_t n_time) {
    // Forward to old pre-CRTP API
    return std::make_shared<AmericanSolverWorkspace>(
        PrivateTag{}, x_min, x_max, n_space, n_time);
}
```

**Step 3: Build incrementally**

```bash
bazel build //src/option:american_option
bazel build //benchmarks:quantlib_accuracy
```

Fix errors one at a time until build succeeds.

**Step 4: Commit build fixes**

```bash
git add .
git commit -m "Fix build errors after reverting CRTP refactoring

Added compatibility shims to restore old API while keeping
revert of problematic PDE solver changes.

References: #[issue-number]"
```

---

## Task 4: Validate Accuracy Restoration

**Files:**
- No files modified (validation only)

**Step 1: Run accuracy benchmark**

```bash
./bazel-bin/benchmarks/quantlib_accuracy 2>&1 | tee accuracy_after_revert.txt
```

**Step 2: Check ATM Put 1Y error**

```bash
grep "ATM Put 1Y" accuracy_after_revert.txt
```

Expected: Error should be back to ~0.35% (similar to CHECKPOINT_1)

**Step 3: Check convergence behavior**

```bash
grep "Grid.*x" accuracy_after_revert.txt | grep Convergence
```

Expected: Finer grids should show BETTER accuracy (decreasing error)
- 51×500: Higher error
- 201×2000: Lower error
- 501×5000: Lowest error

**Step 4: Run full test suite**

```bash
bazel test //...
```

Expected: All tests pass (or at least same pass rate as CHECKPOINT_1)

**Step 5: Document validation results**

Create comparison table:
| Metric | CHECKPOINT_1 | Current Main | After Revert |
|--------|-------------|--------------|--------------|
| ATM Put 1Y error | 0.35% | 14.5% | 0.35% ✓ |
| Convergence | Normal | Inverted | Normal ✓ |
| Tests passing | X/Y | A/B | X/Y ✓ |

---

## Task 5: Identify Specific Breaking Change (Optional Detailed Investigation)

**Files:**
- Various (read-only analysis)

**Step 1: Create test commits for each component of 2809acd**

Since 2809acd was a mega-commit, decompose it:

1. CRTP refactoring of PDESolver
2. PMR workspace changes
3. Boundary condition refactoring
4. Spatial operator changes
5. american_pde_solver.hpp creation

**Step 2: Apply each component incrementally**

```bash
# For each component:
git checkout -b test/component-N
# Manually apply just that component's changes
bazel build //benchmarks:quantlib_accuracy
./bazel-bin/benchmarks/quantlib_accuracy 2>&1 | grep "ATM Put 1Y"
# Record error
git checkout fix/revert-crtp-refactoring
```

**Step 3: Identify which component broke accuracy**

Document which specific change causes error to jump from 0.35% to ~14%

**Step 4: Analyze the breaking change**

For the identified component:
- Read the diff carefully
- Look for sign errors
- Look for coefficient errors
- Look for off-by-one errors
- Look for boundary condition application order changes

**Step 5: Document root cause**

Write detailed analysis of the bug:
- What changed
- Why it broke accuracy
- How to fix it properly

---

## Task 6: Create Pull Request

**Files:**
- `docs/plans/2025-11-19-fix-accuracy-regression.md` (this plan)
- All reverted files

**Step 1: Push branch**

```bash
git push -u origin fix/revert-crtp-refactoring
```

**Step 2: Create PR with detailed description**

```bash
gh pr create --title "Fix 40-100x accuracy regression by reverting CRTP refactoring" --body "$(cat <<'EOF'
## Summary
Fixes accuracy regression from 0.35% to 14.5% introduced in commit 2809acd.

## Root Cause
Commit 2809acd ("Refactor American option solvers using CRTP pattern") was a
mega-commit (32 files, 3224 insertions) that introduced a subtle numerical bug
causing convergence to WORSEN with finer grids.

## Evidence
- CHECKPOINT_1: 0.35% error ✓
- After 2809acd: 14.5% error ✗
- Convergence anomaly: 501×5000 grid shows WORSE accuracy than 51×500 grid

## Fix
Reverted commit 2809acd entirely. Added minimal compatibility shims to maintain
API compatibility with subsequent commits.

## Validation
- ATM Put 1Y error: 14.5% → 0.35% ✓
- Convergence behavior: Restored (finer grids → better accuracy) ✓
- All tests: Passing ✓

## Test Plan
```bash
bazel run //benchmarks:quantlib_accuracy
bazel test //...
```

## Future Work
The CRTP refactoring had good intentions (PMR, API cleanup). We should:
1. Re-apply beneficial changes incrementally with accuracy testing
2. Identify the specific change that broke accuracy
3. Fix root cause and re-apply CRTP correctly

## References
- Investigation: docs/plans/2025-11-19-fix-accuracy-regression.md
- Problematic commit: 2809acd

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 3: Tag reviewers**

Add appropriate reviewers for numerical accuracy and PDE solver changes.

---

## Task 7: Post-Revert Incremental Improvements (Future)

**Files:**
- TBD based on Task 5 findings

**Scope:** After revert is merged, create follow-up PRs to re-apply beneficial changes from 2809acd:

1. **PMR workspace improvements** - Can likely be re-applied safely
2. **API cleanup** - Can be done independently of CRTP
3. **CRTP refactoring** - Fix the specific bug and re-apply correctly

Each change should:
- Be in its own PR
- Include accuracy benchmarks
- Include convergence tests
- Be validated before merge

**Note:** This task is intentionally vague - it will become concrete once Task 5 identifies the specific breaking change.

---

## Success Criteria

✅ Accuracy restored to <1% error on standard benchmarks
✅ Convergence behavior correct (finer grids → better accuracy)
✅ All existing tests pass
✅ PR merged to main
✅ Root cause documented for future reference

## Rollback Plan

If revert introduces new issues:
1. Revert the revert (git revert <revert-commit>)
2. Investigate why simple revert didn't work
3. Consider bisecting between CHECKPOINT_1 and 2809acd to find exact breaking commit
4. May need to revert multiple commits, not just 2809acd

## Notes

- This is a "revert first, fix later" strategy
- Faster than debugging mega-commit line-by-line
- Enables incremental re-application with validation
- Production users need accuracy fix ASAP
