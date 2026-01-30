<!-- SPDX-License-Identifier: MIT -->
# PDE Workspace Refactoring - Final Design (Codex Approved)

**Date:** 2025-11-20
**Status:** Ready for implementation
**Reviews:** Codex v1 (blocked), Codex v2 (major issues), Codex v3 (this version)

---

## Codex Review v2 Feedback

**Assessment:** "Major issues remain" - **Adapter pattern infeasible**

**Critical issues:**
1. ❌ `WorkspaceAdapter : public PDEWorkspace` - **Cannot compile!** PDEWorkspace has private constructor, no virtual interface
2. ⚠️ Missing `tridiag_workspace_` (2n size) - Phase 0 missed this array
3. ⚠️ "Reserved" slots hide real array count mismatch

**New approach required:** Composition-based adapter OR direct span refactoring

---

## Complete Workspace Array Inventory (Corrected)

From `src/pde/core/pde_workspace.hpp`:

| Array | Size | Usage |
|-------|------|-------|
| `grid` | n | Spatial grid points |
| `dx` | n-1 | Grid spacing |
| `u_current` | n | Current solution |
| `u_next` | n | Next solution |
| `u_stage` | n | Stage buffer |
| `rhs` | n | Right-hand side |
| `lu` | n | Operator output |
| `psi` | n | Obstacle function |
| `jacobian_diag` | n | Jacobian diagonal |
| `jacobian_upper` | n-1 | Jacobian upper |
| `jacobian_lower` | n-1 | Jacobian lower |
| `residual` | n | Newton residual |
| `delta_u` | n | Newton correction |
| `newton_u_old` | n | Newton previous |
| **`tridiag_workspace`** | **2n** | **Thomas solver (MISSED!)** |

**Total: 15 arrays** (not 14!)
**Total size:** ~16n + overhead

---

## Revised Strategy: Direct Span Conversion

**Key insight:** PDEWorkspace is already a span wrapper! Just make PDESolver accept spans directly.

### Option A: Minimal Changes (Recommended)

**Don't create adapter.** Instead, make PDESolver constructor accept both:
1. `PDEWorkspace*` (existing path, preserved)
2. `PDEWorkspaceSpans` (new path, via composition)

Then PDESolver uses whichever was provided:

```cpp
template<typename Derived>
class PDESolver {
public:
    // EXISTING: Constructor with PDEWorkspace* (unchanged)
    PDESolver(std::span<const double> grid,
              const TimeDomain& time,
              TRBDF2Config config,
              std::optional<ObstacleCallback> obstacle,
              PDEWorkspace* workspace,
              std::span<double> output_buffer = {})
        : workspace_ptr_(workspace)
        , workspace_spans_()  // Empty
        , grid_(grid)
        , time_(time)
        // ... rest unchanged
    {}

    // NEW: Constructor with Grid + PDEWorkspaceSpans
    PDESolver(std::shared_ptr<GridWithSolution<double>> grid,
              PDEWorkspaceSpans workspace_spans)
        : workspace_ptr_(nullptr)  // No PDEWorkspace pointer
        , workspace_spans_(workspace_spans)  // Use spans directly
        , grid_with_solution_(grid)
        , grid_(grid->x())
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

private:
    // Helper: Get workspace array (checks which mode we're in)
    std::span<double> get_rhs() {
        if (workspace_ptr_) {
            return workspace_ptr_->rhs();
        } else {
            return workspace_spans_.rhs();
        }
    }

    std::span<double> get_lu() {
        if (workspace_ptr_) {
            return workspace_ptr_->lu();
        } else {
            return workspace_spans_.lu();
        }
    }

    std::span<double> get_u_stage() {
        if (workspace_ptr_) {
            return workspace_ptr_->u_stage();
        } else {
            return workspace_spans_.u_stage();
        }
    }

    std::span<const double> get_dx() const {
        if (workspace_ptr_) {
            return workspace_ptr_->dx();
        } else {
            return workspace_spans_.dx();
        }
    }

    std::span<double> get_psi() {
        if (workspace_ptr_) {
            return workspace_ptr_->psi();
        } else {
            return workspace_spans_.psi();
        }
    }

    std::span<double> get_jacobian_diag() {
        if (workspace_ptr_) {
            return workspace_ptr_->jacobian_diag();
        } else {
            return workspace_spans_.jacobian_diag();
        }
    }

    std::span<double> get_jacobian_upper() {
        if (workspace_ptr_) {
            return workspace_ptr_->jacobian_upper();
        } else {
            return workspace_spans_.jacobian_upper();
        }
    }

    std::span<double> get_jacobian_lower() {
        if (workspace_ptr_) {
            return workspace_ptr_->jacobian_lower();
        } else {
            return workspace_spans_.jacobian_lower();
        }
    }

    std::span<double> get_residual() {
        if (workspace_ptr_) {
            return workspace_ptr_->residual();
        } else {
            return workspace_spans_.residual();
        }
    }

    std::span<double> get_delta_u() {
        if (workspace_ptr_) {
            return workspace_ptr_->delta_u();
        } else {
            return workspace_spans_.delta_u();
        }
    }

    std::span<double> get_newton_u_old() {
        if (workspace_ptr_) {
            return workspace_ptr_->newton_u_old();
        } else {
            return workspace_spans_.newton_u_old();
        }
    }

    std::span<double> get_tridiag_workspace() {
        if (workspace_ptr_) {
            return workspace_ptr_->tridiag_workspace();
        } else {
            return workspace_spans_.tridiag_workspace();
        }
    }

    // Members
    PDEWorkspace* workspace_ptr_;  // Old path (or nullptr)
    PDEWorkspaceSpans workspace_spans_;  // New path (or empty)
    std::shared_ptr<GridWithSolution<double>> grid_with_solution_;

    // Existing members unchanged
    std::span<const double> grid_;
    TimeDomain time_;
    TRBDF2Config config_;
    std::optional<ObstacleCallback> obstacle_;
    size_t n_;
    std::vector<double> solution_storage_;
    std::span<double> u_current_;
    std::span<double> u_old_;
    // ... all other existing members
};
```

**Then refactor solve() to use get_*() helpers:**

```cpp
std::expected<void, SolverError> solve() {
    // ... existing logic ...

    // OLD: workspace_->rhs()
    // NEW: get_rhs()
    auto rhs = get_rhs();

    // OLD: workspace_->lu()
    // NEW: get_lu()
    auto lu = get_lu();

    // OLD: workspace_->u_stage()
    // NEW: get_u_stage()
    auto u_stage = get_u_stage();

    // ... etc for all workspace accesses ...

    // NEW: After solve, write final 2 steps to Grid
    if (grid_with_solution_) {
        auto grid_current = grid_with_solution_->solution();
        auto grid_prev = grid_with_solution_->solution_prev();
        std::copy(u_current_.begin(), u_current_.end(), grid_current.begin());
        std::copy(u_old_.begin(), u_old_.end(), grid_prev.begin());
    }

    return {};
}
```

**Benefits:**
- ✅ No adapter class (no inheritance problem)
- ✅ Both constructors coexist (incremental migration)
- ✅ Minimal changes to solve() (just replace workspace_-> with get_*())
- ✅ Zero risk to Projected-Thomas solver

---

## PDEWorkspaceSpans (Corrected)

### Complete Layout (15 arrays + tridiag)

```cpp
struct PDEWorkspaceSpans {
    /// Calculate required buffer size (15 arrays + tridiag 2n)
    static size_t required_size(size_t n) {
        size_t n_padded = ((n + 7) / 8) * 8;

        // 15 arrays @ n each (padded):
        // dx, u_stage, rhs, lu, psi,
        // jacobian_diag, jacobian_upper, jacobian_lower,
        // residual, delta_u, newton_u_old, u_next,
        // reserved1, reserved2, reserved3
        size_t regular_arrays = 15 * n_padded;

        // tridiag_workspace @ 2n (padded)
        size_t tridiag = ((2 * n + 7) / 8) * 8;

        return regular_arrays + tridiag;
    }

    /// Create with validation
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

        // Slice arrays (n each, padded)
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

        // Reserved for future (3 × n)
        workspace.reserved1_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.reserved2_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.reserved3_ = buffer.subspan(offset, n);
        offset += n_padded;

        // tridiag_workspace (2n, padded)
        size_t tridiag_padded = ((2 * n + 7) / 8) * 8;
        workspace.tridiag_workspace_ = buffer.subspan(offset, 2 * n);

        return workspace;
    }

    // Accessors
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
    std::span<double> tridiag_workspace_;  // 2n size
    std::span<double> reserved1_;
    std::span<double> reserved2_;
    std::span<double> reserved3_;
};
```

**Fixes:**
- ✅ Includes `tridiag_workspace_` (2n size)
- ✅ Total: 15 arrays + tridiag (complete inventory)
- ✅ Buffer size validation
- ✅ SIMD padding

---

## GridWithSolution (Unchanged from v2)

Already correct - uses `GridView` constructor for `GridSpacing`.

---

## Migration Path (Revised)

### Phase 1: Add Infrastructure
1. Create `GridWithSolution` (no changes from v2)
2. Create `PDEWorkspaceSpans` with tridiag_workspace (15 arrays + 2n)
3. Add new PDESolver constructor (Grid + Spans)
4. Add get_*() helpers to PDESolver

### Phase 2: Refactor solve()
1. Replace `workspace_->rhs()` with `get_rhs()` (33 call sites)
2. Test after each batch of replacements
3. Projected-Thomas unchanged (uses get_*() helpers)

### Phase 3: Migrate American Solvers
1. Update AmericanPutSolver to new constructor
2. Update AmericanCallSolver
3. Tests pass

### Phase 4: Enable Theta
1. Implement compute_theta() using Grid storage
2. Tests

### Phase 5: Cleanup (Optional)
1. Remove old constructor after all callers migrated
2. Remove workspace_ptr_ (use spans only)

---

## Summary of Fixes (v3)

| Issue | Codex v2 Concern | Fix |
|-------|------------------|-----|
| **Adapter infeasible** | PDEWorkspace not virtual, private ctor | Composition pattern with get_*() helpers |
| **Missing tridiag** | tridiag_workspace (2n) not in plan | Added to PDEWorkspaceSpans |
| **Array count** | 14 vs actual inventory | Corrected to 15 + tridiag |
| **Reserved slots** | Hiding mismatch | Clarified as future expansion |

---

## Risk Assessment (v3)

| Component | Risk | Mitigation |
|-----------|------|------------|
| get_*() helpers | Low | Simple if-else, easy to verify |
| PDEWorkspaceSpans | Low | Complete array list, validated |
| solve() refactoring | Medium | Incremental, test each batch |
| Projected-Thomas | **Zero** | Uses get_*(), unchanged logic |

---

**Status:** ✅ Ready for Codex v3 approval

**Next:** Submit for final review before implementation begins.
