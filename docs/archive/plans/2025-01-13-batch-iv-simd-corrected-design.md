<!-- SPDX-License-Identifier: MIT -->
# Batch IV Solver with SIMD - Corrected Design

**Date:** 2025-01-13
**Status:** Design (Post-Review, Corrected)
**Authors:** Claude Code + User + Codex Review

---

## ⚠️ IMPLEMENTATION PRIORITY

### VALIDATE BEFORE IMPLEMENTING: OpenMP Baseline Required

**CRITICAL LESSON FROM PR 151:** A batch PDE SIMD implementation (8,000 lines) was closed without merge because simple OpenMP parallel-for (10 lines) was **2.1× faster**. Cache locality issues completely negated SIMD benefits.

**BEFORE implementing any SIMD code, we MUST:**

1. **Implement Simple OpenMP Baseline** (Phase -1, ~1 day):
   ```cpp
   #pragma omp parallel for schedule(dynamic)
   for (size_t i = 0; i < queries.size(); ++i) {
       results[i] = solver.solve(queries[i]);  // Existing scalar path
   }
   ```
   - Use thread-local workspaces (already implemented)
   - Measure throughput on real option chains (SPY, SPX)
   - Record scaling efficiency vs core count
   - **This is the bar SIMD must beat**

2. **Profile Span Distribution** (Phase -1, ~1 day):
   - Instrument scalar solver to log unique m-span counts per chain
   - Test assumption that U ≤ 2 for typical sorted strikes
   - If U > 4 frequently, SIMD grouping may thrash cache (like PR 151)

3. **Decision Gate:**
   - ✅ If OpenMP < target latency AND U ≤ 2: **Proceed with SIMD**
   - ❌ If OpenMP meets target OR U >> 2: **Stop, use OpenMP**

**Only after validation passes should we proceed with SIMD implementation.**

---

### Phase 0: PMR Workspace Infrastructure (IF SIMD Approved)

After OpenMP baseline validation, the **first implementation priority** is to create the PMR-based workspace infrastructure (`BatchIVWorkspace`). This is a pure memory management change with minimal algorithm impact.

**Why PMR first:**
- Separates memory management from algorithmic changes
- Minimal risk (no algorithm modifications)
- Establishes foundation for later SIMD work
- Can be tested in isolation
- Zero impact on existing code
- **Used by both OpenMP baseline AND SIMD path**

**See Implementation Roadmap → Phase -1 and Phase 0 for complete details.**

---

## Executive Summary

This document presents a corrected design for adding SIMD batch processing to the existing IV solver (`IVSolverInterpolated`). The initial design had critical flaws identified by technical review:

1. ❌ **Span divergence**: Assumed single tensor contraction could serve all lanes
2. ❌ **API mismatch**: Required per-lane sigma but API took scalar
3. ❌ **Moneyness bug**: Used `spot/strike` instead of `spot/K_ref`
4. ❌ **Thread safety**: Mutable workspace in const method
5. ❌ **Incomplete validation**: No per-lane tracking or error handling

This corrected design addresses all issues while preserving the 8-10× speedup target.

---

## Goals & Constraints

### Goals
1. **Preserve existing API**: `solve(IVQuery)` remains unchanged (backward compatible)
2. **Add batch processing**: New `solve_batch(queries, workspace)` for option chains
3. **SIMD acceleration**: 8-10× speedup for batches of 8+ strikes with AVX512
4. **Same algorithm**: Newton-Raphson with finite difference vega (no algorithmic changes)
5. **Maintain correctness**: Bit-identical results to scalar path (within FP tolerance)

### Constraints
1. **No analytic vega**: American options require finite differences (free boundary problem)
2. **K_ref contract**: B-spline surface fitted with moneyness = S/K_ref
3. **Thread safety**: No mutable state in solver, caller provides workspace
4. **Validation parity**: Replicate all scalar path validation per lane

---

## Problem Analysis

### Issue 1: Span Divergence (CRITICAL)

**Root Cause:**
The B-spline coefficient tensor is indexed by all four dimensions:
```cpp
int idx = ((im_idx * Nt + jt_idx) * Nv + kv_idx) * Nr + lr_idx;
```

Changing `im_idx` shifts the coefficient tile by `Nt × Nv × Nr` positions. Different moneyness values fall in different knot spans, requiring different coefficient blocks.

**Example:**
```
Strikes:  [90,  95,  100, 105, 110]
Moneyness: [1.11, 1.05, 1.00, 0.95, 0.91]
Spans:     [7,   7,   6,   6,   5]    ← 3 unique spans!
```

**Incorrect Assumption:**
```cpp
// ❌ WRONG: Single G[4] cannot serve all spans
double G[4];
contract_tensor(wt, wv, wr, G);  // Assumes one im span
for (lane in batch) {
    price[lane] = dot(basis_m[lane], G);  // Wrong if span differs!
}
```

**Solution: Unique-Span Grouping**

Contract once per unique span, then process all lanes in that span group:

```cpp
// ✅ CORRECT: Group by span, contract per group
struct SpanGroup {
    int span_idx;
    std::vector<size_t> lanes;
    double G[4];  // Contraction for this span
};

auto groups = partition_by_span(moneyness, batch_size);

for (auto& group : groups) {
    // Contract once per unique span (64 FMAs)
    contract_tensor(wt, wv, wr, group.span_idx, group.G);

    // Evaluate all lanes in this group (4 FMAs each)
    for (size_t lane : group.lanes) {
        double wm[4];
        cubic_basis_nonuniform(tm_, group.span_idx, moneyness[lane], wm);
        prices[lane] = dot_product(wm, group.G);
    }
}
```

**Performance Analysis:**
- Best case (U=1 unique span): `64/8 + 4 = 12 FMAs/strike` → **21× faster** (vs 256 FMAs)
- Realistic (U=2 for sorted strikes): `128/8 + 4 = 20 FMAs/strike` → **13× faster**
- Worst case (U=8): `512/8 + 4 = 68 FMAs/strike` → **4× faster**

**Key insight:** Even worst case is faster! Sorted strikes (typical option chains) give U≈1-2.

---

### Issue 2: Per-Lane Sigma

**Root Cause:**
Newton iteration produces different sigma per lane:
```cpp
simd_t sigma_vec = {0.20, 0.25, 0.22, 0.28, 0.19, 0.26, 0.24, 0.21};
```

But initial API took scalar:
```cpp
void eval_batch(..., double sigma, ...);  // ❌ Can't pass vector!
```

**Solution: Accept Per-Lane Parameters**

```cpp
// ✅ CORRECT: General API accepts spans
void eval_price_and_vega_batch(
    std::span<const double> moneyness,  // Per-lane
    std::span<const double> tau,         // Per-lane (or broadcast from scalar)
    std::span<const double> sigma,       // Per-lane (Newton iterations differ!)
    std::span<const double> rate,        // Per-lane (or broadcast from scalar)
    double vega_epsilon,
    std::span<double> out_prices,
    std::span<double> out_vegas) const;
```

**Convenience overload for option chains:**
```cpp
// Fast path: shared tau, rate; per-lane sigma
void eval_price_and_vega_batch(
    std::span<const double> moneyness,
    double tau,                    // Scalar (shared)
    std::span<const double> sigma, // Vector (per-lane)
    double rate,                   // Scalar (shared)
    double vega_epsilon,
    std::span<double> out_prices,
    std::span<double> out_vegas) const;
```

---

### Issue 3: Moneyness Semantics

**Root Cause:**
B-spline surface was **fitted using `K_ref`** as reference strike. All prices normalized by `K_ref`.

**Incorrect Code:**
```cpp
// ❌ WRONG: Breaks calibration!
moneyness[i] = spot / queries[i].strike;
```

**Correct Code:**
```cpp
// ✅ CORRECT: Match scalar solver (line 103)
moneyness[i] = queries[i].spot / K_ref_;

// Then scale result by strike (match eval_price at line 149)
scaled_price = spline_price * (queries[i].strike / K_ref_);
```

**Why this matters:**
- Surface stores `V(m, τ, σ, r)` where `m = S/K_ref`
- Querying with `m = S/K` accesses wrong region → arbitrage
- Must preserve scaling contract from existing code

---

### Issue 4: Thread Safety

**Root Cause:**
Mutable workspace in const method = data race

**Incorrect Code:**
```cpp
class IVSolverInterpolated {
    mutable BatchIVWorkspace workspace_;  // ❌ Race condition!

    std::vector<IVResult> solve_batch(...) const;  // Multiple threads!
};
```

**Correct Code:**
```cpp
class IVSolverInterpolated {
    // No mutable state!

    // Caller provides workspace (thread-local)
    std::vector<IVResult> solve_batch(
        std::span<const IVQuery> queries,
        BatchIVWorkspace& workspace) const;  // ✅ Thread-safe
};

// Usage:
thread_local BatchIVWorkspace workspace(64);
auto results = solver.solve_batch(queries, workspace);
```

---

### Issue 5: Per-Lane Validation & Tracking

**Missing Infrastructure:**
1. No `validate_query()` per lane
2. No `adaptive_bounds()` per lane
3. No per-lane iteration counts
4. No per-lane residual tracking
5. No per-lane failure reasons

**Solution: Extend Workspace**

```cpp
struct BatchIVWorkspace {
    // Newton state (SoA)
    std::span<double> sigma();
    std::span<double> moneyness();
    std::span<double> prices();
    std::span<double> vegas();

    // Per-lane tracking (NEW)
    std::span<int> iterations();        // Iteration count per lane
    std::span<double> residuals();      // Final error per lane
    std::span<bool> converged_flags();  // Convergence status per lane
    std::span<double> sigma_min();      // Adaptive bounds per lane
    std::span<double> sigma_max();
};
```

---

## Corrected Design

### Architecture Overview

```
Input: std::span<IVQuery> queries (option chain)
    ↓
Validation Phase (per-lane)
├─ validate_query() for each
├─ adaptive_bounds() for each
└─ Reject invalid queries early
    ↓
Group by Shared Parameters
├─ Identify shared (spot, maturity, rate)
└─ Extract per-lane (strikes → moneyness)
    ↓
SIMD Newton Loop (batches of W=8)
├─ For each batch:
│  ├─ Compute moneyness[W] = spot / K_ref (NOT spot/strike!)
│  ├─ Partition by unique m-spans → groups
│  ├─ Newton iterations (masked SIMD):
│  │  ├─ For each unique span group:
│  │  │  ├─ Contract at σ-ε, σ, σ+ε → 3 × G[4]
│  │  │  └─ Evaluate lanes in group → prices[W], vegas[W]
│  │  ├─ Update: σ[W] ← σ[W] - (price[W] - market[W]) / vega[W]
│  │  ├─ Check convergence per lane
│  │  └─ Update masks, track iterations
│  └─ Scale results: price × (strike / K_ref)
└─ Return vector<IVResult>
```

---

## Component Design

**Implementation Order:** Component 3 (BatchIVWorkspace) should be implemented **first**, before any SIMD components. This establishes the memory management infrastructure with zero algorithm changes.

---

### Component 3: BatchIVWorkspace (IMPLEMENT FIRST)

**File:** `src/option/batch_iv_workspace.hpp` (NEW)

**Rationale:** Pure memory management infrastructure. No algorithm changes. Establishes foundation for later SIMD integration.

```cpp
#pragma once

#include "src/pde/memory/workspace_base.hpp"
#include <span>
#include <string>

namespace mango {

/// Workspace for batch IV calculation with SIMD
/// Thread-local: each thread should have its own instance
///
/// Design:
/// - Extends WorkspaceBase for PMR allocation (64-byte aligned)
/// - SoA (Structure of Arrays) layout for SIMD efficiency
/// - Per-lane tracking for validation and convergence
/// - Zero-cost reset() for workspace reuse
///
/// Thread safety:
/// - Each thread must have separate instance (workspace as parameter)
/// - No shared mutable state
/// - Safe for OpenMP parallel regions
class BatchIVWorkspace : public WorkspaceBase {
public:
    /// Create workspace for up to max_batch_size concurrent queries
    /// @param max_batch_size Maximum batch size (default: 64)
    explicit BatchIVWorkspace(size_t max_batch_size = 64);

    // Newton iteration state (SoA layout for SIMD)
    std::span<double> sigma() { return {sigma_, max_size_}; }
    std::span<double> moneyness() { return {moneyness_, max_size_}; }
    std::span<double> market_prices() { return {market_prices_, max_size_}; }
    std::span<double> strikes() { return {strikes_, max_size_}; }

    std::span<double> prices() { return {prices_, max_size_}; }
    std::span<double> vegas() { return {vegas_, max_size_}; }
    std::span<double> residuals() { return {residuals_, max_size_}; }

    // Per-lane tracking (NEW - addresses Issue 5)
    std::span<int> iterations() { return {iterations_, max_size_}; }
    std::span<bool> converged_flags() { return {converged_flags_, max_size_}; }
    std::span<double> sigma_min_bounds() { return {sigma_min_, max_size_}; }
    std::span<double> sigma_max_bounds() { return {sigma_max_, max_size_}; }

    // Failure tracking
    std::span<std::string> failure_reasons() { return {failure_reasons_, max_size_}; }

    size_t max_size() const { return max_size_; }

    /// Reset workspace for reuse (zero-cost PMR release)
    /// WARNING: Invalidates all previously returned spans!
    /// Must be called before each solve_batch() to clear stale state
    void reset() {
        destroy_arrays();  // Destroy placement-new objects first!
        resource_.reset();
        allocate_arrays();
    }

    /// Destructor ensures placement-new objects are destroyed
    ~BatchIVWorkspace() {
        destroy_arrays();
    }

private:
    void allocate_arrays();
    void destroy_arrays();  // NEW: Must destroy std::string objects

    size_t max_size_;

    // Allocated via PMR (64-byte aligned for AVX-512)
    double* sigma_;
    double* moneyness_;
    double* market_prices_;
    double* strikes_;
    double* prices_;
    double* vegas_;
    double* residuals_;
    double* sigma_min_;
    double* sigma_max_;

    int* iterations_;
    bool* converged_flags_;
    std::string* failure_reasons_;
};

}  // namespace mango
```

**Implementation Notes:**

1. **PMR Allocation and Destruction:**
   ```cpp
   void BatchIVWorkspace::allocate_arrays() {
       const size_t padded = pad_to_simd(max_size_);

       // All double arrays: 64-byte aligned
       sigma_ = static_cast<double*>(resource_.allocate(padded * sizeof(double)));
       moneyness_ = static_cast<double*>(resource_.allocate(padded * sizeof(double)));
       prices_ = static_cast<double*>(resource_.allocate(padded * sizeof(double)));
       vegas_ = static_cast<double*>(resource_.allocate(padded * sizeof(double)));
       residuals_ = static_cast<double*>(resource_.allocate(padded * sizeof(double)));
       market_prices_ = static_cast<double*>(resource_.allocate(padded * sizeof(double)));
       strikes_ = static_cast<double*>(resource_.allocate(padded * sizeof(double)));
       sigma_min_ = static_cast<double*>(resource_.allocate(padded * sizeof(double)));
       sigma_max_ = static_cast<double*>(resource_.allocate(padded * sizeof(double)));

       // Non-double arrays
       iterations_ = static_cast<int*>(resource_.allocate(padded * sizeof(int)));
       converged_flags_ = static_cast<bool*>(resource_.allocate(padded * sizeof(bool)));

       // String array (placement new required)
       failure_reasons_ = static_cast<std::string*>(
           resource_.allocate(max_size_ * sizeof(std::string)));
       for (size_t i = 0; i < max_size_; ++i) {
           new (&failure_reasons_[i]) std::string();
       }
   }

   void BatchIVWorkspace::destroy_arrays() {
       // CRITICAL: Destroy placement-new std::string objects before PMR release
       // Even empty strings may hold heap state (SSO not guaranteed)
       if (failure_reasons_) {
           for (size_t i = 0; i < max_size_; ++i) {
               failure_reasons_[i].~basic_string();
           }
           failure_reasons_ = nullptr;
       }
       // Note: PMR-allocated POD types (double, int, bool) don't need destruction
   }
   ```

2. **Memory Layout (SoA):**
   ```
   [sigma_0, sigma_1, ..., sigma_N]     ← Contiguous, SIMD-friendly
   [moneyness_0, moneyness_1, ..., moneyness_N]
   [prices_0, prices_1, ..., prices_N]
   ...
   ```
   vs. AoS (Array of Structures):
   ```
   [sigma_0, moneyness_0, price_0, sigma_1, moneyness_1, price_1, ...]  ← Poor cache locality
   ```

3. **Thread Safety:**
   ```cpp
   // ✅ CORRECT: Each thread has own workspace
   #pragma omp parallel
   {
       thread_local BatchIVWorkspace workspace(64);
       #pragma omp for
       for (size_t i = 0; i < chain_count; ++i) {
           auto results = solver.solve_batch(chains[i], workspace);
       }
   }

   // ❌ WRONG: Shared workspace = data race
   BatchIVWorkspace workspace(64);
   #pragma omp parallel for
   for (size_t i = 0; i < chain_count; ++i) {
       auto results = solver.solve_batch(chains[i], workspace);  // RACE!
   }
   ```

**Testing Strategy:**

```cpp
TEST(BatchIVWorkspaceTest, AllocationSizeCorrect) {
    BatchIVWorkspace ws(64);

    // Verify spans have correct size
    EXPECT_EQ(ws.sigma().size(), 64);
    EXPECT_EQ(ws.moneyness().size(), 64);
    // ... etc
}

TEST(BatchIVWorkspaceTest, AlignmentIsAVX512) {
    BatchIVWorkspace ws(64);

    // All double arrays must be 64-byte aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ws.sigma().data()) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ws.moneyness().data()) % 64, 0);
    // ... etc
}

TEST(BatchIVWorkspaceTest, ResetInvalidatesSpans) {
    BatchIVWorkspace ws(64);

    auto sigma_before = ws.sigma();
    ws.reset();
    auto sigma_after = ws.sigma();

    // Pointers should differ (new allocation)
    EXPECT_NE(sigma_before.data(), sigma_after.data());
}

TEST(BatchIVWorkspaceTest, ThreadSafety) {
    // Each thread creates own workspace
    std::vector<int> results(100);

    #pragma omp parallel for
    for (int i = 0; i < 100; ++i) {
        BatchIVWorkspace ws(8);  // Thread-local
        ws.sigma()[0] = static_cast<double>(i);
        results[i] = static_cast<int>(ws.sigma()[0]);
    }

    // Verify no data races
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(results[i], i);
    }
}

TEST(BatchIVWorkspaceTest, MultipleResetCalls) {
    // NEW: Verify repeated reset() properly destroys and reinitializes strings
    BatchIVWorkspace ws(16);

    for (int round = 0; round < 5; ++round) {
        // Write long strings (force heap allocation beyond SSO)
        for (size_t i = 0; i < 16; ++i) {
            ws.failure_reasons()[i] = std::string(100, 'A' + round);
        }

        // Reset should destroy strings and reallocate
        ws.reset();

        // Spans should be valid with fresh empty strings
        EXPECT_EQ(ws.failure_reasons().size(), 16);
        for (size_t i = 0; i < 16; ++i) {
            EXPECT_TRUE(ws.failure_reasons()[i].empty());
        }
    }
}

TEST(BatchIVWorkspaceTest, LongFailureReasons) {
    // NEW: Verify no heap allocations escape PMR
    BatchIVWorkspace ws(8);

    // Write very long strings (definitely heap-allocated)
    for (size_t i = 0; i < 8; ++i) {
        ws.failure_reasons()[i] = std::string(1000, 'X');
    }

    // Verify write succeeded
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(ws.failure_reasons()[i].size(), 1000);
    }

    // Destroy and reallocate
    ws.reset();

    // Should have fresh empty strings
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_TRUE(ws.failure_reasons()[i].empty());
    }
}

TEST(BatchIVWorkspaceTest, ConcurrentDistinctWorkspaces) {
    // NEW: Mirror real calling pattern with per-thread workspaces
    constexpr size_t num_chains = 100;
    std::vector<std::vector<double>> results(num_chains);

    #pragma omp parallel
    {
        // Each thread has own workspace
        thread_local BatchIVWorkspace workspace(16);

        #pragma omp for
        for (size_t chain = 0; chain < num_chains; ++chain) {
            workspace.reset();  // Clear stale state before each use

            // Simulate solving a batch
            auto sigma_span = workspace.sigma();
            for (size_t i = 0; i < 16; ++i) {
                sigma_span[i] = static_cast<double>(chain * 100 + i);
            }

            // Extract results
            results[chain].assign(sigma_span.begin(), sigma_span.end());
        }
    }

    // Verify no cross-contamination
    for (size_t chain = 0; chain < num_chains; ++chain) {
        ASSERT_EQ(results[chain].size(), 16);
        for (size_t i = 0; i < 16; ++i) {
            EXPECT_EQ(results[chain][i], static_cast<double>(chain * 100 + i));
        }
    }
}
```

**Success Criteria:**
- All unit tests pass (including new destruction/reset tests)
- 64-byte alignment verified for AVX-512
- Zero algorithm changes (no modifications to existing solvers)
- Thread-safe with separate instances
- Proper resource cleanup (no leaks from placement-new strings)
- Ready for SIMD integration

**Additional Considerations (from Technical Review):**

1. **Padded vs Logical Sizes**: Consider exposing both `max_size()` (logical) and `padded_size()` (SIMD-aligned) similar to `PDEWorkspace`. This allows later SIMD kernels to safely operate on padded tails without interface changes.

2. **Future Scratch Buffers**: Phase 1-2 SIMD work will need additional scratch arrays for span grouping:
   - `span_ids[]` - moneyness span index per lane
   - `group_offsets[]` - start index per unique span group
   - `mask_worklist[]` - active lane masks for convergence

   Consider allocating these in Phase 0 to prevent structural churn later.

3. **Reset Contract**: Document that callers **must** call `reset()` before each `solve_batch()` to clear stale convergence flags and failure reasons. Without this, previous batch state bleeds into new queries.

4. **Include Style**: Follow existing codebase pattern with relative includes (e.g., `#include "workspace_base.hpp"` not `#include "src/pde/memory/workspace_base.hpp"`).

5. **Build Dependencies**: Ensure Bazel target links OpenMP before adding `#pragma omp` tests, otherwise tests will compile but not parallelize.

---

### Component 1: BSpline4D_FMA::eval_batch() (IMPLEMENT AFTER WORKSPACE)

**File:** `src/interpolation/bspline_4d.hpp`

```cpp
class BSpline4D_FMA {
public:
    // Existing (unchanged)
    double eval(double m, double tau, double sigma, double r) const;

    // NEW: Batch evaluation with unique-span grouping
    /// Evaluate B-spline for multiple queries
    /// Best performance when queries share tau, sigma, rate
    /// Handles arbitrary moneyness values via span grouping
    [[gnu::target_clones("default","avx2","avx512f")]]
    void eval_batch(
        std::span<const double> moneyness,  // Per-query (can vary)
        std::span<const double> tau,         // Per-query or broadcast
        std::span<const double> sigma,       // Per-query or broadcast
        std::span<const double> rate,        // Per-query or broadcast
        std::span<double> out_prices) const; // Output prices

    // Convenience overload for option chains (shared tau, sigma, rate)
    void eval_batch(
        std::span<const double> moneyness,
        double tau,    // Scalar
        double sigma,  // Scalar
        double rate,   // Scalar
        std::span<double> out_prices) const {
        // Broadcast scalars and forward to general version
        // (implementation detail)
    }
};
```

**Implementation Strategy:**

```cpp
void BSpline4D_FMA::eval_batch(
    std::span<const double> moneyness,
    double tau,
    double sigma,
    double rate,
    std::span<double> out_prices) const
{
    const size_t n = moneyness.size();

    // Find shared knot spans for tau, sigma, rate
    const int jt = find_span_cubic(tt_, tau);
    const int kv = find_span_cubic(tv_, sigma);
    const int lr = find_span_cubic(tr_, rate);

    // Evaluate shared basis functions
    double wt[4], wv[4], wr[4];
    cubic_basis_nonuniform(tt_, jt, tau, wt);
    cubic_basis_nonuniform(tv_, kv, sigma, wv);
    cubic_basis_nonuniform(tr_, lr, wr);

    // Partition moneyness by unique spans
    struct SpanGroup {
        int span_idx;
        std::vector<size_t> lanes;
    };

    std::vector<SpanGroup> groups;
    for (size_t i = 0; i < n; ++i) {
        double m_clamped = clamp_query(moneyness[i], m_.front(), m_.back());
        int im = find_span_cubic(tm_, m_clamped);

        // Find or create group for this span
        auto it = std::find_if(groups.begin(), groups.end(),
            [im](const auto& g) { return g.span_idx == im; });

        if (it == groups.end()) {
            groups.push_back({im, {i}});
        } else {
            it->lanes.push_back(i);
        }
    }

    // Process each span group
    for (const auto& group : groups) {
        // Partial tensor contraction for this span
        double G[4] = {0.0};

        for (int b = 0; b < 4; ++b) {
            for (int c = 0; c < 4; ++c) {
                const double weight_tc = wt[b] * wv[c];
                for (int d = 0; d < 4; ++d) {
                    const double w = weight_tc * wr[d];
                    for (int a = 0; a < 4; ++a) {
                        const size_t idx = coefficient_index(
                            group.span_idx - a, jt - b, kv - c, lr - d);
                        G[a] = std::fma(c_[idx], w, G[a]);
                    }
                }
            }
        }

        // Evaluate all lanes in this group
        for (size_t lane : group.lanes) {
            double wm[4];
            double m_clamped = clamp_query(moneyness[lane], m_.front(), m_.back());
            cubic_basis_nonuniform(tm_, group.span_idx, m_clamped, wm);

            // Dot product (4 FMAs)
            double price = 0.0;
            for (int a = 0; a < 4; ++a) {
                price = std::fma(wm[a], G[a], price);
            }
            out_prices[lane] = price;
        }
    }
}
```

**Performance:**
- Contraction: 64 FMAs × U (unique spans)
- Per-lane: 4 FMAs
- Total: `(64U + 4n) FMAs` where U ≤ n
- For n=8, U=1: 64/8 + 4 = 12 FMAs/strike → **21× faster** (vs 256)

---

### Component 2: BSpline4D_FMA::eval_vega_batch()

**File:** `src/interpolation/bspline_4d.hpp`

```cpp
class BSpline4D_FMA {
public:
    // NEW: Batch price + vega via finite differences
    /// Computes prices and vegas for multiple queries
    /// Uses 3× contractions per unique span (σ-ε, σ, σ+ε)
    [[gnu::target_clones("default","avx2","avx512f")]]
    void eval_price_and_vega_batch(
        std::span<const double> moneyness,
        double tau,                          // Shared
        std::span<const double> sigma,       // Per-lane (Newton iterations!)
        double rate,                         // Shared
        double vega_epsilon,
        std::span<double> out_prices,
        std::span<double> out_vegas) const;
};
```

**Implementation Strategy:**

```cpp
void BSpline4D_FMA::eval_price_and_vega_batch(
    std::span<const double> moneyness,
    double tau,
    std::span<const double> sigma,
    double rate,
    double vega_epsilon,
    std::span<double> out_prices,
    std::span<double> out_vegas) const
{
    const size_t n = moneyness.size();

    // Shared basis for tau, rate
    const int jt = find_span_cubic(tt_, tau);
    const int lr = find_span_cubic(tr_, rate);
    double wt[4], wr[4];
    cubic_basis_nonuniform(tt_, jt, tau, wt);
    cubic_basis_nonuniform(tr_, lr, rate, wr);

    // Partition by unique (m_span, σ_span) pairs
    struct SpanPair {
        int im, kv;
        std::vector<size_t> lanes;
    };

    std::vector<SpanPair> groups;
    for (size_t i = 0; i < n; ++i) {
        double m_clamped = clamp_query(moneyness[i], m_.front(), m_.back());
        double s_clamped = clamp_query(sigma[i], v_.front(), v_.back());

        int im = find_span_cubic(tm_, m_clamped);
        int kv = find_span_cubic(tv_, s_clamped);

        // Find or create group
        auto it = std::find_if(groups.begin(), groups.end(),
            [im, kv](const auto& g) { return g.im == im && g.kv == kv; });

        if (it == groups.end()) {
            groups.push_back({im, kv, {i}});
        } else {
            it->lanes.push_back(i);
        }
    }

    // Process each group
    for (const auto& group : groups) {
        // For first lane in group, get sigma value
        const double sigma_base = sigma[group.lanes[0]];

        // Compute 3 sigma basis functions
        double wv_down[4], wv_base[4], wv_up[4];
        cubic_basis_nonuniform(tv_, group.kv, sigma_base - vega_epsilon, wv_down);
        cubic_basis_nonuniform(tv_, group.kv, sigma_base, wv_base);
        cubic_basis_nonuniform(tv_, group.kv, sigma_base + vega_epsilon, wv_up);

        // Contract 3 times
        double G_down[4], G_base[4], G_up[4];
        contract_for_span(wt, wv_down, wr, group.im, jt, group.kv, lr, G_down);
        contract_for_span(wt, wv_base, wr, group.im, jt, group.kv, lr, G_base);
        contract_for_span(wt, wv_up, wr, group.im, jt, group.kv, lr, G_up);

        // Evaluate each lane in group
        for (size_t lane : group.lanes) {
            double wm[4];
            double m_clamped = clamp_query(moneyness[lane], m_.front(), m_.back());
            cubic_basis_nonuniform(tm_, group.im, m_clamped, wm);

            // 3 dot products
            double price_down = dot_product(wm, G_down);
            double price_base = dot_product(wm, G_base);
            double price_up = dot_product(wm, G_up);

            out_prices[lane] = price_base;
            out_vegas[lane] = (price_up - price_down) / (2.0 * vega_epsilon);
        }
    }
}
```

**Performance:**
- 3 contractions × U groups: 192 FMAs × U
- 3 dot products × n lanes: 12 FMAs × n
- For n=8, U=1: 192/8 + 12 = 36 FMAs/strike → **13× faster** (vs ~500 scalar)

---

### Component 4: IVSolverInterpolated::solve_batch() (IMPLEMENT IN PHASE 3)

**File:** `src/option/iv_solver_interpolated.hpp`

```cpp
class IVSolverInterpolated {
public:
    // EXISTING (unchanged)
    IVResult solve(const IVQuery& query) const;

    // NEW: Batch API
    /// Solve IV for multiple options (typically an option chain)
    ///
    /// Assumptions for best performance:
    /// - All queries share spot, maturity, rate (typical option chain)
    /// - Strikes sorted (reduces unique m-span count)
    ///
    /// Thread safety:
    /// - Caller must provide thread-local workspace
    /// - Multiple threads can call concurrently with separate workspaces
    ///
    /// @param queries Option chain queries
    /// @param workspace Thread-local workspace (caller-owned)
    /// @param config Solver configuration (defaults to constructor config)
    /// @return Results for each query (same order as input)
    [[nodiscard]]
    std::vector<IVResult> solve_batch(
        std::span<const IVQuery> queries,
        BatchIVWorkspace& workspace,
        const IVSolverConfig& config = {}) const;

private:
    const BSpline4D_FMA& price_surface_;
    double K_ref_;
    std::pair<double, double> m_range_, tau_range_, sigma_range_, r_range_;
    IVSolverConfig config_;
};
```

**Implementation (Simplified Outline):**

```cpp
std::vector<IVResult> IVSolverInterpolated::solve_batch(
    std::span<const IVQuery> queries,
    BatchIVWorkspace& workspace,
    const IVSolverConfig& config) const
{
    std::vector<IVResult> results(queries.size());

    // Phase 1: Per-lane validation and setup
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto& q = queries[i];

        // Validate (same as scalar path)
        auto error = validate_query(q);
        if (error.has_value()) {
            results[i] = IVResult{
                .converged = false,
                .failure_reason = *error
            };
            workspace.converged_flags()[i] = false;
            continue;
        }

        // CRITICAL: Use K_ref for moneyness (not strike!)
        workspace.moneyness()[i] = q.spot / K_ref_;
        workspace.market_prices()[i] = q.market_price;
        workspace.strikes()[i] = q.strike;

        // Adaptive bounds per lane
        auto [sig_min, sig_max] = adaptive_bounds(q);
        workspace.sigma_min_bounds()[i] = sig_min;
        workspace.sigma_max_bounds()[i] = sig_max;
        workspace.sigma()[i] = (sig_min + sig_max) / 2.0;  // Initial guess

        workspace.iterations()[i] = 0;
        workspace.converged_flags()[i] = false;
    }

    // Phase 2: Identify shared parameters
    // For now, assume all share spot, maturity, rate (typical chain)
    const double shared_spot = queries[0].spot;
    const double shared_maturity = queries[0].maturity;
    const double shared_rate = queries[0].rate;

    // Phase 3: SIMD Newton loop
    const size_t max_iter = config.max_iterations;

    for (size_t iter = 0; iter < max_iter; ++iter) {
        bool any_active = false;

        // Evaluate prices and vegas (batch with span grouping)
        price_surface_.eval_price_and_vega_batch(
            workspace.moneyness().subspan(0, queries.size()),
            shared_maturity,
            workspace.sigma().subspan(0, queries.size()),  // Per-lane!
            shared_rate,
            config.vega_epsilon,
            workspace.prices().subspan(0, queries.size()),
            workspace.vegas().subspan(0, queries.size()));

        // Update each lane
        for (size_t i = 0; i < queries.size(); ++i) {
            if (workspace.converged_flags()[i]) continue;

            // Scale price by strike (preserve K_ref contract)
            const double scaled_price = workspace.prices()[i] *
                                       (workspace.strikes()[i] / K_ref_);

            const double residual = scaled_price - workspace.market_prices()[i];
            workspace.residuals()[i] = std::abs(residual);

            // Check convergence
            if (workspace.residuals()[i] < config.tolerance) {
                workspace.converged_flags()[i] = true;
                workspace.iterations()[i] = iter + 1;
                continue;
            }

            // Check vega validity
            const double vega = workspace.vegas()[i];
            if (std::abs(vega) < 1e-10) {
                workspace.converged_flags()[i] = false;
                workspace.failure_reasons()[i] = "Vega too small";
                workspace.iterations()[i] = iter + 1;
                continue;
            }

            // Newton update
            const double sigma_new = workspace.sigma()[i] - residual / vega;
            workspace.sigma()[i] = std::clamp(sigma_new,
                                             workspace.sigma_min_bounds()[i],
                                             workspace.sigma_max_bounds()[i]);

            any_active = true;
        }

        if (!any_active) break;  // All converged or failed
    }

    // Phase 4: Extract results
    for (size_t i = 0; i < queries.size(); ++i) {
        if (workspace.converged_flags()[i]) {
            results[i] = IVResult{
                .converged = true,
                .iterations = workspace.iterations()[i],
                .implied_vol = workspace.sigma()[i],
                .final_error = workspace.residuals()[i],
                .vega = workspace.vegas()[i]
            };
        } else if (!workspace.failure_reasons()[i].empty()) {
            results[i] = IVResult{
                .converged = false,
                .iterations = workspace.iterations()[i],
                .implied_vol = workspace.sigma()[i],
                .final_error = workspace.residuals()[i],
                .failure_reason = workspace.failure_reasons()[i],
                .vega = workspace.vegas()[i]
            };
        } else {
            results[i] = IVResult{
                .converged = false,
                .iterations = max_iter,
                .implied_vol = workspace.sigma()[i],
                .final_error = workspace.residuals()[i],
                .failure_reason = "Maximum iterations reached"
            };
        }
    }

    return results;
}
```

---

## Performance Analysis

### Speedup Estimates

**Scalar (current):**
- Span finding: 6 cycles × 3 (vega) = 18 cycles
- Basis computation: 40 FLOPs × 3 = 120 FLOPs
- Tensor contraction: 256 FMAs × 3 = 768 FMAs
- **Total per Newton iteration: ~900 FLOPs/strike**

**SIMD Batch (W=8, U=1 unique span):**
- Span finding: 6 cycles × 1 (shared) = 6 cycles
- Basis computation: 40 FLOPs × 1 (shared) = 5 FLOPs/strike
- Tensor contraction: 192 FMAs × 1 (shared) = 24 FMAs/strike
- Dot products: 12 FMAs/strike
- **Total per Newton iteration: ~45 FLOPs/strike**

**Speedup: 900 / 45 ≈ 20× on FLOPs**

**Realistic (U=2):**
- Contraction: 192 × 2 / 8 = 48 FMAs/strike
- **Total: ~65 FLOPs/strike → 14× speedup**

**Conservative wall-clock: 8-10× (accounting for memory, overhead)**

### Break-Even Analysis

SIMD wins when: `64U/W + 12 < 256`

For W=8: `U < (256-12) × 8 / 64 = 30`

**Conclusion:** Even with 30 unique spans (highly unlikely for sorted strikes), SIMD is still faster!

---

## Testing Strategy

### Unit Tests

**Phase 1: BSpline eval_batch**
```cpp
TEST(BSpline4DTest, EvalBatchMatchesScalar) {
    // Generate random moneyness, tau, sigma, rate
    // Call eval() for each (scalar)
    // Call eval_batch() (SIMD)
    // Compare: EXPECT_NEAR(scalar, batch, 1e-14)
}

TEST(BSpline4DTest, EvalBatchMixedSpans) {
    // Moneyness values spanning 3 different knot intervals
    // Verify correct grouping and contraction
}
```

**Phase 2: Vega batch**
```cpp
TEST(BSpline4DTest, VegaBatchMatchesScalar) {
    // Compare batch vega to scalar finite differences
}
```

**Phase 3: solve_batch**
```cpp
TEST(IVSolverTest, SolveBatchMatchesSolve) {
    // Create 10 queries
    // Solve individually with solve()
    // Solve batch with solve_batch()
    // Compare results (converged, iterations, IV)
}

TEST(IVSolverTest, BatchHandlesFailures) {
    // Mix valid and invalid queries
    // Verify per-lane error handling
}
```

### Integration Tests

```cpp
TEST(IVIntegrationTest, OptionChainRealisticCase) {
    // SPY option chain: 20 strikes, sorted
    // Verify 8-10× speedup vs scalar
    // Verify all results match scalar path
}
```

---

## Implementation Roadmap

**PRIORITY UPDATED:** Based on lessons from PR 151 (closed batch SIMD PR), we MUST validate the OpenMP baseline before implementing SIMD.

**Lesson Learned:** PR 151 implemented 8,000 lines of batch SIMD code only to discover that simple OpenMP parallel-for was 2.1× faster due to cache locality issues. Thread-level parallelism beat SIMD for that workload.

---

### Phase -1: Profiling & OpenMP Baseline (REQUIRED FIRST - 3 days)

**Rationale:** Profile hot paths and validate simple solution before investing weeks in SIMD. Avoid repeating PR 151's mistake.

**Deliverables:**

1. **Profile IV Interpolation Hot Path** (~1 day):
   ```bash
   # Build profiling benchmark
   bazel build -c opt //benchmarks:iv_interpolation_profile

   # Run with perf for detailed profiling
   perf record -g ./bazel-bin/benchmarks/iv_interpolation_profile
   perf report

   # Cache analysis
   perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
     ./bazel-bin/benchmarks/iv_interpolation_profile
   ```

   **Metrics to capture:**
   - B-spline eval time (single call)
   - Vega FD time (3 calls)
   - Newton loop time (full IV solve)
   - Cache miss rates
   - Instructions per cycle (IPC)
   - Time breakdown: eval vs vega vs Newton bookkeeping

   **Span distribution profiling:**
   - Instrument `BSpline4D_FMA::eval()` to log span indices
   - Run on option chain benchmark (8, 16, 32, 64 strikes)
   - Histogram unique m-span counts
   - Validate assumption: U ≤ 2 for sorted strikes?

2. **Simple OpenMP Implementation** (~1 day):
   ```cpp
   // File: tests/iv_solver_openmp_baseline.cc
   std::vector<IVResult> solve_batch_openmp(
       const IVSolverInterpolated& solver,
       std::span<const IVQuery> queries,
       int num_threads)
   {
       std::vector<IVResult> results(queries.size());

       #pragma omp parallel num_threads(num_threads)
       {
           // Thread-local workspace (no contention)
           thread_local BatchIVWorkspace workspace(1);  // Size=1 for scalar

           #pragma omp for schedule(dynamic)
           for (size_t i = 0; i < queries.size(); ++i) {
               workspace.reset();
               results[i] = solver.solve(queries[i]);
           }
       }

       return results;
   }
   ```

3. **Benchmark OpenMP Scaling** (~1 day):
   - Test on synthetic option chains (8, 16, 32, 64 strikes)
   - Measure throughput at 1, 4, 8, 16 threads
   - Calculate scaling efficiency
   - Record cache miss rates (`perf stat`)
   - Compare against profiling results from step 1

**Success Criteria:**
- ✅ Profiling complete with hotspot analysis
- ✅ Span distribution measured on real workloads
- ✅ OpenMP implementation complete and tested
- ✅ Scaling efficiency measured (target: >85% at 16 threads)
- ✅ Cache behavior characterized (miss rates, IPC)

**Decision Gate:**
- ✅ **Proceed to Phase 0** if:
  - OpenMP throughput < target latency (still need SIMD)
  - Span distribution shows U ≤ 2 (SIMD grouping won't thrash)
  - Scaling efficiency < 90% (room for SIMD improvement)

- ❌ **Stop, use OpenMP** if:
  - OpenMP already meets latency target
  - Span distribution shows U >> 2 (SIMD will thrash cache)
  - Scaling efficiency > 90% (diminishing returns)

**Estimated Timeline:** 3 days (vs 8 weeks for full SIMD implementation)

**Files Created:**
- `benchmarks/iv_interpolation_profile.cc` - Profiling benchmark
- `benchmarks/BUILD.bazel` - Added `iv_interpolation_profile` target

**Risk Mitigation:** If OpenMP is sufficient, we save 8 weeks and avoid 1,000+ lines of complex SIMD code.

---

### Phase 0: PMR Workspace Infrastructure (AFTER Phase -1 validation - 1 week)

**Rationale:** Pure memory management change with minimal algorithm impact. Establishes foundation for later SIMD work.

**Tasks:**
- [ ] Create `src/option/batch_iv_workspace.hpp`
- [ ] Create `src/option/batch_iv_workspace.cpp`
- [ ] Implement class extending `WorkspaceBase`:
  - [ ] PMR-based allocation (64-byte aligned)
  - [ ] SoA layout for all arrays
  - [ ] Per-lane tracking infrastructure
  - [ ] Zero-cost `reset()` method
- [ ] Add BUILD.bazel entries
- [ ] Write unit tests:
  - [ ] Allocation sizes and alignment
  - [ ] 64-byte alignment verification (AVX-512)
  - [ ] `reset()` invalidates spans
  - [ ] Thread-safety (separate instances)
- [ ] Documentation and examples

**Success Criteria:**
- All tests pass
- Workspace allocates correctly with PMR
- Zero algorithm changes
- Ready for SIMD integration

**Deliverables:**
- `batch_iv_workspace.hpp` with complete API
- `batch_iv_workspace.cpp` with PMR allocation
- `batch_iv_workspace_test.cc` with full coverage
- Updated BUILD.bazel

---

### Phase 1: BSpline eval_batch (2 weeks)

**Rationale:** Core SIMD kernel for price evaluation with unique-span grouping.

**Tasks:**
- [ ] Implement unique-span grouping algorithm
- [ ] Implement `eval_batch()` with shared tau/sigma/rate
- [ ] Add `[[gnu::target_clones]]` for ISA selection
- [ ] Unit tests (scalar comparison, mixed spans)
- [ ] Benchmark (validate 20× speedup on contractions)

**Success Criteria:**
- Matches scalar path within FP tolerance (1e-14)
- 15-20× speedup on contraction (U=1 case)
- Handles arbitrary span distributions correctly

---

### Phase 2: BSpline vega_batch (2 weeks)

**Rationale:** Finite difference vega with per-lane sigma support.

**Tasks:**
- [ ] Implement `eval_price_and_vega_batch()`
- [ ] Handle per-lane sigma (Newton iteration support)
- [ ] Implement (m_span, σ_span) grouping
- [ ] Unit tests (vega accuracy vs scalar FD)
- [ ] Benchmark (validate 10-14× speedup)

**Success Criteria:**
- Vega matches scalar finite differences
- Handles per-lane sigma correctly
- 10-14× speedup achieved

---

### Phase 3: solve_batch Integration (2 weeks)

**Rationale:** Full Newton loop with per-lane validation and tracking.

**Tasks:**
- [ ] Implement per-lane validation (adaptive bounds, query checks)
- [ ] Implement masked Newton loop (early termination per lane)
- [ ] Preserve K_ref semantics (moneyness = spot/K_ref)
- [ ] Integration tests (chain scenarios, failure handling)
- [ ] End-to-end benchmarks

**Success Criteria:**
- Bit-identical to scalar path (within FP tolerance)
- 8-10× wall-clock speedup on realistic chains
- All validation/error handling preserved

---

### Phase 4: Optimization & Validation (1 week)

**Rationale:** Production readiness and performance tuning.

**Tasks:**
- [ ] Profile realistic option chains (SPY, SPX)
- [ ] Add scalar fallback for small batches (n < 4)
- [ ] Optimize span grouping algorithm
- [ ] Documentation (usage examples, performance characteristics)
- [ ] Code review and cleanup

**Success Criteria:**
- Production-ready code quality
- Comprehensive documentation
- Performance validated on real-world data

**Total: ~8 weeks** (with Phase 0 separated and prioritized)

---

## Lessons Learned from PR 151

### Background: Batch PDE Vectorization Failure

**PR 151** implemented horizontal SIMD vectorization for batch solving of multiple PDE contracts simultaneously. After 8,000+ lines of code and 50 passing tests, it was **closed without merge** because simple OpenMP parallel-for was **2.1× faster**.

### Why PR 151 Failed

1. **Cache Locality Problem (70-80% of overhead)**
   - AoS layout with stride=8: `u[(i-1)*8], u[i*8], u[(i+1)*8]`
   - Required 3 cache lines per stencil (192 bytes) vs 1 for sequential (64 bytes)
   - 3× more cache traffic → 1.78× slower per contract
   - Strided access confused hardware prefetcher

2. **Poor Scaling Efficiency**
   - OpenMP batch mode: 44% scaling efficiency at 16 threads
   - OpenMP single-contract: 93% scaling efficiency at 16 threads
   - Cache contention and false sharing destroyed parallelism

3. **SIMD Efficiency Only 7% of Peak**
   - Theoretical: 8× speedup (AVX-512 width)
   - Actual: 0.56× (44% slower than sequential)
   - Cache overhead completely negated SIMD benefits

4. **The Simple Winner**
   - 10 lines: `#pragma omp parallel for` over single-contract solves
   - 14.8× speedup with 16 threads
   - 2.1× faster than 8,000-line batch SIMD implementation
   - 93% scaling efficiency

### Key Metrics from PR 151

| Approach | Threads | Time | Throughput | Speedup | Efficiency |
|----------|---------|------|------------|---------|------------|
| Batch SIMD (single) | 1 | 75.3 ms | 213 c/s | 1.0× | - |
| OpenMP single | 16 | 5.07 ms | **3,155 c/s** | **14.8×** | **93%** |
| OpenMP batch SIMD | 16 | 10.7 ms | 1,489 c/s | 7.0× | 44% |

**Winner:** OpenMP single-contract (2.1× faster than batch SIMD)

### Critical Lessons for Batch IV SIMD

1. **Measure simple solution first**
   - OpenMP parallel-for is trivial to implement
   - If it meets performance targets, stop
   - Don't assume SIMD will be faster

2. **Cache locality dominates**
   - 3× cache traffic destroyed 8× theoretical SIMD gain
   - Sequential access >> SIMD with strided access
   - Profile cache misses before implementing

3. **Thread parallelism often beats SIMD**
   - 93% scaling efficiency at 16 threads is hard to beat
   - SIMD must beat this bar, not just beat sequential

4. **Complexity cost is real**
   - 8,000 lines of SIMD code
   - 50+ specialized tests
   - High maintenance burden
   - Must deliver clear performance win to justify

### How This Design Differs from PR 151

**Advantages:**

1. **SoA layout** - Not AoS like PR 151
   - `sigma_[0..N]` is contiguous
   - No stride-8 access pattern
   - Better cache locality

2. **No stencil operations**
   - B-spline tensor products are compute-heavy
   - Not memory-bound like PDE stencils
   - All lanes share same 256-coefficient tile

3. **Parameter sharing**
   - Option chains share (spot, maturity, rate)
   - Only strikes differ → minimal divergence
   - Better than heterogeneous PDE contracts

4. **Explicit divergence handling**
   - Unique-span grouping designed from start
   - Per-lane sigma explicit in API
   - Not hidden inside SIMD lanes

**Risks Still Present:**

1. **Span distribution unknown**
   - Design assumes U ≤ 2 (sorted strikes)
   - No actual data validates this
   - If U >> 2, cache thrashing like PR 151

2. **OpenMP might be sufficient**
   - IV solving is already fast (B-spline interpolation)
   - Thread parallelism might hit latency target
   - Won't know until we measure

3. **Complexity still high**
   - ~1,000+ lines for full SIMD implementation
   - Unique-span grouping, per-lane tracking, etc.
   - Must prove benefit > cost

### Validation Strategy (Phase -1)

To avoid repeating PR 151's mistake:

1. ✅ Implement OpenMP baseline (10 lines)
2. ✅ Benchmark on real option chains
3. ✅ Profile span distribution
4. ✅ Measure cache behavior
5. ❌ Only proceed with SIMD if OpenMP insufficient

**This validation is mandatory before any SIMD implementation.**

---

## Vertical SIMD: Micro-Optimizations Within Single Solve

### Phase 0.5 Status: ✅ COMPLETE (2025-01-13)

**Implementation complete. Key finding: Scalar triple is better than SIMD triple.**

#### Summary of Results

| Method | Time (ns) | Speedup | Status |
|--------|-----------|---------|--------|
| Vega FD (baseline) | 515 ns | 1.0× | Original implementation |
| **Scalar triple** | **271 ns** | **1.90×** | ✅ **Integrated into IV solver** |
| SIMD triple | 608 ns | 0.45× | ❌ **Regression - not used** |

**Key findings:**
- **Scalar triple**: Excellent 1.90× speedup by sharing coefficient loads
- **SIMD triple**: 18% slower than FD baseline (narrow SIMD overhead exceeds benefits)
- **Decision**: Use scalar triple in production, keep SIMD variant for research only
- **Integration**: Task 5 updated to use scalar triple instead of SIMD triple

**Lesson learned:** Small-width SIMD (3 lanes) is counterproductive on modern CPUs. Compiler auto-vectorization of scalar code performs better than manual `std::experimental::simd` for narrow operations.

**Impact on horizontal SIMD plans:**
- Phase 1-3 horizontal SIMD must target 8+ lanes (not 3)
- OpenMP baseline validation (Phase -1) now more critical
- Scalar triple sets new baseline: must beat 1.90× improvement

For detailed results, see `docs/profiling-results-2025-01-13.md`.

---

### Can We Use SIMD for a Single Operation?

**Question:** Instead of horizontal SIMD (across 8 options), can we use vertical SIMD (within 1 option's loops)?

**Answer (Updated):** Yes, but scalar approaches often outperform explicit SIMD for narrow operations.

### Vertical SIMD Opportunities

#### 1. **Vega Triple Evaluation** ✅ IMPLEMENTED

**Status:** Complete. Scalar version integrated (1.90× speedup). SIMD version regressed (0.45× slower).

**Problem:** Current code evaluates B-spline 3 times for vega:
```cpp
double price_down = eval(sigma - epsilon);  // 256 FMAs
double price = eval(sigma);                 // 256 FMAs
double price_up = eval(sigma + epsilon);    // 256 FMAs
vega = (price_up - price_down) / (2 * epsilon);
```

**Optimization:** Evaluate all 3 sigma values in one pass (share coefficient loads):
```cpp
using simd3 = std::experimental::fixed_size_simd<double, 4>;  // Use 3 lanes

// Compute basis functions for all 3 sigma values
double wv[3][4];
for (int i = 0; i < 3; ++i) {
    cubic_basis_nonuniform(tv_, kv, sigma_values[i], wv[i]);
}

simd3 accum{0.0, 0.0, 0.0, 0.0};

for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
            // Pack 3 sigma weights into SIMD
            simd3 wv_packed{wv[0][c], wv[1][c], wv[2][c], 0.0};
            double wtab = wm[a] * wt[b];

            for (int d = 0; d < 4; ++d) {
                int idx = coefficient_index(im-a, jt-b, kv-c, lr-d);
                simd3 contrib = coefficients[idx] * wtab * wv_packed * wr[d];
                accum += contrib;
            }
        }
    }
}

auto [price_down, price, price_up, _] = accum;
vega = (price_up - price_down) / (2 * epsilon);
```

**Expected Benefits:**
- Load 256 coefficients **once** instead of 3 times
- ~2.5× faster for vega computation
- Excellent cache behavior (single pass)
- **Avoids PR 151's cache thrashing issue**

**Actual Results (2025-01-13):**

Two implementations tested:

**Scalar Triple (implemented without explicit SIMD):**
```cpp
// BSpline4D_FMA::eval_price_and_vega_triple()
// Single pass, sequential evaluation of (σ-ε, σ, σ+ε)
// Uses std::fma() with #pragma omp simd for auto-vectorization

double price_down = 0.0, price_base = 0.0, price_up = 0.0;
for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
            const double w_down = wm[a] * wt[b] * wv_down[c];
            const double w_base = wm[a] * wt[b] * wv_base[c];
            const double w_up = wm[a] * wt[b] * wv_up[c];
            for (int d = 0; d < 4; ++d) {
                const double coeff = c_[coefficient_index(...)];
                const double w_r = wr[d];
                price_down = std::fma(coeff, w_down * w_r, price_down);
                price_base = std::fma(coeff, w_base * w_r, price_base);
                price_up = std::fma(coeff, w_up * w_r, price_up);
            }
        }
    }
}
```
- **Result:** 271ns (1.90× speedup over 515ns FD baseline)
- **Status:** ✅ Integrated into IV solver (Task 5)
- **Why it works:** Compiler auto-vectorizes loops, zero packing overhead

**SIMD Triple (std::experimental::simd):**
```cpp
// BSpline4D_FMA::eval_price_and_vega_triple_simd()
// Uses fixed_size_simd<double,4> with 3 active lanes
simd_t accum{0.0, 0.0, 0.0, 0.0};
for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
            const simd_t wv_packed{wv_down[c], wv_base[c], wv_up[c], 0.0};
            const simd_t weight_mts = simd_t(wm[a] * wt[b]) * wv_packed;
            for (int d = 0; d < 4; ++d) {
                accum = stdx::fma(simd_t(coeff * w_r), weight_mts, accum);
            }
        }
    }
}
accum.copy_to(results, stdx::element_aligned);
```
- **Result:** 608ns (0.45× "speedup" = 18% SLOWER than FD!)
- **Status:** ❌ Not integrated (regression)
- **Why it failed:** 3-lane SIMD overhead (packing, broadcasts, copy_to) exceeds gains

**Conclusion:**
- Use scalar triple in production (already integrated)
- SIMD triple retained for benchmarking/research only
- Manual SIMD counterproductive for narrow (≤4 lane) operations

#### 2. **B-Spline Innermost Loop (Minor: ~1.2-1.3× speedup)**

**Current code:**
```cpp
// Innermost d-loop: 4 iterations
for (int d = 0; d < 4; ++d) {
    int idx = coefficient_index(im-a, jt-b, kv-c, lr-d);
    sum = std::fma(coefficients[idx], wm[a]*wt[b]*wv[c]*wr[d], sum);
}
```

**Optimization:** Vectorize the 4-element dot product:
```cpp
using simd4 = std::experimental::fixed_size_simd<double, 4>;

double wtabc = wm[a] * wt[b] * wv[c];

// Load 4 contiguous coefficients and weights
const double* coeff_ptr = &coefficients[base_idx];
simd4 coeff_vec = simd4::load(coeff_ptr);
simd4 wr_vec = simd4::load(wr);

// Single vector FMA instead of 4 scalar FMAs
simd4 prod = std::experimental::fma(coeff_vec, wr_vec * simd4(wtabc), simd4{0.0});

// Horizontal reduction
sum += std::experimental::reduce(prod);
```

**Benefits:**
- Reduces instruction count slightly
- ~1.2-1.3× speedup for B-spline evaluation
- Minimal complexity

#### 3. **Newton Loop: NOT Viable**

The Newton iteration is inherently sequential:
```cpp
for (int iter = 0; iter < max_iter; ++iter) {
    double price = eval(sigma);         // Depends on sigma
    double vega = compute_vega(sigma);  // Depends on sigma
    sigma = sigma - (price - target) / vega;  // Update for next iteration
    if (converged) break;               // Early exit
}
```

**No vertical SIMD opportunity** - each iteration depends on the previous.

### Vertical vs Horizontal SIMD Comparison

| Aspect | Vertical SIMD | Horizontal SIMD |
|--------|---------------|-----------------|
| **Target** | Single option's loops | 8 options in parallel |
| **Speedup** | 1.2-2.5× (localized) | 5-8× (end-to-end) |
| **Complexity** | Low (micro-optimization) | High (1,000+ lines) |
| **Cache behavior** | Excellent (sequential) | Risk (span divergence) |
| **Divergence risk** | None (same data) | High (different spans) |
| **Implementation** | ~50 lines | ~1,000 lines |

### Hybrid Recommendation

**Best approach:** Combine both vertical and horizontal SIMD:

1. **Vertical micro-optimizations** (Phase 0.5):
   - Implement vega triple evaluation (~2.5× vega speedup)
   - Optionally: vectorize innermost d-loop (~1.2× price speedup)
   - Total effort: ~2 days
   - Works with both scalar and batch modes

2. **Horizontal batching** (Phase 1-3):
   - Only if OpenMP baseline insufficient
   - Benefits from vertical optimizations (each lane faster)
   - Reduced per-option runtime helps with divergence

3. **OpenMP parallelism** (Phase -1):
   - Thread-level parallelism across options
   - Combined with vertical optimizations for best of both worlds

### Implementation Priority (Updated)

1. **Phase -1**: OpenMP baseline (validate need)
2. **Phase 0**: PMR workspace (shared infrastructure)
3. **Phase 0.5** ✅ **COMPLETE**: Vertical SIMD micro-optimizations
   - ✅ Vega triple evaluation (scalar AND SIMD variants)
   - ❌ Innermost loop vectorization (deferred - scalar triple sufficient)
   - **Benefits both scalar and batch modes**
   - **Result: Use scalar triple (1.90× speedup), NOT SIMD (regression)**
4. **Phase 1-3**: Horizontal SIMD (only if validated)

**Rationale:** Vertical optimizations are low-risk, low-effort wins that improve both scalar and batch paths. Implement these first, then re-evaluate if horizontal SIMD is still needed.

---

## Open Questions

1. **Span grouping data structure**: Use `std::vector` or preallocate fixed-size array?
2. **SIMD in grouping**: Can we vectorize the span partitioning itself?
3. **Adaptive batching**: Should we dynamically choose batch size based on U?
4. **Strike sorting**: Require caller to sort, or sort internally?
5. **Vertical first?**: Should we implement vega triple optimization before horizontal SIMD?

---

## References

- Existing implementation: `src/option/iv_solver_interpolated.cpp`
- B-spline evaluator: `src/interpolation/bspline_4d.hpp`
- Workspace patterns: `src/pde/memory/workspace_base.hpp`
- SIMD stencils: `src/pde/operators/centered_difference_simd_backend.hpp`
- Codex review: (provided separately)

---

## Summary

This corrected design addresses all critical issues identified in review:

1. ✅ **Span divergence**: Unique-span grouping preserves correctness
2. ✅ **Per-lane sigma**: API accepts `std::span<const double>`
3. ✅ **Moneyness semantics**: Preserves `spot/K_ref` contract
4. ✅ **Thread safety**: Workspace as parameter, no mutable state
5. ✅ **Validation parity**: Replicates all scalar checks per lane

**Expected outcome:** 8-10× speedup for realistic option chains while maintaining bit-identical correctness to scalar path.
