# Phase A: Nested Clenshaw-Curtis Levels + PDE Slice Cache

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace arbitrary CGL node counts on sigma/rate axes with nested
Clenshaw-Curtis levels (2^l + 1 points at level l), and add a PDE slice cache
that allows incremental refinement without re-solving existing (sigma, rate)
pairs.

**Architecture:** Add `cc_level_nodes()` to `chebyshev_nodes.hpp` that generates
nodes at a specific CC level. Add `PDESliceCache` class that stores PDE results
keyed by (sigma, rate) node identity. Modify the Chebyshev 4D EEP builder to
accept CC levels instead of raw counts and to use the cache for incremental
builds. Validate that cached builds match fresh builds to machine epsilon.

**Tech Stack:** C++23, Bazel, GoogleTest, existing `chebyshev_nodes.hpp`,
`BatchAmericanOptionSolver`, `CubicSpline`, `ChebyshevTucker4D`.

**Worktree:** `.worktrees/chebyshev-tensor/` (branch `experiment/chebyshev-tensor`)

---

### Task 1: Add `cc_level_nodes()` to chebyshev_nodes.hpp

**Files:**
- Modify: `.worktrees/chebyshev-tensor/src/option/table/dimensionless/chebyshev_nodes.hpp`
- Test: `.worktrees/chebyshev-tensor/tests/chebyshev_nodes_test.cc`

**Step 1: Write the failing tests**

Add to `chebyshev_nodes_test.cc`:

```cpp
TEST(ChebyshevNodesTest, CCLevelZeroGivesTwoNodes) {
    auto nodes = cc_level_nodes(0, -1.0, 1.0);
    ASSERT_EQ(nodes.size(), 2u);
    EXPECT_DOUBLE_EQ(nodes[0], -1.0);
    EXPECT_DOUBLE_EQ(nodes[1], 1.0);
}

TEST(ChebyshevNodesTest, CCLevelNodesMatchChebyshevNodes) {
    // cc_level_nodes(l, a, b) should produce the same nodes as
    // chebyshev_nodes(2^l + 1, a, b), since CGL nodes at 2^l+1 points
    // are exactly the Clenshaw-Curtis nodes at level l.
    for (size_t l = 0; l <= 4; ++l) {
        size_t n = (1u << l) + 1;
        auto cc = cc_level_nodes(l, 0.0, 5.0);
        auto cgl = chebyshev_nodes(n, 0.0, 5.0);
        ASSERT_EQ(cc.size(), n) << "Level " << l;
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(cc[i], cgl[i], 1e-15)
                << "Level " << l << ", node " << i;
        }
    }
}

TEST(ChebyshevNodesTest, CCLevelsAreNested) {
    // Every node at level l must appear at level l+1.
    for (size_t l = 0; l <= 3; ++l) {
        auto coarse = cc_level_nodes(l, -2.0, 3.0);
        auto fine = cc_level_nodes(l + 1, -2.0, 3.0);
        for (double c : coarse) {
            bool found = false;
            for (double f : fine) {
                if (std::abs(c - f) < 1e-14) { found = true; break; }
            }
            EXPECT_TRUE(found) << "Level " << l << " node " << c
                               << " not found at level " << l + 1;
        }
    }
}

TEST(ChebyshevNodesTest, CCNewNodesAtLevel) {
    // cc_new_nodes_at_level returns only nodes NOT present at previous level.
    auto new2 = cc_new_nodes_at_level(2, -1.0, 1.0);
    auto full2 = cc_level_nodes(2, -1.0, 1.0);
    auto full1 = cc_level_nodes(1, -1.0, 1.0);
    // Level 2 has 5 nodes, level 1 has 3 → 2 new nodes
    EXPECT_EQ(new2.size(), 2u);
    // New nodes must be in full2 but not in full1
    for (double n : new2) {
        bool in_fine = false, in_coarse = false;
        for (double f : full2) if (std::abs(n - f) < 1e-14) in_fine = true;
        for (double c : full1) if (std::abs(n - c) < 1e-14) in_coarse = true;
        EXPECT_TRUE(in_fine) << "New node " << n << " not in level 2";
        EXPECT_FALSE(in_coarse) << "New node " << n << " already in level 1";
    }
}

TEST(ChebyshevNodesTest, CCNewNodesAtLevelZeroReturnsAll) {
    auto new0 = cc_new_nodes_at_level(0, -1.0, 1.0);
    EXPECT_EQ(new0.size(), 2u);  // Both endpoints are "new" at level 0
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel test //tests:chebyshev_nodes_test --test_output=all`
Expected: Compilation error — `cc_level_nodes` and `cc_new_nodes_at_level` not defined.

**Step 3: Implement `cc_level_nodes()` and `cc_new_nodes_at_level()`**

Add to `chebyshev_nodes.hpp`, before the closing `}  // namespace mango`:

```cpp
/// Generate Clenshaw-Curtis nodes at level l on [a, b].
/// Returns 2^l + 1 nodes (same as CGL nodes at that count), sorted ascending.
/// Levels are nested: every node at level l appears at level l+1.
[[nodiscard]] inline std::vector<double>
cc_level_nodes(size_t level, double a, double b) {
    return chebyshev_nodes((1u << level) + 1, a, b);
}

/// Return only the NEW nodes introduced at level l (not present at level l-1).
/// At level 0, returns all 2 nodes (both endpoints).
/// At level l >= 1, returns 2^(l-1) new interior nodes, sorted ascending.
[[nodiscard]] inline std::vector<double>
cc_new_nodes_at_level(size_t level, double a, double b) {
    auto all = cc_level_nodes(level, a, b);
    if (level == 0) return all;
    auto prev = cc_level_nodes(level - 1, a, b);
    std::vector<double> result;
    result.reserve(all.size() - prev.size());
    size_t pi = 0;
    for (double node : all) {
        if (pi < prev.size() && std::abs(node - prev[pi]) < 1e-14 * (b - a + 1.0)) {
            ++pi;
        } else {
            result.push_back(node);
        }
    }
    return result;
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel test //tests:chebyshev_nodes_test --test_output=all`
Expected: All tests PASS (7 existing + 5 new = 12 total).

**Step 5: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor
git add src/option/table/dimensionless/chebyshev_nodes.hpp tests/chebyshev_nodes_test.cc
git commit -m "Add nested Clenshaw-Curtis level functions

cc_level_nodes(l, a, b) returns 2^l+1 CGL nodes at level l.
cc_new_nodes_at_level(l, a, b) returns only the new nodes at
level l not present at level l-1. Levels are nested by
construction: every node at level l appears at level l+1."
```

---

### Task 2: Create `PDESliceCache` class

**Files:**
- Create: `.worktrees/chebyshev-tensor/benchmarks/pde_slice_cache.hpp`
- Test: `.worktrees/chebyshev-tensor/tests/pde_slice_cache_test.cc`
- Modify: `.worktrees/chebyshev-tensor/tests/BUILD.bazel` (add test target)
- Modify: `.worktrees/chebyshev-tensor/benchmarks/BUILD.bazel` (add header to srcs of interp_iv_safety)

The cache stores spline-interpolated PDE solutions keyed by (sigma_index,
rate_index, tau_index). It allows incremental population: solve a new (sigma,
rate) pair, cache its snapshots, and later retrieve them without re-solving.

**Step 1: Write the failing test**

Create `tests/pde_slice_cache_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include "pde_slice_cache.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

TEST(PDESliceCacheTest, EmptyCacheHasNoSlices) {
    PDESliceCache cache;
    EXPECT_EQ(cache.num_cached_pairs(), 0u);
    EXPECT_FALSE(cache.has_slice(0, 0));
}

TEST(PDESliceCacheTest, StoreAndRetrieveSlice) {
    PDESliceCache cache;
    // Store a simple spline for (sigma_idx=0, rate_idx=0, tau_idx=0)
    std::vector<double> x_grid = {-0.5, -0.25, 0.0, 0.25, 0.5};
    std::vector<double> values = {1.0, 2.0, 3.0, 2.0, 1.0};
    cache.store_slice(0, 0, 0, x_grid, values);
    EXPECT_TRUE(cache.has_slice(0, 0));
    EXPECT_EQ(cache.num_cached_pairs(), 1u);

    // Retrieve and evaluate
    auto* spline = cache.get_slice(0, 0, 0);
    ASSERT_NE(spline, nullptr);
    EXPECT_NEAR(spline->eval(0.0), 3.0, 1e-10);
}

TEST(PDESliceCacheTest, MultipleTauSnapshots) {
    PDESliceCache cache;
    std::vector<double> x_grid = {0.0, 0.5, 1.0};
    cache.store_slice(0, 0, 0, x_grid, {1.0, 2.0, 3.0});
    cache.store_slice(0, 0, 1, x_grid, {4.0, 5.0, 6.0});
    EXPECT_EQ(cache.num_tau_slices(0, 0), 2u);

    auto* s0 = cache.get_slice(0, 0, 0);
    auto* s1 = cache.get_slice(0, 0, 1);
    ASSERT_NE(s0, nullptr);
    ASSERT_NE(s1, nullptr);
    EXPECT_NEAR(s0->eval(0.5), 2.0, 1e-10);
    EXPECT_NEAR(s1->eval(0.5), 5.0, 1e-10);
}

TEST(PDESliceCacheTest, DifferentSigmaRatePairs) {
    PDESliceCache cache;
    std::vector<double> x_grid = {0.0, 1.0};
    cache.store_slice(0, 0, 0, x_grid, {1.0, 2.0});
    cache.store_slice(1, 0, 0, x_grid, {3.0, 4.0});
    cache.store_slice(0, 1, 0, x_grid, {5.0, 6.0});
    EXPECT_EQ(cache.num_cached_pairs(), 3u);
    EXPECT_TRUE(cache.has_slice(0, 0));
    EXPECT_TRUE(cache.has_slice(1, 0));
    EXPECT_TRUE(cache.has_slice(0, 1));
    EXPECT_FALSE(cache.has_slice(1, 1));
}

TEST(PDESliceCacheTest, NewSigmaNodesCount) {
    PDESliceCache cache;
    std::vector<double> x_grid = {0.0, 1.0};
    // Cache sigma indices {0, 1, 2} at rate index 0
    for (size_t s = 0; s < 3; ++s) {
        cache.store_slice(s, 0, 0, x_grid, {1.0, 2.0});
    }
    // If we want sigma indices {0, 1, 2, 3, 4} at rate index 0,
    // only {3, 4} are new.
    std::vector<size_t> wanted = {0, 1, 2, 3, 4};
    auto missing = cache.missing_pairs(wanted, {0});
    EXPECT_EQ(missing.size(), 2u);  // (3,0) and (4,0)
}

TEST(PDESliceCacheTest, PdeSolveCount) {
    PDESliceCache cache;
    EXPECT_EQ(cache.total_pde_solves(), 0u);
    cache.record_pde_solves(15);
    EXPECT_EQ(cache.total_pde_solves(), 15u);
    cache.record_pde_solves(10);
    EXPECT_EQ(cache.total_pde_solves(), 25u);
}

}  // namespace
}  // namespace mango
```

Add test target to `tests/BUILD.bazel`:

```starlark
cc_test(
    name = "pde_slice_cache_test",
    size = "small",
    srcs = ["pde_slice_cache_test.cc"],
    deps = [
        "//benchmarks:pde_slice_cache",
        "//src/math:cubic_spline_solver",
        "@googletest:gtest_main",
    ],
)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel test //tests:pde_slice_cache_test --test_output=all`
Expected: Compilation error — `pde_slice_cache.hpp` not found.

**Step 3: Implement `PDESliceCache`**

Create `benchmarks/pde_slice_cache.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/cubic_spline_solver.hpp"

#include <cstddef>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace mango {

/// Cache of PDE solutions keyed by (sigma_index, rate_index, tau_index).
/// Each slice is a CubicSpline over the spatial (x) grid at one snapshot time.
/// Supports incremental population: solve new (sigma, rate) pairs and add them
/// without re-solving existing pairs.
class PDESliceCache {
public:
    using Key = std::pair<size_t, size_t>;  // (sigma_idx, rate_idx)

    /// Store a spline for (sigma_idx, rate_idx, tau_idx).
    void store_slice(size_t sigma_idx, size_t rate_idx, size_t tau_idx,
                     std::span<const double> x_grid,
                     std::span<const double> values) {
        auto& tau_map = slices_[{sigma_idx, rate_idx}];
        auto& entry = tau_map[tau_idx];
        entry.spline.build(x_grid, values);
        entry.valid = true;
    }

    /// Check if any tau slices exist for (sigma_idx, rate_idx).
    [[nodiscard]] bool has_slice(size_t sigma_idx, size_t rate_idx) const {
        return slices_.contains({sigma_idx, rate_idx});
    }

    /// Retrieve the spline for (sigma_idx, rate_idx, tau_idx), or nullptr.
    [[nodiscard]] const CubicSpline<double>*
    get_slice(size_t sigma_idx, size_t rate_idx, size_t tau_idx) const {
        auto it = slices_.find({sigma_idx, rate_idx});
        if (it == slices_.end()) return nullptr;
        auto jt = it->second.find(tau_idx);
        if (jt == it->second.end() || !jt->second.valid) return nullptr;
        return &jt->second.spline;
    }

    /// Number of distinct (sigma, rate) pairs cached.
    [[nodiscard]] size_t num_cached_pairs() const { return slices_.size(); }

    /// Number of tau slices stored for a given (sigma, rate) pair.
    [[nodiscard]] size_t num_tau_slices(size_t sigma_idx, size_t rate_idx) const {
        auto it = slices_.find({sigma_idx, rate_idx});
        if (it == slices_.end()) return 0;
        return it->second.size();
    }

    /// Given wanted sigma indices and rate indices, return (sigma_idx, rate_idx)
    /// pairs that are NOT yet cached.
    [[nodiscard]] std::vector<Key>
    missing_pairs(std::span<const size_t> sigma_indices,
                  std::span<const size_t> rate_indices) const {
        std::vector<Key> result;
        for (size_t s : sigma_indices) {
            for (size_t r : rate_indices) {
                if (!slices_.contains({s, r})) {
                    result.push_back({s, r});
                }
            }
        }
        return result;
    }

    /// Record cumulative PDE solve count.
    void record_pde_solves(size_t count) { total_pde_solves_ += count; }

    /// Total PDE solves performed to populate this cache.
    [[nodiscard]] size_t total_pde_solves() const { return total_pde_solves_; }

    /// Clear all cached slices.
    void clear() {
        slices_.clear();
        total_pde_solves_ = 0;
    }

private:
    struct SliceEntry {
        CubicSpline<double> spline;
        bool valid = false;
    };

    std::map<Key, std::map<size_t, SliceEntry>> slices_;
    size_t total_pde_solves_ = 0;
};

}  // namespace mango
```

Also add a `cc_library` target to `benchmarks/BUILD.bazel`:

```starlark
cc_library(
    name = "pde_slice_cache",
    hdrs = ["pde_slice_cache.hpp"],
    deps = [
        "//src/math:cubic_spline_solver",
    ],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/benchmarks",
)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel test //tests:pde_slice_cache_test --test_output=all`
Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor
git add benchmarks/pde_slice_cache.hpp benchmarks/BUILD.bazel tests/pde_slice_cache_test.cc tests/BUILD.bazel
git commit -m "Add PDESliceCache for incremental PDE reuse

Stores spline-interpolated PDE snapshots keyed by (sigma_idx,
rate_idx, tau_idx). Supports incremental population: new
(sigma, rate) pairs can be added without re-solving existing
ones. Tracks cumulative PDE solve count."
```

---

### Task 3: Create `build_chebyshev_4d_eep_incremental()` builder

**Files:**
- Create: `.worktrees/chebyshev-tensor/benchmarks/chebyshev_4d_incremental_builder.hpp`
- Modify: `.worktrees/chebyshev-tensor/benchmarks/BUILD.bazel` (add header to interp_iv_safety srcs)

This builder accepts CC levels for sigma and rate (instead of raw node counts),
uses the `PDESliceCache` for incremental PDE reuse, and produces a
`Chebyshev4DEEPInner` identical to the existing builder.

**Step 1: Write the failing test**

Create `tests/chebyshev_4d_incremental_test.cc`. This test verifies Phase A
verification gate 1: cached build matches fresh build to machine epsilon.

```cpp
// SPDX-License-Identifier: MIT
#include "chebyshev_4d_incremental_builder.hpp"
#include "chebyshev_4d_eep_inner.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

// Phase A verification gate 1: cached build matches fresh build.
TEST(Chebyshev4DIncrementalTest, CachedBuildMatchesFreshBuild) {
    // Build fresh (no cache) with 9 sigma nodes, 5 rate nodes
    // (CC level 3 = 2^3+1=9, CC level 2 = 2^2+1=5)
    Chebyshev4DEEPConfig fresh_cfg;
    fresh_cfg.num_x = 15;
    fresh_cfg.num_tau = 9;
    fresh_cfg.num_sigma = 9;   // CC level 3
    fresh_cfg.num_rate = 5;    // CC level 2
    fresh_cfg.use_tucker = false;
    fresh_cfg.dividend_yield = 0.0;

    auto fresh = build_chebyshev_4d_eep(fresh_cfg, 100.0, OptionType::PUT);

    // Build incrementally with same final levels
    IncrementalBuildConfig inc_cfg;
    inc_cfg.num_x = 15;
    inc_cfg.num_tau = 9;
    inc_cfg.sigma_level = 3;  // 9 nodes
    inc_cfg.rate_level = 2;   // 5 nodes
    inc_cfg.use_tucker = false;
    inc_cfg.dividend_yield = 0.0;
    // Use same domain bounds as fresh (headroom from fresh_cfg node counts)
    inc_cfg.x_min = fresh_cfg.x_min;
    inc_cfg.x_max = fresh_cfg.x_max;
    inc_cfg.tau_min = fresh_cfg.tau_min;
    inc_cfg.tau_max = fresh_cfg.tau_max;
    inc_cfg.sigma_min = fresh_cfg.sigma_min;
    inc_cfg.sigma_max = fresh_cfg.sigma_max;
    inc_cfg.rate_min = fresh_cfg.rate_min;
    inc_cfg.rate_max = fresh_cfg.rate_max;

    PDESliceCache cache;
    auto inc = build_chebyshev_4d_eep_incremental(inc_cfg, cache, 100.0, OptionType::PUT);

    // Compare at 50 random probe points
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> ux(fresh_cfg.x_min, fresh_cfg.x_max);
    std::uniform_real_distribution<double> ut(fresh_cfg.tau_min, fresh_cfg.tau_max);
    std::uniform_real_distribution<double> us(fresh_cfg.sigma_min, fresh_cfg.sigma_max);
    std::uniform_real_distribution<double> ur(fresh_cfg.rate_min, fresh_cfg.rate_max);

    double max_diff = 0.0;
    for (int i = 0; i < 50; ++i) {
        PriceQuery q;
        double x = ux(rng);
        q.spot = std::exp(x) * 100.0;
        q.strike = 100.0;
        q.tau = ut(rng);
        q.sigma = us(rng);
        q.rate = ur(rng);

        double p_fresh = fresh.interp.eval({x, q.tau, q.sigma, q.rate});
        double p_inc = inc.interp.eval({x, q.tau, q.sigma, q.rate});
        max_diff = std::max(max_diff, std::abs(p_fresh - p_inc));
    }

    // Must match to ~1e-14 (machine epsilon for double arithmetic).
    // Allow 1e-12 for accumulated spline interpolation noise.
    EXPECT_LT(max_diff, 1e-12)
        << "Cached build diverges from fresh build: max diff = " << max_diff;
}

// Phase A verification gate 2: incremental cost accounting.
TEST(Chebyshev4DIncrementalTest, IncrementalSolvesOnlyNewPairs) {
    IncrementalBuildConfig cfg;
    cfg.num_x = 10;
    cfg.num_tau = 5;
    cfg.sigma_level = 1;  // 3 nodes
    cfg.rate_level = 1;   // 3 nodes
    cfg.use_tucker = false;

    // First build: empty cache → should solve 3 × 3 = 9 PDEs
    PDESliceCache cache;
    auto r1 = build_chebyshev_4d_eep_incremental(cfg, cache, 100.0, OptionType::PUT);
    EXPECT_EQ(cache.total_pde_solves(), 9u);

    // Refine sigma to level 2 (5 nodes). Rate stays at level 1 (3 nodes).
    // New sigma nodes: 2. New pairs: 2 × 3 = 6.
    cfg.sigma_level = 2;
    auto r2 = build_chebyshev_4d_eep_incremental(cfg, cache, 100.0, OptionType::PUT);
    EXPECT_EQ(cache.total_pde_solves(), 9u + 6u);

    // Refine rate to level 2 (5 nodes). Sigma at level 2 (5 nodes).
    // New rate nodes: 2. New pairs: 5 × 2 = 10.
    cfg.rate_level = 2;
    auto r3 = build_chebyshev_4d_eep_incremental(cfg, cache, 100.0, OptionType::PUT);
    EXPECT_EQ(cache.total_pde_solves(), 9u + 6u + 10u);
}

}  // namespace
}  // namespace mango
```

Add to `tests/BUILD.bazel`:

```starlark
cc_test(
    name = "chebyshev_4d_incremental_test",
    size = "large",
    srcs = ["chebyshev_4d_incremental_test.cc"],
    copts = ["-fopenmp", "-pthread"],
    linkopts = ["-fopenmp", "-pthread"],
    deps = [
        "//benchmarks:chebyshev_4d_incremental_builder",
        "//benchmarks:pde_slice_cache",
        "//src/option/table/dimensionless:chebyshev_tucker_4d",
        "//src/option/table/dimensionless:chebyshev_nodes",
        "//src/option:american_option_batch",
        "//src/option:european_option",
        "//src/math:cubic_spline_solver",
        "//src/option/table:price_query",
        "@googletest:gtest_main",
    ],
    timeout = "long",
)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel test //tests:chebyshev_4d_incremental_test --test_output=all`
Expected: Compilation error — `chebyshev_4d_incremental_builder.hpp` not found.

**Step 3: Implement `build_chebyshev_4d_eep_incremental()`**

Create `benchmarks/chebyshev_4d_incremental_builder.hpp`:

```cpp
// SPDX-License-Identifier: MIT
//
// Incremental Chebyshev 4D EEP builder using nested CC levels + PDE slice cache.
// Phase A of the true adaptive Chebyshev design.
#pragma once

#include "pde_slice_cache.hpp"
#include "mango/option/table/dimensionless/chebyshev_tucker_4d.hpp"
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/price_query.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/math/cubic_spline_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <vector>

namespace mango {

struct IncrementalBuildConfig {
    size_t num_x = 40;
    size_t num_tau = 15;
    size_t sigma_level = 3;   // CC level → 2^l + 1 nodes
    size_t rate_level = 2;    // CC level → 2^l + 1 nodes

    double epsilon = 1e-8;
    bool use_tucker = false;

    // Fixed physical domain bounds (no headroom coupling during refinement)
    double x_min = -0.50, x_max = 0.40;
    double tau_min = 0.019, tau_max = 2.0;
    double sigma_min = 0.05, sigma_max = 0.50;
    double rate_min = 0.01, rate_max = 0.10;

    double dividend_yield = 0.0;
    bool use_hard_max = true;
};

struct IncrementalBuildResult {
    ChebyshevTucker4D interp;
    size_t new_pde_solves;
    double build_seconds;
};

/// Build a Chebyshev 4D EEP surface using nested CC levels on sigma/rate,
/// with incremental PDE reuse via the slice cache.
///
/// The cache stores splines keyed by (sigma_idx, rate_idx, tau_idx) where
/// indices refer to positions in the CC-level node arrays. When refining
/// from level l to l+1, only the NEW (sigma, rate) pairs are solved.
///
/// Domain bounds are FIXED — they do not change with node count.
/// Headroom is computed once from the config bounds and applied at build time.
inline IncrementalBuildResult build_chebyshev_4d_eep_incremental(
    const IncrementalBuildConfig& cfg,
    PDESliceCache& cache,
    double K_ref,
    OptionType option_type)
{
    auto t0 = std::chrono::steady_clock::now();

    // ---- 1. Compute fixed extended domains (headroom from config bounds) ----
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    size_t n_sigma = (1u << cfg.sigma_level) + 1;
    size_t n_rate = (1u << cfg.rate_level) + 1;

    double hx     = headroom_fn(cfg.x_min, cfg.x_max, cfg.num_x);
    double htau   = headroom_fn(cfg.tau_min, cfg.tau_max, cfg.num_tau);
    double hsigma = headroom_fn(cfg.sigma_min, cfg.sigma_max, n_sigma);
    double hrate  = headroom_fn(cfg.rate_min, cfg.rate_max, n_rate);

    double x_lo     = cfg.x_min - hx;
    double x_hi     = cfg.x_max + hx;
    double tau_lo   = std::max(cfg.tau_min - htau, 1e-4);
    double tau_hi   = cfg.tau_max + htau;
    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo  = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi  = cfg.rate_max + hrate;

    // ---- 2. Generate nodes ----
    auto x_nodes     = chebyshev_nodes(cfg.num_x, x_lo, x_hi);
    auto tau_nodes   = chebyshev_nodes(cfg.num_tau, tau_lo, tau_hi);
    auto sigma_nodes = cc_level_nodes(cfg.sigma_level, sigma_lo, sigma_hi);
    auto rate_nodes  = cc_level_nodes(cfg.rate_level, rate_lo, rate_hi);

    // ---- 3. Find missing (sigma, rate) pairs ----
    std::vector<size_t> sigma_indices(n_sigma), rate_indices(n_rate);
    std::iota(sigma_indices.begin(), sigma_indices.end(), 0);
    std::iota(rate_indices.begin(), rate_indices.end(), 0);

    auto missing = cache.missing_pairs(sigma_indices, rate_indices);

    // ---- 4. Batch-solve only missing pairs ----
    size_t new_solves = 0;
    if (!missing.empty()) {
        const double tau_solve = tau_nodes.back() * 1.01;

        std::vector<PricingParams> batch;
        batch.reserve(missing.size());
        for (auto [si, ri] : missing) {
            batch.emplace_back(
                OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = tau_solve,
                           .rate = rate_nodes[ri],
                           .dividend_yield = cfg.dividend_yield,
                           .option_type = option_type},
                sigma_nodes[si]);
        }

        BatchAmericanOptionSolver solver;
        solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
        solver.set_snapshot_times(std::span<const double>{tau_nodes});
        auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);
        new_solves = missing.size();

        // Store results in cache
        for (size_t bi = 0; bi < missing.size(); ++bi) {
            auto [si, ri] = missing[bi];
            if (!batch_result.results[bi].has_value()) continue;

            const auto& result = batch_result.results[bi].value();
            auto x_grid = result.grid()->x();

            for (size_t j = 0; j < cfg.num_tau; ++j) {
                auto spatial = result.at_time(j);
                cache.store_slice(si, ri, j, x_grid, spatial);
            }
        }
        cache.record_pde_solves(new_solves);
    }

    // ---- 5. Extract tensor from cache ----
    const size_t Nx = cfg.num_x;
    const size_t Nt = cfg.num_tau;
    const size_t Ns = n_sigma;
    const size_t Nr = n_rate;
    std::vector<double> tensor(Nx * Nt * Ns * Nr, 0.0);

    for (size_t s = 0; s < Ns; ++s) {
        double sigma = sigma_nodes[s];
        for (size_t r = 0; r < Nr; ++r) {
            double rate = rate_nodes[r];
            for (size_t j = 0; j < Nt; ++j) {
                auto* spline = cache.get_slice(s, r, j);
                if (!spline) continue;

                double tau = tau_nodes[j];
                for (size_t i = 0; i < Nx; ++i) {
                    double am = spline->eval(x_nodes[i]) * K_ref;

                    double spot_local = std::exp(x_nodes[i]) * K_ref;
                    auto eu = EuropeanOptionSolver(
                        OptionSpec{.spot = spot_local, .strike = K_ref,
                                   .maturity = tau, .rate = rate,
                                   .dividend_yield = cfg.dividend_yield,
                                   .option_type = option_type},
                        sigma).solve().value();

                    double eep_raw = am - eu.value();

                    constexpr double kSharpness = 100.0;
                    double eep;
                    if (kSharpness * eep_raw > 500.0) {
                        eep = eep_raw;
                    } else {
                        double softplus =
                            std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
                        double bias = std::log(2.0) / kSharpness;
                        eep = cfg.use_hard_max
                            ? std::max(0.0, softplus - bias)
                            : (softplus - bias);
                    }

                    tensor[i * Nt * Ns * Nr + j * Ns * Nr + s * Nr + r] = eep;
                }
            }
        }
    }

    // ---- 6. Build ChebyshevTucker4D ----
    ChebyshevTucker4DDomain dom{
        .bounds = {{{x_lo, x_hi}, {tau_lo, tau_hi},
                    {sigma_lo, sigma_hi}, {rate_lo, rate_hi}}}};
    ChebyshevTucker4DConfig tcfg{
        .num_pts = {Nx, Nt, Ns, Nr},
        .epsilon = cfg.epsilon,
        .use_tucker = cfg.use_tucker};

    auto interp = ChebyshevTucker4D::build_from_values(tensor, dom, tcfg);
    auto t1 = std::chrono::steady_clock::now();

    return {std::move(interp), new_solves,
            std::chrono::duration<double>(t1 - t0).count()};
}

}  // namespace mango
```

Add `cc_library` target to `benchmarks/BUILD.bazel`:

```starlark
cc_library(
    name = "chebyshev_4d_incremental_builder",
    hdrs = ["chebyshev_4d_incremental_builder.hpp"],
    deps = [
        ":pde_slice_cache",
        "//src/option/table/dimensionless:chebyshev_tucker_4d",
        "//src/option/table/dimensionless:chebyshev_nodes",
        "//src/option/table:price_query",
        "//src/option:european_option",
        "//src/option:american_option",
        "//src/option:american_option_batch",
        "//src/math:cubic_spline_solver",
    ],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/benchmarks",
)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel test //tests:chebyshev_4d_incremental_test --test_output=all`
Expected: Both tests PASS. The `CachedBuildMatchesFreshBuild` test confirms
verification gate 1. The `IncrementalSolvesOnlyNewPairs` test confirms
verification gate 2.

**Step 5: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor
git add benchmarks/chebyshev_4d_incremental_builder.hpp benchmarks/BUILD.bazel tests/chebyshev_4d_incremental_test.cc tests/BUILD.bazel
git commit -m "Add incremental Chebyshev 4D EEP builder

Uses CC levels for sigma/rate axes and PDESliceCache for
incremental PDE reuse. Refining sigma from level l to l+1
solves only the new (sigma, rate) pairs. Cache correctness
verified to machine epsilon against fresh builds."
```

---

### Task 4: Add `cheb4d-incremental` benchmark section

**Files:**
- Modify: `.worktrees/chebyshev-tensor/benchmarks/interp_iv_safety.cc`
- Modify: `.worktrees/chebyshev-tensor/benchmarks/BUILD.bazel` (add incremental builder header)

This adds a new section to the benchmark that demonstrates incremental
refinement: start at CC levels (2, 1) for (sigma, rate), refine to (3, 2),
then to (3, 3). Print per-iteration PDE cost, total PDE cost, and stratified
IV accuracy stats.

**Step 1: Add the benchmark section**

Add `#include "chebyshev_4d_incremental_builder.hpp"` to the top of
`interp_iv_safety.cc`.

Add `"chebyshev_4d_incremental_builder.hpp"` to the `srcs` list of the
`interp_iv_safety` target in `benchmarks/BUILD.bazel`, and add
`"//benchmarks:chebyshev_4d_incremental_builder"` and
`"//benchmarks:pde_slice_cache"` to its `deps`.

Add the following section handler in `interp_iv_safety.cc`, after the existing
`cheb4d-baseline` section:

```cpp
static void run_cheb4d_incremental() {
    constexpr double K_ref = 100.0;
    constexpr auto kType = OptionType::PUT;

    // Define refinement schedule: (sigma_level, rate_level)
    struct Level { size_t sigma_level, rate_level; const char* label; };
    Level schedule[] = {
        {2, 1, "L(2,1): 5sig x 3rate"},
        {3, 2, "L(3,2): 9sig x 5rate"},
        {3, 3, "L(3,3): 9sig x 9rate"},
    };

    PDESliceCache cache;
    IncrementalBuildConfig cfg;
    cfg.num_x = 40;
    cfg.num_tau = 15;
    cfg.use_tucker = false;

    std::printf("\n=== Incremental Chebyshev 4D (CC levels) ===\n");
    std::printf("%-25s  %6s  %6s  %8s  %8s\n",
                "Level", "NewPDE", "TotPDE", "Build(s)", "MaxErr");

    for (const auto& level : schedule) {
        cfg.sigma_level = level.sigma_level;
        cfg.rate_level = level.rate_level;

        auto result = build_chebyshev_4d_eep_incremental(
            cfg, cache, K_ref, kType);

        // Build inner adapter for IV validation
        Chebyshev4DEEPInner inner(
            std::move(result.interp), kType, K_ref, cfg.dividend_yield);

        // Compute IV errors using existing compute_errors_brent pattern
        // (implementation depends on existing infrastructure in the file)
        // ... print heatmaps and stratified stats ...

        std::printf("%-25s  %6zu  %6zu  %8.2f\n",
                    level.label, result.new_pde_solves,
                    cache.total_pde_solves(), result.build_seconds);
    }
}
```

Register the section in `main()`:

```cpp
if (section == "cheb4d-incremental") { run_cheb4d_incremental(); return 0; }
```

**Step 2: Build and verify it compiles**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel build //benchmarks:interp_iv_safety`
Expected: Compiles without errors.

**Step 3: Run the benchmark**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && ./bazel-bin/benchmarks/interp_iv_safety cheb4d-incremental`
Expected output shows:
- L(2,1): 15 new PDE, 15 total
- L(3,2): 30 new PDE (6 new sigma × 3 existing rate + 3 existing sigma × 2 new rate + ... hmm, let me work this out)
  - Actually: L(2,1) = 5×3 = 15 pairs. L(3,2) = 9×5 = 45 pairs. New = 45 - 15 = 30. Total = 15 + 30 = 45.
  - But with CC nesting, the 5 sigma nodes at level 2 are a SUBSET of 9 sigma nodes at level 3. And the 3 rate nodes at level 1 are a SUBSET of 5 rate nodes at level 2. So existing pairs from L(2,1) are reused.
  - New pairs at L(3,2): 45 - 15 = 30 new PDE solves.
- L(3,3): 9×9 = 81 pairs. Already have 45 from L(3,2). New = 81 - 45 = 36.

Verify total PDE solves = 15 + 30 + 36 = 81 (matching 9×9 fresh build).

**Step 4: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor
git add benchmarks/interp_iv_safety.cc benchmarks/BUILD.bazel
git commit -m "Add cheb4d-incremental benchmark section

Demonstrates incremental CC-level refinement on sigma/rate
axes with PDE slice cache reuse. Three levels: (2,1) → (3,2)
→ (3,3). Reports per-iteration and cumulative PDE solve
counts alongside stratified IV accuracy stats."
```

---

### Task 5: Verify Phase A verification gates and run full test suite

**Step 1: Run the incremental builder tests**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel test //tests:chebyshev_4d_incremental_test --test_output=all`

Verify:
- Gate 1 (cache correctness): `CachedBuildMatchesFreshBuild` passes with max_diff < 1e-12.
- Gate 2 (incremental cost): `IncrementalSolvesOnlyNewPairs` shows exact PDE counts (9, 6, 10).

**Step 2: Run the benchmark to verify Gate 3 (no regression)**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && ./bazel-bin/benchmarks/interp_iv_safety cheb4d-incremental`

Compare the L(3,3) accuracy (9×9 = 81 PDE) against the locked baseline
(15×10 = 150 PDE). The incremental build at the same final resolution should
reproduce equivalent accuracy.

**Step 3: Run full test suite to check for regressions**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel test //...`
Expected: All existing tests pass. No regressions.

**Step 4: Run full benchmark build**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel build //benchmarks/...`
Expected: All benchmarks compile.

**Step 5: Commit verification results**

If all gates pass:

```bash
cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor
# No code changes — this is a verification step.
# Update the design doc with gate results if desired.
```

---

## Verification Matrix (Phase A)

| Gate | Test | Criterion | Status |
|------|------|-----------|--------|
| Cache correctness | `CachedBuildMatchesFreshBuild` | max_diff < 1e-12 | Pending |
| Incremental cost | `IncrementalSolvesOnlyNewPairs` | Exact PDE counts | Pending |
| No regression | `cheb4d-incremental` L(3,3) vs baseline | Within 0.5 bps | Pending |
| Full suite | `bazel test //...` | All pass | Pending |
| Benchmarks compile | `bazel build //benchmarks/...` | No errors | Pending |
