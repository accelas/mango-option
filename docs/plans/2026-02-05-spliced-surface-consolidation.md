# Spliced Surface Consolidation — Clean Replacement Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the three separate surface classes with unified `SplicedSurface` type aliases, eliminating ~1200 lines of duplicated code and enabling natural composition.

**Architecture:** Single `SplicedSurface<Inner, Split, Xform, Comb>` template with type aliases for each use case. Strategies and transforms are mix-and-match components.

**Tech Stack:** C++23, Bazel, GoogleTest

---

## Overview

**Remove:**
- `PerMaturityPriceSurface` (per_maturity_price_surface.{hpp,cpp})
- `SegmentedPriceSurface` (segmented_price_surface.{hpp,cpp})
- `SegmentedMultiKRefSurface` (segmented_multi_kref_surface.{hpp,cpp})

**Keep & Extend:**
- `spliced_surface.hpp` — add missing components and type aliases

**Update:**
- `adaptive_grid_builder` — use new types
- `adaptive_grid_types` — simplify result struct
- `iv_solver_factory` — use new builders
- Tests — migrate to new APIs

---

### Task 1: Add missing split strategy — KRefBracket

**Files:**
- Modify: `src/option/table/spliced_surface.hpp`

**Step 1: Add KRefBracket class**

After `LinearBracket` (line ~237), add:

```cpp
/// Split strategy for K_ref bracket interpolation.
/// Finds two K_refs bracketing the query strike and computes linear weights.
class KRefBracket {
public:
    explicit KRefBracket(std::vector<double> k_refs)
        : k_refs_(std::move(k_refs))
    {}

    [[nodiscard]] double key(const PriceQuery& q) const noexcept { return q.strike; }
    [[nodiscard]] size_t num_slices() const noexcept { return k_refs_.size(); }

    [[nodiscard]] Bracket bracket(double strike) const noexcept {
        Bracket br;
        const size_t n = k_refs_.size();
        if (n == 0) {
            return br;
        }
        if (n == 1 || strike <= k_refs_.front()) {
            br.items[0] = SliceWeight{0, 1.0};
            br.size = 1;
            return br;
        }
        if (strike >= k_refs_.back()) {
            br.items[0] = SliceWeight{n - 1, 1.0};
            br.size = 1;
            return br;
        }

        // Find bracketing K_refs
        size_t hi = 1;
        while (hi < n && k_refs_[hi] < strike) {
            ++hi;
        }
        size_t lo = hi - 1;

        double K_lo = k_refs_[lo];
        double K_hi = k_refs_[hi];
        double t = (strike - K_lo) / (K_hi - K_lo);

        br.items[0] = SliceWeight{lo, 1.0 - t};
        br.items[1] = SliceWeight{hi, t};
        br.size = 2;
        return br;
    }

    [[nodiscard]] const std::vector<double>& k_refs() const noexcept { return k_refs_; }

private:
    std::vector<double> k_refs_;
};
```

**Step 2: Build to verify**

Run: `bazel build //src/option/table:spliced_surface`

Expected: Clean compile.

**Step 3: Commit**

```bash
git add src/option/table/spliced_surface.hpp
git commit -m "Add KRefBracket split strategy"
```

---

### Task 2: Add adapters for existing surface types

**Files:**
- Modify: `src/option/table/spliced_surface.hpp`
- Modify: `src/option/table/BUILD.bazel`

**Step 1: Add includes**

At top of `spliced_surface.hpp`:

```cpp
#include "mango/math/black_scholes_analytics.hpp"
#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/american_price_surface.hpp"
```

**Step 2: Add PriceTableSurface3DAdapter**

After `KRefTransform`:

```cpp
/// Adapter wrapping PriceTableSurface<3> for SplicedSurface.
/// Maps PriceQuery {spot, strike, tau, sigma, rate} to 3D coords {m, sigma, rate}.
class PriceTableSurface3DAdapter {
public:
    PriceTableSurface3DAdapter(
        std::shared_ptr<const PriceTableSurface<3>> surface,
        double K_ref)
        : surface_(std::move(surface)), K_ref_(K_ref) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double m = q.spot / q.strike;
        return surface_->value({m, q.sigma, q.rate});
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        double m = q.spot / q.strike;
        double eps = std::max(1e-4, 0.01 * q.sigma);
        double sigma_dn = std::max(1e-4, q.sigma - eps);
        double v_up = surface_->value({m, q.sigma + eps, q.rate});
        double v_dn = surface_->value({m, sigma_dn, q.rate});
        return (v_up - v_dn) / (q.sigma + eps - sigma_dn);
    }

    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }

private:
    std::shared_ptr<const PriceTableSurface<3>> surface_;
    double K_ref_;
};

static_assert(PriceSurface<PriceTableSurface3DAdapter>);
```

**Step 3: Add AmericanPriceSurfaceAdapter**

```cpp
/// Adapter wrapping AmericanPriceSurface for SplicedSurface.
class AmericanPriceSurfaceAdapter {
public:
    explicit AmericanPriceSurfaceAdapter(AmericanPriceSurface surface)
        : surface_(std::move(surface)) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        return surface_.price(q.spot, q.strike, q.tau, q.sigma, q.rate);
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        return surface_.vega(q.spot, q.strike, q.tau, q.sigma, q.rate);
    }

    [[nodiscard]] const AmericanPriceSurface& surface() const { return surface_; }

private:
    AmericanPriceSurface surface_;
};

static_assert(PriceSurface<AmericanPriceSurfaceAdapter>);
```

**Step 4: Update BUILD deps**

```python
cc_library(
    name = "spliced_surface",
    hdrs = ["spliced_surface.hpp"],
    deps = [
        ":american_price_surface",
        ":price_table_metadata",
        ":price_table_surface",
        "//src/math:black_scholes_analytics",
        "//src/option:option_spec",
    ],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option/table",
    include_prefix = "mango/option/table",
)
```

**Step 5: Build to verify**

Run: `bazel build //src/option/table:spliced_surface`

**Step 6: Commit**

```bash
git add src/option/table/spliced_surface.hpp src/option/table/BUILD.bazel
git commit -m "Add surface adapters for SplicedSurface integration"
```

---

### Task 3: Add MaturityTransform for EEP reconstruction

**Files:**
- Modify: `src/option/table/spliced_surface.hpp`

**Step 1: Add MaturityTransform**

After adapters:

```cpp
/// Transform for per-maturity EEP surfaces.
/// Reconstructs American price: P_Am = EEP + P_Eu.
struct MaturityTransform {
    OptionType option_type = OptionType::PUT;
    double dividend_yield = 0.0;

    [[nodiscard]] PriceQuery to_local(size_t, const PriceQuery& q) const noexcept {
        return q;
    }

    [[nodiscard]] double normalize_value(size_t, const PriceQuery& q, double eep) const {
        double p_eu = bs_price(q.spot, q.strike, q.tau, q.sigma, q.rate,
                               dividend_yield, option_type);
        return eep + p_eu;
    }
};
```

**Step 2: Build to verify**

Run: `bazel build //src/option/table:spliced_surface`

**Step 3: Commit**

```bash
git add src/option/table/spliced_surface.hpp
git commit -m "Add MaturityTransform for EEP reconstruction"
```

---

### Task 4: Add type aliases for the three surface patterns

**Files:**
- Modify: `src/option/table/spliced_surface.hpp`

**Step 1: Add type aliases at end of file**

Before closing namespace:

```cpp
// ===========================================================================
// Unified surface type aliases
// ===========================================================================

/// Per-maturity surface: linear τ interpolation over 3D EEP surfaces.
/// Replaces PerMaturityPriceSurface.
using PerMaturitySurface = SplicedSurface<
    PriceTableSurface3DAdapter,
    LinearBracket,
    MaturityTransform,
    WeightedSum>;

/// Segmented surface: dividend segment lookup with spot adjustment.
/// Replaces SegmentedPriceSurface.
/// Inner can be AmericanPriceSurfaceAdapter or PerMaturitySurface.
template<PriceSurface Inner = AmericanPriceSurfaceAdapter>
using SegmentedSurface = SplicedSurface<
    Inner,
    SegmentLookup,
    SegmentedTransform,
    WeightedSum>;

/// Multi-K_ref surface: strike bracket interpolation.
/// Replaces SegmentedMultiKRefSurface.
template<PriceSurface Inner = SegmentedSurface<>>
using MultiKRefSurface = SplicedSurface<
    Inner,
    KRefBracket,
    KRefTransform,
    WeightedSum>;
```

**Step 2: Build to verify**

Run: `bazel build //src/option/table:spliced_surface`

**Step 3: Commit**

```bash
git add src/option/table/spliced_surface.hpp
git commit -m "Add unified type aliases: PerMaturitySurface, SegmentedSurface, MultiKRefSurface"
```

---

### Task 5: Create unified builder functions

**Files:**
- Create: `src/option/table/spliced_surface_builder.hpp`
- Create: `src/option/table/spliced_surface_builder.cpp`
- Modify: `src/option/table/BUILD.bazel`

**Step 1: Create header**

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/spliced_surface.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace mango {

// ===========================================================================
// Per-maturity surface builder
// ===========================================================================

struct PerMaturityConfig {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    std::vector<double> tau_grid;
    double K_ref;
    OptionType option_type;
    double dividend_yield;
};

[[nodiscard]] std::expected<PerMaturitySurface, PriceTableError>
build_per_maturity_surface(PerMaturityConfig config);

// ===========================================================================
// Segmented surface builder
// ===========================================================================

struct SegmentConfig {
    AmericanPriceSurface surface;
    double tau_start;
    double tau_end;
};

struct SegmentedConfig {
    std::vector<SegmentConfig> segments;
    std::vector<Dividend> dividends;
    double K_ref;
    double T;  // expiry in calendar time
};

[[nodiscard]] std::expected<SegmentedSurface<>, PriceTableError>
build_segmented_surface(SegmentedConfig config);

// ===========================================================================
// Multi-K_ref surface builder
// ===========================================================================

struct MultiKRefEntry {
    double K_ref;
    SegmentedSurface<> surface;
};

[[nodiscard]] std::expected<MultiKRefSurface<>, PriceTableError>
build_multi_kref_surface(std::vector<MultiKRefEntry> entries);

}  // namespace mango
```

**Step 2: Create implementation**

```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/table/spliced_surface_builder.hpp"
#include <algorithm>

namespace mango {

std::expected<PerMaturitySurface, PriceTableError>
build_per_maturity_surface(PerMaturityConfig config) {
    if (config.surfaces.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    if (config.surfaces.size() != config.tau_grid.size()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    if (config.tau_grid.size() < 2) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Verify sorted
    for (size_t i = 1; i < config.tau_grid.size(); ++i) {
        if (config.tau_grid[i] <= config.tau_grid[i - 1]) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
    }

    // Verify non-null
    for (const auto& s : config.surfaces) {
        if (!s) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
    }

    // Build adapters
    std::vector<PriceTableSurface3DAdapter> slices;
    slices.reserve(config.surfaces.size());
    for (const auto& s : config.surfaces) {
        slices.emplace_back(s, config.K_ref);
    }

    LinearBracket split(config.tau_grid);
    MaturityTransform xform{config.option_type, config.dividend_yield};
    WeightedSum combine;

    return PerMaturitySurface(
        std::move(slices), std::move(split), std::move(xform), combine);
}

std::expected<SegmentedSurface<>, PriceTableError>
build_segmented_surface(SegmentedConfig config) {
    if (config.segments.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Extract tau bounds and build adapters
    std::vector<double> tau_start, tau_end;
    std::vector<double> tau_min, tau_max;
    std::vector<SurfaceContent> content;
    std::vector<AmericanPriceSurfaceAdapter> slices;

    for (auto& seg : config.segments) {
        tau_start.push_back(seg.tau_start);
        tau_end.push_back(seg.tau_end);
        tau_min.push_back(seg.surface.tau_min());
        tau_max.push_back(seg.surface.tau_max());
        content.push_back(seg.surface.metadata().content);
        slices.emplace_back(std::move(seg.surface));
    }

    SegmentLookup split(std::move(tau_start), std::move(tau_end));
    SegmentedTransform xform{
        .tau_start = split.tau_start_,  // Need to expose or copy
        .tau_min = std::move(tau_min),
        .tau_max = std::move(tau_max),
        .content = std::move(content),
        .dividends = std::move(config.dividends),
        .K_ref = config.K_ref,
        .T = config.T
    };
    WeightedSum combine;

    return SegmentedSurface<>(
        std::move(slices), std::move(split), std::move(xform), combine);
}

std::expected<MultiKRefSurface<>, PriceTableError>
build_multi_kref_surface(std::vector<MultiKRefEntry> entries) {
    if (entries.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Sort by K_ref
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.K_ref < b.K_ref; });

    std::vector<double> k_refs;
    std::vector<SegmentedSurface<>> slices;
    for (auto& e : entries) {
        k_refs.push_back(e.K_ref);
        slices.push_back(std::move(e.surface));
    }

    KRefBracket split(std::move(k_refs));
    KRefTransform xform{split.k_refs()};
    WeightedSum combine;

    return MultiKRefSurface<>(
        std::move(slices), std::move(split), std::move(xform), combine);
}

}  // namespace mango
```

**Step 3: Add BUILD target**

```python
cc_library(
    name = "spliced_surface_builder",
    srcs = ["spliced_surface_builder.cpp"],
    hdrs = ["spliced_surface_builder.hpp"],
    deps = [
        ":spliced_surface",
        "//src/support:error_types",
    ],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option/table",
    include_prefix = "mango/option/table",
)
```

**Step 4: Build to verify**

Run: `bazel build //src/option/table:spliced_surface_builder`

**Step 5: Commit**

```bash
git add src/option/table/spliced_surface_builder.hpp \
        src/option/table/spliced_surface_builder.cpp \
        src/option/table/BUILD.bazel
git commit -m "Add unified builder functions for spliced surfaces"
```

---

### Task 6: Update AdaptiveGridTypes to use new types

**Files:**
- Modify: `src/option/table/adaptive_grid_types.hpp`

**Step 1: Replace includes**

```cpp
#include "mango/option/table/spliced_surface.hpp"
// Remove: #include "mango/option/table/per_maturity_price_surface.hpp"
```

**Step 2: Simplify AdaptiveResult**

Replace the multiple surface fields with:

```cpp
struct AdaptiveResult {
    /// The built surface (4D B-spline for legacy mode)
    std::shared_ptr<const PriceTableSurface<4>> surface = nullptr;

    /// Per-maturity spliced surface (when use_per_maturity=true)
    std::optional<PerMaturitySurface> per_maturity = std::nullopt;

    /// Final axes
    PriceTableAxes<4> axes;

    /// Query price from whichever surface is populated
    [[nodiscard]] double value(const std::array<double, 4>& coords) const {
        if (per_maturity.has_value()) {
            // Convert coords to PriceQuery (assume K_ref=100 for moneyness)
            PriceQuery q{
                .spot = coords[0] * 100.0,
                .strike = 100.0,
                .tau = coords[1],
                .sigma = coords[2],
                .rate = coords[3]
            };
            return per_maturity->price(q);
        }
        return surface ? surface->value(coords)
                       : std::numeric_limits<double>::quiet_NaN();
    }

    // ... rest unchanged
};
```

**Step 3: Build to verify**

Run: `bazel build //src/option/table:adaptive_grid_types`

**Step 4: Commit**

```bash
git add src/option/table/adaptive_grid_types.hpp
git commit -m "Simplify AdaptiveResult to use PerMaturitySurface"
```

---

### Task 7: Update AdaptiveGridBuilder to use new builders

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.hpp`
- Modify: `src/option/table/adaptive_grid_builder.cpp`

**Step 1: Update includes**

```cpp
#include "mango/option/table/spliced_surface_builder.hpp"
// Remove segmented includes if no longer needed
```

**Step 2: Update build_segmented to use new builder**

Replace `SegmentedMultiKRefSurface` return type with `MultiKRefSurface<>`:

```cpp
[[nodiscard]] std::expected<MultiKRefSurface<>, PriceTableError>
build_segmented(const SegmentedAdaptiveConfig& config, ...);
```

**Step 3: Update implementation**

Use `build_segmented_surface()` and `build_multi_kref_surface()` instead of the old builders.

**Step 4: Build to verify**

Run: `bazel build //src/option/table:adaptive_grid_builder`

**Step 5: Commit**

```bash
git add src/option/table/adaptive_grid_builder.hpp \
        src/option/table/adaptive_grid_builder.cpp
git commit -m "Update AdaptiveGridBuilder to use spliced surface builders"
```

---

### Task 8: Update IVSolverFactory

**Files:**
- Modify: `src/option/iv_solver_factory.hpp`
- Modify: `src/option/iv_solver_factory.cpp`

**Step 1: Update includes and types**

Replace `SegmentedMultiKRefSurface` with `MultiKRefSurface<>`.

**Step 2: Update implementation**

Use new builder functions.

**Step 3: Build to verify**

Run: `bazel build //src/option:iv_solver_factory`

**Step 4: Commit**

```bash
git add src/option/iv_solver_factory.hpp src/option/iv_solver_factory.cpp
git commit -m "Update IVSolverFactory to use spliced surfaces"
```

---

### Task 9: Remove old surface classes

**Files:**
- Delete: `src/option/table/per_maturity_price_surface.hpp`
- Delete: `src/option/table/per_maturity_price_surface.cpp`
- Delete: `src/option/table/segmented_price_surface.hpp`
- Delete: `src/option/table/segmented_price_surface.cpp`
- Delete: `src/option/table/segmented_multi_kref_surface.hpp`
- Delete: `src/option/table/segmented_multi_kref_surface.cpp`
- Delete: `src/option/table/segmented_price_table_builder.hpp`
- Delete: `src/option/table/segmented_price_table_builder.cpp`
- Delete: `src/option/table/segmented_multi_kref_builder.hpp`
- Delete: `src/option/table/segmented_multi_kref_builder.cpp`
- Modify: `src/option/table/BUILD.bazel`

**Step 1: Remove files**

```bash
rm src/option/table/per_maturity_price_surface.{hpp,cpp}
rm src/option/table/segmented_price_surface.{hpp,cpp}
rm src/option/table/segmented_multi_kref_surface.{hpp,cpp}
rm src/option/table/segmented_price_table_builder.{hpp,cpp}
rm src/option/table/segmented_multi_kref_builder.{hpp,cpp}
```

**Step 2: Remove BUILD targets**

Remove the corresponding `cc_library` entries from BUILD.bazel.

**Step 3: Build to verify no dangling deps**

Run: `bazel build //src/option/table/...`

**Step 4: Commit**

```bash
git add -A
git commit -m "Remove old surface classes replaced by SplicedSurface"
```

---

### Task 10: Migrate tests

**Files:**
- Delete: `tests/per_maturity_price_surface_test.cc`
- Delete: `tests/segmented_price_surface_test.cc`
- Delete: `tests/segmented_multi_kref_surface_test.cc`
- Modify: `tests/spliced_surface_test.cc` — expand with migrated tests
- Modify: `tests/BUILD.bazel`

**Step 1: Expand spliced_surface_test.cc**

Add tests covering:
- PerMaturitySurface build validation
- SegmentedSurface dividend adjustment
- MultiKRefSurface strike interpolation

**Step 2: Remove old test files**

```bash
rm tests/per_maturity_price_surface_test.cc
rm tests/segmented_price_surface_test.cc
rm tests/segmented_multi_kref_surface_test.cc
```

**Step 3: Update BUILD.bazel**

Remove old test targets, update spliced_surface_test deps.

**Step 4: Run tests**

Run: `bazel test //tests:spliced_surface_test --test_output=errors`

**Step 5: Commit**

```bash
git add -A
git commit -m "Consolidate surface tests into spliced_surface_test"
```

---

### Task 11: Run full test suite

**Step 1: Build everything**

Run: `bazel build //...`

**Step 2: Run all tests**

Run: `bazel test //...`

**Step 3: Fix any failures**

**Step 4: Commit fixes**

```bash
git add -A
git commit -m "Fix test failures after surface consolidation"
```

---

### Task 12: Run benchmark and create PR

**Step 1: Run IV safety benchmark**

Run: `bazel run //benchmarks:interp_iv_safety`

**Step 2: Push branch**

Run: `git push -u origin feature/per-maturity-surface`

**Step 3: Create PR**

```bash
gh pr create --title "Consolidate surface types into unified SplicedSurface abstraction" --body "$(cat <<'EOF'
## Summary

Replaces three separate surface classes with a single unified `SplicedSurface<Inner, Split, Xform, Comb>` template, eliminating ~1200 lines of duplicated code.

## Removed Classes
- `PerMaturityPriceSurface` → `PerMaturitySurface` type alias
- `SegmentedPriceSurface` → `SegmentedSurface<>` type alias
- `SegmentedMultiKRefSurface` → `MultiKRefSurface<>` type alias

## New Architecture
```
SplicedSurface<Inner, Split, Xform, Comb>
├── Split strategies: SegmentLookup, LinearBracket, KRefBracket
├── Transforms: IdentityTransform, SegmentedTransform, MaturityTransform, KRefTransform
├── Combiners: WeightedSum
└── Adapters: PriceTableSurface3DAdapter, AmericanPriceSurfaceAdapter
```

## Benefits
- Single pattern for all "spliced" surfaces
- Natural composition: `MultiKRefSurface<SegmentedSurface<PerMaturitySurface>>`
- Mix-and-match strategies and transforms
- ~1200 lines removed

## Testing
- All existing tests migrated
- Full test suite passing

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
