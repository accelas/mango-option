# Phase C: Local h-Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a piecewise Chebyshev 4D evaluator with 2 τ-bands × 3 x-elements = 6 elements, C∞ overlap blending, boundary detection from cached PDE slices, and shape constraints. Reduce p95 IV error from 979 bps to <100 bps in the 60d-180d σ=15% regime.

**Architecture:** Extend the existing `PiecewiseChebyshev4DEEPInner` in `benchmarks/chebyshev_4d_eep_inner.hpp` with τ-banding, overlap blending, and boundary detection. Build Phase C elements from the `PDESliceCache` (Phase A). Each element is an independent `ChebyshevTucker4D`. At query time, map (x, τ) to elements, blend overlapping evaluations with C∞ bump weights.

**Tech Stack:** C++23, Bazel, ChebyshevTucker4D, PDESliceCache, CubicSpline, EuropeanOptionSolver

**Design spec:** `docs/plans/2026-02-09-phase-c-local-h-refinement-design.md`

**Worktree:** `.worktrees/chebyshev-tensor/` on branch `experiment/chebyshev-tensor`

---

### Task 1: C∞ Bump Weight Function

Implement the smooth partition-of-unity weight function used for overlap blending.

**Files:**
- Create: `benchmarks/bump_blend.hpp`

**Step 1: Write the header**

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <algorithm>

namespace mango {

/// C∞ bump function: ψ(t) = exp(-1 / (1 - (2t-1)²)) for |2t-1| < 1, else 0.
/// Normalized CDF: Ψ(t) = ∫₀ᵗ ψ(s) ds / ∫₀¹ ψ(s) ds.
/// Returns the right-side weight w_right ∈ [0, 1].
/// At t=0: returns 0 (pure left).  At t=1: returns 1 (pure right).
inline double bump_blend_weight(double t) {
    t = std::clamp(t, 0.0, 1.0);

    // Numerical integration via 64-point Gauss-Legendre would be exact,
    // but a lookup + interpolation is simpler and sufficient.
    // Use Simpson's rule with 256 subintervals (precomputed at first call).
    constexpr int N = 256;
    static const auto table = [] {
        auto psi = [](double s) -> double {
            double u = 2.0 * s - 1.0;
            double u2 = u * u;
            if (u2 >= 1.0) return 0.0;
            return std::exp(-1.0 / (1.0 - u2));
        };

        // Build cumulative integral via composite Simpson
        std::array<double, N + 1> cdf{};
        cdf[0] = 0.0;
        double h = 1.0 / N;
        for (int i = 0; i < N; ++i) {
            double a = i * h;
            double b = (i + 1) * h;
            double mid = (a + b) / 2.0;
            cdf[i + 1] = cdf[i] + (h / 6.0) * (psi(a) + 4.0 * psi(mid) + psi(b));
        }
        // Normalize
        double total = cdf[N];
        for (auto& v : cdf) v /= total;
        return cdf;
    }();

    // Interpolate in table
    double idx = t * N;
    int lo = static_cast<int>(idx);
    lo = std::clamp(lo, 0, N - 1);
    double frac = idx - lo;
    return table[lo] * (1.0 - frac) + table[lo + 1] * frac;
}

/// Convenience: given x in overlap zone [a, b], return the right-side weight.
/// Outside [a, b]: clamps to 0 or 1.
inline double overlap_weight_right(double x, double a, double b) {
    if (b <= a) return 0.5;  // degenerate
    double t = (x - a) / (b - a);
    return bump_blend_weight(t);
}

}  // namespace mango
```

**Step 2: Write a unit test**

Create `tests/bump_blend_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "bump_blend.hpp"
#include <cmath>

using namespace mango;

TEST(BumpBlendTest, BoundaryValues) {
    EXPECT_NEAR(bump_blend_weight(0.0), 0.0, 1e-10);
    EXPECT_NEAR(bump_blend_weight(1.0), 1.0, 1e-10);
}

TEST(BumpBlendTest, SymmetryAtMidpoint) {
    EXPECT_NEAR(bump_blend_weight(0.5), 0.5, 1e-6);
}

TEST(BumpBlendTest, Monotonicity) {
    for (int i = 0; i < 100; ++i) {
        double t1 = i / 100.0;
        double t2 = (i + 1) / 100.0;
        EXPECT_LE(bump_blend_weight(t1), bump_blend_weight(t2) + 1e-15)
            << "Non-monotone at t=" << t1;
    }
}

TEST(BumpBlendTest, PartitionOfUnity) {
    // w_left + w_right = 1 for all t
    for (int i = 0; i <= 100; ++i) {
        double t = i / 100.0;
        double w_right = bump_blend_weight(t);
        double w_left = 1.0 - w_right;
        EXPECT_NEAR(w_left + w_right, 1.0, 1e-15);
    }
}

TEST(BumpBlendTest, OverlapWeightRight) {
    EXPECT_NEAR(overlap_weight_right(10.0, 10.0, 20.0), 0.0, 1e-10);
    EXPECT_NEAR(overlap_weight_right(20.0, 10.0, 20.0), 1.0, 1e-10);
    EXPECT_NEAR(overlap_weight_right(15.0, 10.0, 20.0), 0.5, 1e-6);
    // Outside range: clamp
    EXPECT_NEAR(overlap_weight_right(5.0, 10.0, 20.0), 0.0, 1e-10);
    EXPECT_NEAR(overlap_weight_right(25.0, 10.0, 20.0), 1.0, 1e-10);
}
```

**Step 3: Add BUILD rule for test**

Add to `tests/BUILD.bazel`:
```python
cc_test(
    name = "bump_blend_test",
    srcs = ["bump_blend_test.cc"],
    deps = [
        "//benchmarks:bump_blend",
        "@googletest//:gtest_main",
    ],
)
```

Add to `benchmarks/BUILD.bazel`:
```python
cc_library(
    name = "bump_blend",
    hdrs = ["bump_blend.hpp"],
    visibility = ["//tests:__pkg__"],
)
```

**Step 4: Run test**

Run: `bazel test //tests:bump_blend_test --test_output=all`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add benchmarks/bump_blend.hpp tests/bump_blend_test.cc tests/BUILD.bazel benchmarks/BUILD.bazel
git commit -m "Add C∞ bump blend weight function for overlap blending"
```

---

### Task 2: Boundary Detection from Cached PDE Slices

Detect the exercise boundary x\*(τ-band) by scanning cached PDE slices. This determines where to place element breaks.

**Files:**
- Create: `benchmarks/boundary_detector.hpp`
- Create: `tests/boundary_detector_test.cc`

**Step 1: Write the header**

```cpp
// SPDX-License-Identifier: MIT
//
// Exercise boundary detection from cached PDE slices.
// Scans EEP = Am - Eu to locate the zero-crossing x* per τ-band.
#pragma once

#include "pde_slice_cache.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace mango {

struct BoundaryResult {
    double x_star;     // median boundary location
    double delta;      // half-width for boundary element
    size_t n_valid;    // number of valid triples
    size_t n_sampled;  // total triples sampled
};

struct BoundaryDetectorConfig {
    size_t n_scan_points = 200;      // x-points for EEP scan
    double eps_scale = 1e-6;         // ε = max(eps_scale * K_ref, 1e-8)
    double delta_min = 0.10;         // minimum half-width
    double delta_margin = 0.05;      // added to (p90 - p10) / 2
    double valid_fraction = 0.30;    // min valid triples before fallback
};

/// Detect exercise boundary for a τ-band.
///
/// @param cache        Populated PDE slice cache
/// @param tau_nodes    Physical τ values matching cache tau_idx 0..N-1
/// @param sigma_nodes  σ nodes in the cache
/// @param rate_nodes   Rate nodes in the cache
/// @param tau_idx_lo   First tau_idx in this band (inclusive)
/// @param tau_idx_hi   Last tau_idx in this band (inclusive)
/// @param K_ref        Reference strike
/// @param option_type  PUT or CALL
/// @param cfg          Detection parameters
/// @param dividend_yield Continuous dividend yield
/// @param fallback_x_star Optional fallback from other τ-band (NaN if none)
inline BoundaryResult detect_exercise_boundary(
    const PDESliceCache& cache,
    std::span<const double> tau_nodes,
    std::span<const double> sigma_nodes,
    std::span<const double> rate_nodes,
    size_t tau_idx_lo, size_t tau_idx_hi,
    double K_ref, OptionType option_type,
    const BoundaryDetectorConfig& cfg = {},
    double dividend_yield = 0.0,
    double fallback_x_star = std::numeric_limits<double>::quiet_NaN())
{
    double x_min = -0.50, x_max = 0.40;  // physical domain
    double eps = std::max(cfg.eps_scale * K_ref, 1e-8);

    std::vector<double> all_x_stars;
    size_t n_sampled = 0;

    for (size_t si = 0; si < sigma_nodes.size(); ++si) {
        double sigma = sigma_nodes[si];
        for (size_t ri = 0; ri < rate_nodes.size(); ++ri) {
            double rate = rate_nodes[ri];
            for (size_t ti = tau_idx_lo; ti <= tau_idx_hi; ++ti) {
                auto* spline = cache.get_slice(sigma, rate, ti);
                if (!spline) continue;
                n_sampled++;

                double tau = tau_nodes[ti];

                // 1. Evaluate EEP at scan points
                std::vector<double> eep(cfg.n_scan_points);
                std::vector<double> xs(cfg.n_scan_points);
                for (size_t k = 0; k < cfg.n_scan_points; ++k) {
                    double x = x_min + (x_max - x_min) * k /
                               (cfg.n_scan_points - 1);
                    xs[k] = x;

                    double am = spline->eval(x) * K_ref;
                    double spot = std::exp(x) * K_ref;
                    auto eu = EuropeanOptionSolver(
                        OptionSpec{.spot = spot, .strike = K_ref,
                                   .maturity = tau, .rate = rate,
                                   .dividend_yield = dividend_yield,
                                   .option_type = option_type},
                        sigma).solve();
                    if (!eu) continue;
                    eep[k] = am - eu->value();
                }

                // 2. Monotone envelope (running max from right to left for puts)
                if (option_type == OptionType::PUT) {
                    for (int k = static_cast<int>(cfg.n_scan_points) - 2; k >= 0; --k) {
                        eep[k] = std::max(eep[k], eep[k + 1]);
                    }
                }

                // 3. Bracket zero-crossing: scan from OTM (right) toward ITM
                int bracket_hi = -1;
                for (int k = static_cast<int>(cfg.n_scan_points) - 1; k >= 0; --k) {
                    if (eep[k] > eps) {
                        bracket_hi = k;
                        break;
                    }
                }
                if (bracket_hi < 0) continue;  // EEP < eps everywhere

                // Find bracket_lo: the point just right of bracket_hi where eep <= eps
                int bracket_lo = bracket_hi;
                for (int k = bracket_hi + 1;
                     k < static_cast<int>(cfg.n_scan_points); ++k) {
                    if (eep[k] <= eps) {
                        bracket_lo = bracket_hi;
                        bracket_hi = k;
                        break;
                    }
                }

                // 4. Linear interpolation for root
                if (bracket_lo != bracket_hi &&
                    std::abs(eep[bracket_lo] - eep[bracket_hi]) > 1e-15) {
                    double alpha = (eps - eep[bracket_hi]) /
                                   (eep[bracket_lo] - eep[bracket_hi]);
                    double x_star = xs[bracket_hi] +
                                    alpha * (xs[bracket_lo] - xs[bracket_hi]);
                    all_x_stars.push_back(x_star);
                } else {
                    all_x_stars.push_back(xs[bracket_hi]);
                }
            }
        }
    }

    // 5. Compute center and half-width
    if (all_x_stars.size() < cfg.valid_fraction * n_sampled) {
        // Fallback chain
        double x_star = std::isfinite(fallback_x_star) ? fallback_x_star : 0.0;
        return {x_star, cfg.delta_min, all_x_stars.size(), n_sampled};
    }

    std::sort(all_x_stars.begin(), all_x_stars.end());
    size_t n = all_x_stars.size();
    double median = all_x_stars[n / 2];

    size_t p10_idx = n / 10;
    size_t p90_idx = std::min(n * 9 / 10, n - 1);
    double spread = all_x_stars[p90_idx] - all_x_stars[p10_idx];
    double delta = std::max(cfg.delta_min, spread / 2.0 + cfg.delta_margin);

    return {median, delta, n, n_sampled};
}

}  // namespace mango
```

**Step 2: Write a smoke test**

Create `tests/boundary_detector_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "boundary_detector.hpp"

using namespace mango;

TEST(BoundaryDetectorTest, EmptyCacheReturnsFallback) {
    PDESliceCache cache;
    std::vector<double> tau = {0.1, 0.5, 1.0};
    std::vector<double> sigma = {0.20};
    std::vector<double> rate = {0.05};

    auto result = detect_exercise_boundary(
        cache, tau, sigma, rate, 0, 2, 100.0, OptionType::PUT);

    // No valid slices → falls back to x*=0.0
    EXPECT_DOUBLE_EQ(result.x_star, 0.0);
    EXPECT_EQ(result.n_valid, 0u);
}
```

**Step 3: Add BUILD rules**

Add to `benchmarks/BUILD.bazel`:
```python
cc_library(
    name = "boundary_detector",
    hdrs = ["boundary_detector.hpp"],
    deps = [
        ":pde_slice_cache",
        "//src/option:european_option",
        "//src/option:option_spec",
        "//src/option/table/dimensionless:chebyshev_nodes",
    ],
    visibility = ["//tests:__pkg__"],
)
```

Add to `tests/BUILD.bazel`:
```python
cc_test(
    name = "boundary_detector_test",
    srcs = ["boundary_detector_test.cc"],
    deps = [
        "//benchmarks:boundary_detector",
        "//benchmarks:pde_slice_cache",
        "@googletest//:gtest_main",
    ],
)
```

**Step 4: Run tests**

Run: `bazel test //tests:boundary_detector_test --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add benchmarks/boundary_detector.hpp tests/boundary_detector_test.cc \
    tests/BUILD.bazel benchmarks/BUILD.bazel
git commit -m "Add exercise boundary detection from cached PDE slices"
```

---

### Task 3: Piecewise Element Builder with τ-Bands

Build the 6-element tensor set: 2 τ-bands × 3 x-elements each, using the PDE cache and boundary detection.

**Files:**
- Create: `benchmarks/piecewise_element_builder.hpp`

**Step 1: Write the builder**

This header provides:
- `PiecewiseElementConfig` — full config for the 6-element build
- `PiecewiseElementSet` — result: 6 `ChebyshevTucker4D` + metadata
- `build_piecewise_elements()` — main builder function

Key design points from the spec:
- σ and r axes use CC-level nodes (shared across all elements, reuses cache)
- τ-Chebyshev nodes per band map to cached snapshots via exact-match-or-bracket-interpolate
- x-nodes are CGL per element with headroom on outer edges
- EEP = spline.eval(x) · K_ref − European(x, τ, σ, r), with softplus floor

```cpp
// SPDX-License-Identifier: MIT
//
// Build 6 piecewise Chebyshev 4D elements: 2 τ-bands × 3 x-elements.
// Reads from PDESliceCache populated by Phase A incremental builder.
#pragma once

#include "pde_slice_cache.hpp"
#include "boundary_detector.hpp"
#include "mango/option/table/dimensionless/chebyshev_tucker_4d.hpp"
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>

namespace mango {

struct ElementSpec {
    double x_lo, x_hi;       // element x-bounds (with headroom on outer edges)
    double tau_lo, tau_hi;    // element τ-bounds
    size_t num_x;             // CGL nodes in x (15 or 25)
    size_t num_tau;           // CGL nodes in τ
    size_t tau_band;          // 0=short, 1=long
};

struct OverlapZone {
    size_t left_idx, right_idx;
    double x_lo, x_hi;
};

struct PiecewiseElementSet {
    std::vector<ChebyshevTucker4D> elements;   // 6 elements
    std::vector<ElementSpec> specs;             // per-element config
    std::vector<OverlapZone> x_overlaps;       // 2 per τ-band = 4 total

    double tau_blend_lo, tau_blend_hi;         // τ-band overlap [55d, 65d]

    // Shared axes bounds (extended)
    double sigma_lo, sigma_hi;
    double rate_lo, rate_hi;

    size_t total_pde_solves;
    double build_seconds;

    // Boundary detection results
    BoundaryResult short_boundary, long_boundary;
};

struct PiecewiseElementBuildConfig {
    // CC levels for σ and r (reuse Phase A cache)
    size_t sigma_level = 4;
    size_t rate_level = 3;

    // Node counts per element type
    size_t num_x_coarse = 15;      // ITM and OTM elements
    size_t num_x_dense = 25;       // boundary element
    size_t num_tau = 9;            // τ nodes per band

    // τ-band boundaries (in years)
    double tau_short_lo = 0.019;   // ~7d
    double tau_short_hi = 60.0 / 365.0;
    double tau_long_lo = 60.0 / 365.0;
    double tau_long_hi = 2.0;

    // τ-band overlap zone (in years)
    double tau_blend_lo = 55.0 / 365.0;
    double tau_blend_hi = 65.0 / 365.0;

    // x-domain bounds (physical)
    double x_min = -0.50;
    double x_max = 0.40;

    // σ, r physical bounds
    double sigma_min = 0.05;
    double sigma_max = 0.50;
    double rate_min = 0.01;
    double rate_max = 0.10;

    // Fixed headroom references (must match Phase A IncrementalBuildConfig)
    size_t sigma_headroom_ref = 15;
    size_t rate_headroom_ref = 9;

    double epsilon = 1e-8;
    bool use_tucker = false;
    double dividend_yield = 0.0;
    bool use_hard_max = true;
    double K_ref = 100.0;
    OptionType option_type = OptionType::PUT;
};

inline PiecewiseElementSet build_piecewise_elements(
    const PiecewiseElementBuildConfig& cfg,
    const PDESliceCache& cache,
    std::span<const double> cache_tau_nodes)
{
    auto t0 = std::chrono::steady_clock::now();

    // ---- 1. Shared axis setup ----
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    size_t n_sigma = (1u << cfg.sigma_level) + 1;
    size_t n_rate = (1u << cfg.rate_level) + 1;

    double hsigma = headroom_fn(cfg.sigma_min, cfg.sigma_max, cfg.sigma_headroom_ref);
    double hrate = headroom_fn(cfg.rate_min, cfg.rate_max, cfg.rate_headroom_ref);

    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi = cfg.rate_max + hrate;

    auto sigma_nodes = cc_level_nodes(cfg.sigma_level, sigma_lo, sigma_hi);
    auto rate_nodes = cc_level_nodes(cfg.rate_level, rate_lo, rate_hi);

    // ---- 2. Detect exercise boundary per τ-band ----
    // Map τ ranges to cache tau_idx ranges
    auto find_tau_idx_range = [&](double tau_lo, double tau_hi)
        -> std::pair<size_t, size_t> {
        size_t lo = 0, hi = cache_tau_nodes.size() - 1;
        for (size_t i = 0; i < cache_tau_nodes.size(); ++i) {
            if (cache_tau_nodes[i] >= tau_lo) { lo = i; break; }
        }
        for (size_t i = cache_tau_nodes.size(); i > 0; --i) {
            if (cache_tau_nodes[i - 1] <= tau_hi) { hi = i - 1; break; }
        }
        return {lo, hi};
    };

    auto [short_tau_lo, short_tau_hi] = find_tau_idx_range(
        cfg.tau_short_lo, cfg.tau_short_hi);
    auto [long_tau_lo, long_tau_hi] = find_tau_idx_range(
        cfg.tau_long_lo, cfg.tau_long_hi);

    // Detect long-τ boundary first (more reliable)
    auto long_boundary = detect_exercise_boundary(
        cache, cache_tau_nodes, sigma_nodes, rate_nodes,
        long_tau_lo, long_tau_hi,
        cfg.K_ref, cfg.option_type, {}, cfg.dividend_yield);

    auto short_boundary = detect_exercise_boundary(
        cache, cache_tau_nodes, sigma_nodes, rate_nodes,
        short_tau_lo, short_tau_hi,
        cfg.K_ref, cfg.option_type, {}, cfg.dividend_yield,
        long_boundary.x_star);  // fallback to long-τ boundary

    // ---- 3. Build element specs: 3 x-elements per τ-band ----
    struct BandInfo {
        double tau_lo, tau_hi;
        BoundaryResult boundary;
    };
    BandInfo bands[] = {
        {cfg.tau_short_lo, cfg.tau_short_hi, short_boundary},
        {cfg.tau_long_lo, cfg.tau_long_hi, long_boundary},
    };

    std::vector<ElementSpec> specs;
    std::vector<OverlapZone> x_overlaps;
    size_t elem_idx = 0;

    for (size_t band = 0; band < 2; ++band) {
        auto& b = bands[band];
        double x_star = b.boundary.x_star;
        double delta = b.boundary.delta;
        double h_boundary = 2.0 * delta / (cfg.num_x_dense - 1);
        double w_overlap = 2.0 * h_boundary;

        // τ headroom
        double htau = headroom_fn(b.tau_lo, b.tau_hi, cfg.num_tau);
        double tau_lo_ext = std::max(b.tau_lo - htau, 1e-4);
        double tau_hi_ext = b.tau_hi + htau;

        // x-element boundaries (physical)
        double itm_hi = x_star - delta;     // ITM/boundary break
        double otm_lo = x_star + delta;     // boundary/OTM break

        // ITM element: [x_min, itm_hi]
        double hx_itm = headroom_fn(cfg.x_min, itm_hi, cfg.num_x_coarse);
        specs.push_back({
            .x_lo = cfg.x_min - hx_itm, .x_hi = itm_hi + w_overlap,
            .tau_lo = tau_lo_ext, .tau_hi = tau_hi_ext,
            .num_x = cfg.num_x_coarse, .num_tau = cfg.num_tau,
            .tau_band = band});

        // Boundary element: [itm_hi, otm_lo] — dense
        specs.push_back({
            .x_lo = itm_hi - w_overlap, .x_hi = otm_lo + w_overlap,
            .tau_lo = tau_lo_ext, .tau_hi = tau_hi_ext,
            .num_x = cfg.num_x_dense, .num_tau = cfg.num_tau,
            .tau_band = band});

        // OTM element: [otm_lo, x_max]
        double hx_otm = headroom_fn(otm_lo, cfg.x_max, cfg.num_x_coarse);
        specs.push_back({
            .x_lo = otm_lo - w_overlap, .x_hi = cfg.x_max + hx_otm,
            .tau_lo = tau_lo_ext, .tau_hi = tau_hi_ext,
            .num_x = cfg.num_x_coarse, .num_tau = cfg.num_tau,
            .tau_band = band});

        // Overlap zones: ITM-Boundary and Boundary-OTM
        x_overlaps.push_back({
            .left_idx = elem_idx, .right_idx = elem_idx + 1,
            .x_lo = itm_hi - w_overlap, .x_hi = itm_hi + w_overlap});
        x_overlaps.push_back({
            .left_idx = elem_idx + 1, .right_idx = elem_idx + 2,
            .x_lo = otm_lo - w_overlap, .x_hi = otm_lo + w_overlap});

        elem_idx += 3;
    }

    // ---- 4. Build per-element tensors from cache ----
    auto map_tau_to_cache = [&](double tau) -> double {
        // Try exact match first
        for (size_t i = 0; i < cache_tau_nodes.size(); ++i) {
            if (std::abs(tau - cache_tau_nodes[i]) < 1e-12)
                return static_cast<double>(i);  // exact index
        }
        // Bracket and return fractional index for interpolation
        for (size_t i = 0; i + 1 < cache_tau_nodes.size(); ++i) {
            if (tau >= cache_tau_nodes[i] && tau <= cache_tau_nodes[i + 1]) {
                double alpha = (tau - cache_tau_nodes[i]) /
                               (cache_tau_nodes[i + 1] - cache_tau_nodes[i]);
                return i + alpha;
            }
        }
        // Clamp
        return tau < cache_tau_nodes[0] ? 0.0
             : static_cast<double>(cache_tau_nodes.size() - 1);
    };

    std::vector<ChebyshevTucker4D> elements;
    elements.reserve(specs.size());

    for (const auto& spec : specs) {
        auto x_nodes = chebyshev_nodes(spec.num_x, spec.x_lo, spec.x_hi);
        auto tau_nodes_elem = chebyshev_nodes(spec.num_tau, spec.tau_lo, spec.tau_hi);

        size_t Nx = spec.num_x;
        size_t Nt = spec.num_tau;
        size_t Ns = n_sigma;
        size_t Nr = n_rate;
        std::vector<double> tensor(Nx * Nt * Ns * Nr, 0.0);

        for (size_t s = 0; s < Ns; ++s) {
            double sigma = sigma_nodes[s];
            for (size_t r = 0; r < Nr; ++r) {
                double rate = rate_nodes[r];
                for (size_t j = 0; j < Nt; ++j) {
                    double tau = tau_nodes_elem[j];
                    double frac_idx = map_tau_to_cache(tau);
                    size_t idx_lo = static_cast<size_t>(frac_idx);
                    size_t idx_hi = std::min(idx_lo + 1, cache_tau_nodes.size() - 1);
                    double alpha = frac_idx - idx_lo;

                    auto* slice_lo = cache.get_slice(sigma, rate, idx_lo);
                    auto* slice_hi = (idx_lo != idx_hi)
                        ? cache.get_slice(sigma, rate, idx_hi)
                        : slice_lo;

                    if (!slice_lo) continue;

                    for (size_t i = 0; i < Nx; ++i) {
                        double x = x_nodes[i];

                        // Interpolate American price from cache
                        double am_lo = slice_lo->eval(x) * cfg.K_ref;
                        double am = am_lo;
                        if (alpha > 1e-12 && slice_hi && slice_hi != slice_lo) {
                            double am_hi = slice_hi->eval(x) * cfg.K_ref;
                            am = (1.0 - alpha) * am_lo + alpha * am_hi;
                        }

                        // European price
                        double spot = std::exp(x) * cfg.K_ref;
                        auto eu = EuropeanOptionSolver(
                            OptionSpec{.spot = spot, .strike = cfg.K_ref,
                                       .maturity = tau, .rate = rate,
                                       .dividend_yield = cfg.dividend_yield,
                                       .option_type = cfg.option_type},
                            sigma).solve();
                        if (!eu) continue;

                        double eep_raw = am - eu->value();

                        // Softplus floor
                        constexpr double kSharpness = 100.0;
                        double eep;
                        if (kSharpness * eep_raw > 500.0) {
                            eep = eep_raw;
                        } else {
                            double softplus = std::log1p(std::exp(
                                kSharpness * eep_raw)) / kSharpness;
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

        ChebyshevTucker4DDomain dom{
            .bounds = {{{spec.x_lo, spec.x_hi}, {spec.tau_lo, spec.tau_hi},
                        {sigma_lo, sigma_hi}, {rate_lo, rate_hi}}}};
        ChebyshevTucker4DConfig tcfg{
            .num_pts = {Nx, Nt, Ns, Nr},
            .epsilon = cfg.epsilon,
            .use_tucker = cfg.use_tucker};

        elements.push_back(
            ChebyshevTucker4D::build_from_values(tensor, dom, tcfg));
    }

    auto t1 = std::chrono::steady_clock::now();

    return {
        .elements = std::move(elements),
        .specs = std::move(specs),
        .x_overlaps = std::move(x_overlaps),
        .tau_blend_lo = cfg.tau_blend_lo,
        .tau_blend_hi = cfg.tau_blend_hi,
        .sigma_lo = sigma_lo, .sigma_hi = sigma_hi,
        .rate_lo = rate_lo, .rate_hi = rate_hi,
        .total_pde_solves = cache.total_pde_solves(),
        .build_seconds = std::chrono::duration<double>(t1 - t0).count(),
        .short_boundary = short_boundary,
        .long_boundary = long_boundary,
    };
}

}  // namespace mango
```

**Step 2: Add BUILD rule**

Add to `benchmarks/BUILD.bazel`:
```python
cc_library(
    name = "piecewise_element_builder",
    hdrs = ["piecewise_element_builder.hpp"],
    deps = [
        ":pde_slice_cache",
        ":boundary_detector",
        "//src/option/table/dimensionless:chebyshev_tucker_4d",
        "//src/option/table/dimensionless:chebyshev_nodes",
        "//src/option:european_option",
        "//src/option:option_spec",
    ],
    visibility = ["//tests:__pkg__"],
)
```

**Step 3: Verify compilation**

Run: `bazel build //benchmarks:piecewise_element_builder`
Expected: BUILD SUCCESS

**Step 4: Commit**

```bash
git add benchmarks/piecewise_element_builder.hpp benchmarks/BUILD.bazel
git commit -m "Add piecewise element builder with τ-bands and boundary detection"
```

---

### Task 4: Piecewise Evaluator with Overlap Blending

Query-time evaluation: map (x, τ) to element(s), blend overlapping evaluations with C∞ bump weights.

**Files:**
- Create: `benchmarks/piecewise_evaluator.hpp`
- Create: `tests/piecewise_evaluator_test.cc`

**Step 1: Write the evaluator**

```cpp
// SPDX-License-Identifier: MIT
//
// Piecewise Chebyshev 4D evaluator with C∞ overlap blending in (x, τ).
#pragma once

#include "piecewise_element_builder.hpp"
#include "bump_blend.hpp"
#include "mango/option/table/price_query.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

#include <cmath>
#include <vector>

namespace mango {

class PiecewiseBlendedEvaluator {
public:
    PiecewiseBlendedEvaluator(PiecewiseElementSet elements,
                               OptionType type, double K_ref,
                               double dividend_yield)
        : elems_(std::move(elements)), type_(type),
          K_ref_(K_ref), dividend_yield_(dividend_yield) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        double eep = eval_blended(x, q.tau, q.sigma, q.rate);
        auto eu = european(q);
        return eep * (q.strike / K_ref_) + eu;
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        double eep_vega = vega_blended(x, q.tau, q.sigma, q.rate);

        auto eu_result = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return (q.strike / K_ref_) * eep_vega + eu_result.vega();
    }

    const PiecewiseElementSet& element_set() const { return elems_; }

private:
    [[nodiscard]] double european(const PriceQuery& q) const {
        return EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value().value();
    }

    /// Evaluate single element.
    [[nodiscard]] double eval_element(size_t idx, double x, double tau,
                                       double sigma, double rate) const {
        return elems_.elements[idx].eval({x, tau, sigma, rate});
    }

    /// Evaluate partial derivative w.r.t. sigma (axis 2) for single element.
    [[nodiscard]] double partial_sigma_element(
        size_t idx, double x, double tau, double sigma, double rate) const {
        return elems_.elements[idx].partial(2, {x, tau, sigma, rate});
    }

    /// Find elements for a given (x, tau_band).
    /// Returns {elem_idx, weight} pairs.  1 or 2 entries.
    struct WeightedElement { size_t idx; double weight; };

    [[nodiscard]] std::vector<WeightedElement>
    find_elements(double x, size_t band) const {
        // Elements for this band: band*3, band*3+1, band*3+2
        size_t base = band * 3;

        // Check overlap zones
        for (const auto& oz : elems_.x_overlaps) {
            if (oz.left_idx / 3 != band) continue;  // wrong band
            if (x >= oz.x_lo && x <= oz.x_hi) {
                double w_right = overlap_weight_right(x, oz.x_lo, oz.x_hi);
                return {{oz.left_idx, 1.0 - w_right},
                        {oz.right_idx, w_right}};
            }
        }

        // Not in any overlap — find the single element that contains x
        for (size_t i = 0; i < 3; ++i) {
            size_t idx = base + i;
            const auto& spec = elems_.specs[idx];
            if (x >= spec.x_lo && x <= spec.x_hi) {
                return {{idx, 1.0}};
            }
        }

        // Fallback: closest element
        size_t closest = base;
        double best_dist = 1e99;
        for (size_t i = 0; i < 3; ++i) {
            size_t idx = base + i;
            const auto& spec = elems_.specs[idx];
            double mid = (spec.x_lo + spec.x_hi) / 2.0;
            double dist = std::abs(x - mid);
            if (dist < best_dist) { best_dist = dist; closest = idx; }
        }
        return {{closest, 1.0}};
    }

    /// Evaluate EEP with full (x, τ) blending.
    [[nodiscard]] double eval_blended(double x, double tau,
                                       double sigma, double rate) const {
        auto eval_band = [&](size_t band) -> double {
            auto elems = find_elements(x, band);
            double result = 0.0;
            for (const auto& [idx, w] : elems) {
                result += w * eval_element(idx, x, tau, sigma, rate);
            }
            return result;
        };

        // τ-band blending
        if (tau < elems_.tau_blend_lo) {
            return eval_band(0);  // short only
        }
        if (tau > elems_.tau_blend_hi) {
            return eval_band(1);  // long only
        }

        // Blend between τ-bands
        double w_long = overlap_weight_right(
            tau, elems_.tau_blend_lo, elems_.tau_blend_hi);
        return (1.0 - w_long) * eval_band(0) + w_long * eval_band(1);
    }

    /// Evaluate EEP vega with full (x, τ) blending.
    [[nodiscard]] double vega_blended(double x, double tau,
                                       double sigma, double rate) const {
        auto vega_band = [&](size_t band) -> double {
            auto elems = find_elements(x, band);
            double result = 0.0;
            for (const auto& [idx, w] : elems) {
                result += w * partial_sigma_element(idx, x, tau, sigma, rate);
            }
            return result;
        };

        if (tau < elems_.tau_blend_lo) return vega_band(0);
        if (tau > elems_.tau_blend_hi) return vega_band(1);

        double w_long = overlap_weight_right(
            tau, elems_.tau_blend_lo, elems_.tau_blend_hi);
        return (1.0 - w_long) * vega_band(0) + w_long * vega_band(1);
    }

    PiecewiseElementSet elems_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

}  // namespace mango
```

**Step 2: Add BUILD rule**

Add to `benchmarks/BUILD.bazel`:
```python
cc_library(
    name = "piecewise_evaluator",
    hdrs = ["piecewise_evaluator.hpp"],
    deps = [
        ":piecewise_element_builder",
        ":bump_blend",
        "//src/option/table:price_query",
        "//src/option:european_option",
        "//src/option:option_spec",
    ],
    visibility = ["//tests:__pkg__"],
)
```

**Step 3: Verify compilation**

Run: `bazel build //benchmarks:piecewise_evaluator`
Expected: BUILD SUCCESS

**Step 4: Commit**

```bash
git add benchmarks/piecewise_evaluator.hpp benchmarks/BUILD.bazel
git commit -m "Add piecewise blended evaluator with C∞ overlap weights"
```

---

### Task 5: Benchmark Section — cheb4d-piecewise

Wire the piecewise builder+evaluator into `interp_iv_safety.cc` as a new `cheb4d-piecewise` section.

**Files:**
- Modify: `benchmarks/interp_iv_safety.cc`

**Step 1: Add include and builder**

Add to includes (after line 51):
```cpp
#include "piecewise_evaluator.hpp"
```

Add a new builder function (before `main()`):
```cpp
static PiecewiseBlendedEvaluator build_cheb4d_piecewise() {
    using namespace mango;

    // 1. Populate PDE cache with L(4,3): 17σ × 9r = 153 PDE
    PDESliceCache cache;
    IncrementalBuildConfig inc_cfg;
    inc_cfg.num_x = 40;
    inc_cfg.num_tau = 15;
    inc_cfg.sigma_level = 4;
    inc_cfg.rate_level = 3;
    inc_cfg.use_tucker = false;
    inc_cfg.dividend_yield = 0.0;

    auto inc_result = build_chebyshev_4d_eep_incremental(
        inc_cfg, cache, kSpot, OptionType::PUT);

    std::printf("  Phase A: %zu PDE solves, %.1fs\n",
                inc_result.new_pde_solves, inc_result.build_seconds);

    // Reconstruct tau_nodes (must match what incremental builder used)
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double htau = headroom_fn(inc_cfg.tau_min, inc_cfg.tau_max, inc_cfg.num_tau);
    double tau_lo = std::max(inc_cfg.tau_min - htau, 1e-4);
    double tau_hi = inc_cfg.tau_max + htau;
    auto tau_nodes = chebyshev_nodes(inc_cfg.num_tau, tau_lo, tau_hi);

    // 2. Build piecewise elements
    PiecewiseElementBuildConfig pw_cfg;
    pw_cfg.sigma_level = 4;
    pw_cfg.rate_level = 3;
    pw_cfg.sigma_headroom_ref = inc_cfg.sigma_headroom_ref;
    pw_cfg.rate_headroom_ref = inc_cfg.rate_headroom_ref;
    pw_cfg.K_ref = kSpot;
    pw_cfg.option_type = OptionType::PUT;
    pw_cfg.dividend_yield = 0.0;

    auto element_set = build_piecewise_elements(pw_cfg, cache, tau_nodes);

    std::printf("  Phase C: %zu elements, %.1fs build\n",
                element_set.elements.size(), element_set.build_seconds);
    std::printf("  Short boundary: x*=%.3f, δ=%.3f (%zu/%zu valid)\n",
                element_set.short_boundary.x_star,
                element_set.short_boundary.delta,
                element_set.short_boundary.n_valid,
                element_set.short_boundary.n_sampled);
    std::printf("  Long boundary:  x*=%.3f, δ=%.3f (%zu/%zu valid)\n",
                element_set.long_boundary.x_star,
                element_set.long_boundary.delta,
                element_set.long_boundary.n_valid,
                element_set.long_boundary.n_sampled);

    return PiecewiseBlendedEvaluator(
        std::move(element_set), OptionType::PUT, kSpot, 0.0);
}
```

**Step 2: Add section runner**

```cpp
static void run_cheb4d_piecewise() {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev 4D Piecewise (Phase C) — 2 τ-bands × 3 x-elements\n");
    std::printf("================================================================\n\n");

    const auto& q0_prices = get_q0_prices();

    std::printf("--- Building piecewise Chebyshev 4D...\n");
    auto evaluator = build_cheb4d_piecewise();

    std::printf("\n--- Computing IV errors (Brent)...\n");
    std::array<ErrorTable, kNV> all_errors;
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Piecewise Cheb 4D (Phase C) IV Error — σ=%.0f%%",
                      kVols[vi] * 100);
        all_errors[vi] = compute_errors_brent(q0_prices,
            [&](double S, double K, double tau, double sigma, double r) {
                PriceQuery q{.spot = S, .strike = K, .tau = tau,
                             .sigma = sigma, .rate = r};
                return evaluator.price(q);
            }, vi);
        print_heatmap(title, all_errors[vi]);
    }
    print_stratified_stats("piecewise", all_errors);
}
```

**Step 3: Register section**

Add `"cheb4d-piecewise"` to `kSections[]` array and add to `main()`:
```cpp
if (want("cheb4d-piecewise")) run_cheb4d_piecewise();
```

**Step 4: Add deps to BUILD.bazel**

Add `":piecewise_evaluator"` to `interp_iv_safety`'s deps in `benchmarks/BUILD.bazel`.

**Step 5: Verify compilation**

Run: `bazel build //benchmarks:interp_iv_safety`
Expected: BUILD SUCCESS

**Step 6: Run benchmark**

Run: `bazel run //benchmarks:interp_iv_safety -- cheb4d-piecewise`
Expected: Heatmaps and stratified stats printed. Check success gates from design doc Section 7.

**Step 7: Commit**

```bash
git add benchmarks/interp_iv_safety.cc benchmarks/BUILD.bazel
git commit -m "Add cheb4d-piecewise benchmark section for Phase C"
```

---

### Task 6: Shape Constraints (Post-MVP)

Implement per-element shape constraints: EEP non-negativity and put-monotonicity projection. This task should only be done after Task 5's benchmark confirms the piecewise structure works and identifies remaining issues.

**Files:**
- Create: `benchmarks/shape_constraints.hpp`

**Step 1: Write the constraint projector**

Implement:
1. `project_nonnegative()` — evaluate element on dense grid, clip negatives to zero, re-fit via `build_from_values()`
2. `project_monotone_put()` — check ∂EEP/∂x via finite diff, apply running-max envelope, re-fit
3. `apply_shape_constraints()` — orchestrates both, respects overlap boundary freeze (pin nodes to shared average)

The overlap boundary freeze (Section 4 of design):
- Before projection: evaluate both adjacent elements at x-nodes in overlap zone
- Compute shared target: `f_pin = (f_left + f_right) / 2`
- During refit: constrain overlap x-nodes to match pinned values
- After projection: run interface consistency gate (value + derivative mismatch check)

**Step 2: Wire into piecewise builder**

Add an optional `apply_shape_constraints` flag to `PiecewiseElementBuildConfig`. When true, call the projector after building element tensors (between steps 4 and 5 of `build_piecewise_elements()`).

**Step 3: Verify**

Run: `bazel run //benchmarks:interp_iv_safety -- cheb4d-piecewise`
Expected: Same or better results (fewer `---` failures from negative EEP)

**Step 4: Commit**

```bash
git add benchmarks/shape_constraints.hpp benchmarks/piecewise_element_builder.hpp \
    benchmarks/BUILD.bazel
git commit -m "Add shape constraint projection for piecewise elements"
```

---

### Task 7: Verification Gates

Run the full IV safety benchmark and verify Phase C meets the success gates from the design doc.

**Step 1: Run piecewise benchmark**

Run: `bazel run //benchmarks:interp_iv_safety -- cheb4d-piecewise`

**Step 2: Check success gates**

| Metric | Target |
|--------|--------|
| 60d-180d, σ=15% p95 | <100 bps |
| T<60d, σ=30% p95 | <80 bps |
| T>=1y, σ=30% p95 | ≤2 bps (no regression) |
| Solve success rate | ≥130/144 |

**Step 3: Run comparison**

Run: `bazel run //benchmarks:interp_iv_safety -- cheb4d-incremental cheb4d-piecewise`
Expected: Side-by-side comparison showing Phase C improvement over global polynomial.

**Step 4: Run full test suite**

Run: `bazel test //...`
Expected: All tests pass, no regressions.

Run: `bazel build //benchmarks/...`
Expected: All benchmarks compile.

**Step 5: Commit results summary**

If gates pass, commit a note to the design doc or update GitHub issue #353.

---

## File Summary

| File | Action | Task |
|------|--------|------|
| `benchmarks/bump_blend.hpp` | Create | 1 |
| `tests/bump_blend_test.cc` | Create | 1 |
| `benchmarks/boundary_detector.hpp` | Create | 2 |
| `tests/boundary_detector_test.cc` | Create | 2 |
| `benchmarks/piecewise_element_builder.hpp` | Create | 3 |
| `benchmarks/piecewise_evaluator.hpp` | Create | 4 |
| `benchmarks/interp_iv_safety.cc` | Modify | 5 |
| `benchmarks/BUILD.bazel` | Modify | 1,2,3,4,5 |
| `tests/BUILD.bazel` | Modify | 1,2 |
| `benchmarks/shape_constraints.hpp` | Create | 6 |
