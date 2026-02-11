# Chebyshev Surface Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire `ChebyshevInterpolant<4, TuckerTensor<4>>` into the pluggable surface
architecture as a drop-in B-spline alternative, with a PDE-sampling builder and
accuracy benchmark.

**Architecture:** No adapter needed — `ChebyshevInterpolant<N, Storage>` already
satisfies `SurfaceInterpolant`. Compose with existing `EEPSurfaceAdapter`,
`StandardTransform4D`, and `AnalyticalEEP`. Builder samples PDE at CGL nodes.

**Tech Stack:** C++23, Bazel, GoogleTest, BatchAmericanOptionSolver, EuropeanOptionSolver

---

### Task 1: Type Aliases and Surface Header

**Files:**
- Create: `src/option/table/chebyshev/chebyshev_surface.hpp`
- Create: `src/option/table/chebyshev/BUILD.bazel`
- Test: `tests/chebyshev_surface_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/chebyshev_surface_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/price_surface_concept.hpp"
#include "mango/option/table/surface_concepts.hpp"

using namespace mango;

// Static assertions: ChebyshevInterpolant satisfies SurfaceInterpolant
static_assert(SurfaceInterpolant<ChebyshevInterpolant<4, TuckerTensor<4>>, 4>);
static_assert(SurfaceInterpolant<ChebyshevInterpolant<4, RawTensor<4>>, 4>);

// ChebyshevSurface satisfies PriceSurface
static_assert(PriceSurface<ChebyshevSurface>);
static_assert(PriceSurface<ChebyshevRawSurface>);

TEST(ChebyshevSurfaceTest, ConstructAndQuery) {
    // Build a trivial 4D Chebyshev interpolant (constant function = 0.05)
    // to verify the composition chain works end-to-end.
    Domain<4> domain{
        .lo = {-0.5, 0.01, 0.05, 0.01},
        .hi = { 0.5, 2.00, 0.50, 0.10},
    };
    std::array<size_t, 4> num_pts = {5, 5, 5, 5};

    auto interp = ChebyshevInterpolant<4, TuckerTensor<4>>::build(
        [](std::array<double, 4>) { return 0.05; },
        domain, num_pts, 1e-8);

    // Compose: interpolant -> EEPSurfaceAdapter -> PriceTable
    ChebyshevLeaf leaf(
        std::move(interp),
        StandardTransform4D{},
        AnalyticalEEP(OptionType::PUT, 0.02),
        100.0);  // K_ref

    SurfaceBounds bounds{
        .m_min = -0.5, .m_max = 0.5,
        .tau_min = 0.01, .tau_max = 2.0,
        .sigma_min = 0.05, .sigma_max = 0.50,
        .rate_min = 0.01, .rate_max = 0.10,
    };

    ChebyshevSurface surface(std::move(leaf), bounds, OptionType::PUT, 0.02);

    // price() should return a valid positive number for ATM put
    double p = surface.price(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(p, 0.0);
    EXPECT_LT(p, 50.0);  // Reasonable put price

    // vega should be positive
    double v = surface.vega(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(v, 0.0);
}
```

Add to `tests/BUILD.bazel`:

```bazel
cc_test(
    name = "chebyshev_surface_test",
    size = "small",
    srcs = ["chebyshev_surface_test.cc"],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
    deps = [
        "//src/option/table/chebyshev:chebyshev_surface",
        "//src/option/table:price_surface_concept",
        "//src/option/table:surface_concepts",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:chebyshev_surface_test --test_output=all`
Expected: FAIL (header doesn't exist yet)

**Step 3: Write the header and BUILD target**

Create `src/option/table/chebyshev/chebyshev_surface.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/math/chebyshev/tucker_tensor.hpp"
#include "mango/option/table/bounded_surface.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep_surface_adapter.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"

namespace mango {

// Chebyshev leaf: Tucker-compressed 4D interpolant + EEP decomposition
using ChebyshevLeaf = EEPSurfaceAdapter<
    ChebyshevInterpolant<4, TuckerTensor<4>>,
    StandardTransform4D, AnalyticalEEP>;

// Full surface with bounds metadata (satisfies PriceSurface)
using ChebyshevSurface = PriceTable<ChebyshevLeaf>;

// Raw (uncompressed) variant
using ChebyshevRawLeaf = EEPSurfaceAdapter<
    ChebyshevInterpolant<4, RawTensor<4>>,
    StandardTransform4D, AnalyticalEEP>;

using ChebyshevRawSurface = PriceTable<ChebyshevRawLeaf>;

}  // namespace mango
```

Create `src/option/table/chebyshev/BUILD.bazel`:

```bazel
# SPDX-License-Identifier: MIT
cc_library(
    name = "chebyshev_surface",
    hdrs = ["chebyshev_surface.hpp"],
    deps = [
        "//src/math/chebyshev:chebyshev_interpolant",
        "//src/math/chebyshev:raw_tensor",
        "//src/math/chebyshev:tucker_tensor",
        "//src/option/table:bounded_surface",
        "//src/option/table:analytical_eep",
        "//src/option/table:eep_surface_adapter",
        "//src/option/table:standard_transform_4d",
    ],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option/table/chebyshev",
    include_prefix = "mango/option/table/chebyshev",
)
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:chebyshev_surface_test --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/chebyshev/ tests/chebyshev_surface_test.cc tests/BUILD.bazel
git commit -m "Add ChebyshevSurface type aliases"
```

---

### Task 2: Chebyshev Table Builder

**Files:**
- Create: `src/option/table/chebyshev/chebyshev_table_builder.hpp`
- Create: `src/option/table/chebyshev/chebyshev_table_builder.cpp`
- Modify: `src/option/table/chebyshev/BUILD.bazel`
- Modify: `tests/chebyshev_surface_test.cc`

**Step 1: Write the failing test**

Add to `tests/chebyshev_surface_test.cc`:

```cpp
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"

TEST(ChebyshevTableBuilderTest, BuildSucceeds) {
    ChebyshevTableConfig config{
        .num_pts = {12, 8, 8, 5},
        .domain = Domain<4>{
            .lo = {-0.30, 0.02, 0.05, 0.01},
            .hi = { 0.30, 2.00, 0.50, 0.10},
        },
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .tucker_epsilon = 1e-8,
    };

    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value()) << "Builder should succeed";
    EXPECT_GT(result->n_pde_solves, 0u);
    EXPECT_GT(result->build_seconds, 0.0);

    // Query the surface at ATM
    double p = result->surface.price(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(p, 0.0);
    EXPECT_LT(p, 50.0);
}

TEST(ChebyshevTableBuilderTest, IVRoundTrip) {
    // Build surface, price an option, recover IV via Brent
    ChebyshevTableConfig config{
        .num_pts = {16, 10, 10, 6},
        .domain = Domain<4>{
            .lo = {-0.40, 0.02, 0.05, 0.01},
            .hi = { 0.40, 2.00, 0.50, 0.10},
        },
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .tucker_epsilon = 1e-8,
    };

    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value());

    // Get FDM reference price at sigma=0.20
    PricingParams ref_params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);
    auto ref = solve_american_option(ref_params);
    ASSERT_TRUE(ref.has_value());

    // Chebyshev price should be close to FDM
    double cheb_price = result->surface.price(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_NEAR(cheb_price, ref->value(), 0.05);  // within $0.05
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:chebyshev_surface_test --test_output=all`
Expected: FAIL (builder header doesn't exist)

**Step 3: Write the builder header**

Create `src/option/table/chebyshev/chebyshev_table_builder.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/price_table_builder.hpp"

#include <array>
#include <expected>

namespace mango {

struct ChebyshevTableConfig {
    std::array<size_t, 4> num_pts;   // CGL nodes: (m, tau, sigma, rate)
    Domain<4> domain;                // Axis bounds
    double K_ref;
    OptionType option_type;
    double dividend_yield = 0.0;
    double tucker_epsilon = 1e-8;    // 0 = use RawTensor
};

struct ChebyshevTableResult {
    ChebyshevSurface surface;
    size_t n_pde_solves;
    double build_seconds;
};

[[nodiscard]] std::expected<ChebyshevTableResult, PriceTableError>
build_chebyshev_table(const ChebyshevTableConfig& config);

}  // namespace mango
```

**Step 4: Write the builder implementation**

Create `src/option/table/chebyshev/chebyshev_table_builder.cpp`:

The build pipeline:
1. Generate CGL nodes for each of the 4 axes using `chebyshev_nodes()`
2. Create one `PricingParams` per (sigma, rate) pair:
   - spot = K_ref, strike = K_ref, option_type, dividend_yield
   - Solve with `BatchAmericanOptionSolver`:
     - `set_grid_accuracy(GridAccuracyParams{})` (default/Auto)
     - `set_snapshot_times(tau_nodes)` — snapshot at each tau CGL node
     - `solve_batch(params, /*use_shared_grid=*/true)`
3. For each (sigma_idx, rate_idx) result:
   - For each tau snapshot: build `CubicSplineSolver` from PDE spatial solution
   - For each (m_idx, tau_idx): evaluate American price via spline at moneyness node
4. Compute EEP: for each (m, tau, sigma, rate) node:
   - `spot_node = K_ref * exp(m)` where m is the log-moneyness CGL node
   - `eu = EuropeanOptionSolver(spot_node, K_ref, tau, sigma, rate, ...).solve()`
   - `eep = am_price / K_ref - eu.value() / K_ref` (normalized, then scaled by K_ref)
   - Apply softplus floor: `eep = softplus(eep)` for non-negativity
5. Build `ChebyshevInterpolant<4, TuckerTensor<4>>::build_from_values(eep_values, domain, num_pts, tucker_epsilon)`
6. Wrap in `EEPSurfaceAdapter` + `PriceTable`

Key implementation details:
- The batch solver groups by (sigma, rate) — one PDE per unique pair
- n_sigma * n_rate PDE solves total (e.g., 8*5 = 40 PDEs)
- Each PDE gives snapshots at all tau CGL nodes
- Cubic spline interpolates PDE spatial grid at each m CGL node
- The result is a flat array of `n_m * n_tau * n_sigma * n_rate` EEP values

```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"

#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/math/black_scholes_analytics.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include "mango/option/american_option_batch.hpp"

#include <chrono>
#include <cmath>

namespace mango {

std::expected<ChebyshevTableResult, PriceTableError>
build_chebyshev_table(const ChebyshevTableConfig& config) {
    auto t0 = std::chrono::steady_clock::now();

    const size_t n_m     = config.num_pts[0];
    const size_t n_tau   = config.num_pts[1];
    const size_t n_sigma = config.num_pts[2];
    const size_t n_rate  = config.num_pts[3];

    // Generate CGL nodes per axis
    auto m_nodes     = chebyshev_nodes(n_m,     config.domain.lo[0], config.domain.hi[0]);
    auto tau_nodes   = chebyshev_nodes(n_tau,   config.domain.lo[1], config.domain.hi[1]);
    auto sigma_nodes = chebyshev_nodes(n_sigma, config.domain.lo[2], config.domain.hi[2]);
    auto rate_nodes  = chebyshev_nodes(n_rate,  config.domain.lo[3], config.domain.hi[3]);

    // Build batch: one PricingParams per (sigma, rate) pair
    std::vector<PricingParams> batch;
    batch.reserve(n_sigma * n_rate);
    for (size_t si = 0; si < n_sigma; ++si) {
        for (size_t ri = 0; ri < n_rate; ++ri) {
            PricingParams p;
            p.spot = config.K_ref;
            p.strike = config.K_ref;
            p.maturity = config.domain.hi[1];  // max tau
            p.rate = rate_nodes[ri];
            p.dividend_yield = config.dividend_yield;
            p.option_type = config.option_type;
            p.volatility = sigma_nodes[si];
            batch.push_back(p);
        }
    }

    // Solve batch with snapshots at tau CGL nodes
    BatchAmericanOptionSolver solver;
    solver.set_snapshot_times(std::span<const double>(tau_nodes));
    auto batch_result = solver.solve_batch(
        std::span<const PricingParams>(batch), /*use_shared_grid=*/true);

    size_t n_pde_solves = batch_result.n_solves;

    // Extract EEP values at all CGL nodes
    // Tensor layout: values[m_idx * (n_tau * n_sigma * n_rate)
    //                     + tau_idx * (n_sigma * n_rate)
    //                     + sigma_idx * n_rate
    //                     + rate_idx]
    size_t total = n_m * n_tau * n_sigma * n_rate;
    std::vector<double> eep_values(total);

    for (size_t si = 0; si < n_sigma; ++si) {
        for (size_t ri = 0; ri < n_rate; ++ri) {
            size_t batch_idx = si * n_rate + ri;
            const auto& res = batch_result.results[batch_idx];

            if (!res.has_value()) {
                return std::unexpected(PriceTableError{
                    PriceTableErrorCode::ExtractionFailed,
                    static_cast<int>(batch_idx), 0});
            }

            // For each tau snapshot, build cubic spline and evaluate at m nodes
            for (size_t ti = 0; ti < n_tau; ++ti) {
                // Get PDE solution at this snapshot
                auto solution = res->at_time(ti);
                auto grid_x = res->grid_points();

                // Build cubic spline from PDE spatial solution
                CubicSplineSolver<double> spline;
                spline.build(grid_x, solution);

                for (size_t mi = 0; mi < n_m; ++mi) {
                    double m = m_nodes[mi];
                    double spot_node = config.K_ref * std::exp(m);

                    // American price from spline (PDE solution is in spot-space)
                    double am_price = spline.eval(spot_node);

                    // European price (Black-Scholes)
                    double tau = tau_nodes[ti];
                    double sigma = sigma_nodes[si];
                    double rate = rate_nodes[ri];

                    EuropeanOptionSolver eu_solver(
                        OptionSpec{
                            .spot = spot_node,
                            .strike = config.K_ref,
                            .maturity = tau,
                            .rate = rate,
                            .dividend_yield = config.dividend_yield,
                            .option_type = config.option_type},
                        sigma);
                    auto eu = eu_solver.solve();

                    // EEP = (American - European) / K_ref
                    double eep = 0.0;
                    if (eu.has_value()) {
                        eep = (am_price - eu->value()) / config.K_ref;
                    }

                    // Softplus floor for non-negativity
                    // softplus(x) = ln(1 + exp(x * scale)) / scale
                    constexpr double kScale = 200.0;
                    eep = std::log1p(std::exp(eep * kScale)) / kScale;

                    size_t flat = mi * (n_tau * n_sigma * n_rate)
                                + ti * (n_sigma * n_rate)
                                + si * n_rate
                                + ri;
                    eep_values[flat] = eep;
                }
            }
        }
    }

    // Build Chebyshev interpolant from EEP values
    auto interp = ChebyshevInterpolant<4, TuckerTensor<4>>::build_from_values(
        std::span<const double>(eep_values),
        config.domain, config.num_pts, config.tucker_epsilon);

    // Wrap in EEPSurfaceAdapter + PriceTable
    ChebyshevLeaf leaf(
        std::move(interp),
        StandardTransform4D{},
        AnalyticalEEP(config.option_type, config.dividend_yield),
        config.K_ref);

    SurfaceBounds bounds{
        .m_min = config.domain.lo[0], .m_max = config.domain.hi[0],
        .tau_min = config.domain.lo[1], .tau_max = config.domain.hi[1],
        .sigma_min = config.domain.lo[2], .sigma_max = config.domain.hi[2],
        .rate_min = config.domain.lo[3], .rate_max = config.domain.hi[3],
    };

    ChebyshevSurface surface(
        std::move(leaf), bounds, config.option_type, config.dividend_yield);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    return ChebyshevTableResult{
        .surface = std::move(surface),
        .n_pde_solves = n_pde_solves,
        .build_seconds = elapsed,
    };
}

}  // namespace mango
```

Add to `src/option/table/chebyshev/BUILD.bazel`:

```bazel
cc_library(
    name = "chebyshev_table_builder",
    srcs = ["chebyshev_table_builder.cpp"],
    hdrs = ["chebyshev_table_builder.hpp"],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
    deps = [
        ":chebyshev_surface",
        "//src/math/chebyshev:chebyshev_nodes",
        "//src/math:black_scholes_analytics",
        "//src/math:cubic_spline_solver",
        "//src/option:american_option_batch",
    ],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option/table/chebyshev",
    include_prefix = "mango/option/table/chebyshev",
)
```

Update `tests/BUILD.bazel` — add `chebyshev_table_builder` dep to test:

```bazel
cc_test(
    name = "chebyshev_surface_test",
    size = "medium",
    srcs = ["chebyshev_surface_test.cc"],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
    deps = [
        "//src/option:american_option",
        "//src/option/table/chebyshev:chebyshev_surface",
        "//src/option/table/chebyshev:chebyshev_table_builder",
        "//src/option/table:price_surface_concept",
        "//src/option/table:surface_concepts",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:chebyshev_surface_test --test_output=all`
Expected: PASS (both BuildSucceeds and IVRoundTrip)

**Step 6: Commit**

```bash
git add src/option/table/chebyshev/ tests/chebyshev_surface_test.cc tests/BUILD.bazel
git commit -m "Add ChebyshevTableBuilder for 4D EEP surfaces"
```

---

### Task 3: Benchmark Integration

**Files:**
- Modify: `benchmarks/interp_iv_safety.cc`
- Modify: `benchmarks/BUILD.bazel`

**Step 1: Add Chebyshev benchmark section**

Add includes to `benchmarks/interp_iv_safety.cc` (after existing includes):

```cpp
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/math/root_finding.hpp"
```

Add new functions before `main()`:

```cpp
// ============================================================================
// Chebyshev 4D
// ============================================================================

static ChebyshevSurface build_chebyshev_surface() {
    ChebyshevTableConfig config{
        .num_pts = {20, 12, 12, 8},
        .domain = Domain<4>{
            .lo = {-0.50, 0.01, 0.05, 0.01},
            .hi = { 0.40, 2.50, 0.50, 0.10},
        },
        .K_ref = kSpot,
        .option_type = OptionType::PUT,
        .dividend_yield = kDivYield,
        .tucker_epsilon = 1e-8,
    };

    auto result = build_chebyshev_table(config);
    if (!result.has_value()) {
        std::fprintf(stderr, "Chebyshev build failed\n");
        std::exit(1);
    }

    std::printf("  PDE solves: %zu\n", result->n_pde_solves);
    std::printf("  Build time: %.2f s\n", result->build_seconds);
    std::printf("  Compressed size: %zu doubles\n",
                result->surface.inner().interpolant().compressed_size());

    return std::move(result->surface);
}

static ErrorTable compute_errors_chebyshev(
    const PriceGrid& prices,
    const ChebyshevSurface& surface,
    size_t vol_idx) {
    ErrorTable errors{};
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    // Batch FDM IV
    std::vector<IVQuery> queries;
    std::vector<std::pair<size_t, size_t>> query_map;
    queries.reserve(kNT * kNS);

    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < kNS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) {
                errors[ti][si] = std::nan("");
                continue;
            }

            IVQuery q;
            q.spot = kSpot;
            q.strike = kStrikes[si];
            q.maturity = kMaturities[ti];
            q.rate = kRate;
            q.dividend_yield = kDivYield;
            q.option_type = OptionType::PUT;
            q.market_price = price;
            queries.push_back(q);
            query_map.emplace_back(ti, si);
        }
    }

    auto fdm_results = fdm_solver.solve_batch(queries);

    // Chebyshev IV via Brent
    for (size_t i = 0; i < queries.size(); ++i) {
        auto [ti, si] = query_map[i];

        if (!fdm_results.results[i].has_value()) {
            errors[ti][si] = std::nan("");
            continue;
        }

        double fdm_iv = fdm_results.results[i]->implied_vol;
        double target = queries[i].market_price;
        double strike = kStrikes[si];
        double maturity = kMaturities[ti];

        double cheb_iv = brent_solve_iv(
            [&](double vol) {
                return surface.price(kSpot, strike, maturity, vol, kRate);
            },
            target);

        if (std::isnan(cheb_iv)) {
            errors[ti][si] = std::nan("");
            continue;
        }

        errors[ti][si] = std::abs(cheb_iv - fdm_iv) * 10000.0;
    }

    return errors;
}

static void run_chebyshev_4d() {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev 4D Tucker — vanilla (no dividends)\n");
    std::printf("================================================================\n\n");

    auto prices = generate_prices(/*with_dividends=*/false);

    std::printf("--- Building Chebyshev 4D surface...\n");
    auto surface = build_chebyshev_surface();

    std::printf("--- Computing Chebyshev IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Chebyshev 4D IV Error (bps) — σ=%.0f%%",
                      kVols[vi] * 100);
        auto errors = compute_errors_chebyshev(prices, surface, vi);
        print_heatmap(title, errors);
    }
}
```

In `main()`, add after the dividend section:

```cpp
    // Chebyshev 4D
    run_chebyshev_4d();
```

**Step 2: Update benchmarks/BUILD.bazel**

Add `chebyshev_table_builder` and `root_finding` deps to `interp_iv_safety`:

```bazel
    deps = [
        "//src/option:american_option",
        "//src/option:american_option_batch",
        "//src/option:iv_solver",
        "//src/option:interpolated_iv_solver",
        "//src/option/table/chebyshev:chebyshev_table_builder",
        "//src/math:root_finding",
    ],
```

**Step 3: Build and run**

Run: `bazel build //benchmarks:interp_iv_safety`
Expected: Compiles successfully

Run: `bazel-bin/benchmarks/interp_iv_safety`
Expected: Prints vanilla heatmaps, dividend heatmaps, then Chebyshev heatmaps

**Step 4: Run full test suite**

Run: `bazel test //...`
Expected: All tests pass (including new chebyshev_surface_test)

Run: `bazel build //benchmarks/... && bazel build //src/python:mango_option`
Expected: All benchmarks and Python bindings build

**Step 5: Commit**

```bash
git add benchmarks/interp_iv_safety.cc benchmarks/BUILD.bazel
git commit -m "Add Chebyshev 4D section to IV safety benchmark"
```

---

### Task 4: Verify and Clean Up

**Step 1: Run full CI checks**

```bash
bazel test //...
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

Expected: All pass, no warnings.

**Step 2: Run the benchmark end-to-end**

```bash
bazel run //benchmarks:interp_iv_safety
```

Verify output shows:
- Vanilla B-spline heatmaps (existing)
- Dividend heatmaps (existing)
- Chebyshev 4D heatmaps (new)

Compare RMS errors between vanilla and Chebyshev sections.

**Step 3: Commit any fixes**

If any issues found, fix and commit.
