# Chebyshev / Dimensionless Moneyness Coverage (#480) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every remaining table builder (Chebyshev continuous, Chebyshev segmented, non-adaptive Chebyshev table, dimensionless builder, dimensionless reference probe) solves its PDE batch on a materialized grid that covers the log-moneyness nodes it evaluates, so no table node is ever a cubic-spline extrapolation.

**Architecture:** Relocate the #479 helpers `detail::ensure_moneyness_coverage` / `detail::materialize_covering_grid` from the B-spline builder to a backend-neutral target `//src/option/table:covering_grid`, generalize the reach to be node-order independent, and pass a materialized `PDEGridConfig` as `custom_grid` at the five gridless `solve_batch` sites. One red-first regression per site, each with a direct pinned-profile FDM oracle.

**Tech Stack:** C++23, Bazel (Bzlmod), GoogleTest. Build/test with `bazel`.

**Spec:** `docs/superpowers/specs/2026-09-03-chebyshev-coverage-480-design.md` (read it first; task numbering below refers to its site labels S1–S5 and test labels T1–T6).

## Global Constraints

- Every new source file starts with `// SPDX-License-Identifier: MIT` (BUILD files: `# SPDX-License-Identifier: MIT`).
- Library code never uses `printf`/`fprintf`/`std::cout`.
- Commit messages: imperative mood, subject ≤ 50 chars, body wrapped at 72, explain what and why. No attribution lines.
- Every regression test must be run **red on the parent code** before the fix is applied, and its comment must record the measured pre-fix error. Tolerances are pinned from measurement: at least ~10× the measured post-fix deviation (loosen further only for cross-toolchain robustness, and say so in the comment) and at least ~50× below the pre-fix error.
- Every FDM oracle uses an explicit pinned profile: `AmericanOptionSolver::create(params, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)})` then `solve()`. Never `solve_american_option` (default grid).
- Baseline before this plan: `bazel test //...` = 150/150 passing in this worktree. The count must never drop; Task 1 adds one target (151).
- Run tests with `TMPDIR` pointed at the session scratch dir: `TMPDIR=/tmp/codex-skills/$CLAUDE_CODE_SESSION_ID bazel test ...`.
- `bazel` runs may touch `MODULE.bazel.lock`; never commit that change (`git checkout MODULE.bazel.lock` before committing).
- Reference test patterns: `tests/adaptive_surface_build_integration_test.cc:807-893` (#437 e2e regression + `fdm_reference_price` helper) and `tests/price_table_builder_test.cc:426-498` (#479 helper unit tests).

---

## File map

| File | Change |
|---|---|
| `src/option/table/covering_grid.hpp` / `.cpp` | **Create.** Home of the two helpers (moved from the B-spline builder), reach via `std::minmax_element`. |
| `src/option/table/BUILD.bazel` | **Modify.** Add `cc_library(name = "covering_grid")`. |
| `src/option/table/bspline/bspline_builder.hpp` | **Modify.** Replace the two `detail` declarations (lines ~236-253) with `#include "mango/option/table/covering_grid.hpp"`. |
| `src/option/table/bspline/bspline_builder.cpp` | **Modify.** Delete the two definitions (`namespace detail { ... }` block, lines ~233-267); include the new header. |
| `src/option/table/bspline/bspline_adaptive.cpp` | **Modify.** Include the new header directly. |
| `src/option/table/bspline/BUILD.bazel` | **Modify.** `bspline_builder` and `bspline_adaptive` gain dep `//src/option/table:covering_grid`. |
| `tests/covering_grid_test.cc` | **Create.** The four moved #479 helper tests + one order-independence test (T1). |
| `tests/price_table_builder_test.cc` | **Modify.** Remove the moved #437/#479 helper tests (lines 426-498). |
| `tests/BUILD.bazel` | **Modify.** Add `covering_grid_test`; add deps to `dimensionless_adaptive_test`. |
| `src/option/table/chebyshev/chebyshev_adaptive.cpp` | **Modify.** S1 (`make_chebyshev_build_fn`) and S2 (`solve_missing_pde_pairs` gains `m_nodes`) materialize a covering grid. |
| `src/option/table/chebyshev/chebyshev_table_builder.cpp` | **Modify.** S3 materializes a covering grid. |
| `src/option/table/chebyshev/BUILD.bazel` | **Modify.** `chebyshev_adaptive`, `chebyshev_table_builder` gain the dep. |
| `src/option/table/dimensionless/dimensionless_builder.cpp` | **Modify.** S4 materializes a covering grid. |
| `src/option/table/dimensionless/dimensionless_adaptive.cpp` | **Modify.** S5: `reference_eep` → `detail::dimensionless_reference_eep`, materializes a covering grid. |
| `src/option/table/dimensionless/dimensionless_adaptive_detail.hpp` | **Create.** Declaration of `detail::dimensionless_reference_eep`. |
| `src/option/table/dimensionless/BUILD.bazel` | **Modify.** Both targets gain the dep; `dimensionless_adaptive` gains the new hdr. |
| `tests/adaptive_surface_build_integration_test.cc` | **Modify.** T2 (S1) and T3 (S2) regressions. |
| `tests/chebyshev_surface_test.cc` | **Modify.** T5 (S3) regression. |
| `tests/dimensionless_builder_test.cc` | **Modify.** T4 (S4) regression. |
| `tests/dimensionless_adaptive_test.cc` | **Modify.** T6 (S5) regression. |

---

### Task 1: Backend-neutral `covering_grid` target + T1

**Files:**
- Create: `src/option/table/covering_grid.hpp`, `src/option/table/covering_grid.cpp`, `tests/covering_grid_test.cc`
- Modify: `src/option/table/BUILD.bazel`, `src/option/table/bspline/bspline_builder.hpp:236-253`, `src/option/table/bspline/bspline_builder.cpp:233-267`, `src/option/table/bspline/bspline_adaptive.cpp` (includes), `src/option/table/bspline/BUILD.bazel`, `tests/price_table_builder_test.cc:426-498`, `tests/BUILD.bazel`

**Interfaces:**
- Produces (used by every later task):
  ```cpp
  namespace mango::detail {
  void ensure_moneyness_coverage(GridAccuracyParams& accuracy,
                                 std::span<const PricingParams> batch,
                                 std::span<const double> log_moneyness_nodes);
  PDEGridSpec materialize_covering_grid(GridAccuracyParams accuracy,
                                        std::span<const PricingParams> batch,
                                        std::span<const double> log_moneyness_nodes);
  }
  ```
  Include path `mango/option/table/covering_grid.hpp`, Bazel target `//src/option/table:covering_grid`.

- [ ] **Step 1: Create the header**

`src/option/table/covering_grid.hpp`:
```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/grid_spec_types.hpp"
#include "mango/option/option_spec.hpp"
#include <span>

namespace mango::detail {

/// Ensure accuracy.n_sigma is large enough that a shared grid estimated by
/// estimate_batch_pde_grid(batch, accuracy) for a normalized batch
/// (spot = strike = K_ref, x0 = 0) spans every log-moneyness node in
/// `log_moneyness_nodes`: that grid's baseline half-width is
/// n_sigma * max(sigma*sqrt(T)) over `batch`, and any node beyond it would
/// be evaluated by cubic-spline extrapolation of the slice.  The reach is
/// max(|min|, |max|) over the nodes in ANY order (Chebyshev/CC node arrays
/// are handed in directly).  Callers must actually solve on such an
/// estimated grid, passed as a concrete custom grid: the batch solver's
/// gridless routing re-estimates per normalized group (or unions the
/// unwidened per-param grids) and does NOT realize this width.  No-op when
/// either span is empty.
void ensure_moneyness_coverage(GridAccuracyParams& accuracy,
                               std::span<const PricingParams> batch,
                               std::span<const double> log_moneyness_nodes);

/// Widen `accuracy` for moneyness coverage, estimate the shared batch
/// grid, and return it as a concrete PDEGridConfig suitable for
/// solve_batch's custom_grid parameter, which propagates it verbatim into
/// every normalized group and into the regular shared path.
/// mandatory_times stays empty: the batch solver reconstructs per-contract
/// dividend taus itself.
PDEGridSpec materialize_covering_grid(GridAccuracyParams accuracy,
                                      std::span<const PricingParams> batch,
                                      std::span<const double> log_moneyness_nodes);

}  // namespace mango::detail
```

- [ ] **Step 2: Create the implementation (verbatim move first — front()/back() reach stays for now)**

`src/option/table/covering_grid.cpp` (copy the bodies from `bspline_builder.cpp:237-265` exactly; only the parameter name changes):
```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/table/covering_grid.hpp"

#include <algorithm>
#include <cmath>

namespace mango::detail {

void ensure_moneyness_coverage(GridAccuracyParams& accuracy,
                               std::span<const PricingParams> batch,
                               std::span<const double> log_moneyness_nodes)
{
    if (batch.empty() || log_moneyness_nodes.empty()) return;

    const double log_m_min = log_moneyness_nodes.front();
    const double log_m_max = log_moneyness_nodes.back();
    const double required_half_width =
        std::max(std::abs(log_m_min), std::abs(log_m_max));

    // Compute max σ√T across the batch (floor to avoid division by zero)
    double max_sigma_sqrt_T = 0.0;
    for (const auto& p : batch) {
        max_sigma_sqrt_T = std::max(max_sigma_sqrt_T,
                                    p.volatility * std::sqrt(p.maturity));
    }
    max_sigma_sqrt_T = std::max(max_sigma_sqrt_T, 1e-10);

    constexpr double MARGIN = 1.1;  // 10% margin for boundary effects
    double required_n_sigma = (required_half_width / max_sigma_sqrt_T) * MARGIN;
    accuracy.n_sigma = std::max(accuracy.n_sigma, required_n_sigma);
}

PDEGridSpec materialize_covering_grid(GridAccuracyParams accuracy,
                                      std::span<const PricingParams> batch,
                                      std::span<const double> log_moneyness_nodes)
{
    ensure_moneyness_coverage(accuracy, batch, log_moneyness_nodes);
    auto [grid_spec, time_domain] = estimate_batch_pde_grid(batch, accuracy);
    return PDEGridConfig{grid_spec, time_domain.n_steps(), {}};
}

}  // namespace mango::detail
```

- [ ] **Step 3: Add the Bazel target**

In `src/option/table/BUILD.bazel`, after the `pde_cache` library:
```python
cc_library(
    name = "covering_grid",
    srcs = ["covering_grid.cpp"],
    hdrs = ["covering_grid.hpp"],
    deps = [
        "//src/option:grid_spec_types",
        "//src/option:option_spec",
    ],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option/table",
    include_prefix = "mango/option/table",
)
```

- [ ] **Step 4: Remove the old declarations and definitions from the B-spline builder**

`src/option/table/bspline/bspline_builder.hpp`: delete the `ensure_moneyness_coverage` and `materialize_covering_grid` declarations with their doc comments (lines ~236-253, inside `namespace detail`; leave `uniform_grid`/`sqrt_uniform_grid` and the closing `}  // namespace detail`). Add `#include "mango/option/table/covering_grid.hpp"` next to the other `mango/option/...` includes at the top.

`src/option/table/bspline/bspline_builder.cpp`: delete the whole `namespace detail { void ensure_moneyness_coverage(...) {...} PDEGridSpec materialize_covering_grid(...) {...} }  // namespace detail` block (lines ~233-267). Add `#include "mango/option/table/covering_grid.hpp"` at the top (the call at ~line 380 stays).

`src/option/table/bspline/bspline_adaptive.cpp`: add `#include "mango/option/table/covering_grid.hpp"` (calls at ~263, ~272 stay).

`src/option/table/bspline/BUILD.bazel`: add `"//src/option/table:covering_grid",` to the `deps` of both `bspline_builder` and `bspline_adaptive`.

- [ ] **Step 5: Move the four helper tests**

Create `tests/covering_grid_test.cc`:
```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <cmath>
#include <variant>
#include "mango/option/table/covering_grid.hpp"

namespace mango {
namespace {

// ===========================================================================
// Regression tests for issue #437 (moneyness coverage helpers), moved here
// from price_table_builder_test.cc when the helpers became backend-neutral
// (#480).
// ===========================================================================
```
then paste the four tests verbatim from `tests/price_table_builder_test.cc` lines 429-495 (`EnsureMoneynessCoverage.WidensNSigmaWhenAxisUndershoots`, `.LeavesNSigmaWhenCovered`, `.EmptyInputsAreNoOps`, `MaterializeCoveringGrid.ConcreteGridCoversAxisForMultiSigmaBatch`) and close with
```cpp
}  // namespace
}  // namespace mango
```
Delete those tests and their section header from `tests/price_table_builder_test.cc` (lines 426-498, from the `// ====` banner "Regression tests for issue #437" through the end of `ConcreteGridCoversAxisForMultiSigmaBatch`), keeping the file's closing namespaces intact.

In `tests/BUILD.bazel`, right after `price_table_builder_test`:
```python
cc_test(
    name = "covering_grid_test",
    size = "small",
    srcs = ["covering_grid_test.cc"],
    deps = [
        "//src/option/table:covering_grid",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

- [ ] **Step 6: Build and run the moved tests + the B-spline consumers (all green: pure move)**

Run: `bazel test //tests:covering_grid_test //tests:price_table_builder_test //tests:price_table_builder_custom_grid_test //tests:adaptive_surface_build_integration_test`
Expected: all PASS (4 tests in `covering_grid_test`).

- [ ] **Step 7: Write the failing order-independence test (T1)**

Append to `tests/covering_grid_test.cc` before the closing namespaces:
```cpp
// Regression (#480 D3): the reach is the largest-magnitude node wherever it
// sits in the span.  A merely reversed array still has its extremes at
// front()/back(); only an INTERIOR extreme distinguishes minmax from the
// old front()/back() reach, which read {0.0, -0.51, 0.10} as reach 0.10.
TEST(EnsureMoneynessCoverage, ReachIsOrderIndependent) {
    std::vector<mango::PricingParams> batch{mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.1,
                          .rate = 0.05, .dividend_yield = 0.0,
                          .option_type = mango::OptionType::PUT},
        0.20)};
    std::vector<double> sorted = {-0.51, 0.0, 0.10};
    std::vector<double> permuted = {0.0, -0.51, 0.10};

    mango::GridAccuracyParams from_sorted, from_permuted;
    mango::detail::ensure_moneyness_coverage(from_sorted, batch, sorted);
    mango::detail::ensure_moneyness_coverage(from_permuted, batch, permuted);

    const double expected = 0.51 / (0.20 * std::sqrt(0.1)) * 1.1;  // ~8.87
    EXPECT_NEAR(from_sorted.n_sigma, expected, 1e-12);
    EXPECT_NEAR(from_permuted.n_sigma, expected, 1e-12);
}
```

- [ ] **Step 8: Run it, verify it fails**

Run: `bazel test //tests:covering_grid_test --test_output=errors`
Expected: FAIL — `from_permuted.n_sigma` is 5.0 (reach read as 0.10 → 1.74 < default 5.0).

- [ ] **Step 9: Implement the order-independent reach**

In `src/option/table/covering_grid.cpp` replace the two `front()/back()` lines with:
```cpp
    const auto [lo_it, hi_it] = std::minmax_element(
        log_moneyness_nodes.begin(), log_moneyness_nodes.end());
    const double required_half_width =
        std::max(std::abs(*lo_it), std::abs(*hi_it));
```

- [ ] **Step 10: Run, verify green**

Run: `bazel test //tests:covering_grid_test //tests:price_table_builder_test //tests:price_table_builder_custom_grid_test --test_output=errors`
Expected: PASS (5 tests in `covering_grid_test`).

- [ ] **Step 11: Commit**

```bash
git checkout MODULE.bazel.lock
git add src/option/table/covering_grid.hpp src/option/table/covering_grid.cpp \
        src/option/table/BUILD.bazel src/option/table/bspline/ tests/covering_grid_test.cc \
        tests/price_table_builder_test.cc tests/BUILD.bazel
git commit -m "Move covering-grid helpers to a backend-neutral target

The Chebyshev and dimensionless builders need the same
materialize_covering_grid mechanism PR #479 gave the B-spline builders,
but they must not depend on the B-spline builder to get it.  The reach
is now the largest-magnitude node in any order, so callers can hand in
Chebyshev node arrays directly."
```

---

### Task 2: S1 — continuous Chebyshev adaptive build (T2)

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp:397-406` (inside `make_chebyshev_build_fn`), `src/option/table/chebyshev/BUILD.bazel` (`chebyshev_adaptive` deps)
- Test: `tests/adaptive_surface_build_integration_test.cc` (append after `FallbackExplicitGridCoversMoneynessTails`)

**Interfaces:**
- Consumes: `detail::materialize_covering_grid` (Task 1); existing `fdm_reference_price(spot, strike, tau, sigma, rate)` helper at `adaptive_surface_build_integration_test.cc:807` (PUT, q = 0, High profile).
- Produces: nothing new.

- [ ] **Step 1: Write the failing e2e test**

Append to `tests/adaptive_surface_build_integration_test.cc` (inside the anonymous namespace, after the #437 fallback test):
```cpp
// Regression (#480, S1): the continuous Chebyshev build solved its
// (sigma, rate) batch gridless.  extract_chain_domain floors the tau axis
// to a 0.5y spread and build_adaptive_chebyshev adds CC headroom, so for
// this chain the PDE maturity is 1.01 * 0.6875 and the old batch-union
// half-width is 5 * sigma_hi * sqrt(0.694) ~= 5 * 0.225 * 0.833 ~= 0.94
// (the batch is normalized-ineligible: its first param is the sigma_lo =
// 0.01 node, whose margin is far below 0.35).  The moneyness nodes reach
// +-ln(2.5) * (1 + 6/32) ~= +-1.09, so both endpoint nodes were
// cubic-spline extrapolations -- and a Chebyshev interpolant is a global
// polynomial, so the garbage reaches the user's own strikes.
// Pre-fix max abs error on this branch's parent: <FILL FROM STEP 2>
// (at <m>, sigma=<sigma>).
TEST(AdaptiveGridBuilderTest, ChebyshevNodesMatchFdmAtExtremeMoneyness) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    chain.strikes = {40.0, 60.0, 100.0, 160.0, 250.0};
    chain.maturities = {0.05, 0.1};
    chain.implied_vols = {0.10};
    chain.rates = {0.03, 0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;  // relaxed: accuracy is asserted below
    params.max_iter = 2;
    params.validation_samples = 8;

    auto result = build_adaptive_chebyshev(params, chain, OptionType::PUT);
    ASSERT_TRUE(result.has_value())
        << "build failed: " << static_cast<int>(result.error().code);
    ASSERT_NE(result->surface, nullptr);
    const auto& surface = *result->surface;

    // Node span of the fit domain (CC-extended), read back from the
    // interpolant so the test cannot drift from the builder's headroom.
    const auto& dom = surface.inner().interpolant().domain();
    const double K = chain.spot;
    const double tau = dom.hi[1];   // a tau node: isolates m extraction
    const double r = dom.hi[3];     // a rate node
    const auto& sb = result->sample_bounds;

    struct Query { double m; const char* what; };
    const Query queries[] = {
        {dom.lo[0], "node m_lo"},
        {dom.hi[0], "node m_hi"},
        {std::log(100.0 / 250.0), "user strike 250"},
        {std::log(100.0 / 40.0), "user strike 40"},
    };
    // sigma at the node endpoints (on-axis) and the user-facing sample
    // bounds (interpolated in sigma).
    const double sigmas[] = {dom.lo[2], dom.hi[2], sb.sigma_min, sb.sigma_max};

    // Tolerance in $ per K=100: <FILL FROM STEP 5 -- post-fix max deviation
    // and the rationale for the chosen multiple>.
    constexpr double TOL = 1e-3;

    for (const auto& q : queries) {
        for (double sigma : sigmas) {
            const double S = K * std::exp(q.m);
            const double ref = fdm_reference_price(S, K, tau, sigma, r);
            const double got = surface.price(S, K, tau, sigma, r);
            EXPECT_NEAR(got, ref, TOL)
                << q.what << " m=" << q.m << " sigma=" << sigma;
        }
    }
}
```

- [ ] **Step 2: Run it on the unfixed code; record the failure**

Run: `bazel test //tests:adaptive_surface_build_integration_test --test_filter='AdaptiveGridBuilderTest.ChebyshevNodesMatchFdmAtExtremeMoneyness' --test_output=all`
Expected: FAIL. Record the largest `|got - ref|` and its (m, sigma) into the test comment's `Pre-fix max abs error` line. The deep-ITM put end (`node m_lo`, `user strike 250`... i.e. negative m) is expected to be the visibly wrong one. If **no** assertion fails, widen the strikes (e.g. `{30, 60, 100, 160, 330}`) and re-derive the comment's numbers before continuing; do not proceed with a test that is green pre-fix.

- [ ] **Step 3: Apply the S1 fix**

In `src/option/table/chebyshev/chebyshev_adaptive.cpp`, add `#include "mango/option/table/covering_grid.hpp"` to the includes, and in `make_chebyshev_build_fn` replace
```cpp
            BatchAmericanOptionSolver solver;
            solver.set_grid_accuracy(
                make_grid_accuracy(GridAccuracyProfile::Ultra));
            std::vector<double> tau_vec(tau_nodes.begin(), tau_nodes.end());
            solver.set_snapshot_times(std::span<const double>(tau_vec));
            auto batch_result = solver.solve_batch(
                std::span<const PricingParams>(batch), /*use_shared_grid=*/true);
```
with
```cpp
            const auto accuracy = make_grid_accuracy(GridAccuracyProfile::Ultra);
            BatchAmericanOptionSolver solver;
            // grid_accuracy_ still decides normalized-chain eligibility
            // (and its traced route); the grid itself comes from the
            // covering spec below (#480).
            solver.set_grid_accuracy(accuracy);
            std::vector<double> tau_vec(tau_nodes.begin(), tau_nodes.end());
            solver.set_snapshot_times(std::span<const double>(tau_vec));
            // One concrete grid covering every moneyness node, estimated
            // over the batch actually solved (the missing pairs), passed as
            // custom_grid so both the normalized-chain and the shared
            // regular path solve on it.  Gridless solving sized the domain
            // from n_sigma*sigma*sqrt(T) alone and extrapolated the tails
            // (#480, same defect as #437).
            auto covering = detail::materialize_covering_grid(
                accuracy, std::span<const PricingParams>(batch), m_nodes);
            auto batch_result = solver.solve_batch(
                std::span<const PricingParams>(batch), /*use_shared_grid=*/true,
                nullptr, covering);
```
In `src/option/table/chebyshev/BUILD.bazel` add `"//src/option/table:covering_grid",` to `chebyshev_adaptive`'s deps.

- [ ] **Step 4: Run, verify green**

Run: `bazel test //tests:adaptive_surface_build_integration_test --test_filter='AdaptiveGridBuilderTest.ChebyshevNodesMatchFdmAtExtremeMoneyness' --test_output=all`
Expected: PASS.

- [ ] **Step 5: Pin the tolerance and finish the comment**

Temporarily print (or read from the EXPECT output with `TOL = 0`) the post-fix max `|got - ref|`; set `TOL` to ≥ 10× that value, ≤ 1/50 of the pre-fix error, and write both numbers plus the rationale into the comment. Remove any temporary printing. Note the test's wall time from the Bazel log; if it exceeds ~90 s, say so in the commit body (the CI split decision is Task 7's).

- [ ] **Step 6: Run the full Chebyshev adaptive suite**

Run: `bazel test //tests:adaptive_surface_build_integration_test //tests:adaptive_grid_builder_test //tests:adaptive_surface_build_slow_test --test_output=errors`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git checkout MODULE.bazel.lock
git add src/option/table/chebyshev/chebyshev_adaptive.cpp src/option/table/chebyshev/BUILD.bazel tests/adaptive_surface_build_integration_test.cc
git commit -m "Solve continuous Chebyshev batch on a covering grid

The continuous adaptive build solved its (sigma, rate) batch gridless,
so the PDE domain was sized from n_sigma*sigma*sqrt(T) alone and the
outer moneyness nodes were cubic-spline extrapolations.  A Chebyshev
interpolant is a global polynomial, so the garbage reached user strikes.
Materialize one covering grid over the missing batch and pass it as
custom_grid (#480, same mechanism as #479)."
```

---

### Task 3: S2 — segmented Chebyshev (`solve_missing_pde_pairs`) (T3)

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp` (`solve_missing_pde_pairs` ~176-224 and its two callers ~517-519, ~571-573)
- Test: `tests/adaptive_surface_build_integration_test.cc` (append after Task 2's test)

**Interfaces:**
- Consumes: `detail::materialize_covering_grid` (Task 1).
- Produces: `solve_missing_pde_pairs(cache, K_ref, option_type, dividend_yield, discrete_dividends, m_nodes, tau_nodes, sigma_nodes, rate_nodes)` (file-static; new `std::span<const double> m_nodes` parameter placed before `tau_nodes`).

- [ ] **Step 1: Write the failing test with both oracles**

Append to `tests/adaptive_surface_build_integration_test.cc`:
```cpp
// Padded-timeline coverage oracle for the segmented Chebyshev path: the
// table solves a normalized contract with maturity 1.01 * tau_max and the
// dividend's calendar time anchored to THAT maturity, so at the tau_query
// snapshot the event sits at tau = 1.01*tau_query - calendar_time, not at
// tau_query - calendar_time.  Reproduce exactly that contract on an
// explicit wide grid and read the snapshot at the queried moneyness, so
// the comparison isolates spatial coverage from the (pre-existing,
// out-of-scope) timing skew.  Returns a dollar price for strike K.
double segmented_coverage_oracle(double S, double K, double tau_query,
                                 double sigma, double rate,
                                 const std::vector<Dividend>& dividends) {
    PricingParams p(
        OptionSpec{.spot = K, .strike = K, .maturity = tau_query * 1.01,
                   .rate = rate, .dividend_yield = 0.0,
                   .option_type = OptionType::PUT},
        sigma);
    p.discrete_dividends = dividends;

    BatchAmericanOptionSolver solver;
    const std::vector<double> snaps = {tau_query};
    solver.set_snapshot_times(std::span<const double>(snaps));
    auto grid_spec = GridSpec<double>::sinh_spaced(-1.5, 1.5, 3001, 2.0).value();
    const std::vector<PricingParams> batch = {p};
    auto res = solver.solve_batch(
        std::span<const PricingParams>(batch), /*use_shared_grid=*/true,
        nullptr, PDEGridSpec{PDEGridConfig{grid_spec, 4000, {}}});
    EXPECT_TRUE(res.results[0].has_value());
    const auto& r = res.results[0].value();
    CubicSpline<double> spline;
    auto err = spline.build(r.grid()->x(), r.at_time(0));
    EXPECT_FALSE(err.has_value());
    return spline.eval(std::log(S / K)) * K;   // at_time() is V/K
}

// User-contract oracle: the option the user actually asked about, with the
// dividend at its true calendar time (High profile, pinned).
double dividend_fdm_reference_price(double S, double K, double tau,
                                    double sigma, double rate,
                                    const std::vector<Dividend>& dividends) {
    PricingParams p(
        OptionSpec{.spot = S, .strike = K, .maturity = tau, .rate = rate,
                   .dividend_yield = 0.0, .option_type = OptionType::PUT},
        sigma);
    p.discrete_dividends = dividends;
    auto solver = AmericanOptionSolver::create(
        p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
    EXPECT_TRUE(solver.has_value());
    auto ref = solver->solve();
    EXPECT_TRUE(ref.has_value());
    return ref->value_at(S);
}

// Regression (#480, S2): the segmented Chebyshev build solved its dividend
// batch gridless.  A batch with discrete dividends is normalized-ineligible,
// so it solved on the batch-union grid: half-width 5 * sigma_hi * sqrt(T)
// with T = 1.01 * 0.25 and sigma_hi = 0.15 + 0.075 (CC headroom on the
// [0.05, 0.15] sample range) ~= 0.57, left edge nudged by the dividend
// extension.  The moneyness nodes span the user range [ln 0.49, ln 2]
// (dividend-widened) plus 3/32 of headroom ~= [-0.85, 0.83], so every
// endpoint node was extrapolated; the queried S = 50 and S = 200 are
// themselves outside the old domain.
// Pre-fix max abs error on this branch's parent (coverage oracle):
// <FILL FROM STEP 2> (at S=<S>, sigma=<sigma>).
TEST(AdaptiveGridBuilderTest, SegmentedChebyshevTailsMatchFdmAtExtremeMoneyness) {
    const std::vector<Dividend> dividends = {
        Dividend{.calendar_time = 0.1, .amount = 1.0}};
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = dividends,
        .maturity = 0.25,
        .kref_config = {.K_refs = {100.0}},   // single K_ref: no strike blend
    };
    IVGrid grid{
        .moneyness = {std::log(0.5), 0.0, std::log(2.0)},  // log(S/K) here
        .vol = {0.10},                                       // -> [0.05, 0.15]
        .rate = {0.03, 0.05},
    };

    auto surface = build_chebyshev_segmented_manual(seg_config, grid);
    ASSERT_TRUE(surface.has_value())
        << "build failed: " << static_cast<int>(surface.error().code);

    const double K = 100.0;
    const double tau = 0.25;
    const double r = 0.05;

    // Coverage oracle tolerance ($ per K=100): <FILL FROM STEP 5>.
    constexpr double TOL_COVERAGE = 1e-3;
    // User-contract tolerance: must exceed the measured timing-skew
    // discrepancy between the two oracles (<FILL>) and sit well below the
    // pre-fix error.
    constexpr double TOL_USER = 2e-2;

    for (double S : {50.0, 200.0}) {
        for (double sigma : {0.05, 0.15}) {
            const double got = surface->price(S, K, tau, sigma, r);
            const double cov = segmented_coverage_oracle(S, K, tau, sigma, r, dividends);
            EXPECT_NEAR(got, cov, TOL_COVERAGE)
                << "coverage oracle S=" << S << " sigma=" << sigma;
            const double usr = dividend_fdm_reference_price(S, K, tau, sigma, r, dividends);
            EXPECT_NEAR(got, usr, TOL_USER)
                << "user oracle S=" << S << " sigma=" << sigma;
        }
    }
}
```
Add `#include "mango/math/cubic_spline_solver.hpp"` and `#include "mango/option/grid_spec_types.hpp"` to the test's includes, and `"//src/math:cubic_spline_solver",` plus `"//src/option:grid_spec_types",` to `adaptive_surface_build_integration_test`'s deps in `tests/BUILD.bazel` (`//src/pde/core:grid` is already there).

- [ ] **Step 2: Run it on the unfixed code; record the failure**

Run: `bazel test //tests:adaptive_surface_build_integration_test --test_filter='AdaptiveGridBuilderTest.SegmentedChebyshevTailsMatchFdmAtExtremeMoneyness' --test_output=all`
Expected: FAIL on the coverage-oracle assertions (S = 50 is the deep-ITM put end). Record the largest `|got - cov|` into the comment. If nothing fails, widen `moneyness` (e.g. `{ln 0.4, 0, ln 2.5}`) and re-derive the comment.

- [ ] **Step 3: Apply the S2 fix**

In `chebyshev_adaptive.cpp`, change `solve_missing_pde_pairs`'s signature to
```cpp
static size_t solve_missing_pde_pairs(
    ChebyshevPDECache& cache,
    double K_ref,
    OptionType option_type,
    double dividend_yield,
    const std::vector<Dividend>& discrete_dividends,
    std::span<const double> m_nodes,
    std::span<const double> tau_nodes,
    std::span<const double> sigma_nodes,
    std::span<const double> rate_nodes)
```
and replace its solver block
```cpp
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    std::vector<double> tau_vec(tau_nodes.begin(), tau_nodes.end());
    solver.set_snapshot_times(std::span<const double>(tau_vec));
    auto batch_result = solver.solve_batch(
        std::span<const PricingParams>(batch), /*use_shared_grid=*/true);
```
with
```cpp
    const auto accuracy = make_grid_accuracy(GridAccuracyProfile::Ultra);
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(accuracy);  // routing eligibility only
    std::vector<double> tau_vec(tau_nodes.begin(), tau_nodes.end());
    solver.set_snapshot_times(std::span<const double>(tau_vec));
    // A dividend batch is normalized-ineligible and solved on the
    // batch-union grid, whose half-width n_sigma*sigma_max*sqrt(T) knows
    // nothing about the moneyness nodes; materialize a covering grid over
    // the missing batch and pass it as custom_grid (#480).  Dividend event
    // times are rebuilt per contract by the batch solver, so the config's
    // empty mandatory_times is safe.
    auto covering = detail::materialize_covering_grid(
        accuracy, std::span<const PricingParams>(batch), m_nodes);
    auto batch_result = solver.solve_batch(
        std::span<const PricingParams>(batch), /*use_shared_grid=*/true,
        nullptr, covering);
```
Update both callers to pass `m_nodes` before `tau_nodes`:
- in `make_segmented_chebyshev_build_fn`: `solve_missing_pde_pairs(cache, config.K_ref, config.option_type, config.dividend_yield, config.discrete_dividends, m_nodes, tau_nodes, sigma_nodes, rate_nodes);`
- in `build_chebyshev_segmented_pieces`: `solve_missing_pde_pairs(cache, K_ref, option_type, dividend_yield, discrete_dividends, m_nodes, tau_nodes, sigma_nodes, rate_nodes);`

- [ ] **Step 4: Run, verify green**

Run: `bazel test //tests:adaptive_surface_build_integration_test --test_filter='AdaptiveGridBuilderTest.SegmentedChebyshevTailsMatchFdmAtExtremeMoneyness' --test_output=all`
Expected: PASS on the coverage oracle. If the user-contract assertion still fails, measure `|cov - usr|` (the timing skew) and set `TOL_USER` above it; it must still be ≤ 1/50 of the pre-fix error, otherwise report the numbers and stop.

- [ ] **Step 5: Pin tolerances; finish comments**

Fill in the post-fix coverage-oracle deviation, the measured skew, and rationale. Remove temporary printing.

- [ ] **Step 6: Run the segmented suites**

Run: `bazel test //tests:adaptive_surface_build_integration_test //tests:adaptive_surface_build_slow_test //tests:price_table_data_test //tests:iv_solver_factory_test --test_output=errors`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git checkout MODULE.bazel.lock
git add src/option/table/chebyshev/chebyshev_adaptive.cpp tests/adaptive_surface_build_integration_test.cc tests/BUILD.bazel
git commit -m "Solve segmented Chebyshev batch on a covering grid

The dividend batch is normalized-ineligible and solved on the batch-union
grid, whose half-width n_sigma*sigma_max*sqrt(T) is unrelated to the
moneyness nodes the leaves evaluate, so the outer nodes were cubic-spline
extrapolations.  Pass the moneyness nodes into solve_missing_pde_pairs
and solve on a materialized covering grid (#480)."
```

---

### Task 4: S3 — non-adaptive `build_chebyshev_table` (T5)

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_table_builder.cpp:116-120`, `src/option/table/chebyshev/BUILD.bazel` (`chebyshev_table_builder` deps)
- Test: `tests/chebyshev_surface_test.cc` (append)

**Interfaces:**
- Consumes: `detail::materialize_covering_grid` (Task 1).

- [ ] **Step 1: Write the failing test**

Append to `tests/chebyshev_surface_test.cc` (file uses `using namespace mango;`, tests are top-level):
```cpp
// Regression (#480, S3): build_chebyshev_table solved its batch gridless
// with the default accuracy (n_sigma = 5), so the PDE half-width was
// 5 * 0.20 * sqrt(0.1) ~= 0.32 while the moneyness nodes reach +-0.7:
// both endpoint nodes were cubic-spline extrapolations.  All queried
// coordinates are CGL nodes (endpoints), so only extraction is measured.
// Pre-fix max abs error on this branch's parent: <FILL FROM STEP 2>.
TEST(ChebyshevTableBuilderTest, TailsMatchFdmAtExtremeMoneyness) {
    ChebyshevTableConfig config{
        .num_pts = {9, 5, 3, 3},
        .domain = {.lo = {-0.7, 0.01, 0.10, 0.02},
                   .hi = { 0.7, 0.10, 0.20, 0.06}},
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
    };
    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value());

    const double K = 100.0;
    const double tau = 0.10;   // tau node (domain hi)
    const double r = 0.06;     // rate node (domain hi)

    // Tolerance ($ per K=100): <FILL FROM STEP 5>.
    constexpr double TOL = 1e-3;

    for (double m : {-0.7, 0.7}) {
        for (double sigma : {0.10, 0.20}) {
            const double S = K * std::exp(m);
            PricingParams p(
                OptionSpec{.spot = S, .strike = K, .maturity = tau,
                           .rate = r, .dividend_yield = 0.0,
                           .option_type = OptionType::PUT},
                sigma);
            auto solver = AmericanOptionSolver::create(
                p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
            ASSERT_TRUE(solver.has_value());
            auto ref = solver->solve();
            ASSERT_TRUE(ref.has_value());
            const double got = result->surface.price(S, K, tau, sigma, r);
            EXPECT_NEAR(got, ref->value_at(S), TOL)
                << "m=" << m << " sigma=" << sigma;
        }
    }
}
```

- [ ] **Step 2: Run on the unfixed code; record the failure**

Run: `bazel test //tests:chebyshev_surface_test --test_filter='ChebyshevTableBuilderTest.TailsMatchFdmAtExtremeMoneyness' --test_output=all`
Expected: FAIL (m = −0.7, deep-ITM put). Record the max error in the comment.

- [ ] **Step 3: Apply the S3 fix**

In `chebyshev_table_builder.cpp` add `#include "mango/option/table/covering_grid.hpp"` and replace
```cpp
    BatchAmericanOptionSolver solver;
    solver.set_snapshot_times(std::span<const double>(tau_nodes));
    auto batch_result = solver.solve_batch(
        std::span<const PricingParams>(batch), /*use_shared_grid=*/true);
```
with
```cpp
    BatchAmericanOptionSolver solver;
    solver.set_snapshot_times(std::span<const double>(tau_nodes));
    // Solve on one concrete grid that covers every moneyness node; the
    // solver's own (default-accuracy) estimate is sized from
    // n_sigma*sigma*sqrt(T) alone and extrapolated the tails (#480).
    auto covering = detail::materialize_covering_grid(
        GridAccuracyParams{}, std::span<const PricingParams>(batch), m_nodes);
    auto batch_result = solver.solve_batch(
        std::span<const PricingParams>(batch), /*use_shared_grid=*/true,
        nullptr, covering);
```
Add `"//src/option/table:covering_grid",` to `chebyshev_table_builder`'s deps in `src/option/table/chebyshev/BUILD.bazel`.

- [ ] **Step 4: Run, verify green**

Run: `bazel test //tests:chebyshev_surface_test --test_output=errors`
Expected: PASS.

- [ ] **Step 5: Pin the tolerance; then run the consumers of `build_chebyshev_table`**

Run: `bazel test //tests:chebyshev_surface_test //tests:greeks_accuracy_test //tests:price_table_data_test //tests:chebyshev_pde_cache_test --test_output=errors`
Expected: PASS. If `greeks_accuracy_test` moves a pinned number, record old vs new in the commit body (do not loosen its tolerance without saying why).

- [ ] **Step 6: Commit**

```bash
git checkout MODULE.bazel.lock
git add src/option/table/chebyshev/chebyshev_table_builder.cpp src/option/table/chebyshev/BUILD.bazel tests/chebyshev_surface_test.cc
git commit -m "Solve non-adaptive Chebyshev table on a covering grid

build_chebyshev_table had the same gridless solve as the adaptive
builders and extrapolated its outer moneyness nodes; it was not listed
in #480 but falls under the same invariant."
```

---

### Task 5: S4 — `solve_dimensionless_pde` (T4)

**Files:**
- Modify: `src/option/table/dimensionless/dimensionless_builder.cpp:33-56`, `src/option/table/dimensionless/BUILD.bazel` (`dimensionless_builder` deps)
- Test: `tests/dimensionless_builder_test.cc` (append inside the anonymous namespace)

- [ ] **Step 1: Write the failing test**

```cpp
// Regression (#480, S4): solve_dimensionless_pde solved each kappa slice
// gridless with sigma_eff = sqrt(2) and T = 1.01 * tau'_max, so the PDE
// half-width was 5 * sqrt(2) * sqrt(0.00404) ~= 0.45 while the x nodes
// reach +-0.7: both endpoint nodes were cubic-spline extrapolations.
// Oracle: the same contract solved directly at spot = K e^x (High profile),
// dollar value divided by K (the table stores V/K).
// Pre-fix max abs error on this branch's parent: <FILL FROM STEP 2>.
TEST(DimensionlessBuilderTest, TailsMatchFdmAtExtremeMoneyness) {
    DimensionlessAxes axes;
    axes.log_moneyness = {-0.7, -0.35, 0.0, 0.35, 0.7};
    axes.tau_prime = {0.002, 0.004};
    axes.ln_kappa = {-1.0, 0.0, 0.5};
    const double K = 100.0;

    auto pde = solve_dimensionless_pde(axes, K, OptionType::PUT);
    ASSERT_TRUE(pde.has_value());

    const size_t Nt = axes.tau_prime.size();
    const size_t Nk = axes.ln_kappa.size();
    const size_t j = Nt - 1;                 // tau' = 0.004
    const double tp = axes.tau_prime[j];

    // Tolerance (V/K): <FILL FROM STEP 5>.
    constexpr double TOL = 1e-4;

    for (size_t i : {size_t{0}, axes.log_moneyness.size() - 1}) {
        const double x = axes.log_moneyness[i];
        for (size_t k = 0; k < Nk; ++k) {
            const double kappa = std::exp(axes.ln_kappa[k]);
            PricingParams p(
                OptionSpec{.spot = K * std::exp(x), .strike = K,
                           .maturity = tp, .rate = kappa,
                           .dividend_yield = 0.0,
                           .option_type = OptionType::PUT},
                std::sqrt(2.0));
            auto solver = AmericanOptionSolver::create(
                p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
            ASSERT_TRUE(solver.has_value());
            auto ref = solver->solve();
            ASSERT_TRUE(ref.has_value());
            const double got = pde->values[(i * Nt + j) * Nk + k];
            EXPECT_NEAR(got, ref->value() / K, TOL)
                << "x=" << x << " ln_kappa=" << axes.ln_kappa[k];
        }
    }
}
```

- [ ] **Step 2: Run on the unfixed code; record the failure**

Run: `bazel test //tests:dimensionless_builder_test --test_filter='DimensionlessBuilderTest.TailsMatchFdmAtExtremeMoneyness' --test_output=all`
Expected: FAIL at x = −0.7. Record the max error.

- [ ] **Step 3: Apply the S4 fix**

In `dimensionless_builder.cpp` add `#include "mango/option/table/covering_grid.hpp"`, hoist the accuracy out of the loop, and pass a covering grid:
```cpp
    const double sigma_eff = std::sqrt(2.0);
    const double pde_maturity = axes.tau_prime.back() * 1.01;
    const auto accuracy = make_grid_accuracy(GridAccuracyProfile::Ultra);
    int n_pde_solves = 0;

    for (size_t k = 0; k < Nk; ++k) {
        const double kappa = std::exp(axes.ln_kappa[k]);

        PricingParams params(
            OptionSpec{
                .spot = K_ref,
                .strike = K_ref,
                .maturity = pde_maturity,
                .rate = kappa,
                .dividend_yield = 0.0,
                .option_type = option_type},
            sigma_eff);

        BatchAmericanOptionSolver batch_solver;
        batch_solver.set_grid_accuracy(accuracy);
        batch_solver.set_snapshot_times(
            std::span<const double>{axes.tau_prime.data(), axes.tau_prime.size()});

        std::vector<PricingParams> batch = {params};
        // The solver's own estimate has half-width n_sigma*sqrt(2)*sqrt(T)
        // and extrapolated the outer x nodes for short tau'_max; solve on
        // a grid materialized to cover axes.log_moneyness (#480).
        auto covering = detail::materialize_covering_grid(
            accuracy, std::span<const PricingParams>(batch), axes.log_moneyness);
        auto batch_result = batch_solver.solve_batch(batch, true, nullptr, covering);
        ++n_pde_solves;
```
Add `"//src/option/table:covering_grid",` to `dimensionless_builder`'s deps.

- [ ] **Step 4: Run, verify green; pin tolerance**

Run: `bazel test //tests:dimensionless_builder_test --test_output=all`
Expected: PASS. Fill in the tolerance rationale.

- [ ] **Step 5: Run the dimensionless suites**

Run: `bazel test //tests:dimensionless_builder_test //tests:dimensionless_3d_surface_test //tests:dimensionless_iv_test //tests:dimensionless_adaptive_test --test_output=errors`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git checkout MODULE.bazel.lock
git add src/option/table/dimensionless/dimensionless_builder.cpp src/option/table/dimensionless/BUILD.bazel tests/dimensionless_builder_test.cc
git commit -m "Solve dimensionless PDE slices on a covering grid

Each kappa slice was solved gridless with half-width
n_sigma*sqrt(2)*sqrt(1.01*tau'_max), which undershoots the x axis
for short-dated / low-vol domains and extrapolated the outer nodes
(#480)."
```

---

### Task 6: S5 — `reference_eep` → `detail::dimensionless_reference_eep` (T6)

**Files:**
- Create: `src/option/table/dimensionless/dimensionless_adaptive_detail.hpp`
- Modify: `src/option/table/dimensionless/dimensionless_adaptive.cpp:34-70` (+ the two call sites of `reference_eep`), `src/option/table/dimensionless/BUILD.bazel` (`dimensionless_adaptive` hdrs + deps), `tests/BUILD.bazel` (`dimensionless_adaptive_test` deps)
- Test: `tests/dimensionless_adaptive_test.cc`

**Interfaces:**
- Produces:
  ```cpp
  namespace mango::detail {
  double dimensionless_reference_eep(double x0, double tau_prime_0,
                                     double ln_kappa_0, double K_ref,
                                     OptionType option_type);
  }
  ```
  include path `mango/option/table/dimensionless/dimensionless_adaptive_detail.hpp`, owned by `//src/option/table/dimensionless:dimensionless_adaptive`.

- [ ] **Step 1: Expose the probe (no behaviour change yet)**

Create `src/option/table/dimensionless/dimensionless_adaptive_detail.hpp`:
```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"

namespace mango::detail {

/// Ground-truth probe for the dimensionless adaptive loop: the normalized
/// early-exercise premium max(V/K - European/K, 0) at a single
/// dimensionless point (x0, tau'_0, ln kappa_0), from a direct PDE solve
/// with sigma_eff = sqrt(2), r = kappa, q = 0.  Returns 0.0 if the solve
/// or the slice spline fails.  Exposed only so the probe's own PDE domain
/// coverage can be regression-tested (#480).
double dimensionless_reference_eep(double x0, double tau_prime_0,
                                   double ln_kappa_0, double K_ref,
                                   OptionType option_type);

}  // namespace mango::detail
```
In `dimensionless_adaptive.cpp`: add `#include "mango/option/table/dimensionless/dimensionless_adaptive_detail.hpp"`; move `reference_eep` out of the anonymous namespace into `namespace detail { ... }` renamed `dimensionless_reference_eep` (same body); replace the call `reference_eep(x0, tp0, lk0, K_ref, params.option_type)` in the probe loop with `detail::dimensionless_reference_eep(...)`.

In `src/option/table/dimensionless/BUILD.bazel`, `dimensionless_adaptive`: `hdrs = ["dimensionless_builder.hpp", "dimensionless_adaptive_detail.hpp"]` and add `"//src/option/table:covering_grid",` to its deps.

Run: `bazel build //src/option/table/dimensionless/... && bazel test //tests:dimensionless_adaptive_test`
Expected: builds, PASS.

- [ ] **Step 2: Write the failing test**

Append inside the anonymous namespace of `tests/dimensionless_adaptive_test.cc`, adding includes `"mango/option/table/dimensionless/dimensionless_adaptive_detail.hpp"`, `"mango/option/table/dimensionless/dimensionless_european.hpp"`, `"mango/option/american_option.hpp"`, `<algorithm>`:
```cpp
// Regression (#480, S5): the adaptive loop's ground-truth probe solved a
// normalized contract with maturity max(1.01 tau'_0, 0.02) gridless, so
// its half-width was 5 * sqrt(2) * sqrt(0.02) ~= 1.0 and any probe with
// |x0| > 1 read a cubic-spline extrapolation -- a wrong reference that
// silently misdirects refinement.  This probe sits 0.3 beyond that edge.
// Oracle: the same contract solved directly at spot = K e^x0 (High
// profile); its dollar value is divided by K before subtracting the
// normalized European, matching what the probe returns.
// Pre-fix abs error on this branch's parent: <FILL FROM STEP 3>.
TEST(DimensionlessAdaptiveTest, ReferenceEepCoversFarProbe) {
    const double x0 = -1.3, tp = 0.005, lk = 0.0, K = 100.0;
    const double kappa = std::exp(lk);

    const double got = detail::dimensionless_reference_eep(
        x0, tp, lk, K, OptionType::PUT);

    PricingParams p(
        OptionSpec{.spot = K * std::exp(x0), .strike = K, .maturity = tp,
                   .rate = kappa, .dividend_yield = 0.0,
                   .option_type = OptionType::PUT},
        std::sqrt(2.0));
    auto solver = AmericanOptionSolver::create(
        p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
    ASSERT_TRUE(solver.has_value());
    auto am = solver->solve();
    ASSERT_TRUE(am.has_value());
    const double ref = std::max(
        am->value() / K - dimensionless_european(x0, tp, kappa, OptionType::PUT),
        0.0);

    // Tolerance (V/K): <FILL FROM STEP 5>.
    constexpr double TOL = 1e-4;
    EXPECT_NEAR(got, ref, TOL);
    EXPECT_GT(ref, 0.0) << "probe must sit where the EEP is non-trivial";
}
```
In `tests/BUILD.bazel`, `dimensionless_adaptive_test` deps add `"//src/option:american_option",` and `"//src/option/table/dimensionless:dimensionless_european",`.

- [ ] **Step 3: Run on the unfixed code; record the failure**

Run: `bazel test //tests:dimensionless_adaptive_test --test_filter='DimensionlessAdaptiveTest.ReferenceEepCoversFarProbe' --test_output=all`
Expected: FAIL. If it passes (extrapolation happens to be benign here), move the probe further out (x0 = −1.6, then −2.0) until it fails, update the comment, and only then continue.

- [ ] **Step 4: Apply the S5 fix**

In `detail::dimensionless_reference_eep` replace
```cpp
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    ...
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/false);
```
with
```cpp
    const auto accuracy = make_grid_accuracy(GridAccuracyProfile::Ultra);
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(accuracy);
    ... (snapshot + batch construction unchanged) ...
    // The probe is read at x0, which the solver's own n_sigma*sigma*sqrt(T)
    // domain need not contain; materialize a grid that covers it (#480).
    const std::array<double, 1> reach = {x0};
    auto covering = detail::materialize_covering_grid(
        accuracy, std::span<const PricingParams>(batch),
        std::span<const double>(reach));
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/false,
                                     nullptr, covering);
```
Add `#include "mango/option/table/covering_grid.hpp"` and `<array>`.

- [ ] **Step 5: Run, verify green; pin tolerance**

Run: `bazel test //tests:dimensionless_adaptive_test --test_output=all`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git checkout MODULE.bazel.lock
git add src/option/table/dimensionless/ tests/dimensionless_adaptive_test.cc tests/BUILD.bazel
git commit -m "Cover the dimensionless reference probe's PDE domain

The adaptive loop's ground-truth probe solved gridless and read its
slice spline at x0, which the n_sigma*sigma*sqrt(T) domain need not
contain; a wrong reference silently misdirects refinement.  Expose the
probe as detail::dimensionless_reference_eep so the fix is testable
(#480)."
```

---

### Task 7: Full verification, pinned numbers, docs

**Files:**
- Possibly modify: `CLAUDE.md` (Pattern 4's "549 bps" sentence) and `tests/iv_solver_factory_slow_test.cc:100-140` comments, only if the measurement moved materially.

- [ ] **Step 1: Full suite (includes the nightly `slow` targets, since no tag filter is set locally)**

Run: `bazel test //... --test_output=errors`
Expected: `Executed 151 out of 151 tests: 151 tests pass.`

- [ ] **Step 2: Re-measure the pinned Chebyshev-path numbers**

Run: `bazel test //tests:iv_solver_factory_slow_test --test_filter='IVSolverFactorySegmented.DocumentedAdaptiveDiscreteDividendConfig' --test_output=all`
Read the reported `achieved_max_error`. It was 0.0549 (549 bps). If it changed by more than ~10 %, update the numbers quoted in that test's comment block and in `CLAUDE.md` Pattern 4 ("measures 549 bps against the 2,000 bps viability bound") to the new measurement; if it got *worse*, stop and report — that is not expected from removing extrapolation.

- [ ] **Step 3: CI parity builds**

Run: `bazel build //benchmarks/... //src/python:mango_option`
Expected: both build clean, no warnings (`-Werror` is on for `//src/...`).

- [ ] **Step 4: Runtime check for CI placement**

From the Bazel log of Step 1, note wall time for `adaptive_surface_build_integration_test` (T2 + T3 landed there). If the test target now exceeds its `long` timeout headroom or T2/T3 together add more than ~2 minutes, move T2 and T3 to `tests/adaptive_surface_build_slow_test.cc` (the nightly split, tagged `slow`, never `manual`) and say so in the commit body.

- [ ] **Step 5: Commit any doc/pin updates**

```bash
git checkout MODULE.bazel.lock
git add -A CLAUDE.md tests/
git commit -m "Refresh pinned Chebyshev accuracy numbers after #480"
```
(skip if nothing changed).

---

## Deviations log

Record here, during execution, anything done differently from this plan or the spec and why (the pre-merge review reads it).

- Task 3b (controller-added): boundary clearance `reach + 3·σ_max√T` in
  `ensure_moneyness_coverage` (spec D11). Tasks 2/3 tolerance rationales
  re-justified (values unchanged).
- Task 2: tolerances split by query class (`TOL_NODE = 1e-5` on the node
  endpoints, `TOL_USER = 0.2` on the user strikes). Measured pre-fix error at
  the user strikes was only 0.0139, so the spec's claim that endpoint-node
  garbage visibly contaminates user strikes through the global polynomial was
  overstated for this chain; the node-endpoint assertions are the discriminator
  (27.85 pre-fix → 6.7e-9 post-fix).
- Task 3: residual ≈0.021 at S = 50 after Task 3b is off-node Chebyshev fit
  error over a 9-node m axis, not coverage (edge-node sweep max 9.8e-5);
  left as-is, `TOL_COVERAGE = 0.25` documents it.

## Follow-ups to file at finish (not in this PR)

- Segmented Chebyshev dividend-timing skew: the table's PDE contract uses `maturity = 1.01·τ_max` and anchors `Dividend::calendar_time` to that padded maturity, so at any snapshot the event sits 1 % of τ_max earlier than the user's contract implies (spec D10).

---

## Rework tasks (composable coverage API — spec §"Rework", D12–D16)

Global constraints above still bind. Additional invariant for every rework
task: **no pinned number changes.** T2–T6, the #479 regressions, the B-spline
bit-identity goldens, and the 549 bps pin must pass untouched; if any moves,
stop and report — the rework is API-only.

### Task 8: `log_moneyness_coverage` on `GridAccuracyParams`, honoured by the estimators and every `solve_batch` routing

**Files:**
- Modify: `src/option/grid_spec_types.hpp`, `src/option/grid_spec_types.cpp`
- Modify: `src/option/american_option_batch.cpp` (`is_normalized_eligible`, `trace_ineligibility_reason`: judge on a copy with coverage cleared; `solve_regular_batch`: resolve a `GridAccuracyParams` custom grid over the batch / per contract)
- Create: `tests/grid_spec_types_test.cc`; Modify: `tests/BUILD.bazel` (new small `grid_spec_types_test` on `//src/option:grid_spec_types`), `tests/american_option_batch_test.cc` (five solver-level tests), `tests/covering_grid_test.cc` (temporary exact-identity test against the helper; the file is deleted in Task 9)

**Interfaces (produced):**
```cpp
struct LogMoneynessRange { double lo = 0.0, hi = 0.0;
    static std::optional<LogMoneynessRange> of(std::span<const double> nodes);
    [[nodiscard]] double reach_from(double x0) const; };
// GridAccuracyParams gains (appended after max_time_steps):
std::optional<LogMoneynessRange> log_moneyness_coverage;   // default nullopt
double coverage_clearance_sigmas = 3.0;
PDEGridConfig estimate_batch_pde_grid_config(std::span<const PricingParams> batch,
                                             const GridAccuracyParams& accuracy);
```

- [ ] **Step 1: Write the failing estimator tests** — `tests/grid_spec_types_test.cc` (SPDX header; `#include <gtest/gtest.h>`, `"mango/option/grid_spec_types.hpp"`, `<cmath>`, `<limits>`, `<vector>`; `namespace mango { namespace {`):

```cpp
PricingParams put(double sigma, double T, double spot = 100.0) {
    return PricingParams(OptionSpec{.spot = spot, .strike = 100.0, .maturity = T,
        .rate = 0.05, .dividend_yield = 0.0, .option_type = OptionType::PUT}, sigma);
}

TEST(LogMoneynessRange, OfIsOrderIndependentAndEmptyIsNullopt) {
    std::vector<double> permuted = {0.0, -0.51, 0.10};
    auto r = LogMoneynessRange::of(permuted);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->lo, -0.51);
    EXPECT_DOUBLE_EQ(r->hi, 0.10);
    EXPECT_FALSE(LogMoneynessRange::of({}).has_value());
}

TEST(LogMoneynessRange, ReachIsMeasuredFromX0) {
    LogMoneynessRange r{-0.5, 0.5};
    EXPECT_DOUBLE_EQ(r.reach_from(0.0), 0.5);
    EXPECT_DOUBLE_EQ(r.reach_from(0.2), 0.7);    // inside: to the far endpoint
    EXPECT_DOUBLE_EQ(r.reach_from(2.0), 2.5);    // outside: to the far endpoint
}

// D11 rule on a single normalized contract: edge = max(1.1*reach, reach + 3s).
TEST(EstimatePdeGrid, SingleContractCoversWithItsOwnSigmaSqrtT) {
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.5, 0.5};
    auto [grid, td] = estimate_pde_grid(put(0.20, 0.25), acc);   // s = 0.1
    EXPECT_NEAR(grid.x_min(), -(0.5 + 3.0 * 0.1), 1e-9);
    EXPECT_NEAR(grid.x_max(),  (0.5 + 3.0 * 0.1), 1e-9);
}

// Review round 1: the range is absolute ln(S/K); a contract off the money
// must still get [lo - 3s, hi + 3s] inside its domain (x0 = ln 1.2 ~ 0.182).
TEST(EstimatePdeGrid, OffTheMoneyContractStillCoversTheAbsoluteRange) {
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.5, 0.5};
    auto [grid, td] = estimate_pde_grid(put(0.20, 0.25, /*spot=*/120.0), acc);  // s = 0.1
    EXPECT_LE(grid.x_min(), -0.8 + 1e-9);
    EXPECT_GE(grid.x_max(),  0.8 - 1e-9);
}

TEST(EstimatePdeGrid, ClearanceIsConfigurable) {
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.5, 0.5};
    acc.coverage_clearance_sigmas = 6.0;
    auto [grid, td] = estimate_pde_grid(put(0.20, 0.25), acc);   // s = 0.1
    EXPECT_NEAR(grid.x_min(), -(0.5 + 6.0 * 0.1), 1e-9);
}

TEST(EstimatePdeGrid, CoverageInsideNSigmaDomainLeavesGridUnchanged) {
    GridAccuracyParams plain;
    GridAccuracyParams covered = plain;
    covered.log_moneyness_coverage = LogMoneynessRange{-0.51, 0.51};   // 0.51/0.5 + 3 = 4.02 < 5
    auto [g1, t1] = estimate_pde_grid(put(0.50, 1.0), plain);
    auto [g2, t2] = estimate_pde_grid(put(0.50, 1.0), covered);
    EXPECT_DOUBLE_EQ(g1.x_min(), g2.x_min());
    EXPECT_EQ(g1.n_points(), g2.n_points());
    EXPECT_EQ(t1.n_steps(), t2.n_steps());
}

TEST(EstimatePdeGrid, NonFiniteEndpointDisablesCoverageAndNegativeClearanceIsZero) {
    GridAccuracyParams plain;
    GridAccuracyParams nan = plain;
    nan.log_moneyness_coverage = LogMoneynessRange{std::numeric_limits<double>::quiet_NaN(), 0.5};
    auto [g1, t1] = estimate_pde_grid(put(0.20, 0.25), plain);
    auto [g2, t2] = estimate_pde_grid(put(0.20, 0.25), nan);
    EXPECT_DOUBLE_EQ(g1.x_min(), g2.x_min());

    GridAccuracyParams neg;
    neg.log_moneyness_coverage = LogMoneynessRange{-2.0, 2.0};
    neg.coverage_clearance_sigmas = -4.0;
    auto [g3, t3] = estimate_pde_grid(put(0.20, 0.25), neg);
    EXPECT_NEAR(g3.x_min(), -(2.0 * 1.1), 1e-9);   // the 10% floor alone
}

// Batch: one n_sigma for all, from the largest sigma*sqrt(T); the union's edge
// is max(1.1*reach, reach + 3*s_max).  Identical to widening n_sigma by hand.
TEST(EstimateBatchPdeGrid, CoverageEdgeIsReachPlusThreeSigmaMaxSqrtT) {
    std::vector<PricingParams> batch = {put(0.10, 0.1), put(0.15, 0.1), put(0.20, 0.1)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.51, 0.51};
    const double s = 0.20 * std::sqrt(0.1);
    const double expected_edge = std::max(0.51 * 1.1, 0.51 + 3.0 * s);   // ~0.700
    auto [grid, td] = estimate_batch_pde_grid(batch, acc);
    EXPECT_NEAR(grid.x_min(), -expected_edge, 1e-9);
    EXPECT_NEAR(grid.x_max(),  expected_edge, 1e-9);
    GridAccuracyParams manual;
    manual.n_sigma = std::max(manual.n_sigma, expected_edge / s);
    auto [grid2, td2] = estimate_batch_pde_grid(batch, manual);
    EXPECT_DOUBLE_EQ(grid.x_min(), grid2.x_min());
    EXPECT_EQ(grid.n_points(), grid2.n_points());
    EXPECT_EQ(td.n_steps(), td2.n_steps());
}

TEST(EstimateBatchPdeGridConfig, WrapsTheSharedGrid) {
    std::vector<PricingParams> batch = {put(0.10, 0.1), put(0.20, 0.1)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-0.51, 0.51};
    auto [grid, td] = estimate_batch_pde_grid(batch, acc);
    auto config = estimate_batch_pde_grid_config(batch, acc);
    EXPECT_DOUBLE_EQ(config.grid_spec.x_min(), grid.x_min());
    EXPECT_EQ(config.n_time, td.n_steps());
    EXPECT_TRUE(config.mandatory_times.empty());
}

// Exact goldens recorded from detail::materialize_covering_grid on the
// parent revision (see covering_grid_test.cc's identity test, Step 3), so
// the fold cannot drift after the helper is deleted.  Values: <FILL IN STEP 5>.
TEST(EstimateBatchPdeGrid, GoldensMatchTheRetiredHelper) {
    // (a) clamp-binding Ultra chain batch (T2-like): sigma nodes over
    //     [0.01, 0.225] at T = 0.694375, coverage [-1.0881, 1.0881].
    // (b) dividend batch: sigma {0.05, 0.15}, T = 0.2525, one dividend
    //     {calendar_time 0.1, amount 1.0}, coverage [-0.8452, 0.8250].
    // For each: EXPECT_DOUBLE_EQ on x_min, x_max; EXPECT_EQ on n_points, n_time.
}
```
`tests/BUILD.bazel`, next to `american_option_batch_test`:
```python
cc_test(
    name = "grid_spec_types_test",
    size = "small",
    srcs = ["grid_spec_types_test.cc"],
    deps = [
        "//src/option:grid_spec_types",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```
Run: `bazel test //tests:grid_spec_types_test` — expected: does not compile.

- [ ] **Step 2: Write the failing solver tests** (append to `tests/american_option_batch_test.cc`; add `<cmath>`, `<variant>`):

```cpp
namespace {
PricingParams atm_put(double sigma, double T, std::vector<Dividend> divs = {}) {
    PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = T,
        .rate = 0.05, .dividend_yield = 0.0, .option_type = OptionType::PUT}, sigma);
    p.discrete_dividends = std::move(divs);
    return p;
}
}  // namespace

// D14: eligibility is judged with coverage cleared.  Reach 3.0 makes the
// coverage-widened first-contract grid 7.2 wide (> MAX_WIDTH = 5.8) while its
// base grid (half-width 1.0, margin 1.0) is eligible, so the batch must STILL
// route through the normalized chain: each group solves on its own grid,
// covering with its own clearance, and the two grids differ.  Without D14
// the batch would fall to the shared regular path and both results would
// share one grid.
TEST(BatchAmericanOptionSolver, CoverageDoesNotChangeRoutingAndEveryGroupCovers) {
    std::vector<PricingParams> batch = {atm_put(0.20, 1.0), atm_put(0.40, 1.0)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-3.0, 3.0};
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(acc);
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/true);
    ASSERT_EQ(result.failed_count, 0u);
    auto x0 = result.results[0]->grid()->x();
    auto x1 = result.results[1]->grid()->x();
    EXPECT_LE(x0.front(), -(3.0 + 3.0 * 0.20) + 1e-9);
    EXPECT_LE(x1.front(), -(3.0 + 3.0 * 0.40) + 1e-9);
    EXPECT_NE(x0.front(), x1.front()) << "per-group grids expected (normalized routing)";
}

// Shared regular route (dividends make the batch ineligible): one grid for
// all, its edge set by the LARGEST sigma*sqrt(T).
TEST(BatchAmericanOptionSolver, SharedRegularRouteCoversWithSigmaMax) {
    const std::vector<Dividend> divs = {Dividend{.calendar_time = 0.5, .amount = 1.0}};
    std::vector<PricingParams> batch = {atm_put(0.20, 1.0, divs), atm_put(0.40, 1.0, divs)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-1.5, 1.5};
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(acc);
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/true);
    ASSERT_EQ(result.failed_count, 0u);
    auto x0 = result.results[0]->grid()->x();
    auto x1 = result.results[1]->grid()->x();
    EXPECT_DOUBLE_EQ(x0.front(), x1.front());
    EXPECT_LE(x0.front(), -(1.5 + 3.0 * 0.40) + 1e-9);
    EXPECT_GE(x0.back(),   (1.5 + 3.0 * 0.40) - 1e-9);
}

// Per-contract route: a probe 1.3 beyond the strike with s = 0.2*sqrt(0.02).
TEST(BatchAmericanOptionSolver, PerContractRouteHonoursCoverage) {
    std::vector<PricingParams> batch = {atm_put(0.20, 0.02)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-1.3, -1.3};
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(acc);
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/false);
    ASSERT_TRUE(result.results[0].has_value());
    EXPECT_LE(result.results[0]->grid()->x().front(),
              -(1.3 + 3.0 * 0.20 * std::sqrt(0.02)) + 1e-9);
}

// D15: an accuracy spec passed as custom_grid is estimated the same way the
// solver's own accuracy is -- over the batch on the shared path (edge from
// sigma_max, not from params[0]) ...
TEST(BatchAmericanOptionSolver, AccuracyCustomGridIsEstimatedOverTheBatch) {
    const std::vector<Dividend> divs = {Dividend{.calendar_time = 0.5, .amount = 1.0}};
    std::vector<PricingParams> batch = {atm_put(0.20, 1.0, divs), atm_put(0.40, 1.0, divs)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-1.5, 1.5};
    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/true, nullptr, PDEGridSpec{acc});
    ASSERT_EQ(result.failed_count, 0u);
    EXPECT_LE(result.results[0]->grid()->x().front(), -(1.5 + 3.0 * 0.40) + 1e-9);
}

// ... and per contract otherwise (two sigmas -> two different grids; before
// the repair both contracts reused params[0]'s grid).
TEST(BatchAmericanOptionSolver, AccuracyCustomGridIsEstimatedPerContract) {
    std::vector<PricingParams> batch = {atm_put(0.20, 1.0), atm_put(0.40, 1.0)};
    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/false, nullptr,
                                     PDEGridSpec{GridAccuracyParams{}});
    ASSERT_EQ(result.failed_count, 0u);
    EXPECT_NEAR(result.results[0]->grid()->x().front(), -1.0, 1e-9);   // 5 * 0.2
    EXPECT_NEAR(result.results[1]->grid()->x().front(), -2.0, 1e-9);   // 5 * 0.4
}
```
Run: `bazel test //tests:american_option_batch_test` — expected: does not compile.

- [ ] **Step 3: Write the exact-identity test against the helper** (append to `tests/covering_grid_test.cc`; it dies with the file in Task 9; add `"mango/option/grid_spec_types.hpp"` include and, if needed, `//src/option:grid_spec_types` to its deps):

```cpp
// Rework identity: the estimator with log_moneyness_coverage must produce
// EXACTLY the grid materialize_covering_grid produces (D12 numeric-identity
// claim).  Three batch shapes: clamp-binding Ultra chain, High B-spline-like,
// dividend segmented.
TEST(CoverageRework, EstimatorReproducesTheHelperExactly) {
    struct Case { const char* name; GridAccuracyParams acc; std::vector<PricingParams> batch; std::vector<double> nodes; };
    auto mk = [](double sigma, double T, std::vector<Dividend> divs = {}) {
        PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = T,
            .rate = 0.05, .dividend_yield = 0.0, .option_type = OptionType::PUT}, sigma);
        p.discrete_dividends = std::move(divs);
        return p;
    };
    std::vector<Case> cases;
    cases.push_back({"ultra-chain", make_grid_accuracy(GridAccuracyProfile::Ultra),
        {mk(0.01, 0.694375), mk(0.1175, 0.694375), mk(0.225, 0.694375)}, {-1.0881, 0.0, 1.0881}});
    cases.push_back({"high-bspline", make_grid_accuracy(GridAccuracyProfile::High),
        {mk(0.10, 0.500001), mk(0.15, 0.500001), mk(0.20, 0.500001)}, {-0.3796, 0.0, 0.3796}});
    const std::vector<Dividend> divs = {Dividend{.calendar_time = 0.1, .amount = 1.0}};
    cases.push_back({"dividend", make_grid_accuracy(GridAccuracyProfile::Ultra),
        {mk(0.05, 0.2525, divs), mk(0.15, 0.2525, divs)}, {-0.8452, 0.0, 0.8250}});
    for (auto& c : cases) {
        auto helper = mango::detail::materialize_covering_grid(c.acc, c.batch, c.nodes);
        auto* h = std::get_if<PDEGridConfig>(&helper);
        ASSERT_NE(h, nullptr) << c.name;
        GridAccuracyParams acc = c.acc;
        acc.log_moneyness_coverage = LogMoneynessRange::of(c.nodes);
        auto e = estimate_batch_pde_grid_config(c.batch, acc);
        EXPECT_DOUBLE_EQ(e.grid_spec.x_min(), h->grid_spec.x_min()) << c.name;
        EXPECT_DOUBLE_EQ(e.grid_spec.x_max(), h->grid_spec.x_max()) << c.name;
        EXPECT_EQ(e.grid_spec.n_points(), h->grid_spec.n_points()) << c.name;
        EXPECT_EQ(e.grid_spec.type(), h->grid_spec.type()) << c.name;
        EXPECT_EQ(e.n_time, h->n_time) << c.name;
        auto ge = e.grid_spec.generate(); auto gh = h->grid_spec.generate();
        auto se = ge.view().span(); auto sh = gh.view().span();
        ASSERT_EQ(se.size(), sh.size()) << c.name;
        for (size_t i = 0; i < se.size(); ++i) EXPECT_EQ(se[i], sh[i]) << c.name << " i=" << i;
        // Print x_min/x_max/n_points/n_time for the goldens in Step 5 (remove afterwards).
    }
}
```
Run: `bazel test //tests:covering_grid_test` — expected: does not compile.

- [ ] **Step 4: Implement**

`grid_spec_types.hpp`: add `#include <optional>`, `<span>`; define `LogMoneynessRange` above `GridAccuracyParams`; append the two fields to `GridAccuracyParams` with the spec's doc comment; declare `estimate_batch_pde_grid_config` after `estimate_batch_pde_grid`.

`grid_spec_types.cpp`:
```cpp
std::optional<LogMoneynessRange>
LogMoneynessRange::of(std::span<const double> nodes) {
    if (nodes.empty()) return std::nullopt;
    const auto [lo, hi] = std::minmax_element(nodes.begin(), nodes.end());
    return LogMoneynessRange{*lo, *hi};
}

double LogMoneynessRange::reach_from(double x0) const {
    return std::max(std::abs(lo - x0), std::abs(hi - x0));
}

namespace {

/// n_sigma (in units of s = sigma*sqrt(T)) that puts [lo, hi] at least
/// `clearance` diffusion lengths inside a domain symmetric about x0, with a
/// 10 % widening of the reach as a floor.  Boundary error (and, with discrete
/// dividends, the jump condition's edge fallback) diffuses inward about one
/// sigma*sqrt(T), and a clearance that is a fixed fraction of the reach is far
/// thinner than that whenever the reach is large relative to sigma*sqrt(T)
/// (measured: 0.84 per $100 at the edge nodes of a segmented Chebyshev fit at
/// sigma ~0.19 with a 10 % clearance).
double required_n_sigma(const LogMoneynessRange& range, double x0, double s,
                        double clearance) {
    constexpr double MARGIN = 1.1;
    const double reach_sigmas = range.reach_from(x0) / std::max(s, 1e-10);
    return std::max(reach_sigmas * MARGIN,
                    reach_sigmas + std::max(clearance, 0.0));
}

bool coverage_usable(const GridAccuracyParams& accuracy) {
    return accuracy.log_moneyness_coverage
        && std::isfinite(accuracy.log_moneyness_coverage->lo)
        && std::isfinite(accuracy.log_moneyness_coverage->hi);
}

/// Fold coverage into n_sigma for one contract (its own s) and clear it.
GridAccuracyParams fold_coverage(GridAccuracyParams accuracy, double x0, double s) {
    if (!coverage_usable(accuracy)) { accuracy.log_moneyness_coverage.reset(); return accuracy; }
    accuracy.n_sigma = std::max(accuracy.n_sigma,
        required_n_sigma(*accuracy.log_moneyness_coverage, x0, s,
                         accuracy.coverage_clearance_sigmas));
    accuracy.log_moneyness_coverage.reset();
    return accuracy;
}

/// Fold coverage for a set of contracts sharing one grid: the largest
/// sigma*sqrt(T) sets the clearance and the largest required widening is
/// applied to all, so the contract with s_max alone covers the range.
GridAccuracyParams fold_coverage(GridAccuracyParams accuracy,
                                 std::span<const PricingParams> batch) {
    if (!coverage_usable(accuracy)) { accuracy.log_moneyness_coverage.reset(); return accuracy; }
    double s_max = 0.0;
    for (const auto& p : batch) s_max = std::max(s_max, p.volatility * std::sqrt(p.maturity));
    double required = 0.0;
    for (const auto& p : batch) {
        required = std::max(required, required_n_sigma(
            *accuracy.log_moneyness_coverage, std::log(p.spot / p.strike), s_max,
            accuracy.coverage_clearance_sigmas));
    }
    accuracy.n_sigma = std::max(accuracy.n_sigma, required);
    accuracy.log_moneyness_coverage.reset();
    return accuracy;
}

}  // namespace
```
`estimate_pde_grid`: first statement `const GridAccuracyParams acc = fold_coverage(accuracy, std::log(params.spot / params.strike), params.volatility * std::sqrt(params.maturity));` and every later `accuracy.` reads `acc.`. `estimate_batch_pde_grid`: after the empty-batch early return, `const GridAccuracyParams acc = fold_coverage(accuracy, params);`, use `acc` in the loop and for `alpha` / `max_time_steps`. Add `estimate_batch_pde_grid_config` (estimate, then `PDEGridConfig{grid_spec, time_domain.n_steps(), {}}`).

`american_option_batch.cpp`:
- `is_normalized_eligible` and `trace_ineligibility_reason`: replace `estimate_pde_grid(first, grid_accuracy_)` with
  ```cpp
      // Eligibility is judged on the contract's own kink-region grid, not on
      // the coverage-widened one (D14; #487 tracks judging the grid solved on).
      GridAccuracyParams base = grid_accuracy_;
      base.log_moneyness_coverage.reset();
      auto [grid_spec, time_domain] = estimate_pde_grid(first, base);
  ```
- `solve_regular_batch` (D15): keep the existing `resolve_grid` lambda for the `PDEGridConfig` alternative only. Then
  ```cpp
      const GridAccuracyParams* custom_accuracy =
          custom_grid ? std::get_if<GridAccuracyParams>(&*custom_grid) : nullptr;
      const PDEGridConfig* custom_config =
          custom_grid ? std::get_if<PDEGridConfig>(&*custom_grid) : nullptr;
      const GridAccuracyParams& accuracy = custom_accuracy ? *custom_accuracy : grid_accuracy_;
      // A concrete grid is resolved once and shared verbatim; an accuracy spec
      // is estimated the way the solver's own accuracy is: over the whole
      // batch on the shared path, per contract otherwise.
      std::optional<std::pair<GridSpec<double>, TimeDomain>> resolved_config;
      if (custom_config && !params.empty()) resolved_config = resolve_grid(*custom_config, params[0]);
      std::optional<std::pair<GridSpec<double>, TimeDomain>> shared_grid;
      if (use_shared_grid) {
          shared_grid = resolved_config ? resolved_config
                                        : std::optional{estimate_batch_pde_grid(params, accuracy)};
      }
      ... per contract, non-shared branch:
              auto [grid_spec, time_domain] = resolved_config
                  ? resolved_config.value()
                  : estimate_pde_grid(params[i], accuracy);
  ```
  If anything else in the function used `resolved_custom_grid` (workspace sizing), give it `shared_grid` on the shared path and the per-contract estimate otherwise; report what you found.

- [ ] **Step 5: Run, verify green; record goldens.** `bazel test //tests:grid_spec_types_test //tests:american_option_batch_test //tests:covering_grid_test --test_output=all`. From the identity test's printout fill `GoldensMatchTheRetiredHelper` with the exact `x_min`, `x_max`, `n_points`, `n_time` for cases (a) and (c) as `EXPECT_DOUBLE_EQ` / `EXPECT_EQ`; remove the printout; re-run green. Then the no-behaviour-change check: `bazel test //tests:price_table_builder_test //tests:price_table_builder_custom_grid_test //tests:bspline_bit_identity_test //tests:iv_solver_test //tests:quantlib_accuracy_batch_test //tests:american_option_test --test_output=errors` — all PASS.

- [ ] **Step 6: Commit** (restore the lockfile first): subject "Add log-moneyness coverage to GridAccuracyParams" (46 chars); body: the requirement belongs on the accuracy spec (D12), eligibility judges the base grid (D14), accuracy custom grids are estimated over the batch / per contract (D15).

### Task 9: Builders declare coverage; delete the helper; bind in Python

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp` (S1, S2), `src/option/table/chebyshev/chebyshev_table_builder.cpp` (S3), `src/option/table/dimensionless/dimensionless_builder.cpp` (S4), `src/option/table/dimensionless/dimensionless_adaptive.cpp` (S5), `src/option/table/bspline/bspline_builder.cpp`, `src/option/table/bspline/bspline_builder.hpp`, `src/option/table/bspline/bspline_adaptive.cpp`, the `BUILD.bazel` files under `src/option/table/{chebyshev,dimensionless,bspline}` and `src/option/table` (drop the `covering_grid` dep / target)
- Delete: `src/option/table/covering_grid.hpp`, `src/option/table/covering_grid.cpp`, `tests/covering_grid_test.cc` (+ its `tests/BUILD.bazel` entry)
- Modify: `src/python/mango_bindings.cpp` (bind `LogMoneynessRange` and the two fields), `tests/test_bindings.py` (round-trip test)
- Modify: test comments only where they name `materialize_covering_grid` / `ensure_moneyness_coverage`; no tolerance or number changes.

**Routing preservation rule (D14):** do not add or remove any `set_grid_accuracy` call. Fill in this table in your report from the code (before → after) for every site: which accuracy object the solver holds, whether it is the same object as before, and whether `custom_grid` is explicit:

| Site | `set_grid_accuracy` before | after | custom_grid before | after |
|---|---|---|---|---|
| S1 chebyshev_adaptive (continuous) | Ultra | Ultra + coverage | materialized | `estimate_batch_pde_grid_config` |
| S2 chebyshev_adaptive (segmented) | Ultra | Ultra + coverage | materialized | `estimate_batch_pde_grid_config` |
| S3 chebyshev_table_builder | none | none (accuracy only passed to the estimator) | materialized | `estimate_batch_pde_grid_config(batch, GridAccuracyParams{} + coverage)` |
| S4 dimensionless_builder | Ultra | Ultra + coverage | materialized | `estimate_batch_pde_grid_config` |
| S5 dimensionless_adaptive | Ultra | Ultra + coverage | materialized | none (gridless, `use_shared_grid=false`) |
| bspline_builder primary (`estimate_pde_grid` member) | none | none | explicit (from member) | explicit (member sets coverage, calls `estimate_batch_pde_grid`) |
| bspline_builder fallback | none | none | materialized | `estimate_batch_pde_grid_config` |
| bspline_adaptive fallback / accuracy branch | none | none | materialized | `estimate_batch_pde_grid_config` |

Per-site shape (S1 shown; S2 and S4 the same; S3 with `GridAccuracyParams accuracy;`):
```cpp
            auto accuracy = make_grid_accuracy(GridAccuracyProfile::Ultra);
            // Every moneyness node is read from the slice splines, so the
            // solver must resolve the whole node span (spec D12).
            accuracy.log_moneyness_coverage = LogMoneynessRange::of(m_nodes);
            BatchAmericanOptionSolver solver;
            solver.set_grid_accuracy(accuracy);
            std::vector<double> tau_vec(tau_nodes.begin(), tau_nodes.end());
            solver.set_snapshot_times(std::span<const double>(tau_vec));
            // One shared grid for the whole cohort the cache stores together.
            auto batch_result = solver.solve_batch(
                std::span<const PricingParams>(batch), /*use_shared_grid=*/true,
                nullptr, estimate_batch_pde_grid_config(batch, accuracy));
```
S5: `accuracy.log_moneyness_coverage = LogMoneynessRange{x0, x0};` `solver.set_grid_accuracy(accuracy);` `solver.solve_batch(batch, /*use_shared_grid=*/false);`.
`PriceTableBuilderND::estimate_pde_grid`: replace the `ensure_moneyness_coverage` line with `accuracy.log_moneyness_coverage = LogMoneynessRange::of(axes.grids[0]);`. The two explicit-grid fallbacks and `bspline_adaptive.cpp`'s accuracy branch: `accuracy.log_moneyness_coverage = LogMoneynessRange::of(<m grid>)` and pass `estimate_batch_pde_grid_config(<batch>, accuracy)`; keep their existing `required_n_sigma` emulation; replace the paragraphs about materializing because of routing with one sentence: the cache wants one grid per cohort. Delete the "routing-only" comments at S1–S5.

Python (`src/python/mango_bindings.cpp`, next to `GridAccuracyParams`; `pybind11/stl.h` must be included for `std::optional`):
```cpp
    py::class_<mango::LogMoneynessRange>(m, "LogMoneynessRange")
        .def(py::init<>())
        .def(py::init([](double lo, double hi) { return mango::LogMoneynessRange{lo, hi}; }),
             py::arg("lo"), py::arg("hi"))
        .def_readwrite("lo", &mango::LogMoneynessRange::lo)
        .def_readwrite("hi", &mango::LogMoneynessRange::hi);
    // on GridAccuracyParams:
        .def_readwrite("log_moneyness_coverage", &mango::GridAccuracyParams::log_moneyness_coverage)
        .def_readwrite("coverage_clearance_sigmas", &mango::GridAccuracyParams::coverage_clearance_sigmas)
```
`tests/test_bindings.py` (follow the file's existing style):
```python
def test_grid_accuracy_coverage_roundtrip():
    p = mango.GridAccuracyParams()
    assert p.log_moneyness_coverage is None
    p.log_moneyness_coverage = mango.LogMoneynessRange(-0.5, 0.5)
    p.coverage_clearance_sigmas = 2.5
    assert p.log_moneyness_coverage.lo == -0.5 and p.log_moneyness_coverage.hi == 0.5
    solver = mango.BatchAmericanOptionSolver()
    solver.set_grid_accuracy_params(p)      # reachable from the solver API
```

- [ ] **Step 1:** edit all sites, delete the helper and its test, fix BUILD deps; `grep -rn "covering_grid\|materialize_covering_grid\|ensure_moneyness_coverage" src tests` returns nothing.
- [ ] **Step 2: Numeric-identity check** — `bazel test //tests:grid_spec_types_test //tests:american_option_batch_test //tests:adaptive_surface_build_integration_test //tests:chebyshev_surface_test //tests:dimensionless_builder_test //tests:dimensionless_adaptive_test //tests:price_table_builder_test //tests:price_table_builder_custom_grid_test //tests:bspline_bit_identity_test //tests:greeks_accuracy_test //tests:price_table_data_test //tests:adaptive_grid_builder_test //tests:python_bindings_test --test_output=errors`. All PASS with no tolerance edits (the Task 8 goldens are the exact-identity proof).
- [ ] **Step 3: Commit** (restore the lockfile first): subject "Declare node coverage on the accuracy spec" (42 chars); body: the helper leaked solver internals into seven call sites (spec §Rework, D12–D16); numbers unchanged; Python parity.

### Task 10: Full verification and docs

- [ ] `bazel test //... --test_output=errors` → 151 tests (`covering_grid_test` removed, `grid_spec_types_test` added); `bazel build //benchmarks/... //src/python:mango_option` clean.
- [ ] Re-measure the 549 bps pin (`IVSolverFactorySegmented.DocumentedAdaptiveDiscreteDividendConfig`) — must be 548.64 bps as before.
- [ ] Spec: add a one-line pointer at the top of `### 1. Relocate the covering-grid helpers`: "Superseded by §Rework (D12–D16); kept for the review history." Plan Deviations log: "Rework round: helper replaced by `GridAccuracyParams::log_moneyness_coverage`; no pinned number changed."
- [ ] Commit docs; push to PR #484.
