# Interpolation Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the price surface architecture behind concept-based layers so interpolation schemes, coordinate transforms, EEP strategies, and surface splits are independently pluggable.

**Architecture:** Four concept layers (SurfaceInterpolant → CoordinateTransform → EEPStrategy → SplitPolicy) compose at compile time with zero overhead. Old type aliases (StandardSurface, MultiKRefPriceSurface) are redefined as compositions of new types, preserving the public API. PriceTableSurfaceND stays as the B-spline interpolant; new types wrap it behind concepts.

**Tech Stack:** C++23 concepts, templates, GoogleTest. Bazel build system.

**Design doc:** `docs/plans/2026-02-10-interpolation-refactor-design.md`

---

## Migration Strategy

New types are built alongside old ones, then old aliases are swapped to point at new compositions. Tests pass at every commit. The swap is safe because behavior is identical — only the type machinery changes.

**Key decision:** `EEPSurfaceAdapter` stores `shared_ptr<const PriceTableSurfaceND<N>>` (same ownership as current code). This avoids changing builder interfaces. When Chebyshev arrives, it will use a different interpolant type.

**Affected files:** ~37 files (7 source, 6 headers, 2 impl, 14 tests, 8+ benchmarks). Most only need include updates after the alias swap.

---

### Task 1: Core concepts and BSpline adapter

**Files:**
- Create: `src/option/table/surface_concepts.hpp`
- Create: `src/option/table/bspline/bspline_interpolant.hpp`
- Create: `tests/surface_concepts_test.cc`
- Modify: `tests/BUILD.bazel`
- Modify: `src/option/table/BUILD.bazel`

**Step 1: Write the failing test**

```cpp
// tests/surface_concepts_test.cc
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/surface_concepts.hpp"
#include "mango/option/table/bspline/bspline_interpolant.hpp"

using namespace mango;

TEST(SurfaceConceptsTest, BSplineInterpolantSatisfiesConcept) {
    static_assert(SurfaceInterpolant<SharedBSplineInterp<4>, 4>);
    static_assert(SurfaceInterpolant<SharedBSplineInterp<3>, 3>);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:surface_concepts_test --test_output=all`
Expected: BUILD ERROR (headers don't exist yet)

**Step 3: Write the concepts header**

```cpp
// src/option/table/surface_concepts.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <concepts>

namespace mango {

/// Raw interpolation engine: eval + partial derivative at N-dim coordinates.
/// Implementations: SharedBSplineInterp<N>, ChebyshevTuckerND<N> (future).
template <typename S, size_t N>
concept SurfaceInterpolant = requires(const S& s, std::array<double, N> coords) {
    { s.eval(coords) } -> std::same_as<double>;
    { s.partial(size_t{}, coords) } -> std::same_as<double>;
};

/// Maps 5-param price query to N-dim interpolation coordinates + vega weights.
/// Implementations: StandardTransform4D, DimensionlessTransform3D (future).
template <typename T>
concept CoordinateTransform = requires(const T& t, double spot, double strike,
                                        double tau, double sigma, double rate) {
    { T::kDim } -> std::convertible_to<size_t>;
    { t.to_coords(spot, strike, tau, sigma, rate) }
        -> std::same_as<std::array<double, T::kDim>>;
    { t.vega_weights(spot, strike, tau, sigma, rate) }
        -> std::same_as<std::array<double, T::kDim>>;
};

/// Handles EEP decomposition: American = EEP * scale + European.
/// Implementations: AnalyticalEEP, IdentityEEP.
template <typename E>
concept EEPStrategy = requires(const E& e, double spot, double strike,
                                double tau, double sigma, double rate, double K_ref) {
    { e.european_price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.scale(strike, K_ref) } -> std::same_as<double>;
};

}  // namespace mango
```

**Step 4: Write the BSpline adapter**

```cpp
// src/option/table/bspline/bspline_interpolant.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table_surface.hpp"
#include <array>
#include <memory>

namespace mango {

/// Adapter that wraps shared_ptr<PriceTableSurfaceND<N>> to satisfy
/// SurfaceInterpolant. Preserves shared ownership semantics.
template <size_t N>
class SharedBSplineInterp {
public:
    explicit SharedBSplineInterp(std::shared_ptr<const PriceTableSurfaceND<N>> surface)
        : surface_(std::move(surface)) {}

    [[nodiscard]] double eval(const std::array<double, N>& coords) const {
        return surface_->value(coords);
    }

    [[nodiscard]] double partial(size_t axis, const std::array<double, N>& coords) const {
        return surface_->partial(axis, coords);
    }

    /// Access underlying surface (for metadata, axes, etc.)
    [[nodiscard]] const PriceTableSurfaceND<N>& surface() const { return *surface_; }

private:
    std::shared_ptr<const PriceTableSurfaceND<N>> surface_;
};

}  // namespace mango
```

**Step 5: Add BUILD targets**

Add to `src/option/table/BUILD.bazel`:
```python
cc_library(
    name = "surface_concepts",
    hdrs = ["surface_concepts.hpp"],
)

cc_library(
    name = "bspline_interpolant",
    hdrs = ["bspline/bspline_interpolant.hpp"],
    deps = [":price_table_surface"],
)
```

Add to `tests/BUILD.bazel`:
```python
cc_test(
    name = "surface_concepts_test",
    srcs = ["surface_concepts_test.cc"],
    deps = [
        "//src/option/table:surface_concepts",
        "//src/option/table:bspline_interpolant",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:surface_concepts_test --test_output=all`
Expected: PASS

**Step 7: Commit**

```bash
git add src/option/table/surface_concepts.hpp src/option/table/bspline/bspline_interpolant.hpp \
        tests/surface_concepts_test.cc src/option/table/BUILD.bazel tests/BUILD.bazel
git commit -m "Add SurfaceInterpolant concept and BSpline adapter"
```

---

### Task 2: Coordinate transform and EEP strategies

**Files:**
- Create: `src/option/table/transforms/standard_4d.hpp`
- Create: `src/option/table/eep/analytical_eep.hpp`
- Create: `src/option/table/eep/identity_eep.hpp`
- Modify: `tests/surface_concepts_test.cc`
- Modify: `src/option/table/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing tests**

Append to `tests/surface_concepts_test.cc`:

```cpp
#include "mango/option/table/transforms/standard_4d.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/identity_eep.hpp"

TEST(SurfaceConceptsTest, StandardTransform4DSatisfiesConcept) {
    static_assert(CoordinateTransform<StandardTransform4D>);
    static_assert(StandardTransform4D::kDim == 4);
}

TEST(SurfaceConceptsTest, AnalyticalEEPSatisfiesConcept) {
    static_assert(EEPStrategy<AnalyticalEEP>);
}

TEST(SurfaceConceptsTest, IdentityEEPSatisfiesConcept) {
    static_assert(EEPStrategy<IdentityEEP>);
}

TEST(StandardTransform4DTest, ToCoordsReturnsLogMoneyness) {
    StandardTransform4D xform;
    auto c = xform.to_coords(110.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_NEAR(c[0], std::log(110.0 / 100.0), 1e-12);  // x = ln(S/K)
    EXPECT_NEAR(c[1], 0.5, 1e-12);   // tau
    EXPECT_NEAR(c[2], 0.20, 1e-12);  // sigma
    EXPECT_NEAR(c[3], 0.05, 1e-12);  // rate
}

TEST(StandardTransform4DTest, VegaWeightsOnlySigmaAxis) {
    StandardTransform4D xform;
    auto w = xform.vega_weights(110.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_EQ(w[0], 0.0);
    EXPECT_EQ(w[1], 0.0);
    EXPECT_EQ(w[2], 1.0);
    EXPECT_EQ(w[3], 0.0);
}

TEST(IdentityEEPTest, EuropeanPriceIsZero) {
    IdentityEEP eep;
    EXPECT_EQ(eep.european_price(100, 100, 0.5, 0.20, 0.05), 0.0);
    EXPECT_EQ(eep.european_vega(100, 100, 0.5, 0.20, 0.05), 0.0);
}

TEST(IdentityEEPTest, ScaleIsStrikeOverKRef) {
    IdentityEEP eep;
    EXPECT_NEAR(eep.scale(110.0, 100.0), 1.1, 1e-12);
}

TEST(AnalyticalEEPTest, ScaleIsStrikeOverKRef) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    EXPECT_NEAR(eep.scale(110.0, 100.0), 1.1, 1e-12);
}

TEST(AnalyticalEEPTest, EuropeanPriceIsPositiveForATMPut) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double p = eep.european_price(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(p, 0.0);
    EXPECT_LT(p, 20.0);  // Reasonable range
}

TEST(AnalyticalEEPTest, EuropeanVegaIsPositive) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double v = eep.european_vega(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(v, 0.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:surface_concepts_test --test_output=all`
Expected: BUILD ERROR (headers don't exist yet)

**Step 3: Write StandardTransform4D**

```cpp
// src/option/table/transforms/standard_4d.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cmath>

namespace mango {

/// Identity coordinate transform for 4D (x, tau, sigma, rate) surfaces.
/// Direct sigma axis — vega is a single partial derivative.
struct StandardTransform4D {
    static constexpr size_t kDim = 4;

    [[nodiscard]] std::array<double, 4> to_coords(
        double spot, double strike, double tau, double sigma, double rate) const noexcept {
        return {std::log(spot / strike), tau, sigma, rate};
    }

    [[nodiscard]] std::array<double, 4> vega_weights(
        double /*spot*/, double /*strike*/, double /*tau*/,
        double /*sigma*/, double /*rate*/) const noexcept {
        return {0.0, 0.0, 1.0, 0.0};
    }
};

}  // namespace mango
```

**Step 4: Write AnalyticalEEP**

```cpp
// src/option/table/eep/analytical_eep.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

namespace mango {

/// EEP strategy using closed-form Black-Scholes European pricing.
/// Handles dividend yield. Used for standard (non-segmented) surfaces.
class AnalyticalEEP {
public:
    AnalyticalEEP(OptionType option_type, double dividend_yield)
        : option_type_(option_type), dividend_yield_(dividend_yield) {}

    [[nodiscard]] double european_price(
        double spot, double strike, double tau, double sigma, double rate) const {
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                .rate = rate, .dividend_yield = dividend_yield_,
                .option_type = option_type_}, sigma).solve().value();
        return eu.value();
    }

    [[nodiscard]] double european_vega(
        double spot, double strike, double tau, double sigma, double rate) const {
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                .rate = rate, .dividend_yield = dividend_yield_,
                .option_type = option_type_}, sigma).solve().value();
        return eu.vega();
    }

    [[nodiscard]] double scale(double strike, double K_ref) const noexcept {
        return strike / K_ref;
    }

private:
    OptionType option_type_;
    double dividend_yield_;
};

}  // namespace mango
```

**Step 5: Write IdentityEEP**

```cpp
// src/option/table/eep/identity_eep.hpp
// SPDX-License-Identifier: MIT
#pragma once

namespace mango {

/// No EEP decomposition. Surface stores V/K_ref directly.
/// european_price/vega return 0, scale returns K/K_ref.
/// Used for segmented dividend segments (NormalizedPrice content).
struct IdentityEEP {
    [[nodiscard]] double european_price(
        double, double, double, double, double) const noexcept { return 0.0; }

    [[nodiscard]] double european_vega(
        double, double, double, double, double) const noexcept { return 0.0; }

    [[nodiscard]] double scale(double strike, double K_ref) const noexcept {
        return strike / K_ref;
    }
};

}  // namespace mango
```

**Step 6: Add BUILD deps and run**

Add to `src/option/table/BUILD.bazel`:
```python
cc_library(
    name = "standard_transform_4d",
    hdrs = ["transforms/standard_4d.hpp"],
)

cc_library(
    name = "analytical_eep",
    hdrs = ["eep/analytical_eep.hpp"],
    deps = ["//src/option:european_option"],
)

cc_library(
    name = "identity_eep",
    hdrs = ["eep/identity_eep.hpp"],
)
```

Update test deps to include new targets.

Run: `bazel test //tests:surface_concepts_test --test_output=all`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/option/table/transforms/ src/option/table/eep/ \
        tests/surface_concepts_test.cc src/option/table/BUILD.bazel tests/BUILD.bazel
git commit -m "Add StandardTransform4D and EEP strategies"
```

---

### Task 3: EEPSurfaceAdapter template

**Files:**
- Create: `src/option/table/eep_surface_adapter.hpp`
- Modify: `tests/surface_concepts_test.cc`
- Modify: `src/option/table/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Append to `tests/surface_concepts_test.cc`:

```cpp
#include "mango/option/table/eep_surface_adapter.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/eep_transform.hpp"  // Old EEPPriceTableInner for comparison

// Build a small test surface for comparison tests.
// Uses the existing builder pipeline.
static std::shared_ptr<const mango::PriceTableSurface> make_test_surface() {
    using namespace mango;
    auto setup = PriceTableBuilder<4>::from_vectors(
        {-0.2, -0.1, 0.0, 0.1, 0.2},     // log-moneyness
        {0.1, 0.5, 1.0},                   // maturities
        {0.15, 0.20, 0.30},                // vols
        {0.03, 0.05},                      // rates
        100.0,                              // K_ref
        GridAccuracyParams{},
        OptionType::PUT, 0.02);
    auto& [builder, axes] = *setup;
    EEPDecomposer decomposer{OptionType::PUT, 100.0, 0.02};
    auto result = builder.build(axes, SurfaceContent::EarlyExercisePremium,
        [&](PriceTensor& t, const PriceTableAxes& a) { decomposer.decompose(t, a); });
    return result->surface;
}

TEST(EEPSurfaceAdapterTest, MatchesOldEEPPriceTableInner) {
    auto surface = make_test_surface();
    double K_ref = surface->metadata().K_ref;

    // Old path
    EEPPriceTableInner old_inner(surface, OptionType::PUT, K_ref, 0.02);

    // New path
    SharedBSplineInterp<4> interp(surface);
    StandardTransform4D xform;
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    EEPSurfaceAdapter adapter(std::move(interp), xform, eep, K_ref);

    // Compare at several query points
    struct TestPoint { double spot, strike, tau, sigma, rate; };
    TestPoint points[] = {
        {100, 100, 0.5, 0.20, 0.05},  // ATM
        {110, 100, 0.5, 0.20, 0.05},  // ITM put
        {90,  100, 0.5, 0.20, 0.05},  // OTM put
        {100, 100, 0.1, 0.30, 0.03},  // Short maturity
        {100, 100, 1.0, 0.15, 0.05},  // Long maturity
    };

    for (const auto& p : points) {
        PriceQuery q{p.spot, p.strike, p.tau, p.sigma, p.rate};
        double old_price = old_inner.price(q);
        double new_price = adapter.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(old_price, new_price, 1e-12)
            << "Mismatch at S=" << p.spot << " K=" << p.strike;

        double old_vega = old_inner.vega(q);
        double new_vega = adapter.vega(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(old_vega, new_vega, 1e-12)
            << "Vega mismatch at S=" << p.spot << " K=" << p.strike;
    }
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:surface_concepts_test --test_output=all`
Expected: BUILD ERROR (eep_surface_adapter.hpp doesn't exist)

**Step 3: Write EEPSurfaceAdapter**

```cpp
// src/option/table/eep_surface_adapter.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/surface_concepts.hpp"
#include <algorithm>

namespace mango {

/// Composes interpolant + coordinate transform + EEP strategy into a
/// complete price surface with price() and vega() methods.
///
/// Replaces EEPPriceTableInner (with AnalyticalEEP) and PriceTableInner
/// (with IdentityEEP). Any combination of interpolant, transform, and
/// EEP strategy works without code duplication.
template <typename Interp, typename Xform, typename EEP>
class EEPSurfaceAdapter {
public:
    EEPSurfaceAdapter(Interp interp, Xform xform, EEP eep, double K_ref)
        : interp_(std::move(interp))
        , xform_(std::move(xform))
        , eep_(std::move(eep))
        , K_ref_(K_ref)
    {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        double eep_val = std::max(0.0, raw);
        return eep_val * eep_.scale(strike, K_ref_)
             + eep_.european_price(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        if (raw <= 0.0) {
            // EEP clamped to zero — EEP vega is zero (consistent with price clamp)
            return eep_.european_vega(spot, strike, tau, sigma, rate);
        }
        auto w = xform_.vega_weights(spot, strike, tau, sigma, rate);
        double eep_vega = 0.0;
        for (size_t i = 0; i < Xform::kDim; ++i) {
            eep_vega += w[i] * interp_.partial(i, coords);
        }
        return eep_vega * eep_.scale(strike, K_ref_)
             + eep_.european_vega(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] const Interp& interpolant() const noexcept { return interp_; }
    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }

private:
    Interp interp_;
    Xform xform_;
    EEP eep_;
    double K_ref_;
};

}  // namespace mango
```

**Step 4: Add BUILD target and run**

Add to `src/option/table/BUILD.bazel`:
```python
cc_library(
    name = "eep_surface_adapter",
    hdrs = ["eep_surface_adapter.hpp"],
    deps = [":surface_concepts"],
)
```

Run: `bazel test //tests:surface_concepts_test --test_output=all`
Expected: ALL PASS

**Step 5: Also verify old EEPPriceTableInner has no vega clamp (design issue #1 fixed)**

The test above already verifies price/vega match. Note: the old `EEPPriceTableInner::vega()` does NOT clamp when EEP < 0, but the new adapter does. For typical surfaces this won't differ (EEP is non-negative after softplus floor), but the new behavior is mathematically correct.

**Step 6: Run all existing tests to verify no regressions**

Run: `bazel test //tests:surface_concepts_test //tests:eep_integration_test --test_output=all`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/option/table/eep_surface_adapter.hpp \
        tests/surface_concepts_test.cc src/option/table/BUILD.bazel tests/BUILD.bazel
git commit -m "Add EEPSurfaceAdapter template"
```

---

### Task 4: SplitSurface framework and split policies

**Files:**
- Create: `src/option/table/split_surface.hpp`
- Create: `src/option/table/splits/tau_segment.hpp`
- Create: `src/option/table/splits/multi_kref.hpp`
- Create: `tests/split_surface_test.cc`
- Modify: `src/option/table/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing tests**

```cpp
// tests/split_surface_test.cc
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"

using namespace mango;

// Mock inner that returns spot * strike (easy to verify routing)
struct MockInner {
    double offset = 0.0;
    double price(double spot, double strike, double tau, double sigma, double rate) const {
        return spot / strike + offset;
    }
    double vega(double spot, double strike, double tau, double sigma, double rate) const {
        return 0.1 + offset;
    }
};

// --- TauSegmentSplit tests ---

TEST(TauSegmentSplitTest, RoutesToCorrectSegment) {
    TauSegmentSplit split({0.0, 0.25, 0.50}, {0.25, 0.50, 1.0}, {}, {}, 100.0);

    auto br = split.bracket(100, 100, 0.10, 0.20, 0.05);  // tau=0.10 → segment 0
    EXPECT_EQ(br.count, 1u);
    EXPECT_EQ(br.entries[0].index, 0u);
    EXPECT_NEAR(br.entries[0].weight, 1.0, 1e-12);

    auto br2 = split.bracket(100, 100, 0.30, 0.20, 0.05);  // tau=0.30 → segment 1
    EXPECT_EQ(br2.entries[0].index, 1u);

    auto br3 = split.bracket(100, 100, 0.75, 0.20, 0.05);  // tau=0.75 → segment 2
    EXPECT_EQ(br3.entries[0].index, 2u);
}

TEST(TauSegmentSplitTest, ToLocalRemapsTauAndStrike) {
    // Segment 1: tau_start=0.25, tau_min=0.0, tau_max=0.24
    TauSegmentSplit split({0.0, 0.25}, {0.25, 0.50},
                          {0.0, 0.0}, {0.24, 0.24}, 100.0);

    auto [ls, lk, lt, lv, lr] = split.to_local(1, 110.0, 95.0, 0.30, 0.20, 0.05);
    EXPECT_NEAR(lt, 0.05, 1e-12);       // 0.30 - 0.25 = 0.05
    EXPECT_NEAR(lk, 100.0, 1e-12);      // strike → K_ref
    EXPECT_NEAR(ls, 110.0, 1e-12);      // spot unchanged
    EXPECT_NEAR(lv, 0.20, 1e-12);       // sigma unchanged
    EXPECT_NEAR(lr, 0.05, 1e-12);       // rate unchanged
}

TEST(TauSegmentSplitTest, NormalizeMultipliesByKRef) {
    TauSegmentSplit split({0.0}, {1.0}, {0.0}, {1.0}, 100.0);
    EXPECT_NEAR(split.normalize(0, 95.0, 0.5), 50.0, 1e-12);  // 0.5 * 100
}

TEST(TauSegmentSplitTest, DenormalizeIsIdentity) {
    TauSegmentSplit split({0.0}, {1.0}, {0.0}, {1.0}, 100.0);
    EXPECT_NEAR(split.denormalize(42.0, 100, 100, 0.5, 0.2, 0.05), 42.0, 1e-12);
}

// --- MultiKRefSplit tests ---

TEST(MultiKRefSplitTest, BracketsCorrectly) {
    MultiKRefSplit split({80.0, 100.0, 120.0});

    // Below first → clamp to index 0
    auto br = split.bracket(100, 70.0, 0.5, 0.20, 0.05);
    EXPECT_EQ(br.count, 1u);
    EXPECT_EQ(br.entries[0].index, 0u);

    // Between 80 and 100 → interpolate
    auto br2 = split.bracket(100, 90.0, 0.5, 0.20, 0.05);
    EXPECT_EQ(br2.count, 2u);
    EXPECT_EQ(br2.entries[0].index, 0u);  // K_ref=80
    EXPECT_EQ(br2.entries[1].index, 1u);  // K_ref=100
    EXPECT_NEAR(br2.entries[0].weight, 0.5, 1e-12);
    EXPECT_NEAR(br2.entries[1].weight, 0.5, 1e-12);

    // Above last → clamp to last
    auto br3 = split.bracket(100, 130.0, 0.5, 0.20, 0.05);
    EXPECT_EQ(br3.count, 1u);
    EXPECT_EQ(br3.entries[0].index, 2u);
}

TEST(MultiKRefSplitTest, ToLocalSetsStrikeToKRef) {
    MultiKRefSplit split({80.0, 100.0, 120.0});
    auto [ls, lk, lt, lv, lr] = split.to_local(1, 110.0, 95.0, 0.5, 0.20, 0.05);
    EXPECT_NEAR(lk, 100.0, 1e-12);      // strike → K_ref[1]
    EXPECT_NEAR(ls, 110.0, 1e-12);      // spot unchanged
}

TEST(MultiKRefSplitTest, NormalizeDividesByKRef) {
    MultiKRefSplit split({80.0, 100.0, 120.0});
    EXPECT_NEAR(split.normalize(1, 95.0, 50.0), 0.5, 1e-12);  // 50.0 / 100.0
}

TEST(MultiKRefSplitTest, DenormalizeMultipliesByStrike) {
    MultiKRefSplit split({80.0, 100.0, 120.0});
    EXPECT_NEAR(split.denormalize(0.5, 100, 95.0, 0.5, 0.2, 0.05), 47.5, 1e-12);
}

// --- SplitSurface integration ---

TEST(SplitSurfaceTest, SingleSegmentPassesThrough) {
    // Single segment, identity split
    TauSegmentSplit split({0.0}, {1.0}, {0.0}, {1.0}, 100.0);
    MockInner inner{.offset = 0.0};
    SplitSurface<MockInner, TauSegmentSplit> surface({std::move(inner)}, std::move(split));

    // MockInner returns spot/strike, normalize multiplies by K_ref=100
    // spot=110, strike=100 (but to_local sets strike=K_ref=100)
    // MockInner sees spot=110, strike=100 → 110/100 = 1.1
    // normalize: 1.1 * 100 = 110
    double p = surface.price(110.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_NEAR(p, 110.0, 1e-12);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:split_surface_test --test_output=all`
Expected: BUILD ERROR

**Step 3: Write SplitSurface and SplitPolicy**

```cpp
// src/option/table/split_surface.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <concepts>
#include <tuple>
#include <vector>

namespace mango {

struct BracketResult {
    struct Entry { size_t index; double weight; };
    std::array<Entry, 2> entries{};
    size_t count = 0;
};

template <typename S>
concept SplitPolicy = requires(const S& s, double spot, double strike,
                                double tau, double sigma, double rate) {
    { s.bracket(spot, strike, tau, sigma, rate) } -> std::same_as<BracketResult>;
    { s.to_local(size_t{}, spot, strike, tau, sigma, rate) }
        -> std::same_as<std::tuple<double, double, double, double, double>>;
    { s.normalize(size_t{}, strike, double{}) } -> std::same_as<double>;
    { s.denormalize(double{}, spot, strike, tau, sigma, rate) } -> std::same_as<double>;
};

/// Composable surface split. Routes queries to pieces via SplitPolicy,
/// with per-slice remapping and value normalization.
template <typename Inner, typename Split>
class SplitSurface {
public:
    SplitSurface(std::vector<Inner> pieces, Split split)
        : pieces_(std::move(pieces)), split_(std::move(split)) {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        auto br = split_.bracket(spot, strike, tau, sigma, rate);
        double result = 0.0;
        for (size_t i = 0; i < br.count; ++i) {
            auto [ls, lk, lt, lv, lr] = split_.to_local(
                br.entries[i].index, spot, strike, tau, sigma, rate);
            double raw = pieces_[br.entries[i].index].price(ls, lk, lt, lv, lr);
            double norm = split_.normalize(br.entries[i].index, strike, raw);
            result += br.entries[i].weight * norm;
        }
        return split_.denormalize(result, spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        auto br = split_.bracket(spot, strike, tau, sigma, rate);
        double result = 0.0;
        for (size_t i = 0; i < br.count; ++i) {
            auto [ls, lk, lt, lv, lr] = split_.to_local(
                br.entries[i].index, spot, strike, tau, sigma, rate);
            double raw = pieces_[br.entries[i].index].vega(ls, lk, lt, lv, lr);
            double norm = split_.normalize(br.entries[i].index, strike, raw);
            result += br.entries[i].weight * norm;
        }
        return split_.denormalize(result, spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] size_t num_pieces() const noexcept { return pieces_.size(); }

private:
    std::vector<Inner> pieces_;
    Split split_;
};

}  // namespace mango
```

**Step 4: Write TauSegmentSplit**

```cpp
// src/option/table/splits/tau_segment.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/split_surface.hpp"
#include <algorithm>
#include <tuple>
#include <vector>

namespace mango {

/// Split policy for tau-segmented surfaces (discrete dividends).
/// Merges SegmentLookup + SegmentedTransform from old architecture.
class TauSegmentSplit {
public:
    TauSegmentSplit(std::vector<double> tau_start, std::vector<double> tau_end,
                    std::vector<double> tau_min, std::vector<double> tau_max,
                    double K_ref)
        : tau_start_(std::move(tau_start))
        , tau_end_(std::move(tau_end))
        , tau_min_(std::move(tau_min))
        , tau_max_(std::move(tau_max))
        , K_ref_(K_ref)
    {}

    [[nodiscard]] BracketResult bracket(
        double /*spot*/, double /*strike*/, double tau,
        double /*sigma*/, double /*rate*/) const noexcept {
        BracketResult br;
        const size_t n = tau_start_.size();
        if (n == 0) return br;

        size_t idx = 0;
        for (size_t i = n; i > 0; --i) {
            const size_t j = i - 1;
            if (j == 0) {
                if (tau >= tau_start_[j] && tau <= tau_end_[j]) { idx = j; break; }
            } else {
                if (tau > tau_start_[j] && tau <= tau_end_[j]) { idx = j; break; }
            }
        }
        if (tau <= tau_start_.front()) idx = 0;
        else if (tau >= tau_end_.back()) idx = n - 1;

        br.entries[0] = {idx, 1.0};
        br.count = 1;
        return br;
    }

    [[nodiscard]] std::tuple<double, double, double, double, double>
    to_local(size_t i, double spot, double /*strike*/,
             double tau, double sigma, double rate) const noexcept {
        double local_tau = std::clamp(tau - tau_start_[i], tau_min_[i], tau_max_[i]);
        double local_spot = spot > 0.0 ? spot : 1e-8;
        return {local_spot, K_ref_, local_tau, sigma, rate};
    }

    [[nodiscard]] double normalize(size_t /*i*/, double /*strike*/,
                                    double raw) const noexcept {
        return raw * K_ref_;
    }

    [[nodiscard]] double denormalize(double combined, double /*spot*/, double /*strike*/,
                                      double /*tau*/, double /*sigma*/,
                                      double /*rate*/) const noexcept {
        return combined;
    }

private:
    std::vector<double> tau_start_, tau_end_, tau_min_, tau_max_;
    double K_ref_;
};

}  // namespace mango
```

**Step 5: Write MultiKRefSplit**

```cpp
// src/option/table/splits/multi_kref.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/split_surface.hpp"
#include <tuple>
#include <vector>

namespace mango {

/// Split policy for multi-K_ref surfaces (discrete dividends).
/// Merges KRefBracket + KRefTransform from old architecture.
class MultiKRefSplit {
public:
    explicit MultiKRefSplit(std::vector<double> k_refs)
        : k_refs_(std::move(k_refs)) {}

    [[nodiscard]] BracketResult bracket(
        double /*spot*/, double strike, double /*tau*/,
        double /*sigma*/, double /*rate*/) const noexcept {
        BracketResult br;
        const size_t n = k_refs_.size();
        if (n == 0) return br;
        if (n == 1 || strike <= k_refs_.front()) {
            br.entries[0] = {0, 1.0};
            br.count = 1;
            return br;
        }
        if (strike >= k_refs_.back()) {
            br.entries[0] = {n - 1, 1.0};
            br.count = 1;
            return br;
        }
        size_t hi = 1;
        while (hi < n && k_refs_[hi] < strike) ++hi;
        size_t lo = hi - 1;
        double t = (strike - k_refs_[lo]) / (k_refs_[hi] - k_refs_[lo]);
        br.entries[0] = {lo, 1.0 - t};
        br.entries[1] = {hi, t};
        br.count = 2;
        return br;
    }

    [[nodiscard]] std::tuple<double, double, double, double, double>
    to_local(size_t i, double spot, double /*strike*/,
             double tau, double sigma, double rate) const noexcept {
        return {spot, k_refs_[i], tau, sigma, rate};
    }

    [[nodiscard]] double normalize(size_t i, double /*strike*/,
                                    double raw) const noexcept {
        return raw / k_refs_[i];
    }

    [[nodiscard]] double denormalize(double combined, double /*spot*/, double strike,
                                      double /*tau*/, double /*sigma*/,
                                      double /*rate*/) const noexcept {
        return combined * strike;
    }

    [[nodiscard]] const std::vector<double>& k_refs() const noexcept { return k_refs_; }

private:
    std::vector<double> k_refs_;
};

}  // namespace mango
```

**Step 6: Add BUILD targets, run tests**

Run: `bazel test //tests:split_surface_test --test_output=all`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/option/table/split_surface.hpp src/option/table/splits/ \
        tests/split_surface_test.cc src/option/table/BUILD.bazel tests/BUILD.bazel
git commit -m "Add SplitSurface framework and split policies"
```

---

### Task 5: PriceTable wrapper and new type aliases

**Files:**
- Create: `src/option/table/bounded_surface.hpp`
- Modify: `tests/surface_concepts_test.cc`
- Modify: `src/option/table/BUILD.bazel`

**Step 1: Write the failing test**

Append to `tests/surface_concepts_test.cc`:

```cpp
#include "mango/option/table/bounded_surface.hpp"
#include "mango/option/table/price_surface_concept.hpp"

TEST(PriceTableTest, SatisfiesPriceSurfaceConcept) {
    using StandardAdapter = EEPSurfaceAdapter<SharedBSplineInterp<4>,
                                               StandardTransform4D, AnalyticalEEP>;
    static_assert(PriceSurface<PriceTable<StandardAdapter>>);
}
```

**Step 2: Write PriceTable**

```cpp
// src/option/table/bounded_surface.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"

namespace mango {

/// Bounds metadata for a price surface.
struct SurfaceBounds {
    double m_min, m_max;
    double tau_min, tau_max;
    double sigma_min, sigma_max;
    double rate_min, rate_max;
};

/// Adds bounds and metadata to any inner surface that has price()/vega().
/// Satisfies the PriceSurface concept required by InterpolatedIVSolver.
template <typename Inner>
class PriceTable {
public:
    PriceTable(Inner inner, SurfaceBounds bounds,
                   OptionType option_type, double dividend_yield)
        : inner_(std::move(inner))
        , bounds_(bounds)
        , option_type_(option_type)
        , dividend_yield_(dividend_yield)
    {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        return inner_.price(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        return inner_.vega(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double m_min() const noexcept { return bounds_.m_min; }
    [[nodiscard]] double m_max() const noexcept { return bounds_.m_max; }
    [[nodiscard]] double tau_min() const noexcept { return bounds_.tau_min; }
    [[nodiscard]] double tau_max() const noexcept { return bounds_.tau_max; }
    [[nodiscard]] double sigma_min() const noexcept { return bounds_.sigma_min; }
    [[nodiscard]] double sigma_max() const noexcept { return bounds_.sigma_max; }
    [[nodiscard]] double rate_min() const noexcept { return bounds_.rate_min; }
    [[nodiscard]] double rate_max() const noexcept { return bounds_.rate_max; }
    [[nodiscard]] OptionType option_type() const noexcept { return option_type_; }
    [[nodiscard]] double dividend_yield() const noexcept { return dividend_yield_; }

    [[nodiscard]] const Inner& inner() const noexcept { return inner_; }

private:
    Inner inner_;
    SurfaceBounds bounds_;
    OptionType option_type_;
    double dividend_yield_;
};

}  // namespace mango
```

**Step 3: Run tests**

Run: `bazel test //tests:surface_concepts_test --test_output=all`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/option/table/bounded_surface.hpp \
        tests/surface_concepts_test.cc src/option/table/BUILD.bazel
git commit -m "Add PriceTable wrapper"
```

---

### Task 6: Bridge — redirect type aliases to new types

This is the critical task. Old type aliases (`StandardSurface`, `StandardSurface`, `SegmentedPriceSurface`, `MultiKRefInner`, `MultiKRefPriceSurface`) get redefined as compositions of new types. All 116 tests must pass.

**Files:**
- Modify: `src/option/table/standard_surface.hpp`
- Modify: `src/option/table/standard_surface.cpp`
- Modify: `src/option/table/spliced_surface_builder.hpp`
- Modify: `src/option/table/spliced_surface_builder.cpp`

**Step 1: Redefine type aliases in standard_surface.hpp**

Replace the old aliases with:
```cpp
// src/option/table/standard_surface.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/bspline/bspline_interpolant.hpp"
#include "mango/option/table/bounded_surface.hpp"
#include "mango/option/table/eep_surface_adapter.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/identity_eep.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
// Keep old includes for transition (consumers may still use them)
#include "mango/option/table/eep_transform.hpp"
#include "mango/option/table/price_table_inner.hpp"
#include "mango/option/table/spliced_surface.hpp"
#include <expected>
#include <memory>
#include <string>

namespace mango {

// ===========================================================================
// New type aliases — concept-based layered architecture
// ===========================================================================

/// Leaf adapter for standard (EEP) surfaces
using StandardLeaf = EEPSurfaceAdapter<SharedBSplineInterp<4>,
                                        StandardTransform4D, AnalyticalEEP>;

/// Standard surface wrapper (satisfies PriceSurface concept)
using StandardSurface = PriceTable<StandardLeaf>;

/// Leaf adapter for segmented surfaces (no EEP decomposition)
using SegmentedLeaf = EEPSurfaceAdapter<SharedBSplineInterp<4>,
                                         StandardTransform4D, IdentityEEP>;

/// Tau-segmented surface
using SegmentedPriceSurface = SplitSurface<SegmentedLeaf, TauSegmentSplit>;

/// Multi-K_ref surface (outer split over K_refs of segmented inner)
using MultiKRefInner = SplitSurface<SegmentedPriceSurface, MultiKRefSplit>;

/// Multi-K_ref wrapper (satisfies PriceSurface concept)
using MultiKRefPriceSurface = PriceTable<MultiKRefInner>;

// ===========================================================================
// Legacy aliases for gradual migration
// ===========================================================================

// Keep StandardSurface for any code that references it directly.
// It was SplicedSurface<EEPPriceTableInner, ...>; now it's the leaf.
using StandardSurface = StandardLeaf;

/// Create a StandardSurface from a pre-built EEP surface.
[[nodiscard]] std::expected<StandardSurface, std::string>
make_bspline_surface(
    std::shared_ptr<const PriceTableSurface> surface,
    OptionType type);

// Alias for InterpolatedIVSolver template
using DefaultInterpolatedIVSolver = InterpolatedIVSolver<StandardSurface>;

}  // namespace mango
```

Wait — `DefaultInterpolatedIVSolver` is defined in `interpolated_iv_solver.hpp`, not here. Don't move it. Just update the aliases.

**Step 2: Rewrite make_bspline_surface in standard_surface.cpp**

```cpp
std::expected<StandardSurface, std::string>
make_bspline_surface(
    std::shared_ptr<const PriceTableSurface> surface,
    OptionType type)
{
    // ... same validation as current code ...

    double K_ref = meta.K_ref;
    double dividend_yield = meta.dividends.dividend_yield;
    const auto& axes = surface->axes();

    SharedBSplineInterp<4> interp(surface);
    StandardTransform4D xform;
    AnalyticalEEP eep(type, dividend_yield);
    StandardLeaf leaf(std::move(interp), xform, eep, K_ref);

    SurfaceBounds bounds{
        .m_min = meta.m_min,
        .m_max = meta.m_max,
        .tau_min = axes.grids[1].front(),
        .tau_max = axes.grids[1].back(),
        .sigma_min = axes.grids[2].front(),
        .sigma_max = axes.grids[2].back(),
        .rate_min = axes.grids[3].front(),
        .rate_max = axes.grids[3].back(),
    };

    return StandardSurface(std::move(leaf), bounds, type, dividend_yield);
}
```

**Step 3: Rewrite spliced_surface_builder to use new types**

Update `spliced_surface_builder.cpp`:
- `build_segmented_surface()` constructs `SplitSurface<SegmentedLeaf, TauSegmentSplit>`
- `build_multi_kref_surface()` constructs `SplitSurface<SegmentedPriceSurface, MultiKRefSplit>`

Update `spliced_surface_builder.hpp`:
- `SegmentConfig` unchanged (still uses shared_ptr<PriceTableSurface>)
- `MultiKRefEntry.surface` type changes to `SegmentedPriceSurface`

**Step 4: Run ALL tests**

Run: `bazel test //... --test_output=errors`
Expected: ALL 116 PASS

This is the highest-risk step. If tests fail, check:
- `StandardSurface` still satisfies `PriceSurface` concept
- `SplitSurface` `price()/vega()` produce identical values to `SplicedSurface`
- `PriceTable` forwards all methods correctly
- `SegmentedPriceSurface` tau routing matches `SegmentLookup`
- `MultiKRefInner` K_ref bracketing matches `KRefBracket`

**Step 5: Also build benchmarks and python bindings**

Run: `bazel build //benchmarks/... //src/python:mango_option`
Expected: BUILD SUCCESS (may need include updates)

**Step 6: Commit**

```bash
git add src/option/table/standard_surface.hpp src/option/table/standard_surface.cpp \
        src/option/table/spliced_surface_builder.hpp src/option/table/spliced_surface_builder.cpp
git commit -m "Redirect surface aliases to new concept-based types"
```

---

### Task 7: Update remaining consumers and delete old code

**Files:**
- Modify: `src/option/interpolated_iv_solver.hpp` — remove old includes, update template instantiations
- Modify: `src/option/interpolated_iv_solver.cpp` — use new types in factory
- Modify: `src/option/table/adaptive_grid_builder.cpp` — update surface wrapping
- Modify: `src/option/table/adaptive_grid_types.hpp` — update result types
- Modify: `src/option/table/segmented_price_table_builder.cpp` — use new leaf type
- Modify: Various test files — update includes
- Modify: Various benchmark files — update includes
- Modify: `src/python/mango_bindings.cpp` — update includes

**Step 1: Update interpolated_iv_solver**

In `interpolated_iv_solver.hpp`:
- Replace `#include "mango/option/table/spliced_surface.hpp"` with new headers
- `AnyIVSolver` variant types stay the same (StandardSurface, MultiKRefPriceSurface)
- Template instantiations stay the same

In `interpolated_iv_solver.cpp`:
- `wrap_surface()` uses `make_bspline_surface()` (already updated)
- `wrap_multi_kref_surface()` constructs `MultiKRefPriceSurface` using new types
- `build_multi_kref_manual()` uses new `SegmentedLeaf` + `TauSegmentSplit`

**Step 2: Update adaptive_grid_builder**

- `adaptive_grid_types.hpp`: `AdaptiveResult.surface` stays as `shared_ptr<const PriceTableSurface>` (the B-spline surface is still built by the builder)
- `adaptive_grid_builder.cpp`: surface wrapping at the end uses `make_bspline_surface()`

**Step 3: Update segmented_price_table_builder**

Uses `build_segmented_surface()` which is already updated in Task 6.

**Step 4: Update test includes**

Most test files include `standard_surface.hpp` which now pulls in the new headers. Tests that directly reference old types (e.g., `SplicedSurface` in `spliced_surface_test.cc`) need updating:
- `spliced_surface_test.cc` — test the new `SplitSurface` type instead
- `price_surface_concept_test.cc` — already satisfied by `PriceTable`

**Step 5: Run ALL tests**

Run: `bazel test //... --test_output=errors`
Expected: ALL PASS

**Step 6: Build everything**

Run: `bazel build //... && bazel build //benchmarks/... && bazel build //src/python:mango_option`
Expected: ALL BUILD SUCCESS

**Step 7: Commit**

```bash
git add -A
git commit -m "Update all consumers to new surface types"
```

---

### Task 8: Delete old code and final cleanup

**Files to delete:**
- `src/option/table/spliced_surface.hpp` — replaced by `split_surface.hpp`
- `src/option/table/price_table_inner.hpp` — replaced by `EEPSurfaceAdapter<..., IdentityEEP>`
- `src/option/table/eep_transform.hpp` — replaced by `eep_surface_adapter.hpp` + `eep/analytical_eep.hpp`
- `src/option/table/eep_transform.cpp` — logic moved to new files

**Note:** `price_table_surface.hpp` is NOT deleted. It stays as the B-spline surface — `SharedBSplineInterp<N>` wraps it.

**Step 1: Remove old includes from standard_surface.hpp**

Delete the legacy includes added in Task 6:
```cpp
// Remove these:
#include "mango/option/table/eep_transform.hpp"
#include "mango/option/table/price_table_inner.hpp"
#include "mango/option/table/spliced_surface.hpp"
```

**Step 2: Remove old files**

Delete the files listed above. Remove their BUILD targets.

**Step 3: Fix any remaining include references**

Grep for `eep_transform.hpp`, `price_table_inner.hpp`, `spliced_surface.hpp` across the entire codebase. Update or remove each reference.

The `EEPDecomposer` struct (build-time helper) currently in `eep_transform.hpp` needs a new home. Move it to `src/option/table/eep/eep_decomposer.hpp` (it's build-time only, separate from query-time adapter).

**Step 4: Run full verification**

```bash
bazel test //...
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

Expected: ALL PASS, ALL BUILD

**Step 5: Commit**

```bash
git add -A
git commit -m "Remove old SplicedSurface and inner adapter types"
```

---

## Verification Checklist

After all tasks complete:

- [ ] `bazel test //...` — all 116 tests pass
- [ ] `bazel build //benchmarks/...` — all benchmarks compile
- [ ] `bazel build //src/python:mango_option` — python bindings compile
- [ ] No references to deleted types remain (grep for `SplicedSurface`, `EEPPriceTableInner`, `PriceTableInner`)
- [ ] New concepts are tested (`surface_concepts_test`, `split_surface_test`)
- [ ] `EEPSurfaceAdapter` produces identical output to old `EEPPriceTableInner`
- [ ] `SplitSurface<..., TauSegmentSplit>` produces identical output to old `SegmentedSurface`
- [ ] `SplitSurface<..., MultiKRefSplit>` produces identical output to old `MultiKRefSurface`
