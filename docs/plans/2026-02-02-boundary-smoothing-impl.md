# K_ref Boundary Smoothing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove artificial kinks at K_ref boundaries in SegmentedMultiKRefSurface by fixing interpolation edge cases.

**Architecture:** All changes are in `interp_across_krefs()` and the `price()`/`vega()` entry logic in `segmented_multi_kref_surface.cpp`. Three interpolation fixes (C1 Hermite, virtual edge points, smooth extrapolation) plus test-only diagnostics. No new files, no header changes.

**Tech Stack:** C++23, GoogleTest, Bazel

---

### Task 1: Add C1 smoothness tests (failing)

Write tests that detect the current C0 kinks. These fail now and pass after fixes.

**Files:**
- Modify: `tests/segmented_multi_kref_surface_test.cc`

**Step 1: Write the failing smoothness test**

Add a test that checks numerical derivative continuity across K_ref boundaries.
The test builds a 3-point surface ({80, 100, 120}), then evaluates price at
strikes near K_ref=100 using **off-boundary points only** (avoids the
exact-match branch that bypasses interpolation).

```cpp
// Numerical derivative smoothness test at K_ref boundaries
TEST(SegmentedMultiKRefSurfaceTest, C1SmoothnessAtKRefBoundary) {
    auto s80 = build_surface(80.0);
    auto s100 = build_surface(100.0);
    auto s120 = build_surface(120.0);

    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    entries.push_back({80.0, std::move(s80)});
    entries.push_back({100.0, std::move(s100)});
    entries.push_back({120.0, std::move(s120)});

    auto surface = SegmentedMultiKRefSurface::create(std::move(entries));
    ASSERT_TRUE(surface.has_value());

    // Use off-boundary points to avoid the exact-match branch in price()
    // which bypasses interpolation and could mask discontinuities
    constexpr double spot = 100.0, tau = 0.5, sigma = 0.20, rate = 0.05;
    constexpr double h = 0.5;  // step size for finite differences

    // Compute left and right derivatives near K_ref=100 using 4-point stencil
    // Left derivative:  (p(K-h) - p(K-2h)) / h
    // Right derivative: (p(K+2h) - p(K+h)) / h
    double K = 100.0;
    double p_minus2h = surface->price(spot, K - 2*h, tau, sigma, rate);
    double p_minus_h = surface->price(spot, K - h, tau, sigma, rate);
    double p_plus_h  = surface->price(spot, K + h, tau, sigma, rate);
    double p_plus2h  = surface->price(spot, K + 2*h, tau, sigma, rate);

    double deriv_left  = (p_minus_h - p_minus2h) / h;
    double deriv_right = (p_plus2h - p_plus_h) / h;

    // For C1 continuity, these should be close
    // Tolerance: relative difference < 10% of the average absolute derivative
    double avg_deriv = 0.5 * (std::abs(deriv_left) + std::abs(deriv_right));
    if (avg_deriv > 1e-10) {  // skip if derivative is near zero
        double rel_diff = std::abs(deriv_left - deriv_right) / avg_deriv;
        EXPECT_LT(rel_diff, 0.10)
            << "Derivative discontinuity at K_ref=100: "
            << "left=" << deriv_left << " right=" << deriv_right;
    }
}

// Also test vega smoothness at K_ref boundaries
TEST(SegmentedMultiKRefSurfaceTest, VegaSmoothnessAtKRefBoundary) {
    auto s80 = build_surface(80.0);
    auto s100 = build_surface(100.0);
    auto s120 = build_surface(120.0);

    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    entries.push_back({80.0, std::move(s80)});
    entries.push_back({100.0, std::move(s100)});
    entries.push_back({120.0, std::move(s120)});

    auto surface = SegmentedMultiKRefSurface::create(std::move(entries));
    ASSERT_TRUE(surface.has_value());

    constexpr double spot = 100.0, tau = 0.5, sigma = 0.20, rate = 0.05;
    constexpr double h = 0.5;

    double K = 100.0;
    double v_minus2h = surface->vega(spot, K - 2*h, tau, sigma, rate);
    double v_minus_h = surface->vega(spot, K - h, tau, sigma, rate);
    double v_plus_h  = surface->vega(spot, K + h, tau, sigma, rate);
    double v_plus2h  = surface->vega(spot, K + 2*h, tau, sigma, rate);

    double deriv_left  = (v_minus_h - v_minus2h) / h;
    double deriv_right = (v_plus2h - v_plus_h) / h;

    double avg_deriv = 0.5 * (std::abs(deriv_left) + std::abs(deriv_right));
    if (avg_deriv > 1e-10) {
        double rel_diff = std::abs(deriv_left - deriv_right) / avg_deriv;
        EXPECT_LT(rel_diff, 0.10)
            << "Vega derivative discontinuity at K_ref=100: "
            << "left=" << deriv_left << " right=" << deriv_right;
    }
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:segmented_multi_kref_surface_test --test_filter=C1Smoothness --test_output=all`

Expected: FAIL (current piecewise linear interpolation with 3 points produces C0 kink)

**Step 3: Commit**

```bash
git add tests/segmented_multi_kref_surface_test.cc
git commit -m "Add failing C1 smoothness tests for K_ref boundaries"
```

---

### Task 2: Implement C1 Hermite for < 4 K_ref points

Replace the piecewise linear fallback with Fritsch-Carlson C1 Hermite.

**Files:**
- Modify: `src/option/table/segmented_multi_kref_surface.cpp:102-180`

**Context:** The current `interp_across_krefs()` uses piecewise linear when
`entries.size() < 4`. This gives C0 only. We need a proper C1 Hermite that:
- Handles non-uniform spacing in log(K_ref) correctly
- Uses weighted slopes for endpoints (not simple one-sided differences)
- Applies monotonicity limiter only when data is monotone
- Falls back to C1 central difference when data is non-monotone
- Supports extrapolation (t outside [x[0], x[n-1]])

**Step 1: Add C1 Hermite helper**

Add a static helper function above `interp_across_krefs()`:

```cpp
// C1 Hermite interpolation (Fritsch-Carlson) for n=2 or n=3 points.
// x[] are strictly increasing positions in log(K_ref) space.
// y[] are normalized values (value/K_ref).
// t can be inside or outside [x[0], x[n-1]] (extrapolation supported).
static double hermite_interp(const double* x, const double* y, size_t n,
                             double t) {
    if (n == 2) {
        double u = (t - x[0]) / (x[1] - x[0]);
        return (1.0 - u) * y[0] + u * y[1];
    }

    // n == 3: Full Fritsch-Carlson for non-uniform spacing
    const double h0 = x[1] - x[0];
    const double h1 = x[2] - x[1];
    const double d0 = (y[1] - y[0]) / h0;
    const double d1 = (y[2] - y[1]) / h1;

    // Weighted endpoint slopes (correct for non-uniform spacing)
    double m0 = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    double m2 = ((2.0 * h1 + h0) * d1 - h1 * d0) / (h0 + h1);

    // Interior slope
    double m1;
    if (d0 * d1 > 0.0) {
        // Data is monotone over both intervals: weighted harmonic mean
        const double w1 = 2.0 * h1 + h0;
        const double w2 = 2.0 * h0 + h1;
        m1 = (w1 + w2) / (w1 / d0 + w2 / d1);
    } else {
        // Non-monotone data: C1 central difference (no monotone forcing)
        m1 = (h0 * d1 + h1 * d0) / (h0 + h1);
    }

    // Monotonicity limiter on ALL slopes (only when data is monotone)
    if (d0 * d1 > 0.0) {
        auto limit = [](double& m, double d_near) {
            if (m * d_near <= 0.0) { m = 0.0; return; }
            double lim = 3.0 * std::abs(d_near);
            if (std::abs(m) > lim) m = std::copysign(lim, m);
        };
        limit(m0, d0);
        limit(m1, d0);
        limit(m1, d1);
        limit(m2, d1);
    }

    // Choose interval (supports extrapolation: t can be outside [x[0], x[2]])
    size_t i = (t <= x[1]) ? 0 : 1;

    const double hi = x[i + 1] - x[i];
    const double u = (t - x[i]) / hi;
    const double u2 = u * u;
    const double u3 = u2 * u;

    const double mi  = (i == 0) ? m0 : m1;
    const double mi1 = (i == 0) ? m1 : m2;

    // Hermite basis (slopes scaled by interval width)
    return (2.0 * u3 - 3.0 * u2 + 1.0) * y[i]
         + (u3 - 2.0 * u2 + u)          * (mi * hi)
         + (-2.0 * u3 + 3.0 * u2)       * y[i + 1]
         + (u3 - u2)                     * (mi1 * hi);
}
```

**Step 2: Replace the linear fallback in `interp_across_krefs()`**

Replace lines 149-156 (the `if (n < 4)` block) with:

```cpp
    if (n < 4) {
        // C1 Hermite for 2-3 points (Fritsch-Carlson for n=3, linear for n=2)
        std::array<double, 3> xs, ys;
        for (size_t i = 0; i < n; ++i) {
            xs[i] = std::log(entries[i].K_ref);
            ys[i] = eval_fn(i) / entries[i].K_ref;
        }
        double result = hermite_interp(xs.data(), ys.data(), n, log_strike);
        return std::max(result, 0.0) * strike;  // clamp non-negative
    }
```

**Step 3: Run tests to verify the smoothness tests pass**

Run: `bazel test //tests:segmented_multi_kref_surface_test --test_output=all`

Expected: All tests pass including the new C1 smoothness tests.

**Step 4: Commit**

```bash
git add src/option/table/segmented_multi_kref_surface.cpp
git commit -m "Replace linear fallback with C1 Hermite for < 4 K_refs"
```

---

### Task 3: Virtual edge points for Catmull-Rom

Replace index clamping with linearly extrapolated virtual points at edges.

**Files:**
- Modify: `src/option/table/segmented_multi_kref_surface.cpp:158-177`
- Modify: `tests/segmented_multi_kref_surface_test.cc`

**Step 1: Write a smoothness test for 5+ K_ref points with asymmetric spacing**

Use asymmetric K_ref spacing to make the edge clamping issue visible.

```cpp
TEST(SegmentedMultiKRefSurfaceTest, C1SmoothnessAtEdgeIntervals) {
    // Asymmetric spacing makes edge clamping visible
    // (uniform spacing can hide the issue because duplicate gives same slope)
    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    for (double K : {70.0, 80.0, 100.0, 115.0, 140.0}) {
        entries.push_back({K, build_surface(K)});
    }

    auto surface = SegmentedMultiKRefSurface::create(std::move(entries));
    ASSERT_TRUE(surface.has_value());

    constexpr double spot = 100.0, tau = 0.5, sigma = 0.20, rate = 0.05;
    constexpr double h = 0.5;

    // Check smoothness near edge K_refs (K=80 is first interior, K=115 is last interior)
    for (double K : {80.0, 115.0}) {
        double p_m2 = surface->price(spot, K - 2*h, tau, sigma, rate);
        double p_m1 = surface->price(spot, K - h, tau, sigma, rate);
        double p_p1 = surface->price(spot, K + h, tau, sigma, rate);
        double p_p2 = surface->price(spot, K + 2*h, tau, sigma, rate);

        double deriv_left  = (p_m1 - p_m2) / h;
        double deriv_right = (p_p2 - p_p1) / h;

        double avg_deriv = 0.5 * (std::abs(deriv_left) + std::abs(deriv_right));
        if (avg_deriv > 1e-10) {
            double rel_diff = std::abs(deriv_left - deriv_right) / avg_deriv;
            EXPECT_LT(rel_diff, 0.15)
                << "Edge derivative discontinuity at K=" << K
                << ": left=" << deriv_left << " right=" << deriv_right;
        }
    }

    // Same for vega
    for (double K : {80.0, 115.0}) {
        double v_m2 = surface->vega(spot, K - 2*h, tau, sigma, rate);
        double v_m1 = surface->vega(spot, K - h, tau, sigma, rate);
        double v_p1 = surface->vega(spot, K + h, tau, sigma, rate);
        double v_p2 = surface->vega(spot, K + 2*h, tau, sigma, rate);

        double deriv_left  = (v_m1 - v_m2) / h;
        double deriv_right = (v_p2 - v_p1) / h;

        double avg_deriv = 0.5 * (std::abs(deriv_left) + std::abs(deriv_right));
        if (avg_deriv > 1e-10) {
            double rel_diff = std::abs(deriv_left - deriv_right) / avg_deriv;
            EXPECT_LT(rel_diff, 0.15)
                << "Vega edge discontinuity at K=" << K
                << ": left=" << deriv_left << " right=" << deriv_right;
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:segmented_multi_kref_surface_test --test_filter=C1SmoothnessAtEdge --test_output=all`

**Step 3: Replace clamping with virtual points**

In `interp_across_krefs()`, replace the index selection and x/y array construction
(lines ~158-177):

```cpp
    // Select 4 entries centered on the bracket [lo_idx, lo_idx+1]
    size_t i1 = lo_idx;
    size_t i2 = lo_idx + 1;

    // Use virtual points at edges (linear extrapolation) instead of clamping
    std::array<double, 4> x, y;
    x[1] = std::log(entries[i1].K_ref);
    x[2] = std::log(entries[i2].K_ref);
    y[1] = eval_fn(i1) / entries[i1].K_ref;
    y[2] = eval_fn(i2) / entries[i2].K_ref;

    if (i1 > 0) {
        size_t i0 = i1 - 1;
        x[0] = std::log(entries[i0].K_ref);
        y[0] = eval_fn(i0) / entries[i0].K_ref;
    } else {
        // Virtual left point: linear extrapolation from [i1, i2]
        x[0] = 2.0 * x[1] - x[2];
        y[0] = 2.0 * y[1] - y[2];
    }

    if (i2 + 1 < n) {
        size_t i3 = i2 + 1;
        x[3] = std::log(entries[i3].K_ref);
        y[3] = eval_fn(i3) / entries[i3].K_ref;
    } else {
        // Virtual right point: linear extrapolation from [i1, i2]
        x[3] = 2.0 * x[2] - x[1];
        y[3] = 2.0 * y[2] - y[1];
    }

    // Clamp result to non-negative (Catmull-Rom can overshoot at edges)
    double result = catmull_rom(x, y, log_strike);
    return std::max(result, 0.0) * strike;
```

**Step 4: Run tests**

Run: `bazel test //tests:segmented_multi_kref_surface_test --test_output=all`

Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/option/table/segmented_multi_kref_surface.cpp tests/segmented_multi_kref_surface_test.cc
git commit -m "Use virtual edge points for Catmull-Rom instead of clamping"
```

---

### Task 4: Smooth extrapolation outside K_ref range

Let Catmull-Rom naturally extrapolate for modest distances beyond the K_ref range.
Bound extrapolation in **log(K_ref) space** (not linear space) to avoid
overshooting with irregular grids. Clamp output to non-negative.

**Files:**
- Modify: `src/option/table/segmented_multi_kref_surface.cpp:182-228` (price and vega)
- Modify: `tests/segmented_multi_kref_surface_test.cc`

**Step 1: Write tests for smooth transition at K_ref boundaries**

```cpp
TEST(SegmentedMultiKRefSurfaceTest, SmoothExtrapolationOutsideRange) {
    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    for (double K : {80.0, 100.0, 120.0}) {
        entries.push_back({K, build_surface(K)});
    }

    auto surface = SegmentedMultiKRefSurface::create(std::move(entries));
    ASSERT_TRUE(surface.has_value());

    constexpr double spot = 100.0, tau = 0.5, sigma = 0.20, rate = 0.05;
    constexpr double h = 0.5;

    // Check price smoothness at the left boundary (K_ref=80)
    {
        double K = 80.0;
        double p_m2 = surface->price(spot, K - 2*h, tau, sigma, rate);
        double p_m1 = surface->price(spot, K - h, tau, sigma, rate);
        double p_p1 = surface->price(spot, K + h, tau, sigma, rate);
        double p_p2 = surface->price(spot, K + 2*h, tau, sigma, rate);

        double deriv_left  = (p_m1 - p_m2) / h;
        double deriv_right = (p_p2 - p_p1) / h;

        double avg_deriv = 0.5 * (std::abs(deriv_left) + std::abs(deriv_right));
        if (avg_deriv > 1e-10) {
            double rel_diff = std::abs(deriv_left - deriv_right) / avg_deriv;
            EXPECT_LT(rel_diff, 0.25)
                << "Extrapolation kink at left boundary K=" << K
                << ": left_deriv=" << deriv_left << " right_deriv=" << deriv_right;
        }
    }

    // Check vega smoothness at the right boundary (K_ref=120)
    {
        double K = 120.0;
        double v_m2 = surface->vega(spot, K - 2*h, tau, sigma, rate);
        double v_m1 = surface->vega(spot, K - h, tau, sigma, rate);
        double v_p1 = surface->vega(spot, K + h, tau, sigma, rate);
        double v_p2 = surface->vega(spot, K + 2*h, tau, sigma, rate);

        double deriv_left  = (v_m1 - v_m2) / h;
        double deriv_right = (v_p2 - v_p1) / h;

        double avg_deriv = 0.5 * (std::abs(deriv_left) + std::abs(deriv_right));
        if (avg_deriv > 1e-10) {
            double rel_diff = std::abs(deriv_left - deriv_right) / avg_deriv;
            EXPECT_LT(rel_diff, 0.25)
                << "Vega extrapolation kink at right boundary K=" << K
                << ": left_deriv=" << deriv_left << " right_deriv=" << deriv_right;
        }
    }

    // Extrapolated prices must be finite and non-negative
    for (double K_out : {75.0, 70.0, 125.0, 130.0}) {
        double p = surface->price(spot, K_out, tau, sigma, rate);
        EXPECT_TRUE(std::isfinite(p)) << "K=" << K_out;
        EXPECT_GE(p, 0.0) << "K=" << K_out;  // non-negative guard

        double v = surface->vega(spot, K_out, tau, sigma, rate);
        EXPECT_TRUE(std::isfinite(v)) << "vega K=" << K_out;
    }
}
```

**Step 2: Implement smooth extrapolation in `price()`**

Replace the `price()` method. Key changes:
- Extrapolation bounds in **log space** (one log-interval beyond range)
- Clamp to nearest surface beyond the extrapolation limit
- Non-negative guard on output

```cpp
double SegmentedMultiKRefSurface::price(double spot, double strike,
                                         double tau, double sigma,
                                         double rate) const {
    const size_t n = entries_.size();

    // Single entry: use it directly
    if (n == 1) {
        return entries_.front().surface.price(spot, strike, tau, sigma, rate);
    }

    // Extrapolation bounds in log space (one log-interval beyond range)
    double log_K_min = std::log(entries_.front().K_ref);
    double log_K_max = std::log(entries_.back().K_ref);
    double log_left_spacing = (n > 1)
        ? std::log(entries_[1].K_ref) - log_K_min : 0.0;
    double log_right_spacing = (n > 1)
        ? log_K_max - std::log(entries_[n - 2].K_ref) : 0.0;
    double log_strike = std::log(strike);

    if (log_strike < log_K_min - log_left_spacing) {
        return entries_.front().surface.price(spot, strike, tau, sigma, rate);
    }
    if (log_strike > log_K_max + log_right_spacing) {
        return entries_.back().surface.price(spot, strike, tau, sigma, rate);
    }

    // Exact match at boundary K_refs (must check before extrapolation path)
    if (strike == entries_.front().K_ref) {
        return entries_.front().surface.price(spot, strike, tau, sigma, rate);
    }
    if (strike == entries_.back().K_ref) {
        return entries_.back().surface.price(spot, strike, tau, sigma, rate);
    }

    // For strikes in the extrapolation zone or interior, use interpolation
    size_t lo_idx;
    if (strike < entries_.front().K_ref) {
        lo_idx = 0;
    } else if (strike > entries_.back().K_ref) {
        lo_idx = n - 2;
    } else {
        lo_idx = find_bracket(strike);
        // Exact match at interior K_ref
        if (strike == entries_[lo_idx].K_ref) {
            return entries_[lo_idx].surface.price(spot, strike, tau, sigma, rate);
        }
    }

    return interp_across_krefs(entries_, strike, lo_idx, [&](size_t i) {
        return entries_[i].surface.price(spot, strike, tau, sigma, rate);
    });
}
```

**Step 3: Apply the same pattern to `vega()`**

Same log-space extrapolation logic as price().

```cpp
double SegmentedMultiKRefSurface::vega(double spot, double strike,
                                        double tau, double sigma,
                                        double rate) const {
    const size_t n = entries_.size();

    if (n == 1) {
        return entries_.front().surface.vega(spot, strike, tau, sigma, rate);
    }

    double log_K_min = std::log(entries_.front().K_ref);
    double log_K_max = std::log(entries_.back().K_ref);
    double log_left_spacing = (n > 1)
        ? std::log(entries_[1].K_ref) - log_K_min : 0.0;
    double log_right_spacing = (n > 1)
        ? log_K_max - std::log(entries_[n - 2].K_ref) : 0.0;
    double log_strike = std::log(strike);

    if (log_strike < log_K_min - log_left_spacing) {
        return entries_.front().surface.vega(spot, strike, tau, sigma, rate);
    }
    if (log_strike > log_K_max + log_right_spacing) {
        return entries_.back().surface.vega(spot, strike, tau, sigma, rate);
    }

    // Exact match at boundary K_refs
    if (strike == entries_.front().K_ref) {
        return entries_.front().surface.vega(spot, strike, tau, sigma, rate);
    }
    if (strike == entries_.back().K_ref) {
        return entries_.back().surface.vega(spot, strike, tau, sigma, rate);
    }

    size_t lo_idx;
    if (strike < entries_.front().K_ref) {
        lo_idx = 0;
    } else if (strike > entries_.back().K_ref) {
        lo_idx = n - 2;
    } else {
        lo_idx = find_bracket(strike);
        if (strike == entries_[lo_idx].K_ref) {
            return entries_[lo_idx].surface.vega(spot, strike, tau, sigma, rate);
        }
    }

    return interp_across_krefs(entries_, strike, lo_idx, [&](size_t i) {
        return entries_[i].surface.vega(spot, strike, tau, sigma, rate);
    });
}
```

**Step 4: Run tests**

Run: `bazel test //tests:segmented_multi_kref_surface_test --test_output=all`

Expected: All tests pass including extrapolation smoothness and non-negative guard.

**Step 5: Commit**

```bash
git add src/option/table/segmented_multi_kref_surface.cpp tests/segmented_multi_kref_surface_test.cc
git commit -m "Smooth extrapolation outside K_ref range in log space"
```

---

### Task 5: Boundary disagreement metric and full regression

Add a diagnostic metric and run the full test suite.

**Files:**
- Modify: `tests/segmented_multi_kref_surface_test.cc`

**Step 1: Add boundary disagreement metric**

```cpp
// Diagnostic: measure cross-surface disagreement at K_ref boundaries
struct BoundaryDiagnostics {
    double max_price_diff = 0.0;   // max |P_left(K) - P_right(K)| / K
    double mean_price_diff = 0.0;
    size_t sample_count = 0;
};

static BoundaryDiagnostics measure_boundary_disagreement(
    const std::vector<SegmentedMultiKRefSurface::Entry>& entries,
    double spot) {
    BoundaryDiagnostics result;
    if (entries.size() < 2) return result;

    std::vector<double> taus = {0.25, 0.5, 0.75};
    std::vector<double> sigmas = {0.15, 0.20, 0.30};
    std::vector<double> rates = {0.03, 0.05, 0.07};

    double sum_diff = 0.0;
    size_t count = 0;

    for (size_t i = 0; i + 1 < entries.size(); ++i) {
        double K = entries[i + 1].K_ref;
        for (double tau : taus) {
            for (double sigma : sigmas) {
                for (double rate : rates) {
                    double p_left = entries[i].surface.price(spot, K, tau, sigma, rate);
                    double p_right = entries[i + 1].surface.price(spot, K, tau, sigma, rate);
                    double diff = std::abs(p_left - p_right) / K;
                    result.max_price_diff = std::max(result.max_price_diff, diff);
                    sum_diff += diff;
                    count++;
                }
            }
        }
    }

    result.mean_price_diff = count > 0 ? sum_diff / count : 0.0;
    result.sample_count = count;
    return result;
}
```

**Step 2: Add a test using the metric**

```cpp
TEST(SegmentedMultiKRefSurfaceTest, BoundaryDisagreementMetric) {
    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    entries.push_back({80.0, build_surface(80.0)});
    entries.push_back({100.0, build_surface(100.0)});
    entries.push_back({120.0, build_surface(120.0)});

    auto diag = measure_boundary_disagreement(entries, 100.0);

    EXPECT_TRUE(std::isfinite(diag.max_price_diff));
    EXPECT_GE(diag.sample_count, 1u);

    // RecordProperty makes the values visible in test XML output
    // without polluting stdout in CI
    ::testing::Test::RecordProperty("max_price_diff",
        std::to_string(diag.max_price_diff));
    ::testing::Test::RecordProperty("mean_price_diff",
        std::to_string(diag.mean_price_diff));
}
```

**Step 3: Run all tests**

Run: `bazel test //tests:segmented_multi_kref_surface_test --test_output=all`

Expected: All tests pass.

**Step 4: Run the full test suite for regressions**

Run: `bazel test //...`

Expected: All tests pass (no regressions).

**Step 5: Run the IV factory test to verify Newton convergence**

Run: `bazel test //tests:iv_solver_factory_test --test_output=all`

Expected: All tests pass, including the segmented path test.

**Step 6: Commit**

```bash
git add tests/segmented_multi_kref_surface_test.cc
git commit -m "Add boundary disagreement metric and full regression check"
```
