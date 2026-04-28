# Numerical EEP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve B-spline fitting quality for chained segments in the segmented price table builder by decomposing the American price into smooth EEP + European components via a second unconstrained PDE solve.

**Architecture:** For each chained segment, run two PDE solves per (σ,r) pair — one American (with obstacle), one European (without). Subtract to get smooth numerical EEP. Store two B-spline surfaces per segment; reconstruct by summing at query time.

**Tech Stack:** C++23, Bazel, GoogleTest, TR-BDF2 PDE solver with Brennan-Schwartz projection

**Design doc:** `docs/plans/2026-02-07-numerical-eep-design.md`

---

### Task 1: Add `NumericalEEP` to `SurfaceContent` enum

**Files:**
- Modify: `src/option/table/price_table_metadata.hpp:11-14`

**Step 1: Add the enum value**

In `src/option/table/price_table_metadata.hpp`, add `NumericalEEP = 2` to the enum:

```cpp
enum class SurfaceContent : uint8_t {
    RawPrice = 0,              ///< Raw American option prices
    EarlyExercisePremium = 1,  ///< P_Am - P_Eu (requires reconstruction)
    NumericalEEP = 2,         ///< EEP from PDE subtraction, requires companion European surface
};
```

**Step 2: Build to verify no compile errors**

Run: `bazel build //src/option/table:price_table_metadata`
Expected: BUILD SUCCESS

**Step 3: Commit**

```bash
git add src/option/table/price_table_metadata.hpp
git commit -m "Add NumericalEEP to SurfaceContent enum"
```

---

### Task 2: Add `projection_enabled` flag to `AmericanOptionSolver`

**Files:**
- Modify: `src/option/american_option.hpp:107-116`
- Modify: `src/option/american_option.cpp:348-394`
- Test: `tests/american_option_test.cc`

**Step 1: Write the failing test**

Add to `tests/american_option_test.cc`:

```cpp
// European PDE (projection disabled) should produce value <= American
TEST(AmericanOptionTest, ProjectionDisabledProducesEuropeanValue) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT},
        0.20);

    // American solve (projection enabled, default)
    auto am = solve_american_option(params);
    ASSERT_TRUE(am.has_value());
    double am_price = am->value_at(100.0);

    // European solve (projection disabled)
    auto [grid_spec, time_domain] = estimate_pde_grid(params);
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n),
                                     std::pmr::get_default_resource());
    auto ws = PDEWorkspace::from_buffer(buffer, n).value();
    auto solver = AmericanOptionSolver::create(params, ws).value();
    solver.set_projection_enabled(false);
    auto eu = solver.solve();
    ASSERT_TRUE(eu.has_value());
    double eu_price = eu->value_at(100.0);

    // European <= American (early exercise adds value)
    EXPECT_LE(eu_price, am_price + 1e-10);
    // European should still be positive
    EXPECT_GT(eu_price, 0.0);
    // Difference (EEP) should be small but positive for ATM put
    EXPECT_GT(am_price - eu_price, 0.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:american_option_test --test_filter=*ProjectionDisabled* --test_output=all`
Expected: FAIL — `set_projection_enabled` does not exist

**Step 3: Add the flag to AmericanOptionSolver**

In `src/option/american_option.hpp`, add after `set_initial_condition` (line 112):

```cpp
/// Disable obstacle projection for European PDE solve.
/// When false, the solver runs plain Thomas instead of projected Thomas,
/// producing an unconstrained (European-style) solution.
void set_projection_enabled(bool enabled) { projection_enabled_ = enabled; }
bool projection_enabled() const { return projection_enabled_; }
```

Add private member after `custom_ic_` (line 116):

```cpp
bool projection_enabled_ = true;
```

**Step 4: Pass the flag through to the CRTP solver**

In `src/option/american_option.cpp`, in `AmericanOptionSolver::solve()` (line 378), add before the `std::visit` call:

```cpp
bool projection = projection_enabled_;
```

Inside the visit lambda, after `pde_solver.set_config(trbdf2_config_);` (line 385), add:

```cpp
pde_solver.set_projection_enabled(projection);
```

**Step 5: Add `set_projection_enabled` to CRTP solvers**

In `src/option/american_option.cpp`, add to both `AmericanPutSolver` (after line 133) and `AmericanCallSolver` (after line 230):

```cpp
void set_projection_enabled(bool enabled) {
    PDESolver<AmericanPutSolver>::set_projection_enabled(enabled);
}
```

(Same for `AmericanCallSolver` with its template parameter.)

**Step 6: Add `set_projection_enabled` and flag to `PDESolver`**

In `src/pde/core/pde_solver.hpp`, add public method (after `has_obstacle()`, line 187):

```cpp
void set_projection_enabled(bool enabled) { projection_enabled_ = enabled; }
```

Add private member (alongside other members):

```cpp
bool projection_enabled_ = true;
```

**Step 7: Gate the three projection sites**

In `src/pde/core/pde_solver.hpp`:

1. **`initialize()`** (line 103): change `apply_obstacle(t, u_current);` to:
```cpp
if (projection_enabled_) apply_obstacle(t, u_current);
```

2. **`process_temporal_events()`** (line 270): change `apply_obstacle(event.time, u_current);` to:
```cpp
if (projection_enabled_) apply_obstacle(event.time, u_current);
```

3. **`solve_implicit_stage_projected()`** (line 525): after building the Jacobian and RHS, gate the deep-ITM locking and projected solve. When projection disabled, use plain Thomas:

After `rhs_with_bc` is built (around line 581) and before the deep-ITM locking block (line 629), add:

```cpp
if (!projection_enabled_) {
    // European mode: plain Thomas solve, no obstacle projection
    auto result = solve_thomas<double>(
        workspace_.jacobian(),
        rhs_with_bc,
        u,
        workspace_.tridiag_workspace()
    );
    if (!result.ok()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::LinearSolveFailure,
            .iterations = 1,
            .residual = std::numeric_limits<double>::infinity()
        });
    }
    apply_boundary_conditions(u, t);
    return {};
}
```

This early-returns before the obstacle evaluation, deep-ITM locking, and projected Thomas.

**Step 8: Run test to verify it passes**

Run: `bazel test //tests:american_option_test --test_filter=*ProjectionDisabled* --test_output=all`
Expected: PASS

**Step 9: Run full test suite**

Run: `bazel test //tests:american_option_test --test_output=errors`
Expected: All tests pass (existing behavior unchanged when flag is true)

**Step 10: Commit**

```bash
git add src/pde/core/pde_solver.hpp src/option/american_option.hpp src/option/american_option.cpp tests/american_option_test.cc
git commit -m "Add projection_enabled flag for European PDE solves"
```

---

### Task 3: Handle `NumericalEEP` in `SegmentedTransform`

**Files:**
- Modify: `src/option/table/spliced_surface.hpp:405-435`
- Test: `tests/segmented_price_surface_test.cc`

**Step 1: Write the failing test**

Add to `tests/segmented_price_surface_test.cc`:

```cpp
TEST(SegmentedTransformTest, NumericalEEPPinsStrikeNoSpotAdjust) {
    SegmentedTransform transform;
    transform.tau_start = {0.0, 0.5};
    transform.tau_min = {0.01, 0.0};
    transform.tau_max = {0.5, 0.5};
    transform.content = {SurfaceContent::EarlyExercisePremium,
                         SurfaceContent::NumericalEEP};
    transform.dividends = {{.calendar_time = 0.5, .amount = 2.0}};
    transform.K_ref = 100.0;
    transform.T = 1.0;

    PriceQuery q{.spot = 95.0, .strike = 110.0, .tau = 0.8, .sigma = 0.2, .rate = 0.05};

    // Segment 1 is NumericalEEP: should pin strike, no spot adjustment
    auto local = transform.to_local(1, q);
    EXPECT_DOUBLE_EQ(local.strike, 100.0);  // pinned to K_ref
    EXPECT_DOUBLE_EQ(local.spot, 95.0);     // no adjustment

    // normalize_value for NumericalEEP should multiply by K_ref
    double raw = 0.05;  // normalized V/K_ref
    EXPECT_DOUBLE_EQ(transform.normalize_value(1, q, raw), 5.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:segmented_price_surface_test --test_filter=*NumericalEEP* --test_output=all`
Expected: FAIL — NumericalEEP not handled

**Step 3: Update `to_local()` and `normalize_value()`**

In `src/option/table/spliced_surface.hpp`, modify `SegmentedTransform::to_local()` (line 405):

```cpp
[[nodiscard]] PriceQuery to_local(size_t i, const PriceQuery& q) const {
    PriceQuery out = q;

    // Convert to local segment time and clamp.
    out.tau = std::clamp(q.tau - tau_start[i], tau_min[i], tau_max[i]);

    // Spot adjustment for analytical EEP segments only.
    if (content[i] == SurfaceContent::EarlyExercisePremium) {
        double t_query = T - q.tau;
        double t_boundary = T - tau_start[i];
        out.spot = compute_spot_adjustment(q.spot, t_query, t_boundary);
    }

    // RawPrice and NumericalEEP segments are only valid at K_ref.
    if (content[i] == SurfaceContent::RawPrice ||
        content[i] == SurfaceContent::NumericalEEP) {
        out.strike = K_ref;
    }

    if (out.spot <= 0.0) {
        out.spot = 1e-8;
    }

    return out;
}
```

Modify `normalize_value()` (line 430):

```cpp
[[nodiscard]] double normalize_value(size_t i, const PriceQuery&, double raw) const noexcept {
    if (content[i] == SurfaceContent::RawPrice ||
        content[i] == SurfaceContent::NumericalEEP) {
        return raw * K_ref;
    }
    return raw;
}
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:segmented_price_surface_test --test_filter=*NumericalEEP* --test_output=all`
Expected: PASS

**Step 5: Run all segmented tests**

Run: `bazel test //tests:segmented_price_surface_test --test_output=errors`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/option/table/spliced_surface.hpp tests/segmented_price_surface_test.cc
git commit -m "Handle NumericalEEP in SegmentedTransform"
```

---

### Task 4: Extend `AmericanPriceSurface` for two-surface reconstruction

**Files:**
- Modify: `src/option/table/american_price_surface.hpp:20-71`
- Modify: `src/option/table/american_price_surface.cpp:19-168`
- Test: `tests/american_price_surface_test.cc`

**Step 1: Write the failing test**

Add to `tests/american_price_surface_test.cc`:

```cpp
TEST(AmericanPriceSurfaceTest, NumericalEEPSumsTwoSurfaces) {
    auto eep_surface = make_test_surface(SurfaceContent::NumericalEEP, 1.0);
    auto eu_surface = make_test_surface(SurfaceContent::NumericalEEP, 3.0);

    auto result = AmericanPriceSurface::create(eep_surface, OptionType::PUT, eu_surface);
    ASSERT_TRUE(result.has_value());

    // price() should sum EEP + European (both normalized, strike pinned to K_ref=100)
    double price = result->price(100.0, 100.0, 0.5, 0.20, 0.05);
    // Both surfaces return their constant value at any coords
    // price = eep_value + eu_value (normalized, denormalized by SegmentedTransform)
    EXPECT_GT(price, 0.0);
}

TEST(AmericanPriceSurfaceTest, NumericalEEPRejectsWithoutCompanion) {
    auto surface = make_test_surface(SurfaceContent::NumericalEEP);
    auto result = AmericanPriceSurface::create(surface, OptionType::PUT);
    EXPECT_FALSE(result.has_value());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:american_price_surface_test --test_filter=*NumericalEEP* --test_output=all`
Expected: FAIL — `create()` overload does not exist

**Step 3: Add companion surface to `AmericanPriceSurface`**

In `src/option/table/american_price_surface.hpp`, add a new `create()` overload (after line 25):

```cpp
/// Create from EEP + companion European surface (NumericalEEP content).
static std::expected<AmericanPriceSurface, ValidationError> create(
    std::shared_ptr<const PriceTableSurface<4>> eep_surface,
    OptionType type,
    std::shared_ptr<const PriceTableSurface<4>> eu_surface);
```

Add private member (after line 70):

```cpp
std::shared_ptr<const PriceTableSurface<4>> eu_surface_;  ///< Companion European surface (NumericalEEP only)
```

Update the private constructor (line 64) to accept the optional European surface:

```cpp
AmericanPriceSurface(std::shared_ptr<const PriceTableSurface<4>> surface,
                     OptionType type, double K_ref, double dividend_yield,
                     std::shared_ptr<const PriceTableSurface<4>> eu_surface = nullptr);
```

**Step 4: Implement the new `create()` overload**

In `src/option/table/american_price_surface.cpp`, add after the existing `create()`:

```cpp
std::expected<AmericanPriceSurface, ValidationError>
AmericanPriceSurface::create(
    std::shared_ptr<const PriceTableSurface<4>> eep_surface,
    OptionType type,
    std::shared_ptr<const PriceTableSurface<4>> eu_surface)
{
    if (!eep_surface || !eu_surface) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 0});
    }

    const auto& meta = eep_surface->metadata();
    if (meta.content != SurfaceContent::NumericalEEP) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 0});
    }

    if (meta.K_ref <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, meta.K_ref, 0});
    }

    // Validate companion surface axes match
    const auto& eu_meta = eu_surface->metadata();
    if (std::abs(eu_meta.K_ref - meta.K_ref) > 1e-12) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, eu_meta.K_ref, 0});
    }

    return AmericanPriceSurface(
        std::move(eep_surface), type, meta.K_ref,
        meta.dividends.dividend_yield, std::move(eu_surface));
}
```

Also update the existing `create()` to reject `NumericalEEP` without companion:

In the existing `create()`, add `NumericalEEP` check (after line 30):

```cpp
if (meta.content == SurfaceContent::NumericalEEP) {
    // NumericalEEP requires companion European surface — use the 3-arg overload
    return std::unexpected(ValidationError{
        ValidationErrorCode::InvalidBounds, 0.0, 0});
}
```

Update constructor to store eu_surface:

```cpp
AmericanPriceSurface::AmericanPriceSurface(
    std::shared_ptr<const PriceTableSurface<4>> surface,
    OptionType type, double K_ref, double dividend_yield,
    std::shared_ptr<const PriceTableSurface<4>> eu_surface)
    : surface_(std::move(surface))
    , type_(type)
    , K_ref_(K_ref)
    , dividend_yield_(dividend_yield)
    , eu_surface_(std::move(eu_surface)) {}
```

**Step 5: Implement NumericalEEP branches in price() and Greeks**

In `src/option/table/american_price_surface.cpp`, update `price()`:

```cpp
double AmericanPriceSurface::price(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    if (surface_->metadata().content == SurfaceContent::NumericalEEP) {
        assert(eu_surface_ && "NumericalEEP requires companion European surface");
        double m = spot / K_ref_;
        return surface_->value({m, tau, sigma, rate})
             + eu_surface_->value({m, tau, sigma, rate});
    }
    if (surface_->metadata().content == SurfaceContent::RawPrice) {
        assert(strike == K_ref_ && "RawPrice surfaces require strike == K_ref");
        double m = spot / K_ref_;
        return surface_->value({m, tau, sigma, rate});
    }
    // EarlyExercisePremium
    double m = spot / strike;
    double eep = surface_->value({m, tau, sigma, rate});
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .option_type = type_}, sigma).solve().value();
    return eep * (strike / K_ref_) + eu.value();
}
```

Update `delta()`:

```cpp
double AmericanPriceSurface::delta(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    if (surface_->metadata().content == SurfaceContent::NumericalEEP) {
        double m = spot / K_ref_;
        return (1.0 / K_ref_) * (surface_->partial(0, {m, tau, sigma, rate})
                                + eu_surface_->partial(0, {m, tau, sigma, rate}));
    }
    // ... existing EEP code unchanged ...
```

Update `gamma()`:

```cpp
double AmericanPriceSurface::gamma(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    if (surface_->metadata().content == SurfaceContent::NumericalEEP) {
        double m = spot / K_ref_;
        return (1.0 / (K_ref_ * K_ref_)) *
            (surface_->second_partial(0, {m, tau, sigma, rate})
           + eu_surface_->second_partial(0, {m, tau, sigma, rate}));
    }
    // ... existing EEP code unchanged ...
```

Update `vega()`:

```cpp
double AmericanPriceSurface::vega(double spot, double strike, double tau,
                                  double sigma, double rate) const {
    if (surface_->metadata().content == SurfaceContent::NumericalEEP) {
        double m = spot / K_ref_;
        return surface_->partial(2, {m, tau, sigma, rate})
             + eu_surface_->partial(2, {m, tau, sigma, rate});
    }
    if (surface_->metadata().content == SurfaceContent::RawPrice) {
        // ... existing FD vega ...
```

Update `theta()`:

```cpp
double AmericanPriceSurface::theta(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    if (surface_->metadata().content == SurfaceContent::NumericalEEP) {
        double m = spot / K_ref_;
        return -(surface_->partial(1, {m, tau, sigma, rate})
               + eu_surface_->partial(1, {m, tau, sigma, rate}));
    }
    // ... existing EEP code unchanged ...
```

**Step 6: Run tests**

Run: `bazel test //tests:american_price_surface_test --test_output=all`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/option/table/american_price_surface.hpp src/option/table/american_price_surface.cpp tests/american_price_surface_test.cc
git commit -m "Extend AmericanPriceSurface for NumericalEEP two-surface reconstruction"
```

---

### Task 5: Implement numerical EEP in segmented builder

**Files:**
- Modify: `src/option/table/segmented_price_table_builder.cpp:182-338`
- Test: `tests/segmented_price_table_builder_test.cc`

**Step 1: Write the failing test**

Add to `tests/segmented_price_table_builder_test.cc`:

```cpp
TEST(SegmentedPriceTableBuilderTest, NumericalEEPProducesReasonablePrice) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0,
                      .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},
        .grid = IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
        .use_numerical_eep = true,  // new flag
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    // Query in the chained segment (tau > T - t_div = 0.5)
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.8, .sigma = 0.20, .rate = 0.05};
    double price = result->price(q);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, 50.0);

    // Vega should be finite and positive
    double vega = result->vega(q);
    EXPECT_TRUE(std::isfinite(vega));
    EXPECT_GT(vega, 0.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:segmented_price_table_builder_test --test_filter=*NumericalEEP* --test_output=all`
Expected: FAIL — `use_numerical_eep` field does not exist

**Step 3: Add config flag**

In `src/option/table/segmented_price_table_builder.hpp`, add to `Config` (after `pde_accuracy`, line 48):

```cpp
/// Use numerical EEP for chained segments (two PDE solves per sigma/rate pair).
/// Improves B-spline fitting quality by decomposing into smooth EEP + European.
bool use_numerical_eep = false;
```

**Step 4: Implement the dual-solve pipeline**

In `src/option/table/segmented_price_table_builder.cpp`, inside the `for` loop (line 182), replace the chained segment path. After the American batch solve (line 278-279), add the European solve and EEP computation when `config.use_numerical_eep` is true:

After step 5 (American batch solve, line 278-279) and step 6 (failure rate check, line 284-291), add:

```cpp
        // ---- Numerical EEP path for chained segments ----
        if (!is_last_segment && config.use_numerical_eep) {
            // Step 5b: European batch solve (projection disabled)
            BatchAmericanOptionSolver eu_batch_solver;
            eu_batch_solver.set_snapshot_times(axes.grids[1]);

            // Wrap setup callback to also disable projection
            auto eu_setup_callback = [&setup_callback](
                size_t index, AmericanOptionSolver& solver) {
                if (setup_callback) setup_callback(index, solver);
                solver.set_projection_enabled(false);
            };

            auto eu_batch_result = eu_batch_solver.solve_batch(
                batch_params, true, eu_setup_callback, custom_grid);

            // Step 6: Union failure masks
            // (Mark both as failed if either failed)
            for (size_t idx = 0; idx < batch_result.results.size(); ++idx) {
                bool am_failed = !batch_result.results[idx].has_value();
                bool eu_failed = !eu_batch_result.results[idx].has_value();
                if (am_failed || eu_failed) {
                    if (!am_failed)
                        batch_result.results[idx] = std::unexpected(SolverError{
                            .code = SolverErrorCode::InvalidConfiguration, .iterations = 0});
                    if (!eu_failed)
                        eu_batch_result.results[idx] = std::unexpected(SolverError{
                            .code = SolverErrorCode::InvalidConfiguration, .iterations = 0});
                }
            }

            // Step 7a: Extract American tensor (as RawPrice)
            builder.set_surface_content(SurfaceContent::RawPrice);
            auto am_extraction = builder.extract_tensor(batch_result, axes);
            if (!am_extraction.has_value()) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::ExtractionFailed});
            }

            // Step 7b: Extract European tensor (as RawPrice)
            auto eu_extraction = builder.extract_tensor(eu_batch_result, axes);
            if (!eu_extraction.has_value()) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::ExtractionFailed});
            }

            // Step 8: Compute EEP = softplus(K_ref * (Am - Eu)) / K_ref
            auto& eep_tensor = am_extraction->tensor;  // reuse for EEP
            auto eu_tensor_view = eu_extraction->tensor.view;
            constexpr double kSharpness = 100.0;
            const double bias = std::log(2.0) / kSharpness;

            for (size_t idx = 0; idx < eep_tensor.data.size(); ++idx) {
                double am_val = eep_tensor.data[idx];
                double eu_val = eu_extraction->tensor.data[idx];
                if (std::isnan(am_val) || std::isnan(eu_val)) {
                    eep_tensor.data[idx] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                double eep_dollar = K_ref * (am_val - eu_val);
                if (kSharpness * eep_dollar > 500.0) {
                    eep_tensor.data[idx] = eep_dollar / K_ref;
                } else {
                    double sp = std::log1p(std::exp(kSharpness * eep_dollar)) / kSharpness;
                    eep_tensor.data[idx] = std::max(0.0, sp - bias) / K_ref;
                }
            }

            // Step 9: Repair both tensors
            // Union failed indices for both
            auto eep_failed_pde = am_extraction->failed_pde;
            for (auto idx : eu_extraction->failed_pde) {
                if (std::find(eep_failed_pde.begin(), eep_failed_pde.end(), idx)
                    == eep_failed_pde.end()) {
                    eep_failed_pde.push_back(idx);
                }
            }
            auto eep_failed_spline = am_extraction->failed_spline;
            for (const auto& idx : eu_extraction->failed_spline) {
                eep_failed_spline.push_back(idx);
            }

            auto eep_repair = builder.repair_failed_slices(
                eep_tensor, eep_failed_pde, eep_failed_spline, axes);
            if (!eep_repair.has_value()) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::RepairFailed});
            }

            auto eu_repair = builder.repair_failed_slices(
                eu_extraction->tensor, eu_extraction->failed_pde,
                eu_extraction->failed_spline, axes);
            if (!eu_repair.has_value()) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::RepairFailed});
            }

            // Step 10: Fit two B-spline surfaces
            auto eep_fit = builder.fit_coeffs(eep_tensor, axes);
            if (!eep_fit.has_value()) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
            }

            auto eu_fit = builder.fit_coeffs(eu_extraction->tensor, axes);
            if (!eu_fit.has_value()) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
            }

            // Step 11: Build AmericanPriceSurface with both surfaces
            PriceTableMetadata eep_metadata{
                .K_ref = K_ref,
                .dividends = {.dividend_yield = config.dividends.dividend_yield},
                .content = SurfaceContent::NumericalEEP,
            };

            auto eep_surface = PriceTableSurface<4>::build(
                axes, std::move(eep_fit->coefficients), eep_metadata);
            if (!eep_surface.has_value()) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
            }

            PriceTableMetadata eu_metadata{
                .K_ref = K_ref,
                .dividends = {.dividend_yield = config.dividends.dividend_yield},
                .content = SurfaceContent::RawPrice,  // European stored as RawPrice
            };

            auto eu_surface_built = PriceTableSurface<4>::build(
                axes, std::move(eu_fit->coefficients), eu_metadata);
            if (!eu_surface_built.has_value()) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
            }

            auto aps = AmericanPriceSurface::create(
                *eep_surface, config.option_type, *eu_surface_built);
            if (!aps.has_value()) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
            }

            segment_configs.push_back(SegmentConfig{
                .surface = std::move(*aps),
                .tau_start = tau_start,
                .tau_end = tau_end,
            });
            prev_surface_ptr = &segment_configs.back().surface;
            continue;  // skip the existing single-surface path below
        }
```

**Step 5: Run test**

Run: `bazel test //tests:segmented_price_table_builder_test --test_filter=*NumericalEEP* --test_output=all`
Expected: PASS

**Step 6: Run all segmented tests**

Run: `bazel test //tests:segmented_price_table_builder_test --test_output=errors`
Expected: All pass (existing tests use `use_numerical_eep = false` by default)

**Step 7: Commit**

```bash
git add src/option/table/segmented_price_table_builder.hpp src/option/table/segmented_price_table_builder.cpp tests/segmented_price_table_builder_test.cc
git commit -m "Implement numerical EEP dual-solve pipeline for chained segments"
```

---

### Task 6: European PDE grid convergence test

**Files:**
- Test: `tests/american_option_test.cc`

**Step 1: Write the convergence test**

```cpp
// European PDE should converge toward BSM as grid refines
TEST(AmericanOptionTest, EuropeanPDEConvergesToBSM) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT},
        0.20);

    // BSM reference
    auto bsm = EuropeanOptionSolver(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT}, 0.20).solve().value();
    double bsm_price = bsm.value();

    // Solve European PDE at two accuracy levels
    double error_low = 0.0, error_high = 0.0;
    for (auto [accuracy, error_ptr] : {
        std::pair{GridAccuracyParams{.tol = 5e-3}, &error_low},
        std::pair{GridAccuracyParams{.tol = 1e-5}, &error_high}}) {

        auto [grid_spec, time_domain] = estimate_pde_grid(params, accuracy);
        size_t n = grid_spec.n_points();
        std::pmr::vector<double> buffer(PDEWorkspace::required_size(n),
                                         std::pmr::get_default_resource());
        auto ws = PDEWorkspace::from_buffer(buffer, n).value();
        auto solver = AmericanOptionSolver::create(params, ws,
            PDEGridConfig{.grid_spec = grid_spec, .n_time = time_domain.n_steps()}).value();
        solver.set_projection_enabled(false);
        auto result = solver.solve().value();
        *error_ptr = std::abs(result.value_at(100.0) - bsm_price);
    }

    // Higher accuracy should produce smaller error
    EXPECT_LT(error_high, error_low);
    // Both should be reasonably close (not wildly wrong)
    EXPECT_LT(error_low, 0.5);   // < $0.50 for low accuracy
    EXPECT_LT(error_high, 0.01); // < $0.01 for high accuracy
}
```

**Step 2: Run test**

Run: `bazel test //tests:american_option_test --test_filter=*ConvergesToBSM* --test_output=all`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/american_option_test.cc
git commit -m "Add European PDE grid convergence test"
```

---

### Task 7: Greeks consistency test for NumericalEEP

**Files:**
- Test: `tests/segmented_price_table_builder_test.cc`

**Step 1: Write the Greeks FD consistency test**

```cpp
TEST(SegmentedPriceTableBuilderTest, NumericalEEPGreeksMatchFD) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0,
                      .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},
        .grid = IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
        .use_numerical_eep = true,
    };

    auto surface = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(surface.has_value());

    // Query in chained segment
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.8, .sigma = 0.20, .rate = 0.05};

    double price = surface->price(q);
    double vega = surface->vega(q);
    double delta = surface->delta(q);

    // FD vega
    constexpr double eps_sigma = 1e-4;
    PriceQuery q_up = q; q_up.sigma += eps_sigma;
    PriceQuery q_dn = q; q_dn.sigma -= eps_sigma;
    double fd_vega = (surface->price(q_up) - surface->price(q_dn)) / (2.0 * eps_sigma);
    EXPECT_NEAR(vega, fd_vega, 0.1);  // within $0.10

    // FD delta
    constexpr double eps_spot = 0.5;
    PriceQuery q_sup = q; q_sup.spot += eps_spot;
    PriceQuery q_sdn = q; q_sdn.spot -= eps_spot;
    double fd_delta = (surface->price(q_sup) - surface->price(q_sdn)) / (2.0 * eps_spot);
    EXPECT_NEAR(delta, fd_delta, 0.05);
}
```

**Step 2: Run test**

Run: `bazel test //tests:segmented_price_table_builder_test --test_filter=*GreeksMatchFD* --test_output=all`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/segmented_price_table_builder_test.cc
git commit -m "Add Greeks FD consistency test for numerical EEP"
```

---

### Task 8: Update Python bindings

**Files:**
- Modify: `src/python/mango_bindings.cpp:700-703` (enum)
- Modify: `src/python/mango_bindings.cpp:831-890` (AmericanPriceSurface create)

**Step 1: Add NumericalEEP to Python enum**

In `src/python/mango_bindings.cpp`, add after `EarlyExercisePremium` (line 703):

```cpp
    py::enum_<mango::SurfaceContent>(m, "SurfaceContent")
        .value("RawPrice", mango::SurfaceContent::RawPrice)
        .value("EarlyExercisePremium", mango::SurfaceContent::EarlyExercisePremium)
        .value("NumericalEEP", mango::SurfaceContent::NumericalEEP);
```

**Step 2: Add create() overload to Python bindings**

In the `AmericanPriceSurface` binding section (around line 835), add:

```cpp
    .def_static("create_numerical_eep",
        [](std::shared_ptr<const mango::PriceTableSurface<4>> eep,
           mango::OptionType type,
           std::shared_ptr<const mango::PriceTableSurface<4>> eu) {
            return mango::AmericanPriceSurface::create(eep, type, eu);
        },
        py::arg("eep_surface"), py::arg("option_type"), py::arg("eu_surface"))
```

**Step 3: Build Python bindings**

Run: `bazel build //src/python:mango_option`
Expected: BUILD SUCCESS

**Step 4: Commit**

```bash
git add src/python/mango_bindings.cpp
git commit -m "Add NumericalEEP to Python bindings"
```

---

### Task 9: Full regression test

**Step 1: Run all tests**

Run: `bazel test //...`
Expected: All tests pass

**Step 2: Build benchmarks**

Run: `bazel build //benchmarks/...`
Expected: BUILD SUCCESS

**Step 3: Commit any fixups, tag as complete**

If any tests needed fixing, commit the fixes. Then update the design doc status.
