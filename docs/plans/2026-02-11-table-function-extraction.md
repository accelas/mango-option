# Table Function Extraction Refactoring

**Date:** 2026-02-11
**Status:** Planning
**Priority:** P2 - Code quality improvement

## Goal

Reduce function sizes in `src/option/table/` by extracting named helpers. This is purely mechanical extraction with no architectural changes. The refactoring improves readability and maintainability without changing any public APIs or behavior.

**Priority order:**
1. Shared utilities first (K_ref resolution, domain expansion)
2. Phase extractions from `run_refinement()` and builders
3. Pipeline deduplication in B-spline builders

**Success criteria:**
- `run_refinement()` drops from 255 to ~60 lines
- `SegmentedPriceTableBuilder::build()` drops from 272 to ~50 lines
- `build_dimensionless_bspline()` drops from 76 to ~40 lines
- All tests pass (`bazel test //...`)
- Benchmarks compile (`bazel build //benchmarks:interp_iv_safety`)

## Architecture

**No architectural changes** — this is pure refactoring. The changes are:

- **Shared utilities extraction** (tasks 1-2): K_ref resolution and domain expansion logic is currently duplicated 3 times across Chebyshev and B-spline adaptive builders. Extract to `adaptive_refinement.cpp`.

- **Phase extraction** (tasks 3-5): `run_refinement()` has 3 clear phases (sample generation, evaluation, result saving). Extract each to a named helper.

- **Pipeline deduplication** (task 6): The B-spline surface assembly pipeline (extract → repair → EEP → fit) appears in 3 places. Extract to `assemble_surface()` method.

- **Segmented builder cleanup** (tasks 7-8): The segmented builder's 272-line `build()` function has two large blocks (grid expansion, per-segment build loop). Extract each.

- **Repair phase split** (task 9): `repair_failed_slices()` has two distinct phases (spline failures, PDE failures). Split for clarity.

- **Dimensionless builder cleanup** (task 10): Extract grid setup from `build_dimensionless_bspline()`.

**Testing strategy:**
- TDD for shared utilities (tasks 1-2): Write unit tests first
- Refactor verification for all other tasks: Existing tests serve as regression suite
- Run full test suite after each task

## Tech Stack

- **Language:** C++23
- **Build:** Bazel
- **Test framework:** GoogleTest
- **Files modified:** `src/option/table/adaptive_refinement.{hpp,cpp}`, `src/option/table/bspline/bspline_builder.{hpp,cpp}`, `src/option/table/bspline/bspline_adaptive.cpp`, `src/option/table/bspline/bspline_segmented_builder.cpp`, `src/option/table/chebyshev/chebyshev_adaptive.cpp`, `src/option/interpolated_iv_solver.cpp`

---

## Task 1: Extract `resolve_k_refs()` shared utility

**Goal:** Extract K_ref resolution logic currently duplicated 3 times.

**Files:**
- Modify: `src/option/table/adaptive_refinement.hpp` — add declaration
- Modify: `src/option/table/adaptive_refinement.cpp` — add implementation
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp` — replace inline K_ref generation in `build_adaptive_chebyshev_segmented()` (lines 821-841) and `build_chebyshev_segmented_manual()` (similar block)
- Modify: `src/option/table/bspline/bspline_adaptive.cpp` — replace inline K_ref generation in `build_adaptive_bspline_segmented()` (lines 693-718)
- Modify: `tests/adaptive_grid_builder_test.cc` — add unit test

**Steps:**

1. **Write failing test** in `tests/adaptive_grid_builder_test.cc`:
   ```cpp
   TEST(ResolveKRefsTest, ExplicitKRefs) {
       KRefConfig config{.K_refs = {80.0, 100.0, 120.0}};
       auto result = mango::resolve_k_refs(config, 100.0);
       ASSERT_TRUE(result.has_value());
       EXPECT_EQ(result->size(), 3);
       EXPECT_DOUBLE_EQ(result->at(0), 80.0);
   }

   TEST(ResolveKRefsTest, GeneratedKRefs) {
       KRefConfig config{.count = 5, .span = 0.4};
       auto result = mango::resolve_k_refs(config, 100.0);
       ASSERT_TRUE(result.has_value());
       EXPECT_EQ(result->size(), 5);
       // Verify log-spaced distribution
   }

   TEST(ResolveKRefsTest, ErrorCases) {
       KRefConfig bad_count{.count = 0, .span = 0.4};
       EXPECT_FALSE(mango::resolve_k_refs(bad_count, 100.0).has_value());

       KRefConfig bad_span{.count = 3, .span = 0.0};
       EXPECT_FALSE(mango::resolve_k_refs(bad_span, 100.0).has_value());
   }
   ```

2. **Add declaration** to `src/option/table/adaptive_refinement.hpp`:
   ```cpp
   [[nodiscard]] std::expected<std::vector<double>, PriceTableError>
   resolve_k_refs(const KRefConfig& config, double spot);
   ```

3. **Add implementation** to `src/option/table/adaptive_refinement.cpp`:
   ```cpp
   std::expected<std::vector<double>, PriceTableError>
   resolve_k_refs(const KRefConfig& config, double spot) {
       // If K_refs explicitly provided, sort and return
       if (!config.K_refs.empty()) {
           std::vector<double> sorted = config.K_refs;
           std::sort(sorted.begin(), sorted.end());
           return sorted;
       }

       // Generate from count/span
       if (config.count < 1) {
           return std::unexpected(PriceTableError{
               .code = PriceTableErrorCode::INVALID_PARAMETERS,
               .message = "K_ref count must be >= 1"});
       }
       if (config.span <= 0.0) {
           return std::unexpected(PriceTableError{
               .code = PriceTableErrorCode::INVALID_PARAMETERS,
               .message = "K_ref span must be > 0"});
       }

       std::vector<double> K_refs(config.count);
       if (config.count == 1) {
           K_refs[0] = spot;
       } else {
           double log_min = std::log(spot) - config.span / 2.0;
           double log_max = std::log(spot) + config.span / 2.0;
           double delta = (log_max - log_min) / (config.count - 1);
           for (size_t i = 0; i < config.count; ++i) {
               K_refs[i] = std::exp(log_min + i * delta);
           }
       }
       return K_refs;
   }
   ```

4. **Run test** to verify implementation: `bazel test //tests:adaptive_grid_builder_test`

5. **Replace call site** in `src/option/table/chebyshev/chebyshev_adaptive.cpp` in `build_adaptive_chebyshev_segmented()` (lines 821-841):
   - Remove inline K_ref generation logic
   - Replace with: `auto K_refs = TRY(resolve_k_refs(config.kref_config, config.spot));`

6. **Replace call site** in `src/option/table/chebyshev/chebyshev_adaptive.cpp` in `build_chebyshev_segmented_manual()` (similar block):
   - Replace with same `TRY(resolve_k_refs(...))` call

7. **Replace call site** in `src/option/table/bspline/bspline_adaptive.cpp` in `build_adaptive_bspline_segmented()` (lines 693-718):
   - Remove inline K_ref generation logic
   - Replace with: `auto K_refs = TRY(resolve_k_refs(config.kref_config, config.spot));`

8. **Verify** full test suite: `bazel test //...`

9. **Verify** benchmarks compile: `bazel build //benchmarks:interp_iv_safety`

**Commit:** `Extract resolve_k_refs() shared utility`

---

## Task 2: Extract `expand_segmented_domain()` shared utility

**Goal:** Extract domain expansion logic currently duplicated 3 times.

**Files:**
- Modify: `src/option/table/adaptive_refinement.hpp` — add `DomainBounds` struct and declaration
- Modify: `src/option/table/adaptive_refinement.cpp` — add implementation
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp` — replace inline domain expansion in `build_adaptive_chebyshev_segmented()` (lines 843-872) and `build_chebyshev_segmented_manual()`
- Modify: `src/option/table/bspline/bspline_adaptive.cpp` — replace inline domain expansion in `probe_and_build()` (lines 189-227)
- Modify: `tests/adaptive_grid_builder_test.cc` — add unit test

**Steps:**

1. **Write failing test** in `tests/adaptive_grid_builder_test.cc`:
   ```cpp
   TEST(ExpandSegmentedDomainTest, NoDividends) {
       IVGrid domain{
           .moneyness = {0.8, 1.0, 1.2},
           .tau = {0.25, 0.5, 1.0},
           .vol = {0.20, 0.30},
           .rate = {0.03, 0.05}
       };
       double maturity = 1.0;
       std::vector<Dividend> divs;
       auto result = mango::expand_segmented_domain(domain, maturity, 0.02, divs, 100.0);
       ASSERT_TRUE(result.has_value());
       // Verify expansion by standard spreads
       EXPECT_LT(result->min_m, 0.8);
       EXPECT_GT(result->max_m, 1.2);
   }

   TEST(ExpandSegmentedDomainTest, WithDividends) {
       IVGrid domain{.moneyness = {0.8, 1.0, 1.2}, .tau = {0.5, 1.0}, .vol = {0.20}, .rate = {0.05}};
       std::vector<Dividend> divs = {Dividend{.calendar_time = 0.25, .amount = 2.0}};
       auto result = mango::expand_segmented_domain(domain, 1.0, 0.0, divs, 100.0);
       ASSERT_TRUE(result.has_value());
       // Verify moneyness lower bound expanded for dividends
       EXPECT_LT(result->min_m, std::log(0.8) - 2.0/100.0);
   }

   TEST(ExpandSegmentedDomainTest, EmptyDomain) {
       IVGrid empty_domain{};
       auto result = mango::expand_segmented_domain(empty_domain, 1.0, 0.0, {}, 100.0);
       EXPECT_FALSE(result.has_value());
   }
   ```

2. **Add struct** to `src/option/table/adaptive_refinement.hpp`:
   ```cpp
   struct DomainBounds {
       double min_m, max_m;
       double min_tau, max_tau;
       double min_vol, max_vol;
       double min_rate, max_rate;
   };

   [[nodiscard]] std::expected<DomainBounds, PriceTableError>
   expand_segmented_domain(const IVGrid& domain,
                          double maturity,
                          double dividend_yield,
                          const std::vector<Dividend>& discrete_dividends,
                          double min_K_ref);
   ```

3. **Add implementation** to `src/option/table/adaptive_refinement.cpp`:
   ```cpp
   std::expected<DomainBounds, PriceTableError>
   expand_segmented_domain(const IVGrid& domain,
                          double maturity,
                          double dividend_yield,
                          const std::vector<Dividend>& discrete_dividends,
                          double min_K_ref) {
       // Validate domain non-empty
       if (domain.moneyness.empty() || domain.tau.empty() ||
           domain.vol.empty() || domain.rate.empty()) {
           return std::unexpected(PriceTableError{
               .code = PriceTableErrorCode::INVALID_PARAMETERS,
               .message = "Domain grids must be non-empty"});
       }

       // Convert moneyness (S/K) to log-moneyness (ln S/K)
       std::vector<double> log_m(domain.moneyness.size());
       std::transform(domain.moneyness.begin(), domain.moneyness.end(),
                     log_m.begin(), [](double m) { return std::log(m); });

       // Expand moneyness lower bound for cumulative dividends
       double cum_div = 0.0;
       for (const auto& div : discrete_dividends) {
           if (div.calendar_time <= maturity) {
               cum_div += div.amount;
           }
       }
       double min_log_m = *std::min_element(log_m.begin(), log_m.end());
       if (cum_div > 0.0) {
           min_log_m -= cum_div / min_K_ref;
       }
       double max_log_m = *std::max_element(log_m.begin(), log_m.end());

       // Standard spreads: m=0.10, vol=0.10, rate=0.04, tau=0.1
       auto expand = [](double min_val, double max_val, double spread) {
           return std::make_pair(min_val - spread, max_val + spread);
       };

       auto [min_m_exp, max_m_exp] = expand(min_log_m, max_log_m, 0.10);
       auto [min_vol_exp, max_vol_exp] = expand(
           *std::min_element(domain.vol.begin(), domain.vol.end()),
           *std::max_element(domain.vol.begin(), domain.vol.end()),
           0.10);
       auto [min_rate_exp, max_rate_exp] = expand(
           *std::min_element(domain.rate.begin(), domain.rate.end()),
           *std::max_element(domain.rate.begin(), domain.rate.end()),
           0.04);

       double min_tau = *std::min_element(domain.tau.begin(), domain.tau.end());
       double max_tau = *std::max_element(domain.tau.begin(), domain.tau.end());
       auto [min_tau_exp, max_tau_exp] = expand(min_tau, max_tau, 0.1);

       // Cap tau at maturity
       max_tau_exp = std::min(max_tau_exp, maturity);

       return DomainBounds{
           .min_m = min_m_exp, .max_m = max_m_exp,
           .min_tau = min_tau_exp, .max_tau = max_tau_exp,
           .min_vol = min_vol_exp, .max_vol = max_vol_exp,
           .min_rate = min_rate_exp, .max_rate = max_rate_exp
       };
   }
   ```

4. **Run test** to verify implementation: `bazel test //tests:adaptive_grid_builder_test`

5. **Replace call site** in `src/option/table/chebyshev/chebyshev_adaptive.cpp` in `build_adaptive_chebyshev_segmented()` (lines 843-872):
   - Remove inline domain expansion logic
   - Replace with:
     ```cpp
     auto bounds = TRY(expand_segmented_domain(
         config.domain, config.maturity, config.dividend_yield,
         config.discrete_dividends, K_refs.front()));

     // Add Chebyshev CC headroom
     double cc_headroom_m = (bounds.max_m - bounds.min_m) * 0.15;
     // ... apply headroom to all 4 axes
     ```

6. **Replace call site** in `src/option/table/chebyshev/chebyshev_adaptive.cpp` in `build_chebyshev_segmented_manual()`:
   - Replace with same pattern (expand + add CC headroom)

7. **Replace call site** in `src/option/table/bspline/bspline_adaptive.cpp` in `probe_and_build()` (lines 189-227):
   - Remove inline domain expansion logic
   - Replace with:
     ```cpp
     auto bounds = TRY(expand_segmented_domain(
         config.domain, maturity, config.dividend_yield,
         config.discrete_dividends, K_refs.front()));

     // Add B-spline support headroom
     double headroom_m = 3.0 * (bounds.max_m - bounds.min_m) / (n_m - 1);
     // ... apply headroom to all 4 axes
     ```

8. **Verify** full test suite: `bazel test //...`

9. **Verify** benchmarks compile: `bazel build //benchmarks:interp_iv_safety`

**Commit:** `Extract expand_segmented_domain() shared utility`

---

## Task 3: Extract `generate_validation_samples()` from `run_refinement()`

**Goal:** Extract sample generation phase from `run_refinement()` (lines 378-428).

**Files:**
- Modify: `src/option/table/adaptive_refinement.cpp`

**Steps:**

1. **Extract static function** near top of file:
   ```cpp
   static std::vector<std::array<double, 4>> generate_validation_samples(
       const AdaptiveGridParams& params,
       size_t iteration,
       const std::array<std::pair<double, double>, 4>& bounds,
       const std::array<std::vector<size_t>, 4>& focus_bins,
       bool focus_active) {
       // Move lines 378-428 here
       // ... Sobol/Random/Focus logic
   }
   ```

2. **Replace call site** in `run_refinement()`:
   ```cpp
   auto samples = generate_validation_samples(
       params, iteration, bounds, focus_bins, focus_active);
   ```

3. **Verify** full test suite: `bazel test //...`

**Commit:** `Extract generate_validation_samples() from run_refinement()`

---

## Task 4: Extract `evaluate_samples()` from `run_refinement()`

**Goal:** Extract sample evaluation phase from `run_refinement()` (lines 430-472).

**Files:**
- Modify: `src/option/table/adaptive_refinement.cpp`

**Steps:**

1. **Add struct** near top of file:
   ```cpp
   struct ValidationResult {
       double max_error;
       double avg_error;
       size_t valid_samples;
       size_t pde_solves;
       ErrorBins error_bins;
   };
   ```

2. **Extract static function**:
   ```cpp
   static std::expected<ValidationResult, PriceTableError>
   evaluate_samples(
       const std::vector<std::array<double, 4>>& samples,
       const SurfaceHandle& handle,
       const ValidateFn& validate_fn,
       const ComputeErrorFn& compute_error,
       const RefinementContext& ctx) {
       // Move lines 430-472 here
       // ... loop over samples, compute errors, accumulate stats
   }
   ```

3. **Replace call site** in `run_refinement()`:
   ```cpp
   auto eval_result = TRY(evaluate_samples(
       samples, handle, validate_fn, compute_error, ctx));
   double max_error = eval_result.max_error;
   double avg_error = eval_result.avg_error;
   // ... use eval_result fields
   ```

4. **Verify** full test suite: `bazel test //...`

**Commit:** `Extract evaluate_samples() from run_refinement()`

---

## Task 5: Extract `save_refinement_result()` from `run_refinement()`

**Goal:** Extract result-packing code duplicated at lines 491-503 and 513-524.

**Files:**
- Modify: `src/option/table/adaptive_refinement.cpp`

**Steps:**

1. **Extract static function**:
   ```cpp
   static void save_refinement_result(
       RefinementResult& result,
       const std::vector<double>& moneyness,
       const std::vector<double>& tau,
       const std::vector<double>& vol,
       const std::vector<double>& rate,
       double max_error,
       double avg_error,
       bool target_met) {
       result.final_grid = IVGrid{
           .moneyness = moneyness,
           .tau = tau,
           .vol = vol,
           .rate = rate
       };
       result.iterations_run = /* from context */;
       result.max_error_bps = max_error;
       result.avg_error_bps = avg_error;
       result.target_met = target_met;
   }
   ```

2. **Replace both call sites** (early exit and loop exit):
   ```cpp
   save_refinement_result(result, current_moneyness, current_tau,
                         current_vol, current_rate, max_error, avg_error, true);
   ```

3. **Verify** full test suite: `bazel test //...`

**After tasks 3-5,** `run_refinement()` should drop from 255 to ~60 lines.

**Commit:** `Extract save_refinement_result() from run_refinement()`

---

## Task 6: Extract `assemble_surface()` method on `PriceTableBuilderND`

**Goal:** Deduplicate B-spline surface assembly pipeline appearing in 3 places.

**Files:**
- Modify: `src/option/table/bspline/bspline_builder.hpp` — add method declaration
- Modify: `src/option/table/bspline/bspline_builder.cpp` — add implementation, refactor `build()` to use it
- Modify: `src/option/table/bspline/bspline_adaptive.cpp` — refactor `build_cached_surface()` to use it
- Modify: `src/option/table/bspline/bspline_segmented_builder.cpp` — refactor per-segment build in `SegmentedPriceTableBuilder::build()` to use it

**Steps:**

1. **Add declaration** to `src/option/table/bspline/bspline_builder.hpp` in `PriceTableBuilderND`:
   ```cpp
   template <size_t N>
   [[nodiscard]] std::expected<std::shared_ptr<const PriceTableSurface>, PriceTableError>
   assemble_surface(
       const BatchAmericanOptionResult& batch,
       const PriceTableAxes& axes,
       double K_ref,
       const DividendSpec& div_spec,
       bool apply_eep = true) const;
   ```

2. **Add implementation** to `src/option/table/bspline/bspline_builder.cpp`:
   ```cpp
   template <size_t N>
   std::expected<std::shared_ptr<const PriceTableSurface>, PriceTableError>
   PriceTableBuilderND<N>::assemble_surface(
       const BatchAmericanOptionResult& batch,
       const PriceTableAxes& axes,
       double K_ref,
       const DividendSpec& div_spec,
       bool apply_eep) const {

       // Phase 1: Extract tensor
       auto values_tensor = TRY(extract_tensor(batch));

       // Phase 2: Repair failed slices
       auto repaired = TRY(repair_failed_slices(values_tensor, batch, axes, K_ref, div_spec));

       // Phase 3: Optional EEP decomposition
       std::vector<double> final_values;
       if (apply_eep) {
           auto [eep, residual] = TRY(eep_decompose(repaired, axes.log_m, K_ref));
           final_values = std::move(residual);
       } else {
           final_values = std::move(repaired);
       }

       // Phase 4: Fit coefficients
       auto coeffs = TRY(fit_coeffs(final_values, axes));

       // Phase 5: Build surface
       return PriceTableSurface::build(
           axes, std::move(coeffs), K_ref, div_spec, apply_eep);
   }

   // Explicit instantiation
   template class PriceTableBuilderND<4>;
   ```

3. **Refactor `build()`** in `src/option/table/bspline/bspline_builder.cpp`:
   - Replace ~40 lines of extract → repair → EEP → fit → build with:
     ```cpp
     return assemble_surface(batch_result, axes, K_ref_, div_spec_, true);
     ```

4. **Refactor `build_cached_surface()`** in `src/option/table/bspline/bspline_adaptive.cpp`:
   - Replace surface assembly logic with:
     ```cpp
     return builder.assemble_surface(batch_result, axes, K_ref, div_spec, true);
     ```

5. **Refactor per-segment build** in `src/option/table/bspline/bspline_segmented_builder.cpp`:
   - In the loop body of `SegmentedPriceTableBuilder::build()`, replace surface assembly with:
     ```cpp
     auto surface = TRY(builder_.assemble_surface(
         batch_result, axes, seg_K_ref, div_spec, false));  // no EEP for segmented
     ```

6. **Verify** full test suite: `bazel test //...`

7. **Verify** benchmarks compile: `bazel build //benchmarks:interp_iv_safety`

**Commit:** `Extract assemble_surface() method on PriceTableBuilderND`

---

## Task 7: Extract `expand_log_moneyness_grid()` from `SegmentedPriceTableBuilder::build()`

**Goal:** Extract log-moneyness grid expansion logic (lines 155-219).

**Files:**
- Modify: `src/option/table/bspline/bspline_segmented_builder.cpp`

**Steps:**

1. **Extract static function** near top of file:
   ```cpp
   static std::expected<std::vector<double>, PriceTableError>
   expand_log_moneyness_grid(
       const std::vector<double>& input_grid,
       const std::vector<Dividend>& dividends,
       double K_ref,
       double sigma_max,
       double max_segment_width) {
       // Move lines 155-219 here
       // Logic: convert to log-space, refine near dividend jumps, cap segment width
   }
   ```

2. **Replace call site** in `build()`:
   ```cpp
   auto log_m_grid = TRY(expand_log_moneyness_grid(
       config_.grid.moneyness, config_.discrete_dividends,
       K_ref, sigma_max, max_segment_width));
   ```

3. **Verify** full test suite: `bazel test //...`

**Commit:** `Extract expand_log_moneyness_grid() from SegmentedPriceTableBuilder`

---

## Task 8: Extract `build_segment()` from `SegmentedPriceTableBuilder::build()` loop body

**Goal:** Extract per-segment build logic from loop body (lines 232-367).

**Files:**
- Modify: `src/option/table/bspline/bspline_segmented_builder.cpp`

**Steps:**

1. **Extract static function**:
   ```cpp
   static std::expected<std::shared_ptr<const PriceTableSurface>, PriceTableError>
   build_segment(
       size_t seg_idx,
       const std::vector<double>& boundaries,
       const std::vector<Dividend>& dividends,
       const std::vector<double>& log_m_grid,
       const Config& config,
       std::shared_ptr<const PriceTableSurface> prev_surface) {
       // Move loop body here
       // Logic: compute segment bounds, adjust for dividends, build batch, assemble surface
   }
   ```

2. **Replace loop body** in `build()`:
   ```cpp
   for (size_t i = 0; i < boundaries.size() - 1; ++i) {
       auto surface = TRY(build_segment(
           i, boundaries, config_.discrete_dividends, log_m_grid,
           config_, segments.empty() ? nullptr : segments.back()));
       segments.push_back(std::move(surface));
   }
   ```

3. **Verify** full test suite: `bazel test //...`

**After tasks 7-8,** `SegmentedPriceTableBuilder::build()` should drop from 272 to ~50 lines.

**Commit:** `Extract build_segment() from SegmentedPriceTableBuilder loop`

---

## Task 9: Extract `repair_spline_failures()` and `repair_pde_failures()` from `repair_failed_slices()`

**Goal:** Split `repair_failed_slices()` into two clear phases.

**Files:**
- Modify: `src/option/table/bspline/bspline_builder.cpp`

**Steps:**

1. **Extract first phase** as private helper:
   ```cpp
   template <size_t N>
   std::expected<std::vector<double>, PriceTableError>
   PriceTableBuilderND<N>::repair_spline_failures(
       const std::vector<double>& values_tensor,
       const BatchAmericanOptionResult& batch,
       const PriceTableAxes& axes,
       double K_ref,
       const DividendSpec& div_spec) const {
       // Phase 1: Fix spline fitting failures
       // ~50 lines
   }
   ```

2. **Extract second phase** as private helper:
   ```cpp
   template <size_t N>
   std::expected<std::vector<double>, PriceTableError>
   PriceTableBuilderND<N>::repair_pde_failures(
       const std::vector<double>& values_tensor,
       const BatchAmericanOptionResult& batch,
       const PriceTableAxes& axes,
       double K_ref,
       const DividendSpec& div_spec) const {
       // Phase 2: Fix PDE solver failures
       // ~50 lines
   }
   ```

3. **Refactor `repair_failed_slices()`** to become a ~25 line dispatcher:
   ```cpp
   template <size_t N>
   std::expected<std::vector<double>, PriceTableError>
   PriceTableBuilderND<N>::repair_failed_slices(
       const std::vector<double>& values_tensor,
       const BatchAmericanOptionResult& batch,
       const PriceTableAxes& axes,
       double K_ref,
       const DividendSpec& div_spec) const {

       auto after_spline = TRY(repair_spline_failures(
           values_tensor, batch, axes, K_ref, div_spec));

       return repair_pde_failures(
           after_spline, batch, axes, K_ref, div_spec);
   }
   ```

4. **Add declarations** to `src/option/table/bspline/bspline_builder.hpp` as private methods.

5. **Verify** full test suite: `bazel test //...`

**Commit:** `Split repair_failed_slices() into spline and PDE phases`

---

## Task 10: Extract `build_dimensionless_grid()` from `build_dimensionless_bspline()`

**Goal:** Extract grid setup portion from 76-line function.

**Files:**
- Modify: `src/option/interpolated_iv_solver.cpp`

**Steps:**

1. **Extract static function** near top of file (after includes):
   ```cpp
   struct DimensionlessGridSpec {
       std::vector<double> log_m;
       std::vector<double> tau;
       std::vector<double> vol;
   };

   static std::expected<DimensionlessGridSpec, PriceTableError>
   build_dimensionless_grid(
       const IVGrid& input_grid,
       double maturity,
       const GridAccuracyParams& accuracy) {
       // Extract ~35 lines of grid construction logic
       // Convert moneyness to log-space, validate tau <= maturity, etc.
   }
   ```

2. **Replace call site** in `build_dimensionless_bspline()`:
   ```cpp
   auto grid_spec = TRY(build_dimensionless_grid(
       config.grid, config.maturity, config.accuracy));

   // Use grid_spec.log_m, grid_spec.tau, grid_spec.vol
   ```

3. **Verify** full test suite: `bazel test //...`

4. **Verify** benchmarks compile: `bazel build //benchmarks:interp_iv_safety`

**After this task,** `build_dimensionless_bspline()` should drop from 76 to ~40 lines.

**Commit:** `Extract build_dimensionless_grid() from build_dimensionless_bspline()`

---

## Final Verification

After completing all tasks:

1. **Run full test suite:** `bazel test //...` — expect all 116 tests to pass
2. **Build benchmarks:** `bazel build //benchmarks:interp_iv_safety` — expect clean build
3. **Verify line counts** in target functions:
   - `run_refinement()`: ~60 lines (down from 255)
   - `SegmentedPriceTableBuilder::build()`: ~50 lines (down from 272)
   - `build_dimensionless_bspline()`: ~40 lines (down from 76)
   - `repair_failed_slices()`: ~25 lines (down from ~100)

4. **No public API changes:** All extractions are internal helpers

---

## Notes

- Tasks 1-2 are independent shared utilities — do these first
- Tasks 3-5 all modify `run_refinement()` — do in order
- Task 6 is independent and can be done anytime after task 2
- Tasks 7-8 both modify `SegmentedPriceTableBuilder` — do in order
- Task 9 is independent
- Task 10 is independent

**Testing philosophy:**
- TDD for shared utilities (tasks 1-2): Write tests first to drive the API
- Regression verification for refactors (tasks 3-10): Existing tests verify no behavior changes

**Commit granularity:** One commit per task (10 total commits)
