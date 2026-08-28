# Discrete Dividend Wiring (#448 + #440 item 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `compute_vol_surface` converts and forwards discrete dividend schedules into each `IVQuery`, and `InterpolatedIVSolver` rejects queries whose schedule does not match the surface it was built with — closing the double silent-drop path from issues #448 and #440 (item 1).

**Architecture:** Two halves. (1) Simple layer: a testable `convert_discrete_dividends()` helper maps `simple::Dividend{ex_date, amount}` → `mango::Dividend{calendar_time, amount}` relative to the valuation time, filtered per expiry slice; `compute_vol_surface` populates `IVQuery::discrete_dividends` from it. (2) Solver layer: `InterpolatedIVSolver` stores an optional build-time schedule (`std::nullopt` = unknown, for tables deserialized from Parquet which do not persist schedules) and `validate_query` rejects a *non-empty* query schedule that mismatches the known build schedule. Empty query schedules stay valid against segmented surfaces (established contract: the surface's schedule is authoritative — see `tests/discrete_dividend_iv_integration_test.cc`).

**Tech Stack:** C++23, `std::expected`, GoogleTest, Bazel.

**Spec:** GitHub issues #448 (full body) and #440 item 1. No separate design doc — the issues carry locations, suggested fixes, and regression tests; open semantics are pinned in Decisions below.

## Decisions

1. **Full wiring, not the minimal `ComputeError`.** #448 offers "return an error on the discrete alternative" as minimum; we do the real fix.
2. **Validation semantics (#440.1):**
   - Query schedule **empty** → always valid. Existing segmented-surface usage passes empty schedules and prices off the surface's baked-in dividends (`discrete_dividend_iv_integration_test.cc` relies on this; CLAUDE.md Pattern 3 likewise). Rejecting would break the documented API.
   - Query schedule **non-empty**, build schedule **known** → compare against the build schedule restricted to `calendar_time <= query.maturity + tol` (a shorter-dated query legitimately carries only the dividends inside its life). Sizes must match; each entry must match within `kTimeTol = 1e-6` years (~30 s) and `kAmountTol = 1e-6` dollars. Mismatch → new `ValidationErrorCode::DiscreteDividendMismatch` → new `IVErrorCode::DiscreteDividendMismatch`.
   - Query schedule **non-empty**, build schedule **unknown** (`std::nullopt`: segmented table loaded from Parquet — schedules are not persisted, see `src/option/table/serialization/price_table_data.hpp:17-21`) → accept unverified, documented.
   - Continuous surfaces have build schedule known-**empty**, so a non-empty query schedule against them is rejected — this is the loud failure that replaces #448's silent dividend-free solve.
3. **Schedule provenance:** `InterpolatedIVSolver::create()` gains a defaulted `std::optional<std::vector<Dividend>>` parameter (default known-empty — correct for the documented direct-`create` path, which is continuous-only). `AnyPriceTable::make_iv_solver` infers when its caller passes nothing: `*MultiKRefSurface` table types → unknown; all others → known-empty. `make_interpolated_iv_solver` passes `config.discrete_dividends->discrete_dividends` (or known-empty).
4. **Conversion filter:** keep dividends with `0 < t <= tau_max` where `t = compute_tau(val_time, ex_date)`. `t <= 0` covers both already-paid dividends and unparseable timestamps (`compute_tau` returns 0.0 on parse failure). Output sorted by `calendar_time`.
5. **Either/or `DividendSpec` stays.** The continuous+discrete combination the core `PricingParams` supports remains inexpressible in the market-data layer — noted in #448 as a deliberate follow-up decision, out of scope here.
6. **`div_yield` stays 0.0 when the discrete alternative is active** (they are mutually exclusive in `simple::DividendSpec`).

## Global Constraints

- Every new/edited source file starts with `// SPDX-License-Identifier: MIT` (already present in all files touched).
- Library code must not printf/fprintf.
- Regression tests carry the `// Regression:` / `// Bug:` comment format from CLAUDE.md.
- Pre-PR: `bazel test //...`, `bazel build //benchmarks/...`, `bazel build //src/python:mango_option` all green.
- Commit messages: imperative mood, ≤50-char subject, body wrapped at 72.
- Worktree: all work in `/home/kai/work/mango-option/.claude/worktrees/fix-448-discrete-dividends` on branch `worktree-fix-448-discrete-dividends`.

---

### Task 1: Solver-side schedule validation, threaded through the factory

**Files:**
- Modify: `src/support/error_types.hpp:29-45` (ValidationErrorCode) and `:116-135` (IVErrorCode)
- Modify: `src/option/iv_result.hpp:55-92` (`validation_error_to_iv_error`)
- Modify: `src/option/interpolated_iv_solver.hpp` (`create` at ~line 332, private ctor + members at ~104-129, `validate_query` at ~377)
- Modify: `src/option/price_table_factory.hpp:40` (`make_iv_solver` signature)
- Modify: `src/option/price_table_factory.cpp:790-816,883-891` (`make_iv_solver` body, `make_interpolated_iv_solver`)
- Modify: `src/python/mango_bindings.cpp:477,503` (new enum value)
- Test: `tests/discrete_dividend_iv_integration_test.cc` (segmented cases), `tests/iv_solver_factory_test.cc` (continuous rejection)

**Interfaces:**
- Consumes: existing `InterpolatedIVSolver<Surface>::create(Surface, const InterpolatedIVSolverConfig&)`, `AnyPriceTable::make_iv_solver(const InterpolatedIVSolverConfig&)`.
- Produces (later tasks rely on):
  - `ValidationErrorCode::DiscreteDividendMismatch`, `IVErrorCode::DiscreteDividendMismatch`
  - `InterpolatedIVSolver<Surface>::create(Surface surface, const InterpolatedIVSolverConfig& config = {}, std::optional<std::vector<Dividend>> build_dividends = std::vector<Dividend>{})`
  - `AnyPriceTable::make_iv_solver(const InterpolatedIVSolverConfig& config = {}, std::optional<std::vector<Dividend>> build_dividends = std::nullopt)` where `nullopt` means *infer from table type*
  - Behavior: `AnyInterpIVSolver::solve` returns `IVErrorCode::DiscreteDividendMismatch` for a non-empty query schedule that mismatches a known build schedule.

- [ ] **Step 1: Write the failing tests (segmented side)**

Append to `tests/discrete_dividend_iv_integration_test.cc` (fixture `DiscreteDividendIVIntegrationTest` builds a segmented PUT solver, maturity 1.0, one dividend `{0.5, 2.0}`; reuse it):

```cpp
// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: validate_query ignored query.discrete_dividends (#440 item 1)
// Bug: A query carrying a discrete schedule against a mismatched surface
//      priced off the wrong surface with no error.
TEST_F(DiscreteDividendIVIntegrationTest, MatchingQueryScheduleAccepted) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.8,
            .rate = 0.05, .option_type = OptionType::PUT},
        0.20, {{.calendar_time = 0.5, .amount = 2.0}});
    auto price_result = solve_american_option(params);
    ASSERT_TRUE(price_result.has_value());

    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.8;
    query.rate = RateSpec{0.05};
    query.option_type = OptionType::PUT;
    query.market_price = price_result->value();
    query.discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}};

    auto iv_result = solver_->solve(query);
    ASSERT_TRUE(iv_result.has_value())
        << "matching schedule must be accepted; error code: "
        << (iv_result.has_value() ? 0 : static_cast<int>(iv_result.error().code));
    EXPECT_NEAR(iv_result->implied_vol, 0.20, 0.02);
}

// Regression: mismatched dividend amount must be rejected loudly (#440 item 1)
TEST_F(DiscreteDividendIVIntegrationTest, MismatchedAmountRejected) {
    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.8;
    query.rate = RateSpec{0.05};
    query.option_type = OptionType::PUT;
    query.market_price = 5.0;
    query.discrete_dividends = {{.calendar_time = 0.5, .amount = 3.0}};

    auto iv_result = solver_->solve(query);
    ASSERT_FALSE(iv_result.has_value());
    EXPECT_EQ(iv_result.error().code, IVErrorCode::DiscreteDividendMismatch);
}

// Regression: extra dividend in the query must be rejected (#440 item 1)
TEST_F(DiscreteDividendIVIntegrationTest, ExtraDividendRejected) {
    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.8;
    query.rate = RateSpec{0.05};
    query.option_type = OptionType::PUT;
    query.market_price = 5.0;
    query.discrete_dividends = {
        {.calendar_time = 0.25, .amount = 1.0},
        {.calendar_time = 0.5, .amount = 2.0}};

    auto iv_result = solver_->solve(query);
    ASSERT_FALSE(iv_result.has_value());
    EXPECT_EQ(iv_result.error().code, IVErrorCode::DiscreteDividendMismatch);
}

// Prefix rule: a shorter-dated query only carries dividends inside its
// life. Build dividend at t=0.5; a query with maturity 0.4 must pass with
// an empty schedule and be rejected if it claims the t=0.5 dividend.
TEST_F(DiscreteDividendIVIntegrationTest, PrefixWindowSemantics) {
    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.4;
    query.rate = RateSpec{0.05};
    query.option_type = OptionType::PUT;
    query.market_price = 4.0;

    query.discrete_dividends = {};  // dividend at 0.5 is outside [0, 0.4]
    auto ok = solver_->solve(query);
    // May fail numerically for other reasons, but must NOT be a
    // schedule mismatch.
    if (!ok.has_value()) {
        EXPECT_NE(ok.error().code, IVErrorCode::DiscreteDividendMismatch);
    }

    query.discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}};
    auto bad = solver_->solve(query);
    ASSERT_FALSE(bad.has_value());
    EXPECT_EQ(bad.error().code, IVErrorCode::DiscreteDividendMismatch);
}
```

Append to `tests/iv_solver_factory_test.cc` (self-contained — do not depend on other fixtures in that file):

```cpp
// Regression: a continuous surface must loudly reject a query carrying a
// discrete dividend schedule (#448 / #440 item 1)
// Bug: the schedule was silently ignored and the query priced dividend-free.
TEST(IVSolverFactoryDividendValidation, ContinuousSurfaceRejectsDiscreteQuery) {
    mango::IVSolverFactoryConfig config{
        .option_type = mango::OptionType::PUT,
        .spot = 100.0,
        .grid = mango::IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.10, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.04, 0.06, 0.08},
        },
        .backend = mango::BSplineBackend{
            .maturity_grid = {0.25, 0.5, 0.75, 1.0},
        },
    };
    auto solver = mango::make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value());

    mango::IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.8;
    query.rate = mango::RateSpec{0.04};
    query.option_type = mango::OptionType::PUT;
    query.market_price = 6.0;
    query.discrete_dividends = {{.calendar_time = 0.5, .amount = 1.5}};

    auto result = solver->solve(query);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::IVErrorCode::DiscreteDividendMismatch);

    // Same query without the schedule still solves (sanity that the
    // rejection is schedule-driven, not incidental).
    query.discrete_dividends.clear();
    auto ok = solver->solve(query);
    EXPECT_TRUE(ok.has_value());
}
```

- [ ] **Step 2: Run the new tests, verify they fail**

Run: `bazel test //tests:discrete_dividend_iv_integration_test //tests:iv_solver_factory_test --test_output=errors --test_filter='*Mismatch*:*Prefix*:*Matching*:*DividendValidation*'`
(Bazel test_filter flag: use `--test_arg=--gtest_filter=...` if `--test_filter` is not honored.)
Expected: compile failure (`IVErrorCode::DiscreteDividendMismatch` undefined). That counts as the failing state.

- [ ] **Step 3: Add the error codes and mapping**

`src/support/error_types.hpp` — append to `ValidationErrorCode` (after `DividendYieldMismatch`, line 44):

```cpp
    DividendYieldMismatch,
    DiscreteDividendMismatch
```

Append to `IVErrorCode` (after `DividendYieldMismatch`, line 125):

```cpp
    DividendYieldMismatch,
    DiscreteDividendMismatch,
```

`src/option/iv_result.hpp` — in `validation_error_to_iv_error`, after the `DividendYieldMismatch` case (line 83):

```cpp
        case ValidationErrorCode::DiscreteDividendMismatch:
            code = IVErrorCode::DiscreteDividendMismatch;
            break;
```

`src/python/mango_bindings.cpp` — after line 477's case add:

```cpp
                case mango::IVErrorCode::DiscreteDividendMismatch: return "Discrete dividend mismatch";
```

and after line 503's `.value(...)` add:

```cpp
        .value("DiscreteDividendMismatch", mango::IVErrorCode::DiscreteDividendMismatch)
```

- [ ] **Step 4: Store the build schedule in `InterpolatedIVSolver`**

`src/option/interpolated_iv_solver.hpp`:

1. Public `create` declaration (~line 83) becomes:

```cpp
    /// Create solver from a PriceTable
    ///
    /// The surface must provide price(), vega(), and bounds accessors.
    ///
    /// @param surface Pre-built price surface
    /// @param config Solver configuration
    /// @param build_dividends Discrete dividend schedule the surface was
    ///        built with. Defaults to known-empty (correct for continuous
    ///        surfaces — the documented direct-create path). Pass the real
    ///        schedule for segmented surfaces, or std::nullopt for
    ///        "unknown" (e.g. deserialized tables; schedule checks are
    ///        then skipped).
    /// @return IV solver or ValidationError
    static std::expected<InterpolatedIVSolver, ValidationError> create(
        Surface surface,
        const InterpolatedIVSolverConfig& config = {},
        std::optional<std::vector<Dividend>> build_dividends =
            std::vector<Dividend>{});
```

2. Private ctor gains the parameter (after `dividend_yield`), member added after `dividend_yield_`:

```cpp
        OptionType option_type,
        double dividend_yield,
        std::optional<std::vector<Dividend>> build_dividends,
        const InterpolatedIVSolverConfig& config)
        : surface_(std::move(surface))
        , m_range_(m_range)
        , tau_range_(tau_range)
        , sigma_range_(sigma_range)
        , r_range_(r_range)
        , config_(config)
        , option_type_(option_type)
        , dividend_yield_(dividend_yield)
        , build_dividends_(std::move(build_dividends))
    {}
```

```cpp
    OptionType option_type_;
    double dividend_yield_;
    /// Discrete schedule the surface was built with. nullopt = unknown
    /// (deserialized segmented tables) — schedule validation is skipped.
    std::optional<std::vector<Dividend>> build_dividends_;
```

3. `create` template implementation (~line 332): add the parameter and pass it through:

```cpp
template <typename Surface>
std::expected<InterpolatedIVSolver<Surface>, ValidationError>
InterpolatedIVSolver<Surface>::create(
    Surface surface,
    const InterpolatedIVSolverConfig& config,
    std::optional<std::vector<Dividend>> build_dividends)
{
    ...unchanged bounds extraction/validation...
    return InterpolatedIVSolver(
        std::move(surface),
        m_range,
        tau_range,
        sigma_range,
        r_range,
        option_type,
        dividend_yield,
        std::move(build_dividends),
        config);
}
```

4. `validate_query` (~line 377): after the dividend-yield check, before `validate_iv_query`:

```cpp
    // Discrete dividend schedule check (#440 item 1). An empty query
    // schedule is always valid: for segmented surfaces the build-time
    // schedule is authoritative. A non-empty schedule must match the
    // build schedule restricted to the query's life, when it is known.
    if (!query.discrete_dividends.empty() && build_dividends_.has_value()) {
        constexpr double kTimeTol = 1e-6;    // years (~30 seconds)
        constexpr double kAmountTol = 1e-6;  // dollars
        auto by_time = [](const Dividend& a, const Dividend& b) {
            return a.calendar_time < b.calendar_time;
        };
        std::vector<Dividend> expected;
        expected.reserve(build_dividends_->size());
        for (const auto& d : *build_dividends_) {
            if (d.calendar_time <= query.maturity + kTimeTol) {
                expected.push_back(d);
            }
        }
        std::sort(expected.begin(), expected.end(), by_time);
        std::vector<Dividend> actual = query.discrete_dividends;
        std::sort(actual.begin(), actual.end(), by_time);
        if (expected.size() != actual.size()) {
            return ValidationError{
                ValidationErrorCode::DiscreteDividendMismatch,
                static_cast<double>(actual.size()), expected.size()};
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::abs(actual[i].calendar_time -
                         expected[i].calendar_time) > kTimeTol ||
                std::abs(actual[i].amount - expected[i].amount) > kAmountTol) {
                return ValidationError{
                    ValidationErrorCode::DiscreteDividendMismatch,
                    actual[i].amount, i};
            }
        }
    }
```

Confirm `<algorithm>` and `<vector>` are included in `interpolated_iv_solver.hpp`; add if missing.

- [ ] **Step 5: Thread the schedule through the factory**

`src/option/price_table_factory.hpp:40` — signature becomes:

```cpp
    /// @param build_dividends Discrete schedule for validate_query.
    ///        nullopt = infer from table type: segmented (MultiKRef)
    ///        tables get "unknown" (checks skipped — schedules are not
    ///        persisted to Parquet), all others get known-empty.
    [[nodiscard]] std::expected<AnyInterpIVSolver, ValidationError>
    make_iv_solver(const InterpolatedIVSolverConfig& config = {},
                   std::optional<std::vector<Dividend>> build_dividends =
                       std::nullopt) const;
```

`src/option/price_table_factory.cpp:790-805` — body becomes:

```cpp
std::expected<AnyInterpIVSolver, ValidationError>
AnyPriceTable::make_iv_solver(
    const InterpolatedIVSolverConfig& config,
    std::optional<std::vector<Dividend>> build_dividends) const {
    return std::visit([&](const auto& table_ptr)
        -> std::expected<AnyInterpIVSolver, ValidationError> {
        using Table = std::remove_cv_t<
            typename std::decay_t<decltype(table_ptr)>::element_type>;
        using SharedSurface = detail::SharedPriceTableSurface<Table>;
        std::optional<std::vector<Dividend>> divs = std::move(build_dividends);
        if (!divs.has_value() &&
            !std::is_same_v<Table, BSplineMultiKRefSurface> &&
            !std::is_same_v<Table, ChebyshevMultiKRefSurface>) {
            divs = std::vector<Dividend>{};  // continuous: known-empty
        }
        auto solver = InterpolatedIVSolver<SharedSurface>::create(
            SharedSurface(table_ptr), config, std::move(divs));
        if (!solver.has_value()) {
            return std::unexpected(solver.error());
        }
        return make_any_interpolated_solver(std::move(*solver));
    }, impl_->table);
}
```

`src/option/price_table_factory.cpp:883-891` — `make_interpolated_iv_solver` becomes:

```cpp
std::expected<AnyInterpIVSolver, ValidationError>
make_interpolated_iv_solver(const IVSolverFactoryConfig& config) {
    auto table = make_price_table(config);
    if (!table.has_value()) {
        return std::unexpected(table.error());
    }
    std::vector<Dividend> build_dividends;
    if (config.discrete_dividends.has_value()) {
        build_dividends = config.discrete_dividends->discrete_dividends;
    }
    return table->make_iv_solver(config.solver_config,
                                 std::move(build_dividends));
}
```

Note `AnyPriceTable::solve_iv` (line ~808) forwards to `make_iv_solver(config)` with no schedule → inference applies; no change needed.

- [ ] **Step 6: Run the new tests, verify they pass**

Run: `bazel test //tests:discrete_dividend_iv_integration_test //tests:iv_solver_factory_test --test_output=errors`
Expected: PASS (all tests in both targets, old and new).

- [ ] **Step 7: Run the neighboring suites for regressions**

Run: `bazel test //tests:interpolated_iv_solver_test //tests:price_table_factory_test //tests:dimensionless_iv_test --test_output=errors` (skip any target that does not exist; check with `bazel query 'tests(//tests:all)' | grep -e interpolated -e factory -e dimensionless`).
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/support/error_types.hpp src/option/iv_result.hpp \
    src/option/interpolated_iv_solver.hpp src/option/price_table_factory.hpp \
    src/option/price_table_factory.cpp src/python/mango_bindings.cpp \
    tests/discrete_dividend_iv_integration_test.cc tests/iv_solver_factory_test.cc
git commit -m "Validate discrete dividend schedules in interpolated IV solver

A query carrying a discrete dividend schedule was silently priced off
whatever surface the solver held (issue #440 item 1) - dividend-free
for continuous surfaces. Store the build-time schedule in
InterpolatedIVSolver and reject non-empty query schedules that
mismatch it. Empty query schedules remain valid against segmented
surfaces (the surface's schedule is authoritative). Deserialized
segmented tables have unknown schedules (not persisted to Parquet),
so their checks are skipped."
```

---

### Task 2: Simple-layer dividend conversion helper + ChainBuilder setter

**Files:**
- Modify: `src/simple/vol_surface.hpp` (declare helper, after `PriceSource`, ~line 88)
- Modify: `src/simple/vol_surface.cpp` (implement helper; add `#include <algorithm>`)
- Modify: `src/simple/chain_builder.hpp:45-47` (add setter next to `dividend_yield`)
- Test: `tests/simple_vol_surface_test.cc`, `tests/simple_chain_builder_test.cc`

**Interfaces:**
- Consumes: `mango::simple::Dividend{Timestamp ex_date, Price amount}` (`src/simple/option_chain.hpp:21-24`), `compute_tau(const Timestamp&, const Timestamp&)` (`src/simple/timestamp.hpp:76`, returns 0.0 on parse failure), `mango::Dividend{double calendar_time, double amount}` (`src/option/option_spec.hpp:158-161`).
- Produces:
  - `std::vector<mango::Dividend> mango::simple::convert_discrete_dividends(const std::vector<Dividend>& dividends, const Timestamp& val_time, double tau_max)` — keeps `0 < t <= tau_max`, sorted by `calendar_time`.
  - `ChainBuilder<Source>& ChainBuilder<Source>::discrete_dividends(std::vector<Dividend> dividends)` — sets `chain_.dividends` to the vector alternative.

- [ ] **Step 1: Write the failing tests**

Append to `tests/simple_vol_surface_test.cc`:

```cpp
// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: compute_vol_surface dropped discrete dividend schedules (#448)
// Bug: only the double alternative of DividendSpec was read; the
//      vector<Dividend> alternative left div_yield = 0 and never populated
//      IVQuery::discrete_dividends. These tests pin the conversion helper.
TEST(DividendConversionTest, ConvertsExDatesToSortedYearOffsets) {
    Timestamp val{"2026-01-01T00:00:00"};
    std::vector<Dividend> divs = {
        {.ex_date = Timestamp{"2026-07-02T00:00:00"}, .amount = Price{1.50}},
        {.ex_date = Timestamp{"2026-04-02T00:00:00"}, .amount = Price{1.25}},
    };
    auto out = convert_discrete_dividends(divs, val, 1.0);
    ASSERT_EQ(out.size(), 2u);
    // Sorted by calendar time even though the input was not.
    EXPECT_NEAR(out[0].calendar_time, 91.0 / 365.0, 0.01);
    EXPECT_DOUBLE_EQ(out[0].amount, 1.25);
    EXPECT_NEAR(out[1].calendar_time, 182.0 / 365.0, 0.01);
    EXPECT_DOUBLE_EQ(out[1].amount, 1.50);
}

TEST(DividendConversionTest, DropsPastAndPostExpiryDividends) {
    Timestamp val{"2026-01-01T00:00:00"};
    std::vector<Dividend> divs = {
        // Already gone ex before valuation: excluded.
        {.ex_date = Timestamp{"2025-12-15T00:00:00"}, .amount = Price{1.00}},
        // Inside the window: kept.
        {.ex_date = Timestamp{"2026-03-01T00:00:00"}, .amount = Price{1.50}},
        // After expiry (tau_max = 0.5): excluded.
        {.ex_date = Timestamp{"2027-06-01T00:00:00"}, .amount = Price{2.00}},
    };
    auto out = convert_discrete_dividends(divs, val, 0.5);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_DOUBLE_EQ(out[0].amount, 1.50);
    EXPECT_GT(out[0].calendar_time, 0.0);
    EXPECT_LE(out[0].calendar_time, 0.5);
}

TEST(DividendConversionTest, UnparseableExDateIsDropped) {
    Timestamp val{"2026-01-01T00:00:00"};
    std::vector<Dividend> divs = {
        {.ex_date = Timestamp{"not-a-date"}, .amount = Price{1.00}},
    };
    auto out = convert_discrete_dividends(divs, val, 1.0);
    EXPECT_TRUE(out.empty());
}
```

Append to `tests/simple_chain_builder_test.cc`:

```cpp
// Regression: ChainBuilder had no way to attach a discrete dividend
// schedule (#448) — the vector alternative of DividendSpec was
// unreachable through the builder API.
TEST(ChainBuilderTest, DiscreteDividendSchedule) {
    auto chain = ChainBuilder<YFinanceSource>{}
        .symbol("XYZ")
        .spot(100.0)
        .discrete_dividends({
            {.ex_date = Timestamp{"2026-04-01"}, .amount = Price{1.50}},
        })
        .build();
    ASSERT_TRUE(chain.dividends.has_value());
    const auto* divs =
        std::get_if<std::vector<Dividend>>(&*chain.dividends);
    ASSERT_NE(divs, nullptr);
    ASSERT_EQ(divs->size(), 1u);
    EXPECT_DOUBLE_EQ((*divs)[0].amount.to_double(), 1.50);
}
```

(Adjust the `ChainBuilder<YFinanceSource>` spelling and includes to match how the existing tests in each file construct builders — `tests/simple_vol_surface_test.cc:10-17` shows the pattern.)

- [ ] **Step 2: Run tests, verify they fail**

Run: `bazel test //tests:simple_vol_surface_test //tests:simple_chain_builder_test --test_output=errors`
Expected: compile failure (`convert_discrete_dividends` / `discrete_dividends` undeclared).

- [ ] **Step 3: Implement**

`src/simple/vol_surface.hpp`, after the `PriceSource` enum (~line 88):

```cpp
/// Convert a discrete dividend schedule to solver units
///
/// Maps each dividend's ex_date to years from val_time and keeps only
/// dividends payable during the option's life: 0 < t <= tau_max.
/// Dividends whose ex_date fails to parse are dropped (compute_tau
/// returns 0.0 for unparseable timestamps, removed by the t > 0 filter).
/// The result is sorted by calendar time.
[[nodiscard]] std::vector<mango::Dividend> convert_discrete_dividends(
    const std::vector<Dividend>& dividends,
    const Timestamp& val_time,
    double tau_max);
```

`src/simple/vol_surface.cpp` (add `#include <algorithm>` next to `<cmath>`):

```cpp
std::vector<mango::Dividend> convert_discrete_dividends(
    const std::vector<Dividend>& dividends,
    const Timestamp& val_time,
    double tau_max)
{
    std::vector<mango::Dividend> out;
    out.reserve(dividends.size());
    for (const auto& div : dividends) {
        double t = compute_tau(val_time, div.ex_date);
        if (t > 0.0 && t <= tau_max) {
            out.push_back({.calendar_time = t,
                           .amount = div.amount.to_double()});
        }
    }
    std::sort(out.begin(), out.end(),
              [](const mango::Dividend& a, const mango::Dividend& b) {
                  return a.calendar_time < b.calendar_time;
              });
    return out;
}
```

`src/simple/chain_builder.hpp`, next to `dividend_yield` (line 45):

```cpp
    ChainBuilder& discrete_dividends(std::vector<Dividend> dividends) {
        chain_.dividends = std::move(dividends);
        return *this;
    }
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `bazel test //tests:simple_vol_surface_test //tests:simple_chain_builder_test --test_output=errors`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simple/vol_surface.hpp src/simple/vol_surface.cpp \
    src/simple/chain_builder.hpp tests/simple_vol_surface_test.cc \
    tests/simple_chain_builder_test.cc
git commit -m "Add discrete dividend conversion helper and builder setter

simple::Dividend carries {ex_date, amount}; the solver needs
{years-from-valuation, dollars}. convert_discrete_dividends performs
the unit conversion, drops dividends outside (0, tau_max], and sorts
by time. ChainBuilder::discrete_dividends makes the vector alternative
of DividendSpec reachable (issue #448)."
```

---

### Task 3: Wire the schedule through compute_vol_surface + end-to-end regression tests

**Files:**
- Modify: `src/simple/vol_surface.cpp:83-137` (`compute_vol_surface`)
- Modify: `tests/simple_vol_surface_test.cc`
- Modify: `tests/BUILD.bazel` (~line 1040, `simple_vol_surface_test` target)

**Interfaces:**
- Consumes: `convert_discrete_dividends` (Task 2), `DiscreteDividendMismatch` rejection behavior (Task 1), `make_interpolated_iv_solver` + `DiscreteDividendConfig` (`src/option/interpolated_iv_solver.hpp:186-190`), `solve_american_option` (`src/option/american_option.hpp`).
- Produces: `compute_vol_surface` populates `IVQuery::discrete_dividends` per expiry slice when the resolved `DividendSpec` holds the vector alternative; `div_yield` stays 0.0 in that case.

- [ ] **Step 1: Update the BUILD target**

In `tests/BUILD.bazel`, change the `simple_vol_surface_test` target (~line 1040) to:

```python
cc_test(
    name = "simple_vol_surface_test",
    size = "medium",
    srcs = ["simple_vol_surface_test.cc"],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
    deps = [
        "//src/simple:vol_surface",
        "//src/simple:chain_builder",
        "//src/option:american_option",
        "//src/option:interpolated_iv_solver",
        "//src/option:price_table_factory",
        "@googletest//:gtest_main",
    ],
)
```

(Keep any existing deps/attrs the target already has that are not listed here; verify the `//src/simple` target names with `bazel query 'deps(//tests:simple_vol_surface_test, 1)'` before editing.)

- [ ] **Step 2: Write the failing end-to-end tests**

Append to `tests/simple_vol_surface_test.cc` (add includes `"mango/option/american_option.hpp"` and `"mango/option/interpolated_iv_solver.hpp"` at the top):

```cpp
namespace {

// Chain with one expiry (~0.75y), a few strikes around spot=100, and one
// discrete dividend at ~0.25y. Market prices are FDM American prices at
// KNOWN_VOL with that dividend, so the IVs a correct pipeline recovers
// are ~KNOWN_VOL.
constexpr double kKnownVol = 0.20;

OptionChain make_discrete_dividend_chain() {
    std::vector<Dividend> divs = {
        {.ex_date = Timestamp{"2026-04-01T00:00:00"}, .amount = Price{1.50}},
    };
    auto chain = ChainBuilder<YFinanceSource>{}
        .symbol("XYZ")
        .spot(100.0)
        .quote_time("2026-01-01T00:00:00")
        .discrete_dividends(divs)
        .build();

    Timestamp val{"2026-01-01T00:00:00"};
    ExpirySlice slice;
    slice.expiry = Timestamp{"2026-10-01T00:00:00"};
    double tau = compute_tau(val, slice.expiry);
    auto solver_divs = convert_discrete_dividends(divs, val, tau);

    for (double strike : {90.0, 95.0, 100.0, 105.0, 110.0}) {
        mango::PricingParams params(
            mango::OptionSpec{.spot = 100.0, .strike = strike,
                .maturity = tau, .rate = 0.05,
                .option_type = mango::OptionType::PUT},
            kKnownVol, solver_divs);
        auto priced = mango::solve_american_option(params);
        if (!priced.has_value()) continue;
        OptionLeg leg;
        leg.type = mango::OptionType::PUT;
        leg.strike = Price{strike};
        double mid = priced->value();
        leg.bid = Price{mid};
        leg.ask = Price{mid};
        slice.options.push_back(leg);
    }
    chain.expiries.push_back(std::move(slice));
    return chain;
}

}  // namespace

// Regression: compute_vol_surface solved the whole surface dividend-free
// when the chain carried a discrete schedule (#448)
// Bug: the vector<Dividend> alternative of DividendSpec was never read;
//      every IVQuery went out with dividend_yield=0 and no schedule.
//      With solver-side validation (#440 item 1) a dividend-free surface
//      must now reject those queries loudly instead of returning IVs
//      biased by the full dividend effect.
TEST(VolSurfaceDividendTest, ContinuousSolverRejectsDiscreteChain) {
    auto chain = make_discrete_dividend_chain();

    mango::IVSolverFactoryConfig config{
        .option_type = mango::OptionType::PUT,
        .spot = 100.0,
        .grid = mango::IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.10, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.04, 0.06, 0.08},
        },
        .backend = mango::BSplineBackend{
            .maturity_grid = {0.25, 0.5, 0.75, 1.0},
        },
    };
    auto solver = mango::make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value());

    MarketContext ctx;
    ctx.rate = 0.05;
    ctx.valuation_time = Timestamp{"2026-01-01T00:00:00"};

    auto surface = compute_vol_surface(chain, ctx, &*solver);
    ASSERT_TRUE(surface.has_value());
    ASSERT_FALSE(surface->smiles.empty());
    for (const auto& smile : surface->smiles) {
        for (const auto& pt : smile.points) {
            EXPECT_FALSE(pt.iv_mid.has_value())
                << "dividend-free surface must not produce an IV for a "
                   "discrete-dividend chain (strike "
                << pt.strike.to_double() << ")";
        }
    }
}

// Happy path: a segmented solver built with the SAME schedule accepts the
// queries and recovers the known vol.
TEST(VolSurfaceDividendTest, SegmentedSolverRecoversKnownVol) {
    auto chain = make_discrete_dividend_chain();
    Timestamp val{"2026-01-01T00:00:00"};
    const auto& divs = std::get<std::vector<Dividend>>(*chain.dividends);
    auto solver_divs = convert_discrete_dividends(divs, val, 1.0);

    mango::IVSolverFactoryConfig config{
        .option_type = mango::OptionType::PUT,
        .spot = 100.0,
        .grid = mango::IVGrid{
            .moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
            .vol = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        .backend = mango::BSplineBackend{},
        .discrete_dividends = mango::DiscreteDividendConfig{
            .maturity = 1.0,
            .discrete_dividends = solver_divs,
            .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
        },
    };
    auto solver = mango::make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value());

    MarketContext ctx;
    ctx.rate = 0.05;
    ctx.valuation_time = val;

    auto surface = compute_vol_surface(chain, ctx, &*solver);
    ASSERT_TRUE(surface.has_value());
    ASSERT_FALSE(surface->smiles.empty());
    size_t checked = 0;
    for (const auto& smile : surface->smiles) {
        for (const auto& pt : smile.points) {
            ASSERT_TRUE(pt.iv_mid.has_value())
                << "matching schedule must solve (strike "
                << pt.strike.to_double() << ")";
            EXPECT_NEAR(*pt.iv_mid, kKnownVol, 0.02)
                << "strike " << pt.strike.to_double();
            ++checked;
        }
    }
    EXPECT_GE(checked, 3u);
}
```

- [ ] **Step 3: Run tests, verify they fail for the right reason**

Run: `bazel test //tests:simple_vol_surface_test --test_output=all --test_arg=--gtest_filter='VolSurfaceDividendTest.*'`
Expected: `ContinuousSolverRejectsDiscreteChain` FAILS — points DO get IVs (the schedule is dropped, queries validate fine against the dividend-free surface, IVs come back biased). `SegmentedSolverRecoversKnownVol` may already pass (the surface's baked-in schedule dominates); that is expected — it guards against over-strict validation after the fix.

- [ ] **Step 4: Implement the wiring**

`src/simple/vol_surface.cpp`, replace lines 83-89 with:

```cpp
    // Get dividends: continuous yield (indices) or discrete schedule
    // (single stocks). The two are mutually exclusive in DividendSpec.
    double div_yield = 0.0;
    const std::vector<Dividend>* discrete = nullptr;
    auto div_spec = ctx.dividends.value_or(
        chain.dividends.value_or(DividendSpec{0.0}));
    if (const double* yield = std::get_if<double>(&div_spec)) {
        div_yield = *yield;
    } else {
        discrete = &std::get<std::vector<Dividend>>(div_spec);
    }
```

Inside the expiry loop, after the `if (smile.tau <= 0) continue;` (line 100):

```cpp
        // Dividends payable during this expiry's life, in solver units.
        std::vector<mango::Dividend> slice_dividends;
        if (discrete != nullptr) {
            slice_dividends =
                convert_discrete_dividends(*discrete, val_time, smile.tau);
        }
```

In the query construction (after `query.dividend_yield = div_yield;` line 124):

```cpp
            query.discrete_dividends = slice_dividends;
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `bazel test //tests:simple_vol_surface_test --test_output=errors`
Expected: PASS (all tests in the file, including the pre-existing ones).

- [ ] **Step 6: Commit**

```bash
git add src/simple/vol_surface.cpp tests/simple_vol_surface_test.cc tests/BUILD.bazel
git commit -m "Wire discrete dividend schedules through compute_vol_surface

A chain carrying the vector<Dividend> alternative of DividendSpec was
silently solved dividend-free: div_yield stayed 0.0 and
IVQuery::discrete_dividends was never populated, biasing every point
on the surface by the full dividend effect (issue #448). Convert the
schedule per expiry slice and attach it to each query. Combined with
solver-side schedule validation, a mismatched surface now fails loudly
instead of returning confidently wrong IVs."
```

---

### Task 4: Full verification (CI parity)

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `bazel test //...`
Expected: all tests pass (130+ targets; ~2-3 min warm). Any failure: fix before proceeding, or report if pre-existing on main.

- [ ] **Step 2: Benchmarks + Python bindings compile**

Run: `bazel build //benchmarks/... //src/python:mango_option`
Expected: build success, no warnings in project code (warnings are errors per .bazelrc).

- [ ] **Step 3: Rust binding (CLAUDE.md quick reference)**

Run: `bazel test //crates/mango-option:integration_test`
Expected: PASS (Rust does not enumerate IVErrorCode; this is a regression check only).

- [ ] **Step 4: Commit any stragglers, verify clean tree**

Run: `git status --short`
Expected: clean (plan doc already committed).
