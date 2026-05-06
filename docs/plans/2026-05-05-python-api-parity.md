# Python API Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Python binding to workflow-level parity with stable C++ pricing, price-table, persistence, and interpolated-IV capabilities, with the 4D B-spline price-table path treated as a first-class acceptance path.

**Architecture:** Add a C++ `AnyPriceTable` type-erased artifact and a `make_price_table(config)` factory, then bind that artifact as the central Python object for price, Greeks, persistence, and IV. Keep `make_interpolated_iv_solver(config)` as a backward-compatible convenience wrapper over the same table factory. The continuous-dividend `BSplineBackend` path must actively prove it builds and round-trips a `bspline_4d` surface over log-moneyness, maturity, volatility, and rate. Python tests verify binding reachability, automatic conversions, persistence plumbing, and typed exceptions; C++ tests remain responsible for numerical correctness.

**Tech Stack:** C++23, pybind11, Bazel, GoogleTest, Python 3.11 stdlib only for binding tests, Arrow/Parquet C++ for persisted price tables.

---

## File Structure

- Create `src/option/price_table_factory.hpp`: public type-erased price-table artifact, factory, load helper, and persistence options.
- Create `src/option/price_table_factory.cpp`: implementation that reuses/refactors the current interpolated-IV factory build paths to produce reusable price tables.
- Modify `src/option/interpolated_iv_solver.hpp`: keep `IVSolverFactoryConfig`; expose convenience construction through `AnyPriceTable` without changing existing solver API.
- Modify `src/option/interpolated_iv_solver.cpp`: remove duplicate build ownership by delegating `make_interpolated_iv_solver(config)` to `make_price_table(config).make_iv_solver(config.solver_config)`.
- Modify `src/option/BUILD.bazel`: add `price_table_factory` and update `interpolated_iv_solver` deps.
- Modify `src/python/mango_bindings.cpp`: bind `PriceTable`, `make_price_table`, persistence, typed exceptions, conversion helpers, and remove numpy usage.
- Modify `src/python/BUILD.bazel`: add `//src/option:price_table_factory` and Parquet deps.
- Modify `tests/BUILD.bazel`: add C++ factory test, split Python binding tests, and remove numpy dependency.
- Create `tests/price_table_factory_test.cc`: C++ wrapper/factory/persistence reachability tests, including an active 4D B-spline off-grid interpolation path.
- Replace or split `tests/test_bindings.py`: binding reachability and conversion tests using only stdlib, including a Python `BSplineBackend` path that returns `bspline_4d`.
- Modify `BUILD.bazel`: remove wheel `requires = ["numpy"]`.
- Modify `third_party/requirements.txt`: remove `numpy` if no remaining Python target needs it.
- Modify `docs/PYTHON_GUIDE.md`: document `PriceTable`, factory, persistence, conversions, exceptions, and compatibility path.
- Modify `README.md`: update short Python example if it mentions old or stale APIs.

---

### Task 1: Add C++ Failing Tests For Type-Erased Price Tables And 4D B-Spline

**Files:**
- Create: `tests/price_table_factory_test.cc`
- Modify: `tests/BUILD.bazel`

- [ ] **Step 1: Add the failing C++ test file**

Create `tests/price_table_factory_test.cc`:

```cpp
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <vector>

#include "mango/option/price_table_factory.hpp"
#include "mango/option/table/serialization/price_table_data.hpp"

namespace mango {
namespace {

IVSolverFactoryConfig bspline_4d_config() {
    IVSolverFactoryConfig config;
    config.option_type = OptionType::PUT;
    config.spot = 100.0;
    config.dividend_yield = 0.02;
    config.grid.moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};
    config.grid.vol = {0.10, 0.20, 0.30, 0.40};
    config.grid.rate = {0.01, 0.03, 0.05, 0.07};
    BSplineBackend backend;
    backend.maturity_grid = {0.1, 0.25, 0.5, 1.0};
    config.backend = backend;
    return config;
}

PricingParams off_grid_pricing_params() {
    PricingParams p;
    p.spot = 100.0;
    p.strike = 97.0;      // S/K is between the 1.0 and 1.1 moneyness knots
    p.maturity = 0.37;    // between maturity knots
    p.volatility = 0.23;  // between volatility knots
    p.rate = 0.037;       // between rate knots
    p.dividend_yield = 0.02;
    p.option_type = OptionType::PUT;
    return p;
}

TEST(PriceTableFactoryTest, BuildsContinuous4DBSplineAndEvaluatesOffGridPoint) {
    auto table_result = make_price_table(bspline_4d_config());
    ASSERT_TRUE(table_result.has_value()) << "make_price_table failed";

    auto table = std::move(*table_result);
    auto params = off_grid_pricing_params();

    EXPECT_EQ(table.surface_type(), surface_types::kBSpline4D);
    EXPECT_EQ(table.option_type(), OptionType::PUT);
    EXPECT_NEAR(table.dividend_yield(), 0.02, 1e-15);
    EXPECT_GT(table.price(params), 0.0);
    EXPECT_GT(table.vega(params), 0.0);
    EXPECT_TRUE(table.delta(params).has_value());
    EXPECT_TRUE(table.gamma(params).has_value());
    EXPECT_TRUE(table.theta(params).has_value());
    EXPECT_TRUE(table.rho(params).has_value());

    IVQuery query;
    query.spot = params.spot;
    query.strike = params.strike;
    query.maturity = params.maturity;
    query.rate = params.rate;
    query.dividend_yield = params.dividend_yield;
    query.option_type = params.option_type;
    query.market_price = table.price(params);

    auto iv_direct = table.solve_iv(query);
    ASSERT_TRUE(iv_direct.has_value()) << "table.solve_iv failed";
    EXPECT_NEAR(iv_direct->implied_vol, params.volatility, 0.08);

    auto solver = table.make_iv_solver();
    ASSERT_TRUE(solver.has_value()) << "make_iv_solver failed";
    auto iv_via_solver = solver->solve(query);
    ASSERT_TRUE(iv_via_solver.has_value()) << "solver.solve failed";
    EXPECT_NEAR(iv_via_solver->implied_vol, params.volatility, 0.08);
}

TEST(PriceTableFactoryTest, ParquetRoundTripPreserves4DBSplineSurface) {
    auto table_result = make_price_table(bspline_4d_config());
    ASSERT_TRUE(table_result.has_value());
    auto table = std::move(*table_result);
    auto params = off_grid_pricing_params();
    const double price_before = table.price(params);

    auto path = std::filesystem::temp_directory_path() / "mango_any_price_table_test.parquet";
    auto save_result = table.save(path);
    ASSERT_TRUE(save_result.has_value()) << "save failed";

    auto loaded_result = load_price_table(path);
    std::filesystem::remove(path);
    ASSERT_TRUE(loaded_result.has_value()) << "load failed";
    auto loaded = std::move(*loaded_result);

    EXPECT_EQ(table.surface_type(), surface_types::kBSpline4D);
    EXPECT_EQ(loaded.surface_type(), surface_types::kBSpline4D);
    EXPECT_NEAR(loaded.price(params), price_before, 1e-10);
    EXPECT_TRUE(loaded.delta(params).has_value());
    EXPECT_TRUE(loaded.make_iv_solver().has_value());
}

TEST(PriceTableFactoryTest, InterpolatedIVSolverConvenienceStillWorks) {
    auto solver_result = make_interpolated_iv_solver(bspline_4d_config());
    ASSERT_TRUE(solver_result.has_value()) << "legacy convenience factory failed";
}

}  // namespace
}  // namespace mango
```

- [ ] **Step 2: Add the Bazel target**

Add near the other price-table tests in `tests/BUILD.bazel`:

```python
cc_test(
    name = "price_table_factory_test",
    size = "medium",
    srcs = ["price_table_factory_test.cc"],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
    deps = [
        "//src/option:price_table_factory",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

- [ ] **Step 3: Run the test and verify it fails**

Run:

```bash
bazel test //tests:price_table_factory_test
```

Expected: FAIL during analysis or compile because `//src/option:price_table_factory` and `mango/option/price_table_factory.hpp` do not exist.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/price_table_factory_test.cc tests/BUILD.bazel
git commit -m "test: specify type-erased price table factory"
```

---

### Task 2: Implement C++ `AnyPriceTable` And Factory

**Files:**
- Create: `src/option/price_table_factory.hpp`
- Create: `src/option/price_table_factory.cpp`
- Modify: `src/option/interpolated_iv_solver.cpp`
- Modify: `src/option/BUILD.bazel`

- [ ] **Step 1: Add the public header**

Create `src/option/price_table_factory.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/serialization/price_table_data.hpp"
#include "mango/support/error_types.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <string>

namespace mango {

enum class PriceTableCompression {
    NONE,
    SNAPPY,
    ZSTD,
};

class AnyPriceTable {
public:
    struct Impl;

    explicit AnyPriceTable(std::unique_ptr<Impl> impl);
    AnyPriceTable(AnyPriceTable&&) noexcept;
    AnyPriceTable& operator=(AnyPriceTable&&) noexcept;
    ~AnyPriceTable();

    [[nodiscard]] std::string surface_type() const;
    [[nodiscard]] OptionType option_type() const noexcept;
    [[nodiscard]] double dividend_yield() const noexcept;

    [[nodiscard]] double price(const PricingParams& params) const;
    [[nodiscard]] double vega(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> delta(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> gamma(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> theta(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> rho(const PricingParams& params) const;

    [[nodiscard]] std::expected<AnyInterpIVSolver, ValidationError>
    make_iv_solver(const InterpolatedIVSolverConfig& config = {}) const;

    [[nodiscard]] std::expected<IVSuccess, IVError>
    solve_iv(const IVQuery& query,
             const InterpolatedIVSolverConfig& config = {}) const;

    [[nodiscard]] PriceTableData to_data() const;

    [[nodiscard]] std::expected<void, PriceTableError>
    save(const std::filesystem::path& path,
         PriceTableCompression compression = PriceTableCompression::ZSTD) const;

private:
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::expected<AnyPriceTable, ValidationError>
make_price_table(const IVSolverFactoryConfig& config);

[[nodiscard]] std::expected<AnyPriceTable, PriceTableError>
load_price_table(const std::filesystem::path& path);

}  // namespace mango
```

- [ ] **Step 2: Add implementation skeleton and variant dispatch**

Create `src/option/price_table_factory.cpp` with the following initial content:

```cpp
// SPDX-License-Identifier: MIT

#include "mango/option/price_table_factory.hpp"

#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_3d_accessor.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/parquet/parquet_io.hpp"
#include "mango/option/table/serialization/from_data.hpp"
#include "mango/option/table/serialization/to_data.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"
#include "mango/math/bspline/bspline_basis.hpp"
#include "mango/math/bspline/bspline_nd_separable.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <variant>

namespace mango {
namespace {

using PriceTableVariant = std::variant<
    BSplinePriceTable,
    BSplineMultiKRefSurface,
    ChebyshevSurface,
    ChebyshevMultiKRefSurface,
    BSpline3DPriceTable,
    Chebyshev3DPriceTable>;

template <typename T>
AnyPriceTable make_any_price_table(T table) {
    return AnyPriceTable(std::make_unique<AnyPriceTable::Impl>(std::move(table)));
}

std::expected<std::vector<double>, ValidationError>
to_log_moneyness(const std::vector<double>& moneyness) {
    std::vector<double> log_m;
    log_m.reserve(moneyness.size());
    for (double m : moneyness) {
        if (m <= 0.0 || !std::isfinite(m)) {
            return std::unexpected(ValidationError{ValidationErrorCode::InvalidBounds, m});
        }
        log_m.push_back(std::log(m));
    }
    return log_m;
}

struct GridBounds {
    double m_min{}, m_max{};
    double sigma_min{}, sigma_max{};
    double rate_min{}, rate_max{};
};

GridBounds extract_bounds(const IVGrid& grid) {
    if (grid.moneyness.empty() || grid.vol.empty() || grid.rate.empty()) return {};
    auto minmax_m = std::minmax_element(grid.moneyness.begin(), grid.moneyness.end());
    auto minmax_v = std::minmax_element(grid.vol.begin(), grid.vol.end());
    auto minmax_r = std::minmax_element(grid.rate.begin(), grid.rate.end());
    if (*minmax_m.first <= 0.0) return {};
    return {
        .m_min = std::log(*minmax_m.first), .m_max = std::log(*minmax_m.second),
        .sigma_min = *minmax_v.first, .sigma_max = *minmax_v.second,
        .rate_min = *minmax_r.first, .rate_max = *minmax_r.second,
    };
}

SurfaceBounds bounds_from_grid(const GridBounds& b, double maturity) {
    return SurfaceBounds{
        .m_min = b.m_min, .m_max = b.m_max,
        .tau_min = 0.0, .tau_max = maturity,
        .sigma_min = b.sigma_min, .sigma_max = b.sigma_max,
        .rate_min = b.rate_min, .rate_max = b.rate_max,
    };
}

ParquetCompression to_parquet_compression(PriceTableCompression c) {
    switch (c) {
        case PriceTableCompression::NONE: return ParquetCompression::NONE;
        case PriceTableCompression::SNAPPY: return ParquetCompression::SNAPPY;
        case PriceTableCompression::ZSTD: return ParquetCompression::ZSTD;
    }
    return ParquetCompression::ZSTD;
}

}  // namespace

struct AnyPriceTable::Impl {
    PriceTableVariant table;

    template <typename T>
    explicit Impl(T t) : table(std::move(t)) {}
};

AnyPriceTable::AnyPriceTable(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
AnyPriceTable::AnyPriceTable(AnyPriceTable&&) noexcept = default;
AnyPriceTable& AnyPriceTable::operator=(AnyPriceTable&&) noexcept = default;
AnyPriceTable::~AnyPriceTable() = default;

std::string AnyPriceTable::surface_type() const {
    return std::visit([](const auto& table) {
        return std::string(surface_type_string<typename std::decay_t<decltype(table)>::inner_type>());
    }, impl_->table);
}

OptionType AnyPriceTable::option_type() const noexcept {
    return std::visit([](const auto& table) { return table.option_type(); }, impl_->table);
}

double AnyPriceTable::dividend_yield() const noexcept {
    return std::visit([](const auto& table) { return table.dividend_yield(); }, impl_->table);
}

double AnyPriceTable::price(const PricingParams& params) const {
    return std::visit([&](const auto& table) {
        return table.price(params.spot, params.strike, params.maturity,
                           params.volatility, get_zero_rate(params.rate, params.maturity));
    }, impl_->table);
}

double AnyPriceTable::vega(const PricingParams& params) const {
    return std::visit([&](const auto& table) {
        return table.vega(params.spot, params.strike, params.maturity,
                          params.volatility, get_zero_rate(params.rate, params.maturity));
    }, impl_->table);
}

std::expected<double, GreekError> AnyPriceTable::delta(const PricingParams& params) const {
    return std::visit([&](const auto& table) { return table.delta(params); }, impl_->table);
}

std::expected<double, GreekError> AnyPriceTable::gamma(const PricingParams& params) const {
    return std::visit([&](const auto& table) { return table.gamma(params); }, impl_->table);
}

std::expected<double, GreekError> AnyPriceTable::theta(const PricingParams& params) const {
    return std::visit([&](const auto& table) { return table.theta(params); }, impl_->table);
}

std::expected<double, GreekError> AnyPriceTable::rho(const PricingParams& params) const {
    return std::visit([&](const auto& table) { return table.rho(params); }, impl_->table);
}

std::expected<AnyInterpIVSolver, ValidationError>
AnyPriceTable::make_iv_solver(const InterpolatedIVSolverConfig& config) const {
    return std::visit([&](const auto& table)
        -> std::expected<AnyInterpIVSolver, ValidationError> {
        using Surface = std::decay_t<decltype(table)>;
        auto solver = InterpolatedIVSolver<Surface>::create(table, config);
        if (!solver.has_value()) return std::unexpected(solver.error());
        return AnyInterpIVSolver(
            std::make_unique<AnyInterpIVSolver::Impl>(std::move(*solver)));
    }, impl_->table);
}

std::expected<IVSuccess, IVError>
AnyPriceTable::solve_iv(const IVQuery& query,
                        const InterpolatedIVSolverConfig& config) const {
    auto solver = make_iv_solver(config);
    if (!solver.has_value()) {
        return std::unexpected(convert_to_iv_error(solver.error()));
    }
    return solver->solve(query);
}

PriceTableData AnyPriceTable::to_data() const {
    return std::visit([](const auto& table) { return mango::to_data(table); }, impl_->table);
}

std::expected<void, PriceTableError>
AnyPriceTable::save(const std::filesystem::path& path,
                    PriceTableCompression compression) const {
    return write_parquet(to_data(), path,
        ParquetWriteOptions{.compression = to_parquet_compression(compression)});
}

}  // namespace mango
```

This skeleton intentionally will not compile yet because `PriceTable<Inner>` does not expose `inner_type` and `AnyInterpIVSolver::Impl` is private. The next steps fix those explicitly.

- [ ] **Step 3: Expose `inner_type` on `PriceTable`**

Modify `src/option/table/price_table.hpp` inside `template <typename Inner> class PriceTable` public section:

```cpp
public:
    using inner_type = Inner;
```

Place it before the constructor.

- [ ] **Step 4: Add a public helper to wrap typed interpolated solvers**

Modify `src/option/interpolated_iv_solver.hpp` after `AnyInterpIVSolver`:

```cpp
template <typename Surface>
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    InterpolatedIVSolver<Surface> solver);
```

Modify `src/option/interpolated_iv_solver.cpp` by replacing the existing anonymous `make_any_solver` helper:

```cpp
template <typename Surface>
AnyInterpIVSolver make_any_interpolated_solver(
    InterpolatedIVSolver<Surface> solver) {
    return AnyInterpIVSolver(
        std::make_unique<AnyInterpIVSolver::Impl>(std::move(solver)));
}

template AnyInterpIVSolver make_any_interpolated_solver<BSplinePriceTable>(
    InterpolatedIVSolver<BSplinePriceTable>);
template AnyInterpIVSolver make_any_interpolated_solver<BSplineMultiKRefSurface>(
    InterpolatedIVSolver<BSplineMultiKRefSurface>);
template AnyInterpIVSolver make_any_interpolated_solver<ChebyshevSurface>(
    InterpolatedIVSolver<ChebyshevSurface>);
template AnyInterpIVSolver make_any_interpolated_solver<ChebyshevMultiKRefSurface>(
    InterpolatedIVSolver<ChebyshevMultiKRefSurface>);
template AnyInterpIVSolver make_any_interpolated_solver<BSpline3DPriceTable>(
    InterpolatedIVSolver<BSpline3DPriceTable>);
template AnyInterpIVSolver make_any_interpolated_solver<Chebyshev3DPriceTable>(
    InterpolatedIVSolver<Chebyshev3DPriceTable>);
```

Then replace every `return make_any_solver(std::move(*solver));` in `src/option/interpolated_iv_solver.cpp` with:

```cpp
return make_any_interpolated_solver(std::move(*solver));
```

Also update `AnyPriceTable::make_iv_solver()` in `src/option/price_table_factory.cpp` to call:

```cpp
return make_any_interpolated_solver(std::move(*solver));
```

- [ ] **Step 5: Move/refactor factory build functions to return `AnyPriceTable`**

In `src/option/price_table_factory.cpp`, implement these private functions by moving the corresponding static build code from `src/option/interpolated_iv_solver.cpp` and returning table wrappers instead of solvers:

```cpp
static std::expected<AnyPriceTable, ValidationError>
build_bspline_continuous_table(const IVSolverFactoryConfig& config,
                               const BSplineBackend& backend);

static std::expected<AnyPriceTable, ValidationError>
build_bspline_segmented_table(const IVSolverFactoryConfig& config,
                              const DiscreteDividendConfig& divs);

static std::expected<AnyPriceTable, ValidationError>
build_chebyshev_continuous_table(const IVSolverFactoryConfig& config,
                                 const ChebyshevBackend& backend);

static std::expected<AnyPriceTable, ValidationError>
build_chebyshev_segmented_table(const IVSolverFactoryConfig& config,
                                const DiscreteDividendConfig& divs);

static std::expected<AnyPriceTable, ValidationError>
build_dimensionless_table(const IVSolverFactoryConfig& config,
                          const DimensionlessBackend& backend);
```

For the B-spline continuous return path, use this exact pattern:

```cpp
auto wrapper = make_bspline_surface(table_result->spline,
                                    config.spot,
                                    config.dividend_yield,
                                    config.option_type);
if (!wrapper.has_value()) {
    return std::unexpected(ValidationError{ValidationErrorCode::InvalidGridSize, 0.0});
}
return make_any_price_table(std::move(*wrapper));
```

For segmented B-spline, after `build_multi_kref_surface(...)` succeeds:

```cpp
auto table = BSplineMultiKRefSurface(
    std::move(*surface),
    bounds_from_grid(extract_bounds(config.grid), divs.maturity),
    config.option_type,
    config.dividend_yield);
return make_any_price_table(std::move(table));
```

For Chebyshev continuous:

```cpp
return make_any_price_table(std::move(result->surface));
```

For Chebyshev segmented:

```cpp
return make_any_price_table(std::move(*surface_result));
```

For dimensionless B-spline and Chebyshev, return the `BSpline3DPriceTable` or `Chebyshev3DPriceTable` created in the current `build_dimensionless_*` helpers instead of immediately creating an IV solver.

- [ ] **Step 6: Implement public `make_price_table` and `load_price_table`**

Add to the bottom of `src/option/price_table_factory.cpp` before the closing namespace:

```cpp
std::expected<AnyPriceTable, ValidationError>
make_price_table(const IVSolverFactoryConfig& config) {
    return std::visit([&](const auto& backend)
        -> std::expected<AnyPriceTable, ValidationError> {
        using B = std::decay_t<decltype(backend)>;
        if constexpr (std::is_same_v<B, BSplineBackend>) {
            if (config.discrete_dividends.has_value())
                return build_bspline_segmented_table(config, *config.discrete_dividends);
            return build_bspline_continuous_table(config, backend);
        } else if constexpr (std::is_same_v<B, ChebyshevBackend>) {
            if (config.discrete_dividends.has_value())
                return build_chebyshev_segmented_table(config, *config.discrete_dividends);
            return build_chebyshev_continuous_table(config, backend);
        } else {
            return build_dimensionless_table(config, backend);
        }
    }, config.backend);
}

std::expected<AnyPriceTable, PriceTableError>
load_price_table(const std::filesystem::path& path) {
    auto data = read_parquet(path);
    if (!data.has_value()) return std::unexpected(data.error());

    if (data->surface_type == surface_types::kBSpline4D) {
        auto table = from_data<BSplineLeaf>(*data);
        if (!table.has_value()) return std::unexpected(table.error());
        return make_any_price_table(std::move(*table));
    }
    if (data->surface_type == surface_types::kBSpline4DSegmented) {
        auto table = from_data<BSplineMultiKRefInner>(*data);
        if (!table.has_value()) return std::unexpected(table.error());
        return make_any_price_table(std::move(*table));
    }
    if (data->surface_type == surface_types::kChebyshev4D ||
        data->surface_type == surface_types::kChebyshev4DRaw) {
        auto table = from_data<ChebyshevLeaf>(*data);
        if (!table.has_value()) return std::unexpected(table.error());
        return make_any_price_table(std::move(*table));
    }
    if (data->surface_type == surface_types::kChebyshev4DSegmented) {
        auto table = from_data<ChebyshevMultiKRefInner>(*data);
        if (!table.has_value()) return std::unexpected(table.error());
        return make_any_price_table(std::move(*table));
    }
    if (data->surface_type == surface_types::kBSpline3D) {
        auto table = from_data<BSpline3DLeaf>(*data);
        if (!table.has_value()) return std::unexpected(table.error());
        return make_any_price_table(std::move(*table));
    }
    if (data->surface_type == surface_types::kChebyshev3D ||
        data->surface_type == surface_types::kChebyshev3DRaw) {
        auto table = from_data<Chebyshev3DLeaf>(*data);
        if (!table.has_value()) return std::unexpected(table.error());
        return make_any_price_table(std::move(*table));
    }

    return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
}
```

- [ ] **Step 7: Delegate existing solver factory to price-table factory**

Modify `src/option/interpolated_iv_solver.cpp` public `make_interpolated_iv_solver` implementation:

```cpp
std::expected<AnyInterpIVSolver, ValidationError>
make_interpolated_iv_solver(const IVSolverFactoryConfig& config) {
    auto table = make_price_table(config);
    if (!table.has_value()) return std::unexpected(table.error());
    return table->make_iv_solver(config.solver_config);
}
```

Add `#include "mango/option/price_table_factory.hpp"` to `src/option/interpolated_iv_solver.cpp`.

- [ ] **Step 8: Add Bazel library target**

Add to `src/option/BUILD.bazel` after `interpolated_iv_solver`:

```python
cc_library(
    name = "price_table_factory",
    srcs = ["price_table_factory.cpp"],
    hdrs = ["price_table_factory.hpp"],
    deps = [
        ":interpolated_iv_solver",
        ":option_grid",
        ":option_spec",
        "//src/option/table:price_table",
        "//src/option/table:adaptive_grid_types",
        "//src/option/table:analytical_eep",
        "//src/option/table:dimensionless_transform_3d",
        "//src/option/table/bspline:bspline_3d_surface",
        "//src/option/table/bspline:bspline_adaptive",
        "//src/option/table/bspline:bspline_builder",
        "//src/option/table/bspline:bspline_segmented_builder",
        "//src/option/table/bspline:bspline_surface",
        "//src/option/table/bspline:bspline_tensor_accessor",
        "//src/option/table/chebyshev:chebyshev_3d_surface",
        "//src/option/table/chebyshev:chebyshev_adaptive",
        "//src/option/table/chebyshev:chebyshev_surface",
        "//src/option/table/chebyshev:chebyshev_table_builder",
        "//src/option/table/dimensionless:dimensionless_3d_accessor",
        "//src/option/table/dimensionless:dimensionless_builder",
        "//src/option/table/parquet:parquet_io",
        "//src/option/table/serialization:from_data",
        "//src/option/table/serialization:to_data",
        "//src/math/bspline:bspline_basis",
        "//src/math/bspline:bspline_nd_separable",
        "//src/math/chebyshev:chebyshev_nodes",
        "//src/support:error_types",
    ],
    copts = [
        "-Wall",
        "-Wextra",
        "-O3",
        "-march=native",
        "-fopenmp",
    ],
    linkopts = ["-fopenmp"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option",
    include_prefix = "mango/option",
)
```

Add `":price_table_factory"` to the root `//:mango_option` deps in `BUILD.bazel`.

- [ ] **Step 9: Run C++ tests**

Run:

```bash
bazel test //tests:price_table_factory_test //tests:price_table_data_test //tests:parquet_io_test //tests:interpolated_iv_solver_test
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/option/price_table_factory.hpp src/option/price_table_factory.cpp src/option/interpolated_iv_solver.hpp src/option/interpolated_iv_solver.cpp src/option/table/price_table.hpp src/option/BUILD.bazel BUILD.bazel
git commit -m "feat: add reusable price table factory"
```

---

### Task 3: Replace Python Smoke Script With Binding Reachability Tests

**Files:**
- Modify: `tests/test_bindings.py`
- Modify: `tests/BUILD.bazel`

- [ ] **Step 1: Replace numpy-based smoke style with stdlib tests**

Replace `tests/test_bindings.py` with:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import math
import pathlib
import tempfile

import mango_option as mo


def make_pricing_params(rate=0.05):
    p = mo.PricingParams()
    p.spot = 100.0
    p.strike = 100.0
    p.maturity = 0.5
    p.volatility = 0.20
    p.rate = rate
    p.dividend_yield = 0.02
    p.option_type = mo.OptionType.PUT
    return p


def make_bspline_4d_off_grid_params():
    p = make_pricing_params()
    p.strike = 97.0
    p.maturity = 0.37
    p.volatility = 0.23
    p.rate = 0.037
    return p


def make_iv_query(price, params=None, rate=0.05):
    q = mo.IVQuery()
    if params is None:
        q.spot = 100.0
        q.strike = 100.0
        q.maturity = 0.5
        q.rate = rate
        q.dividend_yield = 0.02
        q.option_type = mo.OptionType.PUT
    else:
        q.spot = params.spot
        q.strike = params.strike
        q.maturity = params.maturity
        q.rate = params.rate
        q.dividend_yield = params.dividend_yield
        q.option_type = params.option_type
    q.market_price = price
    return q


def make_price_table_config():
    config = mo.PriceTableConfig()
    config.option_type = mo.OptionType.PUT
    config.spot = 100.0
    config.dividend_yield = 0.02
    config.grid.moneyness = [0.8, 0.9, 1.0, 1.1, 1.2]
    config.grid.vol = [0.10, 0.20, 0.30, 0.40]
    config.grid.rate = [0.01, 0.03, 0.05, 0.07]
    backend = mo.BSplineBackend()
    backend.maturity_grid = [0.1, 0.25, 0.5, 1.0]
    config.backend = backend
    return config


def assert_finite_number(value):
    assert isinstance(value, float)
    assert math.isfinite(value)


def test_rate_spec_conversions():
    p = make_pricing_params(rate=1)
    assert p.rate == 1.0
    p.rate = 0.05
    assert p.rate == 0.05
    curve = mo.YieldCurve.flat(0.04)
    p.rate = curve
    assert isinstance(p.rate, mo.YieldCurve)

    try:
        p.rate = "0.05"
        raise AssertionError("string rate should fail")
    except mo.TypeConversionError:
        pass


def test_sequence_conversions_for_vectors_and_axes():
    config = make_price_table_config()
    config.grid.moneyness = (0.8, 0.9, 1.0, 1.1, 1.2)
    assert list(config.grid.moneyness) == [0.8, 0.9, 1.0, 1.1, 1.2]

    axes = mo.PriceTableAxes()
    axes.grids = [
        [0.8, 1.0, 1.2, 1.4],
        (0.1, 0.5, 1.0, 1.5),
        [0.1, 0.2, 0.3, 0.4],
        (0.01, 0.03, 0.05, 0.07),
    ]
    assert axes.shape() == (4, 4, 4, 4)
    assert axes.total_points() == 256
    assert list(axes.grids[0]) == [0.8, 1.0, 1.2, 1.4]
    axes.names = ("moneyness", "maturity", "vol", "rate")
    assert list(axes.names) == ["moneyness", "maturity", "vol", "rate"]


def test_optional_and_backend_variant_conversions():
    config = make_price_table_config()
    assert config.adaptive is None
    adaptive = mo.AdaptiveGridParams()
    adaptive.target_iv_error = 0.001
    config.adaptive = adaptive
    assert isinstance(config.adaptive, mo.AdaptiveGridParams)
    config.adaptive = None
    assert config.adaptive is None

    bspline = mo.BSplineBackend()
    bspline.maturity_grid = [0.25, 0.5, 1.0]
    config.backend = bspline
    assert isinstance(config.backend, mo.BSplineBackend)

    cheb = mo.ChebyshevBackend()
    cheb.maturity = 1.0
    cheb.num_pts = [8, 6, 6, 4]
    config.backend = cheb
    assert isinstance(config.backend, mo.ChebyshevBackend)

    dim = mo.DimensionlessBackend()
    dim.maturity = 1.0
    dim.interpolant = mo.DimensionlessInterpolant.BSPLINE
    config.backend = dim
    assert isinstance(config.backend, mo.DimensionlessBackend)

    try:
        config.backend = object()
        raise AssertionError("invalid backend should fail")
    except mo.TypeConversionError:
        pass


def test_dividend_conversions():
    p = make_pricing_params()
    p.discrete_dividends = [mo.Dividend(0.25, 1.0), mo.Dividend(0.75, 1.0)]
    assert len(p.discrete_dividends) == 2
    p.discrete_dividends = [(0.25, 1.0), (0.75, 1.0)]
    assert len(p.discrete_dividends) == 2
    assert p.discrete_dividends[0].calendar_time == 0.25

    config = make_price_table_config()
    divs = mo.DiscreteDividendConfig()
    divs.maturity = 1.0
    divs.discrete_dividends = [(0.25, 1.0), (0.75, 1.0)]
    config.discrete_dividends = divs
    assert config.discrete_dividends is not None
    config.discrete_dividends = None
    assert config.discrete_dividends is None


def test_bspline_4d_price_table_workflow_and_persistence_paths():
    table = mo.make_price_table(make_price_table_config())
    assert table.surface_type == "bspline_4d"

    p = make_bspline_4d_off_grid_params()
    price = table.price(p)
    assert_finite_number(price)
    assert_finite_number(table.vega(p))
    assert_finite_number(table.delta(p))
    assert_finite_number(table.gamma(p))
    assert_finite_number(table.theta(p))
    assert_finite_number(table.rho(p))

    q = make_iv_query(price, p)
    iv = table.solve_iv(q)
    assert isinstance(iv, mo.IVSuccess)

    solver = table.make_iv_solver()
    success, result, error = solver.solve(q)
    assert success
    assert isinstance(result, mo.IVSuccess)

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "surface.parquet"
        table.save(path)
        loaded = mo.PriceTable.load(path)
        assert loaded.surface_type == table.surface_type
        assert_finite_number(loaded.price(p))


def test_legacy_interpolated_iv_solver_factory_still_works():
    solver = mo.make_interpolated_iv_solver(make_price_table_config())
    table = mo.make_price_table(make_price_table_config())
    price = table.price(make_pricing_params())
    success, result, error = solver.solve(make_iv_query(price))
    assert success
    assert isinstance(result, mo.IVSuccess)


def test_typed_exceptions_for_validation_and_persistence():
    config = make_price_table_config()
    config.grid.moneyness = [-1.0, 0.9, 1.0, 1.1]
    try:
        mo.make_price_table(config)
        raise AssertionError("invalid moneyness should fail")
    except mo.ValidationError as e:
        assert hasattr(e, "code")

    try:
        mo.PriceTable.load("/tmp/does-not-exist-mango-option.parquet")
        raise AssertionError("missing file should fail")
    except mo.PriceTableError as e:
        assert hasattr(e, "code")


def main():
    tests = [
        test_rate_spec_conversions,
        test_sequence_conversions_for_vectors_and_axes,
        test_optional_and_backend_variant_conversions,
        test_dividend_conversions,
        test_bspline_4d_price_table_workflow_and_persistence_paths,
        test_legacy_interpolated_iv_solver_factory_still_works,
        test_typed_exceptions_for_validation_and_persistence,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Remove numpy dependency from Python test target**

Modify `tests/BUILD.bazel` `python_bindings_test` target:

```python
py_test(
    name = "python_bindings_test",
    size = "medium",
    srcs = ["test_bindings.py"],
    main = "test_bindings.py",
    data = ["//src/python:mango_option"],
    imports = ["../src/python"],
)
```

- [ ] **Step 3: Run the Python test and verify it fails**

Run:

```bash
bazel test //tests:python_bindings_test
```

Expected: FAIL because `PriceTableConfig`, `PriceTable`, typed exceptions, dimensionless backend binding, tuple dividend conversion, and no-numpy `PriceTableAxes` conversion are not implemented.

- [ ] **Step 4: Commit failing binding tests**

```bash
git add tests/test_bindings.py tests/BUILD.bazel
git commit -m "test: specify python price table parity binding"
```

---

### Task 4: Add Python Exceptions And Conversion Helpers

**Files:**
- Modify: `src/python/mango_bindings.cpp`

- [ ] **Step 1: Remove numpy include and add helper includes**

At the top of `src/python/mango_bindings.cpp`, replace:

```cpp
#include <pybind11/numpy.h>
```

with:

```cpp
#include <array>
#include <filesystem>
#include <sstream>
```

Add includes:

```cpp
#include "mango/option/price_table_factory.hpp"
```

- [ ] **Step 2: Add module exception handles and raise helpers**

Above `PYBIND11_MODULE`, add:

```cpp
namespace {

py::object g_validation_error;
py::object g_price_table_error;
py::object g_solver_error;
py::object g_type_conversion_error;

[[noreturn]] void raise_with_code(
    const py::object& exc_type,
    const std::string& message,
    int code) {
    py::object exc = exc_type(message);
    exc.attr("code") = code;
    PyErr_SetObject(exc_type.ptr(), exc.ptr());
    throw py::error_already_set();
}

[[noreturn]] void raise_validation_error(const mango::ValidationError& err) {
    raise_with_code(g_validation_error,
        "validation error code " + std::to_string(static_cast<int>(err.code)) +
        " value=" + std::to_string(err.value),
        static_cast<int>(err.code));
}

[[noreturn]] void raise_price_table_error(const mango::PriceTableError& err) {
    raise_with_code(g_price_table_error,
        "price table error code " + std::to_string(static_cast<int>(err.code)) +
        " axis=" + std::to_string(err.axis_index) +
        " count=" + std::to_string(err.count),
        static_cast<int>(err.code));
}

[[noreturn]] void raise_iv_error(const mango::IVError& err) {
    raise_with_code(g_solver_error,
        "IV solver error code " + std::to_string(static_cast<int>(err.code)),
        static_cast<int>(err.code));
}

[[noreturn]] void raise_type_conversion_error(const std::string& message) {
    raise_with_code(g_type_conversion_error, message, -1);
}

std::string python_path_to_string(const py::object& obj) {
    py::object fspath = py::module_::import("os").attr("fspath");
    return fspath(obj).cast<std::string>();
}

std::vector<double> python_to_double_vector(const py::handle& obj,
                                            const char* field_name) {
    if (!PySequence_Check(obj.ptr())) {
        raise_type_conversion_error(std::string(field_name) + " must be a sequence of floats");
    }
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    std::vector<double> out;
    out.reserve(seq.size());
    for (py::handle item : seq) {
        out.push_back(py::cast<double>(item));
    }
    return out;
}

std::vector<mango::Dividend> python_to_dividends(const py::handle& obj) {
    if (!PySequence_Check(obj.ptr())) {
        raise_type_conversion_error("discrete_dividends must be a sequence");
    }
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    std::vector<mango::Dividend> out;
    out.reserve(seq.size());
    for (py::handle item : seq) {
        if (py::isinstance<mango::Dividend>(item)) {
            out.push_back(py::cast<mango::Dividend>(item));
            continue;
        }
        if (!PySequence_Check(item.ptr())) {
            raise_type_conversion_error("dividend must be Dividend or (time, amount)");
        }
        py::sequence pair = py::reinterpret_borrow<py::sequence>(item);
        if (pair.size() != 2) {
            raise_type_conversion_error("dividend tuple must have exactly two values");
        }
        out.push_back(mango::Dividend{
            pair[0].cast<double>(),
            pair[1].cast<double>(),
        });
    }
    return out;
}

py::list dividends_to_python(const std::vector<mango::Dividend>& dividends) {
    py::list result;
    for (const auto& d : dividends) result.append(d);
    return result;
}

}  // namespace
```

- [ ] **Step 3: Initialize exception classes in module init**

At the start of `PYBIND11_MODULE`, after `m.doc()`:

```cpp
auto mango_error = py::exception<py::error_already_set>(m, "MangoError");
g_validation_error = py::reinterpret_borrow<py::object>(
    PyErr_NewException("mango_option.ValidationError", mango_error.ptr(), nullptr));
g_price_table_error = py::reinterpret_borrow<py::object>(
    PyErr_NewException("mango_option.PriceTableError", mango_error.ptr(), nullptr));
g_solver_error = py::reinterpret_borrow<py::object>(
    PyErr_NewException("mango_option.SolverException", mango_error.ptr(), nullptr));
g_type_conversion_error = py::reinterpret_borrow<py::object>(
    PyErr_NewException("mango_option.TypeConversionError", mango_error.ptr(), nullptr));
m.attr("ValidationError") = g_validation_error;
m.attr("PriceTableError") = g_price_table_error;
m.attr("SolverException") = g_solver_error;
m.attr("TypeConversionError") = g_type_conversion_error;
```

If `py::exception<py::error_already_set>` does not compile, replace it with:

```cpp
py::object mango_error = py::reinterpret_borrow<py::object>(
    PyErr_NewException("mango_option.MangoError", PyExc_Exception, nullptr));
m.attr("MangoError") = mango_error;
```

and keep the derived exception setup unchanged.

- [ ] **Step 4: Update rate conversion failure**

Replace `throw py::type_error("rate must be a float or YieldCurve");` in `python_to_rate_spec` with:

```cpp
raise_type_conversion_error("rate must be a float, int, or YieldCurve");
```

- [ ] **Step 5: Run compile check**

Run:

```bash
bazel build //src/python:mango_option
```

Expected: FAIL only if exception class setup needs the fallback from Step 3; after applying fallback, PASS.

- [ ] **Step 6: Commit**

```bash
git add src/python/mango_bindings.cpp
git commit -m "feat: add python binding exceptions and converters"
```

---

### Task 5: Bind Missing Config Fields And Automatic Conversions

**Files:**
- Modify: `src/python/mango_bindings.cpp`

- [ ] **Step 1: Bind missing config fields**

In `InterpolatedIVSolverConfig`, add:

```cpp
.def_readwrite("vega_threshold", &mango::InterpolatedIVSolverConfig::vega_threshold)
```

In `AdaptiveGridParams`, add:

```cpp
.def_readwrite("validation_samples", &mango::AdaptiveGridParams::validation_samples)
.def_readwrite("refinement_factor", &mango::AdaptiveGridParams::refinement_factor)
.def_readwrite("lhs_seed", &mango::AdaptiveGridParams::lhs_seed)
.def_readwrite("vega_floor", &mango::AdaptiveGridParams::vega_floor)
.def_readwrite("max_failure_rate", &mango::AdaptiveGridParams::max_failure_rate)
```

In `ChebyshevBackend`, add list conversion for `num_pts`:

```cpp
.def_property("num_pts",
    [](const mango::ChebyshevBackend& b) {
        return py::make_tuple(b.num_pts[0], b.num_pts[1], b.num_pts[2], b.num_pts[3]);
    },
    [](mango::ChebyshevBackend& b, const py::handle& obj) {
        if (!PySequence_Check(obj.ptr())) {
            raise_type_conversion_error("num_pts must be a sequence of four integers");
        }
        py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
        if (seq.size() != 4) {
            raise_type_conversion_error("num_pts must contain exactly four values");
        }
        for (size_t i = 0; i < 4; ++i) b.num_pts[i] = seq[i].cast<size_t>();
    })
```

- [ ] **Step 2: Bind Dimensionless backend and enum**

Add after `ChebyshevBackend`:

```cpp
py::enum_<mango::DimensionlessBackend::Interpolant>(m, "DimensionlessInterpolant")
    .value("BSPLINE", mango::DimensionlessBackend::Interpolant::BSpline)
    .value("CHEBYSHEV", mango::DimensionlessBackend::Interpolant::Chebyshev);

py::class_<mango::DimensionlessBackend>(m, "DimensionlessBackend")
    .def(py::init<>())
    .def_readwrite("maturity", &mango::DimensionlessBackend::maturity)
    .def_readwrite("interpolant", &mango::DimensionlessBackend::interpolant)
    .def_property("chebyshev_pts",
        [](const mango::DimensionlessBackend& b) {
            return py::make_tuple(b.chebyshev_pts[0], b.chebyshev_pts[1], b.chebyshev_pts[2]);
        },
        [](mango::DimensionlessBackend& b, const py::handle& obj) {
            if (!PySequence_Check(obj.ptr())) {
                raise_type_conversion_error("chebyshev_pts must be a sequence of three integers");
            }
            py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
            if (seq.size() != 3) {
                raise_type_conversion_error("chebyshev_pts must contain exactly three values");
            }
            for (size_t i = 0; i < 3; ++i) b.chebyshev_pts[i] = seq[i].cast<size_t>();
        });
```

- [ ] **Step 3: Accept dimensionless backend in factory config**

Modify `IVSolverFactoryConfig.backend` setter:

```cpp
if (py::isinstance<mango::BSplineBackend>(obj))
    c.backend = obj.cast<mango::BSplineBackend>();
else if (py::isinstance<mango::ChebyshevBackend>(obj))
    c.backend = obj.cast<mango::ChebyshevBackend>();
else if (py::isinstance<mango::DimensionlessBackend>(obj))
    c.backend = obj.cast<mango::DimensionlessBackend>();
else
    raise_type_conversion_error("backend must be BSplineBackend, ChebyshevBackend, or DimensionlessBackend");
```

- [ ] **Step 4: Alias `PriceTableConfig` to existing factory config**

After binding `IVSolverFactoryConfig`, add:

```cpp
m.attr("PriceTableConfig") = m.attr("IVSolverFactoryConfig");
```

This preserves one C++ config type and gives Python the domain-centered name.

- [ ] **Step 5: Replace dividend `def_readwrite` with automatic conversion properties**

For `DividendSpec`, replace `def_readwrite("discrete_dividends", ...)` with:

```cpp
.def_property("discrete_dividends",
    [](const mango::DividendSpec& d) { return dividends_to_python(d.discrete_dividends); },
    [](mango::DividendSpec& d, const py::handle& obj) {
        d.discrete_dividends = python_to_dividends(obj);
    })
```

For `PricingParams`, replace `def_readwrite("discrete_dividends", ...)` with the same property against `p.discrete_dividends`.

For `DiscreteDividendConfig`, replace `def_readwrite("discrete_dividends", ...)` with the same property against `d.discrete_dividends`.

- [ ] **Step 6: Replace `PriceTableAxes` numpy conversion**

Replace `PriceTableAxes.grids` property with:

```cpp
.def_property("grids",
    [](const mango::PriceTableAxes& self) {
        py::list result;
        for (const auto& grid : self.grids) {
            py::list values;
            for (double v : grid) values.append(v);
            result.append(values);
        }
        return result;
    },
    [](mango::PriceTableAxes& self, const py::handle& grids_obj) {
        if (!PySequence_Check(grids_obj.ptr())) {
            raise_type_conversion_error("grids must be a sequence of four sequences");
        }
        py::sequence grids = py::reinterpret_borrow<py::sequence>(grids_obj);
        if (grids.size() != 4) {
            raise_validation_error(mango::ValidationError{
                mango::ValidationErrorCode::InvalidGridSize,
                static_cast<double>(grids.size())});
        }
        for (size_t i = 0; i < 4; ++i) {
            self.grids[i] = python_to_double_vector(grids[i], "grid");
        }
    })
```

Keep `names` as a sequence property, but change wrong-size errors to `raise_validation_error(...)`.

- [ ] **Step 7: Run conversion-focused test**

Run:

```bash
bazel test //tests:python_bindings_test
```

Expected: still FAIL because `PriceTable` and `make_price_table` are not bound, but conversion-related failures should be reduced to those missing symbols.

- [ ] **Step 8: Commit**

```bash
git add src/python/mango_bindings.cpp
git commit -m "feat: complete python config conversions"
```

---

### Task 6: Bind `PriceTable`, Factory, Persistence, And Table IV

**Files:**
- Modify: `src/python/mango_bindings.cpp`
- Modify: `src/python/BUILD.bazel`

- [ ] **Step 1: Add compression enum**

In `PYBIND11_MODULE`, add near other enums:

```cpp
py::enum_<mango::PriceTableCompression>(m, "PriceTableCompression")
    .value("NONE", mango::PriceTableCompression::NONE)
    .value("SNAPPY", mango::PriceTableCompression::SNAPPY)
    .value("ZSTD", mango::PriceTableCompression::ZSTD);
```

- [ ] **Step 2: Add helper to unwrap expected Greek values**

Above `PYBIND11_MODULE`, add:

```cpp
double greek_or_raise(std::expected<double, mango::GreekError> result,
                      const char* name) {
    if (!result.has_value()) {
        raise_with_code(g_solver_error,
            std::string(name) + " failed", static_cast<int>(result.error().code));
    }
    return *result;
}
```

- [ ] **Step 3: Bind `PriceTable`**

Add before `InterpolatedIVSolver` binding:

```cpp
py::class_<mango::AnyPriceTable>(m, "PriceTable")
    .def_property_readonly("surface_type", &mango::AnyPriceTable::surface_type)
    .def_property_readonly("option_type", &mango::AnyPriceTable::option_type)
    .def_property_readonly("dividend_yield", &mango::AnyPriceTable::dividend_yield)
    .def("price", &mango::AnyPriceTable::price, py::arg("params"))
    .def("vega", &mango::AnyPriceTable::vega, py::arg("params"))
    .def("delta", [](const mango::AnyPriceTable& t, const mango::PricingParams& p) {
        return greek_or_raise(t.delta(p), "delta");
    }, py::arg("params"))
    .def("gamma", [](const mango::AnyPriceTable& t, const mango::PricingParams& p) {
        return greek_or_raise(t.gamma(p), "gamma");
    }, py::arg("params"))
    .def("theta", [](const mango::AnyPriceTable& t, const mango::PricingParams& p) {
        return greek_or_raise(t.theta(p), "theta");
    }, py::arg("params"))
    .def("rho", [](const mango::AnyPriceTable& t, const mango::PricingParams& p) {
        return greek_or_raise(t.rho(p), "rho");
    }, py::arg("params"))
    .def("make_iv_solver",
        [](const mango::AnyPriceTable& table,
           std::optional<mango::InterpolatedIVSolverConfig> config) {
            auto result = table.make_iv_solver(config.value_or(mango::InterpolatedIVSolverConfig{}));
            if (!result.has_value()) raise_validation_error(result.error());
            return std::move(*result);
        },
        py::arg("solver_config") = py::none())
    .def("solve_iv",
        [](const mango::AnyPriceTable& table,
           const mango::IVQuery& query,
           std::optional<mango::InterpolatedIVSolverConfig> config) {
            auto result = table.solve_iv(query, config.value_or(mango::InterpolatedIVSolverConfig{}));
            if (!result.has_value()) raise_iv_error(result.error());
            return *result;
        },
        py::arg("query"),
        py::arg("solver_config") = py::none())
    .def("save",
        [](const mango::AnyPriceTable& table,
           const py::object& path,
           mango::PriceTableCompression compression) {
            auto result = table.save(python_path_to_string(path), compression);
            if (!result.has_value()) raise_price_table_error(result.error());
        },
        py::arg("path"),
        py::arg("compression") = mango::PriceTableCompression::ZSTD)
    .def_static("load",
        [](const py::object& path) {
            auto result = mango::load_price_table(python_path_to_string(path));
            if (!result.has_value()) raise_price_table_error(result.error());
            return std::move(*result);
        },
        py::arg("path"));
```

- [ ] **Step 4: Bind `make_price_table`**

Add near `make_interpolated_iv_solver`:

```cpp
m.def("make_price_table",
    [](const mango::IVSolverFactoryConfig& config) {
        auto result = mango::make_price_table(config);
        if (!result.has_value()) raise_validation_error(result.error());
        return std::move(*result);
    },
    py::arg("config"),
    "Build a reusable price table from a factory configuration");
```

- [ ] **Step 5: Update `make_interpolated_iv_solver` error mapping**

Replace the current `throw py::value_error(...)` failure in `make_interpolated_iv_solver` with:

```cpp
raise_validation_error(result.error());
```

Leave the return tuple style of `InterpolatedIVSolver.solve` and `solve_batch` unchanged.

- [ ] **Step 6: Update Python binding Bazel deps**

Modify `src/python/BUILD.bazel` deps:

```python
deps = [
    "//src/option:iv_solver",
    "//src/option:interpolated_iv_solver",
    "//src/option:price_table_factory",
    "//src/option:american_option",
    "//src/option:american_option_batch",
    "//src/option:option_grid",
    "//src/option/table/parquet:parquet_io",
    "//src/option/table/serialization:from_data",
    "//src/option/table/serialization:to_data",
    "//src/option/table/bspline:bspline_surface",
    "//src/option/table/bspline:bspline_builder",
    "//src/math:root_finding",
    "//src/support:error_types",
]
```

- [ ] **Step 7: Run Python binding test**

Run:

```bash
bazel test //tests:python_bindings_test
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/python/mango_bindings.cpp src/python/BUILD.bazel
git commit -m "feat: bind reusable python price table"
```

---

### Task 7: Remove Hard Numpy Dependency

**Files:**
- Modify: `BUILD.bazel`
- Modify: `third_party/requirements.txt`
- Modify: `MODULE.bazel.lock` only by running Bazel lock update if required by the repo workflow

- [ ] **Step 1: Remove wheel requirement**

In root `BUILD.bazel`, remove:

```python
requires = ["numpy"],
```

from the `py_wheel(name = "mango_option_wheel", ...)` target.

- [ ] **Step 2: Empty `third_party/requirements.txt`**

Replace `third_party/requirements.txt` contents with an empty file.

- [ ] **Step 3: Search for remaining numpy usage**

Run:

```bash
rg -n "numpy|pybind11/numpy|py::array|array_t|requirement\\(\"numpy\"\\)|requires = \\[\"numpy\"\\]" .
```

Expected: only historical references in `docs/archive/` and `MODULE.bazel.lock` may remain. No active source, tests, or BUILD files should require numpy.

- [ ] **Step 4: Run Python build and test**

Run:

```bash
bazel test //tests:python_bindings_test
bazel build //src/python:mango_option //:mango_option_wheel
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add BUILD.bazel third_party/requirements.txt MODULE.bazel.lock
git commit -m "build: remove hard numpy dependency from python binding"
```

If `MODULE.bazel.lock` is unchanged, omit it from `git add`.

---

### Task 8: Update Python Documentation

**Files:**
- Modify: `docs/PYTHON_GUIDE.md`
- Modify: `README.md`

- [ ] **Step 1: Replace stale single-option examples**

In `docs/PYTHON_GUIDE.md`, replace `params.type` with `params.option_type` everywhere.

- [ ] **Step 2: Replace stale `PriceTableWorkspace` section**

Replace the current "Saving and Loading Price Tables" section with:

````markdown
### Reusable Price Tables

Build a reusable interpolation surface once, then use it for pricing, Greeks,
fast implied volatility, and persistence. With `BSplineBackend` and no discrete
dividends, this builds the standard 4D B-spline table over log-moneyness,
maturity, volatility, and rate.

```python
import mango_option as mo

config = mo.PriceTableConfig()
config.option_type = mo.OptionType.PUT
config.spot = 100.0
config.dividend_yield = 0.02
config.grid.moneyness = [0.8, 0.9, 1.0, 1.1, 1.2]
config.grid.vol = [0.10, 0.20, 0.30, 0.40]
config.grid.rate = [0.01, 0.03, 0.05, 0.07]

backend = mo.BSplineBackend()
backend.maturity_grid = [0.1, 0.25, 0.5, 1.0]
config.backend = backend

table = mo.make_price_table(config)
assert table.surface_type == "bspline_4d"

params = mo.PricingParams()
params.spot = 100.0
params.strike = 100.0
params.maturity = 0.5
params.volatility = 0.20
params.rate = 0.05
params.dividend_yield = 0.02
params.option_type = mo.OptionType.PUT

print(table.price(params))
print(table.delta(params))
print(table.gamma(params))

query = mo.IVQuery()
query.spot = 100.0
query.strike = 100.0
query.maturity = 0.5
query.rate = 0.05
query.dividend_yield = 0.02
query.option_type = mo.OptionType.PUT
query.market_price = table.price(params)

iv = table.solve_iv(query)
solver = table.make_iv_solver()
success, result, error = solver.solve(query)  # Back-compatible tuple API

table.save("spy_puts.parquet")
loaded = mo.PriceTable.load("spy_puts.parquet")
```

`bspline_4d` is the main high-throughput interpolation path. Use it when the
surface domain is described by log-moneyness, maturity, volatility, and rate.
Use segmented B-spline surfaces for discrete cash dividends and dimensionless
3D surfaces only when those model constraints are intentional.
````

- [ ] **Step 3: Add conversion notes**

Add a "Python Conversions" section:

```markdown
## Python Conversions

The binding accepts Python-native values for common inputs:

- rates: `float`, `int`, or `YieldCurve`
- grids and vector fields: lists or tuples of numbers
- dividend schedules: `Dividend` objects or `(time, amount)` pairs
- optional config fields: assign `None` to clear
- persistence paths: strings or `pathlib.Path`

The core binding does not require numpy, pandas, pyarrow, or dataframe objects.
```

- [ ] **Step 4: Add error notes**

Add:

```markdown
## Errors

New price-table APIs raise typed exceptions:

- `ValidationError`
- `PriceTableError`
- `SolverException`
- `TypeConversionError`

Existing IV solver methods keep the back-compatible `(success, result, error)`
return shape.
```

- [ ] **Step 5: Update README Python example if needed**

Ensure `README.md` Python snippets use `option_type`, do not mention `PriceTableWorkspace` or numpy, and preserve the existing message that the primary interpolation table is 4D B-spline over moneyness, maturity, vol, and rate.

- [ ] **Step 6: Commit**

```bash
git add docs/PYTHON_GUIDE.md README.md
git commit -m "docs: document python price table parity API"
```

---

### Task 9: Full Verification And Coverage Check

**Files:**
- No source edits expected

- [ ] **Step 1: Run focused tests**

Run:

```bash
bazel test //tests:python_bindings_test //tests:price_table_factory_test //tests:price_table_builder_test //tests:price_table_builder_factories_test //tests:price_table_greeks_test //tests:price_table_data_test //tests:parquet_io_test //tests:interpolated_iv_solver_test
```

Expected: PASS. The disabled historical `price_table_4d_integration_test` target does not count toward acceptance; active 4D B-spline coverage comes from `price_table_factory_test`, `price_table_builder_test`, `price_table_builder_factories_test`, `price_table_greeks_test`, and `interpolated_iv_solver_test`.

- [ ] **Step 2: Build Python binding and wheel**

Run:

```bash
bazel build //src/python:mango_option //:mango_option_wheel
```

Expected: PASS.

- [ ] **Step 3: Run binding coverage**

Run:

```bash
bazel coverage --instrumentation_filter=//src/python --combined_report=lcov //tests:python_bindings_test
```

Expected: PASS. Record line/function/branch coverage from `bazel-out/_coverage/_coverage_report.dat` in the final implementation notes. Do not treat numerical coverage percentage as the acceptance gate; acceptance is the binding reachability matrix in `tests/test_bindings.py`.

- [ ] **Step 4: Search for stale API names and hard deps**

Run:

```bash
rg -n "PriceTableWorkspace|params\\.type|grid_accuracy|numpy|pybind11/numpy|py::array|array_t" README.md docs/PYTHON_GUIDE.md src/python tests BUILD.bazel third_party/requirements.txt
rg -n "bspline_4d|4D B-spline|log-moneyness, maturity, volatility, and rate" README.md docs/PYTHON_GUIDE.md tests/test_bindings.py tests/price_table_factory_test.cc
```

Expected: the stale-name grep has no matches except intentional compatibility mentions of existing APIs. `grid_accuracy` should not appear as a Python `IVSolverConfig` field. The 4D B-spline grep must find active docs and tests naming `bspline_4d` and the four axes.

- [ ] **Step 5: Commit any verification-only doc correction**

If Step 4 found stale docs, fix them and commit:

```bash
git add README.md docs/PYTHON_GUIDE.md
git commit -m "docs: remove stale python binding references"
```

Skip this commit if no files changed.

---

## Self-Review Checklist

- Spec coverage:
  - Python API parity is workflow-level: covered by `PriceTable`, `make_price_table`, compatibility wrapper, docs.
  - Stable scope excludes `mango::simple`: documented in `CONTEXT.md` and not included in tasks.
  - Persistence included: covered by C++ factory tests, Python save/load binding, Parquet deps.
  - 4D B-spline path is first-class: covered by active C++ factory/builder/Greek/IV tests, Python `bspline_4d` reachability test, and docs naming the four axes.
  - Price and Greeks on reusable table: covered by C++ and Python reachability tests.
  - Direct IV and reusable solver from table: covered by C++ and Python tests.
  - Typed exceptions: covered by binding helpers and Python tests.
  - Conversion coverage: covered by Python tests for rates, sequences, optionals, backends, dividends, enums, nested configs, and paths.
  - No hard numpy/dataframe dependency: covered by dependency removal task and grep.
- Placeholder scan: no task contains TBD, TODO, "similar to", or unbounded "handle edge cases" language.
- Type consistency:
  - Python uses `PriceTableConfig` alias for `IVSolverFactoryConfig`.
  - Python artifact is `PriceTable`.
  - C++ artifact is `AnyPriceTable`.
  - Existing `AnyInterpIVSolver` remains the Python `InterpolatedIVSolver`.
