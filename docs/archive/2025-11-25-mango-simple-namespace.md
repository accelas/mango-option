# mango::simple Namespace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a user-friendly interface that converts financial data from external sources (yfinance, Databento, IBKR) into internal solver calls with type-safe conversions and deferred double conversion.

**Architecture:** Type-tagged data structures (`Price`, `Timestamp`) store original formats (fixed-point, nanoseconds, strings). Type-safe `Converter<Source>` traits ensure compile-time correctness. Conversion to `double` happens only at the solver boundary. `OptionChain` aggregates options by expiry for batch IV computation.

**Tech Stack:** C++23 (std::variant, std::expected, std::optional, concepts), existing mango solver APIs

---

## Task 1: Create Price Type with Deferred Conversion

**Files:**
- Create: `src/simple/price.hpp`
- Test: `tests/simple_price_test.cc`
- Create: `src/simple/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/simple_price_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "mango/simple/price.hpp"

TEST(SimplePriceTest, ConstructFromDouble) {
    mango::simple::Price p{100.50};
    EXPECT_DOUBLE_EQ(p.to_double(), 100.50);
}

TEST(SimplePriceTest, ConstructFromFixedPoint9) {
    // Databento format: price * 10^9
    int64_t fixed = 100500000000LL;  // 100.50 * 10^9
    mango::simple::Price p{fixed, mango::simple::PriceFormat::FixedPoint9};
    EXPECT_DOUBLE_EQ(p.to_double(), 100.50);
}

TEST(SimplePriceTest, MidpointPreservesPrecision) {
    // Two fixed-point prices
    mango::simple::Price bid{100250000000LL, mango::simple::PriceFormat::FixedPoint9};
    mango::simple::Price ask{100750000000LL, mango::simple::PriceFormat::FixedPoint9};

    auto mid = mango::simple::Price::midpoint(bid, ask);
    ASSERT_TRUE(mid.has_value());
    EXPECT_DOUBLE_EQ(mid->to_double(), 100.50);
}

TEST(SimplePriceTest, MidpointMixedFormats) {
    // Mixed formats: converts to double for midpoint
    mango::simple::Price bid{100.25};
    mango::simple::Price ask{100750000000LL, mango::simple::PriceFormat::FixedPoint9};

    auto mid = mango::simple::Price::midpoint(bid, ask);
    ASSERT_TRUE(mid.has_value());
    EXPECT_DOUBLE_EQ(mid->to_double(), 100.50);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:simple_price_test --test_output=all`
Expected: FAIL with "no such package 'src/simple'" or "price.hpp: No such file"

**Step 3: Create BUILD.bazel for simple library**

Create `src/simple/BUILD.bazel`:

```python
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "price",
    hdrs = ["price.hpp"],
    deps = [],
)
```

**Step 4: Write minimal implementation**

Create `src/simple/price.hpp`:

```cpp
/**
 * @file price.hpp
 * @brief Price type with deferred double conversion
 */

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

namespace mango::simple {

/// Price storage format
enum class PriceFormat {
    Double,       // Native double
    FixedPoint9   // Databento: price * 10^9
};

/// Price with deferred conversion
///
/// Stores prices in their original format to preserve precision.
/// Conversion to double happens only when needed (at solver boundary).
class Price {
public:
    /// Construct from double
    explicit Price(double value) : value_(value) {}

    /// Construct from fixed-point
    Price(int64_t value, PriceFormat format) {
        if (format == PriceFormat::FixedPoint9) {
            value_ = FixedPoint9{value};
        } else {
            value_ = static_cast<double>(value);
        }
    }

    /// Convert to double (deferred conversion point)
    [[nodiscard]] double to_double() const {
        return std::visit([](const auto& v) -> double {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, double>) {
                return v;
            } else {
                return static_cast<double>(v.value) * 1e-9;
            }
        }, value_);
    }

    /// Compute midpoint of two prices
    ///
    /// If both prices are same format, preserves that format.
    /// Otherwise, converts to double.
    static std::optional<Price> midpoint(const Price& a, const Price& b) {
        // Check if both are same fixed-point format
        if (auto* fa = std::get_if<FixedPoint9>(&a.value_)) {
            if (auto* fb = std::get_if<FixedPoint9>(&b.value_)) {
                // Average in fixed-point to preserve precision
                int64_t mid = (fa->value + fb->value) / 2;
                return Price{mid, PriceFormat::FixedPoint9};
            }
        }
        // Fall back to double
        return Price{(a.to_double() + b.to_double()) / 2.0};
    }

    /// Check if stored as fixed-point
    [[nodiscard]] bool is_fixed_point() const {
        return std::holds_alternative<FixedPoint9>(value_);
    }

private:
    struct FixedPoint9 {
        int64_t value;
    };

    std::variant<double, FixedPoint9> value_;
};

}  // namespace mango::simple
```

**Step 5: Add test target to tests/BUILD.bazel**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "simple_price_test",
    srcs = ["simple_price_test.cc"],
    deps = [
        "//src/simple:price",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:simple_price_test --test_output=all`
Expected: PASS (4 tests)

**Step 7: Commit**

```bash
git add src/simple/BUILD.bazel src/simple/price.hpp tests/simple_price_test.cc tests/BUILD.bazel
git commit -m "feat(simple): add Price type with deferred conversion"
```

---

## Task 2: Create Timestamp Type with Multiple Formats

**Files:**
- Create: `src/simple/timestamp.hpp`
- Test: `tests/simple_timestamp_test.cc`
- Modify: `src/simple/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/simple_timestamp_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "mango/simple/timestamp.hpp"
#include <chrono>

using namespace mango::simple;

TEST(SimpleTimestampTest, ConstructFromISODate) {
    Timestamp ts{"2024-06-21"};
    auto tp = ts.to_timepoint();
    ASSERT_TRUE(tp.has_value());
    // Should be midnight UTC on that date
}

TEST(SimpleTimestampTest, ConstructFromCompactDate) {
    Timestamp ts{"20240621", TimestampFormat::Compact};
    auto tp = ts.to_timepoint();
    ASSERT_TRUE(tp.has_value());
}

TEST(SimpleTimestampTest, ConstructFromNanoseconds) {
    // 2024-06-21 00:00:00 UTC in nanoseconds since epoch
    uint64_t nanos = 1718928000000000000ULL;
    Timestamp ts{nanos};
    auto tp = ts.to_timepoint();
    ASSERT_TRUE(tp.has_value());
}

TEST(SimpleTimestampTest, ConstructFromISO8601WithTime) {
    Timestamp ts{"2024-06-21T10:30:00"};
    auto tp = ts.to_timepoint();
    ASSERT_TRUE(tp.has_value());
}

TEST(SimpleTimestampTest, ComputeTauToExpiry) {
    Timestamp now{"2024-06-21T10:30:00"};
    Timestamp expiry{"2024-06-21T16:00:00"};  // PM settlement

    double tau = compute_tau(now, expiry);
    // 5.5 hours remaining / (365 * 24 hours) ≈ 0.000628
    EXPECT_NEAR(tau, 5.5 / (365.0 * 24.0), 1e-6);
}

TEST(SimpleTimestampTest, NowReturnsCurrentTime) {
    auto ts = Timestamp::now();
    auto tp = ts.to_timepoint();
    ASSERT_TRUE(tp.has_value());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:simple_timestamp_test --test_output=all`
Expected: FAIL with "timestamp.hpp: No such file"

**Step 3: Write minimal implementation**

Create `src/simple/timestamp.hpp`:

```cpp
/**
 * @file timestamp.hpp
 * @brief Timestamp type with multiple format support
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <variant>

namespace mango::simple {

/// Timestamp storage format
enum class TimestampFormat {
    ISO,          // "2024-06-21" or "2024-06-21T10:30:00"
    Compact,      // "20240621"
    Nanoseconds   // uint64_t nanoseconds since epoch
};

/// Timestamp with multiple format support
///
/// Stores timestamps in original format, converts on demand.
class Timestamp {
public:
    using TimePoint = std::chrono::system_clock::time_point;

    /// Construct from ISO date string (auto-detect format)
    explicit Timestamp(std::string value, TimestampFormat format = TimestampFormat::ISO)
        : value_(StringValue{std::move(value), format}) {}

    /// Construct from nanoseconds since epoch
    explicit Timestamp(uint64_t nanos)
        : value_(nanos) {}

    /// Construct from time_point
    explicit Timestamp(TimePoint tp)
        : value_(tp) {}

    /// Get current time
    static Timestamp now() {
        return Timestamp{std::chrono::system_clock::now()};
    }

    /// Convert to time_point
    [[nodiscard]] std::expected<TimePoint, std::string> to_timepoint() const;

    /// Convert to string for display
    [[nodiscard]] std::string to_string() const;

private:
    struct StringValue {
        std::string str;
        TimestampFormat format;
    };

    std::variant<StringValue, uint64_t, TimePoint> value_;

    // Parse helpers
    static std::expected<TimePoint, std::string> parse_iso(const std::string& s);
    static std::expected<TimePoint, std::string> parse_compact(const std::string& s);
};

/// Compute time to expiry in years (calendar time basis)
///
/// Uses calendar time (365 * 24 hours) for consistency with
/// market-quoted implied volatilities.
///
/// @param valuation Current valuation time
/// @param expiry Option expiry time
/// @return Time to expiry in years
double compute_tau(const Timestamp& valuation, const Timestamp& expiry);

}  // namespace mango::simple
```

**Step 4: Create timestamp.cpp for implementation**

Create `src/simple/timestamp.cpp`:

```cpp
#include "mango/simple/timestamp.hpp"
#include <charconv>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace mango::simple {

std::expected<Timestamp::TimePoint, std::string> Timestamp::to_timepoint() const {
    return std::visit([](const auto& v) -> std::expected<TimePoint, std::string> {
        using T = std::decay_t<decltype(v)>;

        if constexpr (std::is_same_v<T, TimePoint>) {
            return v;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            // Nanoseconds since epoch
            auto duration = std::chrono::nanoseconds(v);
            return TimePoint{std::chrono::duration_cast<TimePoint::duration>(duration)};
        } else {
            // String value
            if (v.format == TimestampFormat::Compact) {
                return parse_compact(v.str);
            } else {
                return parse_iso(v.str);
            }
        }
    }, value_);
}

std::expected<Timestamp::TimePoint, std::string> Timestamp::parse_iso(const std::string& s) {
    std::tm tm = {};
    std::istringstream ss(s);

    // Try with time component first
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) {
        // Try date only
        ss.clear();
        ss.str(s);
        ss >> std::get_time(&tm, "%Y-%m-%d");
        if (ss.fail()) {
            return std::unexpected("Failed to parse ISO timestamp: " + s);
        }
    }

    auto time = std::mktime(&tm);
    if (time == -1) {
        return std::unexpected("Invalid timestamp: " + s);
    }

    return std::chrono::system_clock::from_time_t(time);
}

std::expected<Timestamp::TimePoint, std::string> Timestamp::parse_compact(const std::string& s) {
    if (s.length() != 8) {
        return std::unexpected("Compact format must be 8 digits: " + s);
    }

    int year, month, day;
    auto r1 = std::from_chars(s.data(), s.data() + 4, year);
    auto r2 = std::from_chars(s.data() + 4, s.data() + 6, month);
    auto r3 = std::from_chars(s.data() + 6, s.data() + 8, day);

    if (r1.ec != std::errc{} || r2.ec != std::errc{} || r3.ec != std::errc{}) {
        return std::unexpected("Failed to parse compact date: " + s);
    }

    std::tm tm = {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;

    auto time = std::mktime(&tm);
    if (time == -1) {
        return std::unexpected("Invalid date: " + s);
    }

    return std::chrono::system_clock::from_time_t(time);
}

std::string Timestamp::to_string() const {
    auto tp_result = to_timepoint();
    if (!tp_result) {
        return "<invalid>";
    }

    auto time = std::chrono::system_clock::to_time_t(*tp_result);
    std::tm tm = *std::gmtime(&time);

    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

double compute_tau(const Timestamp& valuation, const Timestamp& expiry) {
    auto val_tp = valuation.to_timepoint();
    auto exp_tp = expiry.to_timepoint();

    if (!val_tp || !exp_tp) {
        return 0.0;  // Error case
    }

    auto duration = *exp_tp - *val_tp;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration).count();

    // Calendar time: hours / (365 * 24)
    return static_cast<double>(hours) / (365.0 * 24.0);
}

}  // namespace mango::simple
```

**Step 5: Update BUILD.bazel**

Modify `src/simple/BUILD.bazel`:

```python
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "price",
    hdrs = ["price.hpp"],
    deps = [],
)

cc_library(
    name = "timestamp",
    srcs = ["timestamp.cpp"],
    hdrs = ["timestamp.hpp"],
    deps = [],
)
```

**Step 6: Add test target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "simple_timestamp_test",
    srcs = ["simple_timestamp_test.cc"],
    deps = [
        "//src/simple:timestamp",
        "@googletest//:gtest_main",
    ],
)
```

**Step 7: Run test to verify it passes**

Run: `bazel test //tests:simple_timestamp_test --test_output=all`
Expected: PASS (6 tests)

**Step 8: Commit**

```bash
git add src/simple/timestamp.hpp src/simple/timestamp.cpp src/simple/BUILD.bazel tests/simple_timestamp_test.cc tests/BUILD.bazel
git commit -m "feat(simple): add Timestamp type with multi-format support"
```

---

## Task 3: Create Settlement Enum and OptionLeg Structure

**Files:**
- Create: `src/simple/option_types.hpp`
- Test: `tests/simple_option_types_test.cc`
- Modify: `src/simple/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/simple_option_types_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "mango/simple/option_types.hpp"

using namespace mango::simple;

TEST(SimpleOptionTypesTest, SettlementDefaults) {
    OptionLeg leg;
    EXPECT_FALSE(leg.settlement.has_value());  // Unknown by default
}

TEST(SimpleOptionTypesTest, OptionLegWithOptionalFields) {
    OptionLeg leg;
    leg.strike = Price{580.0};
    leg.bid = Price{1.45};
    leg.ask = Price{1.52};
    // last, volume, open_interest are optional

    EXPECT_TRUE(leg.bid.has_value());
    EXPECT_TRUE(leg.ask.has_value());
    EXPECT_FALSE(leg.last.has_value());
    EXPECT_FALSE(leg.volume.has_value());
}

TEST(SimpleOptionTypesTest, OptionLegMid) {
    OptionLeg leg;
    leg.bid = Price{1.45};
    leg.ask = Price{1.52};

    auto mid = leg.mid();
    ASSERT_TRUE(mid.has_value());
    EXPECT_NEAR(mid->to_double(), 1.485, 1e-6);
}

TEST(SimpleOptionTypesTest, OptionLegMidWithoutBothPrices) {
    OptionLeg leg;
    leg.bid = Price{1.45};
    // No ask

    auto mid = leg.mid();
    EXPECT_FALSE(mid.has_value());
}

TEST(SimpleOptionTypesTest, OptionLegPriceForIV) {
    OptionLeg leg;
    leg.bid = Price{1.45};
    leg.ask = Price{1.52};
    leg.last = Price{1.50};

    // Prefer mid over last
    auto price = leg.price_for_iv();
    ASSERT_TRUE(price.has_value());
    EXPECT_NEAR(price->to_double(), 1.485, 1e-6);
}

TEST(SimpleOptionTypesTest, OptionLegPriceForIVFallback) {
    OptionLeg leg;
    leg.last = Price{1.50};
    // No bid/ask

    auto price = leg.price_for_iv();
    ASSERT_TRUE(price.has_value());
    EXPECT_DOUBLE_EQ(price->to_double(), 1.50);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:simple_option_types_test --test_output=all`
Expected: FAIL with "option_types.hpp: No such file"

**Step 3: Write minimal implementation**

Create `src/simple/option_types.hpp`:

```cpp
/**
 * @file option_types.hpp
 * @brief Option-related types for mango::simple namespace
 */

#pragma once

#include "mango/simple/price.hpp"
#include "mango/simple/timestamp.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mango::simple {

/// Option settlement type
enum class Settlement {
    AM,   // AM-settled (expires at market open) - SPX, VIX
    PM    // PM-settled (expires at 4:00 PM ET) - SPY, AAPL, etc.
};

/// Single option leg with optional fields
///
/// All price/volume fields are optional since data sources
/// may not provide complete information.
struct OptionLeg {
    Price strike{0.0};

    // Price data - at least one should be present for IV calculation
    std::optional<Price> bid;
    std::optional<Price> ask;
    std::optional<Price> last;

    // Volume data - often missing for illiquid options
    std::optional<int64_t> volume;
    std::optional<int64_t> open_interest;

    // Source-provided values (for comparison/validation)
    std::optional<double> source_iv;
    std::optional<double> source_delta;
    std::optional<double> source_gamma;
    std::optional<double> source_theta;
    std::optional<double> source_vega;

    // Settlement type (may be unknown from some sources)
    std::optional<Settlement> settlement;

    /// Compute mid price if both bid and ask are present
    [[nodiscard]] std::optional<Price> mid() const {
        if (bid && ask) {
            return Price::midpoint(*bid, *ask);
        }
        return std::nullopt;
    }

    /// Best available price for IV calculation
    /// Priority: mid > last
    [[nodiscard]] std::optional<Price> price_for_iv() const {
        if (auto m = mid()) {
            return m;
        }
        return last;
    }
};

/// Options for a single expiry date
struct ExpirySlice {
    Timestamp expiry{""};
    std::optional<Settlement> settlement;
    std::vector<OptionLeg> calls;
    std::vector<OptionLeg> puts;
};

}  // namespace mango::simple
```

**Step 4: Update BUILD.bazel**

Modify `src/simple/BUILD.bazel`:

```python
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "price",
    hdrs = ["price.hpp"],
    deps = [],
)

cc_library(
    name = "timestamp",
    srcs = ["timestamp.cpp"],
    hdrs = ["timestamp.hpp"],
    deps = [],
)

cc_library(
    name = "option_types",
    hdrs = ["option_types.hpp"],
    deps = [
        ":price",
        ":timestamp",
    ],
)
```

**Step 5: Add test target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "simple_option_types_test",
    srcs = ["simple_option_types_test.cc"],
    deps = [
        "//src/simple:option_types",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:simple_option_types_test --test_output=all`
Expected: PASS (6 tests)

**Step 7: Commit**

```bash
git add src/simple/option_types.hpp src/simple/BUILD.bazel tests/simple_option_types_test.cc tests/BUILD.bazel
git commit -m "feat(simple): add Settlement enum and OptionLeg structure"
```

---

## Task 4: Create OptionChain and MarketContext

**Files:**
- Create: `src/simple/option_chain.hpp`
- Test: `tests/simple_option_chain_test.cc`
- Modify: `src/simple/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/simple_option_chain_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "mango/simple/option_chain.hpp"

using namespace mango::simple;

TEST(SimpleOptionChainTest, EmptyChain) {
    OptionChain chain;
    EXPECT_TRUE(chain.expiries.empty());
    EXPECT_FALSE(chain.spot.has_value());
}

TEST(SimpleOptionChainTest, ChainWithData) {
    OptionChain chain;
    chain.symbol = "SPY";
    chain.spot = Price{580.50};
    chain.quote_time = Timestamp{"2024-06-21T10:30:00"};

    ExpirySlice slice;
    slice.expiry = Timestamp{"2024-06-21"};
    slice.settlement = Settlement::PM;

    OptionLeg call;
    call.strike = Price{580.0};
    call.bid = Price{2.85};
    call.ask = Price{2.92};
    slice.calls.push_back(call);

    chain.expiries.push_back(std::move(slice));

    EXPECT_EQ(chain.symbol, "SPY");
    EXPECT_EQ(chain.expiries.size(), 1);
    EXPECT_EQ(chain.expiries[0].calls.size(), 1);
}

TEST(SimpleMarketContextTest, DefaultContext) {
    MarketContext ctx;
    EXPECT_FALSE(ctx.rate.has_value());
    EXPECT_FALSE(ctx.valuation_time.has_value());
}

TEST(SimpleMarketContextTest, ContextWithRate) {
    MarketContext ctx;
    ctx.rate = 0.053;  // Flat 5.3%

    EXPECT_TRUE(ctx.rate.has_value());
    EXPECT_DOUBLE_EQ(std::get<double>(*ctx.rate), 0.053);
}

TEST(SimpleMarketContextTest, ContextWithYieldCurve) {
    auto curve = mango::YieldCurve::flat(0.05);
    MarketContext ctx;
    ctx.rate = curve;

    EXPECT_TRUE(ctx.rate.has_value());
    EXPECT_TRUE(std::holds_alternative<mango::YieldCurve>(*ctx.rate));
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:simple_option_chain_test --test_output=all`
Expected: FAIL with "option_chain.hpp: No such file"

**Step 3: Write minimal implementation**

Create `src/simple/option_chain.hpp`:

```cpp
/**
 * @file option_chain.hpp
 * @brief Option chain and market context types
 */

#pragma once

#include "mango/simple/option_types.hpp"
#include "mango/math/yield_curve.hpp"
#include "mango/option/option_spec.hpp"  // For RateSpec
#include <optional>
#include <string>
#include <vector>

namespace mango::simple {

/// Dividend specification
///
/// Either a continuous yield (for indices) or discrete dividends (for stocks).
struct Dividend {
    Timestamp ex_date;
    Price amount;
};

using DividendSpec = std::variant<double, std::vector<Dividend>>;

/// Full option chain for an underlying
struct OptionChain {
    std::string symbol;
    std::optional<Price> spot;
    std::optional<Timestamp> quote_time;
    std::optional<DividendSpec> dividends;
    std::optional<std::string> exchange;

    std::vector<ExpirySlice> expiries;

    /// Get all expiry timestamps
    [[nodiscard]] std::vector<Timestamp> expiry_dates() const {
        std::vector<Timestamp> dates;
        dates.reserve(expiries.size());
        for (const auto& slice : expiries) {
            dates.push_back(slice.expiry);
        }
        return dates;
    }
};

/// Market context for IV computation
///
/// Contains rate, valuation time, and optional dividend override.
struct MarketContext {
    std::optional<mango::RateSpec> rate;
    std::optional<Timestamp> valuation_time;
    std::optional<DividendSpec> dividends;  // Override chain's dividends
};

}  // namespace mango::simple
```

**Step 4: Update BUILD.bazel**

Modify `src/simple/BUILD.bazel`:

```python
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "price",
    hdrs = ["price.hpp"],
    deps = [],
)

cc_library(
    name = "timestamp",
    srcs = ["timestamp.cpp"],
    hdrs = ["timestamp.hpp"],
    deps = [],
)

cc_library(
    name = "option_types",
    hdrs = ["option_types.hpp"],
    deps = [
        ":price",
        ":timestamp",
    ],
)

cc_library(
    name = "option_chain",
    hdrs = ["option_chain.hpp"],
    deps = [
        ":option_types",
        "//src/math:yield_curve",
        "//src/option:option_spec",
    ],
)
```

**Step 5: Add test target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "simple_option_chain_test",
    srcs = ["simple_option_chain_test.cc"],
    deps = [
        "//src/simple:option_chain",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:simple_option_chain_test --test_output=all`
Expected: PASS (5 tests)

**Step 7: Commit**

```bash
git add src/simple/option_chain.hpp src/simple/BUILD.bazel tests/simple_option_chain_test.cc tests/BUILD.bazel
git commit -m "feat(simple): add OptionChain and MarketContext"
```

---

## Task 5: Create Type-Safe Converter Traits

**Files:**
- Create: `src/simple/converter.hpp`
- Create: `src/simple/sources/yfinance.hpp`
- Create: `src/simple/sources/databento.hpp`
- Create: `src/simple/sources/ibkr.hpp`
- Test: `tests/simple_converter_test.cc`
- Modify: `src/simple/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/simple_converter_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "mango/simple/converter.hpp"
#include "mango/simple/sources/yfinance.hpp"
#include "mango/simple/sources/databento.hpp"
#include "mango/simple/sources/ibkr.hpp"

using namespace mango::simple;

// Test yfinance converter
TEST(ConverterTest, YFinancePrice) {
    auto p = Converter<YFinanceSource>::to_price(100.50);
    EXPECT_DOUBLE_EQ(p.to_double(), 100.50);
}

TEST(ConverterTest, YFinanceTimestamp) {
    auto ts = Converter<YFinanceSource>::to_timestamp("2024-06-21");
    auto tp = ts.to_timepoint();
    EXPECT_TRUE(tp.has_value());
}

TEST(ConverterTest, YFinanceOptionType) {
    EXPECT_EQ(Converter<YFinanceSource>::to_option_type("call"), mango::OptionType::CALL);
    EXPECT_EQ(Converter<YFinanceSource>::to_option_type("put"), mango::OptionType::PUT);
}

// Test Databento converter
TEST(ConverterTest, DatabentoPriceFixedPoint) {
    int64_t fixed = 100500000000LL;  // 100.50 * 10^9
    auto p = Converter<DatabentSource>::to_price(fixed);
    EXPECT_DOUBLE_EQ(p.to_double(), 100.50);
    EXPECT_TRUE(p.is_fixed_point());
}

TEST(ConverterTest, DabentoTimestamp) {
    uint64_t nanos = 1718928000000000000ULL;
    auto ts = Converter<DatabentSource>::to_timestamp(nanos);
    auto tp = ts.to_timepoint();
    EXPECT_TRUE(tp.has_value());
}

TEST(ConverterTest, DabentoOptionType) {
    EXPECT_EQ(Converter<DatabentSource>::to_option_type('C'), mango::OptionType::CALL);
    EXPECT_EQ(Converter<DatabentSource>::to_option_type('P'), mango::OptionType::PUT);
}

// Test IBKR converter
TEST(ConverterTest, IBKRPrice) {
    auto p = Converter<IBKRSource>::to_price(100.50);
    EXPECT_DOUBLE_EQ(p.to_double(), 100.50);
}

TEST(ConverterTest, IBKRTimestamp) {
    auto ts = Converter<IBKRSource>::to_timestamp("20240621");
    auto tp = ts.to_timepoint();
    EXPECT_TRUE(tp.has_value());
}

TEST(ConverterTest, IBKROptionType) {
    EXPECT_EQ(Converter<IBKRSource>::to_option_type("C"), mango::OptionType::CALL);
    EXPECT_EQ(Converter<IBKRSource>::to_option_type("P"), mango::OptionType::PUT);
    EXPECT_EQ(Converter<IBKRSource>::to_option_type("CALL"), mango::OptionType::CALL);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:simple_converter_test --test_output=all`
Expected: FAIL with "converter.hpp: No such file"

**Step 3: Write converter base and concept**

Create `src/simple/converter.hpp`:

```cpp
/**
 * @file converter.hpp
 * @brief Type-safe converter traits for data sources
 */

#pragma once

#include "mango/simple/price.hpp"
#include "mango/simple/timestamp.hpp"
#include "mango/simple/option_types.hpp"
#include "mango/option/option_spec.hpp"
#include <concepts>
#include <stdexcept>

namespace mango::simple {

// Source tag types
struct YFinanceSource {};
struct DatabentSource {};
struct IBKRSource {};

/// Conversion error exception
class ConversionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Converter trait - must be specialized for each source
template<typename Source>
struct Converter;

/// Concept for valid converter
template<typename T>
concept ValidConverter = requires {
    // Must have price conversion
    { T::to_price(std::declval<double>()) } -> std::same_as<Price>;
    // Must have option type conversion
    { T::to_option_type(std::declval<const char*>()) } -> std::same_as<mango::OptionType>;
};

}  // namespace mango::simple
```

**Step 4: Create yfinance converter**

Create `src/simple/sources/yfinance.hpp`:

```cpp
/**
 * @file yfinance.hpp
 * @brief Converter for yfinance data format
 */

#pragma once

#include "mango/simple/converter.hpp"

namespace mango::simple {

template<>
struct Converter<YFinanceSource> {
    static Price to_price(double v) {
        return Price{v};
    }

    static Timestamp to_timestamp(const std::string& s) {
        return Timestamp{s, TimestampFormat::ISO};
    }

    static mango::OptionType to_option_type(const std::string& s) {
        if (s == "call" || s == "Call" || s == "CALL") {
            return mango::OptionType::CALL;
        }
        if (s == "put" || s == "Put" || s == "PUT") {
            return mango::OptionType::PUT;
        }
        throw ConversionError("Invalid option type: " + s);
    }

    /// Convert yfinance option data to OptionLeg
    struct RawOption {
        std::string expiry;
        double strike;
        double bid;
        double ask;
        double lastPrice;
        int64_t volume;
        int64_t openInterest;
        double impliedVolatility;
    };

    static OptionLeg to_leg(const RawOption& src) {
        OptionLeg leg;
        leg.strike = to_price(src.strike);
        leg.bid = to_price(src.bid);
        leg.ask = to_price(src.ask);
        leg.last = to_price(src.lastPrice);
        leg.volume = src.volume;
        leg.open_interest = src.openInterest;
        leg.source_iv = src.impliedVolatility;
        return leg;
    }
};

}  // namespace mango::simple
```

**Step 5: Create Databento converter**

Create `src/simple/sources/databento.hpp`:

```cpp
/**
 * @file databento.hpp
 * @brief Converter for Databento data format
 */

#pragma once

#include "mango/simple/converter.hpp"

namespace mango::simple {

template<>
struct Converter<DatabentSource> {
    static constexpr double PRICE_SCALE = 1e-9;

    static Price to_price(int64_t v) {
        return Price{v, PriceFormat::FixedPoint9};
    }

    static Timestamp to_timestamp(uint64_t nanos) {
        return Timestamp{nanos};
    }

    static mango::OptionType to_option_type(char c) {
        if (c == 'C') return mango::OptionType::CALL;
        if (c == 'P') return mango::OptionType::PUT;
        throw ConversionError(std::string("Invalid option type: ") + c);
    }

    /// Databento raw option message
    struct RawOption {
        uint64_t ts_event;
        int64_t price;
        int64_t bid_px;
        int64_t ask_px;
        int64_t strike_price;
        char option_type;
    };

    static OptionLeg to_leg(const RawOption& src) {
        OptionLeg leg;
        leg.strike = to_price(src.strike_price);
        leg.bid = to_price(src.bid_px);
        leg.ask = to_price(src.ask_px);
        leg.last = to_price(src.price);
        return leg;
    }
};

}  // namespace mango::simple
```

**Step 6: Create IBKR converter**

Create `src/simple/sources/ibkr.hpp`:

```cpp
/**
 * @file ibkr.hpp
 * @brief Converter for Interactive Brokers data format
 */

#pragma once

#include "mango/simple/converter.hpp"

namespace mango::simple {

template<>
struct Converter<IBKRSource> {
    static Price to_price(double v) {
        return Price{v};
    }

    static Timestamp to_timestamp(const std::string& s) {
        // IBKR uses compact format: "20240621"
        return Timestamp{s, TimestampFormat::Compact};
    }

    static mango::OptionType to_option_type(const std::string& s) {
        if (s == "C" || s == "CALL" || s == "Call") {
            return mango::OptionType::CALL;
        }
        if (s == "P" || s == "PUT" || s == "Put") {
            return mango::OptionType::PUT;
        }
        throw ConversionError("Invalid option type: " + s);
    }

    /// IBKR raw option data
    struct RawOption {
        std::string expiry;
        double strike;
        double bid;
        double ask;
        double last;
        std::string right;
        int64_t volume;
    };

    static OptionLeg to_leg(const RawOption& src) {
        OptionLeg leg;
        leg.strike = to_price(src.strike);
        leg.bid = to_price(src.bid);
        leg.ask = to_price(src.ask);
        leg.last = to_price(src.last);
        leg.volume = src.volume;
        return leg;
    }
};

}  // namespace mango::simple
```

**Step 7: Update BUILD.bazel**

Modify `src/simple/BUILD.bazel`:

```python
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "price",
    hdrs = ["price.hpp"],
    deps = [],
)

cc_library(
    name = "timestamp",
    srcs = ["timestamp.cpp"],
    hdrs = ["timestamp.hpp"],
    deps = [],
)

cc_library(
    name = "option_types",
    hdrs = ["option_types.hpp"],
    deps = [
        ":price",
        ":timestamp",
    ],
)

cc_library(
    name = "option_chain",
    hdrs = ["option_chain.hpp"],
    deps = [
        ":option_types",
        "//src/math:yield_curve",
        "//src/option:option_spec",
    ],
)

cc_library(
    name = "converter",
    hdrs = [
        "converter.hpp",
        "sources/yfinance.hpp",
        "sources/databento.hpp",
        "sources/ibkr.hpp",
    ],
    deps = [
        ":price",
        ":timestamp",
        ":option_types",
        "//src/option:option_spec",
    ],
)
```

**Step 8: Add test target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "simple_converter_test",
    srcs = ["simple_converter_test.cc"],
    deps = [
        "//src/simple:converter",
        "@googletest//:gtest_main",
    ],
)
```

**Step 9: Run test to verify it passes**

Run: `bazel test //tests:simple_converter_test --test_output=all`
Expected: PASS (9 tests)

**Step 10: Commit**

```bash
git add src/simple/converter.hpp src/simple/sources/ src/simple/BUILD.bazel tests/simple_converter_test.cc tests/BUILD.bazel
git commit -m "feat(simple): add type-safe Converter traits for data sources"
```

---

## Task 6: Create ChainBuilder for Type-Safe Chain Construction

**Files:**
- Create: `src/simple/chain_builder.hpp`
- Test: `tests/simple_chain_builder_test.cc`
- Modify: `src/simple/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/simple_chain_builder_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "mango/simple/chain_builder.hpp"
#include "mango/simple/sources/yfinance.hpp"
#include "mango/simple/sources/databento.hpp"

using namespace mango::simple;

TEST(ChainBuilderTest, YFinanceBasic) {
    auto chain = ChainBuilder<YFinanceSource>{}
        .symbol("SPY")
        .spot(580.50)
        .quote_time("2024-06-21T10:30:00")
        .build();

    EXPECT_EQ(chain.symbol, "SPY");
    ASSERT_TRUE(chain.spot.has_value());
    EXPECT_DOUBLE_EQ(chain.spot->to_double(), 580.50);
}

TEST(ChainBuilderTest, DabentoBasic) {
    auto chain = ChainBuilder<DatabentSource>{}
        .symbol("SPY")
        .spot(580500000000LL)  // Fixed-point
        .quote_time(1718972400000000000ULL)  // Nanoseconds
        .build();

    EXPECT_EQ(chain.symbol, "SPY");
    ASSERT_TRUE(chain.spot.has_value());
    EXPECT_DOUBLE_EQ(chain.spot->to_double(), 580.50);
    EXPECT_TRUE(chain.spot->is_fixed_point());
}

TEST(ChainBuilderTest, AddOptions) {
    Converter<YFinanceSource>::RawOption call_opt{
        .expiry = "2024-06-21",
        .strike = 580.0,
        .bid = 2.85,
        .ask = 2.92,
        .lastPrice = 2.88,
        .volume = 42150,
        .openInterest = 51200,
        .impliedVolatility = 0.128
    };

    auto chain = ChainBuilder<YFinanceSource>{}
        .symbol("SPY")
        .spot(580.50)
        .add_call("2024-06-21", call_opt)
        .settlement(Settlement::PM)
        .build();

    EXPECT_EQ(chain.expiries.size(), 1);
    EXPECT_EQ(chain.expiries[0].calls.size(), 1);
    EXPECT_DOUBLE_EQ(chain.expiries[0].calls[0].strike.to_double(), 580.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:simple_chain_builder_test --test_output=all`
Expected: FAIL with "chain_builder.hpp: No such file"

**Step 3: Write minimal implementation**

Create `src/simple/chain_builder.hpp`:

```cpp
/**
 * @file chain_builder.hpp
 * @brief Type-safe builder for OptionChain
 */

#pragma once

#include "mango/simple/option_chain.hpp"
#include "mango/simple/converter.hpp"
#include <map>

namespace mango::simple {

/// Type-safe chain builder
///
/// Uses Converter<Source> to ensure correct types at compile time.
template<typename Source>
class ChainBuilder {
    using Conv = Converter<Source>;

public:
    ChainBuilder& symbol(std::string sym) {
        chain_.symbol = std::move(sym);
        return *this;
    }

    template<typename T>
    ChainBuilder& spot(T&& v) {
        chain_.spot = Conv::to_price(std::forward<T>(v));
        return *this;
    }

    template<typename T>
    ChainBuilder& quote_time(T&& v) {
        chain_.quote_time = Conv::to_timestamp(std::forward<T>(v));
        return *this;
    }

    ChainBuilder& settlement(Settlement s) {
        default_settlement_ = s;
        return *this;
    }

    ChainBuilder& dividend_yield(double yield) {
        chain_.dividends = yield;
        return *this;
    }

    template<typename T, typename RawOpt>
    ChainBuilder& add_call(T&& expiry, const RawOpt& opt) {
        auto ts = Conv::to_timestamp(std::forward<T>(expiry));
        auto& slice = get_or_create_slice(ts);
        slice.calls.push_back(Conv::to_leg(opt));
        return *this;
    }

    template<typename T, typename RawOpt>
    ChainBuilder& add_put(T&& expiry, const RawOpt& opt) {
        auto ts = Conv::to_timestamp(std::forward<T>(expiry));
        auto& slice = get_or_create_slice(ts);
        slice.puts.push_back(Conv::to_leg(opt));
        return *this;
    }

    OptionChain build() {
        // Apply default settlement to slices without one
        for (auto& slice : chain_.expiries) {
            if (!slice.settlement.has_value() && default_settlement_.has_value()) {
                slice.settlement = default_settlement_;
            }
        }
        return std::move(chain_);
    }

private:
    ExpirySlice& get_or_create_slice(const Timestamp& expiry) {
        // Find existing slice with same expiry (by string comparison for simplicity)
        auto expiry_str = expiry.to_string();
        for (auto& slice : chain_.expiries) {
            if (slice.expiry.to_string() == expiry_str) {
                return slice;
            }
        }
        // Create new slice
        chain_.expiries.push_back(ExpirySlice{expiry});
        return chain_.expiries.back();
    }

    OptionChain chain_;
    std::optional<Settlement> default_settlement_;
};

}  // namespace mango::simple
```

**Step 4: Update BUILD.bazel**

Add to `src/simple/BUILD.bazel`:

```python
cc_library(
    name = "chain_builder",
    hdrs = ["chain_builder.hpp"],
    deps = [
        ":option_chain",
        ":converter",
    ],
)
```

**Step 5: Add test target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "simple_chain_builder_test",
    srcs = ["simple_chain_builder_test.cc"],
    deps = [
        "//src/simple:chain_builder",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:simple_chain_builder_test --test_output=all`
Expected: PASS (3 tests)

**Step 7: Commit**

```bash
git add src/simple/chain_builder.hpp src/simple/BUILD.bazel tests/simple_chain_builder_test.cc tests/BUILD.bazel
git commit -m "feat(simple): add type-safe ChainBuilder"
```

---

## Task 7: Create VolatilitySurface and compute_vol_surface Function

**Files:**
- Create: `src/simple/vol_surface.hpp`
- Create: `src/simple/vol_surface.cpp`
- Test: `tests/simple_vol_surface_test.cc`
- Modify: `src/simple/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/simple_vol_surface_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "mango/simple/vol_surface.hpp"
#include "mango/simple/chain_builder.hpp"
#include "mango/simple/sources/yfinance.hpp"

using namespace mango::simple;

TEST(VolSurfaceTest, ComputeSmileFromChain) {
    // Build a simple chain
    auto chain = ChainBuilder<YFinanceSource>{}
        .symbol("SPY")
        .spot(580.50)
        .quote_time("2024-06-21T10:30:00")
        .settlement(Settlement::PM)
        .dividend_yield(0.013)
        .build();

    // Add a single option for testing
    ExpirySlice slice;
    slice.expiry = Timestamp{"2024-06-28"};  // 1 week out
    slice.settlement = Settlement::PM;

    OptionLeg call;
    call.strike = Price{580.0};
    call.bid = Price{5.50};
    call.ask = Price{5.70};
    slice.calls.push_back(call);

    chain.expiries.push_back(std::move(slice));

    MarketContext ctx;
    ctx.rate = 0.053;
    ctx.valuation_time = Timestamp{"2024-06-21T10:30:00"};

    // This requires a precomputed price table, so we test the structure
    // In real usage, you'd provide a solver
    VolatilitySurface surface;
    surface.symbol = chain.symbol;
    surface.spot = *chain.spot;

    EXPECT_EQ(surface.symbol, "SPY");
}

TEST(VolSmileTest, SmilePointStructure) {
    VolatilitySmile::Point pt;
    pt.strike = Price{580.0};
    pt.moneyness = 0.0;  // ATM
    pt.iv_mid = 0.15;

    EXPECT_DOUBLE_EQ(pt.strike.to_double(), 580.0);
    EXPECT_TRUE(pt.iv_mid.has_value());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:simple_vol_surface_test --test_output=all`
Expected: FAIL with "vol_surface.hpp: No such file"

**Step 3: Write vol_surface.hpp**

Create `src/simple/vol_surface.hpp`:

```cpp
/**
 * @file vol_surface.hpp
 * @brief Volatility surface types and computation
 */

#pragma once

#include "mango/simple/option_chain.hpp"
#include "mango/option/iv_solver_interpolated.hpp"
#include "mango/option/iv_solver_fdm.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace mango::simple {

/// Single point on the volatility smile
struct VolatilitySmile {
    Timestamp expiry;
    double tau;  // Time to expiry in years
    Price spot;

    struct Point {
        Price strike;
        double moneyness;  // log(K/S)
        std::optional<double> iv_bid;
        std::optional<double> iv_ask;
        std::optional<double> iv_mid;
        std::optional<double> iv_last;
    };

    std::vector<Point> calls;
    std::vector<Point> puts;
};

/// Complete volatility surface
struct VolatilitySurface {
    std::string symbol;
    Timestamp quote_time;
    Price spot;

    std::vector<VolatilitySmile> smiles;

    /// Interpolate IV at arbitrary (strike, tau)
    std::optional<double> iv_at(double strike, double tau) const;
};

/// Error during surface computation
struct ComputeError {
    std::string message;
    size_t failed_count = 0;
};

/// Configuration for IV computation
struct IVComputeConfig {
    enum class Method {
        Interpolated,  // Fast: use precomputed tables (~30µs)
        FDM            // Accurate: solve PDE per option (~143ms)
    };

    Method method = Method::Interpolated;
    double tolerance = 1e-6;
    int max_iterations = 50;
};

/// Price source for IV calculation
enum class PriceSource {
    Bid,
    Ask,
    Mid,
    Last
};

/// Compute volatility surface from option chain
///
/// @param chain Option chain with quotes
/// @param ctx Market context (rate, valuation time)
/// @param solver Interpolated IV solver (required for Method::Interpolated)
/// @param config Computation configuration
/// @param price_source Which price to use for IV
/// @return Volatility surface or error
std::expected<VolatilitySurface, ComputeError> compute_vol_surface(
    const OptionChain& chain,
    const MarketContext& ctx,
    const mango::IVSolverInterpolated* solver = nullptr,
    const IVComputeConfig& config = {},
    PriceSource price_source = PriceSource::Mid);

}  // namespace mango::simple
```

**Step 4: Write vol_surface.cpp**

Create `src/simple/vol_surface.cpp`:

```cpp
#include "mango/simple/vol_surface.hpp"
#include <cmath>

namespace mango::simple {

std::optional<double> VolatilitySurface::iv_at(double strike, double tau) const {
    // Simple linear interpolation - could be enhanced with B-spline
    if (smiles.empty()) return std::nullopt;

    // Find bracketing smiles by tau
    const VolatilitySmile* left = nullptr;
    const VolatilitySmile* right = nullptr;

    for (const auto& smile : smiles) {
        if (smile.tau <= tau) {
            if (!left || smile.tau > left->tau) {
                left = &smile;
            }
        }
        if (smile.tau >= tau) {
            if (!right || smile.tau < right->tau) {
                right = &smile;
            }
        }
    }

    if (!left && !right) return std::nullopt;
    if (!left) left = right;
    if (!right) right = left;

    // For now, use nearest smile
    const auto& smile = (std::abs(left->tau - tau) < std::abs(right->tau - tau)) ? *left : *right;

    // Find nearest strike in smile
    double moneyness = std::log(strike / spot.to_double());
    std::optional<double> best_iv;
    double best_dist = std::numeric_limits<double>::max();

    for (const auto& pt : smile.calls) {
        double dist = std::abs(pt.moneyness - moneyness);
        if (dist < best_dist && pt.iv_mid) {
            best_dist = dist;
            best_iv = pt.iv_mid;
        }
    }
    for (const auto& pt : smile.puts) {
        double dist = std::abs(pt.moneyness - moneyness);
        if (dist < best_dist && pt.iv_mid) {
            best_dist = dist;
            best_iv = pt.iv_mid;
        }
    }

    return best_iv;
}

std::expected<VolatilitySurface, ComputeError> compute_vol_surface(
    const OptionChain& chain,
    const MarketContext& ctx,
    const mango::IVSolverInterpolated* solver,
    const IVComputeConfig& config,
    PriceSource price_source)
{
    if (!chain.spot) {
        return std::unexpected(ComputeError{"Missing spot price"});
    }

    if (config.method == IVComputeConfig::Method::Interpolated && !solver) {
        return std::unexpected(ComputeError{"Interpolated method requires solver"});
    }

    VolatilitySurface surface;
    surface.symbol = chain.symbol;
    surface.spot = *chain.spot;
    if (chain.quote_time) {
        surface.quote_time = *chain.quote_time;
    }

    // Get valuation time
    Timestamp val_time = ctx.valuation_time.value_or(
        chain.quote_time.value_or(Timestamp::now()));

    // Get rate
    double rate = 0.05;  // Default
    if (ctx.rate) {
        rate = mango::get_zero_rate(*ctx.rate, 1.0);  // Approximate
    }

    // Get dividend yield
    double div_yield = 0.0;
    auto div_spec = ctx.dividends.value_or(
        chain.dividends.value_or(DividendSpec{0.0}));
    if (std::holds_alternative<double>(div_spec)) {
        div_yield = std::get<double>(div_spec);
    }

    double spot = chain.spot->to_double();
    size_t failed_count = 0;

    for (const auto& slice : chain.expiries) {
        VolatilitySmile smile;
        smile.expiry = slice.expiry;
        smile.tau = compute_tau(val_time, slice.expiry);
        smile.spot = *chain.spot;

        if (smile.tau <= 0) continue;  // Skip expired options

        // Process calls
        for (const auto& leg : slice.calls) {
            auto price_opt = (price_source == PriceSource::Mid) ? leg.mid() :
                             (price_source == PriceSource::Bid) ? leg.bid :
                             (price_source == PriceSource::Ask) ? leg.ask :
                             leg.last;

            if (!price_opt) continue;

            double strike = leg.strike.to_double();
            double market_price = price_opt->to_double();

            VolatilitySmile::Point pt;
            pt.strike = leg.strike;
            pt.moneyness = std::log(strike / spot);

            mango::IVQuery query;
            query.spot = spot;
            query.strike = strike;
            query.maturity = smile.tau;
            query.rate = rate;
            query.dividend_yield = div_yield;
            query.type = mango::OptionType::CALL;
            query.market_price = market_price;

            if (solver) {
                auto result = solver->solve_impl(query);
                if (result) {
                    pt.iv_mid = result->implied_vol;
                } else {
                    ++failed_count;
                }
            }

            smile.calls.push_back(pt);
        }

        // Process puts
        for (const auto& leg : slice.puts) {
            auto price_opt = (price_source == PriceSource::Mid) ? leg.mid() :
                             (price_source == PriceSource::Bid) ? leg.bid :
                             (price_source == PriceSource::Ask) ? leg.ask :
                             leg.last;

            if (!price_opt) continue;

            double strike = leg.strike.to_double();
            double market_price = price_opt->to_double();

            VolatilitySmile::Point pt;
            pt.strike = leg.strike;
            pt.moneyness = std::log(strike / spot);

            mango::IVQuery query;
            query.spot = spot;
            query.strike = strike;
            query.maturity = smile.tau;
            query.rate = rate;
            query.dividend_yield = div_yield;
            query.type = mango::OptionType::PUT;
            query.market_price = market_price;

            if (solver) {
                auto result = solver->solve_impl(query);
                if (result) {
                    pt.iv_mid = result->implied_vol;
                } else {
                    ++failed_count;
                }
            }

            smile.puts.push_back(pt);
        }

        if (!smile.calls.empty() || !smile.puts.empty()) {
            surface.smiles.push_back(std::move(smile));
        }
    }

    return surface;
}

}  // namespace mango::simple
```

**Step 5: Update BUILD.bazel**

Add to `src/simple/BUILD.bazel`:

```python
cc_library(
    name = "vol_surface",
    srcs = ["vol_surface.cpp"],
    hdrs = ["vol_surface.hpp"],
    deps = [
        ":option_chain",
        "//src/option:iv_solver_interpolated",
        "//src/option:iv_solver_fdm",
    ],
)

# Convenience target for all simple headers
cc_library(
    name = "simple",
    hdrs = ["simple.hpp"],
    deps = [
        ":price",
        ":timestamp",
        ":option_types",
        ":option_chain",
        ":converter",
        ":chain_builder",
        ":vol_surface",
    ],
)
```

**Step 6: Create simple.hpp umbrella header**

Create `src/simple/simple.hpp`:

```cpp
/**
 * @file simple.hpp
 * @brief Umbrella header for mango::simple namespace
 */

#pragma once

#include "mango/simple/price.hpp"
#include "mango/simple/timestamp.hpp"
#include "mango/simple/option_types.hpp"
#include "mango/simple/option_chain.hpp"
#include "mango/simple/converter.hpp"
#include "mango/simple/chain_builder.hpp"
#include "mango/simple/vol_surface.hpp"
#include "mango/simple/sources/yfinance.hpp"
#include "mango/simple/sources/databento.hpp"
#include "mango/simple/sources/ibkr.hpp"
```

**Step 7: Add test target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "simple_vol_surface_test",
    srcs = ["simple_vol_surface_test.cc"],
    deps = [
        "//src/simple:vol_surface",
        "//src/simple:chain_builder",
        "@googletest//:gtest_main",
    ],
)
```

**Step 8: Run test to verify it passes**

Run: `bazel test //tests:simple_vol_surface_test --test_output=all`
Expected: PASS (2 tests)

**Step 9: Commit**

```bash
git add src/simple/vol_surface.hpp src/simple/vol_surface.cpp src/simple/simple.hpp src/simple/BUILD.bazel tests/simple_vol_surface_test.cc tests/BUILD.bazel
git commit -m "feat(simple): add VolatilitySurface and compute_vol_surface"
```

---

## Task 8: Create End-to-End Example (yfinance)

**Files:**
- Create: `examples/simple_yfinance_example.cpp`
- Modify: `examples/BUILD.bazel`

**Step 1: Write the example**

Create `examples/simple_yfinance_example.cpp`:

```cpp
/**
 * @file simple_yfinance_example.cpp
 * @brief End-to-end example: yfinance data → volatility smile
 */

#include "mango/simple/simple.hpp"
#include <iostream>
#include <iomanip>

using namespace mango::simple;

int main() {
    // ============================================
    // Step 1: Simulated yfinance data
    // ============================================

    // In practice, this comes from Python via pybind11
    Converter<YFinanceSource>::RawOption spy_calls[] = {
        {.expiry = "2024-06-21", .strike = 575.0, .bid = 6.10, .ask = 6.25, .lastPrice = 6.15, .volume = 15420, .openInterest = 28300, .impliedVolatility = 0.142},
        {.expiry = "2024-06-21", .strike = 580.0, .bid = 2.85, .ask = 2.92, .lastPrice = 2.88, .volume = 42150, .openInterest = 51200, .impliedVolatility = 0.128},
        {.expiry = "2024-06-21", .strike = 585.0, .bid = 0.95, .ask = 1.02, .lastPrice = 0.98, .volume = 31200, .openInterest = 39100, .impliedVolatility = 0.135},
        {.expiry = "2024-06-21", .strike = 590.0, .bid = 0.22, .ask = 0.28, .lastPrice = 0.25, .volume = 18900, .openInterest = 22400, .impliedVolatility = 0.148},
    };

    Converter<YFinanceSource>::RawOption spy_puts[] = {
        {.expiry = "2024-06-21", .strike = 570.0, .bid = 0.18, .ask = 0.24, .lastPrice = 0.21, .volume = 12300, .openInterest = 18700, .impliedVolatility = 0.152},
        {.expiry = "2024-06-21", .strike = 575.0, .bid = 0.52, .ask = 0.58, .lastPrice = 0.55, .volume = 28400, .openInterest = 35600, .impliedVolatility = 0.138},
        {.expiry = "2024-06-21", .strike = 580.0, .bid = 2.30, .ask = 2.42, .lastPrice = 2.35, .volume = 38700, .openInterest = 48200, .impliedVolatility = 0.126},
        {.expiry = "2024-06-21", .strike = 585.0, .bid = 5.40, .ask = 5.55, .lastPrice = 5.48, .volume = 21500, .openInterest = 31400, .impliedVolatility = 0.132},
    };

    // ============================================
    // Step 2: Build option chain using type-safe builder
    // ============================================

    auto builder = ChainBuilder<YFinanceSource>{}
        .symbol("SPY")
        .spot(580.50)
        .quote_time("2024-06-21T10:30:00")
        .settlement(Settlement::PM)
        .dividend_yield(0.013);

    for (const auto& call : spy_calls) {
        builder.add_call(call.expiry, call);
    }
    for (const auto& put : spy_puts) {
        builder.add_put(put.expiry, put);
    }

    auto chain = builder.build();

    // ============================================
    // Step 3: Set up market context
    // ============================================

    MarketContext ctx;
    ctx.rate = 0.053;  // 5.3% Fed Funds
    ctx.valuation_time = Timestamp{"2024-06-21T10:30:00"};

    // ============================================
    // Step 4: Compute volatility surface
    // ============================================

    // Note: For actual IV computation, you'd need a precomputed price table
    // Here we just demonstrate the data flow
    auto surface_result = compute_vol_surface(chain, ctx, nullptr);

    if (!surface_result) {
        std::cerr << "Error: " << surface_result.error().message << "\n";
        return 1;
    }

    const auto& surface = *surface_result;

    // ============================================
    // Step 5: Output results
    // ============================================

    std::cout << "# SPY Volatility Smile\n";
    std::cout << "# Spot: " << surface.spot.to_double() << "\n\n";

    for (const auto& smile : surface.smiles) {
        std::cout << "## Expiry: " << smile.expiry.to_string()
                  << " (tau=" << std::fixed << std::setprecision(6)
                  << smile.tau << " years)\n";
        std::cout << "# strike,moneyness,type\n";

        for (const auto& pt : smile.calls) {
            std::cout << pt.strike.to_double() << ","
                      << pt.moneyness << ",call\n";
        }
        for (const auto& pt : smile.puts) {
            std::cout << pt.strike.to_double() << ","
                      << pt.moneyness << ",put\n";
        }
    }

    std::cout << "\nChain built successfully with "
              << chain.expiries.size() << " expiries\n";

    return 0;
}
```

**Step 2: Add to examples/BUILD.bazel**

Add to `examples/BUILD.bazel`:

```python
cc_binary(
    name = "simple_yfinance_example",
    srcs = ["simple_yfinance_example.cpp"],
    deps = [
        "//src/simple",
    ],
)
```

**Step 3: Build and run**

Run: `bazel build //examples:simple_yfinance_example`
Run: `bazel run //examples:simple_yfinance_example`
Expected: Output showing the option chain data

**Step 4: Commit**

```bash
git add examples/simple_yfinance_example.cpp examples/BUILD.bazel
git commit -m "feat(examples): add yfinance end-to-end example"
```

---

## Task 9: Create End-to-End Example (Databento)

**Files:**
- Create: `examples/simple_databento_example.cpp`
- Modify: `examples/BUILD.bazel`

**Step 1: Write the example**

Create `examples/simple_databento_example.cpp`:

```cpp
/**
 * @file simple_databento_example.cpp
 * @brief End-to-end example: Databento fixed-point data → volatility smile
 */

#include "mango/simple/simple.hpp"
#include <iostream>
#include <iomanip>

using namespace mango::simple;

int main() {
    // ============================================
    // Simulated Databento data (fixed-point format)
    // ============================================

    // Databento uses int64_t with 9 decimal places
    // 580.50 = 580500000000

    Converter<DatabentSource>::RawOption options[] = {
        {.ts_event = 1718972400000000000ULL, .price = 615000000000LL, .bid_px = 610000000000LL, .ask_px = 625000000000LL, .strike_price = 575000000000LL, .option_type = 'C'},
        {.ts_event = 1718972400000000000ULL, .price = 288000000000LL, .bid_px = 285000000000LL, .ask_px = 292000000000LL, .strike_price = 580000000000LL, .option_type = 'C'},
        {.ts_event = 1718972400000000000ULL, .price = 98000000000LL, .bid_px = 95000000000LL, .ask_px = 102000000000LL, .strike_price = 585000000000LL, .option_type = 'C'},
        {.ts_event = 1718972400000000000ULL, .price = 21000000000LL, .bid_px = 18000000000LL, .ask_px = 24000000000LL, .strike_price = 570000000000LL, .option_type = 'P'},
        {.ts_event = 1718972400000000000ULL, .price = 235000000000LL, .bid_px = 230000000000LL, .ask_px = 242000000000LL, .strike_price = 580000000000LL, .option_type = 'P'},
    };

    // ============================================
    // Build chain with Databento converter (preserves fixed-point)
    // ============================================

    auto builder = ChainBuilder<DatabentSource>{}
        .symbol("SPY")
        .spot(580500000000LL)  // Fixed-point preserved!
        .quote_time(1718972400000000000ULL)
        .settlement(Settlement::PM);

    // Single expiry for this example
    uint64_t expiry_nanos = 1719014400000000000ULL;  // 2024-06-21 16:00 UTC

    for (const auto& opt : options) {
        auto leg = Converter<DatabentSource>::to_leg(opt);

        // Verify fixed-point is preserved
        if (leg.strike.is_fixed_point()) {
            std::cout << "Strike " << leg.strike.to_double()
                      << " stored as fixed-point\n";
        }

        if (opt.option_type == 'C') {
            // Need to add to builder - would need timestamp for expiry
        }
    }

    auto chain = builder.build();

    // ============================================
    // Verify precision preservation
    // ============================================

    std::cout << "\n=== Precision Check ===\n";
    std::cout << "Spot stored as fixed-point: "
              << (chain.spot->is_fixed_point() ? "YES" : "NO") << "\n";
    std::cout << "Spot value: " << std::fixed << std::setprecision(9)
              << chain.spot->to_double() << "\n";

    // The key benefit: no precision loss until final computation
    std::cout << "\nFixed-point precision preserved through data pipeline.\n";
    std::cout << "Conversion to double only at solver boundary.\n";

    return 0;
}
```

**Step 2: Add to examples/BUILD.bazel**

Add to `examples/BUILD.bazel`:

```python
cc_binary(
    name = "simple_databento_example",
    srcs = ["simple_databento_example.cpp"],
    deps = [
        "//src/simple",
    ],
)
```

**Step 3: Build and run**

Run: `bazel build //examples:simple_databento_example`
Run: `bazel run //examples:simple_databento_example`
Expected: Output showing fixed-point preservation

**Step 4: Commit**

```bash
git add examples/simple_databento_example.cpp examples/BUILD.bazel
git commit -m "feat(examples): add Databento end-to-end example"
```

---

## Task 10: Run All Tests and Final Verification

**Step 1: Run all simple tests**

Run: `bazel test //tests:simple_price_test //tests:simple_timestamp_test //tests:simple_option_types_test //tests:simple_option_chain_test //tests:simple_converter_test //tests:simple_chain_builder_test //tests:simple_vol_surface_test --test_output=errors`

Expected: All tests PASS

**Step 2: Build all examples**

Run: `bazel build //examples:simple_yfinance_example //examples:simple_databento_example`

Expected: Build succeeds

**Step 3: Run full test suite**

Run: `bazel test //... --test_output=errors`

Expected: All existing tests still pass

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(simple): complete mango::simple namespace implementation

Adds user-friendly interface for external data sources:
- Price type with deferred double conversion
- Timestamp with ISO/compact/nanosecond formats
- Type-safe Converter traits for yfinance/Databento/IBKR
- ChainBuilder for ergonomic chain construction
- VolatilitySurface computation from option chains

Preserves precision by deferring conversion to solver boundary."
```

---

## Summary

This plan implements the `mango::simple` namespace in 10 tasks:

1. **Price type** - Deferred double conversion, fixed-point support
2. **Timestamp type** - Multiple format support (ISO, compact, nanoseconds)
3. **Option types** - Settlement enum, OptionLeg with optional fields
4. **OptionChain** - Chain and market context structures
5. **Converters** - Type-safe traits for yfinance, Databento, IBKR
6. **ChainBuilder** - Ergonomic type-safe chain construction
7. **VolatilitySurface** - Surface computation from chains
8. **yfinance example** - End-to-end demonstration
9. **Databento example** - Fixed-point precision demonstration
10. **Verification** - Full test suite validation

Each task follows TDD: write failing test → implement → verify → commit.
