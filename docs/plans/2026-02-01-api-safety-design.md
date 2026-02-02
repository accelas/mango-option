# API Safety: Prevent Silent Wrong Results

Four changes that make the API reject misuse at compile time or with
clear runtime errors, instead of returning wrong numbers silently.

## 1. Add `option_type()` to PriceSurface concept

**Problem:** `IVSolverInterpolated::solve()` accepts any `IVQuery`
regardless of option type. A CALL query against a PUT surface returns
a wrong IV with no error.

**Fix:**

Add `option_type()` to the `PriceSurface` concept:

```cpp
// price_surface_concept.hpp
template <typename S>
concept PriceSurface = requires(const S& s, double spot, double strike,
                                double tau, double sigma, double rate) {
    // ... existing requirements ...
    { s.option_type() } -> std::same_as<OptionType>;
};
```

Add the accessor to both implementations:
- `AmericanPriceSurface::option_type()` — returns stored `type_`
- `SegmentedPriceSurface::option_type()` — returns first segment's type

Validate in `IVSolverInterpolated::solve()`:

```cpp
if (query.type != surface_.option_type()) {
    return std::unexpected(IVError{
        .code = IVErrorCode::InvalidGridConfig,
        .iterations = 0, .final_error = 0.0, .last_vol = std::nullopt});
}
```

**Files:** `price_surface_concept.hpp`, `american_price_surface.hpp`,
`american_price_surface.cpp`, `segmented_price_surface.hpp`,
`segmented_price_surface.cpp`, `iv_solver_interpolated.hpp`

## 2. Expose `dividend_yield` on AmericanPriceSurface

**Problem:** `AmericanPriceSurface::price()` bakes in the construction-time
`dividend_yield` but accepts no yield parameter. A caller with a different
yield gets wrong prices silently.

**Fix:**

Add `dividend_yield()` accessor to `AmericanPriceSurface` (and the
`PriceSurface` concept). The IV solver already extracts `dividend_yield`
from the query's `OptionSpec`. Validate consistency in
`IVSolverInterpolated::validate_query()`:

```cpp
// price_surface_concept.hpp — add to concept
{ s.dividend_yield() } -> std::convertible_to<double>;
```

```cpp
// iv_solver_interpolated.hpp — in validate_query()
if (std::abs(query.dividend_yield - surface_.dividend_yield()) > 1e-10) {
    return ValidationError{ValidationErrorCode::InvalidBounds, 0.0, 0};
}
```

For `SegmentedPriceSurface`, expose `dividend_yield()` from the metadata
of the first segment.

**Files:** `price_surface_concept.hpp`, `american_price_surface.hpp`,
`american_price_surface.cpp`, `segmented_price_surface.hpp`,
`segmented_price_surface.cpp`, `iv_solver_interpolated.hpp`

## 3. Replace `pair<double, double>` with `Dividend` struct

**Problem:** Discrete dividends use `pair<double, double>` throughout.
Nothing prevents swapping calendar time and amount, or passing
time-to-expiry where calendar time is expected.

**Fix:**

Define a named struct in `option_spec.hpp`:

```cpp
struct Dividend {
    double calendar_time;  // years from valuation date
    double amount;         // dollar amount
};
```

Replace `std::vector<std::pair<double, double>>` with
`std::vector<Dividend>` in:
- `PricingParams::discrete_dividends`
- `IVSolverConfig::discrete_dividends`
- `PriceTableConfig::discrete_dividends`
- `PriceTableMetadata::discrete_dividends`
- `SegmentedMultiKRefBuilder::Config::dividends`
- `SegmentedPriceSurface::Config::dividends`

Remove `SegmentedPriceSurface::DividendEntry` (now redundant with
`Dividend`).

Update all call sites to use `{.calendar_time = ..., .amount = ...}`.

**Files:** Every file that touches discrete dividends (~33 files across
src/ and tests/).

## 4. Remove positional constructors from PricingParams and IVQuery

**Problem:** `PricingParams` has constructors with 7-8 positional
`double` parameters. Swapping `rate` and `volatility` compiles and runs.

**Fix:**

Delete the positional constructors from `PricingParams` (lines 228-276)
and `IVQuery` (lines 164-179). Keep:
- Default constructor
- `PricingParams(const OptionSpec&, double volatility, dividends)`

Both structs support designated initializers through their base
(`OptionSpec`) plus own fields. All existing call sites migrate to either:

```cpp
// Preferred: designated initializers via OptionSpec + volatility
PricingParams params;
params.spot = 100.0;
params.strike = 100.0;
// ...
```

Or the kept `PricingParams(spec, volatility)` constructor.

Update ~15 call sites in tests and benchmarks.

**Files:** `option_spec.hpp`, plus test/benchmark files that use
positional constructors.

## Implementation Order

1. `Dividend` struct (item 3) — broadest change, no behavioral risk
2. Remove positional constructors (item 4) — mechanical migration
3. `option_type()` on concept + validation (item 1) — behavioral change
4. `dividend_yield()` on concept + validation (item 2) — behavioral change

Items 1-2 share the same files and should land together.
Items 3-4 are independent and can be separate commits.

## Testing

- Add regression test: PUT query against CALL surface returns error
- Add regression test: mismatched dividend_yield returns error
- Existing tests verify that correct usage still works after constructor removal
- Dividend struct change is compile-verified (wrong field names won't build)
