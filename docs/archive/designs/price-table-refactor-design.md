<!-- SPDX-License-Identifier: MIT -->
# Price Table Refactor – Full Design

## Objectives

- Unify the price-table pipeline with solver conventions: config → builder → immutable surface.
- Make tensor handling dimension-agnostic (N axes) via templates.
- Provide explicit ownership (no dangling spans) using mdspan views and aligned arenas.
- Bake a known discrete-dividend schedule into the precomputed surface so intraday IV queries see the right payouts.

---

## 1. Core Data Structures

```cpp
// Axis metadata (names optional; help debugging/serialization)
template <size_t N>
struct PriceTableAxes {
    std::array<std::vector<double>, N> grids;   // axis 0..N-1
    std::array<std::string, N> names;           // e.g. {"m", "tau", "sigma", "rate"}
};

// Arena keeps 64-byte alignment and shared ownership
class AlignedArena {
public:
    explicit AlignedArena(size_t bytes, size_t align = 64);
    double* allocate(size_t count);
    std::shared_ptr<AlignedArena> share() const;
private:
    std::pmr::vector<std::byte> buffer_;
    size_t align_;
};

// Tensor wrapper owns the arena and exposes an mdspan view
template <size_t N>
struct PriceTensor {
    std::shared_ptr<AlignedArena> arena;
    std::experimental::mdspan<double, std::experimental::dextents<size_t, N>> view;
};
```

- AlignedArena replaces the ad-hoc vector plus pointer math. Every mdspan references the arena so memory lives as long as any consumer holds the tensor.

---

## 2. Immutable Surface

```cpp
template <size_t N>
class PriceTableSurface {
public:
    struct Metadata {
        double K_ref;
        double dividend_yield;
        std::vector<std::pair<double,double>> discrete_dividends; // fixed schedule
    };

    static std::expected<PriceTableSurface, std::string>
        build(PriceTableAxes<N> axes,
              std::vector<double> coeffs,
              Metadata metadata);

    const PriceTableAxes<N>& axes() const;
    const Metadata& metadata() const;
    auto coeffs() const { return tensor_.view; }   // mdspan

    double value(const std::array<double, N>& coords) const;
    double partial(size_t axis, const std::array<double, N>& coords) const;

private:
    PriceTableAxes<N> axes_;
    PriceTensor<N> tensor_;
    Metadata meta_;
};
```

- Construction moves the owned axes/coeff vectors into the arena and keeps metadata (including the discrete schedule) alongside.
- Consumers hold `std::shared_ptr<const PriceTableSurface<N>>`, analogous to AmericanOptionResult.

---

## 3. Config and Builder Pipeline

```cpp
struct PriceTableConfig {
    OptionType option_type;
    GridSpec<double> grid_estimator;
    size_t n_time;
    double dividend_yield;
    std::vector<std::pair<double,double>> discrete_dividends;
};

template <size_t N>
class PriceTableBuilder {
public:
    explicit PriceTableBuilder(PriceTableConfig cfg);

    std::expected<PriceTableSurface<N>, std::string>
        build(const PriceTableAxes<N>& axes);

private:
    PriceTableConfig config_;

    std::vector<AmericanOptionParams>
        make_batch(const PriceTableAxes<N>& axes, double maturity_limit);

    BatchAmericanOptionResult
        solve_batch(const std::vector<AmericanOptionParams>& batch);

    PriceTensor<N>
        extract_tensor(const BatchAmericanOptionResult& batch,
                       const PriceTableAxes<N>& axes);

    std::vector<double>
        fit_coeffs(const PriceTensor<N>& tensor,
                   const PriceTableAxes<N>& axes);
};
```

**Pipeline steps:**

1. **make_batch**: Uses `std::ranges::views::cartesian_product` over axes [2..N-1] (vol, rate, …) to create PDE parameter sets. Each AmericanOptionParams gets `config_.discrete_dividends` so every solve bakes the known schedule.

2. **solve_batch**: Reuses BatchAmericanOptionSolver, sets grid accuracy/time steps from config, and registers axes.grids[1] as snapshot times.

3. **extract_tensor**: Calls a recursion helper to map the recorded solutions into a PriceTensor<N> mdspan. Each slice is resampled via cubic spline (or whichever 1D interpolator we use).

4. **fit_coeffs**: Uses BSplineNDSeparable (already dimension-agnostic) to fit coefficients, returning a flat vector. This vector is moved into PriceTableSurface::build.

---

## 4. Recursion Helpers (Dimension Agnostic)

```cpp
template <size_t Axis, size_t N, typename Func>
void for_each_axis_index(const PriceTableAxes<N>& axes, Func&& fn);

template <size_t N, typename SliceFunc>
void for_each_slice(PriceTensor<N>& tensor, SliceFunc&& fn);
```

- `for_each_axis_index` drives batch param generation, letting you write "do something for every combination of axis indices" without hard-coded loops.
- `for_each_slice` extracts (N-1)-dimensional slices for interpolation/fitting, mimicking the pattern already used in BSplineNDSeparable.

---

## 5. Ownership & API Consistency

- **Ownership**: Every price table consumer works with `std::shared_ptr<const PriceTableSurface<N>>`. No raw spans leak; the mdspan view keeps the arena alive.
- **API alignment**: `PriceTableSurface::value()` and `partial()` mirror `AmericanOptionResult::value_at` and `IVSolverInterpolated::eval_price/vega`, so IV solvers can swap between PDE-forward evaluation and the precomputed table without changing call sites.
- **Discrete dividends**: The schedule lives in `PriceTableConfig → PriceTableSurface::Metadata`. Batch PDE solves use it, and IV fallback solvers receive it from the surface metadata, keeping both paths consistent.

---

## 6. Extensibility

- Adding a new axis (e.g., discrete dividend scenarios, volatility regimes) is as simple as declaring `PriceTableAxes<5>` and feeding the builder a fifth grid. The recursion helpers and builder pipeline adapt automatically.
- The tensor is ready for streaming: the mdspan view + metadata can be serialized straight into Arrow/IPC without copying.

---

## Summary

This design achieves the goals:

- Template-based dimension handling like BSplineND.
- Proper ownership via arenas and mdspan.
- Fixed discrete-dividend support via config + metadata.
- API parity with existing solver components.

You can implement it incrementally: introduce PriceTableAxes/PriceTableSurface alongside the existing 4D code, then migrate the builder/extractor stages one by one.
