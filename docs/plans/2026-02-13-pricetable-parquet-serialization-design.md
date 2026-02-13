# Price Table Parquet Serialization Design

## Goal

Add two-layer serialization to all price table surface types:

1. **Data layer** — `to_data()` / `from_data()` on `PriceTable<Inner>`,
   converting between live surfaces and plain vector structs.  No I/O
   dependencies.  Usable in benchmarks, tests, or any context.

2. **I/O layer** — Parquet read/write as a thin wrapper over the data
   layer.  Files readable from Python (pandas, polars, DuckDB).

## Prior Art

Arrow IPC serialization was added in PR #160 and removed during the
C++23 refactor (#248).  The CRC64 checksum utility survives in
`src/support/crc64.hpp`.

## Scope

### Supported surface types (all 7)

| # | Type alias | Dims | Interpolant | Structure |
|---|-----------|------|-------------|-----------|
| 1 | `BSplinePriceTable` | 4D | B-spline | Standard (EEP) |
| 2 | `BSplineMultiKRefSurface` | 4D | B-spline | Segmented multi-K_ref |
| 3 | `ChebyshevSurface` | 4D | Chebyshev (Tucker) | Standard (EEP) |
| 4 | `ChebyshevRawSurface` | 4D | Chebyshev (Raw) | Standard (EEP) |
| 5 | `ChebyshevMultiKRefSurface` | 4D | Chebyshev (Raw) | Segmented multi-K_ref |
| 6 | `BSpline3DPriceTable` | 3D | B-spline | Standard (EEP) |
| 7 | `Chebyshev3DPriceTable` | 3D | Chebyshev (Tucker) | Standard (EEP) |

Tucker-based surfaces (#3, #7) expand to raw tensor on serialize.
On deserialize they reconstruct as Raw-storage equivalents
(`ChebyshevRawSurface`, `Chebyshev3DRawPriceTable`).  Eval results
are identical; in-memory footprint is slightly larger.

### Out of scope

- `SegmentedDimensionlessSurface` (3D segmented — not yet a PriceTable)
- Native Tucker factor serialization (can add later)
- Streaming / incremental writes
- Schema migration tooling

## Layer 1: Data descriptor

### `PriceTableData` struct

```cpp
/// Serializable representation of any PriceTable surface.
/// No Arrow, no I/O — just plain vectors.
struct PriceTableData {
    /// Identifies which PriceTable<Inner> to reconstruct.
    std::string surface_type;

    OptionType option_type = OptionType::PUT;
    double dividend_yield = 0.0;
    DividendSpec dividends;
    double maturity = 0.0;

    /// Per-segment interpolant data.
    struct Segment {
        int32_t segment_id = 0;
        double K_ref = 0.0;
        double tau_start = 0.0, tau_end = 0.0;
        double tau_min = 0.0, tau_max = 0.0;

        /// "bspline" or "chebyshev"
        std::string interp_type;
        size_t ndim = 4;  // 3 or 4

        /// Domain bounds (ndim values each)
        std::vector<double> domain_lo, domain_hi;
        std::vector<int32_t> num_pts;

        /// B-spline: grid and knot vectors per axis
        std::vector<std::vector<double>> grids;   // ndim vectors
        std::vector<std::vector<double>> knots;   // ndim vectors

        /// B-spline coefficients or Chebyshev raw values (flattened)
        std::vector<double> values;
    };
    std::vector<Segment> segments;

    // Build stats (informational, not needed for reconstruction)
    size_t n_pde_solves = 0;
    double precompute_time_seconds = 0.0;
};
```

### API on PriceTable

```cpp
template <typename Inner>
class PriceTable {
public:
    /// Extract data descriptor (no I/O deps).
    [[nodiscard]] PriceTableData to_data() const;

    /// Reconstruct surface from data descriptor.
    /// Returns error if surface_type does not match Inner.
    [[nodiscard]] static std::expected<PriceTable, PriceTableError>
    from_data(const PriceTableData& data);
};
```

Usage (benchmark quick-load):

```cpp
// Build once, extract data
PriceTableData data = surface.to_data();

// Reconstruct from vectors — no PDE, microseconds
auto loaded = BSplinePriceTable::from_data(data).value();
double p = loaded.price(spot, strike, tau, sigma, rate);
```

### Surface-type strings

| surface_type string | PriceTable alias | Notes |
|---|---|---|
| `bspline_4d` | `BSplinePriceTable` | Standard EEP |
| `bspline_4d_segmented` | `BSplineMultiKRefSurface` | Multi-K_ref |
| `chebyshev_4d` | `ChebyshevSurface` | Tucker → Raw on serialize |
| `chebyshev_4d_raw` | `ChebyshevRawSurface` | Raw storage |
| `chebyshev_4d_segmented` | `ChebyshevMultiKRefSurface` | Multi-K_ref |
| `bspline_3d` | `BSpline3DPriceTable` | Dimensionless |
| `chebyshev_3d` | `Chebyshev3DPriceTable` | Tucker → Raw on serialize |

### Reconstruction from data

For standard (EEP) surfaces (1 segment):

1. Reconstruct interpolant from segment vectors:
   - B-spline → `BSplineND<T,N>::create(grids, knots, coeffs)`
   - Chebyshev → `ChebyshevInterpolant<N,RawTensor<N>>::build_from_values(...)`
2. Wrap: `SharedInterp` or direct → `TransformLeaf` → `EEPLayer`
3. Derive `SurfaceBounds` from domain
4. Construct `PriceTable<Inner>(inner, bounds, option_type, dividend_yield)`

For segmented (multi-K_ref) surfaces (N segments):

1. Reconstruct each segment's interpolant
2. Group by K_ref → build `TauSegmentSplit` per K_ref
3. Compose `SplitSurface<Leaf, TauSegmentSplit>` per K_ref
4. Compose `SplitSurface<TauSeg, MultiKRefSplit>` across K_refs
5. Wrap in `PriceTable<Inner>`

### Generic overload-based dispatch

Extraction and reconstruction use free-function overloads that walk the
compositional type tree.  No `to_segments()` member functions on internal
types — the dispatch is entirely external.

**Extraction (`to_data` direction):**

```cpp
// Base case: extract one segment from a TransformLeaf holding a B-spline
template <size_t N, typename Xform>
void extract_segments(
    const TransformLeaf<SharedInterp<BSplineND<double, N>, N>, Xform>& leaf,
    std::vector<PriceTableData::Segment>& out,
    double K_ref, double tau_start, double tau_end,
    double tau_min, double tau_max);

// Base case: extract one segment from a TransformLeaf holding Chebyshev (Raw)
template <size_t N, typename Xform>
void extract_segments(
    const TransformLeaf<ChebyshevInterpolant<N, RawTensor<N>>, Xform>& leaf,
    std::vector<PriceTableData::Segment>& out,
    double K_ref, double tau_start, double tau_end,
    double tau_min, double tau_max);

// Base case: Chebyshev (Tucker) — expand to raw, then same as above
template <size_t N, typename Xform>
void extract_segments(
    const TransformLeaf<ChebyshevInterpolant<N, TuckerTensor<N>>, Xform>& leaf,
    std::vector<PriceTableData::Segment>& out,
    double K_ref, double tau_start, double tau_end,
    double tau_min, double tau_max);

// Recursive: EEPLayer delegates to its leaf
template <typename Leaf, typename EEP>
void extract_segments(const EEPLayer<Leaf, EEP>& layer, ...);

// Recursive: SplitSurface<Inner, TauSegmentSplit> iterates tau segments
template <typename Inner>
void extract_segments(const SplitSurface<Inner, TauSegmentSplit>& split, ...);

// Recursive: SplitSurface<Inner, MultiKRefSplit> iterates K_ref groups
template <typename Inner>
void extract_segments(const SplitSurface<Inner, MultiKRefSplit>& split, ...);
```

**Reconstruction (`from_data` direction):**

```cpp
// reconstruct<Inner>(segments, ...) → Inner
// Overloads mirror extract_segments, building the type tree bottom-up.

template <size_t N, typename Xform>
auto reconstruct_leaf(const PriceTableData::Segment& seg)
    -> std::expected<TransformLeaf<...>, PriceTableError>;

template <typename Inner>
auto reconstruct_eep(const PriceTableData::Segment& seg,
                     OptionType opt, double q)
    -> std::expected<EEPLayer<Inner, AnalyticalEEP>, PriceTableError>;

template <typename Inner>
auto reconstruct_segmented(std::span<const PriceTableData::Segment> segs)
    -> std::expected<SplitSurface<...>, PriceTableError>;
```

**Extensibility:** A new surface type that composes existing interpolants
(e.g., `NumericalEEPLayer<TransformLeaf<SharedInterp<...>>>`) only needs
one new `extract_segments` overload for `NumericalEEPLayer` and one
`reconstruct_eep` overload.  The leaf-level B-spline/Chebyshev overloads
are reused automatically.

## Layer 2: Parquet I/O

### API

```cpp
/// Write PriceTableData to Parquet.
[[nodiscard]] std::expected<void, PriceTableError>
write_parquet(const PriceTableData& data,
              const std::filesystem::path& path,
              const ParquetWriteOptions& opts = {});

/// Read PriceTableData from Parquet.
[[nodiscard]] std::expected<PriceTableData, PriceTableError>
read_parquet(const std::filesystem::path& path);
```

Parquet I/O operates entirely on `PriceTableData` — no templates,
no surface types.  The caller chains:

```cpp
// Write: surface → data → parquet
write_parquet(surface.to_data(), "spy.parquet");

// Read: parquet → data → surface
auto data = read_parquet("spy.parquet").value();
auto surface = BSplinePriceTable::from_data(data).value();
```

Convenience methods on PriceTable delegate to this chain:

```cpp
template <typename Inner>
class PriceTable {
    auto to_parquet(path, opts) { return write_parquet(to_data(), path, opts); }
    static auto from_parquet(path) { return from_data(read_parquet(path).value()); }
};
```

### Parquet schema

Same as original design — see "Row schema" below.

### File-level key-value metadata

| Key | Value type | Example |
|-----|-----------|---------|
| `mango.format_version` | string | `"2.0"` |
| `mango.surface_type` | string | see table above |
| `mango.option_type` | string | `"PUT"` or `"CALL"` |
| `mango.dividend_yield` | string | `"0.02"` |
| `mango.discrete_dividends` | JSON | `[{"t":0.25,"amount":1.50}]` |
| `mango.maturity` | string | `"1.0"` |
| `mango.n_pde_solves` | string | `"15000"` |
| `mango.precompute_time_seconds` | string | `"42.5"` |
| `mango.created_timestamp` | string | ISO-8601 |

### Row schema — one row per interpolant segment

| Column | Parquet type | Notes |
|--------|-------------|-------|
| `segment_id` | INT32 | 0-indexed per K_ref group |
| `K_ref` | DOUBLE | Reference strike |
| `tau_start` | DOUBLE | Segment global tau start |
| `tau_end` | DOUBLE | Segment global tau end |
| `tau_min` | DOUBLE | Local tau lower bound |
| `tau_max` | DOUBLE | Local tau upper bound |
| `interp_type` | STRING | `"bspline"` or `"chebyshev"` |
| `ndim` | INT32 | 3 or 4 |
| `domain_lo` | LIST\<DOUBLE\> | Domain lower bounds |
| `domain_hi` | LIST\<DOUBLE\> | Domain upper bounds |
| `num_pts` | LIST\<INT32\> | Grid points per axis |
| `grid_0`..`grid_3` | LIST\<DOUBLE\> | B-spline grid vectors; empty for Chebyshev or unused dims |
| `knots_0`..`knots_3` | LIST\<DOUBLE\> | B-spline knot vectors; empty for Chebyshev or unused dims |
| `values` | LIST\<DOUBLE\> | Coefficients (B-spline) or function values (Chebyshev) |
| `checksum_values` | INT64 | CRC64 |

3D surfaces use `grid_0`..`grid_2` / `knots_0`..`knots_2`; columns
`grid_3` / `knots_3` are empty lists.

## Build integration

```python
# third_party/arrow/BUILD.bazel — add parquet target
cc_library(name = "parquet", linkopts = ["-lparquet"],
           deps = [":arrow"], visibility = ["//visibility:public"])

# src/option/table/serialization/BUILD.bazel — data layer (no Arrow dep)
cc_library(name = "price_table_data", ...)

# src/option/table/parquet/BUILD.bazel — I/O layer
cc_library(name = "parquet_io", deps = [":price_table_data", "//third_party/arrow:parquet", ...])
```

## Testing

1. **Data round-trip** for each of the 7 surface types: build surface →
   `to_data()` → `from_data()` → verify prices match at sample points.
2. **Parquet round-trip** for each type: `to_parquet` → `from_parquet` →
   verify prices.
3. **CRC64 corruption**: flip byte in Parquet values → read returns error.
4. **Type mismatch**: write `bspline_4d`, read as `ChebyshevRawSurface` → error.
5. **Compression**: ZSTD, Snappy, None all round-trip correctly.
6. **Tucker expansion**: `ChebyshevSurface.to_data()` produces raw values →
   `ChebyshevRawSurface::from_data()` gives identical prices.
7. **3D surfaces**: round-trip `BSpline3DPriceTable` and `Chebyshev3DPriceTable`.
