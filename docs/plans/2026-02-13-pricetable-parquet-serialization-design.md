# Price Table Parquet Serialization Design

## Goal

Add Parquet-based serialization to price table surfaces so that
precomputed tables can be saved to disk and reloaded without
re-running the PDE build.  The same files must be readable from
Python (pandas, polars, DuckDB) for cross-language analytics.

## Prior Art

Arrow IPC serialization was added in PR #160 and removed during the
C++23 refactor (#248).  The CRC64 checksum utility survives in
`src/support/crc64.hpp`.  The old schema documented 33 fields and
covered the 4-D B-spline case only.

## Scope

### In scope

- Serialize and deserialize all three price-table surface types:
  - `BSplinePriceTable` (standard, continuous dividends only)
  - `BSplineMultiKRefSurface` (segmented, discrete dividends)
  - `ChebyshevMultiKRefSurface` (segmented Chebyshev)
- Full round-trip: write → read → immediately queryable surface
- CRC64 integrity checks on coefficient / value data
- ZSTD, Snappy, and uncompressed output

### Out of scope

- `SegmentedDimensionlessSurface` (experimental, schema can extend later)
- Streaming / incremental writes
- Schema migration tooling

## API

Serialization lives on `PriceTable<Inner>`:

```cpp
template <typename Inner>
class PriceTable {
public:
    [[nodiscard]] std::expected<void, PriceTableError>
    to_parquet(const std::filesystem::path& path,
               const ParquetWriteOptions& opts = {}) const;

    [[nodiscard]] static std::expected<PriceTable, PriceTableError>
    from_parquet(const std::filesystem::path& path);
};
```

Usage:

```cpp
// Write
surface.to_parquet("spy_put.parquet");

// Read — fully reconstructed, ready to eval
auto surface = ChebyshevMultiKRefSurface::from_parquet("spy.parquet").value();
double p = surface.price(spot, strike, tau, sigma, rate);
```

If the file's `mango.surface_type` does not match the template
specialization, `from_parquet` returns an error.

### Options

```cpp
enum class CompressionType { ZSTD, SNAPPY, NONE };

struct ParquetWriteOptions {
    CompressionType compression = CompressionType::ZSTD;
};
```

## Parquet Schema

### File-level key-value metadata

| Key | Value type | Example |
|-----|-----------|---------|
| `mango.format_version` | string | `"2.0"` |
| `mango.surface_type` | string | `"bspline_standard"`, `"bspline_segmented"`, `"chebyshev_segmented"` |
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
| `tau_min` | DOUBLE | Local tau lower bound (for TauSegmentSplit clamping) |
| `tau_max` | DOUBLE | Local tau upper bound |
| `interp_type` | STRING | `"bspline"` or `"chebyshev"` |
| `domain_lo` | LIST\<DOUBLE\> | Domain lower bounds (N values) |
| `domain_hi` | LIST\<DOUBLE\> | Domain upper bounds (N values) |
| `num_pts` | LIST\<INT32\> | Grid points per axis (N values) |
| `grid_0` | LIST\<DOUBLE\> | Grid vector, axis 0 (moneyness); empty for Chebyshev |
| `grid_1` | LIST\<DOUBLE\> | Grid vector, axis 1 (tau) |
| `grid_2` | LIST\<DOUBLE\> | Grid vector, axis 2 (vol) |
| `grid_3` | LIST\<DOUBLE\> | Grid vector, axis 3 (rate) |
| `knots_0` | LIST\<DOUBLE\> | Knot vector, axis 0; empty for Chebyshev |
| `knots_1` | LIST\<DOUBLE\> | Knot vector, axis 1 |
| `knots_2` | LIST\<DOUBLE\> | Knot vector, axis 2 |
| `knots_3` | LIST\<DOUBLE\> | Knot vector, axis 3 |
| `values` | LIST\<DOUBLE\> | B-spline coefficients or Chebyshev function values |
| `checksum_values` | INT64 | CRC64 of values (stored as signed, reinterpret on read) |

### Row layout per surface type

**`bspline_standard`** — 1 row, `segment_id=0`, `tau_start=0`,
`tau_end=maturity`, `interp_type="bspline"`.

**`bspline_segmented`** — one row per (K_ref, tau-segment) pair.
Rows ordered by K_ref then segment_id.

**`chebyshev_segmented`** — same structure; `grid_*` and `knots_*`
columns are empty lists, domain encoded in `domain_lo` / `domain_hi`.

## Reconstruction

On `from_parquet`:

1. Read file-level metadata; validate `format_version` and
   `surface_type`.
2. Read all rows.  For each row:
   - Validate CRC64 checksum against `values` data.
   - Reconstruct interpolant:
     - B-spline → `BSplineND<double,4>::create(grids, knots, coeffs)`
     - Chebyshev → `ChebyshevInterpolant<4,RawTensor<4>>::build_from_values(values, domain, num_pts)`
   - Wrap in `TransformLeaf<Interp, StandardTransform4D>` with
     the row's `K_ref`.
3. Group segments by K_ref → build per-K_ref `TauSegmentSplit` from
   `tau_start`, `tau_end`, `tau_min`, `tau_max`, `K_ref`.
4. Compose `SplitSurface<Leaf, TauSegmentSplit>` per K_ref.
5. Compose `SplitSurface<TauSeg, MultiKRefSplit>` from distinct K_refs.
6. Wrap in `PriceTable<Inner>` with `option_type`, `dividend_yield`,
   bounds derived from domain.

Cost: microseconds (validation + vector moves), no PDE work.

## Inner-type serialization concept

```cpp
template <typename T>
concept ParquetSerializable = requires(const T& t, ParquetSegmentWriter& w) {
    { t.write_segments(w) } -> std::same_as<void>;
    { T::read_segments(ParquetSegmentReader&) } -> std::same_as<std::expected<T, PriceTableError>>;
};
```

Each layer implements this recursively:
- `SplitSurface::write_segments` iterates pieces, delegates to inner
- `TransformLeaf::write_segments` writes one row (the interpolant data)
- `BSplineND` / `ChebyshevInterpolant` provide accessors for grids,
  knots, domain, values

## Build integration

### Bazel targets

```python
# third_party/arrow/BUILD.bazel — add parquet library
cc_library(
    name = "parquet",
    linkopts = ["-lparquet"],
    deps = [":arrow"],
    visibility = ["//visibility:public"],
)

# src/option/table/parquet/BUILD.bazel
cc_library(
    name = "parquet_io",
    srcs = ["parquet_writer.cpp", "parquet_reader.cpp"],
    hdrs = ["parquet_writer.hpp", "parquet_reader.hpp"],
    deps = [
        "//src/option/table:price_table",
        "//src/option/table/bspline:bspline_surface",
        "//src/option/table/chebyshev:chebyshev_surface",
        "//src/math:bspline_nd",
        "//src/math/chebyshev:chebyshev_interpolant",
        "//src/support:crc64",
        "//third_party/arrow:parquet",
    ],
)
```

### CI

Add `libparquet-dev` alongside existing `libarrow-dev`.

## Testing

1. **Round-trip (standard B-spline):** Build `PriceTableResult<4>` →
   wrap as `BSplinePriceTable` → `to_parquet` → `from_parquet` →
   verify bitwise-identical grids, knots, coefficients.

2. **Round-trip (segmented B-spline):** Build with 2 K_refs and 3
   dividend dates → serialize → deserialize → verify all segments.

3. **Round-trip (Chebyshev segmented):** Same with Chebyshev
   interpolants → verify domain, num_pts, values.

4. **CRC64 corruption:** Write valid file, flip a byte in the values
   column → verify `from_parquet` returns checksum error.

5. **Type mismatch:** Write a `bspline_standard` file, attempt to
   read as `ChebyshevMultiKRefSurface` → verify error.

6. **Compression:** Round-trip with ZSTD, Snappy, and uncompressed.

7. **Forward compatibility:** Read a file with extra unknown metadata
   keys → verify it still loads.

8. **Price accuracy:** After round-trip, evaluate `surface.price()`
   at 100 random points → verify results match the original surface
   within floating-point epsilon.

## File size estimates

| Grid | Coefficients | Grids+Knots | Total |
|------|-------------|-------------|-------|
| 50×30×20×10 | 2.4 MB | 2 KB | ~2.4 MB |
| 20×15×10×5 | 120 KB | 0.5 KB | ~120 KB |
| Segmented (3 K_ref × 4 seg) | ~10 MB | 24 KB | ~10 MB |

ZSTD compression typically reduces B-spline coefficient data by
40-60%.
