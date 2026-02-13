# Price Table Parquet Serialization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two-layer serialization (data + Parquet I/O) to all 7 `PriceTable<Inner>` types, enabling fast surface reconstruction from vectors and cross-language Parquet files.

**Architecture:** Layer 1 converts between live surfaces and a `PriceTableData` struct of plain vectors (no I/O deps). Layer 2 maps `PriceTableData` to/from Parquet via Arrow C++. `PriceTable<Inner>` exposes `to_data()`/`from_data()` and convenience `to_parquet()`/`from_parquet()`.

**Tech Stack:** C++23, Apache Arrow C++ (Parquet module), Bazel, GoogleTest

**Design doc:** `docs/plans/2026-02-13-pricetable-parquet-serialization-design.md`

---

### Task 1: Add serialization accessors to internal types

Several internal types have private members without public accessors.
Add const accessors needed for data extraction.

**Files:**
- Modify: `src/option/table/split_surface.hpp:116`
- Modify: `src/option/table/splits/tau_segment.hpp:68`
- Modify: `src/math/chebyshev/raw_tensor.hpp:67`
- Modify: `src/math/chebyshev/chebyshev_interpolant.hpp:154`
- Modify: `src/math/chebyshev/tucker_tensor.hpp:395`

**Step 1: Add accessors to `SplitSurface`**

In `src/option/table/split_surface.hpp`, after `num_pieces()` (line 116):

```cpp
[[nodiscard]] const std::vector<Inner>& pieces() const noexcept { return pieces_; }
[[nodiscard]] const Split& split() const noexcept { return split_; }
```

**Step 2: Add accessors to `TauSegmentSplit`**

In `src/option/table/splits/tau_segment.hpp`, before `private:` (line 68):

```cpp
[[nodiscard]] const std::vector<double>& tau_start() const noexcept { return tau_start_; }
[[nodiscard]] const std::vector<double>& tau_end() const noexcept { return tau_end_; }
[[nodiscard]] const std::vector<double>& tau_min() const noexcept { return tau_min_; }
[[nodiscard]] const std::vector<double>& tau_max() const noexcept { return tau_max_; }
[[nodiscard]] double K_ref() const noexcept { return K_ref_; }
```

**Step 3: Add `values()` accessor to `RawTensor`**

In `src/math/chebyshev/raw_tensor.hpp`, after `shape()` (line 67):

```cpp
[[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }
```

**Step 4: Add `storage()` accessor to `ChebyshevInterpolant`**

In `src/math/chebyshev/chebyshev_interpolant.hpp`, after `num_pts()` (line 154):

```cpp
[[nodiscard]] const Storage& storage() const noexcept { return storage_; }
```

**Step 5: Add `expand()` to `TuckerTensor`**

In `src/math/chebyshev/tucker_tensor.hpp`, after `ranks()` (line 395):

```cpp
/// Expand Tucker decomposition to full raw tensor.
/// Returns vector of size product(shape).
[[nodiscard]] std::vector<double> expand() const {
    TuckerResult<N> result;
    result.core = core_;
    result.factors = factors_;
    result.shape = shape_;
    result.ranks = ranks_;
    return tucker_reconstruct<N>(result);
}

[[nodiscard]] const std::array<size_t, N>& shape() const { return shape_; }
```

Note: `TuckerTensor` already has `ranks()` but not `shape()`.
Check first — if `shape()` already exists, skip that accessor.

**Step 6: Build and test**

Run: `bazel test //...`
Expected: All 129 tests pass (additive changes only)

**Step 7: Commit**

```
git add src/option/table/split_surface.hpp \
        src/option/table/splits/tau_segment.hpp \
        src/math/chebyshev/raw_tensor.hpp \
        src/math/chebyshev/chebyshev_interpolant.hpp \
        src/math/chebyshev/tucker_tensor.hpp
git commit -m "Add const accessors for serialization"
```

---

### Task 2: Create `PriceTableData` struct and BUILD target

The data descriptor is a plain struct of vectors — no Arrow, no
templates, no surface-type dependencies.

**Files:**
- Create: `src/option/table/serialization/price_table_data.hpp`
- Create: `src/option/table/serialization/BUILD.bazel`

**Step 1: Create `price_table_data.hpp`**

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace mango {

/// Serializable representation of any PriceTable surface.
/// Plain vectors — no I/O dependencies.
struct PriceTableData {
    std::string surface_type;

    OptionType option_type = OptionType::PUT;
    double dividend_yield = 0.0;
    DividendSpec dividends;
    double maturity = 0.0;

    struct Segment {
        int32_t segment_id = 0;
        double K_ref = 0.0;
        double tau_start = 0.0, tau_end = 0.0;
        double tau_min = 0.0, tau_max = 0.0;
        std::string interp_type;  // "bspline" or "chebyshev"
        size_t ndim = 4;

        std::vector<double> domain_lo, domain_hi;
        std::vector<int32_t> num_pts;
        std::vector<std::vector<double>> grids;   // ndim vectors
        std::vector<std::vector<double>> knots;   // ndim vectors
        std::vector<double> values;               // coefficients or raw values
    };
    std::vector<Segment> segments;

    size_t n_pde_solves = 0;
    double precompute_time_seconds = 0.0;
};

}  // namespace mango
```

**Step 2: Create BUILD.bazel**

```python
# SPDX-License-Identifier: MIT

cc_library(
    name = "price_table_data",
    hdrs = ["price_table_data.hpp"],
    deps = [
        "//src/option:option_spec",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 3: Build**

Run: `bazel build //src/option/table/serialization:price_table_data`
Expected: Compiles

**Step 4: Commit**

```
git add src/option/table/serialization/price_table_data.hpp \
        src/option/table/serialization/BUILD.bazel
git commit -m "Add PriceTableData descriptor struct"
```

---

### Task 3: Implement `to_data()` — extract data from all surface types

Traverse each surface type's layer tree and produce `PriceTableData`.

**Files:**
- Create: `src/option/table/serialization/to_data.hpp`
- Create: `src/option/table/serialization/to_data.cpp`
- Modify: `src/option/table/serialization/BUILD.bazel`

**Step 1: Write the failing test**

In `tests/price_table_data_test.cc` (created in Task 5):

```cpp
TEST(PriceTableDataTest, BSpline4DToDataProducesOneSegment) {
    auto surface = build_small_bspline_4d();
    auto data = surface.to_data();
    EXPECT_EQ(data.surface_type, "bspline_4d");
    ASSERT_EQ(data.segments.size(), 1);
    EXPECT_EQ(data.segments[0].interp_type, "bspline");
    EXPECT_EQ(data.segments[0].ndim, 4);
    EXPECT_FALSE(data.segments[0].values.empty());
}
```

**Step 2: Implement extraction helpers**

For each leaf type, implement a function that produces one `Segment`:

```cpp
// Extract from a B-spline TransformLeaf (4D or 3D)
template <size_t N>
PriceTableData::Segment extract_bspline_segment(
    const TransformLeaf<SharedInterp<BSplineND<double, N>, N>,
                         auto>& leaf,
    int32_t segment_id, double tau_start, double tau_end,
    double tau_min, double tau_max);

// Extract from a Chebyshev TransformLeaf (RawTensor)
template <size_t N>
PriceTableData::Segment extract_chebyshev_segment(
    const TransformLeaf<ChebyshevInterpolant<N, RawTensor<N>>,
                         auto>& leaf,
    int32_t segment_id, double tau_start, double tau_end,
    double tau_min, double tau_max);
```

For Tucker-based ChebyshevInterpolant, call `storage().expand()` to
get raw values, then produce the same Segment as RawTensor.

**Step 3: Implement `to_data()` for each PriceTable specialization**

The implementation traverses the type tree:

- **Standard (EEP) types** (BSplinePriceTable, ChebyshevRawSurface,
  ChebyshevSurface, BSpline3DPriceTable, Chebyshev3DPriceTable):
  `table.inner().leaf()` → extract one segment.  Set `tau_start=0`,
  `tau_end=tau_max`, `tau_min=table.tau_min()`, `tau_max=table.tau_max()`.

- **Segmented multi-K_ref types** (BSplineMultiKRefSurface,
  ChebyshevMultiKRefSurface):
  `table.inner().pieces()` → iterate K_ref pieces.
  Each piece: `.pieces()` → iterate tau segments.
  Extract TauSegmentSplit data for each segment's tau boundaries.

**Step 4: Wire `to_data()` into PriceTable**

In `price_table.hpp`, add:

```cpp
[[nodiscard]] PriceTableData to_data() const;
```

The implementation lives in `to_data.cpp` with explicit template
instantiations for all 7 Inner types.

**Step 5: Update BUILD.bazel**

```python
cc_library(
    name = "to_data",
    srcs = ["to_data.cpp"],
    hdrs = ["to_data.hpp"],
    deps = [
        ":price_table_data",
        "//src/option/table:price_table",
        "//src/option/table:split_surface",
        "//src/option/table/bspline:bspline_surface",
        "//src/option/table/bspline:bspline_adaptive",
        "//src/option/table/chebyshev:chebyshev_surface",
        "//src/option/table/chebyshev:chebyshev_adaptive",
        "//src/option/table/bspline:bspline_3d_surface",
        "//src/option/table/chebyshev:chebyshev_3d_surface",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 6: Build**

Run: `bazel build //src/option/table/serialization:to_data`

**Step 7: Commit**

```
git add src/option/table/serialization/to_data.hpp \
        src/option/table/serialization/to_data.cpp \
        src/option/table/serialization/BUILD.bazel \
        src/option/table/price_table.hpp
git commit -m "Implement to_data() for all 7 surface types"
```

---

### Task 4: Implement `from_data()` — reconstruct surfaces from vectors

Reconstruct each surface type from `PriceTableData`.

**Files:**
- Create: `src/option/table/serialization/from_data.hpp`
- Create: `src/option/table/serialization/from_data.cpp`
- Modify: `src/option/table/serialization/BUILD.bazel`

**Step 1: Implement interpolant reconstruction**

```cpp
// B-spline: grids + knots + coefficients → BSplineND
template <size_t N>
std::expected<std::shared_ptr<const BSplineND<double, N>>, PriceTableError>
make_bspline_from_segment(const PriceTableData::Segment& seg);

// Chebyshev: domain + num_pts + values → ChebyshevInterpolant
template <size_t N>
std::expected<ChebyshevInterpolant<N, RawTensor<N>>, PriceTableError>
make_chebyshev_from_segment(const PriceTableData::Segment& seg);
```

**Step 2: Implement surface assembly**

For standard (EEP) surfaces:
1. Reconstruct interpolant from single segment
2. Wrap in `SharedInterp` (B-spline) or use directly (Chebyshev)
3. Wrap in `TransformLeaf` with appropriate transform
4. Wrap in `EEPLayer` with `AnalyticalEEP(option_type, dividend_yield)`
5. Derive `SurfaceBounds` from domain
6. Construct `PriceTable<Inner>`

For segmented multi-K_ref:
1. Group segments by K_ref
2. For each K_ref group: build leaves + `TauSegmentSplit` + `SplitSurface`
3. Build `MultiKRefSplit` + outer `SplitSurface`
4. Construct `PriceTable<Inner>`

**Step 3: Wire `from_data()` into PriceTable**

```cpp
template <typename Inner>
[[nodiscard]] static std::expected<PriceTable<Inner>, PriceTableError>
PriceTable<Inner>::from_data(const PriceTableData& data);
```

Explicit instantiations for all 7 Inner types in `from_data.cpp`.

**Step 4: Validate surface_type**

If `data.surface_type` doesn't match the expected string for the
template Inner type, return `PriceTableError` with
`PriceTableErrorCode::InvalidConfig`.

**Step 5: Build**

Run: `bazel build //src/option/table/serialization:from_data`

**Step 6: Commit**

```
git add src/option/table/serialization/from_data.hpp \
        src/option/table/serialization/from_data.cpp \
        src/option/table/serialization/BUILD.bazel \
        src/option/table/price_table.hpp
git commit -m "Implement from_data() for all 7 surface types"
```

---

### Task 5: Tests — data layer round-trips for all 7 types

**Files:**
- Create: `tests/price_table_data_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Add test target**

```python
cc_test(
    name = "price_table_data_test",
    size = "large",
    srcs = ["price_table_data_test.cc"],
    deps = [
        "//src/option/table/serialization:to_data",
        "//src/option/table/serialization:from_data",
        "//src/option/table/bspline:bspline_builder",
        "//src/option/table/bspline:bspline_surface",
        "//src/option/table/bspline:bspline_adaptive",
        "//src/option/table/chebyshev:chebyshev_adaptive",
        "//src/option/table/chebyshev:chebyshev_surface",
        "//src/option/table/bspline:bspline_3d_surface",
        "//src/option/table/chebyshev:chebyshev_3d_surface",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 2: Write round-trip tests**

For each of the 7 surface types:
1. Build a small surface (minimal grid for speed)
2. Call `to_data()`
3. Verify segment count and interp_type
4. Call `from_data()`
5. Evaluate `price()` at 20+ sample points
6. Verify prices match within 1e-12

Include tests for:
- **Tucker expansion**: `ChebyshevSurface.to_data()` then
  `ChebyshevRawSurface::from_data()` — prices must match.
- **Type mismatch**: `BSplinePriceTable.to_data()` then
  `ChebyshevRawSurface::from_data()` → returns error.

**Step 3: Run**

Run: `bazel test //tests:price_table_data_test --test_output=all`

**Step 4: Commit**

```
git add tests/price_table_data_test.cc tests/BUILD.bazel
git commit -m "Add data-layer round-trip tests for all 7 surface types"
```

---

### Task 6: Set up Arrow/Parquet build dependency

**Files:**
- Modify: `third_party/arrow/BUILD.bazel`

**Step 1: Add Parquet target**

Append to `third_party/arrow/BUILD.bazel`:

```python
cc_library(
    name = "parquet",
    hdrs = [],
    includes = ["/usr/include"],
    linkopts = ["-lparquet"],
    deps = [":arrow"],
    visibility = ["//visibility:public"],
)
```

**Step 2: Verify**

Run: `bazel build //third_party/arrow:parquet`

**Step 3: Commit**

```
git add third_party/arrow/BUILD.bazel
git commit -m "Add Parquet build target to third_party/arrow"
```

---

### Task 7: Implement Parquet writer (`write_parquet`)

Map `PriceTableData` to Parquet.  No templates — pure data.

**Files:**
- Create: `src/option/table/parquet/parquet_io.hpp`
- Create: `src/option/table/parquet/parquet_io.cpp`
- Create: `src/option/table/parquet/BUILD.bazel`

**Step 1: Implement `write_parquet(PriceTableData, path, opts)`**

1. Build Arrow schema (20 columns from the design doc)
2. For each segment in `data.segments`, populate Arrow builders
3. Compute CRC64 checksum on each segment's values
4. Encode file-level metadata from `data.*` fields
5. Write with `parquet::arrow::WriteTable()`

Map `CompressionType` to Arrow compression codec.

**Step 2: Implement `read_parquet(path) → PriceTableData`**

1. Open with `parquet::arrow::FileReader`
2. Read into Arrow Table
3. Extract file metadata → populate `data.*`
4. For each row: extract columns → populate `Segment`
5. Validate CRC64 checksum; return error on mismatch

**Step 3: Wire convenience methods on PriceTable**

In `price_table.hpp`:

```cpp
[[nodiscard]] std::expected<void, PriceTableError>
to_parquet(const std::filesystem::path& path,
           const ParquetWriteOptions& opts = {}) const {
    return mango::write_parquet(to_data(), path, opts);
}

[[nodiscard]] static std::expected<PriceTable, PriceTableError>
from_parquet(const std::filesystem::path& path) {
    auto data = mango::read_parquet(path);
    if (!data.has_value()) return std::unexpected(data.error());
    return from_data(*data);
}
```

**Step 4: BUILD.bazel**

```python
# SPDX-License-Identifier: MIT

cc_library(
    name = "parquet_io",
    srcs = ["parquet_io.cpp"],
    hdrs = ["parquet_io.hpp"],
    deps = [
        "//src/option/table/serialization:price_table_data",
        "//src/support:crc64",
        "//third_party/arrow",
        "//third_party/arrow:parquet",
    ],
    visibility = ["//visibility:public"],
)
```

Note: `parquet_io` depends on `price_table_data` only, not on
any surface types.  This keeps Arrow quarantined.

**Step 5: Build**

Run: `bazel build //src/option/table/parquet:parquet_io`

**Step 6: Commit**

```
git add src/option/table/parquet/parquet_io.hpp \
        src/option/table/parquet/parquet_io.cpp \
        src/option/table/parquet/BUILD.bazel \
        src/option/table/price_table.hpp
git commit -m "Implement Parquet read/write for PriceTableData"
```

---

### Task 8: Tests — Parquet round-trips

**Files:**
- Create: `tests/parquet_io_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Add test target**

**Step 2: Write tests**

- **Parquet round-trip** for each of the 7 types: build surface →
  `to_parquet()` → `from_parquet()` → verify prices.
- **CRC64 corruption**: manually set wrong checksum in
  `PriceTableData`, write to Parquet, read → verify error.
- **Type mismatch**: write `bspline_4d`, read as `ChebyshevRawSurface` → error.
- **Compression**: round-trip with ZSTD, Snappy, None.
- **Forward compat**: add unknown metadata key to file → loads fine.

**Step 3: Run**

Run: `bazel test //tests:parquet_io_test --test_output=all`

**Step 4: Commit**

```
git add tests/parquet_io_test.cc tests/BUILD.bazel
git commit -m "Add Parquet round-trip tests for all surface types"
```

---

### Task 9: Run full test suite and CI pre-flight

**Step 1:** `bazel test //...` — all tests pass
**Step 2:** `bazel build //benchmarks/...` — benchmarks compile
**Step 3:** `bazel build //src/python:mango_option` — Python bindings compile
**Step 4:** Fix any issues, commit

---

## Type reference

```
#1 BSplinePriceTable
    = PriceTable<EEPLayer<TransformLeaf<SharedInterp<BSplineND<double,4>,4>, StandardTransform4D>, AnalyticalEEP>>

#2 BSplineMultiKRefSurface
    = PriceTable<SplitSurface<SplitSurface<TransformLeaf<SharedInterp<BSplineND<double,4>,4>, StandardTransform4D>, TauSegmentSplit>, MultiKRefSplit>>

#3 ChebyshevSurface
    = PriceTable<EEPLayer<TransformLeaf<ChebyshevInterpolant<4,TuckerTensor<4>>, StandardTransform4D>, AnalyticalEEP>>

#4 ChebyshevRawSurface
    = PriceTable<EEPLayer<TransformLeaf<ChebyshevInterpolant<4,RawTensor<4>>, StandardTransform4D>, AnalyticalEEP>>

#5 ChebyshevMultiKRefSurface
    = PriceTable<SplitSurface<SplitSurface<TransformLeaf<ChebyshevInterpolant<4,RawTensor<4>>, StandardTransform4D>, TauSegmentSplit>, MultiKRefSplit>>

#6 BSpline3DPriceTable
    = PriceTable<EEPLayer<TransformLeaf<SharedInterp<BSplineND<double,3>,3>, DimensionlessTransform3D>, AnalyticalEEP>>

#7 Chebyshev3DPriceTable
    = PriceTable<EEPLayer<TransformLeaf<ChebyshevInterpolant<3,TuckerTensor<3>>, DimensionlessTransform3D>, AnalyticalEEP>>
```

## Accessor summary

| Type | Existing | To add |
|------|----------|--------|
| `PriceTable<Inner>` | `inner()`, `option_type()`, `dividend_yield()`, bounds | `to_data()`, `from_data()`, `to_parquet()`, `from_parquet()` |
| `EEPLayer` | `leaf()`, `interpolant()`, `K_ref()` | — |
| `TransformLeaf` | `interpolant()`, `K_ref()` | — |
| `SharedInterp<T,N>` | `get()` → `const T&` | — |
| `BSplineND<T,N>` | `grid(d)`, `knots(d)`, `coefficients()`, `dimensions()` | — |
| `ChebyshevInterpolant<N,S>` | `domain()`, `num_pts()` | `storage()` |
| `RawTensor<N>` | `shape()`, `compressed_size()` | `values()` |
| `TuckerTensor<N>` | `ranks()`, `compressed_size()` | `expand()`, `shape()` |
| `SplitSurface<I,S>` | `num_pieces()` | `pieces()`, `split()` |
| `TauSegmentSplit` | — | `tau_start()`, `tau_end()`, `tau_min()`, `tau_max()`, `K_ref()` |
| `MultiKRefSplit` | `k_refs()` | — |
