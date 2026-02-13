# Price Table Parquet Serialization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `to_parquet()` / `from_parquet()` methods to `PriceTable<Inner>` supporting BSpline (standard + segmented) and Chebyshev segmented surfaces.

**Architecture:** Serialization is a method on `PriceTable<Inner>`. Inner types expose const accessors for their data. A shared Parquet schema maps each interpolant segment to one row. Arrow C++ handles all Parquet I/O.

**Tech Stack:** C++23, Apache Arrow C++ (Parquet module), Bazel, GoogleTest

**Design doc:** `docs/plans/2026-02-13-pricetable-parquet-serialization-design.md`

---

### Task 1: Add serialization accessors to internal types

Several internal types have private members without public accessors.
Add const accessors needed for serialization traversal.

**Files:**
- Modify: `src/option/table/split_surface.hpp:116-121`
- Modify: `src/option/table/splits/tau_segment.hpp:68-71`
- Modify: `src/math/chebyshev/raw_tensor.hpp:66-72`
- Modify: `src/math/chebyshev/chebyshev_interpolant.hpp:150-200`

**Step 1: Add accessors to `SplitSurface`**

In `src/option/table/split_surface.hpp`, after `num_pieces()` (line 116), add:

```cpp
[[nodiscard]] const std::vector<Inner>& pieces() const noexcept { return pieces_; }
[[nodiscard]] const Split& split() const noexcept { return split_; }
```

**Step 2: Add accessors to `TauSegmentSplit`**

In `src/option/table/splits/tau_segment.hpp`, before `private:` (line 68), add:

```cpp
[[nodiscard]] const std::vector<double>& tau_start() const noexcept { return tau_start_; }
[[nodiscard]] const std::vector<double>& tau_end() const noexcept { return tau_end_; }
[[nodiscard]] const std::vector<double>& tau_min() const noexcept { return tau_min_; }
[[nodiscard]] const std::vector<double>& tau_max() const noexcept { return tau_max_; }
[[nodiscard]] double K_ref() const noexcept { return K_ref_; }
```

**Step 3: Add `values()` accessor to `RawTensor`**

In `src/math/chebyshev/raw_tensor.hpp`, after `shape()` (line 67), add:

```cpp
[[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }
```

**Step 4: Add `storage()` accessor to `ChebyshevInterpolant`**

In `src/math/chebyshev/chebyshev_interpolant.hpp`, after `num_pts()` (line 154), add:

```cpp
[[nodiscard]] const Storage& storage() const noexcept { return storage_; }
```

**Step 5: Build and test**

Run: `bazel test //...` from the worktree
Expected: All 129 tests pass (accessors are additive, no behavior change)

**Step 6: Commit**

```
git add src/option/table/split_surface.hpp \
        src/option/table/splits/tau_segment.hpp \
        src/math/chebyshev/raw_tensor.hpp \
        src/math/chebyshev/chebyshev_interpolant.hpp
git commit -m "Add const accessors for Parquet serialization"
```

---

### Task 2: Set up Arrow/Parquet build dependency

Wire the Parquet library into the Bazel build and verify it links.

**Files:**
- Modify: `third_party/arrow/BUILD.bazel`
- Create: `src/option/table/parquet/BUILD.bazel`
- Create: `src/option/table/parquet/parquet_io.hpp` (stub)
- Create: `src/option/table/parquet/parquet_io.cpp` (stub)

**Step 1: Add Parquet target to `third_party/arrow/BUILD.bazel`**

Append after the existing `arrow` library:

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

**Step 2: Create `src/option/table/parquet/BUILD.bazel`**

```python
# SPDX-License-Identifier: MIT

cc_library(
    name = "parquet_io",
    srcs = ["parquet_io.cpp"],
    hdrs = ["parquet_io.hpp"],
    deps = [
        "//src/option/table:price_table",
        "//src/option/table:split_surface",
        "//src/option/table/bspline:bspline_surface",
        "//src/option/table/chebyshev:chebyshev_surface",
        "//src/option/table/chebyshev:chebyshev_adaptive",
        "//src/math:bspline_nd",
        "//src/math/chebyshev:chebyshev_interpolant",
        "//src/math/chebyshev:raw_tensor",
        "//src/support:crc64",
        "//third_party/arrow:parquet",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 3: Create stub header `src/option/table/parquet/parquet_io.hpp`**

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace mango {

enum class CompressionType { ZSTD, SNAPPY, NONE };

struct ParquetWriteOptions {
    CompressionType compression = CompressionType::ZSTD;
};

/// Write a price table to Parquet.
template <typename Inner>
[[nodiscard]] std::expected<void, PriceTableError>
write_parquet(const PriceTable<Inner>& table,
              const std::filesystem::path& path,
              const ParquetWriteOptions& opts = {});

/// Read a price table from Parquet.
template <typename Inner>
[[nodiscard]] std::expected<PriceTable<Inner>, PriceTableError>
read_parquet(const std::filesystem::path& path);

}  // namespace mango
```

**Step 4: Create stub source `src/option/table/parquet/parquet_io.cpp`**

```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/table/parquet/parquet_io.hpp"

// Explicit template instantiation stubs — will be implemented in Task 4/5.

namespace mango {
}  // namespace mango
```

**Step 5: Verify the build links against libparquet**

Run: `bazel build //src/option/table/parquet:parquet_io`
Expected: Build succeeds (verifies Arrow + Parquet libraries are found)

**Step 6: Commit**

```
git add third_party/arrow/BUILD.bazel \
        src/option/table/parquet/BUILD.bazel \
        src/option/table/parquet/parquet_io.hpp \
        src/option/table/parquet/parquet_io.cpp
git commit -m "Add Parquet build dependency and IO stub"
```

---

### Task 3: Implement Parquet schema and writer helpers

Build the Arrow schema and row-writing utilities used by `write_parquet()`.

**Files:**
- Create: `src/option/table/parquet/schema.hpp`
- Create: `src/option/table/parquet/schema.cpp`
- Modify: `src/option/table/parquet/BUILD.bazel` (add schema target)

**Step 1: Create `src/option/table/parquet/schema.hpp`**

Define:
- `arrow::Schema` factory (`make_parquet_schema()`)
- `ParquetSegmentRow` struct holding one row of data
- Helper to append a `ParquetSegmentRow` to an Arrow `RecordBatchBuilder`
- Helper to build file-level key-value metadata

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include <arrow/api.h>
#include <arrow/type.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mango {

/// One row of the Parquet table (one interpolant segment).
struct ParquetSegmentRow {
    int32_t segment_id = 0;
    double K_ref = 0.0;
    double tau_start = 0.0;
    double tau_end = 0.0;
    double tau_min = 0.0;
    double tau_max = 0.0;
    std::string interp_type;              // "bspline" or "chebyshev"
    std::vector<double> domain_lo;        // N values
    std::vector<double> domain_hi;        // N values
    std::vector<int32_t> num_pts;         // N values
    std::vector<double> grid_0, grid_1, grid_2, grid_3;
    std::vector<double> knots_0, knots_1, knots_2, knots_3;
    std::vector<double> values;           // coefficients or function values
    int64_t checksum_values = 0;          // CRC64, stored as int64
};

/// Build the Arrow schema for price table Parquet files.
std::shared_ptr<arrow::Schema> make_parquet_schema();

/// Build file-level key-value metadata.
std::shared_ptr<arrow::KeyValueMetadata> make_file_metadata(
    const std::string& surface_type,
    const std::string& option_type,
    double dividend_yield,
    const std::string& discrete_dividends_json,
    double maturity,
    size_t n_pde_solves = 0,
    double precompute_time_seconds = 0.0);

/// Build an Arrow Table from a vector of segment rows + file metadata.
arrow::Result<std::shared_ptr<arrow::Table>>
build_arrow_table(const std::vector<ParquetSegmentRow>& rows,
                  std::shared_ptr<arrow::KeyValueMetadata> metadata);

}  // namespace mango
```

**Step 2: Implement `schema.cpp`**

The schema has 20 columns matching the design doc. Use `arrow::list(arrow::float64())` for LIST<DOUBLE> columns, `arrow::int32()` / `arrow::float64()` / `arrow::utf8()` for scalars. Use `arrow::int64()` for checksums (reinterpret_cast uint64→int64).

Build the Arrow Table from vectors of segment rows by populating `arrow::DoubleBuilder`, `arrow::ListBuilder`, etc.

This is mechanical Arrow API code. Key Arrow types:
- `arrow::DoubleBuilder` for scalar doubles
- `arrow::Int32Builder` for int32 scalars
- `arrow::StringBuilder` for strings
- `arrow::ListBuilder` + `arrow::DoubleBuilder` for LIST<DOUBLE>
- `arrow::ListBuilder` + `arrow::Int32Builder` for LIST<INT32>

**Step 3: Update BUILD.bazel**

Add a `schema` target:

```python
cc_library(
    name = "schema",
    srcs = ["schema.cpp"],
    hdrs = ["schema.hpp"],
    deps = [
        "//third_party/arrow",
        "//third_party/arrow:parquet",
    ],
    visibility = ["//visibility:public"],
)
```

And add `":schema"` to the `parquet_io` deps.

**Step 4: Build**

Run: `bazel build //src/option/table/parquet:schema`
Expected: Compiles successfully

**Step 5: Commit**

```
git add src/option/table/parquet/schema.hpp \
        src/option/table/parquet/schema.cpp \
        src/option/table/parquet/BUILD.bazel
git commit -m "Add Parquet schema and Arrow table builder"
```

---

### Task 4: Implement `write_parquet()` — extract segments from surfaces

Extract `ParquetSegmentRow` vectors from each surface type, then write
to Parquet via the Arrow table builder.

**Files:**
- Modify: `src/option/table/parquet/parquet_io.hpp`
- Modify: `src/option/table/parquet/parquet_io.cpp`

**Step 1: Write the failing test first (in Task 6's test file)**

Before implementing, write a round-trip test that calls `write_parquet`
on a small B-spline surface → assert file exists → read back. This
will fail with link errors until the implementation is done.

**Step 2: Implement surface traversal**

The writer needs to traverse the type tree and extract rows:

For **`BSplinePriceTable`** (`PriceTable<EEPLayer<TransformLeaf<SharedBSplineInterp<4>, StandardTransform4D>, AnalyticalEEP>>`):
- `table.inner()` → `EEPLayer`
- `.leaf()` → `TransformLeaf`
- `.interpolant()` → `SharedBSplineInterp<4>`
- `.get()` → `BSplineND<double, 4>`
- Extract: `.grid(d)`, `.knots(d)`, `.coefficients()`
- Single row, `interp_type = "bspline"`, `segment_id = 0`
- `K_ref` from `.K_ref()`, `tau_start = 0`, `tau_end = grid(1).back()`

For **`BSplineMultiKRefSurface`** (`PriceTable<SplitSurface<SplitSurface<TransformLeaf<...>, TauSegmentSplit>, MultiKRefSplit>>`):
- `table.inner()` → outer `SplitSurface` (MultiKRefSplit)
- `.pieces()` → vector of tau-segmented surfaces per K_ref
- `.split().k_refs()` → K_ref values
- For each K_ref piece: `.pieces()` → vector of `TransformLeaf` leaves
  - `.split().tau_start()`, `.tau_end()`, `.tau_min()`, `.tau_max()`
  - Each leaf: `.interpolant().get()` → `BSplineND<double, 4>`
  - One row per leaf

For **`ChebyshevMultiKRefSurface`** (same structure but Chebyshev leaves):
- Same traversal as B-spline segmented
- Each leaf: `.interpolant()` → `ChebyshevInterpolant<4, RawTensor<4>>`
  - `.domain()` → `Domain<4>` (lo/hi arrays)
  - `.num_pts()` → `array<size_t, 4>`
  - `.storage().values()` → raw tensor data
  - `.storage().shape()` → tensor shape
- `interp_type = "chebyshev"`, grid/knots columns empty

**Step 3: Write Parquet file**

Use `parquet::arrow::WriteTable()` with the Arrow Table from Task 3.
Map `CompressionType` to `arrow::Compression::type`:
- ZSTD → `arrow::Compression::ZSTD`
- SNAPPY → `arrow::Compression::SNAPPY`
- NONE → `arrow::Compression::UNCOMPRESSED`

**Step 4: Compute CRC64 checksums**

For each row, compute `CRC64::compute(values.data(), values.size())`
and store in `checksum_values`. Use the existing `src/support/crc64.hpp`.

**Step 5: Template instantiation**

In `parquet_io.cpp`, explicitly instantiate `write_parquet` for:
- `BSplineLeaf` (BSplinePriceTable's Inner)
- `BSplineMultiKRefInner` (BSplineMultiKRefSurface's Inner)
- `ChebyshevMultiKRefInner` (ChebyshevMultiKRefSurface's Inner)

**Step 6: Build**

Run: `bazel build //src/option/table/parquet:parquet_io`
Expected: Compiles successfully

**Step 7: Commit**

```
git add src/option/table/parquet/parquet_io.hpp \
        src/option/table/parquet/parquet_io.cpp
git commit -m "Implement write_parquet for all surface types"
```

---

### Task 5: Implement `read_parquet()` — reconstruct surfaces

Read Parquet rows, validate checksums, reconstruct the full typed surface.

**Files:**
- Modify: `src/option/table/parquet/parquet_io.cpp`

**Step 1: Read and validate**

Use `parquet::arrow::FileReader` to read the Parquet file:
1. Open file with `arrow::io::ReadableFile::Open(path)`
2. Create `parquet::arrow::FileReader`
3. Read all row groups into an `arrow::Table`
4. Extract file-level metadata (`table->schema()->metadata()`)
5. Validate `mango.format_version` and `mango.surface_type`
6. If `surface_type` doesn't match the template Inner type, return error

**Step 2: Extract rows**

For each row in the Arrow Table:
- Read scalar columns with `arrow::DoubleArray`, `arrow::Int32Array`, `arrow::StringArray`
- Read LIST columns: cast to `arrow::ListArray`, extract value arrays
- Populate `ParquetSegmentRow` structs
- Validate CRC64: recompute from `values` data, compare with `checksum_values`

**Step 3: Reconstruct interpolants**

For `interp_type == "bspline"`:
```cpp
std::array<std::vector<double>, 4> grids = {row.grid_0, row.grid_1, row.grid_2, row.grid_3};
std::array<std::vector<double>, 4> knots = {row.knots_0, row.knots_1, row.knots_2, row.knots_3};
auto spline = BSplineND<double, 4>::create(grids, knots, row.values).value();
auto shared = std::make_shared<const BSplineND<double, 4>>(std::move(*spline));
SharedBSplineInterp<4> interp(shared);
TransformLeaf leaf(std::move(interp), StandardTransform4D{}, row.K_ref);
```

For `interp_type == "chebyshev"`:
```cpp
Domain<4> domain;
for (size_t d = 0; d < 4; ++d) { domain.lo[d] = row.domain_lo[d]; domain.hi[d] = row.domain_hi[d]; }
std::array<size_t, 4> npts;
for (size_t d = 0; d < 4; ++d) npts[d] = row.num_pts[d];
auto interp = ChebyshevInterpolant<4, RawTensor<4>>::build_from_values(
    std::span<const double>(row.values), domain, npts);
TransformLeaf leaf(std::move(interp), StandardTransform4D{}, row.K_ref);
```

**Step 4: Assemble surface tree**

For **BSplinePriceTable** (1 row):
- Wrap leaf in `EEPLayer` with `AnalyticalEEP(option_type, dividend_yield)`
- Derive `SurfaceBounds` from grid bounds
- Construct `PriceTable<BSplineLeaf>(leaf, bounds, option_type, dividend_yield)`

For **BSplineMultiKRefSurface** / **ChebyshevMultiKRefSurface**:
- Group rows by K_ref
- For each K_ref group:
  - Sort segments by `segment_id`
  - Build `TauSegmentSplit` from `tau_start`, `tau_end`, `tau_min`, `tau_max`, `K_ref`
  - Build `SplitSurface<Leaf, TauSegmentSplit>(leaves, tau_split)`
- Collect distinct K_refs → build `MultiKRefSplit(k_refs)`
- Build `SplitSurface<TauSeg, MultiKRefSplit>(per_kref_surfaces, kref_split)`
- Derive `SurfaceBounds` from min/max across all segments' domains
- Construct `PriceTable<Inner>(inner, bounds, option_type, dividend_yield)`

**Step 5: Parse metadata**

Parse `mango.discrete_dividends` JSON (simple format: `[{"t":0.25,"amount":1.50}]`).
Use a minimal hand-parser (the format is trivial fixed-schema JSON)
or nlohmann-json if already available. Check if the project has a JSON
dependency first. If not, a simple `sscanf`-based parser suffices for
this fixed schema.

**Step 6: Template instantiation**

In `parquet_io.cpp`, explicitly instantiate `read_parquet` for the same
three Inner types as Task 4.

**Step 7: Build**

Run: `bazel build //src/option/table/parquet:parquet_io`
Expected: Compiles successfully

**Step 8: Commit**

```
git add src/option/table/parquet/parquet_io.cpp
git commit -m "Implement read_parquet with full surface reconstruction"
```

---

### Task 6: Add `to_parquet` / `from_parquet` methods to PriceTable

Wire the free functions into `PriceTable<Inner>` as member methods.

**Files:**
- Modify: `src/option/table/price_table.hpp`

**Step 1: Add methods**

In `PriceTable<Inner>`, after the existing accessor methods, add:

```cpp
#if __has_include("mango/option/table/parquet/parquet_io.hpp")
    /// Serialize this price table to a Parquet file.
    [[nodiscard]] std::expected<void, PriceTableError>
    to_parquet(const std::filesystem::path& path,
               const ParquetWriteOptions& opts = {}) const {
        return write_parquet(*this, path, opts);
    }

    /// Deserialize a price table from a Parquet file.
    [[nodiscard]] static std::expected<PriceTable, PriceTableError>
    from_parquet(const std::filesystem::path& path) {
        return read_parquet<Inner>(path);
    }
#endif
```

Alternatively, if `__has_include` is undesirable (it makes PriceTable
depend on parquet_io.hpp at include time), use a simpler approach:
just include the header unconditionally and add `//src/option/table/parquet:parquet_io`
to the `price_table` BUILD target deps. The parquet dependency becomes
mandatory.

**Decision for implementer:** If you want Parquet to be optional
(not everyone has Arrow installed), use the `__has_include` guard.
If Parquet is always available in CI and dev, just include directly.
Check with the project owner's preference. The design doc says Arrow
is already in CI (`libarrow-dev`), so direct include is likely fine.

**Step 2: Build**

Run: `bazel build //src/option/table/parquet:parquet_io`
Expected: Compiles

**Step 3: Commit**

```
git add src/option/table/price_table.hpp
git commit -m "Add to_parquet/from_parquet methods to PriceTable"
```

---

### Task 7: Write tests — standard B-spline round-trip

**Files:**
- Create: `tests/parquet_io_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Add test target to `tests/BUILD.bazel`**

```python
cc_test(
    name = "parquet_io_test",
    size = "large",
    srcs = ["parquet_io_test.cc"],
    deps = [
        "//src/option/table/parquet:parquet_io",
        "//src/option/table/bspline:bspline_builder",
        "//src/option/table/bspline:bspline_surface",
        "//src/option/table/bspline:bspline_adaptive",
        "//src/option/table/chebyshev:chebyshev_adaptive",
        "//src/math:bspline_nd",
        "//src/support:crc64",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 2: Write the standard B-spline round-trip test**

```cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <cstdlib>

#include "mango/option/table/parquet/parquet_io.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"

namespace {

class ParquetIOTest : public ::testing::Test {
protected:
    std::filesystem::path tmp_path() {
        auto* tmpdir = std::getenv("TEST_TMPDIR");
        return std::filesystem::path(tmpdir ? tmpdir : "/tmp")
             / "parquet_io_test.parquet";
    }
};

TEST_F(ParquetIOTest, BSplineStandardRoundTrip) {
    // Build a small price table
    std::vector<double> moneyness = {-0.3, -0.1, 0.0, 0.1, 0.3};
    std::vector<double> maturity = {0.1, 0.25, 0.5, 1.0};
    std::vector<double> vol = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> rate = {0.02, 0.03, 0.05, 0.07};
    double K_ref = 100.0;

    auto setup = mango::PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, vol, rate, K_ref,
        mango::GridAccuracyParams{}, mango::OptionType::PUT);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());

    // Wrap as BSplinePriceTable
    auto surface = mango::make_bspline_surface(
        result->spline, result->K_ref,
        result->dividends.dividend_yield,
        mango::OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    // Write
    auto path = tmp_path();
    auto write_result = mango::write_parquet(*surface, path);
    ASSERT_TRUE(write_result.has_value()) << "Write failed";
    EXPECT_TRUE(std::filesystem::exists(path));

    // Read
    auto loaded = mango::read_parquet<mango::BSplineLeaf>(path);
    ASSERT_TRUE(loaded.has_value()) << "Read failed";

    // Verify prices match at several query points
    for (double m : {-0.2, 0.0, 0.2}) {
        for (double tau : {0.15, 0.5}) {
            double spot = K_ref * std::exp(m);
            double p_orig = surface->price(spot, K_ref, tau, 0.20, 0.05);
            double p_loaded = loaded->price(spot, K_ref, tau, 0.20, 0.05);
            EXPECT_NEAR(p_orig, p_loaded, 1e-12)
                << "Mismatch at m=" << m << " tau=" << tau;
        }
    }
}

}  // namespace
```

**Step 3: Run the test**

Run: `bazel test //tests:parquet_io_test --test_output=all`
Expected: PASS

**Step 4: Commit**

```
git add tests/parquet_io_test.cc tests/BUILD.bazel
git commit -m "Add standard B-spline Parquet round-trip test"
```

---

### Task 8: Write tests — checksum corruption, type mismatch, compression

**Files:**
- Modify: `tests/parquet_io_test.cc`

**Step 1: CRC64 corruption test**

Write the file, then flip a byte in the Parquet file (read raw bytes,
modify the values column region, write back). Verify `read_parquet`
returns a checksum error.

Alternatively: write with correct checksums, then manually construct
a row with wrong checksum and write it — simpler and more reliable.

**Step 2: Type mismatch test**

Write a `bspline_standard` file, attempt to read as
`ChebyshevMultiKRefSurface` → verify error.

**Step 3: Compression variants test**

Round-trip with `CompressionType::ZSTD`, `SNAPPY`, and `NONE`.
Verify all produce correct results.

**Step 4: Run tests**

Run: `bazel test //tests:parquet_io_test --test_output=all`
Expected: All pass

**Step 5: Commit**

```
git add tests/parquet_io_test.cc
git commit -m "Add corruption, type-mismatch, and compression tests"
```

---

### Task 9: Write tests — segmented B-spline round-trip

Build a segmented B-spline surface with discrete dividends, serialize,
deserialize, verify prices match.

**Files:**
- Modify: `tests/parquet_io_test.cc`

**Step 1: Build segmented surface**

Use `BSplineSegmentedBuilder` or the `build_adaptive_bspline_segmented`
convenience function with a small grid and 1-2 discrete dividends.
Requires `SegmentedAdaptiveConfig` with K_refs and dividend schedule.

**Step 2: Write and read**

```cpp
auto write_res = mango::write_parquet(surface, path);
auto loaded = mango::read_parquet<mango::BSplineMultiKRefInner>(path);
```

**Step 3: Verify prices**

Sample 50+ random points across the domain, compare original vs loaded.
Tolerance: 1e-12 (bitwise identical coefficients → identical prices).

**Step 4: Run and commit**

Run: `bazel test //tests:parquet_io_test --test_output=all`

```
git add tests/parquet_io_test.cc
git commit -m "Add segmented B-spline Parquet round-trip test"
```

---

### Task 10: Write tests — Chebyshev segmented round-trip

Same pattern as Task 9 but with `ChebyshevMultiKRefSurface`.

**Files:**
- Modify: `tests/parquet_io_test.cc`

**Step 1: Build Chebyshev surface**

Use `build_chebyshev_segmented_manual` or `build_adaptive_chebyshev_segmented`
with a small config.

**Step 2: Write, read, verify prices**

Same flow as Task 9. Use `ChebyshevMultiKRefInner` as the template
parameter for `read_parquet`.

**Step 3: Run and commit**

```
git add tests/parquet_io_test.cc
git commit -m "Add Chebyshev segmented Parquet round-trip test"
```

---

### Task 11: Run full test suite and CI pre-flight

**Step 1: Run all tests**

Run: `bazel test //...`
Expected: All tests pass (existing 129 + new parquet tests)

**Step 2: Build benchmarks**

Run: `bazel build //benchmarks/...`
Expected: Build succeeds

**Step 3: Build Python bindings**

Run: `bazel build //src/python:mango_option`
Expected: Build succeeds

**Step 4: Final commit if any fixups needed**

---

## Accessor summary

| Type | Existing accessors | Accessors to add |
|------|-------------------|-----------------|
| `PriceTable<Inner>` | `inner()`, `option_type()`, `dividend_yield()`, bounds | — |
| `EEPLayer<Leaf, EEP>` | `leaf()`, `interpolant()`, `K_ref()` | — |
| `TransformLeaf<I, X>` | `interpolant()`, `K_ref()` | — |
| `SharedInterp<T, N>` | `get()` → `const T&` | — |
| `BSplineND<T, N>` | `grid(d)`, `knots(d)`, `coefficients()`, `dimensions()` | — |
| `ChebyshevInterpolant<N, S>` | `domain()`, `num_pts()` | `storage()` |
| `RawTensor<N>` | `shape()`, `compressed_size()` | `values()` |
| `SplitSurface<I, S>` | `num_pieces()` | `pieces()`, `split()` |
| `TauSegmentSplit` | — | `tau_start()`, `tau_end()`, `tau_min()`, `tau_max()`, `K_ref()` |
| `MultiKRefSplit` | `k_refs()` | — |

## Type alias reference

```
BSplinePriceTable = PriceTable<EEPLayer<TransformLeaf<SharedInterp<BSplineND<double,4>,4>, StandardTransform4D>, AnalyticalEEP>>

BSplineMultiKRefSurface = PriceTable<SplitSurface<SplitSurface<TransformLeaf<SharedInterp<BSplineND<double,4>,4>, StandardTransform4D>, TauSegmentSplit>, MultiKRefSplit>>

ChebyshevMultiKRefSurface = PriceTable<SplitSurface<SplitSurface<TransformLeaf<ChebyshevInterpolant<4,RawTensor<4>>, StandardTransform4D>, TauSegmentSplit>, MultiKRefSplit>>
```
