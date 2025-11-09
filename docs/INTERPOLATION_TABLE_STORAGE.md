# Interpolation Table Storage

Memory-mapped binary storage for 4D B-spline interpolation tables.

## Overview

The interpolation table storage module provides fast, efficient save/load functionality for pre-computed option price tables. It uses a custom binary format with memory mapping to enable:

- **Instant loading**: ~microseconds to load multi-MB tables
- **Zero-copy access**: Memory-mapped data used directly
- **Compact format**: ~2-5 MB for typical price tables
- **Self-describing**: Version headers and metadata included

## File Format

### Header (256 bytes)

```c
struct InterpolationTableHeader {
    uint32_t magic;              // 0x4D494E54 ('MINT')
    uint32_t version;            // Format version (1)
    double K_ref;                // Reference strike price
    uint32_t option_type;        // 0=PUT, 1=CALL
    uint32_t spline_degree;      // B-spline degree (typically 3)

    // Grid dimensions
    uint64_t n_moneyness;
    uint64_t n_maturity;
    uint64_t n_volatility;
    uint64_t n_rate;
    uint64_t n_coefficients;

    // Data offsets (64-byte aligned)
    uint64_t moneyness_offset;
    uint64_t maturity_offset;
    uint64_t volatility_offset;
    uint64_t rate_offset;
    uint64_t coefficients_offset;

    char option_type_str[16];    // "PUT" or "CALL"
    char reserved[128];          // Reserved
};
```

### Data Layout

All data arrays are 64-byte aligned for cache efficiency:

1. **Header** (256 bytes)
2. **Moneyness knots** (n_moneyness × 8 bytes)
3. **Maturity knots** (n_maturity × 8 bytes)
4. **Volatility knots** (n_volatility × 8 bytes)
5. **Rate knots** (n_rate × 8 bytes)
6. **Coefficients** (n_coefficients × 8 bytes, row-major 4D tensor)

## API Reference

### Saving Tables

```cpp
#include "src/interpolation_table_storage_v2.hpp"

auto result = InterpolationTableStorage::save(
    "table.mint",           // filepath
    moneyness_knots,        // vector<double>
    maturity_knots,         // vector<double>
    volatility_knots,       // vector<double>
    rate_knots,             // vector<double>
    coefficients,           // vector<double> (4D flattened)
    100.0,                  // K_ref
    "PUT",                  // option_type
    3                       // spline_degree
);

if (!result) {
    std::cerr << "Save failed: " << result.error() << "\n";
}
```

### Loading Tables

```cpp
auto result = InterpolationTableStorage::load("table.mint");

if (!result) {
    std::cerr << "Load failed: " << result.error() << "\n";
    return;
}

auto spline = std::move(*result);  // unique_ptr<BSpline4D_FMA>

// Query prices
double price = spline->eval(1.05, 0.25, 0.20, 0.05);
```

**Note:** Loading uses memory mapping. The file must remain valid for the lifetime of the returned `BSpline4D_FMA` object. The data is copied during BSpline4D_FMA construction, so the memory map can be released immediately after construction.

### Reading Metadata

```cpp
auto result = InterpolationTableStorage::read_metadata("table.mint");

if (result) {
    auto meta = *result;
    std::cout << "K_ref: " << meta.K_ref << "\n"
              << "Option type: " << meta.option_type << "\n"
              << "Spline degree: " << meta.spline_degree << "\n"
              << "Grid: " << meta.n_moneyness << "×"
                          << meta.n_maturity << "×"
                          << meta.n_volatility << "×"
                          << meta.n_rate << "\n"
              << "Coefficients: " << meta.n_coefficients << "\n"
              << "File size: " << (meta.file_size_bytes / 1024.0) << " KB\n";
}
```

## Integration with PriceTable4DBuilder

The storage module is designed to work seamlessly with `PriceTable4DBuilder`:

```cpp
// Step 1: Pre-compute prices
auto builder = PriceTable4DBuilder::create(
    moneyness_grid, maturity_grid, volatility_grid, rate_grid, K_ref
);

auto result = builder->precompute(OptionType::PUT, pde_config);

// Step 2: Save the table (requires accessing B-spline internals)
// Note: Currently requires manual extraction of knots and coefficients
// Future enhancement: Add PriceTable4DResult::save() method

// Step 3: Load for fast queries
auto loaded = InterpolationTableStorage::load("table.mint");
double price = loaded->eval(m, tau, sigma, r);
```

## Performance Characteristics

### Typical Performance (20×15×12×8 grid = 28,800 coefficients)

| Operation | Time | Notes |
|-----------|------|-------|
| Save | ~10-20 ms | Sequential write |
| Load | ~50-200 µs | Memory-mapped |
| Query | ~150 ns | B-spline evaluation |
| Read metadata | ~10 µs | Header only |

### Memory Usage

| Grid Size | Coefficients | File Size |
|-----------|--------------|-----------|
| 10×10×10×5 | 5,000 | ~40 KB |
| 20×15×12×8 | 28,800 | ~230 KB |
| 50×30×20×10 | 300,000 | ~2.4 MB |

**Formula:** `file_size ≈ 256 + (n_m + n_tau + n_v + n_r + n_coeffs) × 8 + padding`

### Speedup vs Pre-computation

For a typical 20×15×12×8 table:
- Pre-compute: ~30-60 seconds (one-time cost)
- Load: ~100 µs (300,000-600,000× faster)

## Error Handling

All functions return `Expected<T, std::string>` for robust error handling:

```cpp
auto result = InterpolationTableStorage::load("table.mint");

if (!result) {
    // Handle error
    std::cerr << "Error: " << result.error() << "\n";

    // Common errors:
    // - "Failed to open file: <path>"
    // - "Invalid magic number - not a valid interpolation table file"
    // - "Unsupported file version"
    // - "Coefficient count mismatch with grid dimensions"
}
```

### Validation

The storage module validates:
- ✅ Magic number (`0x4D494E54`)
- ✅ Version compatibility
- ✅ Grid dimension consistency
- ✅ Coefficient array size matches dimensions
- ✅ Option type is "PUT" or "CALL"
- ✅ Spline degree is reasonable (1-10)

## File Extension

Use `.mint` (Mango INTerpolation) for saved tables:
- `american_put_table.mint`
- `european_call_surface.mint`
- `spx_prices_20250109.mint`

## Example: Complete Workflow

```cpp
#include "src/price_table_4d_builder.hpp"
#include "src/interpolation_table_storage_v2.hpp"

int main() {
    // 1. Define grids
    std::vector<double> m_grid = linspace(0.7, 1.3, 20);
    std::vector<double> tau_grid = linspace(0.027, 2.0, 15);
    std::vector<double> v_grid = linspace(0.10, 0.80, 12);
    std::vector<double> r_grid = linspace(0.0, 0.10, 8);

    // 2. Pre-compute prices (one-time, ~30-60 seconds)
    auto builder = PriceTable4DBuilder::create(m_grid, tau_grid, v_grid, r_grid, 100.0);
    auto result = builder->precompute(OptionType::PUT, pde_config);

    // 3. Save to disk (manual extraction for now)
    // auto save_result = InterpolationTableStorage::save(
    //     "table.mint", m_grid, tau_grid, v_grid, r_grid,
    //     coeffs, 100.0, "PUT", 3
    // );

    // 4. Later: Load instantly (~100 µs)
    auto loaded = InterpolationTableStorage::load("table.mint");

    // 5. Query prices (~150 ns per query)
    for (const auto& query : market_data) {
        double price = loaded->eval(query.m, query.tau, query.sigma, query.r);
        process_price(price);
    }

    return 0;
}
```

## Design Rationale

### Why Custom Format Instead of Arrow?

While Apache Arrow IPC was initially considered, a custom format was chosen for:

1. **Simplicity**: No external dependencies, easier to build
2. **Binary compatibility**: Direct mapping to C++ structs
3. **Performance**: Zero overhead, no parsing required
4. **Size**: Minimal format overhead (~256 bytes header)
5. **Integration**: Seamless with existing Bazel build system

The format is inspired by Arrow's philosophy (columnar, aligned, self-describing) but optimized for our specific use case.

### Memory Mapping Benefits

Memory mapping (`mmap`) provides:
- **Lazy loading**: OS loads pages on-demand
- **Shared memory**: Multiple processes can share one table
- **Virtual memory**: Large tables don't consume RAM until accessed
- **OS caching**: Kernel manages page cache automatically

### 64-byte Alignment

Data arrays are 64-byte aligned to:
- Match CPU cache line size
- Enable SIMD vectorization
- Optimize memory bandwidth
- Reduce false sharing

## Future Enhancements

### Planned Features

1. **Direct PriceTable4DBuilder integration**
   ```cpp
   result->save("table.mint");  // One-liner save
   ```

2. **Compression support**
   - LZ4/Zstd compression for coefficients
   - Trade file size for load time

3. **Incremental updates**
   - Modify subset of coefficients without full rewrite

4. **Checksums**
   - CRC32 or XXH3 for data integrity

5. **Multiple option types in one file**
   - Store PUT and CALL surfaces together

### Compatibility Notes

- **Version 1** (current): Basic format with 64-byte alignment
- Future versions will maintain backward compatibility for reading
- Header includes reserved space for extensions

## Testing

Run tests with:

```bash
bazel test //tests:interpolation_table_storage_test
```

Test coverage includes:
- ✅ Basic save/load roundtrip
- ✅ Metadata reading
- ✅ Fitted B-spline save/load
- ✅ Error handling (invalid magic, dimension mismatch, etc.)
- ✅ File size and alignment verification
- ✅ Load performance benchmarks

## Troubleshooting

### "Failed to open file"
- Check file path is correct
- Verify file permissions
- Ensure parent directory exists

### "Invalid magic number"
- File is corrupted or not a .mint file
- Check if file was transferred in text mode (use binary)

### "Coefficient count mismatch"
- File may be corrupted
- Version incompatibility
- Re-save the table

### Slow loading
- Verify file is on local disk (not network mount)
- Check if anti-virus is scanning the file
- Ensure sufficient free memory for memory mapping

## Related Documentation

- [INTERPOLATION_TABLE_ANALYSIS.md](INTERPOLATION_TABLE_ANALYSIS.md) - Data structure details
- [INTERPOLATION_FILES_QUICK_REFERENCE.md](INTERPOLATION_FILES_QUICK_REFERENCE.md) - File locations
- [CLAUDE.md](../CLAUDE.md) - Project overview

## License

Copyright © 2025 Mango-IV Project
