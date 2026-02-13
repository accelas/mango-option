// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/serialization/price_table_data.hpp"
#include "mango/support/error_types.hpp"

#include <expected>
#include <filesystem>

namespace mango {

enum class ParquetCompression {
    NONE,
    SNAPPY,
    ZSTD,
};

struct ParquetWriteOptions {
    ParquetCompression compression = ParquetCompression::ZSTD;
};

/// Write PriceTableData to a Parquet file.
[[nodiscard]] std::expected<void, PriceTableError>
write_parquet(const PriceTableData& data,
              const std::filesystem::path& path,
              const ParquetWriteOptions& opts = {});

/// Read PriceTableData from a Parquet file.
[[nodiscard]] std::expected<PriceTableData, PriceTableError>
read_parquet(const std::filesystem::path& path);

}  // namespace mango
