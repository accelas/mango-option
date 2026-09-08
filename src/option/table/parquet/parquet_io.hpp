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

/// Write format 4.0 with original ratio endpoints. Segmented tables also
/// require strike bounds and fixed-expiry provenance; invalid metadata is refused.
/// All numerical model/domain metadata is bound to the payload checksum.
[[nodiscard]] std::expected<void, PriceTableError>
write_parquet(const PriceTableData& data,
              const std::filesystem::path& path,
              const ParquetWriteOptions& opts = {});

/// Read format 4.0 PriceTableData. Older formats are refused because they
/// lack original ratio endpoints or other required model/domain provenance.
[[nodiscard]] std::expected<PriceTableData, PriceTableError>
read_parquet(const std::filesystem::path& path);

}  // namespace mango
