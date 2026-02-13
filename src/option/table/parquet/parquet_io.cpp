// SPDX-License-Identifier: MIT
#include "mango/option/table/parquet/parquet_io.hpp"
#include "mango/support/crc64.hpp"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace mango {

namespace {

// ============================================================================
// Constants
// ============================================================================

constexpr const char* FORMAT_VERSION = "2.0";

// ============================================================================
// Helpers
// ============================================================================

/// Return a SerializationFailed error.
PriceTableError serialization_error() {
    return PriceTableError{PriceTableErrorCode::SerializationFailed};
}

/// Check an Arrow Status and return PriceTableError on failure.
#define MANGO_ARROW_CHECK(expr)                   \
    do {                                          \
        auto _s = (expr);                         \
        if (!_s.ok()) {                           \
            return std::unexpected(               \
                serialization_error());           \
        }                                         \
    } while (0)

/// Check an Arrow Result and assign; return PriceTableError on failure.
#define MANGO_ARROW_ASSIGN(var, expr)             \
    auto _result_##var = (expr);                  \
    if (!_result_##var.ok()) {                    \
        return std::unexpected(                   \
            serialization_error());               \
    }                                             \
    auto var = std::move(_result_##var).ValueUnsafe()

std::expected<double, PriceTableError> parse_double(const std::string& s) {
    try {
        return std::stod(s);
    } catch (...) {
        return std::unexpected(serialization_error());
    }
}

std::expected<size_t, PriceTableError> parse_size_t(const std::string& s) {
    try {
        return std::stoull(s);
    } catch (...) {
        return std::unexpected(serialization_error());
    }
}

std::string option_type_to_string(OptionType t) {
    return t == OptionType::PUT ? "PUT" : "CALL";
}

std::expected<OptionType, PriceTableError>
string_to_option_type(const std::string& s) {
    if (s == "PUT") return OptionType::PUT;
    if (s == "CALL") return OptionType::CALL;
    return std::unexpected(serialization_error());
}

std::string double_to_string(double v) {
    std::ostringstream oss;
    oss.precision(17);
    oss << v;
    return oss.str();
}

std::string size_to_string(size_t v) {
    return std::to_string(v);
}

// ============================================================================
// Schema
// ============================================================================

std::shared_ptr<arrow::Schema> make_parquet_schema(
    const std::shared_ptr<arrow::KeyValueMetadata>& metadata) {

    auto list_double = arrow::list(arrow::float64());
    auto list_int32 = arrow::list(arrow::int32());

    auto fields = arrow::FieldVector{
        arrow::field("segment_id", arrow::int32()),
        arrow::field("K_ref", arrow::float64()),
        arrow::field("tau_start", arrow::float64()),
        arrow::field("tau_end", arrow::float64()),
        arrow::field("tau_min", arrow::float64()),
        arrow::field("tau_max", arrow::float64()),
        arrow::field("interp_type", arrow::utf8()),
        arrow::field("ndim", arrow::int32()),
        arrow::field("domain_lo", list_double),
        arrow::field("domain_hi", list_double),
        arrow::field("num_pts", list_int32),
        arrow::field("grid_0", list_double),
        arrow::field("grid_1", list_double),
        arrow::field("grid_2", list_double),
        arrow::field("grid_3", list_double),
        arrow::field("knots_0", list_double),
        arrow::field("knots_1", list_double),
        arrow::field("knots_2", list_double),
        arrow::field("knots_3", list_double),
        arrow::field("values", list_double),
        arrow::field("checksum_values", arrow::int64()),
    };

    return arrow::schema(fields, metadata);
}

// ============================================================================
// Write helpers
// ============================================================================

/// Append a std::vector<double> to a ListBuilder<DoubleBuilder>.
arrow::Status append_double_list(
    arrow::ListBuilder& list_builder,
    const std::vector<double>& vec) {

    ARROW_RETURN_NOT_OK(list_builder.Append());
    auto& value_builder =
        static_cast<arrow::DoubleBuilder&>(*list_builder.value_builder());
    for (double v : vec) {
        ARROW_RETURN_NOT_OK(value_builder.Append(v));
    }
    return arrow::Status::OK();
}

/// Append a std::vector<int32_t> to a ListBuilder<Int32Builder>.
arrow::Status append_int32_list(
    arrow::ListBuilder& list_builder,
    const std::vector<int32_t>& vec) {

    ARROW_RETURN_NOT_OK(list_builder.Append());
    auto& value_builder =
        static_cast<arrow::Int32Builder&>(*list_builder.value_builder());
    for (int32_t v : vec) {
        ARROW_RETURN_NOT_OK(value_builder.Append(v));
    }
    return arrow::Status::OK();
}

/// Append an empty list to a ListBuilder.
arrow::Status append_empty_double_list(arrow::ListBuilder& list_builder) {
    ARROW_RETURN_NOT_OK(list_builder.Append());
    // No values appended -> empty list
    return arrow::Status::OK();
}

// ============================================================================
// Read helpers
// ============================================================================

/// Extract a list-of-double column value at row i.
std::vector<double> read_double_list(
    const std::shared_ptr<arrow::Array>& col, int64_t row) {

    auto list_arr =
        std::static_pointer_cast<arrow::ListArray>(col);
    auto values_arr =
        std::static_pointer_cast<arrow::DoubleArray>(list_arr->values());

    int32_t start = list_arr->value_offset(row);
    int32_t end = list_arr->value_offset(row + 1);

    std::vector<double> result;
    result.reserve(static_cast<size_t>(end - start));
    for (int32_t j = start; j < end; ++j) {
        result.push_back(values_arr->Value(j));
    }
    return result;
}

/// Extract a list-of-int32 column value at row i.
std::vector<int32_t> read_int32_list(
    const std::shared_ptr<arrow::Array>& col, int64_t row) {

    auto list_arr =
        std::static_pointer_cast<arrow::ListArray>(col);
    auto values_arr =
        std::static_pointer_cast<arrow::Int32Array>(list_arr->values());

    int32_t start = list_arr->value_offset(row);
    int32_t end = list_arr->value_offset(row + 1);

    std::vector<int32_t> result;
    result.reserve(static_cast<size_t>(end - start));
    for (int32_t j = start; j < end; ++j) {
        result.push_back(values_arr->Value(j));
    }
    return result;
}

}  // anonymous namespace

// ============================================================================
// write_parquet
// ============================================================================

std::expected<void, PriceTableError>
write_parquet(const PriceTableData& data,
              const std::filesystem::path& path,
              const ParquetWriteOptions& opts) {

    auto pool = arrow::default_memory_pool();

    // ---- File-level metadata ----
    auto metadata = std::make_shared<arrow::KeyValueMetadata>();
    metadata->Append("mango.format_version", FORMAT_VERSION);
    metadata->Append("mango.surface_type", data.surface_type);
    metadata->Append("mango.option_type", option_type_to_string(data.option_type));
    metadata->Append("mango.dividend_yield", double_to_string(data.dividend_yield));
    metadata->Append("mango.maturity", double_to_string(data.maturity));
    metadata->Append("mango.bounds_m_min", double_to_string(data.bounds_m_min));
    metadata->Append("mango.bounds_m_max", double_to_string(data.bounds_m_max));
    metadata->Append("mango.bounds_tau_min", double_to_string(data.bounds_tau_min));
    metadata->Append("mango.bounds_tau_max", double_to_string(data.bounds_tau_max));
    metadata->Append("mango.bounds_sigma_min", double_to_string(data.bounds_sigma_min));
    metadata->Append("mango.bounds_sigma_max", double_to_string(data.bounds_sigma_max));
    metadata->Append("mango.bounds_rate_min", double_to_string(data.bounds_rate_min));
    metadata->Append("mango.bounds_rate_max", double_to_string(data.bounds_rate_max));
    metadata->Append("mango.n_pde_solves", size_to_string(data.n_pde_solves));
    metadata->Append("mango.precompute_time_seconds",
                     double_to_string(data.precompute_time_seconds));

    auto schema = make_parquet_schema(metadata);

    // ---- Column builders ----
    arrow::Int32Builder segment_id_b(pool);
    arrow::DoubleBuilder k_ref_b(pool);
    arrow::DoubleBuilder tau_start_b(pool);
    arrow::DoubleBuilder tau_end_b(pool);
    arrow::DoubleBuilder tau_min_b(pool);
    arrow::DoubleBuilder tau_max_b(pool);
    arrow::StringBuilder interp_type_b(pool);
    arrow::Int32Builder ndim_b(pool);

    arrow::ListBuilder domain_lo_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));
    arrow::ListBuilder domain_hi_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));
    arrow::ListBuilder num_pts_b(pool,
        std::make_shared<arrow::Int32Builder>(pool));

    arrow::ListBuilder grid_0_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));
    arrow::ListBuilder grid_1_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));
    arrow::ListBuilder grid_2_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));
    arrow::ListBuilder grid_3_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));

    arrow::ListBuilder knots_0_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));
    arrow::ListBuilder knots_1_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));
    arrow::ListBuilder knots_2_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));
    arrow::ListBuilder knots_3_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));

    arrow::ListBuilder values_b(pool,
        std::make_shared<arrow::DoubleBuilder>(pool));
    arrow::Int64Builder checksum_b(pool);

    // ---- Populate rows ----
    for (const auto& seg : data.segments) {
        MANGO_ARROW_CHECK(segment_id_b.Append(seg.segment_id));
        MANGO_ARROW_CHECK(k_ref_b.Append(seg.K_ref));
        MANGO_ARROW_CHECK(tau_start_b.Append(seg.tau_start));
        MANGO_ARROW_CHECK(tau_end_b.Append(seg.tau_end));
        MANGO_ARROW_CHECK(tau_min_b.Append(seg.tau_min));
        MANGO_ARROW_CHECK(tau_max_b.Append(seg.tau_max));
        MANGO_ARROW_CHECK(interp_type_b.Append(seg.interp_type));
        MANGO_ARROW_CHECK(ndim_b.Append(static_cast<int32_t>(seg.ndim)));

        MANGO_ARROW_CHECK(append_double_list(domain_lo_b, seg.domain_lo));
        MANGO_ARROW_CHECK(append_double_list(domain_hi_b, seg.domain_hi));
        MANGO_ARROW_CHECK(append_int32_list(num_pts_b, seg.num_pts));

        // grids: up to 4 axes; pad with empty lists if fewer
        arrow::ListBuilder* grid_builders[] = {
            &grid_0_b, &grid_1_b, &grid_2_b, &grid_3_b};
        for (size_t d = 0; d < 4; ++d) {
            if (d < seg.grids.size()) {
                MANGO_ARROW_CHECK(
                    append_double_list(*grid_builders[d], seg.grids[d]));
            } else {
                MANGO_ARROW_CHECK(
                    append_empty_double_list(*grid_builders[d]));
            }
        }

        // knots: up to 4 axes; pad with empty lists if fewer
        arrow::ListBuilder* knots_builders[] = {
            &knots_0_b, &knots_1_b, &knots_2_b, &knots_3_b};
        for (size_t d = 0; d < 4; ++d) {
            if (d < seg.knots.size()) {
                MANGO_ARROW_CHECK(
                    append_double_list(*knots_builders[d], seg.knots[d]));
            } else {
                MANGO_ARROW_CHECK(
                    append_empty_double_list(*knots_builders[d]));
            }
        }

        MANGO_ARROW_CHECK(append_double_list(values_b, seg.values));

        uint64_t crc = CRC64::compute(seg.values.data(), seg.values.size());
        MANGO_ARROW_CHECK(
            checksum_b.Append(static_cast<int64_t>(crc)));
    }

    // ---- Finalize arrays ----
    std::shared_ptr<arrow::Array> segment_id_a, k_ref_a, tau_start_a,
        tau_end_a, tau_min_a, tau_max_a, interp_type_a, ndim_a,
        domain_lo_a, domain_hi_a, num_pts_a,
        grid_0_a, grid_1_a, grid_2_a, grid_3_a,
        knots_0_a, knots_1_a, knots_2_a, knots_3_a,
        values_a, checksum_a;

    MANGO_ARROW_CHECK(segment_id_b.Finish(&segment_id_a));
    MANGO_ARROW_CHECK(k_ref_b.Finish(&k_ref_a));
    MANGO_ARROW_CHECK(tau_start_b.Finish(&tau_start_a));
    MANGO_ARROW_CHECK(tau_end_b.Finish(&tau_end_a));
    MANGO_ARROW_CHECK(tau_min_b.Finish(&tau_min_a));
    MANGO_ARROW_CHECK(tau_max_b.Finish(&tau_max_a));
    MANGO_ARROW_CHECK(interp_type_b.Finish(&interp_type_a));
    MANGO_ARROW_CHECK(ndim_b.Finish(&ndim_a));
    MANGO_ARROW_CHECK(domain_lo_b.Finish(&domain_lo_a));
    MANGO_ARROW_CHECK(domain_hi_b.Finish(&domain_hi_a));
    MANGO_ARROW_CHECK(num_pts_b.Finish(&num_pts_a));
    MANGO_ARROW_CHECK(grid_0_b.Finish(&grid_0_a));
    MANGO_ARROW_CHECK(grid_1_b.Finish(&grid_1_a));
    MANGO_ARROW_CHECK(grid_2_b.Finish(&grid_2_a));
    MANGO_ARROW_CHECK(grid_3_b.Finish(&grid_3_a));
    MANGO_ARROW_CHECK(knots_0_b.Finish(&knots_0_a));
    MANGO_ARROW_CHECK(knots_1_b.Finish(&knots_1_a));
    MANGO_ARROW_CHECK(knots_2_b.Finish(&knots_2_a));
    MANGO_ARROW_CHECK(knots_3_b.Finish(&knots_3_a));
    MANGO_ARROW_CHECK(values_b.Finish(&values_a));
    MANGO_ARROW_CHECK(checksum_b.Finish(&checksum_a));

    // ---- Build table ----
    auto table = arrow::Table::Make(schema, {
        segment_id_a, k_ref_a, tau_start_a, tau_end_a, tau_min_a, tau_max_a,
        interp_type_a, ndim_a,
        domain_lo_a, domain_hi_a, num_pts_a,
        grid_0_a, grid_1_a, grid_2_a, grid_3_a,
        knots_0_a, knots_1_a, knots_2_a, knots_3_a,
        values_a, checksum_a,
    });

    // ---- Writer properties ----
    auto props_builder = parquet::WriterProperties::Builder();
    switch (opts.compression) {
        case ParquetCompression::NONE:
            props_builder.compression(arrow::Compression::UNCOMPRESSED);
            break;
        case ParquetCompression::SNAPPY:
            props_builder.compression(arrow::Compression::SNAPPY);
            break;
        case ParquetCompression::ZSTD:
            props_builder.compression(arrow::Compression::ZSTD);
            break;
    }
    auto writer_props = props_builder.build();

    auto arrow_props = parquet::ArrowWriterProperties::Builder()
        .store_schema()->build();

    // ---- Write file ----
    MANGO_ARROW_ASSIGN(outfile,
        arrow::io::FileOutputStream::Open(path.string()));

    MANGO_ARROW_CHECK(parquet::arrow::WriteTable(
        *table, pool, outfile, /*chunk_size=*/1024,
        writer_props, arrow_props));

    return {};
}

// ============================================================================
// read_parquet
// ============================================================================

std::expected<PriceTableData, PriceTableError>
read_parquet(const std::filesystem::path& path) {

    auto pool = arrow::default_memory_pool();

    // ---- Open file ----
    MANGO_ARROW_ASSIGN(infile,
        arrow::io::ReadableFile::Open(path.string()));

    MANGO_ARROW_ASSIGN(reader,
        parquet::arrow::OpenFile(infile, pool));

    std::shared_ptr<arrow::Table> table;
    {
        auto status = reader->ReadTable(&table);
        if (!status.ok()) {
            return std::unexpected(serialization_error());
        }
    }

    // ---- Read file-level metadata ----
    auto kv = table->schema()->metadata();
    if (!kv) {
        return std::unexpected(serialization_error());
    }

    PriceTableData data;

    // Helper to find metadata value by key
    auto get_meta = [&](const std::string& key)
        -> std::expected<std::string, PriceTableError> {
        auto idx = kv->FindKey(key);
        if (idx < 0) {
            return std::unexpected(serialization_error());
        }
        return kv->value(idx);
    };

    // Validate format version first
    {
        auto v = get_meta("mango.format_version");
        if (!v) return std::unexpected(v.error());
        if (*v != FORMAT_VERSION) {
            return std::unexpected(serialization_error());
        }
    }
    {
        auto v = get_meta("mango.surface_type");
        if (!v) return std::unexpected(v.error());
        data.surface_type = *v;
    }
    {
        auto v = get_meta("mango.option_type");
        if (!v) return std::unexpected(v.error());
        auto opt = string_to_option_type(*v);
        if (!opt) return std::unexpected(opt.error());
        data.option_type = *opt;
    }
    {
        auto v = get_meta("mango.dividend_yield");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.dividend_yield = *parsed;
    }
    {
        auto v = get_meta("mango.maturity");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.maturity = *parsed;
    }

    // ---- Read SurfaceBounds ----
    {
        auto v = get_meta("mango.bounds_m_min");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.bounds_m_min = *parsed;
    }
    {
        auto v = get_meta("mango.bounds_m_max");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.bounds_m_max = *parsed;
    }
    {
        auto v = get_meta("mango.bounds_tau_min");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.bounds_tau_min = *parsed;
    }
    {
        auto v = get_meta("mango.bounds_tau_max");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.bounds_tau_max = *parsed;
    }
    {
        auto v = get_meta("mango.bounds_sigma_min");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.bounds_sigma_min = *parsed;
    }
    {
        auto v = get_meta("mango.bounds_sigma_max");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.bounds_sigma_max = *parsed;
    }
    {
        auto v = get_meta("mango.bounds_rate_min");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.bounds_rate_min = *parsed;
    }
    {
        auto v = get_meta("mango.bounds_rate_max");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.bounds_rate_max = *parsed;
    }

    {
        auto v = get_meta("mango.n_pde_solves");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_size_t(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.n_pde_solves = *parsed;
    }
    {
        auto v = get_meta("mango.precompute_time_seconds");
        if (!v) return std::unexpected(v.error());
        auto parsed = parse_double(*v);
        if (!parsed) return std::unexpected(parsed.error());
        data.precompute_time_seconds = *parsed;
    }

    // Also set dividend_yield in the DividendSpec
    data.dividends.dividend_yield = data.dividend_yield;

    // ---- Column accessors ----
    // Get column by name, checking existence
    auto get_col = [&](const std::string& name)
        -> std::expected<std::shared_ptr<arrow::ChunkedArray>, PriceTableError> {
        auto col = table->GetColumnByName(name);
        if (!col) {
            return std::unexpected(serialization_error());
        }
        return col;
    };

    // For simplicity, combine chunks into a single array per column
    auto get_array = [&](const std::string& name)
        -> std::expected<std::shared_ptr<arrow::Array>, PriceTableError> {
        auto col_result = get_col(name);
        if (!col_result) return std::unexpected(col_result.error());
        auto col = *col_result;

        if (col->num_chunks() == 0) {
            return std::unexpected(serialization_error());
        }
        if (col->num_chunks() == 1) {
            return col->chunk(0);
        }
        // Combine chunks
        auto combined = arrow::Concatenate(col->chunks(), pool);
        if (!combined.ok()) {
            return std::unexpected(serialization_error());
        }
        return *combined;
    };

    // Get all column arrays
    auto segment_id_res = get_array("segment_id");
    auto k_ref_res = get_array("K_ref");
    auto tau_start_res = get_array("tau_start");
    auto tau_end_res = get_array("tau_end");
    auto tau_min_res = get_array("tau_min");
    auto tau_max_res = get_array("tau_max");
    auto interp_type_res = get_array("interp_type");
    auto ndim_res = get_array("ndim");
    auto domain_lo_res = get_array("domain_lo");
    auto domain_hi_res = get_array("domain_hi");
    auto num_pts_res = get_array("num_pts");
    auto grid_0_res = get_array("grid_0");
    auto grid_1_res = get_array("grid_1");
    auto grid_2_res = get_array("grid_2");
    auto grid_3_res = get_array("grid_3");
    auto knots_0_res = get_array("knots_0");
    auto knots_1_res = get_array("knots_1");
    auto knots_2_res = get_array("knots_2");
    auto knots_3_res = get_array("knots_3");
    auto values_res = get_array("values");
    auto checksum_res = get_array("checksum_values");

    // Check all columns exist
    if (!segment_id_res || !k_ref_res || !tau_start_res || !tau_end_res ||
        !tau_min_res || !tau_max_res || !interp_type_res || !ndim_res ||
        !domain_lo_res || !domain_hi_res || !num_pts_res ||
        !grid_0_res || !grid_1_res || !grid_2_res || !grid_3_res ||
        !knots_0_res || !knots_1_res || !knots_2_res || !knots_3_res ||
        !values_res || !checksum_res) {
        return std::unexpected(serialization_error());
    }

    // Validate column types before casting
    auto check_type = [](const std::shared_ptr<arrow::Array>& arr,
                         arrow::Type::type expected) {
        return arr->type_id() == expected;
    };

    if (!check_type(*segment_id_res, arrow::Type::INT32) ||
        !check_type(*k_ref_res, arrow::Type::DOUBLE) ||
        !check_type(*tau_start_res, arrow::Type::DOUBLE) ||
        !check_type(*tau_end_res, arrow::Type::DOUBLE) ||
        !check_type(*tau_min_res, arrow::Type::DOUBLE) ||
        !check_type(*tau_max_res, arrow::Type::DOUBLE) ||
        !check_type(*interp_type_res, arrow::Type::STRING) ||
        !check_type(*ndim_res, arrow::Type::INT32) ||
        !check_type(*domain_lo_res, arrow::Type::LIST) ||
        !check_type(*domain_hi_res, arrow::Type::LIST) ||
        !check_type(*num_pts_res, arrow::Type::LIST) ||
        !check_type(*grid_0_res, arrow::Type::LIST) ||
        !check_type(*grid_1_res, arrow::Type::LIST) ||
        !check_type(*grid_2_res, arrow::Type::LIST) ||
        !check_type(*grid_3_res, arrow::Type::LIST) ||
        !check_type(*knots_0_res, arrow::Type::LIST) ||
        !check_type(*knots_1_res, arrow::Type::LIST) ||
        !check_type(*knots_2_res, arrow::Type::LIST) ||
        !check_type(*knots_3_res, arrow::Type::LIST) ||
        !check_type(*values_res, arrow::Type::LIST) ||
        !check_type(*checksum_res, arrow::Type::INT64)) {
        return std::unexpected(serialization_error());
    }

    auto segment_id_a = std::static_pointer_cast<arrow::Int32Array>(*segment_id_res);
    auto k_ref_a = std::static_pointer_cast<arrow::DoubleArray>(*k_ref_res);
    auto tau_start_a = std::static_pointer_cast<arrow::DoubleArray>(*tau_start_res);
    auto tau_end_a = std::static_pointer_cast<arrow::DoubleArray>(*tau_end_res);
    auto tau_min_a = std::static_pointer_cast<arrow::DoubleArray>(*tau_min_res);
    auto tau_max_a = std::static_pointer_cast<arrow::DoubleArray>(*tau_max_res);
    auto interp_type_a = std::static_pointer_cast<arrow::StringArray>(*interp_type_res);
    auto ndim_a = std::static_pointer_cast<arrow::Int32Array>(*ndim_res);
    auto checksum_a = std::static_pointer_cast<arrow::Int64Array>(*checksum_res);

    int64_t n_rows = table->num_rows();
    data.segments.resize(static_cast<size_t>(n_rows));

    for (int64_t i = 0; i < n_rows; ++i) {
        auto& seg = data.segments[static_cast<size_t>(i)];

        seg.segment_id = segment_id_a->Value(i);
        seg.K_ref = k_ref_a->Value(i);
        seg.tau_start = tau_start_a->Value(i);
        seg.tau_end = tau_end_a->Value(i);
        seg.tau_min = tau_min_a->Value(i);
        seg.tau_max = tau_max_a->Value(i);
        seg.interp_type = interp_type_a->GetString(i);
        seg.ndim = static_cast<size_t>(ndim_a->Value(i));

        seg.domain_lo = read_double_list(*domain_lo_res, i);
        seg.domain_hi = read_double_list(*domain_hi_res, i);
        seg.num_pts = read_int32_list(*num_pts_res, i);

        // Read grids: grid_0..grid_3, only include non-empty
        std::shared_ptr<arrow::Array> grid_arrays[] = {
            *grid_0_res, *grid_1_res, *grid_2_res, *grid_3_res};
        seg.grids.clear();
        for (size_t d = 0; d < seg.ndim && d < 4; ++d) {
            auto vec = read_double_list(grid_arrays[d], i);
            seg.grids.push_back(std::move(vec));
        }

        // Read knots: knots_0..knots_3, only include non-empty
        std::shared_ptr<arrow::Array> knots_arrays[] = {
            *knots_0_res, *knots_1_res, *knots_2_res, *knots_3_res};
        seg.knots.clear();
        for (size_t d = 0; d < seg.ndim && d < 4; ++d) {
            auto vec = read_double_list(knots_arrays[d], i);
            seg.knots.push_back(std::move(vec));
        }

        seg.values = read_double_list(*values_res, i);

        // Verify checksum
        uint64_t stored_crc =
            static_cast<uint64_t>(checksum_a->Value(i));
        uint64_t computed_crc =
            CRC64::compute(seg.values.data(), seg.values.size());
        if (stored_crc != computed_crc) {
            return std::unexpected(
                PriceTableError{PriceTableErrorCode::SerializationFailed});
        }
    }

    return data;
}

}  // namespace mango
