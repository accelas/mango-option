#include "src/option/table/price_table_workspace.hpp"
#include "src/support/crc64.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <chrono>
#include <fstream>
#include <optional>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

namespace mango {

std::expected<void, std::string> PriceTableWorkspace::validate_inputs(
    std::span<const double> m_grid,
    std::span<const double> tau_grid,
    std::span<const double> sigma_grid,
    std::span<const double> r_grid,
    std::span<const double> coefficients)
{
    // Validate grid sizes
    if (m_grid.size() < 4) {
        return std::unexpected("Log-moneyness grid must have >= 4 points");
    }
    if (tau_grid.size() < 4) {
        return std::unexpected("Maturity grid must have >= 4 points");
    }
    if (sigma_grid.size() < 4) {
        return std::unexpected("Volatility grid must have >= 4 points");
    }
    if (r_grid.size() < 4) {
        return std::unexpected("Rate grid must have >= 4 points");
    }

    // Validate coefficient size
    size_t expected_size = m_grid.size() * tau_grid.size() *
                          sigma_grid.size() * r_grid.size();
    if (coefficients.size() != expected_size) {
        return std::unexpected("Coefficient size mismatch: expected " +
                         std::to_string(expected_size) + ", got " +
                         std::to_string(coefficients.size()));
    }

    // Validate monotonicity
    auto is_sorted = [](std::span<const double> v) {
        return std::is_sorted(v.begin(), v.end());
    };

    if (!is_sorted(m_grid)) {
        return std::unexpected("Log-moneyness grid must be sorted ascending");
    }
    if (!is_sorted(tau_grid)) {
        return std::unexpected("Maturity grid must be sorted ascending");
    }
    if (!is_sorted(sigma_grid)) {
        return std::unexpected("Volatility grid must be sorted ascending");
    }
    if (!is_sorted(r_grid)) {
        return std::unexpected("Rate grid must be sorted ascending");
    }

    return {};
}

std::expected<PriceTableWorkspace, std::string> PriceTableWorkspace::allocate_and_initialize(
    std::span<const double> m_grid,
    std::span<const double> tau_grid,
    std::span<const double> sigma_grid,
    std::span<const double> r_grid,
    std::span<const double> coefficients,
    double K_ref,
    double dividend_yield)
{
    PriceTableWorkspace ws;

    // Compute knot vectors (clamped cubic B-spline)
    auto knots_m = clamped_knots_cubic(m_grid);
    auto knots_tau = clamped_knots_cubic(tau_grid);
    auto knots_sigma = clamped_knots_cubic(sigma_grid);
    auto knots_r = clamped_knots_cubic(r_grid);

    // Calculate total arena size
    size_t total_size = m_grid.size() + tau_grid.size() +
                       sigma_grid.size() + r_grid.size() +
                       knots_m.size() + knots_tau.size() +
                       knots_sigma.size() + knots_r.size() +
                       coefficients.size();

    // Allocate with 64-byte alignment for AVX-512
    // Use over-allocation to ensure alignment
    ws.arena_.resize(total_size + 8);  // +8 for alignment padding

    // Find 64-byte aligned start within arena
    auto arena_ptr = reinterpret_cast<std::uintptr_t>(ws.arena_.data());
    auto aligned_offset = (64 - (arena_ptr % 64)) % 64;
    double* aligned_start = ws.arena_.data() + aligned_offset / sizeof(double);

    // Copy data into arena
    double* ptr = aligned_start;

    std::memcpy(ptr, m_grid.data(), m_grid.size() * sizeof(double));
    ws.log_moneyness_ = std::span<const double>(ptr, m_grid.size());
    ptr += m_grid.size();

    std::memcpy(ptr, tau_grid.data(), tau_grid.size() * sizeof(double));
    ws.maturity_ = std::span<const double>(ptr, tau_grid.size());
    ptr += tau_grid.size();

    std::memcpy(ptr, sigma_grid.data(), sigma_grid.size() * sizeof(double));
    ws.volatility_ = std::span<const double>(ptr, sigma_grid.size());
    ptr += sigma_grid.size();

    std::memcpy(ptr, r_grid.data(), r_grid.size() * sizeof(double));
    ws.rate_ = std::span<const double>(ptr, r_grid.size());
    ptr += r_grid.size();

    std::memcpy(ptr, knots_m.data(), knots_m.size() * sizeof(double));
    ws.knots_m_ = std::span<const double>(ptr, knots_m.size());
    ptr += knots_m.size();

    std::memcpy(ptr, knots_tau.data(), knots_tau.size() * sizeof(double));
    ws.knots_tau_ = std::span<const double>(ptr, knots_tau.size());
    ptr += knots_tau.size();

    std::memcpy(ptr, knots_sigma.data(), knots_sigma.size() * sizeof(double));
    ws.knots_sigma_ = std::span<const double>(ptr, knots_sigma.size());
    ptr += knots_sigma.size();

    std::memcpy(ptr, knots_r.data(), knots_r.size() * sizeof(double));
    ws.knots_r_ = std::span<const double>(ptr, knots_r.size());
    ptr += knots_r.size();

    std::memcpy(ptr, coefficients.data(), coefficients.size() * sizeof(double));
    ws.coefficients_ = std::span<const double>(ptr, coefficients.size());

    ws.K_ref_ = K_ref;
    ws.dividend_yield_ = dividend_yield;

    return ws;
}

// Helper for zero-copy loading from raw buffers (friend of PriceTableWorkspace)
std::expected<PriceTableWorkspace, std::string> allocate_and_initialize_from_buffers(
    const double* m_data, size_t n_m,
    const double* tau_data, size_t n_tau,
    const double* sigma_data, size_t n_sigma,
    const double* r_data, size_t n_r,
    const double* coeff_data, size_t n_coeffs,
    double K_ref,
    double dividend_yield)
{
    PriceTableWorkspace ws;

    // Compute knot vectors (clamped cubic B-spline)
    // We need to create temporary vectors for clamped_knots_cubic
    std::vector<double> m_grid(m_data, m_data + n_m);
    std::vector<double> tau_grid(tau_data, tau_data + n_tau);
    std::vector<double> sigma_grid(sigma_data, sigma_data + n_sigma);
    std::vector<double> r_grid(r_data, r_data + n_r);

    auto knots_m = clamped_knots_cubic(m_grid);
    auto knots_tau = clamped_knots_cubic(tau_grid);
    auto knots_sigma = clamped_knots_cubic(sigma_grid);
    auto knots_r = clamped_knots_cubic(r_grid);

    // Calculate total arena size
    size_t total_size = n_m + n_tau + n_sigma + n_r +
                       knots_m.size() + knots_tau.size() +
                       knots_sigma.size() + knots_r.size() +
                       n_coeffs;

    // Allocate with 64-byte alignment for AVX-512
    ws.arena_.resize(total_size + 8);  // +8 for alignment padding

    // Find 64-byte aligned start within arena
    auto arena_ptr = reinterpret_cast<std::uintptr_t>(ws.arena_.data());
    auto aligned_offset = (64 - (arena_ptr % 64)) % 64;
    double* aligned_start = ws.arena_.data() + aligned_offset / sizeof(double);

    // Copy data into arena (single copy from Arrow buffers)
    double* ptr = aligned_start;

    std::memcpy(ptr, m_data, n_m * sizeof(double));
    ws.log_moneyness_ = std::span<const double>(ptr, n_m);
    ptr += n_m;

    std::memcpy(ptr, tau_data, n_tau * sizeof(double));
    ws.maturity_ = std::span<const double>(ptr, n_tau);
    ptr += n_tau;

    std::memcpy(ptr, sigma_data, n_sigma * sizeof(double));
    ws.volatility_ = std::span<const double>(ptr, n_sigma);
    ptr += n_sigma;

    std::memcpy(ptr, r_data, n_r * sizeof(double));
    ws.rate_ = std::span<const double>(ptr, n_r);
    ptr += n_r;

    std::memcpy(ptr, knots_m.data(), knots_m.size() * sizeof(double));
    ws.knots_m_ = std::span<const double>(ptr, knots_m.size());
    ptr += knots_m.size();

    std::memcpy(ptr, knots_tau.data(), knots_tau.size() * sizeof(double));
    ws.knots_tau_ = std::span<const double>(ptr, knots_tau.size());
    ptr += knots_tau.size();

    std::memcpy(ptr, knots_sigma.data(), knots_sigma.size() * sizeof(double));
    ws.knots_sigma_ = std::span<const double>(ptr, knots_sigma.size());
    ptr += knots_sigma.size();

    std::memcpy(ptr, knots_r.data(), knots_r.size() * sizeof(double));
    ws.knots_r_ = std::span<const double>(ptr, knots_r.size());
    ptr += knots_r.size();

    std::memcpy(ptr, coeff_data, n_coeffs * sizeof(double));
    ws.coefficients_ = std::span<const double>(ptr, n_coeffs);

    ws.K_ref_ = K_ref;
    ws.dividend_yield_ = dividend_yield;

    return ws;
}

std::expected<PriceTableWorkspace, std::string> PriceTableWorkspace::create(
    std::span<const double> log_m_grid,
    std::span<const double> tau_grid,
    std::span<const double> sigma_grid,
    std::span<const double> r_grid,
    std::span<const double> coefficients,
    double K_ref,
    double dividend_yield,
    double m_min,
    double m_max)
{
    // Validate inputs first
    auto validation = validate_inputs(log_m_grid, tau_grid, sigma_grid, r_grid, coefficients);
    if (!validation) {
        return std::unexpected(validation.error());
    }

    // Allocate and initialize workspace
    auto result = allocate_and_initialize(log_m_grid, tau_grid, sigma_grid, r_grid,
                                          coefficients, K_ref, dividend_yield);
    if (result) {
        result->m_min_ = m_min;
        result->m_max_ = m_max;
    }
    return result;
}

std::expected<void, std::string> PriceTableWorkspace::save(
    const std::string& filepath,
    const std::string& ticker,
    uint8_t option_type) const
{
    // Create Arrow schema according to spec v1.0
    auto schema = arrow::schema({
        // Metadata (scalar fields)
        arrow::field("format_version", arrow::uint32()),
        arrow::field("created_timestamp", arrow::timestamp(arrow::TimeUnit::MICRO)),
        arrow::field("mango_version", arrow::utf8()),  // Placeholder: "dev" for now

        // Option parameters
        arrow::field("ticker", arrow::utf8()),
        arrow::field("option_type", arrow::uint8()),
        arrow::field("K_ref", arrow::float64()),
        arrow::field("dividend_yield", arrow::float64()),

        // Moneyness bounds (original S/K before log transform)
        arrow::field("m_min", arrow::float64()),
        arrow::field("m_max", arrow::float64()),

        // Grid dimensions
        arrow::field("n_log_moneyness", arrow::uint32()),
        arrow::field("n_maturity", arrow::uint32()),
        arrow::field("n_volatility", arrow::uint32()),
        arrow::field("n_rate", arrow::uint32()),

        // Grid vectors (1D arrays)
        arrow::field("log_moneyness", arrow::list(arrow::float64())),
        arrow::field("maturity", arrow::list(arrow::float64())),
        arrow::field("volatility", arrow::list(arrow::float64())),
        arrow::field("rate", arrow::list(arrow::float64())),

        // Knot vectors
        arrow::field("knots_log_moneyness", arrow::list(arrow::float64())),
        arrow::field("knots_maturity", arrow::list(arrow::float64())),
        arrow::field("knots_volatility", arrow::list(arrow::float64())),
        arrow::field("knots_rate", arrow::list(arrow::float64())),

        // Coefficients (4D tensor in row-major layout)
        arrow::field("coefficients", arrow::list(arrow::float64())),

        // Optional: raw prices (not yet implemented, save as null)
        arrow::field("prices_raw", arrow::list(arrow::float64()), /*nullable=*/true),

        // Fitting statistics (placeholders: 0.0 until fitting is implemented)
        arrow::field("max_residual_axis0oneyness", arrow::float64()),
        arrow::field("max_residual_axis0aturity", arrow::float64()),
        arrow::field("max_residual_volatility", arrow::float64()),
        arrow::field("max_residual_axis3ate", arrow::float64()),
        arrow::field("max_residual_overall", arrow::float64()),
        arrow::field("condition_number_max", arrow::float64()),

        // Build metadata (placeholders: 0 until build tracking is implemented)
        arrow::field("n_pde_solves", arrow::uint32()),
        arrow::field("precompute_time_seconds", arrow::float64()),
        arrow::field("pde_n_space", arrow::uint32()),
        arrow::field("pde_n_time", arrow::uint32()),

        // Checksums (placeholder 0 for now, Task 7 adds CRC64)
        arrow::field("checksum_coefficients", arrow::uint64()),
        arrow::field("checksum_grids", arrow::uint64()),
    });

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    auto [n_m, n_tau, n_sigma, n_r] = dimensions();

    // Build column data using Arrow builders
    arrow::UInt32Builder format_version_builder;
    arrow::TimestampBuilder timestamp_builder(arrow::timestamp(arrow::TimeUnit::MICRO),
                                             arrow::default_memory_pool());
    arrow::StringBuilder mango_version_builder;
    arrow::StringBuilder ticker_builder;
    arrow::UInt8Builder option_type_builder;
    arrow::DoubleBuilder k_ref_builder;
    arrow::DoubleBuilder dividend_builder;

    arrow::DoubleBuilder m_min_builder, m_max_builder;
    arrow::UInt32Builder n_m_builder, n_tau_builder, n_sigma_builder, n_r_builder;

    // List builders for grid vectors
    arrow::ListBuilder log_moneyness_list_builder(arrow::default_memory_pool(),
        std::make_shared<arrow::DoubleBuilder>());
    arrow::ListBuilder maturity_list_builder(arrow::default_memory_pool(),
        std::make_shared<arrow::DoubleBuilder>());
    arrow::ListBuilder volatility_list_builder(arrow::default_memory_pool(),
        std::make_shared<arrow::DoubleBuilder>());
    arrow::ListBuilder rate_list_builder(arrow::default_memory_pool(),
        std::make_shared<arrow::DoubleBuilder>());

    // List builders for knot vectors
    arrow::ListBuilder knots_m_list_builder(arrow::default_memory_pool(),
        std::make_shared<arrow::DoubleBuilder>());
    arrow::ListBuilder knots_tau_list_builder(arrow::default_memory_pool(),
        std::make_shared<arrow::DoubleBuilder>());
    arrow::ListBuilder knots_sigma_list_builder(arrow::default_memory_pool(),
        std::make_shared<arrow::DoubleBuilder>());
    arrow::ListBuilder knots_r_list_builder(arrow::default_memory_pool(),
        std::make_shared<arrow::DoubleBuilder>());

    // List builder for coefficients
    arrow::ListBuilder coeffs_list_builder(arrow::default_memory_pool(),
        std::make_shared<arrow::DoubleBuilder>());

    // List builder for prices_raw (nullable)
    arrow::ListBuilder prices_raw_list_builder(arrow::default_memory_pool(),
        std::make_shared<arrow::DoubleBuilder>());

    // Builders for fitting statistics
    arrow::DoubleBuilder max_res_m_builder, max_res_tau_builder, max_res_sigma_builder;
    arrow::DoubleBuilder max_res_r_builder, max_res_overall_builder, cond_num_builder;

    // Builders for build metadata
    arrow::UInt32Builder n_pde_solves_builder, pde_n_space_builder, pde_n_time_builder;
    arrow::DoubleBuilder precompute_time_builder;

    arrow::UInt64Builder checksum_coeffs_builder, checksum_grids_builder;

    // Append scalar values
    if (!format_version_builder.Append(1).ok() ||
        !timestamp_builder.Append(micros).ok() ||
        !mango_version_builder.Append("dev").ok() ||  // Placeholder: "dev"
        !ticker_builder.Append(ticker).ok() ||
        !option_type_builder.Append(option_type).ok() ||
        !k_ref_builder.Append(K_ref_).ok() ||
        !dividend_builder.Append(dividend_yield_).ok() ||
        !m_min_builder.Append(m_min_).ok() ||
        !m_max_builder.Append(m_max_).ok() ||
        !n_m_builder.Append(static_cast<uint32_t>(n_m)).ok() ||
        !n_tau_builder.Append(static_cast<uint32_t>(n_tau)).ok() ||
        !n_sigma_builder.Append(static_cast<uint32_t>(n_sigma)).ok() ||
        !n_r_builder.Append(static_cast<uint32_t>(n_r)).ok())
    {
        return std::unexpected("Failed to append scalar values");
    }

    // Helper to append list data
    auto append_list = [](arrow::ListBuilder& list_builder,
                         std::span<const double> data) -> bool {
        if (!list_builder.Append().ok()) return false;
        auto* value_builder = static_cast<arrow::DoubleBuilder*>(list_builder.value_builder());
        return value_builder->AppendValues(data.data(), data.size()).ok();
    };

    // Append grid vectors
    if (!append_list(log_moneyness_list_builder, log_moneyness_) ||
        !append_list(maturity_list_builder, maturity_) ||
        !append_list(volatility_list_builder, volatility_) ||
        !append_list(rate_list_builder, rate_))
    {
        return std::unexpected("Failed to append grid vectors");
    }

    // Append knot vectors
    if (!append_list(knots_m_list_builder, knots_m_) ||
        !append_list(knots_tau_list_builder, knots_tau_) ||
        !append_list(knots_sigma_list_builder, knots_sigma_) ||
        !append_list(knots_r_list_builder, knots_r_))
    {
        return std::unexpected("Failed to append knot vectors");
    }

    // Append coefficients
    if (!append_list(coeffs_list_builder, coefficients_)) {
        return std::unexpected("Failed to append coefficients");
    }

    // Append prices_raw as null (not yet implemented)
    if (!prices_raw_list_builder.AppendNull().ok()) {
        return std::unexpected("Failed to append prices_raw (null)");
    }

    // Append fitting statistics (placeholders: 0.0)
    if (!max_res_m_builder.Append(0.0).ok() ||
        !max_res_tau_builder.Append(0.0).ok() ||
        !max_res_sigma_builder.Append(0.0).ok() ||
        !max_res_r_builder.Append(0.0).ok() ||
        !max_res_overall_builder.Append(0.0).ok() ||
        !cond_num_builder.Append(0.0).ok())
    {
        return std::unexpected("Failed to append fitting statistics");
    }

    // Append build metadata (placeholders: 0)
    if (!n_pde_solves_builder.Append(0).ok() ||
        !precompute_time_builder.Append(0.0).ok() ||
        !pde_n_space_builder.Append(0).ok() ||
        !pde_n_time_builder.Append(0).ok())
    {
        return std::unexpected("Failed to append build metadata");
    }

    // Compute CRC64 checksums for data integrity
    // 1. Checksum for coefficients array
    uint64_t checksum_coefficients = CRC64::compute(coefficients_.data(), coefficients_.size());

    // 2. Checksum for all grid vectors (concatenated)
    // Allocate temporary buffer for concatenated grids
    std::vector<double> all_grids;
    all_grids.reserve(log_moneyness_.size() + maturity_.size() +
                     volatility_.size() + rate_.size());
    all_grids.insert(all_grids.end(), log_moneyness_.begin(), log_moneyness_.end());
    all_grids.insert(all_grids.end(), maturity_.begin(), maturity_.end());
    all_grids.insert(all_grids.end(), volatility_.begin(), volatility_.end());
    all_grids.insert(all_grids.end(), rate_.begin(), rate_.end());
    uint64_t checksum_grids = CRC64::compute(all_grids.data(), all_grids.size());

    // Append checksums (real CRC64 values)
    if (!checksum_coeffs_builder.Append(checksum_coefficients).ok() ||
        !checksum_grids_builder.Append(checksum_grids).ok())
    {
        return std::unexpected("Failed to append checksums");
    }

    // Finish all builders and create arrays
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(35);  // Updated to match 35 fields (added m_min, m_max)

    auto finish_builder = [&arrays](arrow::ArrayBuilder& builder) -> bool {
        std::shared_ptr<arrow::Array> array;
        if (!builder.Finish(&array).ok()) return false;
        arrays.push_back(array);
        return true;
    };

    if (!finish_builder(format_version_builder) ||
        !finish_builder(timestamp_builder) ||
        !finish_builder(mango_version_builder) ||
        !finish_builder(ticker_builder) ||
        !finish_builder(option_type_builder) ||
        !finish_builder(k_ref_builder) ||
        !finish_builder(dividend_builder) ||
        !finish_builder(m_min_builder) ||
        !finish_builder(m_max_builder) ||
        !finish_builder(n_m_builder) ||
        !finish_builder(n_tau_builder) ||
        !finish_builder(n_sigma_builder) ||
        !finish_builder(n_r_builder) ||
        !finish_builder(log_moneyness_list_builder) ||
        !finish_builder(maturity_list_builder) ||
        !finish_builder(volatility_list_builder) ||
        !finish_builder(rate_list_builder) ||
        !finish_builder(knots_m_list_builder) ||
        !finish_builder(knots_tau_list_builder) ||
        !finish_builder(knots_sigma_list_builder) ||
        !finish_builder(knots_r_list_builder) ||
        !finish_builder(coeffs_list_builder) ||
        !finish_builder(prices_raw_list_builder) ||
        !finish_builder(max_res_m_builder) ||
        !finish_builder(max_res_tau_builder) ||
        !finish_builder(max_res_sigma_builder) ||
        !finish_builder(max_res_r_builder) ||
        !finish_builder(max_res_overall_builder) ||
        !finish_builder(cond_num_builder) ||
        !finish_builder(n_pde_solves_builder) ||
        !finish_builder(precompute_time_builder) ||
        !finish_builder(pde_n_space_builder) ||
        !finish_builder(pde_n_time_builder) ||
        !finish_builder(checksum_coeffs_builder) ||
        !finish_builder(checksum_grids_builder))
    {
        return std::unexpected("Failed to finish builders");
    }

    // Create record batch (single row)
    auto record_batch = arrow::RecordBatch::Make(schema, 1, arrays);

    // Open file for writing
    auto file_result = arrow::io::FileOutputStream::Open(filepath);
    if (!file_result.ok()) {
        return std::unexpected("Failed to open file: " + file_result.status().ToString());
    }

    // Write using Arrow IPC (Feather V2 format)
    auto writer_result = arrow::ipc::MakeFileWriter(*file_result, schema);
    if (!writer_result.ok()) {
        return std::unexpected("Failed to create Arrow writer: " + writer_result.status().ToString());
    }

    auto writer = *writer_result;
    if (!writer->WriteRecordBatch(*record_batch).ok()) {
        return std::unexpected("Failed to write record batch");
    }

    if (!writer->Close().ok()) {
        return std::unexpected("Failed to close Arrow writer");
    }

    return {};
}

std::expected<PriceTableWorkspace, PriceTableWorkspace::LoadError>
PriceTableWorkspace::load(const std::string& filepath)
{
    // 1. Check if file exists
    std::ifstream test_file(filepath);
    if (!test_file.good()) {
        return std::unexpected(LoadError::FILE_NOT_FOUND);
    }
    test_file.close();

    // 2. Open file using Arrow memory-mapped IO for zero-copy
    auto mmap_result = arrow::io::MemoryMappedFile::Open(filepath, arrow::io::FileMode::READ);
    if (!mmap_result.ok()) {
        return std::unexpected(LoadError::MMAP_FAILED);
    }
    auto mmap_file = *mmap_result;

    // 3. Create IPC file reader
    auto reader_result = arrow::ipc::RecordBatchFileReader::Open(mmap_file);
    if (!reader_result.ok()) {
        // Check if it's a generic Arrow read error or specifically not an Arrow file
        std::string error_msg = reader_result.status().ToString();
        if (error_msg.find("Not an Arrow file") != std::string::npos ||
            error_msg.find("Invalid") != std::string::npos) {
            return std::unexpected(LoadError::NOT_ARROW_FILE);
        }
        return std::unexpected(LoadError::ARROW_READ_ERROR);
    }
    auto reader = *reader_result;

    // 4. Read the single record batch
    if (reader->num_record_batches() != 1) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }

    auto batch_result = reader->ReadRecordBatch(0);
    if (!batch_result.ok()) {
        return std::unexpected(LoadError::ARROW_READ_ERROR);
    }
    auto batch = *batch_result;

    // 5. Helper to extract scalar fields
    auto get_scalar = [&batch](const std::string& name) -> std::shared_ptr<arrow::Scalar> {
        auto column = batch->GetColumnByName(name);
        if (!column) return nullptr;
        auto scalar_result = column->GetScalar(0);
        if (!scalar_result.ok()) return nullptr;
        return *scalar_result;
    };

    // 6a. Helper to get raw Arrow buffer pointer (zero-copy)
    struct BufferView {
        const double* data;
        size_t size;
    };

    auto get_arrow_buffer = [&batch](const std::string& name) -> std::optional<BufferView> {
        auto column = batch->GetColumnByName(name);
        if (!column) return std::nullopt;

        auto list_array = std::dynamic_pointer_cast<arrow::ListArray>(column);
        if (!list_array || list_array->length() != 1) return std::nullopt;

        int64_t start = list_array->value_offset(0);
        int64_t end = list_array->value_offset(1);
        int64_t list_length = end - start;

        auto values = std::dynamic_pointer_cast<arrow::DoubleArray>(list_array->values());
        if (!values) return std::nullopt;

        return BufferView{values->raw_values() + start, static_cast<size_t>(list_length)};
    };

    // 6b. Fallback helper for cases where we still need vectors (CRC computation)
    auto get_list_values = [&batch](const std::string& name) -> std::vector<double> {
        auto column = batch->GetColumnByName(name);
        if (!column) return {};

        auto list_array = std::dynamic_pointer_cast<arrow::ListArray>(column);
        if (!list_array) return {};

        // For a single-row batch, extract the list at row 0
        if (list_array->length() != 1) return {};

        // Get the slice of values for row 0
        int64_t start = list_array->value_offset(0);
        int64_t end = list_array->value_offset(1);
        int64_t list_length = end - start;

        auto values = std::dynamic_pointer_cast<arrow::DoubleArray>(list_array->values());
        if (!values) return {};

        std::vector<double> result(list_length);
        for (int64_t i = 0; i < list_length; ++i) {
            result[i] = values->Value(start + i);
        }

        return result;
    };

    // 7. Validate format version
    auto version_scalar = get_scalar("format_version");
    if (!version_scalar) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }
    auto version_uint32 = std::dynamic_pointer_cast<arrow::UInt32Scalar>(version_scalar);
    if (!version_uint32) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }
    uint32_t format_version = version_uint32->value;
    if (format_version != 1) {
        return std::unexpected(LoadError::UNSUPPORTED_VERSION);
    }

    // 8. Extract dimensions
    auto n_m_scalar = get_scalar("n_log_moneyness");
    auto n_tau_scalar = get_scalar("n_maturity");
    auto n_sigma_scalar = get_scalar("n_volatility");
    auto n_r_scalar = get_scalar("n_rate");

    if (!n_m_scalar || !n_tau_scalar || !n_sigma_scalar || !n_r_scalar) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }

    auto n_m_uint32 = std::dynamic_pointer_cast<arrow::UInt32Scalar>(n_m_scalar);
    auto n_tau_uint32 = std::dynamic_pointer_cast<arrow::UInt32Scalar>(n_tau_scalar);
    auto n_sigma_uint32 = std::dynamic_pointer_cast<arrow::UInt32Scalar>(n_sigma_scalar);
    auto n_r_uint32 = std::dynamic_pointer_cast<arrow::UInt32Scalar>(n_r_scalar);

    if (!n_m_uint32 || !n_tau_uint32 || !n_sigma_uint32 || !n_r_uint32) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }

    uint32_t n_m = n_m_uint32->value;
    uint32_t n_tau = n_tau_uint32->value;
    uint32_t n_sigma = n_sigma_uint32->value;
    uint32_t n_r = n_r_uint32->value;

    // 9. Validate dimensions (must be >= 4 for cubic B-splines)
    if (n_m < 4 || n_tau < 4 || n_sigma < 4 || n_r < 4) {
        return std::unexpected(LoadError::INSUFFICIENT_GRID_POINTS);
    }

    // 10. Extract metadata
    auto k_ref_scalar = get_scalar("K_ref");
    auto div_scalar = get_scalar("dividend_yield");
    auto m_min_scalar = get_scalar("m_min");
    auto m_max_scalar = get_scalar("m_max");

    if (!k_ref_scalar || !div_scalar || !m_min_scalar || !m_max_scalar) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }

    auto k_ref_double = std::dynamic_pointer_cast<arrow::DoubleScalar>(k_ref_scalar);
    auto div_double = std::dynamic_pointer_cast<arrow::DoubleScalar>(div_scalar);
    auto m_min_double = std::dynamic_pointer_cast<arrow::DoubleScalar>(m_min_scalar);
    auto m_max_double = std::dynamic_pointer_cast<arrow::DoubleScalar>(m_max_scalar);

    if (!k_ref_double || !div_double || !m_min_double || !m_max_double) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }

    double K_ref = k_ref_double->value;
    double dividend_yield = div_double->value;
    double m_min = m_min_double->value;
    double m_max = m_max_double->value;

    // 11. Extract grid buffer views (zero-copy)
    auto m_view = get_arrow_buffer("log_moneyness");
    auto tau_view = get_arrow_buffer("maturity");
    auto sigma_view = get_arrow_buffer("volatility");
    auto r_view = get_arrow_buffer("rate");

    if (!m_view || !tau_view || !sigma_view || !r_view) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }

    // 12. Validate array sizes match metadata
    if (m_view->size != n_m || tau_view->size != n_tau ||
        sigma_view->size != n_sigma || r_view->size != n_r) {
        return std::unexpected(LoadError::SIZE_MISMATCH);
    }

    // 13. Extract knot vectors
    auto knots_m = get_list_values("knots_log_moneyness");
    auto knots_tau = get_list_values("knots_maturity");
    auto knots_sigma = get_list_values("knots_volatility");
    auto knots_r = get_list_values("knots_rate");

    // 14. Validate knot vector sizes (should be n + 4 for clamped cubic B-splines)
    // Note: Schema doc incorrectly stated n + 8, but clamped_knots_cubic() returns n + 4
    if (knots_m.size() != n_m + 4 || knots_tau.size() != n_tau + 4 ||
        knots_sigma.size() != n_sigma + 4 || knots_r.size() != n_r + 4) {
        return std::unexpected(LoadError::SIZE_MISMATCH);
    }

    // 14b. Validate knot values match recomputed knots from grids
    // Recompute knots from loaded grids (need temporary vectors)
    std::vector<double> m_grid(m_view->data, m_view->data + m_view->size);
    std::vector<double> tau_grid(tau_view->data, tau_view->data + tau_view->size);
    std::vector<double> sigma_grid(sigma_view->data, sigma_view->data + sigma_view->size);
    std::vector<double> r_grid(r_view->data, r_view->data + r_view->size);

    auto knots_m_computed = clamped_knots_cubic(m_grid);
    auto knots_tau_computed = clamped_knots_cubic(tau_grid);
    auto knots_sigma_computed = clamped_knots_cubic(sigma_grid);
    auto knots_r_computed = clamped_knots_cubic(r_grid);

    // Compare with tolerance for floating-point errors
    auto knots_match = [](std::span<const double> a, std::span<const double> b) {
        if (a.size() != b.size()) return false;
        return std::equal(a.begin(), a.end(), b.begin(),
                         [](double x, double y) { return std::abs(x - y) < 1e-14; });
    };

    if (!knots_match(knots_m, knots_m_computed) ||
        !knots_match(knots_tau, knots_tau_computed) ||
        !knots_match(knots_sigma, knots_sigma_computed) ||
        !knots_match(knots_r, knots_r_computed)) {
        return std::unexpected(LoadError::CORRUPTED_KNOTS);
    }

    // 15. Extract coefficients buffer view (zero-copy)
    auto coeffs_view = get_arrow_buffer("coefficients");
    if (!coeffs_view) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }

    // 16. Validate coefficient size
    size_t expected_coeffs = static_cast<size_t>(n_m) * n_tau * n_sigma * n_r;
    if (coeffs_view->size != expected_coeffs) {
        return std::unexpected(LoadError::COEFFICIENT_SIZE_MISMATCH);
    }

    // 17. Validate grid monotonicity (using grids already created for knot validation)
    auto is_sorted = [](std::span<const double> v) {
        return std::is_sorted(v.begin(), v.end());
    };

    if (!is_sorted(m_grid) || !is_sorted(tau_grid) ||
        !is_sorted(sigma_grid) || !is_sorted(r_grid)) {
        return std::unexpected(LoadError::GRID_NOT_SORTED);
    }

    // 18. Extract and validate CRC64 checksums
    auto checksum_coeffs_scalar = get_scalar("checksum_coefficients");
    auto checksum_grids_scalar = get_scalar("checksum_grids");

    if (!checksum_coeffs_scalar || !checksum_grids_scalar) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }

    auto checksum_coeffs_uint64 = std::dynamic_pointer_cast<arrow::UInt64Scalar>(checksum_coeffs_scalar);
    auto checksum_grids_uint64 = std::dynamic_pointer_cast<arrow::UInt64Scalar>(checksum_grids_scalar);

    if (!checksum_coeffs_uint64 || !checksum_grids_uint64) {
        return std::unexpected(LoadError::SCHEMA_MISMATCH);
    }

    uint64_t stored_checksum_coeffs = checksum_coeffs_uint64->value;
    uint64_t stored_checksum_grids = checksum_grids_uint64->value;

    // Compute checksums directly from Arrow buffers (zero-copy)
    uint64_t computed_checksum_coeffs = CRC64::compute(coeffs_view->data, coeffs_view->size);

    // For grids, we need to concatenate for CRC (one unavoidable copy)
    std::vector<double> all_grids;
    all_grids.reserve(m_grid.size() + tau_grid.size() +
                     sigma_grid.size() + r_grid.size());
    all_grids.insert(all_grids.end(), m_grid.begin(), m_grid.end());
    all_grids.insert(all_grids.end(), tau_grid.begin(), tau_grid.end());
    all_grids.insert(all_grids.end(), sigma_grid.begin(), sigma_grid.end());
    all_grids.insert(all_grids.end(), r_grid.begin(), r_grid.end());
    uint64_t computed_checksum_grids = CRC64::compute(all_grids.data(), all_grids.size());

    // Validate checksums
    if (computed_checksum_coeffs != stored_checksum_coeffs) {
        return std::unexpected(LoadError::CORRUPTED_COEFFICIENTS);
    }

    if (computed_checksum_grids != stored_checksum_grids) {
        return std::unexpected(LoadError::CORRUPTED_GRIDS);
    }

    // 19. Create workspace using zero-copy path (single memcpy from Arrow buffers into arena)
    auto ws_result = allocate_and_initialize_from_buffers(
        m_view->data, m_view->size,
        tau_view->data, tau_view->size,
        sigma_view->data, sigma_view->size,
        r_view->data, r_view->size,
        coeffs_view->data, coeffs_view->size,
        K_ref, dividend_yield);

    if (!ws_result) {
        return std::unexpected(LoadError::ARROW_READ_ERROR);
    }

    // 20. Set moneyness bounds
    auto& ws = ws_result.value();
    ws.m_min_ = m_min;
    ws.m_max_ = m_max;

    // 21. Verify alignment of loaded data
    // Note: We rely on allocate_and_initialize() to ensure proper alignment
    // The alignment is best-effort and may not always be perfect for all grid sizes

    // Optional alignment check (disabled for now as it's not critical for correctness)
    // auto check_alignment = [](const void* ptr) -> bool {
    //     auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    //     return (addr % 64) == 0;
    // };
    // if (!check_alignment(ws.log_moneyness_.data()) ||
    //     !check_alignment(ws.coefficients_.data())) {
    //     return std::unexpected(LoadError::INVALID_ALIGNMENT);
    // }

    return std::move(ws);
}

}  // namespace mango
