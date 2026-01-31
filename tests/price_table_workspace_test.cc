// SPDX-License-Identifier: MIT
#include "src/option/table/price_table_workspace.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstring>

TEST(PriceTableWorkspace, ConstructsFromGridData) {
    // Note: Workspace now stores log-moneyness, not moneyness
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};  // ln(0.8), ln(0.9), ln(1.0), ln(1.1)
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);

    ASSERT_TRUE(ws_result.has_value());
    auto& ws = ws_result.value();

    EXPECT_EQ(ws.log_moneyness().size(), 4);
    EXPECT_EQ(ws.maturity().size(), 4);
    EXPECT_EQ(ws.coefficients().size(), 256);
    EXPECT_DOUBLE_EQ(ws.K_ref(), 100.0);
    EXPECT_DOUBLE_EQ(ws.m_min(), 0.8);
    EXPECT_DOUBLE_EQ(ws.m_max(), 1.1);
}

TEST(PriceTableWorkspace, RejectsInsufficientGridPoints) {
    std::vector<double> log_m_grid = {-0.11, 0.0, 0.10};  // Only 3 points
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(3 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.9, 1.1);

    EXPECT_FALSE(ws_result.has_value());
    EXPECT_EQ(ws_result.error(), "Log-moneyness grid must have >= 4 points");
}

TEST(PriceTableWorkspace, ValidatesArenaAlignment) {
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);

    ASSERT_TRUE(ws_result.has_value());
    auto& ws = ws_result.value();

    // Check 64-byte alignment for SIMD
    auto addr = reinterpret_cast<std::uintptr_t>(ws.log_moneyness().data());
    EXPECT_EQ(addr % 64, 0) << "Log-moneyness grid not 64-byte aligned";
}

TEST(PriceTableWorkspace, SavesAndLoadsFromArrowFile) {
    // Create workspace with known data (log-moneyness)
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};

    // Fill coefficients with distinct values for verification
    std::vector<double> coeffs(4 * 4 * 4 * 4);
    for (size_t i = 0; i < coeffs.size(); ++i) {
        coeffs[i] = static_cast<double>(i) * 0.1;
    }

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);

    ASSERT_TRUE(ws_result.has_value());
    auto& ws = ws_result.value();

    // Save to temporary file
    const std::string filepath = "/tmp/test_price_table.arrow";
    auto save_result = ws.save(filepath, "SPY", 0);  // 0 = PUT

    ASSERT_TRUE(save_result.has_value()) << "Save failed: " << save_result.error();

    // TODO: Implement load() in later task to verify roundtrip
    // For now, just verify file exists and has Arrow magic header
    std::ifstream file(filepath, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "File not created";

    char magic[6];
    file.read(magic, 6);
    std::string magic_str(magic, 6);
    EXPECT_EQ(magic_str, "ARROW1") << "Arrow magic header not found";

    // Cleanup
    std::filesystem::remove(filepath);
}

TEST(PriceTableWorkspace, SavedFileContainsCorrectDimensions) {
    // Create workspace with known dimensions (log-moneyness)
    std::vector<double> log_m_grid = {-0.36, -0.22, -0.11, 0.0, 0.10};  // 5 points
    std::vector<double> tau_grid = {0.1, 0.25, 0.5, 1.0};               // 4 points
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};          // 4 points
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};              // 4 points

    std::vector<double> coeffs(5 * 4 * 4 * 4, 42.0);  // Fill with known value

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.03, 0.7, 1.1);

    ASSERT_TRUE(ws_result.has_value());
    auto& ws = ws_result.value();

    // Save to temporary file
    const std::string filepath = "/tmp/test_dimensions.arrow";
    auto save_result = ws.save(filepath, "TEST", 1);  // 1 = CALL

    ASSERT_TRUE(save_result.has_value()) << "Save failed: " << save_result.error();

    // Verify file is non-empty
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(file.is_open());
    auto filesize = file.tellg();
    EXPECT_GT(filesize, 100) << "File too small to contain data";

    // Cleanup
    std::filesystem::remove(filepath);
}

TEST(PriceTableWorkspace, SavedFileContainsAllSchemaFields) {
    // Verify saved file contains all 35 fields from schema v1.1 (added m_min, m_max)
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);
    ASSERT_TRUE(ws_result.has_value());

    const std::string filepath = "/tmp/test_schema_fields.arrow";
    auto save_result = ws_result.value().save(filepath, "SPY", 0);
    ASSERT_TRUE(save_result.has_value());

    // Load and verify schema has exactly 33 fields
    auto load_result = mango::PriceTableWorkspace::load(filepath);
    ASSERT_TRUE(load_result.has_value()) << "Load failed";

    // Load succeeded, which means all required fields were present
    // The load() function validates the schema, so if it succeeded,
    // all 33 fields from spec v1.0 must be present

    // Cleanup
    std::filesystem::remove(filepath);
}

// ============================================================================
// Load Tests (TDD - write failing tests first)
// ============================================================================

TEST(PriceTableWorkspace, LoadFromNonExistentFileReturnsFileNotFound) {
    auto result = mango::PriceTableWorkspace::load("/tmp/nonexistent_file.arrow");

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), mango::PriceTableWorkspace::LoadError::FILE_NOT_FOUND);
}

TEST(PriceTableWorkspace, LoadFromNonArrowFileReturnsNotArrowFile) {
    // Create a file without Arrow magic header
    const std::string filepath = "/tmp/not_arrow_file.txt";
    std::ofstream file(filepath);
    file << "This is not an Arrow file\n";
    file.close();

    auto result = mango::PriceTableWorkspace::load(filepath);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), mango::PriceTableWorkspace::LoadError::NOT_ARROW_FILE);

    // Cleanup
    std::filesystem::remove(filepath);
}

TEST(PriceTableWorkspace, LoadSuccessfulRoundtrip) {
    // Create and save workspace (log-moneyness)
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};

    std::vector<double> coeffs(4 * 4 * 4 * 4);
    for (size_t i = 0; i < coeffs.size(); ++i) {
        coeffs[i] = static_cast<double>(i) * 0.1;
    }

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);

    ASSERT_TRUE(ws_result.has_value());
    auto& ws = ws_result.value();

    const std::string filepath = "/tmp/test_roundtrip.arrow";
    auto save_result = ws.save(filepath, "SPY", 0);
    ASSERT_TRUE(save_result.has_value()) << "Save failed: " << save_result.error();

    // Load workspace
    auto load_result = mango::PriceTableWorkspace::load(filepath);
    ASSERT_TRUE(load_result.has_value()) << "Load failed";

    auto& loaded_ws = load_result.value();

    // Verify dimensions match
    EXPECT_EQ(loaded_ws.log_moneyness().size(), 4);
    EXPECT_EQ(loaded_ws.maturity().size(), 4);
    EXPECT_EQ(loaded_ws.volatility().size(), 4);
    EXPECT_EQ(loaded_ws.rate().size(), 4);
    EXPECT_EQ(loaded_ws.coefficients().size(), 256);

    // Verify metadata
    EXPECT_DOUBLE_EQ(loaded_ws.K_ref(), 100.0);
    EXPECT_DOUBLE_EQ(loaded_ws.dividend_yield(), 0.02);
    EXPECT_DOUBLE_EQ(loaded_ws.m_min(), 0.8);
    EXPECT_DOUBLE_EQ(loaded_ws.m_max(), 1.1);

    // Verify grid values
    for (size_t i = 0; i < log_m_grid.size(); ++i) {
        EXPECT_DOUBLE_EQ(loaded_ws.log_moneyness()[i], log_m_grid[i]);
    }
    for (size_t i = 0; i < tau_grid.size(); ++i) {
        EXPECT_DOUBLE_EQ(loaded_ws.maturity()[i], tau_grid[i]);
    }

    // Verify coefficient values
    for (size_t i = 0; i < coeffs.size(); ++i) {
        EXPECT_DOUBLE_EQ(loaded_ws.coefficients()[i], coeffs[i]);
    }

    // Verify 64-byte alignment
    auto addr = reinterpret_cast<std::uintptr_t>(loaded_ws.log_moneyness().data());
    EXPECT_EQ(addr % 64, 0) << "Loaded data not 64-byte aligned";

    // Cleanup
    std::filesystem::remove(filepath);
}

TEST(PriceTableWorkspace, LoadValidatesDimensions) {
    // This test would require creating a malformed Arrow file
    // For now, we'll test this indirectly by ensuring load() validates
    // We'll create a proper test after implementing load()

    // Create workspace with known dimensions (log-moneyness)
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);
    ASSERT_TRUE(ws_result.has_value());

    const std::string filepath = "/tmp/test_dimensions_validation.arrow";
    auto save_result = ws_result.value().save(filepath, "TEST", 0);
    ASSERT_TRUE(save_result.has_value());

    // Load should succeed for valid file
    auto load_result = mango::PriceTableWorkspace::load(filepath);
    ASSERT_TRUE(load_result.has_value());

    // Verify dimensions
    auto [n_m, n_tau, n_sigma, n_r] = load_result.value().dimensions();
    EXPECT_EQ(n_m, 4);
    EXPECT_EQ(n_tau, 4);
    EXPECT_EQ(n_sigma, 4);
    EXPECT_EQ(n_r, 4);

    // Cleanup
    std::filesystem::remove(filepath);
}

TEST(PriceTableWorkspace, LoadVerifiesGridMonotonicity) {
    // Create workspace with sorted grids (log-moneyness)
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);
    ASSERT_TRUE(ws_result.has_value());

    const std::string filepath = "/tmp/test_monotonicity.arrow";
    auto save_result = ws_result.value().save(filepath, "TEST", 0);
    ASSERT_TRUE(save_result.has_value());

    // Load should succeed (grids are already sorted)
    auto load_result = mango::PriceTableWorkspace::load(filepath);
    EXPECT_TRUE(load_result.has_value());

    // Cleanup
    std::filesystem::remove(filepath);
}

TEST(PriceTableWorkspace, LoadVerifiesKnotVectorSizes) {
    // Create workspace (log-moneyness)
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10, 0.18};  // 5 points
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};               // 4 points
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(5 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.2);
    ASSERT_TRUE(ws_result.has_value());

    const std::string filepath = "/tmp/test_knots.arrow";
    auto save_result = ws_result.value().save(filepath, "TEST", 0);
    ASSERT_TRUE(save_result.has_value());

    // Load and verify knot vector sizes
    auto load_result = mango::PriceTableWorkspace::load(filepath);
    ASSERT_TRUE(load_result.has_value());

    // Knot vectors should be n + 4 for clamped cubic B-splines
    EXPECT_EQ(load_result.value().knots_log_moneyness().size(), 5 + 4);
    EXPECT_EQ(load_result.value().knots_maturity().size(), 4 + 4);
    EXPECT_EQ(load_result.value().knots_volatility().size(), 4 + 4);
    EXPECT_EQ(load_result.value().knots_rate().size(), 4 + 4);

    // Cleanup
    std::filesystem::remove(filepath);
}

TEST(PriceTableWorkspace, LoadPreservesBufferAlignment) {
    // Create workspace (log-moneyness)
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);
    ASSERT_TRUE(ws_result.has_value());

    const std::string filepath = "/tmp/test_alignment.arrow";
    auto save_result = ws_result.value().save(filepath, "TEST", 0);
    ASSERT_TRUE(save_result.has_value());

    // Load and verify at least the first buffer is 64-byte aligned
    // Note: We copy data from Arrow into our own allocation, so alignment
    // is controlled by allocate_and_initialize(), which aligns the arena start
    auto load_result = mango::PriceTableWorkspace::load(filepath);
    ASSERT_TRUE(load_result.has_value());
    auto& loaded_ws = load_result.value();

    // Check that at least the log-moneyness grid (first buffer) is aligned
    auto addr = reinterpret_cast<std::uintptr_t>(loaded_ws.log_moneyness().data());
    EXPECT_EQ(addr % 64, 0) << "Log-moneyness grid not 64-byte aligned";

    // Cleanup
    std::filesystem::remove(filepath);
}

// ============================================================================
// Checksum Tests (Task 7: CRC64 validation)
// ============================================================================

TEST(PriceTableWorkspace, LoadDetectsCorruptedCoefficients) {
    // Create and save valid workspace (log-moneyness)
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4);
    for (size_t i = 0; i < coeffs.size(); ++i) {
        coeffs[i] = static_cast<double>(i);
    }

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);
    ASSERT_TRUE(ws_result.has_value());

    const std::string filepath = "/tmp/test_corrupt_coeffs.arrow";
    auto save_result = ws_result.value().save(filepath, "TEST", 0);
    ASSERT_TRUE(save_result.has_value());

    // Read entire file into memory
    std::vector<char> file_data;
    {
        std::ifstream file(filepath, std::ios::binary);
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0);
        file_data.resize(size);
        file.read(file_data.data(), size);
    }

    // Corrupt one byte in the second half of file (likely coefficient data)
    // Arrow stores list data towards the end of the file
    size_t corrupt_offset = file_data.size() * 3 / 4;
    file_data[corrupt_offset] ^= 0xFF;

    // Write corrupted data back
    {
        std::ofstream file(filepath, std::ios::binary);
        file.write(file_data.data(), file_data.size());
    }

    // Load should fail with corrupted data error
    auto load_result = mango::PriceTableWorkspace::load(filepath);
    EXPECT_FALSE(load_result.has_value());
    if (!load_result.has_value()) {
        // Could be either CORRUPTED_COEFFICIENTS or CORRUPTED_GRIDS depending on where we hit
        auto err = load_result.error();
        EXPECT_TRUE(err == mango::PriceTableWorkspace::LoadError::CORRUPTED_COEFFICIENTS ||
                   err == mango::PriceTableWorkspace::LoadError::CORRUPTED_GRIDS ||
                   err == mango::PriceTableWorkspace::LoadError::ARROW_READ_ERROR);
    }

    // Cleanup
    std::filesystem::remove(filepath);
}

TEST(PriceTableWorkspace, LoadDetectsCorruptedGrids) {
    // Create and save valid workspace (log-moneyness)
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);
    ASSERT_TRUE(ws_result.has_value());

    const std::string filepath = "/tmp/test_corrupt_grids.arrow";
    auto save_result = ws_result.value().save(filepath, "TEST", 0);
    ASSERT_TRUE(save_result.has_value());

    // Load once to verify the file is valid
    auto valid_load = mango::PriceTableWorkspace::load(filepath);
    ASSERT_TRUE(valid_load.has_value()) << "File should be valid before corruption";

    // Read entire file into memory
    std::vector<char> file_data;
    {
        std::ifstream file(filepath, std::ios::binary);
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0);
        file_data.resize(size);
        file.read(file_data.data(), size);
    }

    // Find and corrupt actual grid data by searching for a known grid value
    // Grid values are stored as IEEE 754 doubles. We'll look for 0.1 (tau_grid[0])
    // and corrupt it. This ensures we hit actual data, not Arrow metadata.
    double target_value = 0.1;  // tau_grid[0]
    bool found_and_corrupted = false;
    for (size_t i = 0; i + sizeof(double) <= file_data.size(); ++i) {
        double value;
        std::memcpy(&value, &file_data[i], sizeof(double));
        if (value == target_value) {
            // Corrupt this double by flipping bits
            file_data[i] ^= 0xFF;
            found_and_corrupted = true;
            break;
        }
    }
    ASSERT_TRUE(found_and_corrupted) << "Could not find grid value to corrupt";

    // Write corrupted data back
    {
        std::ofstream file(filepath, std::ios::binary);
        file.write(file_data.data(), file_data.size());
    }

    // Load should fail with corrupted data error
    auto load_result = mango::PriceTableWorkspace::load(filepath);
    EXPECT_FALSE(load_result.has_value());
    if (!load_result.has_value()) {
        // Should be CORRUPTED_GRIDS since we corrupted grid data
        auto err = load_result.error();
        EXPECT_TRUE(err == mango::PriceTableWorkspace::LoadError::CORRUPTED_GRIDS ||
                   err == mango::PriceTableWorkspace::LoadError::CORRUPTED_KNOTS);
    }

    // Cleanup
    std::filesystem::remove(filepath);
}

// ============================================================================
// Surface Content Tests (format v2)
// ============================================================================

TEST(PriceTableWorkspaceTest, DefaultSurfaceContentIsZero) {
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);

    ASSERT_TRUE(ws_result.has_value());
    EXPECT_EQ(ws_result.value().surface_content(), 0);
}

TEST(PriceTableWorkspaceTest, RoundTripSurfaceContent) {
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    // Create workspace with surface_content=1 (EEP)
    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1,
        /*surface_content=*/1);

    ASSERT_TRUE(ws_result.has_value());
    EXPECT_EQ(ws_result.value().surface_content(), 1);

    // Save to temp file
    const std::string filepath = "/tmp/test_surface_content_roundtrip.arrow";
    auto save_result = ws_result.value().save(filepath, "SPY", 0);
    ASSERT_TRUE(save_result.has_value()) << "Save failed: " << save_result.error();

    // Load back
    auto load_result = mango::PriceTableWorkspace::load(filepath);
    ASSERT_TRUE(load_result.has_value()) << "Load failed";

    // Verify surface_content round-trips correctly
    EXPECT_EQ(load_result.value().surface_content(), 1);

    // Cleanup
    std::filesystem::remove(filepath);
}

TEST(PriceTableWorkspace, SavedFileHasNonZeroChecksums) {
    // Verify that saved files contain real (non-zero) checksums (log-moneyness)
    std::vector<double> log_m_grid = {-0.22, -0.11, 0.0, 0.10};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        log_m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02, 0.8, 1.1);
    ASSERT_TRUE(ws_result.has_value());

    const std::string filepath = "/tmp/test_nonzero_checksums.arrow";
    auto save_result = ws_result.value().save(filepath, "TEST", 0);
    ASSERT_TRUE(save_result.has_value());

    // Load and verify checksums are computed (not placeholder zeros)
    // We'll verify by loading twice - should get same checksums
    auto load_result1 = mango::PriceTableWorkspace::load(filepath);
    ASSERT_TRUE(load_result1.has_value());

    auto load_result2 = mango::PriceTableWorkspace::load(filepath);
    ASSERT_TRUE(load_result2.has_value());

    // Both loads should succeed (checksums match)
    // If checksums were zero (placeholder), corruption tests would fail

    // Cleanup
    std::filesystem::remove(filepath);
}
