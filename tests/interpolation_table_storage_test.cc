/**
 * @file interpolation_table_storage_test.cc
 * @brief Tests for interpolation table save/load functionality
 */

#include "src/interpolation_table_storage_v2.hpp"
#include "src/bspline_4d.hpp"
#include "src/bspline_fitter_4d.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <vector>
#include <fstream>

namespace fs = std::filesystem;

namespace mango {
namespace {

class InterpolationTableStorageTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directory for test files
        test_dir_ = fs::temp_directory_path() / "mango_storage_test";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        // Clean up test files
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }

    std::string get_test_file_path(const std::string& name) {
        return (test_dir_ / name).string();
    }

    fs::path test_dir_;
};

// Test basic save and load roundtrip
TEST_F(InterpolationTableStorageTest, BasicSaveLoad) {
    // Create simple test data (minimum 4 points per dimension for cubic B-splines)
    std::vector<double> m_knots = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_knots = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> v_knots = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_knots = {0.02, 0.04, 0.06, 0.08};

    // Create coefficients (just sequential numbers for testing)
    size_t n_coeffs = m_knots.size() * tau_knots.size() * v_knots.size() * r_knots.size();
    std::vector<double> coefficients(n_coeffs);
    for (size_t i = 0; i < n_coeffs; ++i) {
        coefficients[i] = static_cast<double>(i) / 100.0;
    }

    std::string filepath = get_test_file_path("test_basic.mint");

    // Save
    auto save_result = InterpolationTableStorage::save(
        filepath, m_knots, tau_knots, v_knots, r_knots,
        coefficients, 100.0, "PUT", 3
    );
    ASSERT_TRUE(save_result) << "Save failed: " << save_result.error();

    // Verify file exists
    ASSERT_TRUE(fs::exists(filepath));

    // Load
    auto load_result = InterpolationTableStorage::load(filepath);
    ASSERT_TRUE(load_result) << "Load failed: " << load_result.error();

    auto spline = std::move(*load_result);
    ASSERT_NE(spline, nullptr);

    // Test evaluation (just verify it doesn't crash and returns reasonable values)
    double price = spline->eval(1.0, 0.5, 0.20, 0.04);
    EXPECT_TRUE(std::isfinite(price));
}

// Test metadata reading
TEST_F(InterpolationTableStorageTest, ReadMetadata) {
    std::vector<double> m_knots = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_knots = {0.1, 0.5, 1.0, 1.5};
    std::vector<double> v_knots = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_knots = {0.02, 0.04, 0.06, 0.08};

    size_t n_coeffs = m_knots.size() * tau_knots.size() * v_knots.size() * r_knots.size();
    std::vector<double> coefficients(n_coeffs, 1.0);

    std::string filepath = get_test_file_path("test_metadata.mint");

    // Save
    auto save_result = InterpolationTableStorage::save(
        filepath, m_knots, tau_knots, v_knots, r_knots,
        coefficients, 105.5, "CALL", 3
    );
    ASSERT_TRUE(save_result);

    // Read metadata
    auto meta_result = InterpolationTableStorage::read_metadata(filepath);
    ASSERT_TRUE(meta_result) << "Read metadata failed: " << meta_result.error();

    auto meta = *meta_result;
    EXPECT_DOUBLE_EQ(meta.K_ref, 105.5);
    EXPECT_EQ(meta.option_type, "CALL");
    EXPECT_EQ(meta.spline_degree, 3);
    EXPECT_EQ(meta.n_moneyness, 5);
    EXPECT_EQ(meta.n_maturity, 4);
    EXPECT_EQ(meta.n_volatility, 4);
    EXPECT_EQ(meta.n_rate, 4);
    EXPECT_EQ(meta.n_coefficients, n_coeffs);
    EXPECT_GT(meta.file_size_bytes, 0);
}

// Test with actual B-spline fitting
TEST_F(InterpolationTableStorageTest, FittedSplineSaveLoad) {
    // Create a simple 4D function to fit: f(m,tau,v,r) = m + tau + v + r
    auto test_function = [](double m, double tau, double v, double r) {
        return m + tau + v + r;
    };

    // Create grids (minimum 4 points per dimension for cubic B-splines)
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_grid = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> v_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.04, 0.06, 0.08};

    // Generate data
    size_t n_data = m_grid.size() * tau_grid.size() * v_grid.size() * r_grid.size();
    std::vector<double> data(n_data);
    size_t idx = 0;
    for (double m : m_grid) {
        for (double tau : tau_grid) {
            for (double v : v_grid) {
                for (double r : r_grid) {
                    data[idx++] = test_function(m, tau, v, r);
                }
            }
        }
    }

    // Fit B-spline
    auto fitter_result = BSplineFitter4D::create(m_grid, tau_grid, v_grid, r_grid);
    ASSERT_TRUE(fitter_result.has_value()) << "Fitter creation failed: " << fitter_result.error();

    auto& fitter = fitter_result.value();
    auto fit_result = fitter.fit(data);
    ASSERT_TRUE(fit_result.success) << "Fitting failed: " << fit_result.error_message;

    auto coeffs = fit_result.coefficients;

    std::string filepath = get_test_file_path("test_fitted.mint");

    // Save
    auto save_result = InterpolationTableStorage::save(
        filepath, m_grid, tau_grid, v_grid, r_grid,
        coeffs, 100.0, "PUT", 3
    );
    ASSERT_TRUE(save_result);

    // Load
    auto load_result = InterpolationTableStorage::load(filepath);
    ASSERT_TRUE(load_result);

    auto spline = std::move(*load_result);

    // Test that loaded spline reproduces original function
    double m_test = 0.95;
    double tau_test = 0.75;
    double v_test = 0.18;
    double r_test = 0.035;

    double expected = test_function(m_test, tau_test, v_test, r_test);
    double actual = spline->eval(m_test, tau_test, v_test, r_test);

    EXPECT_NEAR(actual, expected, 0.01) << "Loaded spline evaluation mismatch";
}

// Test error handling: invalid magic number
TEST_F(InterpolationTableStorageTest, InvalidMagicNumber) {
    std::string filepath = get_test_file_path("test_invalid_magic.mint");

    // Create a file with wrong magic number
    std::ofstream file(filepath, std::ios::binary);
    InterpolationTableHeader header{};
    header.magic = 0xDEADBEEF; // Wrong magic
    header.version = 1;
    header.n_moneyness = 5;
    header.n_maturity = 3;
    header.n_volatility = 3;
    header.n_rate = 2;
    header.n_coefficients = 90;
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.close();

    // Try to load
    auto result = InterpolationTableStorage::load(filepath);
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("magic"), std::string::npos);
}

// Test error handling: dimension mismatch
TEST_F(InterpolationTableStorageTest, DimensionMismatch) {
    std::vector<double> m_knots = {0.8, 0.9, 1.0};
    std::vector<double> tau_knots = {0.1, 0.5};
    std::vector<double> v_knots = {0.15, 0.20};
    std::vector<double> r_knots = {0.02};

    // Wrong number of coefficients
    std::vector<double> coefficients = {1.0, 2.0, 3.0}; // Should be 3*2*2*1 = 12

    std::string filepath = get_test_file_path("test_mismatch.mint");

    auto result = InterpolationTableStorage::save(
        filepath, m_knots, tau_knots, v_knots, r_knots,
        coefficients, 100.0, "PUT", 3
    );

    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("mismatch"), std::string::npos);
}

// Test error handling: empty arrays
TEST_F(InterpolationTableStorageTest, EmptyArrays) {
    std::vector<double> empty;
    std::vector<double> valid = {1.0, 2.0};
    std::vector<double> coeffs = {1.0};

    std::string filepath = get_test_file_path("test_empty.mint");

    auto result = InterpolationTableStorage::save(
        filepath, empty, valid, valid, valid,
        coeffs, 100.0, "PUT", 3
    );

    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("Empty"), std::string::npos);
}

// Test error handling: invalid option type
TEST_F(InterpolationTableStorageTest, InvalidOptionType) {
    std::vector<double> knots = {1.0, 2.0};
    std::vector<double> coeffs(16, 1.0); // 2^4 = 16

    std::string filepath = get_test_file_path("test_invalid_type.mint");

    auto result = InterpolationTableStorage::save(
        filepath, knots, knots, knots, knots,
        coeffs, 100.0, "INVALID", 3
    );

    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("option_type"), std::string::npos);
}

// Test file size and alignment
TEST_F(InterpolationTableStorageTest, FileSizeAndAlignment) {
    std::vector<double> m_knots = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau_knots = {0.1, 0.5, 1.0};
    std::vector<double> v_knots = {0.15, 0.20};
    std::vector<double> r_knots = {0.02, 0.04, 0.06, 0.08};

    size_t n_coeffs = m_knots.size() * tau_knots.size() * v_knots.size() * r_knots.size();
    std::vector<double> coefficients(n_coeffs, 1.0);

    std::string filepath = get_test_file_path("test_size.mint");

    auto save_result = InterpolationTableStorage::save(
        filepath, m_knots, tau_knots, v_knots, r_knots,
        coefficients, 100.0, "PUT", 3
    );
    ASSERT_TRUE(save_result);

    // Check file size
    size_t file_size = fs::file_size(filepath);

    // Minimum size: header + all data arrays (with padding)
    size_t min_size = sizeof(InterpolationTableHeader);
    EXPECT_GT(file_size, min_size);

    // File size should be reasonable (not gigantic)
    size_t max_size = min_size + (m_knots.size() + tau_knots.size() +
                                  v_knots.size() + r_knots.size() +
                                  n_coeffs) * sizeof(double) * 2; // 2x for padding
    EXPECT_LT(file_size, max_size);
}

// Performance test: measure load time
TEST_F(InterpolationTableStorageTest, LoadPerformance) {
    // Create a larger table
    std::vector<double> m_knots(20), tau_knots(15), v_knots(10), r_knots(8);
    for (size_t i = 0; i < m_knots.size(); ++i) m_knots[i] = 0.5 + i * 0.1;
    for (size_t i = 0; i < tau_knots.size(); ++i) tau_knots[i] = 0.1 + i * 0.2;
    for (size_t i = 0; i < v_knots.size(); ++i) v_knots[i] = 0.1 + i * 0.05;
    for (size_t i = 0; i < r_knots.size(); ++i) r_knots[i] = 0.01 + i * 0.01;

    size_t n_coeffs = m_knots.size() * tau_knots.size() * v_knots.size() * r_knots.size();
    std::vector<double> coefficients(n_coeffs);
    for (size_t i = 0; i < n_coeffs; ++i) {
        coefficients[i] = std::sin(i * 0.1);
    }

    std::string filepath = get_test_file_path("test_perf.mint");

    auto save_result = InterpolationTableStorage::save(
        filepath, m_knots, tau_knots, v_knots, r_knots,
        coefficients, 100.0, "PUT", 3
    );
    ASSERT_TRUE(save_result);

    // Measure load time
    auto start = std::chrono::high_resolution_clock::now();
    auto load_result = InterpolationTableStorage::load(filepath);
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(load_result);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Load time for " << n_coeffs << " coefficients: "
              << duration.count() << " microseconds\n";

    // Loading should be fast (< 10ms for ~24k coefficients)
    EXPECT_LT(duration.count(), 10000);
}

} // anonymous namespace
} // namespace mango
