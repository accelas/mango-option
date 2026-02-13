// SPDX-License-Identifier: MIT

// Parquet round-trip tests: to_data -> write_parquet -> read_parquet -> from_data -> verify
// Tests all surface types plus checksum verification and compression variants.

#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "mango/option/table/serialization/to_data.hpp"
#include "mango/option/table/serialization/from_data.hpp"
#include "mango/option/table/parquet/parquet_io.hpp"

// Surface type headers
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"

// Dimensionless 3D builder
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_3d_accessor.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/math/bspline_nd_separable.hpp"
#include "mango/math/bspline_nd.hpp"
#include "mango/math/bspline_basis.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"

namespace mango {
namespace {

// ===========================================================================
// Helper: convert S/K moneyness to log-moneyness
// ===========================================================================

std::vector<double> to_log_m(std::initializer_list<double> sk) {
    std::vector<double> out;
    out.reserve(sk.size());
    for (double m : sk) {
        out.push_back(std::log(m));
    }
    return out;
}

// ===========================================================================
// Test fixture: manages temp file creation and cleanup
// ===========================================================================

class ParquetIOTest : public ::testing::Test {
protected:
    std::filesystem::path temp_path_;

    void SetUp() override {
        temp_path_ = std::filesystem::temp_directory_path() /
                     ("mango_parquet_test_" +
                      std::to_string(::testing::UnitTest::GetInstance()
                                         ->current_test_info()->line()) +
                      ".parquet");
    }

    void TearDown() override {
        std::filesystem::remove(temp_path_);
    }
};

// ===========================================================================
// Helper: verify prices match between two surfaces at sample points (4D)
// ===========================================================================

template <typename Surface1, typename Surface2>
void verify_prices_match_4d(const Surface1& original, const Surface2& loaded,
                            double K_ref, double tolerance = 1e-10) {
    struct TestPoint { double spot, strike, tau, sigma, rate; };
    std::vector<TestPoint> test_points = {
        {K_ref,       K_ref,       0.5,  0.25, 0.05},
        {K_ref * 0.9, K_ref,       0.3,  0.20, 0.04},
        {K_ref * 1.1, K_ref,       0.8,  0.30, 0.03},
        {K_ref,       K_ref * 1.1, 1.0,  0.20, 0.05},
        {K_ref,       K_ref * 0.9, 0.25, 0.35, 0.04},
    };

    for (const auto& p : test_points) {
        double orig = original.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        double load = loaded.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(orig, load, tolerance)
            << "Mismatch at spot=" << p.spot << " strike=" << p.strike
            << " tau=" << p.tau << " sigma=" << p.sigma << " rate=" << p.rate;
    }
}

// ===========================================================================
// Helper: verify prices match between two surfaces at sample points (3D)
// ===========================================================================

template <typename Surface1, typename Surface2>
void verify_prices_match_3d(const Surface1& original, const Surface2& loaded,
                            double K_ref, double tolerance = 1e-10) {
    struct TestPoint { double spot, strike, tau, sigma, rate; };
    std::vector<TestPoint> test_points = {
        {K_ref,       K_ref,       1.0,  0.20, 0.05},
        {K_ref * 0.9, K_ref,       0.5,  0.25, 0.03},
        {K_ref * 1.1, K_ref,       1.5,  0.15, 0.04},
    };

    for (const auto& p : test_points) {
        double orig = original.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        double load = loaded.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(orig, load, tolerance)
            << "Mismatch at spot=" << p.spot << " strike=" << p.strike
            << " tau=" << p.tau << " sigma=" << p.sigma << " rate=" << p.rate;
    }
}

// ===========================================================================
// Test 1: BSpline 4D Parquet round-trip
// ===========================================================================

TEST_F(ParquetIOTest, BSpline4DRoundTrip) {
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value()) << "from_vectors failed";
    auto& [builder, axes] = *setup;

    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value()) << "build failed";

    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value()) << "make_bspline_surface failed";

    // to_data -> write_parquet
    auto data = to_data(*surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value()) << "write_parquet failed";

    // read_parquet -> from_data
    auto read_result = read_parquet(temp_path_);
    ASSERT_TRUE(read_result.has_value()) << "read_parquet failed";

    auto loaded = from_data<BSplineLeaf>(*read_result);
    ASSERT_TRUE(loaded.has_value()) << "from_data failed";

    verify_prices_match_4d(*surface, *loaded, 100.0);
}

// ===========================================================================
// Test 2: ChebyshevRaw 4D Parquet round-trip
// ===========================================================================

TEST_F(ParquetIOTest, ChebyshevRaw4DRoundTrip) {
    ChebyshevTableConfig config{
        .num_pts = {8, 6, 6, 4},
        .domain = Domain<4>{
            .lo = {-0.30, 0.02, 0.10, 0.02},
            .hi = { 0.30, 1.50, 0.40, 0.08},
        },
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .tucker_epsilon = 0.0,
    };
    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value()) << "build_chebyshev_table failed";
    auto& surface = std::get<ChebyshevRawSurface>(result->surface);

    auto data = to_data(surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value()) << "write_parquet failed";

    auto read_result = read_parquet(temp_path_);
    ASSERT_TRUE(read_result.has_value()) << "read_parquet failed";

    auto loaded = from_data<ChebyshevRawLeaf>(*read_result);
    ASSERT_TRUE(loaded.has_value()) << "from_data failed";

    verify_prices_match_4d(surface, *loaded, 100.0);
}

// ===========================================================================
// Test 3: Chebyshev Tucker 4D -> Parquet -> ChebyshevRawLeaf round-trip
// ===========================================================================

TEST_F(ParquetIOTest, ChebyshevTucker4DToRawRoundTrip) {
    ChebyshevTableConfig config{
        .num_pts = {8, 6, 6, 4},
        .domain = Domain<4>{
            .lo = {-0.30, 0.02, 0.10, 0.02},
            .hi = { 0.30, 1.50, 0.40, 0.08},
        },
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .tucker_epsilon = 1e-8,
    };
    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value()) << "build_chebyshev_table failed";
    auto& tucker_surface = std::get<ChebyshevSurface>(result->surface);

    auto data = to_data(tucker_surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value()) << "write_parquet failed";

    auto read_result = read_parquet(temp_path_);
    ASSERT_TRUE(read_result.has_value()) << "read_parquet failed";

    auto loaded = from_data<ChebyshevRawLeaf>(*read_result);
    ASSERT_TRUE(loaded.has_value()) << "from_data<ChebyshevRawLeaf> failed";

    verify_prices_match_4d(tucker_surface, *loaded, 100.0);
}

// ===========================================================================
// Test 4: BSpline 3D Parquet round-trip
// ===========================================================================

TEST_F(ParquetIOTest, BSpline3DRoundTrip) {
    constexpr double K_ref = 100.0;

    DimensionlessAxes axes;
    axes.log_moneyness = {-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30};
    axes.tau_prime = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16};
    axes.ln_kappa = {-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8};

    auto pde = solve_dimensionless_pde(axes, K_ref, OptionType::PUT);
    ASSERT_TRUE(pde.has_value())
        << "PDE solve failed: code=" << static_cast<int>(pde.error().code);

    Dimensionless3DAccessor accessor(pde->values, axes, K_ref);
    eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));

    std::array<std::vector<double>, 3> grids = {
        axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
    auto fitter = BSplineNDSeparable<double, 3>::create(grids);
    ASSERT_TRUE(fitter.has_value());
    auto fit = fitter->fit(std::move(pde->values));
    ASSERT_TRUE(fit.has_value());

    std::array<std::vector<double>, 3> bspline_grids = {
        axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
    std::array<std::vector<double>, 3> bspline_knots;
    for (size_t i = 0; i < 3; ++i) {
        bspline_knots[i] = clamped_knots_cubic(bspline_grids[i]);
    }
    auto spline = BSplineND<double, 3>::create(
        bspline_grids, std::move(bspline_knots), std::move(fit->coefficients));
    ASSERT_TRUE(spline.has_value());

    auto spline_ptr = std::make_shared<const BSplineND<double, 3>>(
        std::move(spline.value()));

    SharedBSplineInterp<3> interp(std::move(spline_ptr));
    DimensionlessTransform3D xform;
    BSpline3DTransformLeaf leaf(std::move(interp), xform, K_ref);
    AnalyticalEEP eep(OptionType::PUT, 0.0);
    BSpline3DLeaf eep_leaf(std::move(leaf), std::move(eep));

    const double sigma_min = 0.10;
    const double sigma_max = 0.80;
    SurfaceBounds bounds{
        .m_min = axes.log_moneyness.front(),
        .m_max = axes.log_moneyness.back(),
        .tau_min = 2.0 * axes.tau_prime.front() / (sigma_max * sigma_max),
        .tau_max = 2.0 * axes.tau_prime.back() / (sigma_min * sigma_min),
        .sigma_min = sigma_min,
        .sigma_max = sigma_max,
        .rate_min = 0.005,
        .rate_max = 0.10,
    };

    BSpline3DPriceTable surface(
        std::move(eep_leaf), bounds, OptionType::PUT, 0.0);

    auto data = to_data(surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value()) << "write_parquet failed";

    auto read_result = read_parquet(temp_path_);
    ASSERT_TRUE(read_result.has_value()) << "read_parquet failed";

    auto loaded = from_data<BSpline3DLeaf>(*read_result);
    ASSERT_TRUE(loaded.has_value()) << "from_data failed";

    verify_prices_match_3d(surface, *loaded, K_ref);
}

// ===========================================================================
// Test 5: Chebyshev 3D Tucker -> Parquet -> Chebyshev3DRawLeaf round-trip
// ===========================================================================

TEST_F(ParquetIOTest, Chebyshev3DTuckerToRawRoundTrip) {
    constexpr double K_ref = 100.0;

    DimensionlessAxes axes;
    axes.log_moneyness = {-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30};
    axes.tau_prime = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16};
    axes.ln_kappa = {-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8};

    auto pde = solve_dimensionless_pde(axes, K_ref, OptionType::PUT);
    ASSERT_TRUE(pde.has_value())
        << "PDE solve failed: code=" << static_cast<int>(pde.error().code);

    Dimensionless3DAccessor accessor(pde->values, axes, K_ref);
    eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));

    std::array<size_t, 3> num_pts = {
        axes.log_moneyness.size(),
        axes.tau_prime.size(),
        axes.ln_kappa.size(),
    };
    Domain<3> domain{
        .lo = {axes.log_moneyness.front(), axes.tau_prime.front(), axes.ln_kappa.front()},
        .hi = {axes.log_moneyness.back(), axes.tau_prime.back(), axes.ln_kappa.back()},
    };

    auto cheb = ChebyshevInterpolant<3, TuckerTensor<3>>::build_from_values(
        std::span<const double>(pde->values),
        domain, num_pts, 1e-8);

    DimensionlessTransform3D xform;
    Chebyshev3DTransformLeaf tleaf(std::move(cheb), xform, K_ref);
    AnalyticalEEP eep_fn(OptionType::PUT, 0.0);
    Chebyshev3DLeaf eep_leaf(std::move(tleaf), std::move(eep_fn));

    const double sigma_min = 0.10;
    const double sigma_max = 0.80;
    SurfaceBounds bounds{
        .m_min = axes.log_moneyness.front(),
        .m_max = axes.log_moneyness.back(),
        .tau_min = 2.0 * axes.tau_prime.front() / (sigma_max * sigma_max),
        .tau_max = 2.0 * axes.tau_prime.back() / (sigma_min * sigma_min),
        .sigma_min = sigma_min,
        .sigma_max = sigma_max,
        .rate_min = 0.005,
        .rate_max = 0.10,
    };

    Chebyshev3DPriceTable tucker_surface(
        std::move(eep_leaf), bounds, OptionType::PUT, 0.0);

    auto data = to_data(tucker_surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value()) << "write_parquet failed";

    auto read_result = read_parquet(temp_path_);
    ASSERT_TRUE(read_result.has_value()) << "read_parquet failed";

    auto loaded = from_data<Chebyshev3DRawLeaf>(*read_result);
    ASSERT_TRUE(loaded.has_value()) << "from_data<Chebyshev3DRawLeaf> failed";

    verify_prices_match_3d(tucker_surface, *loaded, K_ref);
}

// ===========================================================================
// Test 6: BSpline segmented Parquet round-trip
// ===========================================================================

TEST_F(ParquetIOTest, BSplineSegmentedRoundTrip) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.02,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}},
        },
        .grid = IVGrid{
            .moneyness = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto bspline_seg = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(bspline_seg.has_value()) << "SegmentedPriceTableBuilder failed";

    auto multi = build_multi_kref_surface({BSplineMultiKRefEntry{
        .K_ref = 100.0, .surface = std::move(*bspline_seg)}});
    ASSERT_TRUE(multi.has_value()) << "build_multi_kref_surface failed";

    SurfaceBounds bounds{
        .m_min = to_log_m({0.8}).front(),
        .m_max = to_log_m({1.2}).front(),
        .tau_min = 0.01,
        .tau_max = 1.0,
        .sigma_min = 0.15,
        .sigma_max = 0.40,
        .rate_min = 0.02,
        .rate_max = 0.05,
    };

    BSplineMultiKRefSurface surface(
        std::move(*multi), bounds, OptionType::PUT, 0.02);

    auto data = to_data(surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value()) << "write_parquet failed";

    auto read_result = read_parquet(temp_path_);
    ASSERT_TRUE(read_result.has_value()) << "read_parquet failed";

    auto loaded = from_data<BSplineMultiKRefInner>(*read_result);
    ASSERT_TRUE(loaded.has_value()) << "from_data failed";

    struct TestPoint { double spot, strike, tau, sigma, rate; };
    std::vector<TestPoint> test_points = {
        {100.0, 100.0, 0.3, 0.25, 0.04},
        {100.0, 100.0, 0.8, 0.20, 0.03},
    };
    for (const auto& p : test_points) {
        double orig = surface.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        double load = loaded->price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(orig, load, 1e-10)
            << "Mismatch at tau=" << p.tau << " sigma=" << p.sigma;
    }
}

// ===========================================================================
// Test 7: CRC64 checksum corruption detection
// ===========================================================================

TEST_F(ParquetIOTest, ChecksumCorruptionDetected) {
    // Build a minimal BSpline 4D surface
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);
    ASSERT_FALSE(data.segments.empty());
    ASSERT_FALSE(data.segments[0].values.empty());

    // 1. Write a valid file and verify it reads back fine
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value()) << "write_parquet failed";

    auto read_result = read_parquet(temp_path_);
    ASSERT_TRUE(read_result.has_value()) << "valid file should read fine";

    // 2. Create a corrupted file: use Arrow to rewrite with bad checksum.
    //    Read the valid file as an Arrow table, replace the checksum column
    //    with wrong values, write it back.
    {
        auto pool = arrow::default_memory_pool();
        auto infile_res = arrow::io::ReadableFile::Open(temp_path_.string());
        ASSERT_TRUE(infile_res.ok());
        auto reader_res = parquet::arrow::OpenFile(
            *infile_res, pool);
        ASSERT_TRUE(reader_res.ok());
        std::shared_ptr<arrow::Table> table;
        ASSERT_TRUE((*reader_res)->ReadTable(&table).ok());

        // Find checksum_values column and replace with zeros
        int crc_idx = table->schema()->GetFieldIndex("checksum_values");
        ASSERT_GE(crc_idx, 0) << "checksum_values column not found";

        arrow::UInt64Builder bad_crc_builder(pool);
        for (int64_t row = 0; row < table->num_rows(); ++row) {
            ASSERT_TRUE(bad_crc_builder.Append(0).ok());
        }
        std::shared_ptr<arrow::Array> bad_crc_array;
        ASSERT_TRUE(bad_crc_builder.Finish(&bad_crc_array).ok());

        auto corrupted_table = table->SetColumn(
            crc_idx, table->schema()->field(crc_idx),
            std::make_shared<arrow::ChunkedArray>(bad_crc_array));
        ASSERT_TRUE(corrupted_table.ok());

        // Write the corrupted table back (overwrite)
        auto outfile_res = arrow::io::FileOutputStream::Open(temp_path_.string());
        ASSERT_TRUE(outfile_res.ok());
        ASSERT_TRUE(parquet::arrow::WriteTable(
            **corrupted_table, pool, *outfile_res, 1024).ok());
    }

    // 3. read_parquet should detect the CRC mismatch and return an error
    auto corrupt_result = read_parquet(temp_path_);
    EXPECT_FALSE(corrupt_result.has_value())
        << "read_parquet should fail on corrupted CRC";
}

// ===========================================================================
// Test 8: Compression variants all round-trip correctly
// ===========================================================================

TEST_F(ParquetIOTest, CompressionVariants) {
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);

    ParquetCompression compressions[] = {
        ParquetCompression::NONE,
        ParquetCompression::SNAPPY,
        ParquetCompression::ZSTD,
    };

    for (auto compression : compressions) {
        auto path = std::filesystem::temp_directory_path() /
                    ("mango_compression_test_" +
                     std::to_string(static_cast<int>(compression)) + ".parquet");

        auto write_result = write_parquet(data, path, {.compression = compression});
        ASSERT_TRUE(write_result.has_value())
            << "write_parquet failed for compression "
            << static_cast<int>(compression);

        auto read_result = read_parquet(path);
        ASSERT_TRUE(read_result.has_value())
            << "read_parquet failed for compression "
            << static_cast<int>(compression);

        auto loaded = from_data<BSplineLeaf>(*read_result);
        ASSERT_TRUE(loaded.has_value())
            << "from_data failed for compression "
            << static_cast<int>(compression);

        // Verify prices match for each compression variant
        verify_prices_match_4d(*surface, *loaded, 100.0);

        std::filesystem::remove(path);
    }
}

// ===========================================================================
// Test 9: Type mismatch via Parquet
// ===========================================================================

TEST_F(ParquetIOTest, TypeMismatchViaParquet) {
    // Build BSpline 4D surface, write to Parquet, read back, try from_data
    // with wrong type
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value());

    auto read_result = read_parquet(temp_path_);
    ASSERT_TRUE(read_result.has_value());

    // bspline_4d data should fail for ChebyshevRawLeaf
    auto cheb_result = from_data<ChebyshevRawLeaf>(*read_result);
    EXPECT_FALSE(cheb_result.has_value())
        << "ChebyshevRawLeaf should reject bspline_4d data from Parquet";

    // bspline_4d data should fail for BSplineMultiKRefInner
    auto seg_result = from_data<BSplineMultiKRefInner>(*read_result);
    EXPECT_FALSE(seg_result.has_value())
        << "BSplineMultiKRefInner should reject bspline_4d data from Parquet";

    // bspline_4d data should fail for BSpline3DLeaf
    auto bsp3d_result = from_data<BSpline3DLeaf>(*read_result);
    EXPECT_FALSE(bsp3d_result.has_value())
        << "BSpline3DLeaf should reject bspline_4d data from Parquet";

    // bspline_4d data should fail for Chebyshev3DRawLeaf
    auto cheb3d_result = from_data<Chebyshev3DRawLeaf>(*read_result);
    EXPECT_FALSE(cheb3d_result.has_value())
        << "Chebyshev3DRawLeaf should reject bspline_4d data from Parquet";
}

// ===========================================================================
// Test 10: Metadata preservation through Parquet round-trip
// ===========================================================================

TEST_F(ParquetIOTest, MetadataPreservation) {
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value());

    auto read_result = read_parquet(temp_path_);
    ASSERT_TRUE(read_result.has_value());
    auto& loaded_data = *read_result;

    // Verify file-level metadata survives Parquet round-trip
    EXPECT_EQ(loaded_data.surface_type, data.surface_type);
    EXPECT_EQ(loaded_data.option_type, data.option_type);
    EXPECT_DOUBLE_EQ(loaded_data.dividend_yield, data.dividend_yield);
    EXPECT_DOUBLE_EQ(loaded_data.maturity, data.maturity);

    // Verify segment-level metadata
    ASSERT_EQ(loaded_data.segments.size(), data.segments.size());
    for (size_t s = 0; s < data.segments.size(); ++s) {
        const auto& orig_seg = data.segments[s];
        const auto& load_seg = loaded_data.segments[s];

        EXPECT_EQ(load_seg.segment_id, orig_seg.segment_id);
        EXPECT_EQ(load_seg.interp_type, orig_seg.interp_type);
        EXPECT_EQ(load_seg.ndim, orig_seg.ndim);
        EXPECT_DOUBLE_EQ(load_seg.K_ref, orig_seg.K_ref);
        EXPECT_DOUBLE_EQ(load_seg.tau_start, orig_seg.tau_start);
        EXPECT_DOUBLE_EQ(load_seg.tau_end, orig_seg.tau_end);
        EXPECT_DOUBLE_EQ(load_seg.tau_min, orig_seg.tau_min);
        EXPECT_DOUBLE_EQ(load_seg.tau_max, orig_seg.tau_max);

        ASSERT_EQ(load_seg.domain_lo.size(), orig_seg.domain_lo.size());
        for (size_t d = 0; d < orig_seg.domain_lo.size(); ++d) {
            EXPECT_DOUBLE_EQ(load_seg.domain_lo[d], orig_seg.domain_lo[d]);
            EXPECT_DOUBLE_EQ(load_seg.domain_hi[d], orig_seg.domain_hi[d]);
        }

        ASSERT_EQ(load_seg.num_pts.size(), orig_seg.num_pts.size());
        for (size_t d = 0; d < orig_seg.num_pts.size(); ++d) {
            EXPECT_EQ(load_seg.num_pts[d], orig_seg.num_pts[d]);
        }

        ASSERT_EQ(load_seg.grids.size(), orig_seg.grids.size());
        for (size_t d = 0; d < orig_seg.grids.size(); ++d) {
            ASSERT_EQ(load_seg.grids[d].size(), orig_seg.grids[d].size());
            for (size_t j = 0; j < orig_seg.grids[d].size(); ++j) {
                EXPECT_DOUBLE_EQ(load_seg.grids[d][j], orig_seg.grids[d][j]);
            }
        }

        ASSERT_EQ(load_seg.knots.size(), orig_seg.knots.size());
        for (size_t d = 0; d < orig_seg.knots.size(); ++d) {
            ASSERT_EQ(load_seg.knots[d].size(), orig_seg.knots[d].size());
            for (size_t j = 0; j < orig_seg.knots[d].size(); ++j) {
                EXPECT_DOUBLE_EQ(load_seg.knots[d][j], orig_seg.knots[d][j]);
            }
        }

        ASSERT_EQ(load_seg.values.size(), orig_seg.values.size());
    }

    // Verify full surface reconstruction works and produces matching prices
    auto loaded = from_data<BSplineLeaf>(loaded_data);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->option_type(), OptionType::PUT);
    EXPECT_NEAR(loaded->dividend_yield(), 0.02, 1e-15);
}

// ===========================================================================
// Test 11: Malformed list child type rejected
// ===========================================================================

TEST_F(ParquetIOTest, MalformedListChildTypeRejected) {
    // Build a valid file first
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value());

    // Read the valid file as an Arrow table
    auto pool = arrow::default_memory_pool();
    auto infile_res = arrow::io::ReadableFile::Open(temp_path_.string());
    ASSERT_TRUE(infile_res.ok());
    auto reader_res = parquet::arrow::OpenFile(*infile_res, pool);
    ASSERT_TRUE(reader_res.ok());
    std::shared_ptr<arrow::Table> table;
    ASSERT_TRUE((*reader_res)->ReadTable(&table).ok());

    // Replace "values" column (list<double>) with list<int32> to create
    // a type mismatch in the nested child element type.
    int val_idx = table->schema()->GetFieldIndex("values");
    ASSERT_GE(val_idx, 0);

    // Build a list<int32> column with the same shape
    arrow::ListBuilder bad_list_builder(pool,
        std::make_shared<arrow::Int32Builder>(pool));
    for (int64_t row = 0; row < table->num_rows(); ++row) {
        ASSERT_TRUE(bad_list_builder.Append().ok());
        auto& val_builder =
            static_cast<arrow::Int32Builder&>(*bad_list_builder.value_builder());
        ASSERT_TRUE(val_builder.Append(42).ok());
    }
    std::shared_ptr<arrow::Array> bad_values_array;
    ASSERT_TRUE(bad_list_builder.Finish(&bad_values_array).ok());

    auto new_field = arrow::field("values", arrow::list(arrow::int32()), false);
    auto corrupted_table = table->SetColumn(
        val_idx, new_field,
        std::make_shared<arrow::ChunkedArray>(bad_values_array));
    ASSERT_TRUE(corrupted_table.ok());

    auto outfile_res = arrow::io::FileOutputStream::Open(temp_path_.string());
    ASSERT_TRUE(outfile_res.ok());
    ASSERT_TRUE(parquet::arrow::WriteTable(
        **corrupted_table, pool, *outfile_res, 1024).ok());

    auto bad_result = read_parquet(temp_path_);
    EXPECT_FALSE(bad_result.has_value())
        << "read_parquet should reject list<int32> where list<double> expected";
}

// ===========================================================================
// Test 12: Null-containing column rejected
// ===========================================================================

TEST_F(ParquetIOTest, NullContainingColumnRejected) {
    // Build a valid file first
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value());

    // Read the valid file as an Arrow table
    auto pool = arrow::default_memory_pool();
    auto infile_res = arrow::io::ReadableFile::Open(temp_path_.string());
    ASSERT_TRUE(infile_res.ok());
    auto reader_res = parquet::arrow::OpenFile(*infile_res, pool);
    ASSERT_TRUE(reader_res.ok());
    std::shared_ptr<arrow::Table> table;
    ASSERT_TRUE((*reader_res)->ReadTable(&table).ok());

    // Replace "K_ref" column with a nullable double column containing a null
    int kref_idx = table->schema()->GetFieldIndex("K_ref");
    ASSERT_GE(kref_idx, 0);

    arrow::DoubleBuilder null_builder(pool);
    for (int64_t row = 0; row < table->num_rows(); ++row) {
        ASSERT_TRUE(null_builder.AppendNull().ok());
    }
    std::shared_ptr<arrow::Array> null_array;
    ASSERT_TRUE(null_builder.Finish(&null_array).ok());
    ASSERT_GT(null_array->null_count(), 0);

    auto nullable_field = arrow::field("K_ref", arrow::float64(), /*nullable=*/true);
    auto corrupted_table = table->SetColumn(
        kref_idx, nullable_field,
        std::make_shared<arrow::ChunkedArray>(null_array));
    ASSERT_TRUE(corrupted_table.ok());

    auto outfile_res = arrow::io::FileOutputStream::Open(temp_path_.string());
    ASSERT_TRUE(outfile_res.ok());
    ASSERT_TRUE(parquet::arrow::WriteTable(
        **corrupted_table, pool, *outfile_res, 1024).ok());

    auto bad_result = read_parquet(temp_path_);
    EXPECT_FALSE(bad_result.has_value())
        << "read_parquet should reject columns containing nulls";
}

// ===========================================================================
// Test 13: K_ref=0 rejected via data layer
// ===========================================================================

TEST_F(ParquetIOTest, ZeroKRefRejected) {
    // Build a valid surface, extract data, then tamper K_ref
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);
    ASSERT_FALSE(data.segments.empty());

    // Tamper K_ref to zero
    data.segments[0].K_ref = 0.0;
    auto bad_result = from_data<BSplineLeaf>(data);
    EXPECT_FALSE(bad_result.has_value())
        << "from_data should reject K_ref=0";

    // Tamper K_ref to negative
    data.segments[0].K_ref = -100.0;
    auto neg_result = from_data<BSplineLeaf>(data);
    EXPECT_FALSE(neg_result.has_value())
        << "from_data should reject negative K_ref";

    // Tamper K_ref to NaN
    data.segments[0].K_ref = std::numeric_limits<double>::quiet_NaN();
    auto nan_result = from_data<BSplineLeaf>(data);
    EXPECT_FALSE(nan_result.has_value())
        << "from_data should reject NaN K_ref";
}

// ===========================================================================
// Test 14: Invalid bounds rejected via data layer
// ===========================================================================

TEST_F(ParquetIOTest, InvalidBoundsRejected) {
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);

    // Inverted bounds (min > max)
    auto bad_data = data;
    bad_data.bounds_sigma_min = 0.50;
    bad_data.bounds_sigma_max = 0.10;
    auto inv_result = from_data<BSplineLeaf>(bad_data);
    EXPECT_FALSE(inv_result.has_value())
        << "from_data should reject inverted bounds";

    // NaN bounds
    bad_data = data;
    bad_data.bounds_tau_min = std::numeric_limits<double>::quiet_NaN();
    auto nan_result = from_data<BSplineLeaf>(bad_data);
    EXPECT_FALSE(nan_result.has_value())
        << "from_data should reject NaN bounds";

    // Inf bounds
    bad_data = data;
    bad_data.bounds_rate_max = std::numeric_limits<double>::infinity();
    auto inf_result = from_data<BSplineLeaf>(bad_data);
    EXPECT_FALSE(inf_result.has_value())
        << "from_data should reject Inf bounds";
}

// ===========================================================================
// Test 15: Tampered K_ref detected via Parquet CRC
// ===========================================================================

TEST_F(ParquetIOTest, TamperedKRefDetectedByCRC) {
    // Build and write a valid file
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value());

    // Read as Arrow table and tamper K_ref column
    auto pool = arrow::default_memory_pool();
    auto infile_res = arrow::io::ReadableFile::Open(temp_path_.string());
    ASSERT_TRUE(infile_res.ok());
    auto reader_res = parquet::arrow::OpenFile(*infile_res, pool);
    ASSERT_TRUE(reader_res.ok());
    std::shared_ptr<arrow::Table> table;
    ASSERT_TRUE((*reader_res)->ReadTable(&table).ok());

    int kref_idx = table->schema()->GetFieldIndex("K_ref");
    ASSERT_GE(kref_idx, 0);

    // Replace K_ref with a different value (but keep checksum unchanged)
    arrow::DoubleBuilder bad_kref_builder(pool);
    for (int64_t row = 0; row < table->num_rows(); ++row) {
        ASSERT_TRUE(bad_kref_builder.Append(999.0).ok());
    }
    std::shared_ptr<arrow::Array> bad_kref_array;
    ASSERT_TRUE(bad_kref_builder.Finish(&bad_kref_array).ok());

    auto corrupted_table = table->SetColumn(
        kref_idx, table->schema()->field(kref_idx),
        std::make_shared<arrow::ChunkedArray>(bad_kref_array));
    ASSERT_TRUE(corrupted_table.ok());

    auto outfile_res = arrow::io::FileOutputStream::Open(temp_path_.string());
    ASSERT_TRUE(outfile_res.ok());
    ASSERT_TRUE(parquet::arrow::WriteTable(
        **corrupted_table, pool, *outfile_res, 1024).ok());

    // read_parquet should detect the CRC mismatch since K_ref is now
    // included in the segment checksum
    auto bad_result = read_parquet(temp_path_);
    EXPECT_FALSE(bad_result.has_value())
        << "read_parquet should detect tampered K_ref via CRC";
}

// ===========================================================================
// Test 16: Tampered metadata detected via metadata checksum
// ===========================================================================

TEST_F(ParquetIOTest, TamperedMetadataDetectedByCRC) {
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);
    auto write_result = write_parquet(data, temp_path_);
    ASSERT_TRUE(write_result.has_value());

    // Read as Arrow table, tamper the option_type metadata, write back.
    // The per-segment CRCs still match, but the metadata checksum should fail.
    auto pool = arrow::default_memory_pool();
    auto infile_res = arrow::io::ReadableFile::Open(temp_path_.string());
    ASSERT_TRUE(infile_res.ok());
    auto reader_res = parquet::arrow::OpenFile(*infile_res, pool);
    ASSERT_TRUE(reader_res.ok());
    std::shared_ptr<arrow::Table> table;
    ASSERT_TRUE((*reader_res)->ReadTable(&table).ok());

    // Replace metadata: change option_type from PUT to CALL
    auto old_meta = table->schema()->metadata();
    ASSERT_TRUE(old_meta != nullptr);
    auto new_meta = old_meta->Copy();
    auto opt_idx = new_meta->FindKey("mango.option_type");
    ASSERT_GE(opt_idx, 0);
    // Build new metadata with tampered value
    auto tampered_meta = std::make_shared<arrow::KeyValueMetadata>();
    for (int i = 0; i < new_meta->size(); ++i) {
        if (i == opt_idx) {
            tampered_meta->Append(new_meta->key(i), "CALL");
        } else {
            tampered_meta->Append(new_meta->key(i), new_meta->value(i));
        }
    }
    auto tampered_schema = table->schema()->WithMetadata(tampered_meta);
    auto tampered_table = table->ReplaceSchemaMetadata(tampered_meta);

    auto outfile_res = arrow::io::FileOutputStream::Open(temp_path_.string());
    ASSERT_TRUE(outfile_res.ok());
    auto props = parquet::ArrowWriterProperties::Builder()
        .store_schema()->build();
    ASSERT_TRUE(parquet::arrow::WriteTable(
        *tampered_table, pool, *outfile_res, 1024,
        parquet::default_writer_properties(), props).ok());

    auto bad_result = read_parquet(temp_path_);
    EXPECT_FALSE(bad_result.has_value())
        << "read_parquet should detect tampered metadata via metadata checksum";
}

}  // namespace
}  // namespace mango
