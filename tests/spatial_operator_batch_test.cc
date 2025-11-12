#include <gtest/gtest.h>
#include "src/pde/operators/spatial_operator.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
#include "src/pde/operators/grid_spacing.hpp"
#include <vector>
#include <experimental/simd>
#include <memory>
#include <cmath>

namespace stdx = std::experimental;

TEST(SpatialOperatorBatch, BitwiseMatchWithSingleContract) {
    constexpr size_t n = 11;
    const size_t batch_width = stdx::native_simd<double>::size();

    // Create grid (exactly like working test)
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = i * 0.1;
    }
    auto grid = mango::GridView<double>(x);
    auto spacing = std::make_shared<mango::operators::GridSpacing<double>>(grid);
    auto pde = mango::operators::BlackScholesPDE<double>(0.2, 0.05, 0.01);
    auto spatial_op = mango::operators::SpatialOperator<
        mango::operators::BlackScholesPDE<double>, double>(pde, spacing);

    // Setup batch input
    std::vector<double> u_batch(n * batch_width);
    for (size_t i = 0; i < n; ++i) {
        for (size_t lane = 0; lane < batch_width; ++lane) {
            u_batch[i * batch_width + lane] = std::exp(-x[i]) * (1.0 + lane * 0.001);
        }
    }

    // Compute batch operator
    std::vector<double> lu_batch(n * batch_width, 0.0);
    spatial_op.apply_interior_batch(0.0, std::span{u_batch}, std::span{lu_batch},
                                   batch_width, 1, n-1);

    // Compute single-contract operator for each lane
    for (size_t lane = 0; lane < batch_width; ++lane) {
        std::vector<double> u_single(n), lu_single(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            u_single[i] = u_batch[i * batch_width + lane];
        }

        spatial_op.apply_interior(0.0, std::span{u_single}, std::span{lu_single}, 1, n-1);

        // Verify bitwise match
        for (size_t i = 1; i < n-1; ++i) {
            EXPECT_EQ(lu_batch[i * batch_width + lane], lu_single[i])
                << "Mismatch at lane=" << lane << " i=" << i;
        }
    }
}

TEST(SpatialOperatorBatch, BoundaryNotTouched) {
    constexpr size_t n = 11;
    const size_t batch_width = stdx::native_simd<double>::size();

    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = i * 0.1;
    }
    auto grid = mango::GridView<double>(x);
    auto spacing = std::make_shared<mango::operators::GridSpacing<double>>(grid);
    auto pde = mango::operators::BlackScholesPDE<double>(0.2, 0.05, 0.01);
    auto spatial_op = mango::operators::SpatialOperator<
        mango::operators::BlackScholesPDE<double>, double>(pde, spacing);

    std::vector<double> u_batch(n * batch_width);
    for (size_t i = 0; i < n; ++i) {
        for (size_t lane = 0; lane < batch_width; ++lane) {
            u_batch[i * batch_width + lane] = std::exp(-x[i]);
        }
    }

    // Initialize lu_batch with sentinel values at boundaries
    std::vector<double> lu_batch(n * batch_width, -999.0);

    spatial_op.apply_interior_batch(0.0, std::span{u_batch}, std::span{lu_batch},
                                   batch_width, 1, n-1);

    // Verify boundaries remain untouched
    for (size_t lane = 0; lane < batch_width; ++lane) {
        EXPECT_DOUBLE_EQ(lu_batch[0 * batch_width + lane], -999.0)
            << "Left boundary should not be modified at lane=" << lane;
        EXPECT_DOUBLE_EQ(lu_batch[(n-1) * batch_width + lane], -999.0)
            << "Right boundary should not be modified at lane=" << lane;
    }
}

TEST(SpatialOperatorBatch, PartialBatchWidth) {
    constexpr size_t n = 11;
    const size_t batch_width = stdx::native_simd<double>::size();
    const size_t partial_width = batch_width / 2 + 1;  // Not multiple of SIMD width

    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = i * 0.1;
    }
    auto grid = mango::GridView<double>(x);
    auto spacing = std::make_shared<mango::operators::GridSpacing<double>>(grid);
    auto pde = mango::operators::BlackScholesPDE<double>(0.2, 0.05, 0.01);
    auto spatial_op = mango::operators::SpatialOperator<
        mango::operators::BlackScholesPDE<double>, double>(pde, spacing);

    // Setup test data with partial batch width
    std::vector<double> u_batch(n * partial_width);
    for (size_t i = 0; i < n; ++i) {
        for (size_t lane = 0; lane < partial_width; ++lane) {
            u_batch[i * partial_width + lane] = std::exp(-x[i]) * (1.0 + lane * 0.001);
        }
    }

    std::vector<double> lu_batch(n * partial_width, 0.0);
    spatial_op.apply_interior_batch(0.0, std::span{u_batch}, std::span{lu_batch},
                                   partial_width, 1, n-1);

    // Verify all lanes including scalar tail
    for (size_t lane = 0; lane < partial_width; ++lane) {
        std::vector<double> u_single(n), lu_single(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            u_single[i] = u_batch[i * partial_width + lane];
        }

        spatial_op.apply_interior(0.0, std::span{u_single}, std::span{lu_single}, 1, n-1);

        for (size_t i = 1; i < n-1; ++i) {
            EXPECT_EQ(lu_batch[i * partial_width + lane], lu_single[i])
                << "Mismatch at lane=" << lane << " i=" << i;
        }
    }
}
