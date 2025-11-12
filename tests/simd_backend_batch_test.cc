#include <gtest/gtest.h>
#include "src/pde/operators/centered_difference_simd_backend.hpp"
#include "src/pde/operators/grid_spacing.hpp"
#include "src/pde/core/grid.hpp"
#include <vector>
#include <experimental/simd>
#include <cmath>

namespace stdx = std::experimental;

TEST(SimdBackendBatch, SecondDerivativeBitwiseMatch) {
    constexpr size_t n = 101;
    const size_t batch_width = stdx::native_simd<double>::size();

    // Create uniform grid
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, n);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    auto spacing = mango::operators::GridSpacing<double>(grid.view());
    mango::operators::SimdBackend<double> backend(spacing);

    // Setup test data (batched)
    std::vector<double> u_batch(n * batch_width);
    for (size_t i = 0; i < n; ++i) {
        for (size_t lane = 0; lane < batch_width; ++lane) {
            u_batch[i * batch_width + lane] = std::sin(i * 0.1) + lane * 0.01;
        }
    }

    // Compute batch second derivative
    std::vector<double> d2u_batch(n * batch_width);
    backend.compute_second_derivative_batch_uniform(
        std::span{u_batch}, std::span{d2u_batch}, batch_width, 1, n-1);

    // Compute single-contract second derivatives (one per lane)
    for (size_t lane = 0; lane < batch_width; ++lane) {
        std::vector<double> u_single(n);
        std::vector<double> d2u_single(n);

        // Extract lane from batch
        for (size_t i = 0; i < n; ++i) {
            u_single[i] = u_batch[i * batch_width + lane];
        }

        // Compute using single-contract path
        backend.compute_second_derivative_uniform(
            std::span{u_single}, std::span{d2u_single}, 1, n-1);

        // Compare results (bitwise for interior points)
        for (size_t i = 1; i < n-1; ++i) {
            EXPECT_EQ(d2u_batch[i * batch_width + lane], d2u_single[i])
                << "Mismatch at lane=" << lane << " i=" << i;
        }
    }
}

TEST(SimdBackendBatch, FirstDerivativeBitwiseMatch) {
    constexpr size_t n = 101;
    const size_t batch_width = stdx::native_simd<double>::size();

    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, n);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    auto spacing = mango::operators::GridSpacing<double>(grid.view());
    mango::operators::SimdBackend<double> backend(spacing);

    // Setup test data
    std::vector<double> u_batch(n * batch_width);
    for (size_t i = 0; i < n; ++i) {
        for (size_t lane = 0; lane < batch_width; ++lane) {
            u_batch[i * batch_width + lane] = std::sin(i * 0.1) + lane * 0.01;
        }
    }

    // Compute batch first derivative
    std::vector<double> du_batch(n * batch_width);
    backend.compute_first_derivative_batch_uniform(
        std::span{u_batch}, std::span{du_batch}, batch_width, 1, n-1);

    // Compare with single-contract
    for (size_t lane = 0; lane < batch_width; ++lane) {
        std::vector<double> u_single(n), du_single(n);
        for (size_t i = 0; i < n; ++i) {
            u_single[i] = u_batch[i * batch_width + lane];
        }

        backend.compute_first_derivative_uniform(
            std::span{u_single}, std::span{du_single}, 1, n-1);

        for (size_t i = 1; i < n-1; ++i) {
            EXPECT_EQ(du_batch[i * batch_width + lane], du_single[i])
                << "Mismatch at lane=" << lane << " i=" << i;
        }
    }
}

TEST(SimdBackendBatch, SecondDerivativeBoundaryNotTouched) {
    constexpr size_t n = 101;
    const size_t batch_width = stdx::native_simd<double>::size();

    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, n);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    auto spacing = mango::operators::GridSpacing<double>(grid.view());
    mango::operators::SimdBackend<double> backend(spacing);

    std::vector<double> u_batch(n * batch_width);
    for (size_t i = 0; i < n; ++i) {
        for (size_t lane = 0; lane < batch_width; ++lane) {
            u_batch[i * batch_width + lane] = std::sin(i * 0.1);
        }
    }

    // Initialize d2u_batch with sentinel values at boundaries
    std::vector<double> d2u_batch(n * batch_width, -999.0);

    backend.compute_second_derivative_batch_uniform(
        std::span{u_batch}, std::span{d2u_batch}, batch_width, 1, n-1);

    // Verify boundaries remain untouched
    for (size_t lane = 0; lane < batch_width; ++lane) {
        EXPECT_DOUBLE_EQ(d2u_batch[0 * batch_width + lane], -999.0)
            << "Left boundary should not be modified at lane=" << lane;
        EXPECT_DOUBLE_EQ(d2u_batch[(n-1) * batch_width + lane], -999.0)
            << "Right boundary should not be modified at lane=" << lane;
    }
}

TEST(SimdBackendBatch, FirstDerivativeBoundaryNotTouched) {
    constexpr size_t n = 101;
    const size_t batch_width = stdx::native_simd<double>::size();

    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, n);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    auto spacing = mango::operators::GridSpacing<double>(grid.view());
    mango::operators::SimdBackend<double> backend(spacing);

    std::vector<double> u_batch(n * batch_width);
    for (size_t i = 0; i < n; ++i) {
        for (size_t lane = 0; lane < batch_width; ++lane) {
            u_batch[i * batch_width + lane] = std::sin(i * 0.1);
        }
    }

    // Initialize du_batch with sentinel values at boundaries
    std::vector<double> du_batch(n * batch_width, -777.0);

    backend.compute_first_derivative_batch_uniform(
        std::span{u_batch}, std::span{du_batch}, batch_width, 1, n-1);

    // Verify boundaries remain untouched
    for (size_t lane = 0; lane < batch_width; ++lane) {
        EXPECT_DOUBLE_EQ(du_batch[0 * batch_width + lane], -777.0)
            << "Left boundary should not be modified at lane=" << lane;
        EXPECT_DOUBLE_EQ(du_batch[(n-1) * batch_width + lane], -777.0)
            << "Right boundary should not be modified at lane=" << lane;
    }
}

TEST(SimdBackendBatch, SecondDerivativePartialBatch) {
    constexpr size_t n = 101;
    const size_t batch_width = stdx::native_simd<double>::size();
    const size_t partial_width = batch_width / 2 + 1;  // Not multiple of SIMD width

    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, n);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    auto spacing = mango::operators::GridSpacing<double>(grid.view());
    mango::operators::SimdBackend<double> backend(spacing);

    // Setup test data with partial batch width
    std::vector<double> u_batch(n * partial_width);
    for (size_t i = 0; i < n; ++i) {
        for (size_t lane = 0; lane < partial_width; ++lane) {
            u_batch[i * partial_width + lane] = std::sin(i * 0.1) + lane * 0.01;
        }
    }

    std::vector<double> d2u_batch(n * partial_width);
    backend.compute_second_derivative_batch_uniform(
        std::span{u_batch}, std::span{d2u_batch}, partial_width, 1, n-1);

    // Verify all lanes including scalar tail
    for (size_t lane = 0; lane < partial_width; ++lane) {
        std::vector<double> u_single(n), d2u_single(n);
        for (size_t i = 0; i < n; ++i) {
            u_single[i] = u_batch[i * partial_width + lane];
        }

        backend.compute_second_derivative_uniform(
            std::span{u_single}, std::span{d2u_single}, 1, n-1);

        for (size_t i = 1; i < n-1; ++i) {
            EXPECT_EQ(d2u_batch[i * partial_width + lane], d2u_single[i])
                << "Mismatch at lane=" << lane << " i=" << i;
        }
    }
}

TEST(SimdBackendBatch, FirstDerivativePartialBatch) {
    constexpr size_t n = 101;
    const size_t batch_width = stdx::native_simd<double>::size();
    const size_t partial_width = batch_width / 2 + 1;

    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, n);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    auto spacing = mango::operators::GridSpacing<double>(grid.view());
    mango::operators::SimdBackend<double> backend(spacing);

    std::vector<double> u_batch(n * partial_width);
    for (size_t i = 0; i < n; ++i) {
        for (size_t lane = 0; lane < partial_width; ++lane) {
            u_batch[i * partial_width + lane] = std::sin(i * 0.1) + lane * 0.01;
        }
    }

    std::vector<double> du_batch(n * partial_width);
    backend.compute_first_derivative_batch_uniform(
        std::span{u_batch}, std::span{du_batch}, partial_width, 1, n-1);

    for (size_t lane = 0; lane < partial_width; ++lane) {
        std::vector<double> u_single(n), du_single(n);
        for (size_t i = 0; i < n; ++i) {
            u_single[i] = u_batch[i * partial_width + lane];
        }

        backend.compute_first_derivative_uniform(
            std::span{u_single}, std::span{du_single}, 1, n-1);

        for (size_t i = 1; i < n-1; ++i) {
            EXPECT_EQ(du_batch[i * partial_width + lane], du_single[i])
                << "Mismatch at lane=" << lane << " i=" << i;
        }
    }
}
