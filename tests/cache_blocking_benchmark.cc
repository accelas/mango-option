#include "src/pde_solver.hpp"
#include "src/spatial_operators.hpp"
#include "src/boundary_conditions.hpp"
#include "src/root_finding.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>

TEST(CacheBlockingBenchmark, LargeGridSpeedup) {
    // Heat equation on large grid
    mango::LaplacianOperator op(0.1);

    // Create grid
    const size_t n = 10000;
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, n).generate();

    // Time domain - 10 steps for reasonable test time
    mango::TimeDomain time(0.0, 0.01, 0.001);  // 10 time steps

    // Root-finding config
    mango::RootFindingConfig root_config;

    // Boundary conditions
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    // Initial condition
    auto ic = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            double dx = x[i] - 0.5;
            u[i] = std::exp(-50.0 * dx * dx);
        }
    };

    // Benchmark: No blocking
    mango::TRBDF2Config config_no_block;
    config_no_block.cache_blocking_threshold = 100000;  // Disable

    mango::PDESolver solver_no_block(grid.span(), time, config_no_block, root_config,
                               left_bc, right_bc, op);
    solver_no_block.initialize(ic);

    auto start_no_block = std::chrono::high_resolution_clock::now();
    auto result1 = solver_no_block.solve();
    auto end_no_block = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(result1.has_value()) << "Solver failed: " << result1.error().message;

    double time_no_block = std::chrono::duration<double>(
        end_no_block - start_no_block).count();

    // Benchmark: With blocking
    mango::TRBDF2Config config_block;
    config_block.cache_blocking_threshold = 5000;  // Enable

    mango::PDESolver solver_block(grid.span(), time, config_block, root_config,
                           left_bc, right_bc, op);
    solver_block.initialize(ic);

    auto start_block = std::chrono::high_resolution_clock::now();
    auto result2 = solver_block.solve();
    auto end_block = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(result2.has_value()) << "Solver failed: " << result2.error().message;

    double time_block = std::chrono::duration<double>(
        end_block - start_block).count();

    double speedup = time_no_block / time_block;

    std::cout << "\n=== Cache-Blocking Benchmark (n=" << n << ") ===" << std::endl;
    std::cout << "No blocking: " << time_no_block << "s" << std::endl;
    std::cout << "With blocking: " << time_block << "s" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Note: Speedup is highly hardware-dependent (CPU cache sizes vary)
    // Design targets 4-8x on typical hardware with 32KB L1 cache
    // Conservative test: expect at least modest improvement
    EXPECT_GE(speedup, 1.0) << "Cache-blocking should not be slower than no blocking";

    if (speedup >= 2.0) {
        std::cout << "SUCCESS: Achieved target speedup of 2x or better!" << std::endl;
    } else {
        std::cout << "NOTE: Speedup < 2x may indicate hardware with large cache or memory bottleneck elsewhere" << std::endl;
    }
}
