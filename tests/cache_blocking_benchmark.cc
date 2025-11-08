#include "src/pde_solver.hpp"
#include "src/spatial_operators.hpp"
#include "src/boundary_conditions.hpp"
#include "src/root_finding.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <cmath>

TEST(CacheBlockingBenchmark, ConfigParameterIgnored) {
    // Verify that cache_blocking_threshold is now ignored (cache blocking removed)
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

    // Run with threshold set high (should have no effect)
    mango::TRBDF2Config config1;
    config1.cache_blocking_threshold = 100000;

    mango::PDESolver solver1(grid.span(), time, config1, root_config,
                             left_bc, right_bc, op);
    solver1.initialize(ic);

    auto start1 = std::chrono::high_resolution_clock::now();
    auto result1 = solver1.solve();
    auto end1 = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(result1.has_value()) << "Solver failed: " << result1.error().message;

    double time1 = std::chrono::duration<double>(end1 - start1).count();

    // Run with threshold set low (should still have no effect)
    mango::TRBDF2Config config2;
    config2.cache_blocking_threshold = 5000;

    mango::PDESolver solver2(grid.span(), time, config2, root_config,
                             left_bc, right_bc, op);
    solver2.initialize(ic);

    auto start2 = std::chrono::high_resolution_clock::now();
    auto result2 = solver2.solve();
    auto end2 = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(result2.has_value()) << "Solver failed: " << result2.error().message;

    double time2 = std::chrono::duration<double>(end2 - start2).count();

    double ratio = time1 / time2;

    std::cout << "\n=== Cache-Blocking Config Test (n=" << n << ") ===" << std::endl;
    std::cout << "High threshold: " << time1 << "s" << std::endl;
    std::cout << "Low threshold: " << time2 << "s" << std::endl;
    std::cout << "Ratio: " << ratio << "x" << std::endl;

    // Since cache blocking is removed, both should run at the same speed
    // Allow for 10% variation due to system noise
    EXPECT_NEAR(ratio, 1.0, 0.1)
        << "Both configs should perform identically (cache blocking removed)";

    std::cout << "NOTE: Cache blocking has been removed. The cache_blocking_threshold "
              << "parameter is ignored.\n"
              << "Both configurations now use single-pass evaluation." << std::endl;
}
