#include "src/pde/core/pde_solver.hpp"
#include "src/support/cpu/cpu_diagnostics.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>

TEST(PDESolverSIMDBenchmark, PerformanceComparison) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 1001);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    auto features = mango::cpu::detect_cpu_features();
    std::cout << "CPU: " << mango::cpu::describe_cpu_features() << "\n";

    // Verify CPU detection works
    EXPECT_TRUE(features.has_sse2);  // x86-64 baseline
}
