#include "src/core/pde_solver.hpp"
#include "src/cpu/feature_detection.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>

TEST(PDESolverSIMDBenchmark, PerformanceComparison) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 1001);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    auto isa = mango::cpu::select_isa_target();
    std::cout << "Running on ISA: " << mango::cpu::isa_target_name(isa) << "\n";

    // Benchmark will be expanded with actual timing once SIMD operators integrated
    EXPECT_TRUE(isa == mango::cpu::ISATarget::DEFAULT ||
                isa == mango::cpu::ISATarget::AVX2 ||
                isa == mango::cpu::ISATarget::AVX512F);
}
