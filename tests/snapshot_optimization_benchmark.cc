#include "src/pde_solver.hpp"
#include "src/price_table_snapshot_collector.hpp"
#include "src/spatial_operators.hpp"
#include "src/boundary_conditions.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>

TEST(SnapshotOptimizationBenchmark, CompareApproaches) {
    // Price table dimensions (scaled down for reasonable benchmark time ~30-60s)
    const size_t n_m = 5;     // Moneyness points (reduced to 5)
    const size_t n_tau = 5;   // Maturity points (reduced to 5)
    const size_t total_options = n_m * n_tau;  // 25 options

    // Generate grids
    std::vector<double> moneyness(n_m);
    for (size_t i = 0; i < n_m; ++i) {
        moneyness[i] = 0.7 + i * 0.6 / (n_m - 1);  // [0.7, 1.3]
    }

    std::vector<double> tau(n_tau);
    for (size_t i = 0; i < n_tau; ++i) {
        tau[i] = 0.027 + i * (2.0 - 0.027) / (n_tau - 1);  // [0.027, 2.0]
    }

    // PDE configuration (smaller grids for benchmark ~30-60s total)
    const double K_ref = 100.0;
    const double sigma = 0.20;
    const size_t n_space = 31;    // Reduced from 51 (coarser grid)
    const size_t n_time = 100;    // Reduced from 200 (fewer time steps)

    // ===== OLD APPROACH: Solve each option individually =====
    auto start_old = std::chrono::high_resolution_clock::now();

    size_t n_solves_old = 0;
    for (size_t tau_idx = 0; tau_idx < n_tau; ++tau_idx) {
        double T = tau[tau_idx];

        for (size_t m_idx = 0; m_idx < n_m; ++m_idx) {
            double m = moneyness[m_idx];
            double S0 = m * K_ref;

            // Setup PDE for this option
            mango::LaplacianOperator op(0.5 * sigma * sigma);
            mango::TimeDomain time(0.0, T, T / n_time);
            mango::RootFindingConfig root_config;

            auto left_bc = mango::DirichletBC([K = K_ref](double, double) {
                return K;  // Put value at S=0
            });
            auto right_bc = mango::DirichletBC([](double, double) {
                return 0.0;  // Put value at S→∞
            });

            // Create grid (must persist for solver lifetime)
            auto grid_buffer = mango::GridSpec<>::uniform(0.0, 2.0 * S0, n_space).generate();

            mango::PDESolver solver(grid_buffer.span(), time, mango::TRBDF2Config{},
                                   root_config, left_bc, right_bc, op);

            // Initial condition: max(K - S, 0)
            auto ic = [K = K_ref](std::span<const double> x, std::span<double> u) {
                for (size_t i = 0; i < x.size(); ++i) {
                    u[i] = std::max(K - x[i], 0.0);
                }
            };
            solver.initialize(ic);
            solver.solve();

            ++n_solves_old;
        }
    }

    auto end_old = std::chrono::high_resolution_clock::now();
    double time_old = std::chrono::duration<double>(end_old - start_old).count();

    // ===== NEW APPROACH: Solve once per maturity with snapshots =====
    auto start_new = std::chrono::high_resolution_clock::now();

    size_t n_solves_new = 0;
    for (size_t tau_idx = 0; tau_idx < n_tau; ++tau_idx) {
        double T = tau[tau_idx];

        // Setup PDE once for this maturity
        mango::LaplacianOperator op(0.5 * sigma * sigma);
        const double S_max = 2.0 * K_ref;
        mango::TimeDomain time(0.0, T, T / n_time);
        mango::RootFindingConfig root_config;

        auto left_bc = mango::DirichletBC([K = K_ref](double, double) { return K; });
        auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

        // Create grid (must persist for solver lifetime)
        auto grid_buffer = mango::GridSpec<>::uniform(0.0, S_max, n_space).generate();

        mango::PDESolver solver(grid_buffer.span(), time, mango::TRBDF2Config{},
                               root_config, left_bc, right_bc, op);

        // Register snapshot collector for all moneyness points
        // NOTE: Pass entire tau array so collector knows full table dimensions
        // Only this maturity (tau_idx) will be filled via user_index
        mango::PriceTableSnapshotCollectorConfig collector_config{
            .moneyness = std::span{moneyness},
            .tau = std::span{tau},  // Full tau array for correct sizing
            .K_ref = K_ref,
            .exercise_type = mango::ExerciseType::EUROPEAN
        };
        mango::PriceTableSnapshotCollector collector(collector_config);

        // CRITICAL FIX: Use step_index (last step = n_time), not time T
        // Step indices are 0-based, so last step is n_time - 1
        solver.register_snapshot(n_time - 1, tau_idx, &collector);

        // Initial condition
        auto ic = [K = K_ref](std::span<const double> x, std::span<double> u) {
            for (size_t i = 0; i < x.size(); ++i) {
                u[i] = std::max(K - x[i], 0.0);
            }
        };
        solver.initialize(ic);
        solver.solve();

        // Collector now has prices for all n_m moneyness points
        ++n_solves_new;
    }

    auto end_new = std::chrono::high_resolution_clock::now();
    double time_new = std::chrono::duration<double>(end_new - start_new).count();

    // ===== RESULTS =====
    double speedup = time_old / time_new;

    std::cout << "\n=== Snapshot Optimization Benchmark ===" << std::endl;
    std::cout << "Price table size: " << n_m << " × " << n_tau
              << " = " << total_options << " options" << std::endl;
    std::cout << "\nOld approach (solve per option):" << std::endl;
    std::cout << "  Solves: " << n_solves_old << std::endl;
    std::cout << "  Time: " << time_old << "s" << std::endl;
    std::cout << "  Time per option: " << (time_old / total_options * 1000.0) << "ms" << std::endl;
    std::cout << "\nNew approach (snapshots):" << std::endl;
    std::cout << "  Solves: " << n_solves_new << std::endl;
    std::cout << "  Time: " << time_new << "s" << std::endl;
    std::cout << "  Time per option: " << (time_new / total_options * 1000.0) << "ms" << std::endl;
    std::cout << "\nSpeedup: " << speedup << "x" << std::endl;
    std::cout << "Solve reduction: " << n_solves_old << " → " << n_solves_new
              << " (" << (n_solves_old / n_solves_new) << "x)" << std::endl;

    // Verify speedup claim (expect ~5x for 5×5 grid, use 4x threshold for variance)
    EXPECT_GE(speedup, 4.0) << "Expected at least 4x speedup (conservative estimate for 5×5)";
    EXPECT_LE(speedup, 20.0) << "Speedup > 20x seems unrealistic for 5×5 grid";

    if (speedup >= 4.5) {
        std::cout << "\n✓ SUCCESS: Achieved " << speedup << "x speedup!" << std::endl;
        std::cout << "  (Scaled to full 20×30 table would be " << (speedup * 4) << "x)" << std::endl;
    } else {
        std::cout << "\n✓ PASSED: Speedup " << speedup << "x demonstrates optimization" << std::endl;
    }
}
