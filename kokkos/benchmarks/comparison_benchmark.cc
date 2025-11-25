// Kokkos American option benchmark for comparison with original implementation
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "kokkos/src/option/american_option.hpp"
#include "kokkos/src/option/batch_solver.hpp"

using namespace mango::kokkos;

int main() {
    Kokkos::initialize();
    {
        std::cout << "=== Kokkos American Option Benchmark ===\n";
        std::cout << "Execution space: " << Kokkos::DefaultExecutionSpace::name() << "\n\n";

        // Match original benchmark parameters
        PricingParams params{
            .strike = 100.0,
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .dividend_yield = 0.02,
            .type = OptionType::Put
        };

        // Warm up
        {
            AmericanOptionSolver<Kokkos::HostSpace> solver(params);
            (void)solver.solve();
        }

        // Show grid estimation
        auto [grid_params, time_params] = estimate_grid_for_option(params);
        std::cout << "Grid estimation:\n";
        std::cout << "  Spatial points: " << grid_params.n_points << "\n";
        std::cout << "  Time steps: " << time_params.n_steps << "\n";
        std::cout << "  Domain: [" << grid_params.x_min << ", " << grid_params.x_max << "]\n\n";

        // Single option benchmark
        std::cout << "--- Single Option ---\n";
        {
            const int N_ITER = 100;
            auto start = std::chrono::high_resolution_clock::now();

            double price = 0, delta = 0;
            for (int i = 0; i < N_ITER; ++i) {
                AmericanOptionSolver<Kokkos::HostSpace> solver(params);
                auto result = solver.solve();
                if (result.has_value()) {
                    price = result->price;
                    delta = result->delta;
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double per_option_ms = total_ms / N_ITER;

            std::cout << "  Price: " << std::fixed << std::setprecision(4) << price << "\n";
            std::cout << "  Delta: " << delta << "\n";
            std::cout << "  Time:  " << per_option_ms << " ms per option\n";
            std::cout << "  Throughput: " << (1000.0 / per_option_ms) << " options/sec\n";
        }

        // Batch benchmark (64 options sequential)
        std::cout << "\n--- Sequential 64 Options ---\n";
        {
            std::vector<PricingParams> batch;
            for (int i = 0; i < 64; ++i) {
                double m = 0.8 + 0.4 * (i % 8) / 7.0;
                batch.push_back(PricingParams{
                    .strike = 100.0,
                    .spot = m * 100.0,
                    .maturity = 0.5 + 0.5 * ((i / 8) % 4) / 3.0,
                    .volatility = 0.15 + 0.10 * ((i / 32) % 2),
                    .rate = 0.05,
                    .dividend_yield = 0.02,
                    .type = OptionType::Put
                });
            }

            const int N_ITER = 10;
            auto start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < N_ITER; ++iter) {
                for (const auto& p : batch) {
                    AmericanOptionSolver<Kokkos::HostSpace> solver(p);
                    (void)solver.solve();
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double per_batch_ms = total_ms / N_ITER;

            std::cout << "  Time:  " << per_batch_ms << " ms per 64 options\n";
            std::cout << "  Throughput: " << (64000.0 / per_batch_ms) << " options/sec\n";
        }

        // Batch benchmark using BatchAmericanSolver
        std::cout << "\n--- Kokkos Batch Solver (64 Options, shared params) ---\n";
        {
            BatchPricingParams batch_params{
                .maturity = 1.0,
                .volatility = 0.20,
                .rate = 0.05,
                .dividend_yield = 0.02,
                .is_put = true
            };

            Kokkos::View<double*, Kokkos::HostSpace> strikes("strikes", 64);
            Kokkos::View<double*, Kokkos::HostSpace> spots("spots", 64);

            for (int i = 0; i < 64; ++i) {
                double m = 0.8 + 0.4 * i / 63.0;
                strikes(i) = 100.0;
                spots(i) = m * 100.0;
            }

            const int N_ITER = 20;
            auto start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < N_ITER; ++iter) {
                BatchAmericanSolver<Kokkos::HostSpace> solver(batch_params, strikes, spots);
                (void)solver.solve();
            }

            auto end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double per_batch_ms = total_ms / N_ITER;

            std::cout << "  Time:  " << per_batch_ms << " ms per 64 options\n";
            std::cout << "  Throughput: " << (64000.0 / per_batch_ms) << " options/sec\n";
        }

        std::cout << "\n=== Feature Parity Summary ===\n";
        std::cout << "Grid estimation now matches original:\n";
        std::cout << "  Original: 101 spatial points, 498 time steps, sinh grid (alpha=2.0)\n";
        std::cout << "  Kokkos:   " << grid_params.n_points << " spatial points, "
                  << time_params.n_steps << " time steps, sinh grid (alpha=" << grid_params.alpha << ")\n";
        std::cout << "\n=== Performance Comparison ===\n";
        std::cout << "Original (OpenMP+PMR, AVX2, TR-BDF2 Newton):\n";
        std::cout << "  Single: ~1.27 ms, Batch 64: ~4.77 ms (13.4k/s)\n";
        std::cout << "Kokkos (feature parity, no SIMD optimization yet):\n";
        std::cout << "  Performance gap is expected - original uses:\n";
        std::cout << "  - Analytical Jacobian with Newton iteration\n";
        std::cout << "  - PMR memory pooling (no allocations per solve)\n";
        std::cout << "  - AVX2/AVX-512 vectorization\n";
        std::cout << "  - OpenMP SIMD for linear algebra\n";
    }
    Kokkos::finalize();
    return 0;
}
