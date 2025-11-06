// Cachegrind test harness for cache-blocking verification
//
// Usage:
//   ./cachegrind_harness --with-blocking
//   ./cachegrind_harness --without-blocking
//
// This program is designed to run under valgrind --tool=cachegrind
// to measure L1 cache performance with and without cache blocking.

#include "src/cpp/pde_solver.hpp"
#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/boundary_conditions.hpp"
#include "src/cpp/root_finding.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [--with-blocking|--without-blocking]\n";
        return 1;
    }

    std::string mode(argv[1]);
    bool use_blocking;

    if (mode == "--with-blocking") {
        use_blocking = true;
    } else if (mode == "--without-blocking") {
        use_blocking = false;
    } else {
        std::cerr << "Invalid mode. Use --with-blocking or --without-blocking\n";
        return 1;
    }

    // Problem setup: Heat equation on large grid
    // Grid size chosen to be above cache blocking threshold
    const size_t n = 10000;
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, n).generate();

    // Time domain - 50 steps to get measurable cache activity
    mango::TimeDomain time(0.0, 0.05, 0.001);  // 50 time steps

    // Heat equation operator
    mango::LaplacianOperator op(0.1);

    // Root-finding config
    mango::RootFindingConfig root_config;

    // Boundary conditions
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    // Initial condition: Gaussian pulse
    auto ic = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            double dx = x[i] - 0.5;
            u[i] = std::exp(-50.0 * dx * dx);
        }
    };

    // Configure solver based on mode
    mango::TRBDF2Config config;
    if (use_blocking) {
        config.cache_blocking_threshold = mango::CacheBlockConfig::default_threshold();
        std::cout << "Running WITH cache blocking (threshold="
                  << config.cache_blocking_threshold << ")\n";
    } else {
        config.cache_blocking_threshold = 100000;  // Effectively disable
        std::cout << "Running WITHOUT cache blocking (threshold="
                  << config.cache_blocking_threshold << ")\n";
    }

    // Create and initialize solver
    mango::PDESolver solver(grid.span(), time, config, root_config,
                           left_bc, right_bc, op);
    solver.initialize(ic);

    // Solve
    std::cout << "Starting solve (n=" << n << ", steps=" << time.n_steps() << ")...\n";
    bool converged = solver.solve();

    if (!converged) {
        std::cerr << "ERROR: Solver failed to converge\n";
        return 1;
    }

    // Get solution (to ensure computation isn't optimized away)
    auto solution = solver.u_current();
    double sum = 0.0;
    for (double val : solution) {
        sum += val;
    }

    std::cout << "Solve complete. Solution checksum: " << sum << "\n";

    return 0;
}
