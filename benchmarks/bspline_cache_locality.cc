/**
 * @file bspline_cache_locality.cc
 * @brief Benchmark comparing cache locality impact of axis ordering in 4D B-spline fitting
 *
 * Measures performance difference between:
 * - Old order: m → τ → σ → r (worst cache locality, m-axis has stride Nt*Nv*Nr)
 * - New order: r → σ → τ → m (best cache locality, r-axis has stride 1)
 *
 * Expected results:
 * - Old order: m-axis stride ~48KB for 50×30×20×10, exceeds L1 cache
 * - New order: r-axis stride 8 bytes, excellent cache locality
 * - Speedup: Hardware-dependent, typically 1.5-3x for large grids
 */

#include "src/bspline_fitter_4d_separable.hpp"
#include "src/bspline_collocation_1d.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <memory>

using namespace mango;

/// Helper: Create linearly spaced grid
std::vector<double> linspace(double start, double end, int n) {
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = start + (end - start) * i / (n - 1);
    }
    return result;
}

/// Test function: smooth separable function
double test_function(double m, double t, double v, double r) {
    return std::exp(-m*m) * std::exp(-t) * std::sin(v * 3.14159) * (1.0 + r);
}

/// Benchmark helper
template<typename Func>
double benchmark(Func&& f, int iterations = 5) {
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(elapsed);
    }

    // Return median time
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

/// Old axis order fitter (for comparison)
/// This replicates the old m → τ → σ → r order
class BSplineFitterOldOrder {
public:
    BSplineFitterOldOrder(const std::vector<double>& m_grid,
                          const std::vector<double>& t_grid,
                          const std::vector<double>& v_grid,
                          const std::vector<double>& r_grid)
        : m_grid_(m_grid), t_grid_(t_grid), v_grid_(v_grid), r_grid_(r_grid)
        , Nm_(m_grid.size()), Nt_(t_grid.size()), Nv_(v_grid.size()), Nr_(r_grid.size())
    {
        solver_m_ = std::make_unique<BSplineCollocation1D>(m_grid_);
        solver_t_ = std::make_unique<BSplineCollocation1D>(t_grid_);
        solver_v_ = std::make_unique<BSplineCollocation1D>(v_grid_);
        solver_r_ = std::make_unique<BSplineCollocation1D>(r_grid_);
    }

    std::vector<double> fit(const std::vector<double>& values, double tolerance = 1e-6) {
        std::vector<double> coeffs = values;
        std::vector<double> slice_m(Nm_), slice_t(Nt_), slice_v(Nv_), slice_r(Nr_);

        // OLD ORDER: m → τ → σ → r (poor cache locality)

        // Step 1: m-axis (stride = Nt*Nv*Nr, WORST cache locality)
        for (size_t j = 0; j < Nt_; ++j) {
            for (size_t k = 0; k < Nv_; ++k) {
                for (size_t l = 0; l < Nr_; ++l) {
                    for (size_t i = 0; i < Nm_; ++i) {
                        slice_m[i] = coeffs[((i * Nt_ + j) * Nv_ + k) * Nr_ + l];
                    }
                    auto result = solver_m_->fit(slice_m, tolerance);
                    for (size_t i = 0; i < Nm_; ++i) {
                        coeffs[((i * Nt_ + j) * Nv_ + k) * Nr_ + l] = result.coefficients[i];
                    }
                }
            }
        }

        // Step 2: τ-axis (stride = Nv*Nr)
        for (size_t i = 0; i < Nm_; ++i) {
            for (size_t k = 0; k < Nv_; ++k) {
                for (size_t l = 0; l < Nr_; ++l) {
                    for (size_t j = 0; j < Nt_; ++j) {
                        slice_t[j] = coeffs[((i * Nt_ + j) * Nv_ + k) * Nr_ + l];
                    }
                    auto result = solver_t_->fit(slice_t, tolerance);
                    for (size_t j = 0; j < Nt_; ++j) {
                        coeffs[((i * Nt_ + j) * Nv_ + k) * Nr_ + l] = result.coefficients[j];
                    }
                }
            }
        }

        // Step 3: σ-axis (stride = Nr)
        for (size_t i = 0; i < Nm_; ++i) {
            for (size_t j = 0; j < Nt_; ++j) {
                for (size_t l = 0; l < Nr_; ++l) {
                    for (size_t k = 0; k < Nv_; ++k) {
                        slice_v[k] = coeffs[((i * Nt_ + j) * Nv_ + k) * Nr_ + l];
                    }
                    auto result = solver_v_->fit(slice_v, tolerance);
                    for (size_t k = 0; k < Nv_; ++k) {
                        coeffs[((i * Nt_ + j) * Nv_ + k) * Nr_ + l] = result.coefficients[k];
                    }
                }
            }
        }

        // Step 4: r-axis (stride = 1, but done LAST)
        for (size_t i = 0; i < Nm_; ++i) {
            for (size_t j = 0; j < Nt_; ++j) {
                for (size_t k = 0; k < Nv_; ++k) {
                    for (size_t l = 0; l < Nr_; ++l) {
                        slice_r[l] = coeffs[((i * Nt_ + j) * Nv_ + k) * Nr_ + l];
                    }
                    auto result = solver_r_->fit(slice_r, tolerance);
                    for (size_t l = 0; l < Nr_; ++l) {
                        coeffs[((i * Nt_ + j) * Nv_ + k) * Nr_ + l] = result.coefficients[l];
                    }
                }
            }
        }

        return coeffs;
    }

private:
    std::vector<double> m_grid_, t_grid_, v_grid_, r_grid_;
    size_t Nm_, Nt_, Nv_, Nr_;
    std::unique_ptr<BSplineCollocation1D> solver_m_, solver_t_, solver_v_, solver_r_;
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "B-Spline Cache Locality Benchmark\n";
    std::cout << "=================================================================\n\n";

    // Test different grid sizes
    struct GridConfig {
        int Nm, Nt, Nv, Nr;
        const char* description;
    };

    std::vector<GridConfig> configs = {
        {20, 15, 10, 8,  "Small  (20×15×10×8)"},
        {50, 30, 20, 10, "Medium (50×30×20×10)"},
        {80, 40, 25, 15, "Large  (80×40×25×15)"},
    };

    for (const auto& config : configs) {
        std::cout << "Grid: " << config.description << "\n";
        std::cout << "  Total points: " << (config.Nm * config.Nt * config.Nv * config.Nr) << "\n";

        // Compute strides
        int stride_r = 1;
        int stride_sigma = config.Nr;
        int stride_tau = config.Nv * config.Nr;
        int stride_m = config.Nt * config.Nv * config.Nr;

        std::cout << "  Memory strides:\n";
        std::cout << "    r-axis: " << stride_r << " doubles = " << (stride_r * 8) << " bytes\n";
        std::cout << "    σ-axis: " << stride_sigma << " doubles = " << (stride_sigma * 8) << " bytes\n";
        std::cout << "    τ-axis: " << stride_tau << " doubles = " << (stride_tau * 8) << " bytes\n";
        std::cout << "    m-axis: " << stride_m << " doubles = " << (stride_m * 8) << " bytes";

        if (stride_m * 8 > 32768) {
            std::cout << " (exceeds L1 cache!)";
        }
        std::cout << "\n\n";

        // Create grids
        auto m_grid = linspace(0.7, 1.3, config.Nm);
        auto t_grid = linspace(0.1, 2.0, config.Nt);
        auto v_grid = linspace(0.1, 0.8, config.Nv);
        auto r_grid = linspace(0.0, 0.1, config.Nr);

        // Generate test data
        std::vector<double> values(config.Nm * config.Nt * config.Nv * config.Nr);
        for (int i = 0; i < config.Nm; ++i) {
            for (int j = 0; j < config.Nt; ++j) {
                for (int k = 0; k < config.Nv; ++k) {
                    for (int l = 0; l < config.Nr; ++l) {
                        size_t idx = ((i * config.Nt + j) * config.Nv + k) * config.Nr + l;
                        values[idx] = test_function(m_grid[i], t_grid[j], v_grid[k], r_grid[l]);
                    }
                }
            }
        }

        // Benchmark OLD order (m → τ → σ → r)
        BSplineFitterOldOrder fitter_old(m_grid, t_grid, v_grid, r_grid);
        auto old_time = benchmark([&]() {
            auto result = fitter_old.fit(values, 1e-6);
        }, 5);

        // Benchmark NEW order (r → σ → τ → m)
        BSplineFitter4DSeparable fitter_new(m_grid, t_grid, v_grid, r_grid);
        auto new_time = benchmark([&]() {
            auto result = fitter_new.fit(values, 1e-6);
        }, 5);

        std::cout << "  OLD order (m → τ → σ → r): " << std::fixed << std::setprecision(2)
                  << old_time << " ms\n";
        std::cout << "  NEW order (r → σ → τ → m): " << new_time << " ms\n";

        double speedup = old_time / new_time;
        std::cout << "  Speedup: " << std::setprecision(2) << speedup << "x\n";

        if (speedup < 1.0) {
            std::cout << "  ⚠️  New order is slower (may indicate small grid or hardware variation)\n";
        } else if (speedup < 1.2) {
            std::cout << "  ✓ Modest improvement (cache effects less pronounced)\n";
        } else if (speedup < 2.0) {
            std::cout << "  ✓✓ Good improvement (clear cache benefit)\n";
        } else {
            std::cout << "  ✓✓✓ Excellent improvement (strong cache optimization)\n";
        }

        std::cout << "\n";
    }

    std::cout << "=================================================================\n";
    std::cout << "Summary\n";
    std::cout << "=================================================================\n";
    std::cout << "The cache-optimized axis order (r → σ → τ → m) processes the\n";
    std::cout << "fastest-varying dimensions first, keeping the working set hot in\n";
    std::cout << "cache during early passes.\n\n";
    std::cout << "For the old order (m → τ → σ → r), the m-axis pass accesses\n";
    std::cout << "memory with stride Nt*Nv*Nr, often exceeding L1 cache size.\n\n";
    std::cout << "Performance improvement varies by:\n";
    std::cout << "  - Grid size (larger grids benefit more)\n";
    std::cout << "  - CPU cache architecture (smaller L1 = bigger benefit)\n";
    std::cout << "  - Memory bandwidth (bottleneck systems benefit more)\n";
    std::cout << "=================================================================\n";

    return 0;
}
