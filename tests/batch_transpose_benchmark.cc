#include <benchmark/benchmark.h>
#include <experimental/simd>
#include <vector>
#include <span>
#include <algorithm>

namespace stdx = std::experimental;

template<typename T>
void pack_contracts_simd(std::span<std::span<const T>> u_soa,
                         std::span<T> u_aos,
                         size_t n, size_t batch_width) {
    using simd_t = stdx::native_simd<T>;
    constexpr size_t simd_width = simd_t::size();

    for (size_t i = 0; i < n; ++i) {
        size_t lane = 0;

        // Vectorized transpose
        for (; lane + simd_width <= batch_width; lane += simd_width) {
            simd_t chunk;
            for (size_t k = 0; k < simd_width; ++k) {
                chunk[k] = u_soa[lane + k][i];
            }
            chunk.copy_to(&u_aos[i * batch_width + lane], stdx::element_aligned);
        }

        // Scalar tail
        for (; lane < batch_width; ++lane) {
            u_aos[i * batch_width + lane] = u_soa[lane][i];
        }
    }
}

template<typename T>
void scatter_to_soa_simd(std::span<const T> lu_aos,
                         std::span<std::span<T>> lu_soa,
                         size_t n, size_t batch_width) {
    using simd_t = stdx::native_simd<T>;
    constexpr size_t simd_width = simd_t::size();

    for (size_t i = 0; i < n; ++i) {
        size_t lane = 0;

        // Vectorized transpose
        for (; lane + simd_width <= batch_width; lane += simd_width) {
            simd_t chunk;
            chunk.copy_from(&lu_aos[i * batch_width + lane], stdx::element_aligned);
            for (size_t k = 0; k < simd_width; ++k) {
                lu_soa[lane + k][i] = chunk[k];
            }
        }

        // Scalar tail
        for (; lane < batch_width; ++lane) {
            lu_soa[lane][i] = lu_aos[i * batch_width + lane];
        }
    }
}

static void BM_Pack(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t batch_width = stdx::native_simd<double>::size();

    // Setup SoA buffers
    std::vector<std::vector<double>> soa(batch_width, std::vector<double>(n, 1.0));
    std::vector<std::span<const double>> soa_spans;
    for (auto& v : soa) soa_spans.push_back(std::span{v});

    std::vector<double> aos(n * batch_width);

    for (auto _ : state) {
        pack_contracts_simd(std::span{soa_spans}, std::span{aos}, n, batch_width);
        benchmark::DoNotOptimize(aos.data());
    }

    state.SetItemsProcessed(state.iterations() * n * batch_width);
}

static void BM_Scatter(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t batch_width = stdx::native_simd<double>::size();

    std::vector<double> aos(n * batch_width, 1.0);
    std::vector<std::vector<double>> soa(batch_width, std::vector<double>(n));
    std::vector<std::span<double>> soa_spans;
    for (auto& v : soa) soa_spans.push_back(std::span{v});

    for (auto _ : state) {
        scatter_to_soa_simd(std::span<const double>{aos}, std::span{soa_spans}, n, batch_width);
        benchmark::DoNotOptimize(soa.data());
    }

    state.SetItemsProcessed(state.iterations() * n * batch_width);
}

static void BM_BatchedStencil(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t batch_width = stdx::native_simd<double>::size();

    std::vector<double> u_aos(n * batch_width, 1.0);
    std::vector<double> d2u_aos(n * batch_width);

    // Simplified batch stencil (uniform grid)
    const double dx2_inv = 1.0;
    using simd_t = stdx::native_simd<double>;
    constexpr size_t simd_width = simd_t::size();

    for (auto _ : state) {
        // Loop over grid points
        for (size_t i = 1; i < n - 1; ++i) {
            // Vectorized contract batches
            size_t lane = 0;
            for (; lane + simd_width <= batch_width; lane += simd_width) {
                simd_t u_left, u_center, u_right;
                u_left.copy_from(&u_aos[(i-1)*batch_width + lane], stdx::element_aligned);
                u_center.copy_from(&u_aos[i*batch_width + lane], stdx::element_aligned);
                u_right.copy_from(&u_aos[(i+1)*batch_width + lane], stdx::element_aligned);

                simd_t result = (u_left + u_right - 2.0 * u_center) * dx2_inv;
                result.copy_to(&d2u_aos[i*batch_width + lane], stdx::element_aligned);
            }

            // Scalar tail
            for (; lane < batch_width; ++lane) {
                size_t idx = i * batch_width + lane;
                d2u_aos[idx] = (u_aos[(i-1)*batch_width + lane] +
                               u_aos[(i+1)*batch_width + lane] -
                               2.0 * u_aos[idx]) * dx2_inv;
            }
        }
        benchmark::DoNotOptimize(d2u_aos.data());
    }

    state.SetItemsProcessed(state.iterations() * (n - 2) * batch_width);
}

BENCHMARK(BM_Pack)->Arg(101)->Arg(501)->Arg(1001);
BENCHMARK(BM_Scatter)->Arg(101)->Arg(501)->Arg(1001);
BENCHMARK(BM_BatchedStencil)->Arg(101)->Arg(501)->Arg(1001);
BENCHMARK_MAIN();
