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

BENCHMARK(BM_Pack)->Arg(101)->Arg(501)->Arg(1001);
BENCHMARK_MAIN();
