#include <benchmark/benchmark.h>
#include <experimental/simd>
#include <vector>
#include <span>
#include <algorithm>

namespace stdx = std::experimental;

// Placeholder for pack/scatter/stencil benchmarks
static void BM_Placeholder(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(state.range(0));
    }
}

BENCHMARK(BM_Placeholder)->Arg(101);
BENCHMARK_MAIN();
