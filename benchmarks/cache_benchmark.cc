#include <benchmark/benchmark.h>
#include <vector>
#include <random>

extern "C" {
#include "../src/price_table.h"
}

// Benchmark slice extraction performance

static OptionPriceTable* create_bench_table(MemoryLayout layout) {
    const size_t n_m = 30, n_tau = 25, n_sigma = 15, n_r = 10;

    std::vector<double> m(n_m), tau(n_tau), sigma(n_sigma), r(n_r);

    for (size_t i = 0; i < n_m; i++) m[i] = 0.8 + i * 0.5 / (n_m - 1);
    for (size_t i = 0; i < n_tau; i++) tau[i] = 0.1 + i * 1.9 / (n_tau - 1);
    for (size_t i = 0; i < n_sigma; i++) sigma[i] = 0.1 + i * 0.4 / (n_sigma - 1);
    for (size_t i = 0; i < n_r; i++) r[i] = 0.0 + i * 0.08 / (n_r - 1);

    OptionPriceTable *table = price_table_create_ex(
        m.data(), n_m, tau.data(), n_tau, sigma.data(), n_sigma,
        r.data(), n_r, nullptr, 0,
        OPTION_PUT, AMERICAN, COORD_RAW, layout);

    // Fill with dummy data
    for (size_t i = 0; i < n_m * n_tau * n_sigma * n_r; i++) {
        table->prices[i] = i * 0.01;
    }

    return table;
}

static void BM_SliceExtraction_M_OUTER(benchmark::State& state) {
    OptionPriceTable *table = create_bench_table(LAYOUT_M_OUTER);
    double slice[30];
    bool contiguous;
    int fixed[] = {-1, 10, 5, 3, 0};

    for (auto _ : state) {
        price_table_extract_slice(table, SLICE_DIM_MONEYNESS, fixed, slice, &contiguous);
        benchmark::DoNotOptimize(slice[0]);
    }

    state.SetLabel(contiguous ? "contiguous" : "strided");
    price_table_destroy(table);
}

static void BM_SliceExtraction_M_INNER(benchmark::State& state) {
    OptionPriceTable *table = create_bench_table(LAYOUT_M_INNER);
    double slice[30];
    bool contiguous;
    int fixed[] = {-1, 10, 5, 3, 0};

    for (auto _ : state) {
        price_table_extract_slice(table, SLICE_DIM_MONEYNESS, fixed, slice, &contiguous);
        benchmark::DoNotOptimize(slice[0]);
    }

    state.SetLabel(contiguous ? "contiguous" : "strided");
    price_table_destroy(table);
}

BENCHMARK(BM_SliceExtraction_M_OUTER);
BENCHMARK(BM_SliceExtraction_M_INNER);

BENCHMARK_MAIN();
