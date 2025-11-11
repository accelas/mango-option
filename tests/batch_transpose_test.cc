#include <gtest/gtest.h>
#include <experimental/simd>
#include <vector>
#include <span>
#include <cmath>

namespace stdx = std::experimental;

// Copy pack_contracts_simd and scatter_to_soa_simd from benchmark file
// (In real implementation, these would be in a header)

template<typename T>
void pack_contracts_simd(std::span<std::span<const T>> u_soa,
                         std::span<T> u_aos,
                         size_t n, size_t batch_width) {
    using simd_t = stdx::native_simd<T>;
    constexpr size_t simd_width = simd_t::size();

    for (size_t i = 0; i < n; ++i) {
        size_t lane = 0;
        for (; lane + simd_width <= batch_width; lane += simd_width) {
            simd_t chunk;
            for (size_t k = 0; k < simd_width; ++k) {
                chunk[k] = u_soa[lane + k][i];
            }
            chunk.copy_to(&u_aos[i * batch_width + lane], stdx::element_aligned);
        }
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
        for (; lane + simd_width <= batch_width; lane += simd_width) {
            simd_t chunk;
            chunk.copy_from(&lu_aos[i * batch_width + lane], stdx::element_aligned);
            for (size_t k = 0; k < simd_width; ++k) {
                lu_soa[lane + k][i] = chunk[k];
            }
        }
        for (; lane < batch_width; ++lane) {
            lu_soa[lane][i] = lu_aos[i * batch_width + lane];
        }
    }
}

TEST(BatchTranspose, BitwiseIdentity) {
    constexpr size_t n = 101;
    const size_t batch_width = stdx::native_simd<double>::size();

    // Setup test data
    std::vector<std::vector<double>> original_soa(batch_width, std::vector<double>(n));
    for (size_t lane = 0; lane < batch_width; ++lane) {
        for (size_t i = 0; i < n; ++i) {
            original_soa[lane][i] = std::sin(i * 0.1) + lane * 0.01;
        }
    }

    // Pack to AoS
    std::vector<double> aos(n * batch_width);
    std::vector<std::span<const double>> soa_spans;
    for (auto& v : original_soa) soa_spans.push_back(std::span{v});
    pack_contracts_simd(std::span{soa_spans}, std::span{aos}, n, batch_width);

    // Scatter back to SoA
    std::vector<std::vector<double>> result_soa(batch_width, std::vector<double>(n));
    std::vector<std::span<double>> result_spans;
    for (auto& v : result_soa) result_spans.push_back(std::span{v});
    scatter_to_soa_simd(std::span<const double>{aos}, std::span{result_spans}, n, batch_width);

    // Verify bitwise match
    for (size_t lane = 0; lane < batch_width; ++lane) {
        for (size_t i = 0; i < n; ++i) {
            EXPECT_EQ(original_soa[lane][i], result_soa[lane][i])
                << "Mismatch at lane=" << lane << " i=" << i;
        }
    }
}
