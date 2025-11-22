#include "src/math/lapack_banded_layout.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace mango {
namespace {

TEST(LapackBandedLayout, MappingFormula) {
    // 4x4 matrix with kl=1, ku=1 (tridiagonal)
    // LAPACK formula: AB(kl + ku + i - j, j) = A(i, j)
    // ldab = 2*kl + ku + 1 = 2*1 + 1 + 1 = 4

    using Extents = std::experimental::dextents<size_t, 2>;
    using Layout = lapack_banded_layout;

    Layout::mapping<Extents> map(Extents{4, 4}, 1, 1);

    // A(0,0) -> AB(1 + 1 + 0 - 0, 0) = AB(2, 0) -> offset = 2 + 0*4 = 2
    EXPECT_EQ(map(0, 0), 2);

    // A(0,1) -> AB(1 + 1 + 0 - 1, 1) = AB(1, 1) -> offset = 1 + 1*4 = 5
    EXPECT_EQ(map(0, 1), 5);

    // A(1,0) -> AB(1 + 1 + 1 - 0, 0) = AB(3, 0) -> offset = 3 + 0*4 = 3
    EXPECT_EQ(map(1, 0), 3);

    // A(1,1) -> AB(1 + 1 + 1 - 1, 1) = AB(2, 1) -> offset = 2 + 1*4 = 6
    EXPECT_EQ(map(1, 1), 6);
}

TEST(LapackBandedLayout, RequiredSpanSize) {
    using Extents = std::experimental::dextents<size_t, 2>;
    using Layout = lapack_banded_layout;

    // 5x5 matrix, kl=2, ku=1
    // ldab = 2*2 + 1 + 1 = 6
    // required_span_size = ldab * n = 6 * 5 = 30

    Layout::mapping<Extents> map(Extents{5, 5}, 2, 1);
    EXPECT_EQ(map.required_span_size(), 30);
}

TEST(LapackBandedLayout, Strides) {
    using Extents = std::experimental::dextents<size_t, 2>;
    using Layout = lapack_banded_layout;

    Layout::mapping<Extents> map(Extents{4, 4}, 1, 1);

    // Column-major: row stride = 1, column stride = ldab
    EXPECT_EQ(map.stride(0), 1);   // Row stride
    EXPECT_EQ(map.stride(1), 4);   // Column stride (ldab = 4)
}

}  // namespace
}  // namespace mango
