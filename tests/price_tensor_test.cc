// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/price_tensor.hpp"

namespace mango {
namespace {

TEST(PriceTensorTest, Create2DTensor) {
    auto result = PriceTensor<2>::create({3, 4});
    ASSERT_TRUE(result.has_value());

    auto tensor = result.value();
    EXPECT_EQ(tensor.view.extent(0), 3);
    EXPECT_EQ(tensor.view.extent(1), 4);
}

TEST(PriceTensorTest, AccessElements) {
    auto tensor = PriceTensor<2>::create({2, 3}).value();

    // Write via mdspan
    tensor.view[0, 0] = 1.0;
    tensor.view[0, 1] = 2.0;
    tensor.view[1, 2] = 6.0;

    // Read via mdspan
    EXPECT_DOUBLE_EQ((tensor.view[0, 0]), 1.0);
    EXPECT_DOUBLE_EQ((tensor.view[0, 1]), 2.0);
    EXPECT_DOUBLE_EQ((tensor.view[1, 2]), 6.0);
}

TEST(PriceTensorTest, Create4DTensor) {
    auto result = PriceTensor<4>::create({5, 4, 3, 2});
    ASSERT_TRUE(result.has_value());

    auto tensor = result.value();
    EXPECT_EQ(tensor.view.extent(0), 5);
    EXPECT_EQ(tensor.view.extent(1), 4);
    EXPECT_EQ(tensor.view.extent(2), 3);
    EXPECT_EQ(tensor.view.extent(3), 2);

    // Total elements = 5*4*3*2 = 120
    size_t total = 1;
    for (size_t i = 0; i < 4; ++i) {
        total *= tensor.view.extent(i);
    }
    EXPECT_EQ(total, 120);
}

TEST(PriceTensorTest, ShapeOverflow) {
    // Request a shape that would overflow size_t
    auto result = PriceTensor<3>::create({SIZE_MAX, SIZE_MAX, SIZE_MAX});
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("overflow"), std::string::npos);
}

TEST(PriceTensorTest, Create1DTensor) {
    auto result = PriceTensor<1>::create({10});
    ASSERT_TRUE(result.has_value());

    auto tensor = result.value();
    EXPECT_EQ(tensor.view.extent(0), 10);

    // Test element access
    tensor.view[5] = 42.0;
    EXPECT_DOUBLE_EQ((tensor.view[5]), 42.0);
}

TEST(PriceTensorTest, Create5DTensor) {
    auto result = PriceTensor<5>::create({2, 3, 4, 5, 6});
    ASSERT_TRUE(result.has_value());

    auto tensor = result.value();
    EXPECT_EQ(tensor.view.extent(0), 2);
    EXPECT_EQ(tensor.view.extent(1), 3);
    EXPECT_EQ(tensor.view.extent(2), 4);
    EXPECT_EQ(tensor.view.extent(3), 5);
    EXPECT_EQ(tensor.view.extent(4), 6);

    // Total elements = 2*3*4*5*6 = 720
    size_t total = 1;
    for (size_t i = 0; i < 5; ++i) {
        total *= tensor.view.extent(i);
    }
    EXPECT_EQ(total, 720);

    // Test element access
    tensor.view[1, 2, 3, 4, 5] = 99.0;
    EXPECT_DOUBLE_EQ((tensor.view[1, 2, 3, 4, 5]), 99.0);
}

// REGRESSION TEST: Verify mdspan dependency compiles correctly
// Issue: Missing @mdspan//:mdspan dependency in BUILD.bazel would cause compile failure
TEST(PriceTensorTest, MdspanTypesCompile) {
    // This test verifies that mdspan types are available and work correctly
    auto tensor = PriceTensor<2>::create({3, 4}).value();

    // Verify mdspan types are accessible
    using MdspanType = decltype(tensor.view);
    using ElementType = typename MdspanType::element_type;

    // Verify element_type is double
    static_assert(std::is_same_v<ElementType, double>,
                  "mdspan element_type must be double");

    // Verify extents work correctly
    EXPECT_EQ(tensor.view.extent(0), 3);
    EXPECT_EQ(tensor.view.extent(1), 4);

    // Verify multidimensional indexing works (this would fail to compile without mdspan)
    tensor.view[0, 0] = 1.0;
    tensor.view[2, 3] = 12.0;
    EXPECT_DOUBLE_EQ((tensor.view[0, 0]), 1.0);
    EXPECT_DOUBLE_EQ((tensor.view[2, 3]), 12.0);
}

// REGRESSION TEST: Verify mdspan works for higher dimensions
TEST(PriceTensorTest, Mdspan4DTypesCompile) {
    auto tensor = PriceTensor<4>::create({2, 3, 4, 5}).value();

    // Verify 4D indexing compiles and works
    tensor.view[0, 1, 2, 3] = 42.0;
    tensor.view[1, 2, 3, 4] = 84.0;

    EXPECT_DOUBLE_EQ((tensor.view[0, 1, 2, 3]), 42.0);
    EXPECT_DOUBLE_EQ((tensor.view[1, 2, 3, 4]), 84.0);

    // Verify extent accessors
    EXPECT_EQ(tensor.view.extent(0), 2);
    EXPECT_EQ(tensor.view.extent(1), 3);
    EXPECT_EQ(tensor.view.extent(2), 4);
    EXPECT_EQ(tensor.view.extent(3), 5);
}

// REGRESSION TEST: Verify storage is 64-byte aligned for SIMD operations
TEST(PriceTensorTest, StorageIsAligned) {
    auto tensor = PriceTensor<2>::create({10, 10}).value();

    // Verify the storage pointer is 64-byte aligned
    uintptr_t addr = reinterpret_cast<uintptr_t>(tensor.storage->data());
    EXPECT_EQ(addr % 64, 0) << "Storage should be 64-byte aligned for AVX-512";
}

// REGRESSION TEST: Verify shared_ptr semantics work correctly
TEST(PriceTensorTest, SharedOwnership) {
    auto tensor1 = PriceTensor<2>::create({5, 5}).value();

    // Write a value
    tensor1.view[2, 2] = 42.0;

    // Copy the tensor (shared_ptr semantics)
    auto tensor2 = tensor1;

    // Both should see the same data
    EXPECT_DOUBLE_EQ((tensor1.view[2, 2]), 42.0);
    EXPECT_DOUBLE_EQ((tensor2.view[2, 2]), 42.0);

    // Modify through one, visible in other
    tensor2.view[2, 2] = 99.0;
    EXPECT_DOUBLE_EQ((tensor1.view[2, 2]), 99.0);

    // They share the same storage
    EXPECT_EQ(tensor1.storage.get(), tensor2.storage.get());
}

} // namespace
} // namespace mango
