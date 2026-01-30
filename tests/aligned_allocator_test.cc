// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/support/aligned_allocator.hpp"
#include <numeric>

namespace mango {
namespace {

TEST(AlignedAllocatorTest, VectorIsAligned) {
    AlignedVector<double> vec(100);

    // Verify 64-byte alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(vec.data());
    EXPECT_EQ(addr % 64, 0) << "Vector should be 64-byte aligned for AVX-512";
}

TEST(AlignedAllocatorTest, DataAccess) {
    AlignedVector<double> vec(10);

    // Write values
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<double>(i * 2);
    }

    // Read values
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec[i], static_cast<double>(i * 2));
    }
}

TEST(AlignedAllocatorTest, EmptyVector) {
    AlignedVector<double> vec;
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
}

TEST(AlignedAllocatorTest, PushBack) {
    AlignedVector<double> vec;
    vec.push_back(1.0);
    vec.push_back(2.0);
    vec.push_back(3.0);

    EXPECT_EQ(vec.size(), 3);
    EXPECT_DOUBLE_EQ(vec[0], 1.0);
    EXPECT_DOUBLE_EQ(vec[1], 2.0);
    EXPECT_DOUBLE_EQ(vec[2], 3.0);

    // Still aligned after push_back
    uintptr_t addr = reinterpret_cast<uintptr_t>(vec.data());
    EXPECT_EQ(addr % 64, 0);
}

TEST(AlignedAllocatorTest, LargeAllocation) {
    // Allocate a large vector
    constexpr size_t size = 1000000;  // 1M elements = 8MB
    AlignedVector<double> vec(size);

    // Fill with values
    std::iota(vec.begin(), vec.end(), 0.0);

    // Verify alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(vec.data());
    EXPECT_EQ(addr % 64, 0);

    // Verify data
    EXPECT_DOUBLE_EQ(vec[0], 0.0);
    EXPECT_DOUBLE_EQ(vec[size - 1], static_cast<double>(size - 1));
}

TEST(AlignedAllocatorTest, CopyConstruct) {
    AlignedVector<double> vec1(10);
    for (size_t i = 0; i < vec1.size(); ++i) {
        vec1[i] = static_cast<double>(i);
    }

    AlignedVector<double> vec2 = vec1;

    // Both should be aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(vec1.data()) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(vec2.data()) % 64, 0);

    // Data should be equal but not the same pointer
    EXPECT_NE(vec1.data(), vec2.data());
    for (size_t i = 0; i < vec1.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec1[i], vec2[i]);
    }
}

TEST(AlignedAllocatorTest, MoveConstruct) {
    AlignedVector<double> vec1(10);
    for (size_t i = 0; i < vec1.size(); ++i) {
        vec1[i] = static_cast<double>(i);
    }
    double* original_ptr = vec1.data();

    AlignedVector<double> vec2 = std::move(vec1);

    // vec2 should have taken ownership
    EXPECT_EQ(vec2.data(), original_ptr);
    EXPECT_EQ(vec2.size(), 10);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(vec2.data()) % 64, 0);

    // Verify data
    for (size_t i = 0; i < vec2.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec2[i], static_cast<double>(i));
    }
}

TEST(AlignedAllocatorTest, ResizePreservesAlignment) {
    AlignedVector<double> vec(10);

    // Resize up
    vec.resize(100);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(vec.data()) % 64, 0);

    // Resize down
    vec.resize(5);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(vec.data()) % 64, 0);

    // Resize back up with value
    vec.resize(50, 42.0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(vec.data()) % 64, 0);
    EXPECT_DOUBLE_EQ(vec[49], 42.0);
}

// REGRESSION TEST: Verify allocator equality comparison works
TEST(AlignedAllocatorTest, AllocatorEquality) {
    AlignedAllocator<double> alloc1;
    AlignedAllocator<double> alloc2;

    // Same type allocators should be equal
    EXPECT_TRUE(alloc1 == alloc2);
    EXPECT_FALSE(alloc1 != alloc2);

    // Rebind allocator should also be equal
    AlignedAllocator<int> alloc3;
    AlignedAllocator<int> alloc4;
    EXPECT_TRUE(alloc3 == alloc4);
}

// REGRESSION TEST: Verify different types with same alignment are still comparable
TEST(AlignedAllocatorTest, CrossTypeEquality) {
    AlignedAllocator<double, 64> alloc_double;
    AlignedAllocator<int, 64> alloc_int;

    // Different types but same alignment should be equal
    EXPECT_TRUE(alloc_double == alloc_int);
}

} // namespace
} // namespace mango
