#include <gtest/gtest.h>
#include "src/option/price_tensor.hpp"
#include "src/support/memory/aligned_arena.hpp"

namespace mango {
namespace {

TEST(PriceTensorTest, Create2DTensor) {
    auto arena = memory::AlignedArena::create(1024).value();

    auto result = PriceTensor<2>::create({3, 4}, arena);
    ASSERT_TRUE(result.has_value());

    auto tensor = result.value();
    EXPECT_EQ(tensor.view.extent(0), 3);
    EXPECT_EQ(tensor.view.extent(1), 4);
    EXPECT_EQ(tensor.arena, arena);
}

TEST(PriceTensorTest, AccessElements) {
    auto arena = memory::AlignedArena::create(1024).value();
    auto tensor = PriceTensor<2>::create({2, 3}, arena).value();

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
    auto arena = memory::AlignedArena::create(10000).value();

    auto result = PriceTensor<4>::create({5, 4, 3, 2}, arena);
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

TEST(PriceTensorTest, ArenaOutOfMemory) {
    auto arena = memory::AlignedArena::create(100).value();  // Too small

    auto result = PriceTensor<3>::create({100, 100, 100}, arena);  // Needs 100*100*100*8 bytes
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("memory"), std::string::npos);
}

TEST(PriceTensorTest, Create1DTensor) {
    auto arena = memory::AlignedArena::create(1024).value();

    auto result = PriceTensor<1>::create({10}, arena);
    ASSERT_TRUE(result.has_value());

    auto tensor = result.value();
    EXPECT_EQ(tensor.view.extent(0), 10);

    // Test element access
    tensor.view[5] = 42.0;
    EXPECT_DOUBLE_EQ((tensor.view[5]), 42.0);
}

TEST(PriceTensorTest, Create5DTensor) {
    auto arena = memory::AlignedArena::create(10000).value();

    auto result = PriceTensor<5>::create({2, 3, 4, 5, 6}, arena);
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

} // namespace
} // namespace mango
