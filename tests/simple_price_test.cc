#include <gtest/gtest.h>
#include "src/simple/price.hpp"

TEST(SimplePriceTest, ConstructFromDouble) {
    mango::simple::Price p{100.50};
    EXPECT_DOUBLE_EQ(p.to_double(), 100.50);
}

TEST(SimplePriceTest, ConstructFromFixedPoint9) {
    // Databento format: price * 10^9
    int64_t fixed = 100500000000LL;  // 100.50 * 10^9
    mango::simple::Price p{fixed, mango::simple::PriceFormat::FixedPoint9};
    EXPECT_DOUBLE_EQ(p.to_double(), 100.50);
}

TEST(SimplePriceTest, MidpointPreservesPrecision) {
    // Two fixed-point prices
    mango::simple::Price bid{100250000000LL, mango::simple::PriceFormat::FixedPoint9};
    mango::simple::Price ask{100750000000LL, mango::simple::PriceFormat::FixedPoint9};

    auto mid = mango::simple::Price::midpoint(bid, ask);
    ASSERT_TRUE(mid.has_value());
    EXPECT_DOUBLE_EQ(mid->to_double(), 100.50);
}

TEST(SimplePriceTest, MidpointMixedFormats) {
    // Mixed formats: converts to double for midpoint
    mango::simple::Price bid{100.25};
    mango::simple::Price ask{100750000000LL, mango::simple::PriceFormat::FixedPoint9};

    auto mid = mango::simple::Price::midpoint(bid, ask);
    ASSERT_TRUE(mid.has_value());
    EXPECT_DOUBLE_EQ(mid->to_double(), 100.50);
}
