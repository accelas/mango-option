#include <gtest/gtest.h>

extern "C" {
#include "../src/price_table.h"
}

TEST(CoordinateSystemTest, EnumValues) {
    EXPECT_EQ(COORD_RAW, 0);
    EXPECT_EQ(COORD_LOG_SQRT, 1);
    EXPECT_EQ(COORD_LOG_VARIANCE, 2);
}

TEST(MemoryLayoutTest, EnumValues) {
    EXPECT_EQ(LAYOUT_M_OUTER, 0);
    EXPECT_EQ(LAYOUT_M_INNER, 1);
    EXPECT_EQ(LAYOUT_BLOCKED, 2);
}
