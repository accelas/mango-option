// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/greek_types.hpp"

using namespace mango;

TEST(GreekTypesTest, EnumValues) {
    // Greek enum has the four first-order types
    EXPECT_NE(static_cast<int>(Greek::Delta), static_cast<int>(Greek::Vega));
    EXPECT_NE(static_cast<int>(Greek::Theta), static_cast<int>(Greek::Rho));
}

TEST(GreekTypesTest, GreekErrorValues) {
    GreekError e1 = GreekError::OutOfDomain;
    GreekError e2 = GreekError::NumericalFailure;
    EXPECT_NE(e1, e2);
}
