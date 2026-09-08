// SPDX-License-Identifier: MIT
#include "mango/option/table/moneyness_bounds.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
namespace {
using namespace mango;
TEST(MoneynessBoundsTest, PreservesRequestedQuoteEndpointsAndRejectsNextSpot) {
    MoneynessDomain domain({.1, .13});
    for (double strike : {1., 3.14, 100., 12345.}) {
        const double lower = strike * .1, upper = strike * .13;
        EXPECT_TRUE(domain.contains_quote(lower, strike));
        EXPECT_TRUE(domain.contains_quote(upper, strike));
        EXPECT_FALSE(domain.contains_quote(std::nextafter(lower, 0), strike));
        EXPECT_FALSE(domain.contains_quote(std::nextafter(upper, INFINITY), strike));
    }
    EXPECT_TRUE(domain.contains_quote(10, 100));
}
TEST(MoneynessBoundsTest, RejectsSubnormalQuoteAliasesWithoutAMinimumCurrencyUnit) {
    const double quantum = std::numeric_limits<double>::denorm_min();
    MoneynessDomain normal({std::exp(-.5), std::exp(.5)});
    EXPECT_FALSE(normal.contains_quote(quantum, 2 * quantum));
    EXPECT_TRUE(normal.contains_quote(quantum, quantum));
    EXPECT_FALSE(normal.contains_quote(0, quantum));
}
TEST(MoneynessBoundsTest, EveryAcceptedExtremeQuoteHasAnEnclosedExactRatio) {
    const double quantum = std::numeric_limits<double>::denorm_min();
    for (double lower : {.1, .3, .607, .92, 1.01, 1.7}) {
        MoneynessDomain domain({lower, lower + .3});
        for (int k = 1; k <= 32; ++k)
            for (int s = 1; s <= 80; ++s) {
                if (domain.contains_quote(s * quantum, k * quantum)) {
                    // Integer subnormal multiples give an independent exact ratio
                    // whose long-double comparison has ample separation here.
                    const long double ratio = static_cast<long double>(s) / k;
                    EXPECT_GE(ratio, domain.enclosure().min);
                    EXPECT_LE(ratio, domain.enclosure().max);
                }
            }
    }
}
} // namespace
