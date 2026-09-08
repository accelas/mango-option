// SPDX-License-Identifier: MIT
#include "mango/math/proof/interval.hpp"
#include <gtest/gtest.h>
#include <limits>
namespace {
using mango::detail::proof::Interval;
TEST(ProofIntervalTest, RetainsPositiveResidualBelowBinary64Resolution) {
    // The represented expression is exactly (1 + epsilon) - 1, not a
    // previously rounded stored coefficient. Its value is epsilon > 0.
    const auto result = (Interval(1) + Interval(1e-25)) - Interval(1);
    EXPECT_TRUE(result.strictly_positive());
    EXPECT_TRUE(result.contains(1e-25));
}
} // namespace

TEST(ProofIntervalTest, EnclosesSignsSubnormalsAndRejectsUndefinedExpressions) {
    using namespace mango::detail::proof;
    const auto a = Interval::hull(-2, 3);
    const auto b = Interval::hull(-4, 5);
    const auto product = a * b;
    EXPECT_EQ(product.lower_bound(), -12);
    EXPECT_EQ(product.upper_bound(), 15);
    EXPECT_TRUE((Interval(0) * a).exact_zero());
    EXPECT_TRUE((Interval(3) - Interval(3)).exact_zero());
    const auto tiny = Interval(std::numeric_limits<double>::denorm_min()) / Interval(2);
    EXPECT_TRUE(tiny.strictly_positive());
    EXPECT_EQ(tiny.lower_bound(), 0);
    EXPECT_EQ(tiny.upper_bound(), std::numeric_limits<double>::denorm_min());
    EXPECT_FALSE((Interval(1) / a).finite());
    EXPECT_FALSE((Interval(0) * Interval(std::numeric_limits<double>::infinity())).finite());
    EXPECT_TRUE(exp(Interval(0)).contains(1));
    EXPECT_TRUE(log(Interval(1)).exact_zero());
    EXPECT_TRUE(sqrt(Interval(4)).contains(2));
    EXPECT_FALSE(log(Interval(0)).finite());
    EXPECT_FALSE(sqrt(Interval(-1)).finite());
}

TEST(ProofIntervalTest, RestrictedExternalExponentRangeCannotRoundImportedConstantsToZero) {
    using namespace mango::detail::proof;
    struct RestoreRange {
        mpfr_exp_t previous = mpfr_get_emin();
        ~RestoreRange() { mpfr_set_emin(previous); }
    } restore;
    ASSERT_EQ(mpfr_set_emin(-10), 0);
    EXPECT_FALSE(Interval(1e-25).finite());
}
