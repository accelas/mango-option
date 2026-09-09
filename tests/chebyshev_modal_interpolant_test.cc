// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/chebyshev_modal_interpolant.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
TEST(ChebyshevModalInterpolantTest,
     ValuesAndDerivativesUseOneStoredPolynomial) {
  const mango::Domain<2> domain{{-2., 1.}, {2., 3.}};
  const std::array<size_t, 2> shape{5, 4};
  std::vector<double> values;
  for (double x : mango::chebyshev_nodes(shape[0], -2., 2.)) {
    for (double y : mango::chebyshev_nodes(shape[1], 1., 3.)) {
      values.push_back(1 + 2 * x + 3 * y + 4 * x * y + 5 * x * x);
    }
  }
  auto result = mango::ChebyshevModalInterpolant<2>::build_from_values(
      values, domain, shape);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->domain().lo, domain.lo);
  EXPECT_EQ(result->domain().hi, domain.hi);
  EXPECT_EQ(result->num_pts(), shape);
  EXPECT_EQ(result->compressed_size(), values.size());
  for (double x : {-2., -.37, 0., 1.19, 2.}) {
    for (double y : {1., 1.23, 2.61, 3.}) {
      const std::array<double, 2> point{x, y};
      EXPECT_NEAR(result->eval(point),
                  1 + 2 * x + 3 * y + 4 * x * y + 5 * x * x, 2e-13);
      EXPECT_NEAR(result->partial(0, point), 2 + 4 * y + 10 * x, 3e-13);
      EXPECT_NEAR(result->partial(1, point), 3 + 4 * x, 3e-13);
      EXPECT_NEAR(result->eval_second_partial(0, point), 10, 1e-12);
      EXPECT_NEAR(result->eval_second_partial(1, point), 0, 1e-12);
      EXPECT_DOUBLE_EQ(result->eval(point), result->polynomial().eval(point));
      EXPECT_DOUBLE_EQ(result->partial(0, point),
                       result->polynomial().partial(0, point));
    }
  }
}

TEST(ChebyshevModalInterpolantTest,
     ExplicitCoefficientsPreserveShapeAndCopyOwnership) {
  std::vector<double> coefficients(257, 0.);
  coefficients[23] = -0.;
  coefficients.back() = 1.;
  auto result = mango::ChebyshevModalInterpolant<1>::build_from_coefficients(
      coefficients, mango::Domain<1>{{-1.}, {1.}}, std::array<size_t, 1>{257});
  ASSERT_TRUE(result);
  coefficients.back() = 9.;
  auto copy = *result;
  EXPECT_NE(copy.polynomial().coefficients().data(),
            result->polynomial().coefficients().data());
  EXPECT_EQ(copy.num_pts()[0], 257);
  EXPECT_EQ(copy.polynomial().shape()[0], 257);
  EXPECT_EQ(std::bit_cast<uint64_t>(copy.polynomial().coefficients()[23]),
            std::bit_cast<uint64_t>(-0.));
  EXPECT_DOUBLE_EQ(copy.eval({1.}), 1.);
  EXPECT_DOUBLE_EQ(copy.partial(0, {1.}), 65536.);
  EXPECT_DOUBLE_EQ(copy.eval_second_partial(0, {1.}), 1431633920.);
  EXPECT_DOUBLE_EQ(copy.partial(0, {2.}), 0.);
  EXPECT_TRUE(std::isnan(copy.partial(1, {0.})));
}

TEST(ChebyshevModalInterpolantTest, DomainShapeAndNonfiniteInputsAreRefused) {
  using Modal = mango::ChebyshevModalInterpolant<1>;
  EXPECT_FALSE(
      Modal::build_from_coefficients(std::array{1., 2.}, {{1.}, {1.}}, {2}));
  EXPECT_FALSE(
      Modal::build_from_coefficients(std::array{1., 2.}, {{2.}, {1.}}, {2}));
  EXPECT_FALSE(
      Modal::build_from_coefficients(std::array{1., 2.}, {{0.}, {1.}}, {3}));
  EXPECT_FALSE(Modal::build_from_values(std::array{1.}, {{0.}, {1.}}, {1}));
  EXPECT_FALSE(Modal::build_from_coefficients(
      std::array{1., std::numeric_limits<double>::infinity()}, {{0.}, {1.}},
      {2}));
  EXPECT_FALSE(Modal::build_from_coefficients(std::vector<double>(258),
                                              {{0.}, {1.}}, {258}));
}
} // namespace
