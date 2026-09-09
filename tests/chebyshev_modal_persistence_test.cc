// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/chebyshev_modal_interpolant.hpp"
#include "mango/option/table/serialization/extract_segments.hpp"
#include "mango/option/table/serialization/reconstruct.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace {
using Modal = mango::ChebyshevModalInterpolant<4>;
using Transform = mango::StandardTransform4D;
using Leaf = mango::TransformLeaf<Modal, Transform>;

Leaf fixture() {
  std::vector<double> coefficients(24);
  coefficients[0] = 10.;
  coefficients[3] = -0.;
  coefficients[8] = 2.;
  coefficients[16] = .5;
  auto interp = Modal::build_from_coefficients(
      coefficients, mango::Domain<4>{{-.2, .05, .1, -.02}, {.2, 1., .5, .08}},
      {3, 2, 2, 2});
  return Leaf(std::move(interp.value()), Transform{}, 100.);
}

TEST(ChebyshevModalPersistenceTest,
     ModalRoundTripStoresCoefficientsAndRejectsNodalType) {
  auto leaf = fixture();
  std::vector<mango::PriceTableData::Segment> segments;
  mango::extract_segments(leaf, segments, 100., 0., 1., .05, 1.);
  ASSERT_EQ(segments.size(), 1);
  const auto &segment = segments.front();
  EXPECT_EQ(segment.interp_type, "chebyshev_modal");
  EXPECT_EQ(segment.num_pts, (std::vector<int32_t>{3, 2, 2, 2}));
  EXPECT_TRUE(segment.grids.empty());
  EXPECT_TRUE(segment.knots.empty());
  const auto coefficients = leaf.interpolant().polynomial().coefficients();
  ASSERT_EQ(segment.values.size(), coefficients.size());
  for (size_t i = 0; i < coefficients.size(); ++i) {
    EXPECT_EQ(std::bit_cast<uint64_t>(segment.values[i]),
              std::bit_cast<uint64_t>(coefficients[i]));
  }
  EXPECT_FALSE(mango::make_chebyshev<4>(segment));
  auto restored =
      mango::reconstruct_chebyshev_modal_leaf<4, Transform>(segment);
  ASSERT_TRUE(restored);
  for (double spot : {85., 93.7, 100., 113.2, 120.}) {
    EXPECT_DOUBLE_EQ(restored->price(spot, 100., .4, .2, .03),
                     leaf.price(spot, 100., .4, .2, .03));
    const std::array<double, 4> query{std::log(spot / 100.), .4, .2, .03};
    EXPECT_DOUBLE_EQ(restored->interpolant().partial(0, query),
                     leaf.interpolant().partial(0, query));
    EXPECT_DOUBLE_EQ(restored->interpolant().eval_second_partial(0, query),
                     leaf.interpolant().eval_second_partial(0, query));
  }
  auto nodal_tag = segment;
  nodal_tag.interp_type = "chebyshev";
  EXPECT_FALSE(mango::make_chebyshev_modal<4>(nodal_tag));
}

TEST(ChebyshevModalPersistenceTest,
     MalformedDomainShapeAndContradictoryStorageAreRefused) {
  std::vector<mango::PriceTableData::Segment> segments;
  mango::extract_segments(fixture(), segments, 100., 0., 1., .05, 1.);
  const auto good = segments.front();
  auto bad = good;
  bad.ndim = 3;
  EXPECT_FALSE(mango::make_chebyshev_modal<4>(bad));
  bad = good;
  bad.domain_hi[0] = bad.domain_lo[0];
  EXPECT_FALSE(mango::make_chebyshev_modal<4>(bad));
  bad = good;
  bad.domain_lo.pop_back();
  EXPECT_FALSE(mango::make_chebyshev_modal<4>(bad));
  bad = good;
  bad.num_pts[0] = -1;
  EXPECT_FALSE(mango::make_chebyshev_modal<4>(bad));
  bad = good;
  bad.values.pop_back();
  EXPECT_FALSE(mango::make_chebyshev_modal<4>(bad));
  bad = good;
  bad.values[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(mango::make_chebyshev_modal<4>(bad));
  bad = good;
  bad.grids = {{0., 1.}};
  EXPECT_FALSE(mango::make_chebyshev_modal<4>(bad));
  bad = good;
  bad.K_ref = 0.;
  EXPECT_FALSE((mango::reconstruct_chebyshev_modal_leaf<4, Transform>(bad)));
}
} // namespace
