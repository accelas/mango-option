// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/chebyshev_modal_interpolant.hpp"
#include "mango/option/table/parquet/parquet_io.hpp"
#include "mango/option/table/serialization/extract_segments.hpp"
#include "mango/option/table/serialization/reconstruct.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include <bit>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>

TEST(ChebyshevModalParquetTest,
     FormatFourPreservesModalBitsAndOriginalModelMetadata) {
  std::vector<double> coefficients(16);
  coefficients[0] = 8.;
  coefficients[3] = -0.;
  coefficients[8] = .5;
  auto modal = mango::ChebyshevModalInterpolant<4>::build_from_coefficients(
      coefficients, {{-.2, .05, .1, -.02}, {.2, 1., .5, .08}}, {2, 2, 2, 2});
  ASSERT_TRUE(modal);
  using Leaf = mango::TransformLeaf<mango::ChebyshevModalInterpolant<4>,
                                    mango::StandardTransform4D>;
  Leaf leaf(std::move(*modal), {}, 100.);
  mango::PriceTableData data;
  // Generic file I/O carries the compositional segment tag. This does not
  // opt an existing financial factory/alias into the modal evaluator.
  data.surface_type = mango::surface_types::kChebyshev4DSegmented;
  data.option_type = mango::OptionType::PUT;
  data.dividend_yield = .02;
  data.strike_bounds = mango::StrikeBounds{100., 100.};
  data.ratio_bounds =
      mango::MoneynessBounds{std::nextafter(.92, 1.), std::nextafter(1.08, 2.)};
  data.fixed_expiry = mango::FixedExpiryMetadata{1., {{.4, 3.}}};
  data.maturity = .9;
  data.bounds_tau_min = .05;
  data.bounds_tau_max = .9;
  data.bounds_m_min = std::log(data.ratio_bounds->min);
  data.bounds_m_max = std::log(data.ratio_bounds->max);
  data.bounds_sigma_min = .1;
  data.bounds_sigma_max = .5;
  data.bounds_rate_min = -.02;
  data.bounds_rate_max = .08;
  mango::extract_segments(leaf, data.segments, 100., 0., 1., .05, .9);
  const auto path =
      std::filesystem::path(::testing::TempDir()) / "modal-format-four.parquet";
  ASSERT_TRUE(mango::write_parquet(data, path));
  auto restored = mango::read_parquet(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(restored);
  ASSERT_EQ(restored->segments.size(), 1);
  const auto &segment = restored->segments.front();
  EXPECT_EQ(segment.interp_type, "chebyshev_modal");
  ASSERT_EQ(segment.values.size(), coefficients.size());
  for (size_t i = 0; i < coefficients.size(); ++i) {
    EXPECT_EQ(std::bit_cast<uint64_t>(segment.values[i]),
              std::bit_cast<uint64_t>(coefficients[i]));
  }
  ASSERT_TRUE(restored->ratio_bounds);
  EXPECT_EQ(restored->ratio_bounds->min, data.ratio_bounds->min);
  EXPECT_EQ(restored->ratio_bounds->max, data.ratio_bounds->max);
  ASSERT_TRUE(restored->strike_bounds);
  EXPECT_DOUBLE_EQ(restored->strike_bounds->min, 100.);
  EXPECT_DOUBLE_EQ(restored->strike_bounds->max, 100.);
  ASSERT_TRUE(restored->fixed_expiry);
  EXPECT_DOUBLE_EQ(restored->fixed_expiry->reference_maturity, 1.);
  ASSERT_EQ(restored->fixed_expiry->discrete_dividends.size(), 1);
  EXPECT_DOUBLE_EQ(restored->fixed_expiry->discrete_dividends[0].calendar_time,
                   .4);
  EXPECT_DOUBLE_EQ(restored->fixed_expiry->discrete_dividends[0].amount, 3.);
  EXPECT_DOUBLE_EQ(restored->dividend_yield, .02);
  EXPECT_DOUBLE_EQ(restored->maturity, .9);
  EXPECT_FALSE(mango::make_chebyshev<4>(segment));
  auto restored_modal = mango::make_chebyshev_modal<4>(segment);
  ASSERT_TRUE(restored_modal);
  const std::array<double, 4> point{.037, .31, .22, .05};
  EXPECT_DOUBLE_EQ(restored_modal->eval(point), leaf.interpolant().eval(point));
  EXPECT_DOUBLE_EQ(restored_modal->partial(0, point),
                   leaf.interpolant().partial(0, point));
}
