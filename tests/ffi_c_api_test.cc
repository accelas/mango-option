// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/ffi/mango_c_api.h"
#include <cmath>
#include <vector>

namespace {

MangoPricingParams make_put_params() {
  MangoPricingParams p{};
  p.spot = 100.0; p.strike = 100.0; p.maturity = 1.0;
  p.dividend_yield = 0.0; p.volatility = 0.25; p.rate_const = 0.05;
  p.tenor_points = nullptr; p.n_tenor_points = 0;
  p.dividends = nullptr; p.n_dividends = 0;
  p.option_type = MANGO_PUT;
  return p;
}

TEST(MangoCApi, PriceAmericanPutAndGreeks) {
  MangoPricingParams p = make_put_params();
  MangoAmericanResult* r = nullptr;
  MangoError err{};
  ASSERT_EQ(mango_price_american(&p, &r, &err), MANGO_OK) << err.message;
  ASSERT_NE(r, nullptr);

  double v = mango_american_value(r);
  EXPECT_GT(v, 0.0);
  EXPECT_TRUE(std::isfinite(mango_american_delta(r)));
  EXPECT_GT(mango_american_gamma(r), 0.0);
  EXPECT_TRUE(std::isfinite(mango_american_theta(r)));

  double off = 0.0;
  ASSERT_EQ(mango_american_value_at(r, 90.0, &off, &err), MANGO_OK) << err.message;
  EXPECT_GT(off, v);  // deeper ITM put worth more

  mango_american_result_free(r);
}

TEST(MangoCApi, SolveIvRoundTrip) {
  MangoPricingParams p = make_put_params();
  MangoAmericanResult* r = nullptr;
  MangoError err{};
  ASSERT_EQ(mango_price_american(&p, &r, &err), MANGO_OK);
  double market = mango_american_value(r);
  mango_american_result_free(r);

  MangoIvQuery q{};
  q.spot = 100.0; q.strike = 100.0; q.maturity = 1.0;
  q.dividend_yield = 0.0; q.market_price = market; q.rate_const = 0.05;
  q.tenor_points = nullptr; q.n_tenor_points = 0;
  q.dividends = nullptr; q.n_dividends = 0;
  q.option_type = MANGO_PUT;

  MangoIvSuccess out{};
  ASSERT_EQ(mango_solve_iv(&q, nullptr, &out, &err), MANGO_OK) << err.message;
  EXPECT_NEAR(out.implied_vol, 0.25, 0.01);
}

TEST(MangoCApi, NegativeSpotIsValidationError) {
  MangoPricingParams p = make_put_params();
  p.spot = -1.0;
  MangoAmericanResult* r = nullptr;
  MangoError err{};
  EXPECT_EQ(mango_price_american(&p, &r, &err), MANGO_ERR_VALIDATION);
  EXPECT_EQ(r, nullptr);
}

TEST(MangoCApi, DiscreteDividendRoundTripIv) {
  std::vector<MangoDividend> divs = {{0.5, 2.0}};
  MangoPricingParams p = make_put_params();
  p.dividends = divs.data(); p.n_dividends = divs.size();
  MangoAmericanResult* r = nullptr;
  MangoError err{};
  ASSERT_EQ(mango_price_american(&p, &r, &err), MANGO_OK);
  double market = mango_american_value(r);
  mango_american_result_free(r);

  MangoIvQuery q{};
  q.spot = 100.0; q.strike = 100.0; q.maturity = 1.0;
  q.dividend_yield = 0.0; q.market_price = market; q.rate_const = 0.05;
  q.dividends = divs.data(); q.n_dividends = divs.size();
  q.option_type = MANGO_PUT;

  MangoIvSuccess out{};
  ASSERT_EQ(mango_solve_iv(&q, nullptr, &out, &err), MANGO_OK) << err.message;
  EXPECT_NEAR(out.implied_vol, 0.25, 0.01);
}

TEST(MangoCApi, InvalidOptionTypeIsValidationError) {
  MangoPricingParams p = make_put_params();
  p.option_type = 7;  // invalid: only 0 (CALL) and 1 (PUT) are valid
  MangoAmericanResult* r = nullptr;
  MangoError err{};
  EXPECT_EQ(mango_price_american(&p, &r, &err), MANGO_ERR_VALIDATION);
  EXPECT_EQ(r, nullptr);
  EXPECT_STREQ(err.message, "invalid option_type");
}

TEST(MangoCApi, NonFiniteIvRateIsValidationError) {
  MangoIvQuery q{};
  q.spot = 100.0; q.strike = 100.0; q.maturity = 1.0;
  q.dividend_yield = 0.0; q.market_price = 5.0;
  q.rate_const = std::nan("");  // non-finite scalar rate
  q.option_type = MANGO_PUT;
  MangoIvSuccess out{};
  MangoError err{};
  // Must be Validation, not Arbitrage (the C++ IV path would otherwise map
  // InvalidRate -> ArbitrageViolation).
  EXPECT_EQ(mango_solve_iv(&q, nullptr, &out, &err), MANGO_ERR_VALIDATION);
}

}  // namespace
