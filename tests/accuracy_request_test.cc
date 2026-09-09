// SPDX-License-Identifier: MIT
#include "mango/option/table/accuracy/request.hpp"
#include "mango/option/table/accuracy/acceptance.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace mango {
namespace {
TEST(AccuracyRequestTest, OneRequestPreservesTargetsAndPolicyWithoutInventingAdmission) {
    const AccuracyRequest defaults;
    EXPECT_DOUBLE_EQ(defaults.max_price_error, .01);
    EXPECT_EQ(defaults.max_iv_error, 2e-5);
    EXPECT_EQ(defaults.policy, AccuracyPolicy::Strict);
    EXPECT_TRUE(defaults.valid());

    const AccuracyRequest request{.max_price_error = .03,
        .max_iv_error = std::nullopt, .policy = AccuracyPolicy::BestEffort};
    const auto assessment = detail::accuracy::assess_request({}, {}, request);
    EXPECT_DOUBLE_EQ(assessment.request().max_price_error, .03);
    EXPECT_FALSE(assessment.request().max_iv_error);
    EXPECT_EQ(assessment.policy(), AccuracyPolicy::BestEffort);
    EXPECT_EQ(assessment.decision(), detail::accuracy::Decision::ViabilityUnassessed);
    EXPECT_EQ(assessment.prerequisites().certificate, PriceProofStatus::NotRun);
}

TEST(AccuracyRequestTest, InvalidTargetsAndPolicyRemainExplicitFailures) {
    for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        AccuracyRequest request;
        request.max_price_error = value;
        EXPECT_FALSE(request.valid());
        EXPECT_EQ(detail::accuracy::assess_request({}, {}, request).decision(),
                  detail::accuracy::Decision::InvalidTargets);
        request = {};
        request.max_iv_error = value;
        EXPECT_FALSE(request.valid());
        EXPECT_EQ(detail::accuracy::assess_request({}, {}, request).decision(),
                  detail::accuracy::Decision::InvalidTargets);
    }
    AccuracyRequest request;
    request.policy = static_cast<AccuracyPolicy>(255);
    EXPECT_FALSE(request.valid());
    EXPECT_EQ(detail::accuracy::assess_request({}, {}, request).decision(),
              detail::accuracy::Decision::InvalidTargets);
}
} // namespace
} // namespace mango
