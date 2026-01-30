// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/simple/timestamp.hpp"
#include <chrono>

using namespace mango::simple;

TEST(SimpleTimestampTest, ConstructFromISODate) {
    Timestamp ts{"2024-06-21"};
    auto tp = ts.to_timepoint();
    ASSERT_TRUE(tp.has_value());
    // Should be midnight UTC on that date
}

TEST(SimpleTimestampTest, ConstructFromCompactDate) {
    Timestamp ts{"20240621", TimestampFormat::Compact};
    auto tp = ts.to_timepoint();
    ASSERT_TRUE(tp.has_value());
}

TEST(SimpleTimestampTest, ConstructFromNanoseconds) {
    // 2024-06-21 00:00:00 UTC in nanoseconds since epoch
    uint64_t nanos = 1718928000000000000ULL;
    Timestamp ts{nanos};
    auto tp = ts.to_timepoint();
    ASSERT_TRUE(tp.has_value());
}

TEST(SimpleTimestampTest, ConstructFromISO8601WithTime) {
    Timestamp ts{"2024-06-21T10:30:00"};
    auto tp = ts.to_timepoint();
    ASSERT_TRUE(tp.has_value());
}

TEST(SimpleTimestampTest, ComputeTauToExpiry) {
    Timestamp now{"2024-06-21T10:30:00"};
    Timestamp expiry{"2024-06-21T16:00:00"};  // PM settlement

    double tau = compute_tau(now, expiry);
    // 5.5 hours remaining / (365 * 24 hours) ≈ 0.000628
    EXPECT_NEAR(tau, 5.5 / (365.0 * 24.0), 1e-6);
}

TEST(SimpleTimestampTest, NowReturnsCurrentTime) {
    auto ts = Timestamp::now();
    auto tp = ts.to_timepoint();
    ASSERT_TRUE(tp.has_value());
}
