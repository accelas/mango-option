// SPDX-License-Identifier: MIT
#include "mango/option/table/eep/eep_decomposer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace mango {
namespace {

struct ConstantEEP {
    double european;

    [[nodiscard]] double european_price(
        double, double, double, double, double) const {
        return european;
    }
    [[nodiscard]] double european_vega(
        double, double, double, double, double) const {
        return 0.0;
    }
    [[nodiscard]] double european_delta(
        double, double, double, double, double) const {
        return 0.0;
    }
    [[nodiscard]] double european_gamma(
        double, double, double, double, double) const {
        return 0.0;
    }
    [[nodiscard]] double european_theta(
        double, double, double, double, double) const {
        return 0.0;
    }
    [[nodiscard]] double european_rho(
        double, double, double, double, double) const {
        return 0.0;
    }
};

static_assert(EEPStrategy<ConstantEEP>);

struct VectorAccessor {
    const std::vector<double>& american_prices;
    std::vector<double>& stored_eep;

    [[nodiscard]] size_t size() const { return american_prices.size(); }
    [[nodiscard]] double american_price(size_t i) const {
        return american_prices[i];
    }
    [[nodiscard]] double spot(size_t) const { return 100.0; }
    [[nodiscard]] double strike() const { return 100.0; }
    [[nodiscard]] double tau(size_t) const { return 1.0; }
    [[nodiscard]] double sigma(size_t) const { return 0.2; }
    [[nodiscard]] double rate(size_t) const { return 0.05; }
    void set_value(size_t i, double value) { stored_eep[i] = value; }
};

TEST(EEPFloorTest, NegativeInputsProjectExactlyToZero) {
    constexpr std::array inputs{
        -std::numeric_limits<double>::denorm_min(),
        -0.005,
        -1.0,
        -5.0,
        -1.0e6,
    };

    for (double input : inputs) {
        EXPECT_DOUBLE_EQ(eep_floor(input), 0.0) << "input=" << input;
    }
}

TEST(EEPFloorTest, PositiveInputsAreExactIdentity) {
    const std::array inputs{
        std::numeric_limits<double>::denorm_min(),
        0.005,
        0.02,
        1.0,
        std::nextafter(5.0, 0.0),
        5.0,
        std::nextafter(5.0, std::numeric_limits<double>::infinity()),
        1.0e6,
    };

    for (double input : inputs) {
        EXPECT_DOUBLE_EQ(eep_floor(input), input) << "input=" << input;
    }
}

TEST(EEPFloorTest, BothSignedZerosProducePositiveZero) {
    EXPECT_DOUBLE_EQ(eep_floor(0.0), 0.0);
    EXPECT_FALSE(std::signbit(eep_floor(0.0)));
    EXPECT_DOUBLE_EQ(eep_floor(-0.0), 0.0);
    EXPECT_FALSE(std::signbit(eep_floor(-0.0)));
}

TEST(EEPFloorTest, ProjectionIsIdempotent) {
    constexpr std::array inputs{-10.0, -0.005, 0.0, 0.005, 0.02, 5.0, 10.0};

    for (double input : inputs) {
        EXPECT_DOUBLE_EQ(eep_floor(eep_floor(input)), eep_floor(input))
            << "input=" << input;
    }
}

TEST(EEPFloorTest, ProjectionIsPositivelyHomogeneous) {
    constexpr std::array inputs{-2.0, -0.005, 0.0, 0.005, 2.0, 5.0};
    constexpr std::array scales{0.5, 2.0, 100.0};

    for (double input : inputs) {
        for (double scale : scales) {
            EXPECT_DOUBLE_EQ(
                eep_floor(scale * input),
                scale * eep_floor(input))
                << "input=" << input << " scale=" << scale;
        }
    }
}

// Regression: eep_floor masked NaN as +0.0 at table-build time (issue #466)
// Bug: std::max(0.0, NaN) returns its first argument, hiding NaN from the
// downstream build_from_values finiteness guard
TEST(EEPFloorTest, NaNPropagates) {
    EXPECT_TRUE(std::isnan(eep_floor(std::nan(""))));
}

TEST(EEPDecomposerTest, BulkAndPerPointPathsUseExactProjection) {
    const std::vector american_prices{9.995, 10.0, 10.005, 10.02, 11.0};
    std::vector<double> stored_eep(american_prices.size(), -1.0);
    const ConstantEEP strategy{.european = 10.0};
    VectorAccessor accessor{american_prices, stored_eep};

    eep_decompose(accessor, strategy);

    for (size_t i = 0; i < american_prices.size(); ++i) {
        const double expected =
            std::max(0.0, american_prices[i] - strategy.european);
        EXPECT_DOUBLE_EQ(stored_eep[i], expected) << "index=" << i;
        EXPECT_DOUBLE_EQ(
            compute_eep(
                american_prices[i],
                100.0,
                100.0,
                1.0,
                0.2,
                0.05,
                strategy),
            expected)
            << "index=" << i;
    }
}

}  // namespace
}  // namespace mango
