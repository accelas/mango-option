#include <gtest/gtest.h>
#include "src/interpolation/bspline_utils.hpp"
#include <vector>
#include <cmath>

namespace mango {
namespace {

// Test fixture for banded solver testing
class BandedSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple test case: fit cubic B-spline to quadratic function
        // y = x^2 on [0, 1] with 11 points
        n_ = 11;
        x_.resize(n_);
        y_.resize(n_);

        for (size_t i = 0; i < n_; ++i) {
            x_[i] = static_cast<double>(i) / (n_ - 1);
            y_[i] = x_[i] * x_[i];  // Quadratic function
        }

        // Create knot vector for cubic B-spline (degree 3)
        // Use clamped_knots_cubic utility function
        knots_ = clamped_knots_cubic(x_);
    }

    size_t n_;
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> knots_;
};

TEST_F(BandedSolverTest, DenseSolverBaseline) {
    // This test uses the CURRENT dense solver as baseline
    // We'll compare banded solver results against this

    // Create fitter (uses current dense implementation)
    // NOTE: Adjust constructor call based on actual BSplineFitter4D API
    // This is a placeholder - actual API may differ

    // For now, test that we can construct the test case
    EXPECT_EQ(x_.size(), n_);
    EXPECT_EQ(y_.size(), n_);
    EXPECT_EQ(knots_.size(), n_ + 4);

    // Verify knot vector structure (clamped at endpoints)
    EXPECT_DOUBLE_EQ(knots_[0], x_.front());
    EXPECT_DOUBLE_EQ(knots_[3], x_.front());  // Multiplicity 4 at start
    EXPECT_DOUBLE_EQ(knots_[knots_.size() - 1], x_.back());
    EXPECT_DOUBLE_EQ(knots_[knots_.size() - 4], x_.back());  // Multiplicity 4 at end

    // Verify monotonicity
    for (size_t i = 1; i < knots_.size(); ++i) {
        EXPECT_GE(knots_[i], knots_[i-1]) << "Knots not monotonic at index " << i;
    }
}

} // namespace
} // namespace mango
