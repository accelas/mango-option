#include <gtest/gtest.h>
#include "kokkos/src/math/bspline_basis.hpp"
#include <cmath>
#include <vector>

namespace mango::kokkos::test {

class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override { Kokkos::initialize(); }
    void TearDown() override { Kokkos::finalize(); }
};

[[maybe_unused]] static ::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class BSplineBasisTest : public ::testing::Test {};

TEST_F(BSplineBasisTest, FindSpanUniform) {
    // Uniform knots: [0,0,0,0, 1,2,3, 4,4,4,4]
    std::vector<double> t = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};

    EXPECT_EQ(find_span_cubic(t.data(), t.size(), 0.0), 3);
    EXPECT_EQ(find_span_cubic(t.data(), t.size(), 0.5), 3);
    EXPECT_EQ(find_span_cubic(t.data(), t.size(), 1.5), 4);
    EXPECT_EQ(find_span_cubic(t.data(), t.size(), 2.5), 5);
    EXPECT_EQ(find_span_cubic(t.data(), t.size(), 3.5), 6);
    EXPECT_EQ(find_span_cubic(t.data(), t.size(), 4.0), 6);
}

TEST_F(BSplineBasisTest, BasisSumToOne) {
    // Partition of unity: basis functions sum to 1
    std::vector<double> t = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};
    double N[4];

    for (double x = 0.1; x < 3.9; x += 0.3) {
        int span = find_span_cubic(t.data(), t.size(), x);
        cubic_basis_nonuniform(t.data(), t.size(), span, x, N);

        double sum = N[0] + N[1] + N[2] + N[3];
        EXPECT_NEAR(sum, 1.0, 1e-10) << "at x=" << x;
    }
}

TEST_F(BSplineBasisTest, BasisNonNegative) {
    std::vector<double> t = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};
    double N[4];

    for (double x = 0.0; x <= 4.0; x += 0.2) {
        int span = find_span_cubic(t.data(), t.size(), x);
        cubic_basis_nonuniform(t.data(), t.size(), span, x, N);

        for (int i = 0; i < 4; ++i) {
            EXPECT_GE(N[i], -1e-14) << "N[" << i << "] at x=" << x;
        }
    }
}

TEST_F(BSplineBasisTest, DerivativeSumToZero) {
    // Derivative of partition of unity is zero
    std::vector<double> t = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};
    double dN[4];

    for (double x = 0.5; x < 3.5; x += 0.3) {
        int span = find_span_cubic(t.data(), t.size(), x);
        cubic_basis_derivative(t.data(), t.size(), span, x, dN);

        double sum = dN[0] + dN[1] + dN[2] + dN[3];
        EXPECT_NEAR(sum, 0.0, 1e-10) << "at x=" << x;
    }
}

TEST_F(BSplineBasisTest, CreateClampedKnots) {
    // Test knot vector creation
    Kokkos::View<double*, Kokkos::HostSpace> x("x", 7);
    for (int i = 0; i < 7; ++i) {
        x(i) = static_cast<double>(i);  // [0,1,2,3,4,5,6]
    }

    Kokkos::View<double*, Kokkos::HostSpace> knots("knots", 11);  // n + 4 = 11
    create_clamped_knots_cubic(x, knots);

    // First 4 knots should equal x[0]
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(knots(i), 0.0);
    }

    // Last 4 knots should equal x[n-1]
    for (int i = 7; i < 11; ++i) {
        EXPECT_DOUBLE_EQ(knots(i), 6.0);
    }

    // Interior knots should be strictly increasing
    for (int i = 1; i < 11; ++i) {
        EXPECT_GE(knots(i), knots(i - 1));
    }
}

TEST_F(BSplineBasisTest, DeviceExecution) {
    // Test that basis functions work in Kokkos parallel_for
    Kokkos::View<double*> t("knots", 11);
    auto t_h = Kokkos::create_mirror_view(t);
    std::vector<double> t_std = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};
    for (int i = 0; i < 11; ++i) t_h(i) = t_std[i];
    Kokkos::deep_copy(t, t_h);

    Kokkos::View<double*> results("results", 10);

    Kokkos::parallel_for("test_basis", 10, KOKKOS_LAMBDA(int idx) {
        double x = 0.4 * idx;
        int span = find_span_cubic(t.data(), 11, x);
        double N[4];
        cubic_basis_nonuniform(t.data(), 11, span, x, N);
        results(idx) = N[0] + N[1] + N[2] + N[3];
    });
    Kokkos::fence();

    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(results_h(i), 1.0, 1e-10) << "at idx=" << i;
    }
}

}  // namespace mango::kokkos::test
