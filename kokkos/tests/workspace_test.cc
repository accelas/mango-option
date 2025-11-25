#include <gtest/gtest.h>
#include "kokkos/src/pde/core/workspace.hpp"

namespace mango::kokkos::test {

// Global setup/teardown for Kokkos - once per test program
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        Kokkos::initialize();
    }
    void TearDown() override {
        Kokkos::finalize();
    }
};

// Register the global environment
[[maybe_unused]] static ::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class WorkspaceTest : public ::testing::Test {
    // No per-test setup/teardown needed for Kokkos
};

TEST_F(WorkspaceTest, CreationSucceeds) {
    auto ws = PDEWorkspace<HostMemSpace>::create(101);
    ASSERT_TRUE(ws.has_value());
    EXPECT_EQ(ws->n(), 101);
}

TEST_F(WorkspaceTest, BuffersHaveCorrectSize) {
    auto ws = PDEWorkspace<HostMemSpace>::create(101).value();

    EXPECT_EQ(ws.rhs().extent(0), 101);
    EXPECT_EQ(ws.u_stage().extent(0), 101);
    EXPECT_EQ(ws.jacobian_diag().extent(0), 101);
    EXPECT_EQ(ws.jacobian_lower().extent(0), 100);  // n-1
    EXPECT_EQ(ws.jacobian_upper().extent(0), 100);  // n-1
}

TEST_F(WorkspaceTest, BuffersAreWritable) {
    auto ws = PDEWorkspace<HostMemSpace>::create(10).value();
    auto rhs = ws.rhs();

    for (size_t i = 0; i < 10; ++i) {
        rhs(i) = static_cast<double>(i);
    }

    EXPECT_DOUBLE_EQ(rhs(5), 5.0);
}

TEST_F(WorkspaceTest, TooSmallRejected) {
    auto ws = PDEWorkspace<HostMemSpace>::create(1);
    EXPECT_FALSE(ws.has_value());
}

}  // namespace mango::kokkos::test
