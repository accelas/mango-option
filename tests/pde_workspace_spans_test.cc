#include "src/pde/core/pde_workspace_spans.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace mango {
namespace {

TEST(PDEWorkspaceSpansTest, RequiredSize) {
    // For n=100:
    // - 12 arrays @ padded(100) = 12 × 104 = 1248
    // - 3 arrays @ padded(99) = 3 × 104 = 312
    // - tridiag @ padded(200) = 200
    // Total = 1760
    size_t n = 100;
    size_t required = PDEWorkspaceSpans::required_size(n);

    size_t n_padded = PDEWorkspaceSpans::pad_to_simd(n);  // 104
    size_t n_minus_1_padded = PDEWorkspaceSpans::pad_to_simd(n - 1);  // 104
    size_t tridiag_padded = PDEWorkspaceSpans::pad_to_simd(2 * n);  // 200

    size_t expected = 12 * n_padded + 3 * n_minus_1_padded + tridiag_padded;
    EXPECT_EQ(required, expected);
}

TEST(PDEWorkspaceSpansTest, CreateFromBuffer) {
    size_t n = 100;
    size_t required = PDEWorkspaceSpans::required_size(n);

    std::vector<double> buffer(required, 0.0);

    auto workspace_result = PDEWorkspaceSpans::from_buffer(buffer, n);
    ASSERT_TRUE(workspace_result.has_value());

    auto workspace = workspace_result.value();

    // Verify sizes
    EXPECT_EQ(workspace.size(), n);
    EXPECT_EQ(workspace.dx().size(), n - 1);
    EXPECT_EQ(workspace.u_stage().size(), n);
    EXPECT_EQ(workspace.rhs().size(), n);
    EXPECT_EQ(workspace.lu().size(), n);
    EXPECT_EQ(workspace.psi().size(), n);
    EXPECT_EQ(workspace.jacobian_diag().size(), n);
    EXPECT_EQ(workspace.jacobian_upper().size(), n - 1);
    EXPECT_EQ(workspace.jacobian_lower().size(), n - 1);
    EXPECT_EQ(workspace.residual().size(), n);
    EXPECT_EQ(workspace.delta_u().size(), n);
    EXPECT_EQ(workspace.newton_u_old().size(), n);
    EXPECT_EQ(workspace.u_next().size(), n);
    EXPECT_EQ(workspace.tridiag_workspace().size(), 2 * n);
}

TEST(PDEWorkspaceSpansTest, BufferTooSmall) {
    size_t n = 100;
    size_t required = PDEWorkspaceSpans::required_size(n);

    std::vector<double> buffer(required - 1, 0.0);  // One element too small

    auto workspace_result = PDEWorkspaceSpans::from_buffer(buffer, n);
    ASSERT_FALSE(workspace_result.has_value());
    EXPECT_TRUE(workspace_result.error().find("too small") != std::string::npos);
}

TEST(PDEWorkspaceSpansTest, GridSizeTooSmall) {
    std::vector<double> buffer(100, 0.0);

    auto workspace_result = PDEWorkspaceSpans::from_buffer(buffer, 1);
    ASSERT_FALSE(workspace_result.has_value());
    EXPECT_TRUE(workspace_result.error().find("at least 2") != std::string::npos);
}

TEST(PDEWorkspaceSpansTest, FromBufferAndGrid) {
    size_t n = 100;
    size_t required = PDEWorkspaceSpans::required_size(n);

    std::vector<double> buffer(required, 0.0);
    std::vector<double> grid(n);

    // Create uniform grid [0, 1]
    for (size_t i = 0; i < n; ++i) {
        grid[i] = static_cast<double>(i) / (n - 1);
    }

    auto workspace_result = PDEWorkspaceSpans::from_buffer_and_grid(buffer, grid, n);
    ASSERT_TRUE(workspace_result.has_value());

    auto workspace = workspace_result.value();

    // Verify dx was computed
    auto dx = workspace.dx();
    double expected_dx = 1.0 / (n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
        EXPECT_NEAR(dx[i], expected_dx, 1e-14);
    }
}

TEST(PDEWorkspaceSpansTest, GridSizeMismatch) {
    size_t n = 100;
    size_t required = PDEWorkspaceSpans::required_size(n);

    std::vector<double> buffer(required, 0.0);
    std::vector<double> grid(n + 1);  // Wrong size

    auto workspace_result = PDEWorkspaceSpans::from_buffer_and_grid(buffer, grid, n);
    ASSERT_FALSE(workspace_result.has_value());
    EXPECT_TRUE(workspace_result.error().find("mismatch") != std::string::npos);
}

TEST(PDEWorkspaceSpansTest, IndependentArrays) {
    size_t n = 100;
    size_t required = PDEWorkspaceSpans::required_size(n);

    std::vector<double> buffer(required, 0.0);

    auto workspace_result = PDEWorkspaceSpans::from_buffer(buffer, n);
    ASSERT_TRUE(workspace_result.has_value());

    auto workspace = workspace_result.value();

    // Write to different arrays
    auto rhs = workspace.rhs();
    auto lu = workspace.lu();
    auto psi = workspace.psi();

    for (size_t i = 0; i < n; ++i) {
        rhs[i] = static_cast<double>(i);
        lu[i] = static_cast<double>(i * 2);
        psi[i] = static_cast<double>(i * 3);
    }

    // Verify independence
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(rhs[i], static_cast<double>(i));
        EXPECT_DOUBLE_EQ(lu[i], static_cast<double>(i * 2));
        EXPECT_DOUBLE_EQ(psi[i], static_cast<double>(i * 3));
    }
}

TEST(PDEWorkspaceSpansTest, ConstAccessors) {
    size_t n = 100;
    size_t required = PDEWorkspaceSpans::required_size(n);

    std::vector<double> buffer(required, 0.0);

    auto workspace_result = PDEWorkspaceSpans::from_buffer(buffer, n);
    ASSERT_TRUE(workspace_result.has_value());

    auto workspace = workspace_result.value();

    // Write via mutable accessor
    auto rhs_mut = workspace.rhs();
    for (size_t i = 0; i < n; ++i) {
        rhs_mut[i] = static_cast<double>(i);
    }

    // Read via const accessor
    const auto& workspace_const = workspace;
    auto rhs_const = workspace_const.rhs();

    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(rhs_const[i], static_cast<double>(i));
    }
}

}  // namespace
}  // namespace mango
