// tests/bspline_collocation_workspace_test.cc
#include "src/math/bspline_collocation_workspace.hpp"
#include "src/support/thread_workspace.hpp"
#include <gtest/gtest.h>
#include <cstdint>

using namespace mango;

TEST(BSplineCollocationWorkspaceTest, RequiredBytesCalculation) {
    // For n=100, bandwidth=4:
    // band_storage: 10*100*8 = 8000 bytes + padding
    // lapack_storage: 10*100*8 = 8000 bytes + padding
    // pivots: 100*4 = 400 bytes + padding
    // coeffs: 100*8 = 800 bytes + padding
    size_t bytes = BSplineCollocationWorkspace<double>::required_bytes(100);

    // Should be at least sum of minimums
    EXPECT_GE(bytes, 8000u + 8000u + 400u + 800u);
    // Should be 64-byte aligned
    EXPECT_EQ(bytes % 64, 0u);
}

TEST(BSplineCollocationWorkspaceTest, FromBytesSuccess) {
    const size_t n = 50;
    ThreadWorkspaceBuffer buffer(BSplineCollocationWorkspace<double>::required_bytes(n));

    auto result = BSplineCollocationWorkspace<double>::from_bytes(buffer.bytes(), n);

    ASSERT_TRUE(result.has_value()) << "from_bytes failed";
    auto& ws = result.value();

    EXPECT_EQ(ws.size(), n);
    EXPECT_EQ(ws.band_storage().size(), 10 * n);  // LDAB=10
    EXPECT_EQ(ws.lapack_storage().size(), 10 * n);
    EXPECT_EQ(ws.pivots().size(), n);
    EXPECT_EQ(ws.coeffs().size(), n);
}

TEST(BSplineCollocationWorkspaceTest, BufferTooSmall) {
    const size_t n = 50;
    size_t required = BSplineCollocationWorkspace<double>::required_bytes(n);

    // Allocate less than required
    std::vector<std::byte> small_buffer(required / 2);

    auto result = BSplineCollocationWorkspace<double>::from_bytes(
        std::span(small_buffer), n);

    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("Buffer too small"), std::string::npos);
}

TEST(BSplineCollocationWorkspaceTest, SpansAre64ByteAligned) {
    const size_t n = 100;
    ThreadWorkspaceBuffer buffer(BSplineCollocationWorkspace<double>::required_bytes(n));

    auto ws = BSplineCollocationWorkspace<double>::from_bytes(buffer.bytes(), n).value();

    // All spans should start at 64-byte aligned addresses
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ws.band_storage().data()) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ws.lapack_storage().data()) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ws.pivots().data()) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ws.coeffs().data()) % 64, 0u);
}

TEST(BSplineCollocationWorkspaceTest, SpansNonOverlapping) {
    const size_t n = 50;
    ThreadWorkspaceBuffer buffer(BSplineCollocationWorkspace<double>::required_bytes(n));

    auto ws = BSplineCollocationWorkspace<double>::from_bytes(buffer.bytes(), n).value();

    auto* band_end = ws.band_storage().data() + ws.band_storage().size();
    auto* lapack_start = ws.lapack_storage().data();
    auto* lapack_end = ws.lapack_storage().data() + ws.lapack_storage().size();
    auto* pivots_start = ws.pivots().data();
    auto* pivots_end = reinterpret_cast<double*>(
        reinterpret_cast<std::byte*>(ws.pivots().data()) + ws.pivots().size() * sizeof(int));
    auto* coeffs_start = ws.coeffs().data();

    // band_storage < lapack_storage
    EXPECT_LE(reinterpret_cast<std::byte*>(band_end),
              reinterpret_cast<std::byte*>(lapack_start));
    // lapack_storage < pivots
    EXPECT_LE(reinterpret_cast<std::byte*>(lapack_end),
              reinterpret_cast<std::byte*>(pivots_start));
    // pivots < coeffs
    EXPECT_LE(reinterpret_cast<std::byte*>(pivots_end),
              reinterpret_cast<std::byte*>(coeffs_start));
}
