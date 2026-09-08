// SPDX-License-Identifier: MIT
// Bit-identity goldens for B-spline collocation fitting (issue #435).
// Pinned for #458's stable generated-knot policy. Factorization-only
// refactors must reproduce every coefficient bit-for-bit; an intentional
// spline-space change requires independently tested accuracy and new goldens.
//
// The goldens are toolchain-pinned: they depend on the LAPACK/BLAS
// implementation and this target's optimization flags. A failure after a
// BLAS upgrade or copt change is environmental, not a fit regression —
// re-generate by emptying the golden arrays and pasting the printed
// literals (see expect_bits).
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "mango/math/bspline/bspline_collocation.hpp"
#include "mango/math/bspline/bspline_nd_separable.hpp"

namespace {

// Compares every coefficient's bit pattern against golden. When golden is
// empty (generation mode), prints the literal list to paste below and fails.
void expect_bits(const std::vector<double>& coeffs,
                 const std::vector<std::uint64_t>& golden) {
    if (golden.size() != coeffs.size()) {
        for (double c : coeffs) {
            std::printf("0x%016llxULL,\n",
                        static_cast<unsigned long long>(std::bit_cast<std::uint64_t>(c)));
        }
        FAIL() << "golden size " << golden.size() << " != coeffs " << coeffs.size()
               << " — paste the printed literals into the golden array";
    }
    for (size_t i = 0; i < coeffs.size(); ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(coeffs[i]), golden[i]) << "index " << i;
    }
}

}  // namespace

TEST(BSplineBitIdentity, Fit1DGolden) {
    std::vector<double> grid{-2.0, -1.3, -0.55, 0.0, 0.4, 1.1, 1.9, 2.6};
    std::vector<double> values(grid.size());
    for (size_t i = 0; i < grid.size(); ++i) {
        values[i] = std::sin(1.3 * grid[i]) + 0.25 * grid[i] * grid[i];
    }
    auto solver = mango::BSplineCollocation1D<double>::create(grid).value();
    auto result = solver.fit(values);
    ASSERT_TRUE(result.has_value());
    const std::vector<std::uint64_t> golden{
        0x3fdf020688bef48aULL,
        0xbfd2c80210122190ULL,
        0xbff187680051b851ULL,
        0xbfd0c04f3ae078d6ULL,
        0x3fec1e677161933bULL,
        0x3ffcb2a792f1a35dULL,
        0x3ff6aaefed5eeb22ULL,
        0x3ff742f2a3712080ULL,
    };
    expect_bits(result->coefficients, golden);
}

TEST(BSplineBitIdentity, FitSeparable3DGolden) {
    std::array<std::vector<double>, 3> grids{{
        {0.0, 0.7, 1.5, 2.6, 4.0},
        {-1.0, -0.5, -0.1, 0.3, 0.8, 1.6},
        {0.0, 0.2, 0.45, 0.7, 1.0, 1.35, 1.8},
    }};
    std::vector<double> values;
    values.reserve(5 * 6 * 7);
    for (double x : grids[0])
        for (double y : grids[1])
            for (double z : grids[2])
                values.push_back(std::exp(-0.3 * x) * std::sin(y) + 0.5 * z
                                 + 0.05 * x * y * z);
    auto fitter = mango::BSplineNDSeparable<double, 3>::create(grids).value();
    auto result = fitter.fit(values);
    ASSERT_TRUE(result.has_value());
    const std::vector<std::uint64_t> golden{
        0xbfeaed548f090ceeULL,
        0xbfe93198d34d512fULL,
        0xbfe575dd17919578ULL,
        0xbfde6331a69aa265ULL,
        0xbfcd4eebd5bdcd4fULL,
        0xbfa75dd179195770ULL,
        0x3fadf783dc3bfdf0ULL,
        0xbfe6f26009253270ULL,
        0xbfe536a44d6976b6ULL,
        0xbfe17ae891adbaf9ULL,
        0xbfd66d489ad2ed6fULL,
        0xbfbac6337c5cc697ULL,
        0x3fb428bb7292283bULL,
        0x3fc769b30e9e6957ULL,
        0xbfd1baafbbe7585aULL,
        0xbfcc867088dfc1bbULL,
        0xbfbb2f0333e1a5c5ULL,
        0x3fb6f31eee407c98ULL,
        0x3fd57883774bdac9ULL,
        0x3fe09a1f9983cb4aULL,
        0x3fe3ef74eed920a8ULL,
        0x3fe51be37714af32ULL,
        0x3fe6d79f32d06aeaULL,
        0x3fea935aee8c26afULL,
        0x3ff06bcf99683575ULL,
        0x3ff45abe8857246bULL,
        0x3ff749ad77461357ULL,
        0x3ff8f45821f0bdfaULL,
        0x3ff05353d7da96f7ULL,
        0x3ff13131b5b874d6ULL,
        0x3ff30f0f939652abULL,
        0x3ff63131b5b874e2ULL,
        0x3ffa2020a4a763b2ULL,
        0x3ffd0f0f939652c2ULL,
        0x3ffeb9ba3e40fd63ULL,
        0x3feffc81c7e042c5ULL,
        0x3ff0dc1ec1cdff41ULL,
        0x3ff2b9fc9fabdd1bULL,
        0x3ff5dc1ec1cdff44ULL,
        0x3ff9cb0db0bcee2aULL,
        0x3ffcb9fc9fabdd24ULL,
        0x3ffe64a74a5687c9ULL,
        0xbfe6e4a1cd9f75c7ULL,
        0xbfe53f15db46b6a4ULL,
        0xbfe1b32382879114ULL,
        0xbfd77e2bb68d6d49ULL,
        0xbfc117a5eecfc28dULL,
        0x3fa4cdc7d786ee72ULL,
        0x3fc1de1ca08c6658ULL,
        0xbfe38253ea9c8895ULL,
        0xbfe1d79aafacb0a0ULL,
        0xbfdc8103c3edbefaULL,
        0xbfd07379a39da59bULL,
        0xbf952236543fd436ULL,
        0x3fc3ece767137071ULL,
        0x3fd05fb24772f707ULL,
        0xbfce257b05c3b77eULL,
        0xbfc74cbbe037c512ULL,
        0xbfb11b2ca88ad913ULL,
        0x3fc033441fc83af9ULL,
        0x3fd7a3050c7a4d18ULL,
        0x3fe19c9a6aeea505ULL,
        0x3fe4e74515994f8cULL,
        0x3fe1f2534b6e5f2dULL,
        0x3fe3bbbd90272591ULL,
        0x3fe794f137dc8246ULL,
        0x3fee0aac7f161487ULL,
        0x3ff313513df342b9ULL,
        0x3ff619678add2ff9ULL,
        0x3ff7d13a0805028eULL,
        0x3febc26f4675ebbaULL,
        0x3fed99e6bded6348ULL,
        0x3ff0c8af1ab26d43ULL,
        0x3ff41cf35ef6b1a0ULL,
        0x3ff84ad13cd48f73ULL,
        0x3ffb68af1ab26d74ULL,
        0x3ffd2e047007c2a4ULL,
        0x3feb31c6d5377791ULL,
        0x3fed11023991943dULL,
        0x3ff08c99b03d8685ULL,
        0x3ff3eee7832f308aULL,
        0x3ff82e6470c098ecULL,
        0x3ffb59667d0a536bULL,
        0x3ffd263349d72023ULL,
        0xbfdcfff12b8cb57eULL,
        0xbfda2b2d78eb24fbULL,
        0xbfd412258205503aULL,
        0xbfc3abb0472b9f4cULL,
        0x3fa81bc5254f3beeULL,
        0x3fc9310a514ab4eeULL,
        0x3fd20a4c4517218cULL,
        0xbfd8b68d9cd2d5cdULL,
        0xbfd5bbd32b328f5dULL,
        0xbfcea20d0a77a109ULL,
        0xbfb22de09608d11bULL,
        0x3fc1f44994f928ffULL,
        0x3fd30faf8624e63cULL,
        0x3fd8ca78880a5944ULL,
        0xbfc3180ce14620d8ULL,
        0xbfb8f4efa9844b3aULL,
        0x3f8c53f59e3f88fbULL,
        0x3fc9b032d5e88505ULL,
        0x3fdbdb3a69d0fdcbULL,
        0x3fe386a52bce52feULL,
        0x3fe6b4de0f5c8c4eULL,
        0x3fd6bbd9dcaf040dULL,
        0x3fda97a69610ca24ULL,
        0x3fe1739c12acc3c0ULL,
        0x3fe86d3f61c9d135ULL,
        0x3ff09765833b3a0cULL,
        0x3ff3db3a2017ce64ULL,
        0x3ff5b62179380ab7ULL,
        0x3fe19507a2fba5c6ULL,
        0x3fe3b673b9bd11acULL,
        0x3fe84d3525d3d391ULL,
        0x3ff0008f3233de39ULL,
        0x3ff4d63f8d398ed0ULL,
        0x3ff871453d9493edULL,
        0x3ffa7db704b10622ULL,
        0x3fe1396841a0124fULL,
        0x3fe3774d67a086d5ULL,
        0x3fe84b625703f7e3ULL,
        0x3ff0331e2b47bae4ULL,
        0x3ff5496b15351212ULL,
        0x3ff914a01a70eacbULL,
        0x3ffb3c729798bd62ULL,
        0xbfd4568d2aa7fa45ULL,
        0xbfd1f81daa165791ULL,
        0xbfc9bbe4db91f246ULL,
        0xbfb135cbfdaeb3b0ULL,
        0x3fb9cb5747b8d627ULL,
        0x3fcceec5cf18b854ULL,
        0x3fd3059bcb1a9513ULL,
        0xbfd154ee88fb9b6fULL,
        0xbfcd69d837357d32ULL,
        0xbfc21b08d631c583ULL,
        0x3f7c038d028eeac7ULL,
        0x3fc8b3e3ab5c4d15ULL,
        0x3fd53c703431113aULL,
        0x3fda48c3a412ba83ULL,
        0xbfbac82a33eaa640ULL,
        0xbfac06a899e539acULL,
        0x3faafa7aab824e62ULL,
        0x3fcdd3ab2edc06f2ULL,
        0x3fdd66b498914840ULL,
        0x3fe41a525562d001ULL,
        0x3fe72c1971d496beULL,
        0x3fcfe324d60dc0eaULL,
        0x3fd416575458df45ULL,
        0x3fdd0313237fa238ULL,
        0x3fe5ff14e8c058dcULL,
        0x3fef663635492f8cULL,
        0x3ff334ae0773d179ULL,
        0x3ff532aa3c8c78a3ULL,
        0x3fd8a950b39e6792ULL,
        0x3fdd80121fb52926ULL,
        0x3fe3f61470909fb2ULL,
        0x3fecb55e652fe9f0ULL,
        0x3ff3d8320ac57787ULL,
        0x3ff7f05f8d9da5a9ULL,
        0x3ffa43edc6813379ULL,
        0x3fd828cd4b8fc216ULL,
        0x3fdd61eb1add5ab4ULL,
        0x3fe450ee454c16c2ULL,
        0x3fedc2069e7fbe73ULL,
        0x3ff4ce203589250cULL,
        0x3ff9398833b71c7dULL,
        0x3ffbbc606139f49cULL,
        0xbfd038765e761267ULL,
        0xbfcc47f6fa5cc8a3ULL,
        0xbfc352346b009fb0ULL,
        0xbfa11fdbe9732291ULL,
        0x3fbd32a1676f6472ULL,
        0x3fccadcb94ff6055ULL,
        0x3fd256e5ca7fb029ULL,
        0xbfcba54fa858f5dfULL,
        0xbfc6d6b0d2e67ff9ULL,
        0xbfb8f83f4969c940ULL,
        0x3fa3942ba354eef1ULL,
        0x3fcab62a8a2b7a10ULL,
        0x3fd57dabe961496eULL,
        0x3fda1cf5de009362ULL,
        0xbfb55c1ab7c980eaULL,
        0xbfa1c272e036d90cULL,
        0x3fb2000e3df90e8eULL,
        0x3fcf8f6347f249c3ULL,
        0x3fddf0a766888117ULL,
        0x3fe44001c7bf21ccULL,
        0x3fe74001c7bf21ceULL,
        0x3fc96e7ad3db639aULL,
        0x3fd1099d7135d46dULL,
        0x3fda5893a84759e8ULL,
        0x3fe4fc46301261b0ULL,
        0x3feecae50584d77cULL,
        0x3ff30d9c61894dfaULL,
        0x3ff52185a01d3736ULL,
        0x3fd3ab2832d625b8ULL,
        0x3fd8de5b660958ddULL,
        0x3fe208c74c9e4618ULL,
        0x3feb6f2db304ac73ULL,
        0x3ff39dfd3fe8bca6ULL,
        0x3ff80463a64f2303ULL,
        0x3ffa8463a64f2308ULL,
        0x3fd344a96ba4e93aULL,
        0x3fd8f41b6d026fb5ULL,
        0x3fe29974cbaadc5fULL,
        0x3fece0741ce79e45ULL,
        0x3ff4e398f2768a2cULL,
        0x3ff9b320cc3bd49aULL,
        0x3ffc6edc87f79054ULL,
    };
    expect_bits(result->coefficients, golden);
}
