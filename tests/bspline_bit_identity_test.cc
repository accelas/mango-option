// SPDX-License-Identifier: MIT
// Bit-identity goldens for B-spline collocation fitting (issue #435).
// Pinned from the pre-refactor implementation; the factor-once refactor
// must reproduce every coefficient bit-for-bit.
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
        0xbfcc9cb8f176c6aeULL,
        0xbff0ffb8e084792aULL,
        0xbfd3e6caf80ab07eULL,
        0x3feea84d040ef82bULL,
        0x3ffc4ac8ecb175c1ULL,
        0x3ff679647416e827ULL,
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
        0xbfe93198d34d5135ULL,
        0xbfe575dd17919569ULL,
        0xbfde6331a69aa250ULL,
        0xbfcd4eebd5bdcd16ULL,
        0xbfa75dd179195783ULL,
        0x3fadf783dc3bfdf0ULL,
        0xbfe692c44f192f36ULL,
        0xbfe4d708935d737dULL,
        0xbfe11b4cd7a1b7b2ULL,
        0xbfd5ae1126bae6daULL,
        0xbfb7c955abfcac74ULL,
        0x3fb7259942f2422dULL,
        0x3fc8e821f6ce765bULL,
        0xbfd1838153a5e401ULL,
        0xbfcc1813b85cd912ULL,
        0xbfba524992dbd3e8ULL,
        0x3fb7cfd88f464e48ULL,
        0x3fd5afb1df8d4f50ULL,
        0x3fe0b5b6cda48576ULL,
        0x3fe40b0c22f9dad3ULL,
        0x3fe47a5ec77c0446ULL,
        0x3fe6361a8337c004ULL,
        0x3fe9f1d63ef37bcdULL,
        0x3ff01b0d419be008ULL,
        0x3ff409fc308acef3ULL,
        0x3ff6f8eb1f79bde1ULL,
        0x3ff8a395ca246885ULL,
        0x3ff065b6983c3026ULL,
        0x3ff14394761a0e08ULL,
        0x3ff3217253f7ebe2ULL,
        0x3ff64394761a0e13ULL,
        0x3ffa32836508fcefULL,
        0x3ffd217253f7ebefULL,
        0x3ffecc1cfea29692ULL,
        0x3feffc81c7e042c5ULL,
        0x3ff0dc1ec1cdff44ULL,
        0x3ff2b9fc9fabdd22ULL,
        0x3ff5dc1ec1cdff4bULL,
        0x3ff9cb0db0bcee31ULL,
        0x3ffcb9fc9fabdd24ULL,
        0x3ffe64a74a5687c9ULL,
        0xbfe6e4a1cd9f75b0ULL,
        0xbfe53f15db46b68bULL,
        0xbfe1b323828790f6ULL,
        0xbfd77e2bb68d6d04ULL,
        0xbfc117a5eecfc21dULL,
        0x3fa4cdc7d786ef99ULL,
        0x3fc1de1ca08c6692ULL,
        0xbfe3310aeda73e96ULL,
        0xbfe185d37b0a88d6ULL,
        0xbfdbdb55a5e82a18ULL,
        0xbfcf9475bea74b1aULL,
        0xbf848d5ecc57b3ddULL,
        0x3fc54f050f355bcfULL,
        0x3fd112a68f307c9fULL,
        0xbfcdc7a6cbfab87fULL,
        0xbfc6ef65de1ba3f0ULL,
        0xbfb062a05913db56ULL,
        0x3fc08dc1f4618432ULL,
        0x3fd7cf258c2121c8ULL,
        0x3fe1b23fde1c1888ULL,
        0x3fe4fcadda513124ULL,
        0x3fe169011b267e98ULL,
        0x3fe3324bd1f40d91ULL,
        0x3fe70b3b8311419cULL,
        0x3fed8084b5824675ULL,
        0x3ff2cdf5be7fe7c3ULL,
        0x3ff5d3d6a516d99cULL,
        0x3ff78b8acb03e31cULL,
        0x3febe1b287a09960ULL,
        0x3fedb88c397ffba6ULL,
        0x3ff0d757efff53dfULL,
        0x3ff42a7f004e36c8ULL,
        0x3ff856f6d8dcd0a2ULL,
        0x3ffb73c9b71bc5ddULL,
        0x3ffd3887584b2e06ULL,
        0x3feb31c6d5377774ULL,
        0x3fed110239919423ULL,
        0x3ff08c99b03d8685ULL,
        0x3ff3eee7832f3082ULL,
        0x3ff82e6470c098f1ULL,
        0x3ffb59667d0a5356ULL,
        0x3ffd263349d7201dULL,
        0xbfdcfff12b8cb568ULL,
        0xbfda2b2d78eb24efULL,
        0xbfd4122582055010ULL,
        0xbfc3abb0472b9f0aULL,
        0x3fa81bc5254f3cecULL,
        0x3fc9310a514ab4f6ULL,
        0x3fd20a4c45172191ULL,
        0xbfd84f95e00526aaULL,
        0xbfd5513dd61bdb2aULL,
        0xbfcdbd4f32c0e702ULL,
        0xbfb0301b60053ac7ULL,
        0x3fc313fdbda5720fULL,
        0x3fd3abc662d4a538ULL,
        0x3fd96d836232d3e4ULL,
        0xbfc2dc9f18e9e917ULL,
        0xbfb881b1b114e097ULL,
        0x3f8faf98ab9596ceULL,
        0x3fc9df63d5eb21c5ULL,
        0x3fdbeeb8b81cfc48ULL,
        0x3fe38edcb9e91f1bULL,
        0x3fe6bc371dc840bdULL,
        0x3fd60de66002cc75ULL,
        0x3fd9e8cbb352513cULL,
        0x3fe11b356e74f25eULL,
        0x3fe81336715d4eb1ULL,
        0x3ff0695a7e97a4ebULL,
        0x3ff3ac6b4eee9f54ULL,
        0x3ff586e36837502fULL,
        0x3fe1a8d4b0515ecbULL,
        0x3fe3c7fe47e527a7ULL,
        0x3fe859e1b5c0fff0ULL,
        0x3ff002cfbba6b9c7ULL,
        0x3ff4d35f5889c6c4ULL,
        0x3ff86a920a48cb76ULL,
        0x3ffa74d7922f830bULL,
        0x3fe1396841a01243ULL,
        0x3fe3774d67a086cfULL,
        0x3fe84b625703f7e3ULL,
        0x3ff0331e2b47baebULL,
        0x3ff5496b15351213ULL,
        0x3ff914a01a70ead5ULL,
        0x3ffb3c729798bd60ULL,
        0xbfd4568d2aa7fa38ULL,
        0xbfd1f81daa165787ULL,
        0xbfc9bbe4db91f210ULL,
        0xbfb135cbfdaeb34cULL,
        0x3fb9cb5747b8d675ULL,
        0x3fcceec5cf18b85eULL,
        0x3fd3059bcb1a9512ULL,
        0xbfd10cb7f3770f4fULL,
        0xbfcccced89bbc780ULL,
        0xbfc163373727f801ULL,
        0x3f8c516412b91baaULL,
        0x3fc9d18d4a99e247ULL,
        0x3fd5e0682f535cd4ULL,
        0x3fdaf8be2679ebbdULL,
        0xbfba74cec62024fbULL,
        0xbfab6c6f40c0d4e3ULL,
        0x3fab79cd13169d15ULL,
        0x3fcde835ba3dc73dULL,
        0x3fdd69e3e57d5942ULL,
        0x3fe419459668681fULL,
        0x3fe7298c61f19254ULL,
        0x3fceef27485b5b73ULL,
        0x3fd39ac8dd3198edULL,
        0x3fdc8427ce2658dfULL,
        0x3fe5bcccba72df79ULL,
        0x3fef20630a994e41ULL,
        0x3ff310723f63a8f9ULL,
        0x3ff50dae4c08018bULL,
        0x3fd8c516bf176ce9ULL,
        0x3fdd9409b9a7cbeeULL,
        0x3fe3f7a8120ce9f3ULL,
        0x3feca8d574880bf9ULL,
        0x3ff3c9121b7b8572ULL,
        0x3ff7daa4a0ba9b28ULL,
        0x3ffa2a720f58a148ULL,
        0x3fd828cd4b8fc207ULL,
        0x3fdd61eb1add5aabULL,
        0x3fe450ee454c16d8ULL,
        0x3fedc2069e7fbe7fULL,
        0x3ff4ce2035892523ULL,
        0x3ff9398833b71c7cULL,
        0x3ffbbc606139f4a2ULL,
        0xbfd038765e761267ULL,
        0xbfcc47f6fa5cc8a6ULL,
        0xbfc352346b009f91ULL,
        0xbfa11fdbe9732216ULL,
        0x3fbd32a1676f64b1ULL,
        0x3fccadcb94ff6055ULL,
        0x3fd256e5ca7fb029ULL,
        0xbfcb321fde62c6d8ULL,
        0xbfc653ba135497aeULL,
        0xbfb7ae5b321d5013ULL,
        0x3fa70c1d6308a79bULL,
        0x3fcbdbc1a38c6286ULL,
        0x3fd62b2a9f8f6d52ULL,
        0x3fdad9a031933839ULL,
        0xbfb5199f8cada669ULL,
        0xbfa14d437f9add1cULL,
        0x3fb229a8483ce28fULL,
        0x3fcf95edb4028780ULL,
        0x3fddeaf9476220a1ULL,
        0x3fe439d452fc3b98ULL,
        0x3fe737eedf4fab79ULL,
        0x3fc8abe2e3598ee0ULL,
        0x3fd0a6589a4172dcULL,
        0x3fd9f10f67d06e10ULL,
        0x3fe4c4f369928099ULL,
        0x3fee8f18946db6d1ULL,
        0x3ff2ee0af665e297ULL,
        0x3ff501017b2383c3ULL,
        0x3fd3c14ec8a73f59ULL,
        0x3fd8eaa5a2591ec3ULL,
        0x3fe2044de2ffcec7ULL,
        0x3feb58e10a101db6ULL,
        0x3ff387a6c0f45621ULL,
        0x3ff7e5b52a63759bULL,
        0x3ffa60f789340d4eULL,
        0x3fd344a96ba4e93aULL,
        0x3fd8f41b6d026fbbULL,
        0x3fe29974cbaadc73ULL,
        0x3fece0741ce79e5bULL,
        0x3ff4e398f2768a39ULL,
        0x3ff9b320cc3bd498ULL,
        0x3ffc6edc87f79054ULL,
    };
    expect_bits(result->coefficients, golden);
}
