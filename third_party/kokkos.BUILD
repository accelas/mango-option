# Kokkos core library - minimal OpenMP build
# Note: This is a simplified build for OpenMP only.
# For production, use Kokkos's CMake build and import via rules_foreign_cc.

# Generate minimal config headers
genrule(
    name = "generate_config",
    outs = [
        "KokkosCore_config.h",
        "KokkosCore_Config_SetupBackend.hpp",
        "KokkosCore_Config_FwdBackend.hpp",
        "KokkosCore_Config_DeclareBackend.hpp",
        "tpls/desul/include/desul/atomics/Config.hpp",
    ],
    cmd = """
cat > $(location KokkosCore_config.h) <<'EOF'
#if !defined(KOKKOS_MACROS_HPP) || defined(KOKKOS_CORE_CONFIG_H)
#error "Do not include KokkosCore_config.h directly; include Kokkos_Macros.hpp instead."
#else
#define KOKKOS_CORE_CONFIG_H
#endif

#define KOKKOS_VERSION 40300
#define KOKKOS_VERSION_MAJOR 4
#define KOKKOS_VERSION_MINOR 3
#define KOKKOS_VERSION_PATCH 0

/* Execution Spaces */
#define KOKKOS_ENABLE_SERIAL
#define KOKKOS_ENABLE_OPENMP

/* General Settings */
#define KOKKOS_ENABLE_CXX23
#define KOKKOS_ENABLE_DEPRECATED_CODE_4
#define KOKKOS_ENABLE_DEPRECATION_WARNINGS
#define KOKKOS_ENABLE_IMPL_MDSPAN

/* Architecture - generic x86_64 */
#define KOKKOS_ARCH_AVX2

EOF

cat > $(location KokkosCore_Config_SetupBackend.hpp) <<'EOF'
#ifndef KOKKOS_CORE_CONFIG_SETUPBACKEND_HPP
#define KOKKOS_CORE_CONFIG_SETUPBACKEND_HPP
// Empty for OpenMP/Serial - no special setup needed
#endif
EOF

cat > $(location KokkosCore_Config_FwdBackend.hpp) <<'EOF'
#ifndef KOKKOS_CORE_CONFIG_FWDBACKEND_HPP
#define KOKKOS_CORE_CONFIG_FWDBACKEND_HPP
// Forward declarations for enabled backends
namespace Kokkos {
class OpenMP;
class Serial;
}
#endif
EOF

cat > $(location KokkosCore_Config_DeclareBackend.hpp) <<'EOF'
#ifndef KOKKOS_CORE_CONFIG_DECLAREBACKEND_HPP
#define KOKKOS_CORE_CONFIG_DECLAREBACKEND_HPP
// Include enabled backend headers with their parallel implementations
#include "decl/Kokkos_Declare_OPENMP.hpp"
#include "decl/Kokkos_Declare_SERIAL.hpp"
#endif
EOF

mkdir -p $$(dirname $(location tpls/desul/include/desul/atomics/Config.hpp))
cat > $(location tpls/desul/include/desul/atomics/Config.hpp) <<'EOF'
#ifndef DESUL_ATOMICS_CONFIG_HPP_
#define DESUL_ATOMICS_CONFIG_HPP_

#define DESUL_ATOMICS_ENABLE_OPENMP

#endif
EOF
""",
)

cc_library(
    name = "kokkos",
    hdrs = glob([
        "core/src/**/*.hpp",
        "core/src/**/*.h",
        "containers/src/**/*.hpp",
        "algorithms/src/**/*.hpp",
        "simd/src/**/*.hpp",
        "tpls/mdspan/include/**/*.hpp",
        "tpls/desul/include/**/*.hpp",
        "tpls/desul/include/**/*.inc",
        "tpls/desul/include/**/*.inc_*",
    ]) + [":generate_config"],
    srcs = glob([
        "core/src/impl/*.cpp",
        "core/src/OpenMP/*.cpp",
        "core/src/Serial/*.cpp",
    ], exclude = [
        "core/src/impl/Kokkos_Spinwait.cpp",
    ]),
    includes = [
        "core/src",
        "containers/src",
        "algorithms/src",
        "simd/src",
        "tpls/mdspan/include",
        "tpls/desul/include",
        ".",  # For generated config
    ],
    defines = [
        "KOKKOS_ENABLE_OPENMP",
        "KOKKOS_ENABLE_SERIAL",
    ],
    copts = ["-fopenmp", "-std=c++23"],
    linkopts = ["-fopenmp"],
    visibility = ["//visibility:public"],
)
