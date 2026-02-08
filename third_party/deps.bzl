# SPDX-License-Identifier: MIT
"""Module extension for non-BCR dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _non_bcr_deps_impl(ctx):
    """Fetch non-BCR dependencies."""
    # mdspan reference implementation (header-only)
    # Provides C++23 std::mdspan polyfill for GCC + libstdc++
    http_archive(
        name = "mdspan",
        urls = ["https://github.com/kokkos/mdspan/archive/refs/tags/mdspan-0.6.0.tar.gz"],
        strip_prefix = "mdspan-mdspan-0.6.0",
        sha256 = "79f94d7f692cbabfbaff6cd0d3434704435c853ee5087b182965fa929a48a592",
        build_file_content = """
cc_library(
    name = "mdspan",
    hdrs = glob([
        "include/**/*.hpp",
        "include/**/*",
    ], exclude = ["include/**/*.txt"]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
""",
    )

    # Eigen 3.4.0 (header-only linear algebra)
    # Used by Chebyshev-Tucker experiment for SVD
    http_archive(
        name = "eigen",
        urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],
        strip_prefix = "eigen-3.4.0",
        sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
        build_file_content = """
cc_library(
    name = "eigen",
    hdrs = glob(["Eigen/**", "unsupported/Eigen/**"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
""",
    )

non_bcr_deps = module_extension(
    implementation = _non_bcr_deps_impl,
)
