# Workspace file for external dependencies not in Bazel Central Registry

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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
