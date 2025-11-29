"""Clang + libc++ toolchain configuration for FuzzTest.

This toolchain uses Clang with libc++ instead of libstdc++, which avoids
the std::span ODR violations that occur with Clang + libstdc++ in C++23.

This toolchain is intended for use in an Earthly container where libc++
is available. It is NOT compatible with the Python bindings (pybind11).

Usage:
    earthly +fuzz-test
    # Or manually in a libc++ environment:
    bazel test --config=fuzz //tests:batch_solver_fuzz_test
"""

load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
     "feature",
     "flag_group",
     "flag_set",
     "tool_path",
     "with_feature_set")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

def _impl(ctx):
    """Implementation of fuzz cc_toolchain_config rule."""

    # Tool paths (use system Clang 19)
    tool_paths = [
        tool_path(name = "gcc", path = "/usr/bin/clang-19"),
        tool_path(name = "g++", path = "/usr/bin/clang++-19"),
        tool_path(name = "cpp", path = "/usr/bin/clang-cpp-19"),
        tool_path(name = "ar", path = "/usr/bin/llvm-ar-19"),
        tool_path(name = "nm", path = "/usr/bin/llvm-nm-19"),
        tool_path(name = "ld", path = "/usr/bin/ld.lld-19"),
        tool_path(name = "as", path = "/usr/bin/clang-19"),
        tool_path(name = "objcopy", path = "/usr/bin/llvm-objcopy-19"),
        tool_path(name = "objdump", path = "/usr/bin/llvm-objdump-19"),
        tool_path(name = "gcov", path = "/usr/bin/llvm-cov-19"),
        tool_path(name = "strip", path = "/usr/bin/llvm-strip-19"),
        tool_path(name = "llvm-cov", path = "/usr/bin/llvm-cov-19"),
    ]

    # C++23 standard feature with libc++
    cxx23_feature = feature(
        name = "cxx23",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-std=c++23",
                            "-stdlib=libc++",
                            "-DMDSPAN_USE_BRACKET_OPERATOR=1",
                        ],
                    ),
                ],
            ),
        ],
    )

    # Enable -fPIC whenever Bazel requests position independent code
    pic_feature = feature(
        name = "pic",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                ],
                flag_groups = [
                    flag_group(flags = ["-fPIC"]),
                ],
            ),
        ],
    )

    supports_pic_feature = feature(
        name = "supports_pic",
        enabled = True,
    )

    dbg_feature = feature(name = "dbg")
    opt_feature = feature(name = "opt")

    compile_mode_feature = feature(
        name = "compile_mode_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(flags = ["-O0", "-g"]),
                ],
                with_features = [with_feature_set(features = ["dbg"])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(flags = ["-O3", "-DNDEBUG"]),
                ],
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
    )

    # C11 standard for C files
    c11_feature = feature(
        name = "c11",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile],
                flag_groups = [
                    flag_group(
                        flags = ["-std=c11"],
                    ),
                ],
            ),
        ],
    )

    # Warning flags (relaxed for external deps)
    warnings_feature = feature(
        name = "warnings",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wall",
                            "-Wextra",
                            "-Wno-unused-parameter",
                            # Suppress warnings in external deps (FuzzTest, protobuf)
                            "-Wno-pass-failed",
                            "-Wno-sign-compare",
                            "-Wno-unused-function",
                            "-Wno-missing-field-initializers",
                            "-Wno-deprecated-declarations",
                        ],
                    ),
                ],
            ),
        ],
    )

    # Treat warnings as errors (for our code)
    werror_feature = feature(
        name = "werror",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-Werror"],
                    ),
                ],
            ),
        ],
    )

    # Baseline x86-64 ISA for portable binaries
    baseline_isa_feature = feature(
        name = "baseline_isa",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-march=x86-64",
                            "-mtune=generic",
                        ],
                    ),
                ],
            ),
        ],
    )

    # Default linker flags with libc++
    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-stdlib=libc++",
                            "-lc++",
                            "-lc++abi",
                            "-lm",
                            "-fuse-ld=lld-19",
                        ],
                    ),
                ],
            ),
        ],
    )

    preprocessor_defines_feature = feature(
        name = "preprocessor_defines",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-D%{preprocessor_defines}"],
                        iterate_over = "preprocessor_defines",
                    ),
                ],
            ),
        ],
    )

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "mango-option-fuzz-toolchain",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "k8",
        target_libc = "local",
        compiler = "clang",
        abi_version = "local",
        abi_libc_version = "local",
        tool_paths = tool_paths,
        features = [
            cxx23_feature,
            c11_feature,
            pic_feature,
            supports_pic_feature,
            dbg_feature,
            opt_feature,
            compile_mode_feature,
            warnings_feature,
            werror_feature,
            baseline_isa_feature,
            default_link_flags_feature,
            preprocessor_defines_feature,
        ],
        cxx_builtin_include_directories = [
            "/usr/lib/llvm-19/lib/clang/19/include",
            "/usr/lib/llvm-19/include/c++/v1",
            "/usr/include/c++/v1",
            "/usr/local/include",
            "/usr/include/x86_64-linux-gnu",
            "/usr/include",
        ],
    )

fuzz_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
