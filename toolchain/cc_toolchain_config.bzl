"""C++ toolchain configuration for mango-iv with C++23 support."""

load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
     "feature",
     "flag_group",
     "flag_set",
     "tool_path")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

def _impl(ctx):
    """Implementation of cc_toolchain_config rule."""

    # Tool paths (use system GCC/G++)
    tool_paths = [
        tool_path(name = "gcc", path = "/usr/bin/gcc"),
        tool_path(name = "g++", path = "/usr/bin/g++"),
        tool_path(name = "cpp", path = "/usr/bin/cpp"),
        tool_path(name = "ar", path = "/usr/bin/ar"),
        tool_path(name = "nm", path = "/usr/bin/nm"),
        tool_path(name = "ld", path = "/usr/bin/ld"),
        tool_path(name = "as", path = "/usr/bin/as"),
        tool_path(name = "objcopy", path = "/usr/bin/objcopy"),
        tool_path(name = "objdump", path = "/usr/bin/objdump"),
        tool_path(name = "gcov", path = "/usr/bin/gcov"),
        tool_path(name = "strip", path = "/usr/bin/strip"),
        tool_path(name = "llvm-cov", path = "/bin/false"),  # Not used
    ]

    # C++23 standard feature
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
                        flags = ["-std=c++23"],
                    ),
                ],
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

    # Warning flags
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
                            "-Wunused-but-set-parameter",
                            "-Wno-free-nonheap-object",
                        ],
                    ),
                ],
            ),
        ],
    )

    # Treat warnings as errors
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

    # Default linker flags
    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-lstdc++",
                            "-lm",
                        ],
                    ),
                ],
            ),
        ],
    )

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "mango-iv-toolchain",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "k8",
        target_libc = "local",
        compiler = "gcc",
        abi_version = "local",
        abi_libc_version = "local",
        tool_paths = tool_paths,
        features = [
            cxx23_feature,
            c11_feature,
            warnings_feature,
            werror_feature,
            baseline_isa_feature,
            default_link_flags_feature,
        ],
        cxx_builtin_include_directories = [
            "/usr/include/c++/14",
            "/usr/include/x86_64-linux-gnu/c++/14",
            "/usr/include/c++/14/backward",
            "/usr/lib/gcc/x86_64-linux-gnu/14/include",
            "/usr/local/include",
            "/usr/include/x86_64-linux-gnu",
            "/usr/include",
            "/usr/lib/linux",
        ],
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
