# SPDX-License-Identifier: MIT
"""Rule to merge all transitive static libraries into a single .a archive."""

def _merge_archive_impl(ctx):
    cc_info = ctx.attr.dep[CcInfo]
    linker_inputs = cc_info.linking_context.linker_inputs.to_list()

    # Collect all static .a files from transitive deps
    archives = []
    for li in linker_inputs:
        for lib in li.libraries:
            if lib.static_library:
                archives.append(lib.static_library)
            elif lib.pic_static_library:
                archives.append(lib.pic_static_library)

    output = ctx.actions.declare_file(ctx.attr.out)

    # Use the C++ toolchain's ar to merge
    cc_toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]
    ar = cc_toolchain.ar_executable

    # Build ar command: extract all .o from each archive, then merge
    # We use a script to handle the temp directory for extraction
    script = ctx.actions.declare_file(ctx.attr.name + "_merge.sh")
    ctx.actions.write(
        output = script,
        content = """\
#!/bin/bash
set -euo pipefail
out="$1"
shift
execroot="$PWD"
tmpdir=$(mktemp -d)
for a in "$@"; do
    (cd "$tmpdir" && ar x "$execroot/$a")
done
ar rcs "$execroot/$out" "$tmpdir"/*.o
rm -rf "$tmpdir"
""",
        is_executable = True,
    )

    ctx.actions.run(
        inputs = archives + [script],
        outputs = [output],
        executable = script,
        arguments = [output.path] + [a.path for a in archives],
        mnemonic = "MergeArchive",
        progress_message = "Merging static archives into %s" % output.short_path,
    )

    return [DefaultInfo(files = depset([output]))]

merge_archive = rule(
    implementation = _merge_archive_impl,
    attrs = {
        "dep": attr.label(providers = [CcInfo]),
        "out": attr.string(mandatory = True),
        "_cc_toolchain": attr.label(
            default = "@bazel_tools//tools/cpp:current_cc_toolchain",
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)
