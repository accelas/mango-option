# MPFR proof dependency

The private proof module uses BCR MPFR `4.2.2.bcr.1` and GMP `6.3.0`, built from
source. No system MPFR development package is required. MPFR supplies explicit
directed rounding for basic arithmetic and exp/log/sqrt; endpoints remain at
proof precision until diagnostic export. See the [MPFR manual](https://www.mpfr.org/mpfr-current/mpfr.html)
and the [Bazel registry overlay](https://github.com/bazelbuild/bazel-central-registry/tree/main/modules/mpfr/4.2.2.bcr.1).

Three local compatibility patches are pinned with their source versions:

- `gmp_bazel7_module.patch` and `mpfr_bazel7_module.patch` give `package_info`
  the exact registry module declaration under `registry_package.bazel`. These
  BCR overlays otherwise refer to a missing `MODULE.bazel` in Bazel 7 source
  repositories. A separate filename also avoids conflicting with versions of
  Bazel that materialize the registry's module declaration themselves.
- The GMP patch additionally sets its root archive to `alwayslink`. Its MPN
  archive references root allocation/assertion objects; the repository's
  linker configuration otherwise leaves that archive cycle unresolved.
- `autoconf_system_extensions.patch` enables `_GNU_SOURCE` for C configure
  probes. The project's toolchain chooses strict C11, while the generated
  M4/GMP configuration enables system extensions for actual source builds.
  Without alignment, probes incorrectly report missing `blksize_t` and
  `struct random_data`, and generated gnulib headers redefine existing types.

The probes' patch changes the new dependency's configure environment; it does
not change project-wide compiler flags or force a system feature-test result.
Reassess these patches when upgrading the pinned dependencies/toolchain.
The resolved graph upgrades `platforms` from the project's former 0.0.11 to
1.0.0 and `rules_cc` transitively; validate bindings and the full module graph
when integrating this branch, rather than relying on a standalone MPFR build.

MPFR is LGPL-3.0-or-later; GMP offers its upstream LGPL/GPL licensing choices.
Retain their distributed licensing files and satisfy the applicable source
and relinking requirements when shipping binaries that link the proof module.
The new kernel is not yet linked into the public pricing or binding libraries.
