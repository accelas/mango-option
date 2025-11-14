# Conan Setup for Apache Arrow

This project uses [Conan](https://conan.io/) to manage the Apache Arrow C++ dependency for price table persistence features.

**Note**: Arrow is required to build the project. Use the automated setup script for quickest installation.

## Prerequisites

Install Conan 2.x:

```bash
# Using pipx (recommended)
pipx install conan

# Or using pip in a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install conan
```

## Quick Start (Automated)

Run the helper script to install and enable Arrow:

```bash
./tools/enable_arrow.sh
```

This script will:
1. Install Conan (if needed)
2. Run `conan install` (~15-30 minutes first time)
3. Enable the Conan extension in MODULE.bazel automatically

To disable later:

```bash
./tools/disable_arrow.sh
```

## Manual Setup (Alternative)

If you prefer manual control:

1. **Create Conan profile** (first time only):

```bash
conan profile detect
```

2. **Install Arrow dependencies**:

```bash
conan install . --output-folder=conan_deps --build=missing
```

This will:
- Download Arrow 14.0.1 and its dependencies
- Build them from source for your platform (~15-30 min)
- Generate Bazel integration files in `conan_deps/`

3. **Enable Conan dependencies in MODULE.bazel**:

Uncomment the Conan extension block in `MODULE.bazel` (lines 22-30):

```python
conan_extension = use_extension("//conan_deps:conan_deps_module_extension.bzl", "conan_extension")
use_repo(
    conan_extension,
    "arrow",
    "boost",
    "bzip2",
    "libbacktrace",
    "zlib",
)
```


## Bazel Integration

After running `conan install`, build with Arrow enabled:

```bash
# Build with Arrow support
bazel build --config=arrow //...

# Test with Arrow support
bazel test --config=arrow //...

# Build without Arrow (default)
bazel build //...
```

In BUILD.bazel files, Arrow dependencies are conditionally included via `select()`:

```python
cc_library(
    name = "my_library",
    deps = select({
        "//:enable_arrow": ["@arrow//:arrow"],
        "//conditions:default": [],
    }),
)
```

## Configuration

The `conanfile.txt` specifies:
- Arrow version 14.0.1
- Static linking (shared=False)
- Minimal build (parquet=False, thrift=False)
- Bazel generators (BazelDeps, BazelToolchain)

## Updating Dependencies

To update to a newer Arrow version:

1. Edit `conanfile.txt` and change the version
2. Re-run `conan install . --output-folder=conan_deps --build=missing`
3. Test that everything builds correctly

## Troubleshooting

### Build failures

If Conan build fails, try:

```bash
# Clean the cache and rebuild
rm -rf conan_deps/
rm -rf ~/.conan2/p/
conan install . --output-folder=conan_deps --build=missing
```

### C++ standard mismatch

The project uses C++23. Ensure your Conan profile is configured correctly:

```bash
# Check profile
conan profile show

# Should show:
#   compiler.cppstd=gnu23
```

If not, edit `~/.conan2/profiles/default` and set `compiler.cppstd=gnu23`.

## CI/CD Integration

Arrow is enabled in CI with aggressive caching to avoid slow builds.

**GitHub Actions caching strategy:**

```yaml
- name: Cache Conan packages
  uses: actions/cache@v4
  with:
    path: ~/.conan2/p
    key: conan-packages-${{ runner.os }}-${{ hashFiles('conanfile.txt') }}
    restore-keys: |
      conan-packages-${{ runner.os }}-
```

**First CI run**: ~20-30 minutes (builds Arrow from source)
**Subsequent runs**: ~2-5 minutes (cache hit, no rebuild)

The cache key includes `conanfile.txt` hash, so changing Arrow options will trigger a rebuild.

See `.github/workflows/ci.yml` for the complete CI configuration.
