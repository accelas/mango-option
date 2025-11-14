# Conan Setup for Apache Arrow (Optional)

This project uses [Conan](https://conan.io/) to manage the Apache Arrow C++ dependency.

**Note**: Arrow support is **optional** and disabled by default. The project builds without Arrow, but you'll need it to use price table persistence features (save/load via Arrow IPC format).

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

## Initial Setup

1. **Create Conan profile** (first time only):

```bash
conan profile detect
```

2. **Install Arrow dependencies**:

```bash
conan install . --output-folder=conan_deps --build=missing
```

This will:
- Download Arrow 14.0.1 and its dependencies (Boost, zlib, etc.)
- Build them from source for your platform
- Generate Bazel integration files in `conan_deps/`

**Note**: The first build will take 15-30 minutes as it compiles Boost and Arrow from source.

3. **Enable Conan integration in MODULE.bazel**:

Uncomment the Conan extension lines in `MODULE.bazel`:

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

Arrow is **disabled by default in CI** to avoid long build times. The CI configuration is defined in `.bazelrc`:

```bash
# CI builds without Arrow (fast, no Conan required)
bazel build --config=ci //...
bazel test --config=ci //...
```

If you need Arrow in CI (not recommended due to 15-30 min build time):

```bash
# Install Conan
pipx install conan

# Configure profile
conan profile detect

# Install dependencies
conan install . --output-folder=conan_deps --build=missing

# Build with Arrow
bazel build --config=arrow //...
```

**Recommendation**: Keep Arrow disabled in CI and only build it locally or in dedicated release pipelines.
