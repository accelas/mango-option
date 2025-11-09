# Distribution Packaging

This directory contains Bazel build rules for creating distributable packages of the mango-iv library.

## Available Packages

### 1. Headers-Only Package
```bash
bazel build //dist:mango-iv-headers
```
**Contents:** Library headers only
**Size:** Minimal (~100KB)
**Use case:** Users who want to integrate headers into their existing build system

### 2. Development Package
```bash
bazel build //dist:mango-iv-dev
```
**Contents:** Headers, examples, BUILD files, documentation
**Size:** Small (~500KB)
**Use case:** Developers who want to build from source using Bazel

### 3. Full Distribution
```bash
bazel build //dist:mango-iv-dist
```
**Contents:** Everything in dev package
**Size:** Small (~500KB)
**Use case:** Complete source distribution for releases

### 4. Benchmark Binaries
```bash
bazel build //dist:mango-iv-benchmarks
```
**Contents:** Pre-compiled benchmark executables
**Size:** Medium (~5-10MB)
**Use case:** Performance testing without building from source

### 5. Complete Package
```bash
bazel build //dist:mango-iv-complete
```
**Contents:** Library + binaries combined
**Size:** Medium (~5-10MB)
**Use case:** One-stop package with everything

## Output Location

All packages are generated in `bazel-bin/dist/` with `.tar.gz` extension:
```
bazel-bin/dist/
├── mango-iv-dist.tar.gz
├── mango-iv-headers.tar.gz
├── mango-iv-dev.tar.gz
├── mango-iv-benchmarks.tar.gz
└── mango-iv-complete.tar.gz
```

## Extracting Packages

```bash
# Extract to current directory
tar -xzf bazel-bin/dist/mango-iv-dist.tar.gz

# Extract to specific location
tar -xzf bazel-bin/dist/mango-iv-dist.tar.gz -C /path/to/destination

# List contents without extracting
tar -tzf bazel-bin/dist/mango-iv-dist.tar.gz
```

## Package Structure

After extraction, you'll find:

```
mango-iv-0.1.0/
├── include/mango/              # All library headers
│   ├── american_option.hpp    # American option pricing
│   ├── iv_solver.hpp          # Implied volatility solver
│   ├── pde_solver.hpp         # PDE solver core
│   ├── bspline_*.hpp          # B-spline interpolation
│   ├── operators/             # Spatial operators
│   │   ├── black_scholes_pde.hpp
│   │   ├── grid_spacing.hpp
│   │   └── operator_factory.hpp
│   ├── common/                # Common utilities
│   │   └── ivcalc_trace.h    # USDT tracing
│   └── 3rd/tl/               # Third-party (expected)
│       └── expected.hpp
├── docs/                      # Documentation
│   ├── README.md             # Project overview
│   ├── CLAUDE.md             # Developer guide
│   ├── TRACING.md            # USDT tracing guide
│   └── TRACING_QUICKSTART.md # Quick tracing tutorial
├── examples/                  # Example source code
│   ├── example_newton_solver.cc
│   └── example_expected_validation.cpp
├── build/                     # BUILD files (for Bazel users)
│   ├── MODULE.bazel
│   ├── BUILD.bazel
│   └── ...
└── VERSION                    # Package version (0.1.0)
```

## Using the Distributed Headers

### Integration with CMake

```cmake
# Add include directory
include_directories(/path/to/mango-iv-0.1.0/include)

# Use in your code
add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE /path/to/mango-iv-0.1.0/include)
```

### Integration with Bazel

Copy the extracted directory to your workspace and add to `MODULE.bazel`:

```python
local_path_override(
    module_name = "mango-iv",
    path = "third_party/mango-iv-0.1.0",
)
```

### Manual Compilation

```bash
# Compile with headers
g++ -std=c++20 -I/path/to/mango-iv-0.1.0/include \
    -O3 -march=native -fopenmp \
    your_app.cpp -o your_app
```

## Version Information

The `VERSION` file in the root of the extracted package contains the version string:

```bash
cat mango-iv-0.1.0/VERSION
# Output: 0.1.0
```

## Updating Package Version

To create a new release version, edit the `VERSION` variable in `dist/BUILD.bazel`:

```python
# Package version (update this for releases)
VERSION = "0.2.0"
```

Then rebuild the packages.

## Continuous Integration

For automated releases, consider adding these commands to your CI pipeline:

```bash
# Build all distribution packages
bazel build //dist:mango-iv-dist
bazel build //dist:mango-iv-headers
bazel build //dist:mango-iv-complete

# Upload to release artifacts
# (implementation depends on your CI system)
```

## License

See the main project for license information.
