<!-- SPDX-License-Identifier: MIT -->
# USDT Tracing Quick Start (5 Minutes)

Get started with zero-overhead tracing of the mango library in under 5 minutes.

## What is USDT Tracing?

USDT (User Statically-Defined Tracing) provides production-safe, zero-overhead instrumentation. When not actively traced, probes are single NOP instructions. When enabled, they capture detailed runtime data without recompiling.

**Benefits:**
- 🚀 Zero overhead when disabled (< 1% when enabled)
- 🔍 Deep visibility into solver behavior
- 🐛 Debug convergence and performance issues
- 📊 Production-safe monitoring

## Prerequisites

```bash
# Install bpftrace (Ubuntu/Debian)
sudo apt-get install bpftrace

# Install bpftrace (RHEL/Fedora)
sudo yum install bpftrace

# Optional: Install systemtap-sdt-dev for compile-time validation
sudo apt-get install systemtap-sdt-dev
```

## Step 1: Build Your Program

**Good news:** USDT is enabled by default! Just build normally:

```bash
# Build an example
bazel build //tests:pde_solver_test

# Or build your own program
bazel build //your:target
```

The binaries automatically include USDT probes (gracefully falls back if systemtap-sdt-dev not installed).

## Step 2: Verify USDT Support (Optional)

```bash
# Check if probes are present
sudo ./tools/mango-trace check ./bazel-bin/tests/pde_solver_test
```

Expected output:
```
✓ USDT notes found in binary
✓ Found 15 mango probes
✓ bpftrace can attach to probes

USDT support: OK
```

## Step 3: Run Your First Trace

**Option A: Use the helper tool (easiest)**

```bash
# Monitor all library activity
sudo ./tools/mango-trace monitor ./bazel-bin/tests/pde_solver_test

# Watch convergence behavior
sudo ./tools/mango-trace monitor ./bazel-bin/tests/pde_solver_test --preset=convergence

# Debug failures
sudo ./tools/mango-trace monitor ./bazel-bin/tests/pde_solver_test --preset=debug

# Profile performance
sudo ./tools/mango-trace monitor ./bazel-bin/tests/pde_solver_test --preset=performance
```

**Option B: Use bpftrace directly**

```bash
# Monitor all activity
sudo bpftrace tools/tracing/monitor_all.bt -c './bazel-bin/tests/pde_solver_test'

# Watch convergence
sudo bpftrace tools/tracing/convergence_watch.bt -c './bazel-bin/tests/pde_solver_test'
```

## Common Use Cases

### Debug Convergence Issues

```bash
sudo ./tools/mango-trace monitor ./bazel-bin/tests/pde_solver_test --preset=debug
```

Shows:
- Convergence failures with diagnostics
- Iteration counts and errors
- Validation failures

### Monitor PDE Solver Progress

```bash
sudo bpftrace tools/tracing/pde_detailed.bt -c './bazel-bin/tests/pde_solver_test'
```

Shows:
- Time stepping progress
- Iterations per step
- Convergence statistics

### Profile Performance

```bash
sudo bpftrace tools/tracing/performance_profile.bt -c './bazel-bin/tests/pde_solver_test'
```

Shows:
- Execution time per module
- Iteration counts
- Timing histograms

### Monitor Implied Volatility Calculations

```bash
sudo bpftrace tools/tracing/iv_detailed.bt -c './bazel-bin/tests/iv_solver_test'
```

Shows:
- Input parameters
- Brent's method iterations
- Convergence behavior

## Available Scripts

All scripts are in `tools/tracing/`:

| Script | Purpose | Use When |
|--------|---------|----------|
| `monitor_all.bt` | High-level overview | General monitoring |
| `convergence_watch.bt` | Convergence tracking | Tuning solver parameters |
| `debug_failures.bt` | Error diagnostics | Something is failing |
| `performance_profile.bt` | Timing analysis | Optimizing performance |
| `pde_detailed.bt` | PDE solver deep dive | PDE-specific debugging |
| `iv_detailed.bt` | IV calculation deep dive | IV-specific debugging |

## Helper Tool Commands

```bash
# List all available probes
sudo ./tools/mango-trace list ./bazel-bin/tests/pde_solver_test

# Validate USDT support
sudo ./tools/mango-trace check ./bazel-bin/tests/pde_solver_test

# Monitor with preset
sudo ./tools/mango-trace monitor <binary> --preset=<name>

# Run specific script
sudo ./tools/mango-trace run convergence_watch.bt <binary>
```

## Attach to Running Process

If your program is already running:

```bash
# Find the process ID
ps aux | grep my_program

# Attach bpftrace to it
sudo bpftrace tools/tracing/monitor_all.bt -p <PID>
```

## Save Output

```bash
# Save to file for later analysis
sudo bpftrace tools/tracing/performance_profile.bt -c './my_program' > trace_output.txt
```

## Troubleshooting

**Problem: "No USDT probes found"**

Solution:
```bash
# Check if probes exist
readelf -n ./bazel-bin/tests/pde_solver_test | grep NT_STAPSDT

# If missing, install systemtap-sdt-dev and rebuild
sudo apt-get install systemtap-sdt-dev
bazel clean
bazel build //tests:pde_solver_test
```

**Problem: "Permission denied"**

Solution: bpftrace requires root privileges
```bash
sudo ./tools/mango-trace monitor <binary>
```

**Problem: "bpftrace: command not found"**

Solution: Install bpftrace
```bash
# Ubuntu/Debian
sudo apt-get install bpftrace

# RHEL/Fedora
sudo yum install bpftrace
```

## Next Steps

- 📖 Read [TRACING.md](TRACING.md) for comprehensive documentation
- 🔧 Explore scripts in `tools/tracing/` directory
- 📝 See [tools/tracing/README.md](tools/tracing/README.md) for script details
- 🎯 Customize scripts for your specific needs

## Example Session

```bash
# 1. Build
bazel build //tests:pde_solver_test

# 2. Check USDT
sudo ./tools/mango-trace check ./bazel-bin/tests/pde_solver_test
# ✓ USDT support: OK

# 3. Monitor
sudo ./tools/mango-trace monitor ./bazel-bin/tests/pde_solver_test --preset=convergence

# Output shows:
# - Convergence iterations per step
# - Success/failure status
# - Timing information
# - Statistics summary
```

## That's It!

You're now tracing the mango library. Use the scripts to:
- ✅ Debug convergence issues
- ✅ Profile performance
- ✅ Monitor production systems
- ✅ Understand solver behavior

For more advanced usage, see [TRACING.md](TRACING.md).
