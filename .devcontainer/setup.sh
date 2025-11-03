#!/bin/bash
set -e

echo "========================================="
echo "Setting up Mango-IV development environment"
echo "========================================="

# Install Bazelisk
echo "Installing Bazelisk..."
if [ ! -f /usr/local/bin/bazel ]; then
    wget -q https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -O /usr/local/bin/bazel
    chmod +x /usr/local/bin/bazel
    echo "✓ Bazelisk installed"
else
    echo "✓ Bazelisk already installed"
fi

# Install bpftrace for USDT tracing (optional, but useful for debugging)
echo "Installing bpftrace for USDT tracing..."
apt-get update -qq && apt-get install -y --no-install-recommends bpftrace || echo "⚠ bpftrace installation failed (optional)"

# Create Bazel cache directory
echo "Setting up Bazel cache..."
mkdir -p /root/.cache/bazel
echo "✓ Bazel cache directory created"

# Verify installation
echo ""
echo "Verifying installation..."
bazel version
git --version
gh --version

echo ""
echo "========================================="
echo "✓ Development environment ready!"
echo "========================================="
echo ""
echo "Quick start commands:"
echo "  bazel build //...         - Build all targets"
echo "  bazel test //tests:...    - Run all tests"
echo "  bazel run //examples:example_heat_equation"
echo ""
echo "For Claude Code CLI, you can now use this Codespace!"
echo ""
