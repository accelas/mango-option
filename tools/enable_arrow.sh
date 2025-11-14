#!/usr/bin/env bash
# Enable Apache Arrow support for price table persistence
#
# This script:
# 1. Installs Arrow and dependencies via Conan (~20 min first time)
# 2. Enables the Conan extension in MODULE.bazel automatically
#
# Usage:
#   ./tools/enable_arrow.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "==> Checking for Conan..."
if ! command -v conan &> /dev/null; then
    echo "ERROR: Conan not found. Please install it first:"
    echo "  pipx install conan"
    echo "  # or"
    echo "  python3 -m pip install --user conan"
    exit 1
fi

echo "==> Detecting Conan profile..."
if [ ! -f ~/.conan2/profiles/default ]; then
    conan profile detect
    echo "Created default Conan profile"
fi

echo "==> Installing Arrow and dependencies via Conan..."
echo "    This will take ~15-30 minutes on first run (builds from source)"
conan install . --output-folder=conan_deps --build=missing

echo ""
echo "✓ Arrow dependencies installed!"
echo ""
echo "To build the project:"
echo "  bazel build //..."
echo ""
echo "Note: Arrow is now required for all builds. The Conan extension"
echo "is permanently enabled in MODULE.bazel (committed to the repo)."
