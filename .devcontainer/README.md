# Codespace Development Container

This devcontainer configuration provides a complete development environment for mango-iv that mirrors the CI environment.

## Features

- **Base Image**: Uses the same `ghcr.io/accelas/mango-iv/ci-env:latest` image as CI
- **Pre-installed Dependencies**:
  - Debian Trixie base with all system packages
  - Build tools: gcc, g++, make, python3
  - Libraries: systemtap-sdt-dev, libquantlib0-dev
  - Bazelisk (automatically installed on first run)
  - Git, GitHub CLI
- **Development Tools**:
  - Zsh with Oh My Zsh
  - VS Code extensions for C/C++ and Bazel
  - bpftrace for USDT tracing support

## Quick Start

### Using GitHub Codespaces

1. **Create a Codespace**:
   - Go to the repository on GitHub
   - Click "Code" → "Codespaces" → "Create codespace on main"
   - Wait for the container to build and setup to complete

2. **Start developing**:
   ```bash
   # Build everything
   bazel build //...

   # Run tests
   bazel test //tests:...

   # Run examples
   bazel run //examples:example_heat_equation
   ```

3. **Use Claude Code CLI**:
   - The Codespace has all dependencies pre-installed
   - You can run Claude Code directly in the terminal
   - All Bazel commands work out of the box

### Using VS Code Remote Containers

1. Install the "Dev Containers" extension
2. Open this repository in VS Code
3. Press F1 and select "Dev Containers: Reopen in Container"
4. Wait for container setup to complete

## Configuration Details

### Image

The devcontainer uses the CI image that's automatically built and published:
- Repository: `ghcr.io/accelas/mango-iv/ci-env`
- Tag: `latest` (always tracks the latest main branch build)

This ensures your development environment exactly matches CI, preventing "works on my machine" issues.

### Bazel Cache

A persistent Bazel cache is mounted to speed up builds:
- Host: `.bazel-cache/` in workspace
- Container: `/root/.cache/bazel`

This cache persists across Codespace rebuilds (but not across Codespace deletions).

### Post-Create Setup

The `setup.sh` script runs after container creation to:
1. Install Bazelisk
2. Set up Bazel cache directories
3. Install optional development tools (bpftrace)
4. Verify installations

### VS Code Extensions

The following extensions are automatically installed:
- **ms-vscode.cpptools**: C/C++ language support
- **ms-vscode.cpptools-extension-pack**: C/C++ extension pack
- **llvm-vs-code-extensions.vscode-clangd**: clangd language server
- **BazelBuild.vscode-bazel**: Bazel build system support
- **stackbuild.bazel-stack-vscode**: Enhanced Bazel integration

## Customization

### Adding Dependencies

To add new system packages:
1. Update `.github/Dockerfile` to add packages to the CI image
2. Push changes to trigger CI image rebuild
3. The devcontainer will automatically use the updated image

### Changing Bazel Settings

Edit `.bazelrc` in the repository root. Changes apply to both local development and CI.

### Adding VS Code Extensions

Edit `.devcontainer/devcontainer.json` and add extension IDs to the `extensions` array.

## Troubleshooting

### Container fails to start

- Check if the CI image is public: `ghcr.io/accelas/mango-iv/ci-env:latest`
- Try rebuilding: "Dev Containers: Rebuild Container"

### Bazel commands fail

- Check Bazelisk installation: `bazel version`
- Clear cache: `bazel clean --expunge`
- Check disk space: `df -h`

### USDT tracing not working

- Install systemtap-sdt-dev: Already included in base image
- Install bpftrace: Run `apt-get install -y bpftrace` (requires sudo in Codespace)
- Check kernel support: `uname -r` (eBPF requires Linux 4.1+)

## Performance Tips

1. **Use the Bazel cache**: It's automatically configured and persisted
2. **Parallel builds**: Bazel automatically detects CPU count
3. **Incremental builds**: Only changed targets are rebuilt
4. **Test filtering**: Use `--test_tag_filters` to run specific test suites

## Claude Code CLI Integration

This devcontainer is specifically designed to work seamlessly with Claude Code CLI:

1. All project dependencies are pre-installed
2. Build tools (Bazel) are configured and ready
3. Git and GitHub CLI are available for source control
4. The environment matches CI, ensuring consistent behavior

Simply open the Codespace and start using Claude Code commands!
