# GitHub Actions CI Configuration

## Dependency Caching Strategy

This project uses a multi-layer caching strategy to speed up CI runs:

### 1. Docker Image Caching (System Dependencies)

**Problem:** Installing system dependencies (`apt-get install`) takes significant time on every CI run.

**Solution:** We maintain a custom Docker image with all dependencies pre-installed.

- **Image Location:** `ghcr.io/<repo>/ci-env:latest`
- **Dockerfile:** `.github/Dockerfile`
- **Build Workflow:** `.github/workflows/docker-build.yml`

The Docker image includes:
- git, wget, ca-certificates
- build-essential (gcc, g++, make)
- python3
- systemtap-sdt-dev (for USDT tracing)
- libquantlib0-dev (for benchmarks)

**How it works:**
1. When `Dockerfile` changes, `docker-build.yml` automatically builds and publishes a new image to GitHub Container Registry (GHCR)
2. The CI workflow (`ci.yml`) uses this cached image, skipping dependency installation
3. Docker layer caching further optimizes image builds

**Rebuilding the image:**
```bash
# Automatic: Push changes to .github/Dockerfile
git add .github/Dockerfile
git commit -m "Update CI dependencies"
git push

# Manual trigger: Go to Actions → Build and Publish Docker Image → Run workflow
```

### 2. Bazel Caching

The CI workflow caches three Bazel-related paths:

1. **Bazelisk binary** (`~/.bazelisk-cache`) - The Bazel launcher itself
2. **Bazel versions** (`~/.cache/bazelisk`) - Downloaded Bazel versions
3. **Build artifacts** (`~/.cache/bazel`) - Compiled objects and dependencies

Key features:
- Build cache key includes all Bazel config files (`.bazelversion`, `MODULE.bazel`, `**/*.bazel`, `**/*.bzl`)
- Restore-keys provide fallback to partial cache hits
- External dependencies (GoogleTest, Benchmark) are cached between runs

## Cache Hit Rates

Typical CI run times:

| Scenario | Time | Notes |
|----------|------|-------|
| Cold start (no cache) | ~5-7 min | First run or Dockerfile changed |
| Warm (full cache) | ~1-2 min | No code changes |
| Partial (code changed) | ~2-4 min | Bazel rebuilds only changed targets |

## First-Time Setup

The first time you push the Dockerfile, you need to:

1. **Make the image public** (optional, for public repos):
   - Go to `github.com/<org>/<repo>/packages`
   - Find the `ci-env` package
   - Settings → Change visibility → Public

2. **Or ensure GHCR credentials work** (done automatically via `GITHUB_TOKEN`)

## Updating Dependencies

To add or update system dependencies:

1. Edit `.github/Dockerfile`
2. Add the new package to the `apt-get install` line
3. Commit and push
4. The `docker-build.yml` workflow will automatically build and publish the new image
5. Future CI runs will use the updated image

Example:
```dockerfile
RUN apt-get update && \
    apt-get install -y \
        git \
        wget \
        ca-certificates \
        build-essential \
        python3 \
        systemtap-sdt-dev \
        libquantlib0-dev \
        NEW_PACKAGE_HERE && \
    rm -rf /var/lib/apt/lists/*
```

## Troubleshooting

**CI fails with "image not found":**
- The Docker image hasn't been built yet
- Manually trigger `docker-build.yml` workflow
- Or temporarily use `debian:trixie` as fallback image

**Slow builds despite caching:**
- Check if cache is being invalidated (Bazel config changes)
- Verify Docker image is being pulled successfully
- Check GitHub Actions cache size limits (10 GB per repository)

**Stale dependencies:**
- Rebuild the Docker image by pushing an update to `Dockerfile`
- Or trigger `docker-build.yml` manually with workflow_dispatch
