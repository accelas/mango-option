# GitHub Actions CI Configuration

## Dependency Caching Strategy

This project uses a multi-layer caching strategy to speed up CI runs:

### 1. Docker Image Caching (System Dependencies)

**Problem:** Installing system dependencies (`apt-get install`) takes significant time on every CI run.

**Solution:** We maintain a custom Docker image with all dependencies pre-installed, using a combination of optimizations:

- **Base Image:** `debian:trixie-slim` (~40% smaller than full trixie)
- **Image Location:** `ghcr.io/<repo>/ci-env:latest`
- **Dockerfile:** `.github/Dockerfile`
- **Build Workflow:** `.github/workflows/docker-build.yml`
- **Build Cache:** GitHub Actions cache (`type=gha`) for Docker layers

The Docker image includes:
- git, wget, ca-certificates
- build-essential (gcc, g++, make)
- python3
- systemtap-sdt-dev (for USDT tracing)
- libquantlib0-dev (for benchmarks)

**Optimizations applied:**
- `debian:trixie-slim` base (smaller footprint)
- `--no-install-recommends` flag (minimal dependencies)
- Aggressive cleanup (`apt-get clean`, remove temp files)
- GitHub Actions cache for Docker build layers

**How it works:**
1. When `Dockerfile` changes, `docker-build.yml` automatically builds and publishes a new image to GitHub Container Registry (GHCR)
2. Docker build layers are cached in GitHub Actions cache (free, 10 GB limit)
3. The CI workflow (`ci.yml`) uses this cached image, skipping dependency installation
4. Subsequent builds reuse cached layers, making rebuilds very fast

**Estimated image size:** 300-400 MB compressed (vs 500-650 MB with full trixie)

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
| Cold start (no cache) | ~5-7 min | First run ever |
| Docker rebuild (Dockerfile changed) | ~3-4 min | GitHub Actions cache speeds up build |
| Warm (full cache) | ~1-2 min | No code changes |
| Partial (code changed) | ~2-4 min | Bazel rebuilds only changed targets |

**Storage usage:**
- GHCR image: ~300-400 MB (compressed)
- GitHub Actions cache: ~500-800 MB (Docker layers)
- Bazel cache: ~100-500 MB (build artifacts)
- **Total: ~1-2 GB** (well within free tiers)

## Cost Considerations

**Public repositories:** FREE
- GHCR storage: Unlimited
- GHCR bandwidth: Unlimited
- GitHub Actions cache: 10 GB free

**Private repositories:**
- GHCR storage: 500 MB free, then $0.25/GB/month
  - Our image (~350 MB): **FREE** (under limit)
- GHCR bandwidth: 1 GB free/month
  - CI pulls are usually free (same network)
- GitHub Actions cache: 10 GB free
  - Our total usage (~1-2 GB): **FREE**

**Bottom line:** Cost is $0 for most cases, or ~$0.05/month worst case for private repos.

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
### Slow Test Workflow

- Nightly coverage for the slow Bazel targets runs via `.github/workflows/slow-tests.yml`. The workflow also supports manual triggering through *Run workflow*.
- Slow tests are the targets tagged with `slow` (currently `implied_volatility_test`, `adaptive_accuracy_test`, and `price_table_slow_test`). They stay out of the fast CI job to keep PR feedback tight.
- To reproduce the nightly job locally:
  ```bash
  bazel test //tests:slow_tests
  ```

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
