# GitHub Actions CI Configuration

## Dependency Caching Strategy

This project uses a multi-layer caching strategy to speed up CI runs:

### 1. Docker Image (System Dependencies)

**Problem:** Installing system dependencies (`apt-get install`) takes 2-3 minutes on every CI run.

**Solution:** A custom Docker image with all dependencies pre-installed.

- **Base Image:** `debian:trixie-slim`
- **Image Location:** `ghcr.io/<repo>/ci-env:latest`
- **Dockerfile:** `.github/Dockerfile`

The Docker image includes:
- git, wget, ca-certificates
- build-essential, clang
- python3, python3-numpy, python3-dev
- systemtap-sdt-dev (USDT tracing)
- libquantlib0-dev (benchmarks)
- liblapacke-dev
- libarrow-dev (Apache Arrow)
- Bazelisk (as `/usr/local/bin/bazel`)

**How it works:**
1. The CI workflow checks if the image exists in GHCR
2. If missing or Dockerfile changed, it builds and pushes automatically
3. The build-and-test job runs inside this pre-built container
4. Docker build layers are cached via GitHub Actions cache

**Rebuilding the image:**
```bash
# Automatic: push changes to .github/Dockerfile
# Manual: Actions → Build and Publish Docker Image → Run workflow
# Or: gh workflow run docker-build.yml
```

### 2. Bazel Caching

The CI workflow caches two Bazel-related paths:

1. **Bazel versions** (`~/.cache/bazelisk`) — keyed on `.bazelversion`
2. **Build artifacts** (`~/.cache/bazel`) — keyed on `.bazelversion` + `MODULE.bazel` + commit SHA

The commit SHA in the cache key ensures each run can save fresh artifacts (GitHub Actions caches are immutable). The `restore-keys` prefix provides partial cache hits from prior runs.

## Expected CI Times

| Scenario | Time |
|----------|------|
| Cold start (no cache, no image) | ~5-7 min |
| Image exists, no Bazel cache | ~3-4 min |
| Warm (image + Bazel cache) | ~1-2 min |
| Partial (code changed) | ~2-3 min |

## First-Time Setup

The first CI run builds and pushes the Docker image automatically. For public repos, make the GHCR package public:

1. Go to `github.com/<org>/<repo>/packages`
2. Find the `ci-env` package
3. Settings → Change visibility → Public

## Updating Dependencies

1. Edit `.github/Dockerfile`
2. Commit and push
3. CI detects the Dockerfile change and rebuilds the image

## Slow Tests

Nightly coverage for slow Bazel targets runs via `.github/workflows/slow-tests.yml` (also supports manual trigger). Slow tests are tagged with `slow` and excluded from the fast CI job.

```bash
# Run locally
bazel test //tests/... --test_tag_filters=slow
```

## Troubleshooting

**CI fails with "image not found":**
- First run: the `docker-image` job builds it automatically
- Manual fix: trigger `docker-build.yml` via workflow_dispatch

**Slow builds despite caching:**
- Check Bazel cache restore in CI logs ("Cache restored from key...")
- Verify Docker image pull succeeds (check `docker-image` job)
- Check GitHub Actions cache size (10 GB limit per repository)

**Stale dependencies:**
- Edit `.github/Dockerfile` and push — CI rebuilds the image automatically
