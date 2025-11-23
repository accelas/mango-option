# Running CI Locally with Act

This project uses [nektos/act](https://github.com/nektos/act) to run GitHub Actions CI workflows locally.

## Installation

### Ubuntu/Debian
```bash
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

### macOS
```bash
brew install act
```

### Other platforms
See: https://github.com/nektos/act#installation

## Prerequisites

- Docker installed and running
- 8GB+ RAM available for Docker containers

## Usage

### List available workflows
```bash
act -l
```

### Run the default CI workflow (push event)
```bash
act
```

### Run CI workflow (pull_request event)
```bash
act pull_request
```

### Run specific job
```bash
act -j build-and-test
```

### Run with specific workflow file
```bash
act -W .github/workflows/ci.yml
```

### Run slow tests workflow
```bash
act -W .github/workflows/slow-tests.yml
```

### Dry run (show what would execute)
```bash
act -n
```

### Run with more detailed output
```bash
act --verbose
```

## Configuration

Project configuration is in `.actrc`:
- Uses medium-sized Ubuntu images for compatibility
- Allocates 8GB memory to containers
- Reuses containers for faster subsequent runs
- Enables verbose output by default

## Common Issues

### Container already exists
```bash
# Clean up existing containers
act --rm
```

### Out of memory
```bash
# Increase memory limit in .actrc
--container-options "--memory=16g"
```

### Cache issues
```bash
# Disable cache actions
act --no-cache-server
```

### Permission denied (Docker)
```bash
# Add your user to docker group
sudo usermod -aG docker $USER
# Then log out and back in
```

## Differences from GitHub Actions

1. **No GitHub-specific features:** GITHUB_TOKEN, repository secrets, etc.
2. **Local filesystem:** Uses your local checkout, not a fresh clone
3. **Docker limitations:** Some container features may not work identically
4. **Cache behavior:** Local cache vs GitHub's cache service

## Tips

- Use `act -l` to see all available jobs before running
- Add `--dryrun` to see what would run without executing
- Use `--reuse` to speed up subsequent runs (default in .actrc)
- Use `--job <job-name>` to run specific jobs only

## Examples

### Run CI on current branch
```bash
act push
```

### Run fast tests only
```bash
act -j build-and-test
```

### Test a specific workflow change
```bash
act -W .github/workflows/ci.yml --dryrun
```

### Debug workflow failure
```bash
act --verbose --job build-and-test
```

## Resources

- Act GitHub: https://github.com/nektos/act
- Act Documentation: https://nektosact.com/
- Supported Actions: https://github.com/nektos/act#default-runners
