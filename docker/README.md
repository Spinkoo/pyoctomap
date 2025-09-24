# Docker Configuration

This folder contains Docker-related files for the octomap2python project.

## Files:

- `Dockerfile` - Base Docker image for development
- `Dockerfile.ci` - CI-optimized Docker image with OctoMap pre-built
- `docker-compose.yml` - Docker Compose configuration for local development
- `build-docker.sh` - Script to build and test with Docker

## Usage:

### Local Development:
```bash
cd docker
docker-compose up
```

### CI Build:
The CI workflow automatically uses the pre-built Docker image.

### Manual Build:
```bash
cd docker
./build-docker.sh
```

## Docker Images:

- **Base Image**: `ubuntu:24.04` with OctoMap and dependencies
- **CI Image**: Pre-built image with OctoMap compiled and ready for CI
