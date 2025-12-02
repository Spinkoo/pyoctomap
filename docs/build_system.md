# Build System Documentation

This page collects the lean, need-to-know details for building PyOctoMap.
For an end-to-end walkthrough see the main `README.md`, and for deep
troubleshooting refer to `docs/troubleshooting.md`.

## Overview

The build tooling automates compilation of the upstream OctoMap C++
libraries, bundles the shared objects into manylinux wheels, and verifies
the Python bindings with a quick smoke test. You can either rely on the
provided scripts or follow the manual outline below.

## Prerequisites

Install typical build essentials plus Python headers before running any
script:

| Distro | Command |
| --- | --- |
| Ubuntu/Debian | `sudo apt install build-essential cmake python3-dev python3-distutils git cython3` |
| Fedora/RHEL/CentOS | `sudo dnf/yum groupinstall "Development Tools"`<br>`sudo dnf/yum install cmake python3-devel git cython3` |
| Arch Linux | `sudo pacman -S base-devel cmake python cython git` |

> For a specific interpreter (e.g., Python 3.14) install the matching
> `python3.x-dev` package so the `Python.h` headers are available.

## Recommended Workflows

### 1. Use the standard Linux build script

```bash
chmod +x build.sh
./build.sh
```

The script checks Python >=3.8, builds OctoMap from `src/octomap`, runs
Cython, produces wheels, bundles shared libs via `auditwheel`, installs
the result, and executes a short import test.

### 2. Produce wheels for every supported Python

```bash
./build-wheel.sh
```

This helper iterates over the supported interpreters in the Docker
environment and stores ready-to-upload wheels in `dist/`.

### 3. Headless / Colab environments

Run the same `build.sh` script after installing `cmake` and
`build-essential` inside the Colab runtime:

```bash
!apt-get update -qq && apt-get install -y -qq cmake build-essential
!git clone --recursive https://github.com/Spinkoo/pyoctomap.git
!cd pyoctomap && chmod +x build.sh && ./build.sh
```

The script disables Qt during the OctoMap configure step, so no GUI
dependencies are required.

### 4. Docker build

```bash
chmod +x docker/build-docker.sh
./docker/build-docker.sh
```

Use this when you prefer a reproducible container image or need to build
on a platform without native toolchains. See `docker/Dockerfile` for the
exact base image and packages.

## Manual Build Outline

If you need to customize any step, the high-level process is:

1. **Clone with submodules**
   ```bash
   git clone --recursive https://github.com/Spinkoo/pyoctomap.git
   cd pyoctomap
   ```
2. **Prepare the environment**
   ```bash
   pip install -U pip setuptools wheel numpy cython auditwheel
   ```
3. **Build OctoMap**
   ```bash
   cd src/octomap
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j"$(nproc)"
   sudo make install && sudo ldconfig
   cd ../../..
   ```
4. **Build and bundle PyOctoMap**
   ```bash
   python -m build  # or python setup.py bdist_wheel
   auditwheel repair dist/*.whl
   ```
5. **Install & verify**
   ```bash
   pip install dist/*.whl
   python -c "import pyoctomap; tree = pyoctomap.OcTree(0.1)"
   ```

## Quick Troubleshooting

- Missing `Python.h` or NumPy headers → install `python3-dev` and
  `python3-numpy` (or `pip install numpy` before building).
- `cannot find -loctomap` during linking → ensure `src/octomap` was built
  and `sudo ldconfig` has been run.
- Colab timeouts → rerun `build.sh` with fewer compilation jobs:
  `MAKEFLAGS=-j2 ./build.sh`.

For exhaustive guidance (Google Colab quirks, memory limits, packaging
options, CI/CD examples, etc.) continue with `docs/troubleshooting.md`
and `docs/performance_guide.md`.
