# Build System Documentation

Detailed documentation of the PyOctoMap build system and scripts.

## Overview

The build system automates the entire process of building PyOctoMap with bundled shared libraries, ensuring zero external dependencies.

## Prerequisites

Before building PyOctoMap from source, you need to install the following system dependencies:

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3-dev python3-distutils build-essential g++ gcc cython3
```

### CentOS/RHEL/Fedora
```bash
# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel gcc-c++ cython3

# Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel gcc-c++ cython3
```

### Arch Linux
```bash
sudo pacman -S python gcc g++ cython
```

> **📝 Note**: Replace `python3-dev` with `python3.x-dev` for specific Python versions (e.g., `python3.14-dev` for Python 3.14). The `python3.x-dev` package contains the header files (`Python.h`) required for compiling C extensions.

## Troubleshooting Build Issues

### Common Build Errors

**Error: `Python.h: No such file or directory`**
```bash
# Install Python development headers
sudo apt install python3-dev  # Ubuntu/Debian
sudo yum install python3-devel  # CentOS/RHEL
sudo dnf install python3-devel  # Fedora
```

**Error: `fatal error: 'numpy/arrayobject.h' file not found`**
```bash
# Install NumPy development headers
pip install numpy
# Or if using system packages:
sudo apt install python3-numpy-dev  # Ubuntu/Debian
```

**Error: `Cython not found`**
```bash
# Install Cython
pip install cython
# Or system package:
sudo apt install cython3  # Ubuntu/Debian
```

**Error: `g++: command not found`**
```bash
# Install C++ compiler
sudo apt install build-essential  # Ubuntu/Debian
sudo yum groupinstall "Development Tools"  # CentOS/RHEL
```

### Python Version Issues

For specific Python versions (e.g., Python 3.14), use the corresponding dev package:
```bash
# For Python 3.14
sudo apt install python3.14-dev python3.14-distutils

# For Python 3.13
sudo apt install python3.13-dev python3.13-distutils
```

## Build Scripts

### Linux Build Script (`build.sh`)

**Location**: `build.sh`
**Purpose**: Automated build and installation for Linux systems

**Features:**
- Python version checking
- Dependency installation
- OctoMap C++ library building
- Clean build process
- Library bundling with auditwheel
- Automatic testing
- Installation verification

**Usage:**
```bash
chmod +x build.sh
./build.sh
```

**What it does:**
1. Checks Python version (3.7+)
2. Installs required dependencies
3. **Builds OctoMap C++ libraries from source** (NEW)
4. Cleans previous builds
5. Builds Cython extensions
6. Creates wheel package
7. Bundles shared libraries
8. Installs package
9. Runs basic tests
10. Provides usage instructions

### Google Colab Installation

**Script**: Use the main `build.sh` script
**Purpose**: Automated build and installation for Google Colab environment

**Features:**
- OctoMap C++ library building from source (built into main script)
- Headless build configuration (no Qt dependencies)
- Automatic dependency checking
- Clean build process
- Library bundling
- Automatic testing

**Usage in Colab:**
```bash
# Install system dependencies first
!apt-get update -qq
!apt-get install -y -qq cmake build-essential

# Clone the repository with submodules
!git clone --recursive https://github.com/Spinkoo/pyoctomap.git
!cd pyoctomap

# Run the main build script
!chmod +x build.sh
!./build.sh
```

**What the build script does:**
1. **Checks Python version** (3.7+)
2. **Installs Python dependencies** (NumPy, Cython, setuptools, wheel)
3. **Builds OctoMap C++ libraries** from source using CMake with headless configuration
4. **Cleans previous builds** - Removes any existing build artifacts
5. **Builds Python package** - Compiles Cython extensions and creates wheel
6. **Bundles shared libraries** - Includes all required C++ libraries
7. **Installs package** - Installs the wheel
8. **Runs basic tests** - Verifies installation works correctly

**Colab-Specific Notes:**
- The main build script automatically handles OctoMap C++ library building
- Headless configuration disables Qt5/Qt6 dependencies for Colab compatibility
- No separate Colab-specific script needed

### Docker Build Script (`build-docker.sh`)

**Location**: `build-docker.sh`
**Purpose**: Build in isolated Docker environment

**Features:**
- Isolated build environment
- Consistent build results
- Cross-platform compatibility
- No system dependencies

**Usage:**
```bash
chmod +x build-docker.sh
./build-docker.sh
```

### Docker Configuration

**Dockerfile**: `docker/Dockerfile`
**Purpose**: Define build environment

**Base Image**: Ubuntu 20.04 LTS
**Includes**:
- Python 3.9+
- Build tools (gcc, g++, cmake)
- OctoMap dependencies
- Cython and NumPy

## Build Process

### Phase 1: Environment Setup

**Python Version Check:**
```bash
python3 --version
# Ensures Python 3.9+ is available
```

**Dependency Installation:**
```bash
pip install numpy cython setuptools wheel
pip install auditwheel  # For Linux library bundling
```

**Environment Variables:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CFLAGS="-O3 -march=native"
export CXXFLAGS="-O3 -march=native"
```

### Phase 2: Source Preparation

**Clean Previous Builds:**
```bash
rm -rf build/ dist/ *.egg-info/
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

**Submodule Initialization:**
```bash
git submodule update --init --recursive
```

### Phase 3: OctoMap Build

**CMake Configuration:**
```bash
cd src/octomap
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Compilation:**
```bash
make -j$(nproc)
sudo make install
```

**Library Installation:**
```bash
sudo ldconfig
```

### Phase 4: Python Package Build

**Cython Compilation:**
```bash
python setup.py build_ext --inplace
```

**Wheel Creation:**
```bash
python setup.py bdist_wheel
```

### Phase 5: Library Bundling

**Auditwheel Process:**
```bash
auditwheel repair dist/*.whl
```

**Library Detection:**
- Automatically finds required libraries
- Creates versioned symlinks
- Validates platform compatibility

**Bundled Libraries:**
- liboctomap.so
- libdynamicedt3D.so
- liboctomath.so
- System dependencies

### Phase 6: Installation & Testing

**Package Installation:**
```bash
pip install dist/*.whl
```

**Functionality Tests:**
```python
python -c "import octomap; print('Import successful')"
python -c "tree = octomap.OcTree(0.1); print('Tree creation successful')"
```

## Configuration Files

### setup.py

**Purpose**: Python package configuration
**Key Features**:
- Cython extension compilation
- Wheel package creation
- Metadata definition
- Dependency specification

**Extension Configuration:**
```python
extensions = [
    Extension(
        "octomap.octomap",
        ["octomap/octomap.pyx"],
        include_dirs=[...],
        libraries=[...],
        library_dirs=[...],
        language="c++"
    )
]
```

### pyproject.toml

**Purpose**: Modern Python packaging configuration
**Features**:
- Build system specification
- Project metadata
- Dependency management
- Tool configuration

### setup.cfg

**Purpose**: Additional package configuration
**Includes**:
- Package discovery
- Data files
- Entry points
- Build options

## Build Dependencies

### System Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install build-essential cmake git
sudo apt install python3-dev python3-pip
sudo apt install libeigen3-dev
```

**CentOS/RHEL:**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install cmake git python3-devel
sudo yum install eigen3-devel
```

### Python Dependencies

**Core Dependencies:**
- numpy >= 1.19.0
- cython >= 0.29.0
- setuptools >= 40.0.0
- wheel >= 0.36.0

**Build Dependencies:**
- auditwheel (Linux)
- delocate (macOS)
- twine (for PyPI upload)

**Optional Dependencies:**
- matplotlib (visualization)
- open3d (3D visualization)
- pytest (testing)

## Build Variants

### Development Build

**Purpose**: Local development and testing
**Features**:
- Debug symbols
- Fast compilation
- Hot reloading

**Commands:**
```bash
python setup.py build_ext --inplace --debug
pip install -e .
```

### Release Build

**Purpose**: Production distribution
**Features**:
- Optimized compilation
- Library bundling
- Wheel packaging

**Commands:**
```bash
python setup.py bdist_wheel
auditwheel repair dist/*.whl
```

### CI/CD Build

**Purpose**: Automated builds
**Features**:
- Isolated environment
- Reproducible builds
- Automated testing

**Configuration:**
```yaml
# GitHub Actions example
- name: Build wheel
  run: |
    pip install build auditwheel
    python -m build
    auditwheel repair dist/*.whl
```

## Troubleshooting Build Issues

### Common Build Errors

**Cython Compilation Errors:**
```bash
# Check Cython version
pip install --upgrade cython

# Check compiler
gcc --version
g++ --version
```

**Library Not Found:**
```bash
# Check library paths
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
sudo ldconfig

# Verify OctoMap installation
pkg-config --libs octomap
```

**Memory Issues:**
```bash
# Reduce parallel jobs
make -j2

# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Google Colab Specific Issues

**Error: `cannot find -ldynamicedt3d: No such file or directory`**
```bash
# This error occurs when OctoMap C++ libraries aren't built first
# Solution: Use the main build.sh script which builds libraries first
!git clone --recursive https://github.com/Spinkoo/pyoctomap.git
!cd pyoctomap && chmod +x build.sh && ./build.sh
```

**Error: `fatal: not a git repository` during cloning**
```bash
# This happens when the submodule isn't properly initialized
# Solution: Clone with --recursive flag
!git clone --recursive https://github.com/Spinkoo/pyoctomap.git
```

**Error: `cmake: command not found`**
```bash
# Install system dependencies first
!apt-get update -qq
!apt-get install -y -qq cmake build-essential
```

**Error: `Python.h: No such file or directory` in Colab**
```bash
# Install Python development headers
!apt-get install -y python3-dev
```

**Build takes too long in Colab**
```bash
# Reduce parallel jobs to avoid memory issues
# The build script automatically uses -j$(nproc) but you can modify it to use fewer cores
!make -j2  # Use only 2 cores instead of all available
```

### Build Optimization

**Compiler Flags:**
```bash
export CFLAGS="-O3 -march=native -mtune=native"
export CXXFLAGS="-O3 -march=native -mtune=native"
```

**Parallel Build:**
```bash
# Use all available cores
make -j$(nproc)

# Or specify number of cores
make -j4
```

**Debug Build:**
```bash
# Enable debug symbols
export CFLAGS="-g -O0"
export CXXFLAGS="-g -O0"
python setup.py build_ext --inplace --debug
```

## Continuous Integration

### GitHub Actions

**Workflow**: `.github/workflows/ci.yml`
**Triggers**: Push, Pull Request
**Platforms**: Ubuntu 20.04, Python 3.9-3.13

> **📝 Note**: Python 3.14 support will be added once it becomes available in the manylinux images. Currently, Python 3.14 is not yet supported by the official manylinux build environment.

**Steps:**
1. Checkout code
2. Setup Python
3. Install dependencies
4. Build OctoMap
5. Build Python package
6. Run tests
7. Upload artifacts

### Docker Build

**Multi-stage Build:**
```dockerfile
# Stage 1: Build OctoMap
FROM ubuntu:20.04 AS octomap-builder
# ... build OctoMap

# Stage 2: Build Python package
FROM ubuntu:20.04 AS python-builder
COPY --from=octomap-builder /usr/local /usr/local
# ... build Python package
```

## Distribution

### Wheel Packages

**Linux Wheels:**
- Platform: linux_x86_64
- Python: 3.9-3.13
- Bundled libraries included

**Future Platforms:**
- Windows (via WSL)
- macOS (universal binaries)
- ARM64 support

### PyPI Upload

**Manual Upload:**
```bash
twine upload dist/*.whl
```

**Automated Upload:**
```bash
# Using GitHub Actions
# Automatic upload on release tags
```

## Maintenance

### Regular Updates

**Dependencies:**
- Update Python versions
- Update OctoMap version
- Update build tools

**Testing:**
- Test on multiple Python versions
- Test on different Linux distributions
- Validate wheel compatibility

**Documentation:**
- Update build instructions
- Document new features
- Maintain troubleshooting guide
