# Build System Documentation

Detailed documentation of the PyOctoMap build system and scripts.

## Overview

The build system automates the entire process of building PyOctoMap with bundled shared libraries, ensuring zero external dependencies.

## Build Scripts

### Linux Build Script (`build.sh`)

**Location**: `build.sh`
**Purpose**: Automated build and installation for Linux systems

**Features:**
- Python version checking
- Dependency installation
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
1. Checks Python version (3.9+)
2. Installs required dependencies
3. Cleans previous builds
4. Builds Cython extensions
5. Creates wheel package
6. Bundles shared libraries
7. Installs package
8. Runs basic tests
9. Provides usage instructions

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
**Platforms**: Ubuntu 20.04, Python 3.9-3.12

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
- Python: 3.9-3.12
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
