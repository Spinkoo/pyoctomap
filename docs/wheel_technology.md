# Wheel Technology Documentation

PyOctoMap uses advanced wheel bundling technology to create self-contained packages with zero external dependencies.

## Overview

Wheel bundling allows us to include all required C++ libraries directly in the Python package, eliminating the need for users to install system libraries.

## Technology Stack

### Linux (auditwheel)

**Tool**: `auditwheel`
**Purpose**: Bundles shared libraries for Linux wheels

**Features:**
- Automatic library detection
- Versioned symlinks creation
- Platform tag validation
- Library compatibility checking

**Process:**
1. Build Cython extensions
2. Create initial wheel
3. Run `auditwheel` to detect dependencies
4. Bundle libraries into wheel
5. Create versioned symlinks
6. Validate final wheel

### macOS (delocate)

**Tool**: `delocate`
**Purpose**: Bundles shared libraries for macOS wheels

**Features:**
- Dynamic library bundling
- Framework support
- Code signing compatibility
- Universal binary support

### Windows (WSL)

**Approach**: Build in WSL environment
**Compatibility**: Windows via WSL2

**Benefits:**
- Full Linux compatibility
- Native performance
- Easy deployment
- No Windows-specific build complexity

## Bundled Libraries

### Core Libraries

**liboctomap.so**
- Primary OctoMap functionality
- 3D occupancy mapping
- Octree operations
- Ray casting algorithms

**libdynamicedt3D.so**
- Dynamic EDT3D implementation
- Distance transform calculations
- Path planning support
- Real-time updates

**liboctomath.so**
- Mathematical utilities
- Vector operations
- Coordinate transformations
- Geometric calculations

### Dependencies

**System Libraries:**
- libc.so.6 (GNU C Library)
- libm.so.6 (Math library)
- libpthread.so.0 (POSIX threads)
- libstdc++.so.6 (C++ standard library)

**Versioned Symlinks:**
```
liboctomap.so -> liboctomap.so.1.10
liboctomap.so.1.10 -> liboctomap.so.1.10.0
liboctomap.so.1.10.0 (actual library)
```

## Build Process

### Docker-Based Build Process

```bash
#!/bin/bash
# build-wheel.sh - Docker-based build for multiple Python versions

# 1. Build wheels using Docker
docker build -f docker/Dockerfile.wheel -t pyoctomap-wheel .

# 2. Extract wheels from container
docker run --name temp-container pyoctomap-wheel
docker cp temp-container:/wheels/ ./dist/
docker rm temp-container

# 3. Test the wheels
pip install dist/*.whl
python -c "import pyoctomap; print('Success!')"
```

### Manual Build (Advanced)

```bash
# For development/testing only
python setup.py bdist_wheel
auditwheel repair dist/*.whl  # Linux only
```

### Manual Build Process

```bash
# 1. Build Cython extensions
python setup.py build_ext --inplace

# 2. Create wheel
python setup.py bdist_wheel

# 3. Bundle libraries
auditwheel repair dist/pyoctomap-*.whl

# 4. Install bundled wheel
pip install dist/pyoctomap-*_linux_x86_64.whl
```

## Wheel Structure

### Directory Layout

```
pyoctomap-1.1.0-cp312-cp312-linux_x86_64.whl
├── octomap/
│   ├── __init__.py
│   ├── octomap.pyx
│   ├── octomap_defs.pxd
│   └── dynamicEDT3D_defs.pxd
├── octomap.libs/
│   ├── liboctomap.so.1.10.0
│   ├── libdynamicedt3D.so.1.0.0
│   ├── liboctomath.so.1.10.0
│   └── [versioned symlinks]
└── METADATA
```

### Library Loading

**Automatic Loading:**
- Libraries loaded automatically on import
- No manual library path configuration
- Works in any Python environment

**Loading Order:**
1. Core math libraries (libm, libc)
2. C++ standard library (libstdc++)
3. OctoMap libraries (liboctomath, liboctomap)
4. Dynamic EDT3D (libdynamicedt3D)

## Platform Support

### Linux Distributions

**Supported:**
- Ubuntu 18.04+ (x86_64)
- CentOS 7+ (x86_64)
- Debian 9+ (x86_64)
- Fedora 30+ (x86_64)

**Architecture:**
- x86_64 (AMD64)
- Future: ARM64 support planned

### Windows (WSL)

**Requirements:**
- WSL2 installed
- Ubuntu 18.04+ in WSL
- Python 3.9+

**Installation:**
```bash
# In WSL
pip install pyoctomap
```

### macOS (Future)

**Planned Support:**
- macOS 10.15+ (Catalina)
- x86_64 and ARM64 (Apple Silicon)
- Universal binaries

## Performance Impact

### Library Bundling Overhead

**Size Impact:**
- Base package: ~2MB
- With bundled libraries: ~15MB
- Total increase: ~7x

**Runtime Impact:**
- Minimal performance overhead
- Libraries loaded once at import
- No repeated loading costs

### Memory Usage

**Library Memory:**
- liboctomap: ~2MB
- libdynamicedt3D: ~1MB
- liboctomath: ~0.5MB
- Total: ~3.5MB

**Application Memory:**
- No additional overhead
- Same as system-installed libraries

## Troubleshooting

### Common Issues

**Library Not Found:**
```bash
# Check if libraries are bundled
python -c "import octomap; print(octomap.__file__)"
ls -la $(python -c "import octomap; print(octomap.__file__)")/../libs/
```

**Version Conflicts:**
```bash
# Check library versions
ldd $(python -c "import octomap; print(octomap.__file__)")/../libs/*.so
```

**Import Errors:**
```python
# Debug import
import sys
print(sys.path)
import octomap
print(octomap.__file__)
```

### Build Issues

**auditwheel Errors:**
```bash
# Check wheel compatibility
auditwheel show dist/*.whl

# Repair wheel
auditwheel repair dist/*.whl --plat linux_x86_64
```

**Library Detection:**
```bash
# Check system libraries
ldd /usr/lib/x86_64-linux-gnu/liboctomap.so

# Check bundled libraries
ldd octomap/libs/liboctomap.so.1.10.0
```

## Best Practices

### Development

**Local Development:**
```bash
# Install in development mode
pip install -e .

# Test with bundled libraries
python -m pytest tests/
```

**CI/CD Integration:**
```yaml
# GitHub Actions example
- name: Build wheel
  run: |
    pip install build auditwheel
    python -m build
    auditwheel repair dist/*.whl
```

### Distribution

**PyPI Upload:**
```bash
# Upload to PyPI
twine upload dist/*.whl
```

**Private Distribution:**
```bash
# Upload to private PyPI
twine upload --repository-url https://your-pypi.com/simple/ dist/*.whl
```

## Future Improvements

### Planned Features

**Multi-Platform Support:**
- Windows native wheels
- macOS universal binaries
- ARM64 support

**Advanced Bundling:**
- Selective library bundling
- Compression optimization
- Lazy loading support

**Performance Optimization:**
- Reduced bundle size
- Faster loading times
- Memory usage optimization
