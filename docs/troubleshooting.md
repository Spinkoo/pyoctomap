# Troubleshooting Guide

Common issues and solutions for PyOctoMap.

## Installation Issues

### Pre-built Wheels Not Available

**Problem**: `pip install pyoctomap` tries to build from source
**Solution**: Use the pre-built manylinux wheels

```bash
# Force wheel installation (recommended)
pip install pyoctomap --only-binary=all

# If that fails, try with no cache
pip install pyoctomap --only-binary=all --no-cache-dir
```

### Python Version Compatibility

**Problem**: Package fails to install or import
**Solution**: Ensure Python 3.9+ is installed

```bash
# Check Python version
python3 --version

# If version is too old, install Python 3.9+
sudo apt update
sudo apt install python3.9 python3.9-pip
```

### Missing Dependencies

**Problem**: ImportError when importing octomap
**Solution**: Install required dependencies

```bash
# Install core dependencies
pip install numpy cython

# For building from source
pip install setuptools wheel

# For library bundling (Linux)
pip install auditwheel
```

### WSL Issues (Windows)

**Problem**: Commands not found in WSL
**Solution**: Ensure WSL2 is properly configured

```bash
# Check WSL version
wsl --list --verbose

# Update to WSL2 if needed
wsl --set-version Ubuntu 2

# Install Python in WSL
sudo apt update
sudo apt install python3 python3-pip
```

## Build Issues

### Cython Compilation Errors

**Problem**: Cython compilation fails
**Solution**: Update compiler toolchain

```bash
# Install build essentials
sudo apt install build-essential

# Update gcc
sudo apt install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
```

### Library Not Found

**Problem**: Cannot find OctoMap libraries
**Solution**: Build OctoMap from source

```bash
# Clone and build OctoMap
cd src/octomap
mkdir build && cd build
cmake ..
make -j4
sudo make install

# Update library path
sudo ldconfig
```

### Memory Issues

**Problem**: Out of memory during build
**Solution**: Increase swap space or reduce parallel jobs

```bash
# Reduce parallel jobs
make -j2

# Or increase swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Runtime Issues

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'octomap'`
**Solution**: Check installation and Python path

```python
# Check if module is installed
import sys
print(sys.path)

# Try importing
try:
    import octomap
    print("OctoMap imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
```

### Library Loading Errors

**Problem**: `OSError: liboctomap.so: cannot open shared object file`
**Solution**: Check library paths and dependencies

```bash
# Check if libraries are bundled
python -c "import octomap; print(octomap.__file__)"
ls -la $(python -c "import octomap; print(octomap.__file__)")/../libs/

# Check library dependencies
ldd $(python -c "import octomap; print(octomap.__file__)")/../libs/*.so
```

### Memory Corruption on Exit

**Problem**: "double free or corruption" messages on exit
**Solution**: This is a known issue with C++ libraries in Python

```python
# Add this at the end of your scripts
import os
os._exit(0)  # Force clean exit
```

**Alternative Solution:**
```python
# Use context managers
with octomap.OcTree(0.1) as tree:
    # Your code here
    pass
# Automatic cleanup
```

## Performance Issues

### Slow Operations

**Problem**: Operations are slower than expected
**Solution**: Check for common performance bottlenecks

```python
# Use batch operations instead of individual updates
points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tree.addPointsBatch(points)  # Fast

# Instead of:
for point in points:
    tree.updateNode(point, True)  # Slow
```

### High Memory Usage

**Problem**: High memory consumption
**Solution**: Optimize tree resolution and update frequency

```python
# Use appropriate resolution
tree = octomap.OcTree(0.1)  # 10cm resolution

# Update inner occupancy less frequently
tree.addPointsBatch(points, update_inner_occupancy=False)
tree.updateInnerOccupancy()  # Call once after batch
```

### Iterator Performance

**Problem**: Iterators are slow
**Solution**: Use appropriate iterator types and limits

```python
# Use leaf iterator for occupied nodes only
for leaf in tree.begin_leafs():
    if tree.isNodeOccupied(leaf):
        # Process occupied nodes
        pass

# Limit iteration depth
for node in tree.begin_tree(maxDepth=5):
    # Process nodes up to depth 5
    pass
```

## API Issues

### Node Access Errors

**Problem**: "Iterator has no current node" errors
**Solution**: This is normal behavior for some coordinates

```python
# Handle None nodes gracefully
for leaf in tree.begin_leafs():
    node = leaf.current_node
    if node is not None:
        # Node exists, process it
        occupied = tree.isNodeOccupied(leaf)
    else:
        # Node doesn't exist at this coordinate (normal)
        continue
```

### Search Failures

**Problem**: `search()` returns None
**Solution**: Check coordinates and tree state

```python
# Check if point is in tree bounds
point = [1.0, 2.0, 3.0]
node = tree.search(point)

if node is None:
    print("Point not found in tree")
    # Check if point is within tree bounds
    print(f"Tree bounds: {tree.getBBXMin()} to {tree.getBBXMax()}")
```

### File I/O Errors

**Problem**: Cannot save or load files
**Solution**: Check file permissions and paths

```python
import os

# Check file permissions
filename = "my_map.bt"
if os.path.exists(filename):
    if not os.access(filename, os.W_OK):
        print("No write permission")
    if not os.access(filename, os.R_OK):
        print("No read permission")

# Use absolute paths
abs_path = os.path.abspath(filename)
tree.write(abs_path)
```

## Debugging Tips

### Enable Debug Output

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your octomap code here
```

### Check Tree State

```python
# Verify tree is properly initialized
print(f"Tree resolution: {tree.getResolution()}")
print(f"Tree size: {tree.size()}")
print(f"Tree depth: {tree.getTreeDepth()}")
print(f"Leaf nodes: {tree.getNumLeafNodes()}")
```

### Memory Profiling

```python
import psutil
import os

# Monitor memory usage
process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# After operations
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

### Performance Profiling

```python
import time
import cProfile

# Time operations
start = time.time()
tree.addPointsBatch(points)
end = time.time()
print(f"Batch operation took: {end - start:.3f} seconds")

# Profile code
cProfile.run('tree.addPointsBatch(points)')
```

## Getting Help

### Check Documentation

1. Read the [API Reference](api_reference.md)
2. Check the [Examples](../examples/) directory
3. Review the [Performance Guide](performance_guide.md)

### Report Issues

When reporting issues, include:

1. **Python version**: `python3 --version`
2. **Operating system**: `uname -a`
3. **Error message**: Complete traceback
4. **Minimal example**: Code that reproduces the issue
5. **Environment**: Virtual environment, system Python, etc.

### Community Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/Spinkoo/pyoctomap/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/Spinkoo/pyoctomap/discussions)
- **Documentation**: [Read the docs](https://github.com/Spinkoo/pyoctomap/tree/main/docs)

## Common Solutions

### Quick Fixes

```bash
# Reinstall package
pip uninstall pyoctomap
pip install pyoctomap

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Rebuild from source
python setup.py clean
python setup.py build_ext --inplace
```

### Environment Reset

```bash
# Create fresh virtual environment
python3 -m venv fresh_env
source fresh_env/bin/activate
pip install pyoctomap
```

### System Update

```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Python packages
pip install --upgrade pip setuptools wheel
```
