# OctoMap2Python

<div align="center">
<img src="images/octomap_core.png" alt="OctoMap Core" width="900">
</div>

A comprehensive Python wrapper for the OctoMap C++ library, providing efficient 3D occupancy mapping capabilities for robotics and computer vision applications. This modernized binding offers enhanced performance, bundled shared libraries for easy deployment, and seamless integration with the Python scientific ecosystem. The package is designed for distribution via PyPI with initial Linux support, followed by Windows compatibility in future releases.

## Features

- **3D Occupancy Mapping**: Efficient octree-based 3D occupancy mapping
- **Probabilistic Updates**: Stochastic occupancy updates with uncertainty handling
- **Path Planning**: Ray casting and collision detection
- **File Operations**: Save/load octree data in binary format
- **Bundled Libraries**: No external dependencies - all C++ libraries included
- **Python Integration**: Clean Python interface with NumPy support
- **Wheel Distribution**: Self-contained wheel packages with bundled shared libraries
- **Linux-first**: Designed for Linux wheels; Windows support is planned next

## Installation
A Python wrapper for the C++ OctoMap library. It provides Python bindings to the OctoMap C++ library and is intended to be distributed via PyPI (Linux first; Windows support is planned next).

**Linux:**
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/Spinkoo/octomap2python.git
cd octomap2python

# Build and install OctoMap C++ library
cd src/octomap
mkdir build
cd build
cmake ..
make
sudo make install

# Return to main project and run automated build script
cd ../../..
chmod +x build.sh
./build.sh
```

For more details on building OctoMap (including Windows instructions), see the [official OctoMap repository](https://github.com/OctoMap/octomap).

When published on PyPI (Linux):
```bash
pip install octomap2python
```

The build scripts will automatically:
- Check Python version and dependencies
- Install required packages (NumPy, Cython, auditwheel/delocate)
- Clean previous builds
- Build the wheel package with **bundled shared libraries**
- Install the package
- Run basic functionality tests

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/Spinkoo/octomap2python.git
cd octomap2python

# Install dependencies
pip install setuptools numpy cython

# For Linux: Install auditwheel for library bundling
pip install auditwheel

# Build and install
python setup.py bdist_wheel
pip install dist/octomap2python-1.1.0-cp312-cp312-linux_x86_64.whl
```

### Optional: Visualization Dependencies

Install visualization packages for the demo scripts:
```bash
# For 2D occupancy grid visualization
pip install matplotlib

# For 3D visualization with Open3D
pip install open3d
```
- For building Open3D from source, **Windows GPU builds using WSL, see [Open3D on WSL](https://github.com/Spinkoo/Open3DWSL).**

### Requirements

- Python 3.9+
- setuptools
- NumPy
- Cython (for building from source)
- auditwheel (Linux) for library bundling

**Optional for visualization:**
- matplotlib (for 2D plotting and occupancy grids)
- open3d (for 3D visualization)

## Quick Start

```python
import octomap

# Create an octree with 0.1m resolution
tree = octomap.OcTree(0.1)

# Add occupied points
tree.updateNode([1.0, 2.0, 3.0], True)
tree.updateNode([1.1, 2.1, 3.1], True)

# Add free space
tree.updateNode([0.5, 0.5, 0.5], False)

# Check occupancy
node = tree.search([1.0, 2.0, 3.0])
if node and tree.isNodeOccupied(node):
    print("Point is occupied!")

# Save to file
tree.write("my_map.bt")

# Load from file
loaded_tree = tree.read("my_map.bt")
```

## Examples

See runnable demos in `examples/`:
- `examples/basic_test.py` — smoke test for core API
- `examples/demo_occupancy_grid.py` — build and visualize a 2D occupancy grid from the octree
- `examples/demo_octomap_open3d.py` — visualize octomap data with Open3D (requires `open3d`)

### Demo Visualizations

**3D OctoMap Scene Visualization:**
<div align="center">
<img src="images/octomap_demo_scene.png" alt="OctoMap Demo Scene" width="700">
</div>


**Occupancy Grid Visualization:**
<div align="center">
<img src="images/occupancy_grid.png" alt="Occupancy Grid" width="700">
</div>

### Room Mapping

```python
import octomap
import numpy as np

# Create octree
tree = octomap.OcTree(0.05)  # 5cm resolution

# Add walls
for x in np.arange(0, 4.0, 0.05):
    for y in np.arange(0, 4.0, 0.05):
        tree.updateNode([x, y, 0], True)  # Floor
        tree.updateNode([x, y, 3.0], True)  # Ceiling

# Add furniture
for x in np.arange(2.0, 3.0, 0.05):
    for y in np.arange(2.0, 2.5, 0.05):
        for z in np.arange(0, 0.8, 0.05):
            tree.updateNode([x, y, z], True)  # Table

print(f"Tree size: {tree.size()} nodes")
```

### Path Planning

```python
def is_path_clear(start, end, tree, steps=50):
    """Simple ray casting for path planning"""
    for i in range(steps + 1):
        t = i / steps
        point = [
            start[0] + t * (end[0] - start[0]),
            start[1] + t * (end[1] - start[1]),
            start[2] + t * (end[2] - start[2])
        ]
        node = tree.search(point)
        if node and tree.isNodeOccupied(node):
            return False, point
    return True, None

# Check if path is clear
start = [0.5, 2.0, 0.5]
end = [2.0, 2.0, 0.5]
clear, obstacle = is_path_clear(start, end, tree)
print(f"Path clear: {clear}")
```

### Probabilistic Mapping

```python
# Multiple sensor readings with uncertainty
sensor_readings = [
    ([1.0, 1.0, 1.0], True),   # First reading
    ([1.0, 1.0, 1.0], True),   # Second reading (increases confidence)
    ([0.5, 0.5, 0.5], False),  # Free space
    ([2.0, 2.0, 2.0], True),   # Uncertain area
    ([2.0, 2.0, 2.0], False),  # Conflicting reading
]

for point, occupied in sensor_readings:
    tree.updateNode(point, occupied)

# Check final occupancy
node = tree.search([1.0, 1.0, 1.0])
if node:
    print(f"Occupied: {tree.isNodeOccupied(node)}")
    print(f"At threshold: {tree.isNodeAtThreshold(node)}")
```

## API Reference

### OcTree Class

#### Constructor
- `OcTree(resolution)` - Create octree with specified resolution

#### Core Methods
- `updateNode(point, occupied)` - Update occupancy at point
- `search(point)` - Find node at point
- `isNodeOccupied(node)` - Check if node is occupied
- `isNodeAtThreshold(node)` - Check if node is at occupancy threshold
- `size()` - Get number of nodes
- `getResolution()` - Get tree resolution

#### File Operations
- `write(filename)` - Save tree to file
- `read(filename)` - Load tree from file (returns new OcTree instance)

#### Iterators
- `SimpleTreeIterator(tree)` - Iterate over all nodes
- `SimpleLeafIterator(tree)` - Iterate over leaf nodes

### OcTreeNode Class

#### Methods
- `getOccupancy()` - Get occupancy probability
- `getValue()` - Get log-odds value
- `setValue(value)` - Set log-odds value
- `hasChildren()` - Check if node has children
- `childExists(i)` - Check if child i exists

## File Format

The wrapper uses OctoMap's binary format (`.bt` files) for saving and loading trees. This format is:
- **Efficient**: Compressed binary format
- **Portable**: Cross-platform compatible
- **Standard**: Compatible with OctoMap tools

## Wheel Bundling Technology

This package uses advanced wheel bundling to include all required C++ libraries:

- **Automatic Library Detection**: Finds and bundles all OctoMap dependencies
- **Versioned Symlinks**: Creates proper library versioning (e.g., `liboctomap.so.1.10`)
- **Platform-Specific Tools**: Uses `auditwheel` (Linux) and `delocate` (macOS)
- **Zero Runtime Dependencies**: No need to install system libraries
- **PyPI Ready**: Can be uploaded to PyPI for easy distribution

**Bundled Libraries:**
- `liboctomap.so` - Core OctoMap functionality
- `libdynamicedt3d.so` - Dynamic EDT3D for distance transforms
- `liboctomath.so` - OctoMap math utilities


## Performance

- **Memory Efficient**: Octree structure minimizes memory usage
- **Fast Updates**: O(log n) insertion and search
- **Scalable**: Handles large 3D environments
- **Real-time**: Suitable for robotics applications

## License

MIT License - see [LICENSE](./LICENSE) file for details.

## Build Scripts

The included build scripts automate the entire build and installation process with **bundled shared libraries**:

**Linux:**
```bash
./build.sh
```

**What it does:**
- ✅ Checks Python version compatibility
- ✅ Installs required dependencies (NumPy, Cython, auditwheel/delocate)
- ✅ Cleans previous build artifacts
- ✅ Builds the wheel package with **bundled shared libraries**
- ✅ Automatically bundles all C++ libraries (liboctomap, libdynamicedt3d, etc.)
- ✅ Creates versioned symlinks for library compatibility
- ✅ Installs the package
- ✅ Runs functionality tests
- ✅ Provides usage instructions

**Key Benefits:**
- 🚀 **Zero Dependencies**: No need to install OctoMap system libraries
- 📦 **Self-Contained**: All libraries bundled in the wheel
- 🔄 **Cross-Platform**: Works on any system with compatible Python version
- 🛠️ **Easy Distribution**: Can be uploaded to PyPI

**Troubleshooting:**
- If the script fails, check that you have Python 3.9+ installed
- Ensure you have `pip` and `python3` available in your PATH
- If Cython compilation fails, try updating your compiler toolchain
- If library bundling fails, ensure auditwheel (Linux) or delocate (macOS) is installed
- **Memory corruption on exit**: If you see "double free or corruption" messages, this is a known issue with C++ libraries in Python. The package works correctly, but to avoid these messages, add `import os; os._exit(0)` at the end of your scripts

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- **Previous work**: [`wkentaro/octomap-python`](https://github.com/wkentaro/octomap-python) - This project builds upon and modernizes the original Python bindings
- **Core library**: [OctoMap](https://octomap.github.io/) - An efficient probabilistic 3D mapping framework based on octrees
- **Build system**: Built with Cython for seamless Python-C++ integration and performance
- **Visualization**: [Open3D](https://www.open3d.org/) - Used for 3D visualization capabilities in demonstration scripts
- **Research support**: Development of this enhanced Python wrapper was supported by the French National Research Agency (ANR) under the France 2030 program, specifically the IRT Nanoelec project (ANR-10-AIRT-05), advancing robotics and 3D mapping research capabilities.

## Contact

- Author: Spinkoo

## Future Directions

- Publish Linux wheels to PyPI, with zero external system dependencies
- Add Windows wheels (next) and improve cross-platform support
- Expand high-level Pythonic API surface and iterator stability
- More visualization examples and integration guides