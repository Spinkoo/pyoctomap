# OctoMap2Python

<div align="center">
<img src="images/octomap_core.png" alt="OctoMap Core" width="900">
</div>

A comprehensive Python wrapper for the OctoMap C++ library, providing efficient 3D occupancy mapping capabilities for robotics and computer vision applications. This modernized binding offers enhanced performance, bundled shared libraries for easy deployment, and seamless integration with the Python scientific ecosystem.

## Features

- **3D Occupancy Mapping**: Efficient octree-based 3D occupancy mapping
- **Probabilistic Updates**: Stochastic occupancy updates with uncertainty handling
- **Path Planning**: Ray casting and collision detection
- **File Operations**: Save/load octree data in binary format
- **Bundled Libraries**: No external dependencies - all C++ libraries included
- **Python Integration**: Clean Python interface with NumPy support
- **Cross-Platform**: Linux native support with Windows compatibility via WSL

## Installation

**Linux / WSL (Windows Subsystem for Linux):**
```bash
# Install system dependencies (ensure compatible libstdc++ version)
sudo apt-get update
sudo apt-get install libstdc++6 cmake build-essential -y

# Clone the repository with submodules
git clone --recursive https://github.com/Spinkoo/octomap2python.git
cd octomap2python

# Build and install OctoMap C++ library
cd src/octomap
mkdir build && cd build
cmake .. && make && sudo make install

# Return to main project and run automated build script
cd ../../..
chmod +x build.sh
./build.sh
```

When published on PyPI (Linux):
```bash
# Basic installation
pip install octomap2python

# Install with visualization tools (includes matplotlib + Open3D)
pip install octomap2python[visualization]
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
# Install system dependencies first
sudo apt-get update
sudo apt-get install libstdc++6 cmake build-essential -y

# Clone the repository
git clone https://github.com/Spinkoo/octomap2python.git
cd octomap2python

# Install Python dependencies
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
# Option 1: Install extras (recommended)
pip install octomap2python[visualization]

# Option 2: Install manually
pip install matplotlib  # For 2D occupancy grid visualization
pip install open3d      # For 3D visualization with Open3D

# Option 3: Install both at once
pip install matplotlib open3d
```
- For building Open3D from source, **Windows GPU builds using WSL, see [Open3D on WSL](https://github.com/Spinkoo/Open3DWSL).**

### Requirements

**System Dependencies:**
- `libstdc++6` - Modern C++ standard library (fixes glibc compatibility issues)
- `cmake` - Build system for OctoMap C++ library
- `build-essential` - GCC compiler and build tools

**Python Dependencies:**
- Python 3.9+
- setuptools
- NumPy
- Cython (for building from source)
- auditwheel (Linux) for library bundling

**Optional for visualization:**
- matplotlib (for 2D plotting and occupancy grids)
- open3d (for 3D visualization)

## Quick Start

### Basic Usage

```python
import octomap
import numpy as np

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
```

### New Vectorized Operations

OctoMap2Python now includes high-performance vectorized operations for better performance:

#### Traditional vs Vectorized Approach

**Traditional (slower):**
```python
# Individual point updates - slower
points = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
for point in points:
    tree.updateNode(point, True)
```

**Vectorized (faster):**
```python
# Batch point updates - 4-5x faster
points = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
tree.addPointsBatch(points)
```

#### Ray Casting with Free Space Marking

**Single Point with Ray Casting:**
```python
# Add point with automatic free space marking
sensor_origin = np.array([0.0, 0.0, 1.5])
point = np.array([2.0, 2.0, 1.0])
tree.addPointWithRayCasting(point, sensor_origin)
```

**Point Cloud with Ray Casting:**
```python
# Add point cloud with ray casting for each point
point_cloud = np.random.rand(1000, 3) * 10
sensor_origin = np.array([0.0, 0.0, 1.5])
success_count = tree.addPointCloudWithRayCasting(point_cloud, sensor_origin)
print(f"Added {success_count} points")
```

#### Batch Operations

**Batch Points with Same Origin:**
```python
# Efficient batch processing
points = np.random.rand(5000, 3) * 10
sensor_origin = np.array([0.0, 0.0, 1.5])
success_count = tree.addPointsBatch(points, update_inner_occupancy=True)
print(f"Added {success_count} points in batch")
```

**Batch Points with Different Origins:**
```python
# Each point can have different sensor origin
points = np.random.rand(100, 3) * 10
origins = np.random.rand(100, 3) * 2
success_count = tree.addPointsBatch(points, origins)
print(f"Added {success_count} points with individual origins")
```

### Performance Comparison

| Operation | Traditional | Vectorized | Speedup |
|-----------|-------------|------------|---------|
| Individual points | 5,000 pts/sec | 20,000 pts/sec | 4x |
| Point cloud | 10,000 pts/sec | 30,000 pts/sec | 3x |
| Batch processing | 15,000 pts/sec | 60,000 pts/sec | 4x |

## Examples

See runnable demos in `examples/`:
- `examples/basic_test.py` — smoke test for core API
- `examples/demo_occupancy_grid.py` — build and visualize a 2D occupancy grid
- `examples/demo_octomap_open3d.py` — visualize octomap data with Open3D

### Demo Visualizations

**3D OctoMap Scene Visualization:**
<div align="center">
<img src="images/octomap_demo_scene.png" alt="OctoMap Demo Scene" width="700">
</div>

**Occupancy Grid Visualization:**
<div align="center">
<img src="images/occupancy_grid.png" alt="Occupancy Grid" width="700">
</div>

## Advanced Usage

### Room Mapping with Ray Casting

```python
import octomap
import numpy as np

# Create octree
tree = octomap.OcTree(0.05)  # 5cm resolution
sensor_origin = np.array([2.0, 2.0, 1.5])

# Add walls with ray casting
wall_points = []
for x in np.arange(0, 4.0, 0.05):
    for y in np.arange(0, 4.0, 0.05):
        wall_points.append([x, y, 0])  # Floor
        wall_points.append([x, y, 3.0])  # Ceiling

# Use vectorized approach for better performance
wall_points = np.array(wall_points)
tree.addPointCloudWithRayCasting(wall_points, sensor_origin)

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

### Iterator Operations

```python
# Iterate over all nodes
for node_it in tree.begin_tree():
    coord = node_it.getCoordinate()
    depth = node_it.getDepth()
    size = node_it.getSize()
    is_leaf = node_it.isLeaf()

# Iterate over leaf nodes only
for leaf_it in tree.begin_leafs():
    coord = leaf_it.getCoordinate()
    occupied = tree.isNodeOccupied(leaf_it)
    if occupied:
        print(f"Occupied leaf at {coord}")

# Iterate over bounding box
bbx_min = np.array([0.0, 0.0, 0.0])
bbx_max = np.array([5.0, 5.0, 5.0])
for bbx_it in tree.begin_leafs_bbx(bbx_min, bbx_max):
    coord = bbx_it.getCoordinate()
    print(f"Node in BBX: {coord}")
```

## Requirements

- Python 3.9+
- NumPy
- Cython (for building from source)

**Optional for visualization:**
- matplotlib (for 2D plotting)
- open3d (for 3D visualization)

## Documentation

- **[Complete API Reference](docs/api_reference.md)** - Detailed API documentation
- **[File Format Guide](docs/file_format.md)** - Supported file formats
- **[Performance Guide](docs/performance_guide.md)** - Optimization tips and benchmarks
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Build System](docs/build_system.md)** - Build process and scripts
- **[Wheel Technology](docs/wheel_technology.md)** - Library bundling details

## License

MIT License - see [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- **Previous work**: [`wkentaro/octomap-python`](https://github.com/wkentaro/octomap-python) - This project builds upon and modernizes the original Python bindings
- **Core library**: [OctoMap](https://octomap.github.io/) - An efficient probabilistic 3D mapping framework based on octrees
- **Build system**: Built with Cython for seamless Python-C++ integration and performance
- **Visualization**: [Open3D](https://www.open3d.org/) - Used for 3D visualization capabilities in demonstration scripts
- **Research support**: Development of this enhanced Python wrapper was supported by the French National Research Agency (ANR) under the France 2030 program, specifically the IRT Nanoelec project (ANR-10-AIRT-05), advancing robotics and 3D mapping research capabilities.