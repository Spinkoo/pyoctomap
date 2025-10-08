# PyOctoMap

<div align="center">
<img src="images/octomap_core.png" alt="OctoMap Core" width="900">
</div>

A comprehensive Python wrapper for the OctoMap C++ library, providing efficient 3D occupancy mapping capabilities for robotics and computer vision applications. This modernized binding offers enhanced performance, bundled shared libraries for easy deployment, and seamless integration with the Python scientific ecosystem.

## Features

- **3D Occupancy Mapping**: Efficient octree-based 3D occupancy mapping
- **Probabilistic Updates**: Stochastic occupancy updates with uncertainty handling
- **Path Planning**: Ray casting and collision detection
- **File Operations**: Save/load octree data in binary format
- **Python Integration**: Clean Python interface with NumPy support
- **Cross-Platform**: Linux native support with Windows compatibility via WSL

## Installation

### Quick Install (Recommended)

For most users, simply install the pre-built wheel:

```bash
pip install pyoctomap
```

**Supported Platforms:**
- Linux (manylinux2014 compatible)
- Python 3.9, 3.10, 3.11, 3.12
- Pre-built wheels available for all supported combinations

> **🚀 ROS Integration**: ROS/ROS2 integration is currently being developed on the [`ros` branch](https://github.com/Spinkoo/pyoctomap/tree/ros), featuring ROS2 message support and real-time point cloud processing.

### Building from Source

If you need to build from source or create custom wheels, we provide a Docker-based build system:

**Linux / WSL (Windows Subsystem for Linux):**
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/Spinkoo/pyoctomap.git
cd pyoctomap

# Build and install OctoMap C++ library
cd src/octomap
mkdir build && cd build
cmake .. && make && sudo make install

# Return to main project and run automated build script
cd ../../..
chmod +x build.sh
./build.sh
```

```bash
# Build wheels for all supported Python versions
./build-wheel.sh

# Or build manually with Docker
docker build -f docker/Dockerfile.wheel -t pyoctomap-wheel .
```

The Docker build creates manylinux-compatible wheels for Python 3.9-3.12, properly bundling all required C++ libraries.

## Quick Start

### Basic Usage

```python
import pyoctomap
import numpy as np

# Create an octree with 0.1m resolution
tree = pyoctomap.OcTree(0.1)

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

PyOctoMap now includes high-performance vectorized operations for better performance:

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

### Batch Operations

For efficient batch processing of point clouds, PyOctoMap provides both precise and fast options:

**Precise Batch Ray-Casting:**
```python
# Accurate batch insertion with ray-casting (hit-stopping for free/occupied)
points = np.random.uniform(-5, 5, (1000, 3))
origin = np.array([0., 0., 0.], dtype=np.float64)
success_count = tree.addPointCloudWithRayCasting(points, origin, discretize=True)
# discretize=True reduces duplicates for dense clouds
```

**Fast Native Batching:**
```python
# Fast C++ batch (full rays, optional discretization and lazy evaluation)
points = np.random.uniform(-5, 5, (1000, 3))
origin = np.array([0., 0., 0.], dtype=np.float64)
tree.insertPointCloudFast(points, origin, discretize=False, lazy_eval=True)
tree.updateInnerOccupancy()  # Manual after lazy

# Equivalent raw C++ access (identical to above)
tree.insertPointCloud(points, origin, discretize=False, lazy_eval=True)
tree.updateInnerOccupancy()
```

Note: `insertPointCloud` and `insertPointCloudFast` share the same underlying logic (consolidated for efficiency) – use either for native batching. For precision-critical tasks, prefer `addPointCloudWithRayCasting`; for speed, use natives with `lazy_eval=True` for deferral. All support NumPy arrays and `max_range` clipping.

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
- `examples/sequential_occupancy_grid_demo.py` — comprehensive sequential occupancy grid with Open3D visualization
- `examples/test_sequential_occupancy_grid.py` — comprehensive test suite for all occupancy grid methods

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
import pyoctomap
import numpy as np

# Create octree
tree = pyoctomap.OcTree(0.05)  # 5cm resolution
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
import pyoctomap
import numpy as np

# Create an octree for path planning
tree = pyoctomap.OcTree(0.1)  # 10cm resolution

# Add some obstacles to the map
obstacles = [
    [1.0, 1.0, 0.5],  # Wall at (1,1)
    [1.5, 1.5, 0.5],  # Another obstacle
    [2.0, 1.0, 0.5],  # Wall at (2,1)
]

for obstacle in obstacles:
    tree.updateNode(obstacle, True)

def is_path_clear(start, end, tree):
    """Efficient ray casting for path planning using OctoMap's built-in castRay"""
    start = np.array(start, dtype=np.float64)
    end = np.array(end, dtype=np.float64)
    
    # Calculate direction vector
    direction = end - start
    ray_length = np.linalg.norm(direction)
    
    if ray_length == 0:
        return True, None
    
    # Normalize direction
    direction = direction / ray_length
    
    # Use OctoMap's efficient castRay method
    end_point = np.zeros(3, dtype=np.float64)
    hit = tree.castRay(start, direction, end_point, 
                      ignoreUnknownCells=True, 
                      maxRange=ray_length)
    
    if hit:
        # Ray hit an obstacle - path is blocked
        return False, end_point
    else:
        # No obstacle found - path is clear
        return True, None

# Check if path is clear
start = [0.5, 2.0, 0.5]
end = [2.0, 2.0, 0.5]
clear, obstacle = is_path_clear(start, end, tree)
if clear:
    print("✅ Path is clear!")
else:
    print(f"❌ Path blocked at: {obstacle}")

# Advanced path planning with multiple waypoints
def plan_path(waypoints, tree):
    """Plan a path through multiple waypoints using ray casting"""
    path_clear = True
    obstacles = []
    
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        clear, obstacle = is_path_clear(start, end, tree)
        
        if not clear:
            path_clear = False
            obstacles.append((i, i+1, obstacle))
    
    return path_clear, obstacles

# Example: Plan path through multiple waypoints
waypoints = [
    [0.0, 0.0, 0.5],
    [1.0, 1.0, 0.5], 
    [2.0, 2.0, 0.5],
    [3.0, 3.0, 0.5]
]

path_clear, obstacles = plan_path(waypoints, tree)
if path_clear:
    print("✅ Complete path is clear!")
else:
    print(f"❌ Path blocked at segments: {obstacles}")
```

### Iterator Operations

PyOctoMap provides three types of iterators for different use cases:

#### Tree Iterator (`begin_tree`) - All Nodes
```python
# Iterate over ALL nodes (inner + leaf nodes) - slower but complete
for node_it in tree.begin_tree():
    coord = node_it.getCoordinate()
    depth = node_it.getDepth()
    size = node_it.getSize()
    is_leaf = node_it.isLeaf()
    
    # Use for: tree structure analysis, debugging, inner node operations
    if not is_leaf:
        print(f"Inner node at depth {depth}, size {size:.2f}m")
```

#### Leaf Iterator (`begin_leafs`) - Leaf Nodes Only  
```python
# Iterate over LEAF nodes only - faster for occupancy queries
for leaf_it in tree.begin_leafs():
    coord = leaf_it.getCoordinate()
    occupied = tree.isNodeOccupied(leaf_it)
    if occupied:
        print(f"Occupied leaf at {coord}")
    
    # Use for: standard occupancy operations, fast iteration
```

#### Bounding Box Iterator (`begin_leafs_bbx`) - Spatial Filtering
```python
# Iterate over leaf nodes within a bounding box
bbx_min = np.array([0.0, 0.0, 0.0])
bbx_max = np.array([5.0, 5.0, 5.0])
for bbx_it in tree.begin_leafs_bbx(bbx_min, bbx_max):
    coord = bbx_it.getCoordinate()
    print(f"Node in BBX: {coord}")
    
    # Use for: region-specific analysis, spatial queries
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
- **Core library**: [OctoMap](https://OctoMap.github.io) - An efficient probabilistic 3D mapping framework based on octrees
- **Build system**: Built with Cython for seamless Python-C++ integration and performance
- **Visualization**: [Open3D](https://www.open3d.org/) - Used for 3D visualization capabilities in demonstration scripts
- **Research support**: Development of this enhanced Python wrapper was supported by the French National Research Agency (ANR) under the France 2030 program, specifically the IRT Nanoelec project (ANR-10-AIRT-05), advancing robotics and 3D mapping research capabilities.