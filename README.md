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
- **Cross-Platform**: Pre-built wheels for Linux (`x86_64`) and macOS (Apple Silicon `arm64`), with Windows compatibility via WSL

## Installation

### Quick Install (Recommended)

Install from PyPI (pre-built manylinux wheel when available):

```bash
pip install pyoctomap
```

> **🚀 ROS Integration**: ROS/ROS2 integration is currently being developed on the [`ros` branch](https://github.com/Spinkoo/pyoctomap/tree/ros), featuring ROS2 message support and real-time point cloud processing.

### Building from Source

> **📋 Prerequisites**: See [Build System Documentation](docs/build_system.md) for detailed system dependencies and troubleshooting guide.

If you need to build from source or create custom wheels, we provide a Docker-based build system:

**Linux / WSL (Windows Subsystem for Linux):**
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/Spinkoo/pyoctomap.git
cd pyoctomap

chmod +x build.sh
./build.sh
```

```bash
# Build wheels for all supported Python versions
./build-wheel.sh

# Or build manually with Docker
docker build -f docker/Dockerfile.wheel -t pyoctomap-wheel .
```

The Docker build creates manylinux-compatible wheels for Python 3.8-3.14, properly bundling all required C++ libraries.

> **📋 Google Colab Users**: See [Build System Documentation](docs/build_system.md) for detailed Colab installation instructions.

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

### Tree Families Overview

PyOctoMap provides multiple octree variants from a single package:

- `OcTree` – standard probabilistic occupancy tree (most users start here)
- `ColorOcTree` – occupancy + RGB color per voxel
- `CountingOcTree` – integer hit counters per voxel
- `OcTreeStamped` – occupancy with per-node timestamps for temporal mapping

See the **[API Reference](docs/api_reference.md)** for a detailed comparison
table and full method documentation.

### Color Occupancy Mapping (ColorOcTree)

```python
import pyoctomap
import numpy as np

tree = pyoctomap.ColorOcTree(0.1)
coord = [1.0, 1.0, 1.0]

tree.updateNode(coord, True)
tree.setNodeColor(coord, 255, 0, 0)  # R, G, B (0-255)
```

**Batch insertion with colors:**
```python
# Insert point cloud with colors in a single operation
points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)  # RGB in [0, 1] range
sensor_origin = np.array([0.0, 0.0, 0.0])  # Optional: for proper ray casting
tree.insertPointCloud(points, sensor_origin=sensor_origin, colors=colors)
```



### Batch Operations (Summary)

For large point clouds, use the unified `insertPointCloud` method:

- `OcTree.insertPointCloud(points, origin, max_range=-1.0, lazy_eval=False, discretize=False)`
- `ColorOcTree.insertPointCloud(points, sensor_origin=None, ..., colors=colors)` — also sets per-point colors
- `OcTreeStamped.insertPointCloud(points, sensor_origin=None, ..., timestamps=ts)` — also sets per-node timestamps

See the **Performance Guide** for practical batch sizing and resolution
recommendations.

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

## Showcase

### 🎨 Photo to 3D Voxel Scene

**[pyocto-map-anything](https://github.com/Spinkoo/pyocto-map-anything)** - Transform single photos into vibrant 3D voxel scenes using AI depth estimation (Depth Anything 3) and PyOctoMap's `ColorOcTree`. This showcase demonstrates the power of combining modern depth estimation models with efficient octree-based mapping, enabling instant 3D reconstruction from 2D images.

**Features:**
- AI-powered depth estimation from single images
- Automatic camera intrinsics estimation (DA3 models)
- Color-integrated voxel mapping with `ColorOcTree`
- Support for multiple depth models (Depth Anything v3, ZoeDepth, DPT)
- High-resolution 3D reconstruction with configurable voxel resolution

Perfect for exploring PyOctoMap's color mapping capabilities and seeing how it integrates with modern computer vision pipelines.

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

# Use batch insertion for better performance
wall_points = np.array(wall_points)
tree.insertPointCloud(wall_points, sensor_origin, lazy_eval=True)
tree.updateInnerOccupancy()

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

### Dynamic Environment Mapping & Iterators

For more complete examples on:

- dynamic environment mapping,
- iterator usage (`begin_tree`, `begin_leafs`, `begin_leafs_bbx`),

refer to the **API Reference** and example scripts in `examples/`.

## Requirements

- Python 3.8+
- NumPy
- Cython (for building from source)

**Optional for visualization:**
- matplotlib (for 2D plotting)
- open3d (for 3D visualization)

## Documentation

- **[Complete API Reference](docs/api_reference.md)** - Detailed API documentation
- **[Build System](docs/build_system.md)** - Prerequisites, build process, and troubleshooting
- **[File Format Guide](docs/file_format.md)** - Supported file formats
- **[Performance Guide](docs/performance_guide.md)** - Optimization tips and benchmarks
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
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