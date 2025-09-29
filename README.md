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
- **Bundled Libraries**: No external dependencies - all C++ libraries included
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

**Note:** Building from source is not recommended for most users as it requires compiling C++ code and managing dependencies.

### ROS/ROS2 Integration

PyOctoMap is designed to work seamlessly with ROS (Robot Operating System):

#### ROS2 (Recommended)
```bash
# Install pyoctomap
pip install pyoctomap

# Install ROS2 dependencies (optional)
pip install pyoctomap[ros]
```

#### ROS1
```bash
# Install pyoctomap
pip install pyoctomap

# For ROS1, you may need to install additional dependencies
pip install sensor_msgs geometry_msgs nav_msgs tf2_ros
```

#### ROS Workspace Integration
```bash
# In your ROS workspace
cd ~/ros2_ws/src
git clone https://github.com/Spinkoo/pyoctomap.git
cd ~/ros2_ws
colcon build --packages-select pyoctomap
```

### For Developers

If you need to build from source or create custom wheels, we provide a Docker-based build system:

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

### ROS2 Integration Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import pyoctomap
import numpy as np

class OctoMapNode(Node):
    def __init__(self):
        super().__init__('octomap_node')
        self.octree = pyoctomap.OcTree(0.1)  # 10cm resolution
        
        # ROS2 subscribers and publishers
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.pointcloud_callback,
            10
        )
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid,
            '/octomap/occupancy_grid',
            10
        )
        
    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points = self.pointcloud2_to_array(msg)
        
        # Add points to octree with ray casting
        sensor_origin = np.array([0.0, 0.0, 1.5])
        success_count = self.octree.addPointCloudWithRayCasting(points, sensor_origin)
        
        self.get_logger().info(f'Added {success_count} points to octree')
        
        # Publish occupancy grid
        self.publish_occupancy_grid()
    
    def publish_occupancy_grid(self):
        # Convert octree to 2D occupancy grid
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = 'map'
        grid_msg.info.resolution = 0.1
        grid_msg.info.width = 100
        grid_msg.info.height = 100
        
        # Fill occupancy data from octree
        occupancy_data = []
        for y in range(100):
            for x in range(100):
                world_x = x * 0.1 - 5.0
                world_y = y * 0.1 - 5.0
                world_z = 0.5  # Fixed height
                
                node = self.octree.search([world_x, world_y, world_z])
                if node and self.octree.isNodeOccupied(node):
                    occupancy_data.append(100)  # Occupied
                else:
                    occupancy_data.append(0)    # Free
        
        grid_msg.data = occupancy_data
        self.occupancy_pub.publish(grid_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OctoMapNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Standard Examples

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

**Optional for ROS integration:**
- rclpy (ROS2 Python client)
- sensor_msgs, geometry_msgs, nav_msgs (ROS message types)
- tf2_ros (ROS2 transforms)

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