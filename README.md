# PyOctoMap - ROS Integration

<div align="center">
<img src="images/octomap_core.png" alt="OctoMap Core" width="900">
</div>

**A ROS-optimized Python wrapper for OctoMap** - the industry-standard 3D occupancy mapping library. Designed specifically for robotics applications, this package provides seamless integration with ROS/ROS2 for real-time 3D mapping, SLAM, and navigation.

## Why PyOctoMap for ROS?

- **🚀 Real-time Performance**: Optimized for robotics with vectorized operations (4x faster than traditional approaches)
- **🗺️ 3D Occupancy Mapping**: Efficient octree-based mapping perfect for robot navigation and SLAM
- **📡 ROS2 Integration**: Native support for PointCloud2, OccupancyGrid, and other ROS message types
- **⚡ Ray Casting**: Built-in ray casting for proper free space marking in 3D environments
- **🔧 Zero Dependencies**: Bundled C++ libraries - no complex setup required
- **🐍 Python Native**: Clean Python interface with NumPy support for rapid prototyping

## Key Features for Robotics

- **3D Occupancy Mapping**: Efficient octree-based 3D occupancy mapping
- **ROS2 Message Integration**: Direct support for PointCloud2, OccupancyGrid, PoseStamped
- **Ray Casting**: Automatic free space marking for proper 3D mapping
- **Vectorized Operations**: 4x faster point cloud processing for real-time applications
- **File Operations**: Save/load octree data in binary format (.bt files)
- **Bundled Libraries**: No external C++ dependencies - works out of the box

## Installation

### Quick Install (Recommended)

```bash
pip install pyoctomap
```

**Supported Platforms:**
- Linux (manylinux2014 compatible)
- Python 3.9, 3.10, 3.11, 3.12

### ROS/ROS2 Integration

PyOctoMap is designed to work seamlessly with ROS (Robot Operating System):

#### ROS2 (Recommended)
```bash
# Install pyoctomap
pip install pyoctomap

# Install ROS2 dependencies via ROS package manager
sudo apt install ros-iron-rclpy ros-iron-sensor-msgs ros-iron-geometry-msgs ros-iron-nav-msgs ros-iron-tf2-ros

# Source your ROS2 workspace
source /opt/ros/iron/setup.bash
```

#### ROS1
```bash
# Install pyoctomap
pip install pyoctomap

# Install ROS1 dependencies via ROS package manager
sudo apt install ros-noetic-rclpy ros-noetic-sensor-msgs ros-noetic-geometry-msgs ros-noetic-nav-msgs ros-noetic-tf2-ros

# Source your ROS1 workspace
source /opt/ros/noetic/setup.bash
```

## Quick Start

### ROS2 Integration (Recommended)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import pyoctomap
import numpy as np

class OctoMapNode(Node):
    def __init__(self):
        super().__init__('octomap_node')
        self.octree = pyoctomap.OcTree(0.1)  # 10cm resolution
        
        # Subscribe to point cloud data
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.pointcloud_callback,
            10
        )
    
    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points = self.pointcloud2_to_array(msg)
        
        # Add points with ray casting for proper 3D mapping
        sensor_origin = np.array([0.0, 0.0, 1.5])
        success_count = self.octree.addPointCloudWithRayCasting(points, sensor_origin)
        
        self.get_logger().info(f'Added {success_count} points to octree')

def main(args=None):
    rclpy.init(args=args)
    node = OctoMapNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Basic Usage (Non-ROS)

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

### Basic Usage Examples

See runnable demos in `examples/`:
- `examples/basic_test.py` — smoke test for core API
- `examples/demo_occupancy_grid.py` — build and visualize a 2D occupancy grid
- `examples/demo_octomap_open3d.py` — visualize octomap data with Open3D
- `examples/sequential_occupancy_grid_demo.py` — comprehensive sequential occupancy grid with Open3D visualization
- `examples/test_sequential_occupancy_grid.py` — comprehensive test suite for all occupancy grid methods


## Requirements

- Python 3.9+
- NumPy

**For ROS integration:**
- ROS2: `ros-iron-rclpy`, `ros-iron-sensor-msgs`, `ros-iron-geometry-msgs`, `ros-iron-nav-msgs`, `ros-iron-tf2-ros`
- ROS1: `ros-noetic-rclpy`, `ros-noetic-sensor-msgs`, `ros-noetic-geometry-msgs`, `ros-noetic-nav-msgs`, `ros-noetic-tf2-ros`
- Install via ROS package manager (not pip)

## License

MIT License - see [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- **Core library**: [OctoMap](https://octomap.github.io) - An efficient probabilistic 3D mapping framework based on octrees
- **Previous work**: [`wkentaro/octomap-python`](https://github.com/wkentaro/octomap-python) - This project builds upon and modernizes the original Python bindings