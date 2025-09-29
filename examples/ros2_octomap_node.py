#!/usr/bin/env python3
"""
ROS2 OctoMap Node Example

This example demonstrates how to use pyoctomap with ROS2 for real-time 3D mapping.
It subscribes to PointCloud2 messages and builds an OctoMap, then publishes
occupancy grid data.

Usage:
    ros2 run pyoctomap ros2_octomap_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import pyoctomap
import numpy as np
import struct


class OctoMapNode(Node):
    def __init__(self):
        super().__init__('octomap_node')
        
        # OctoMap parameters
        self.resolution = 0.1  # 10cm resolution
        self.octree = pyoctomap.OcTree(self.resolution)
        
        # ROS2 parameters
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('sensor_origin_x', 0.0)
        self.declare_parameter('sensor_origin_y', 0.0)
        self.declare_parameter('sensor_origin_z', 1.5)
        
        # Get parameters
        self.resolution = self.get_parameter('resolution').value
        self.max_range = self.get_parameter('max_range').value
        self.sensor_origin = np.array([
            self.get_parameter('sensor_origin_x').value,
            self.get_parameter('sensor_origin_y').value,
            self.get_parameter('sensor_origin_z').value
        ])
        
        # Recreate octree with new resolution
        self.octree = pyoctomap.OcTree(self.resolution)
        
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
        
        self.octomap_pub = self.create_publisher(
            PoseStamped,
            '/octomap/pose',
            10
        )
        
        # Timer for periodic publishing
        self.timer = self.create_timer(1.0, self.publish_occupancy_grid)
        
        self.get_logger().info(f'OctoMap node started with resolution: {self.resolution}m')
    
    def pointcloud2_to_array(self, cloud_msg):
        """Convert ROS2 PointCloud2 message to numpy array"""
        # Get the point cloud data
        cloud_data = cloud_msg.data
        
        # Calculate number of points
        point_step = cloud_msg.point_step
        num_points = len(cloud_data) // point_step
        
        # Extract x, y, z coordinates (assuming they are the first 3 floats)
        points = []
        for i in range(num_points):
            offset = i * point_step
            x = struct.unpack('f', cloud_data[offset:offset+4])[0]
            y = struct.unpack('f', cloud_data[offset+4:offset+8])[0]
            z = struct.unpack('f', cloud_data[offset+8:offset+12])[0]
            
            # Filter out invalid points
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z) or 
                   np.isinf(x) or np.isinf(y) or np.isinf(z)):
                points.append([x, y, z])
        
        return np.array(points)
    
    def pointcloud_callback(self, msg):
        """Process incoming point cloud data"""
        try:
            # Convert PointCloud2 to numpy array
            points = self.pointcloud2_to_array(msg)
            
            if len(points) == 0:
                self.get_logger().warn('Received empty point cloud')
                return
            
            # Add points to octree with ray casting
            success_count = self.octree.addPointCloudWithRayCasting(
                points, 
                self.sensor_origin,
                maxRange=self.max_range
            )
            
            self.get_logger().info(f'Added {success_count}/{len(points)} points to octree')
            
            # Publish octree statistics
            self.publish_octree_stats()
            
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {str(e)}')
    
    def publish_octree_stats(self):
        """Publish octree statistics"""
        stats_msg = PoseStamped()
        stats_msg.header.stamp = self.get_clock().now().to_msg()
        stats_msg.header.frame_id = 'map'
        
        # Use pose message to carry statistics
        stats_msg.pose.position.x = float(self.octree.size())
        stats_msg.pose.position.y = float(self.octree.getResolution())
        stats_msg.pose.position.z = 0.0
        
        self.octomap_pub.publish(stats_msg)
    
    def publish_occupancy_grid(self):
        """Convert octree to 2D occupancy grid and publish"""
        try:
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = self.get_clock().now().to_msg()
            grid_msg.header.frame_id = 'map'
            
            # Grid parameters
            grid_size = 100  # 100x100 grid
            grid_resolution = 0.1  # 10cm per cell
            grid_origin_x = -5.0  # -5m to +5m
            grid_origin_y = -5.0
            
            grid_msg.info.resolution = grid_resolution
            grid_msg.info.width = grid_size
            grid_msg.info.height = grid_size
            grid_msg.info.origin.position.x = grid_origin_x
            grid_msg.info.origin.position.y = grid_origin_y
            grid_msg.info.origin.position.z = 0.0
            grid_msg.info.origin.orientation.w = 1.0
            
            # Fill occupancy data from octree
            occupancy_data = []
            for y in range(grid_size):
                for x in range(grid_size):
                    world_x = x * grid_resolution + grid_origin_x
                    world_y = y * grid_resolution + grid_origin_y
                    world_z = 0.5  # Fixed height for 2D grid
                    
                    node = self.octree.search([world_x, world_y, world_z])
                    if node and self.octree.isNodeOccupied(node):
                        occupancy_data.append(100)  # Occupied
                    else:
                        occupancy_data.append(0)    # Free
            
            grid_msg.data = occupancy_data
            self.occupancy_pub.publish(grid_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing occupancy grid: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = OctoMapNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
