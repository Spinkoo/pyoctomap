#!/usr/bin/env python3
"""
Sequential Occupancy Grid Manager

This module provides a function that takes 3D points sequentially to update 
an occupancy grid and returns points formatted for Open3D visualization.

The main class SequentialOccupancyGrid manages:
- Sequential updates to an OctoMap-based occupancy grid
- Sensor origin handling for ray casting
- Point cloud extraction for Open3D visualization
- Color-coded visualization (occupied, free, uncertain)
- Performance optimization for real-time updates
"""

import numpy as np
import sys
import os
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import time

# Add parent directory to path for proper import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pyoctomap
    print("✅ PyOctoMap import successful!")
except ImportError as e:
    print(f"❌ Failed to import pyoctomap: {e}")
    sys.exit(1)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("✅ Open3D available for visualization")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️ Open3D not available - visualization will be limited")


@dataclass
class VisualizationConfig:
    """Configuration for visualization colors and point sizes"""
    occupied_color: List[float] = None  # RGB [0-1]
    free_color: List[float] = None      # RGB [0-1] 
    uncertain_color: List[float] = None # RGB [0-1]
    point_size: float = 6.0
    use_height_coloring: bool = True
    show_free_space: bool = True
    show_uncertain: bool = True
    
    def __post_init__(self):
        if self.occupied_color is None:
            self.occupied_color = [0.8, 0.2, 0.2]  # Red
        if self.free_color is None:
            self.free_color = [0.2, 0.8, 0.2]      # Green
        if self.uncertain_color is None:
            self.uncertain_color = [0.8, 0.8, 0.2] # Yellow


@dataclass
class MappingStats:
    """Statistics about the occupancy mapping process"""
    total_points_processed: int = 0
    occupied_cells: int = 0
    free_cells: int = 0
    uncertain_cells: int = 0
    tree_size: int = 0
    processing_time: float = 0.0
    last_update_time: float = 0.0


class SequentialOccupancyGrid:
    """
    Sequential Occupancy Grid Manager
    
    This class manages a 3D occupancy grid using OctoMap and provides
    functionality for sequential point updates and Open3D visualization.
    """
    
    def __init__(self, 
                 resolution: float = 0.05,
                 sensor_origin: Union[List[float], np.ndarray] = None,
                 prob_hit: float = 0.7,
                 prob_miss: float = 0.3,
                 clamping_thres_min: float = 0.12,
                 clamping_thres_max: float = 0.97):
        """
        Initialize the Sequential Occupancy Grid
        
        Args:
            resolution: Voxel resolution in meters
            sensor_origin: Default sensor origin [x, y, z] in meters
            prob_hit: Probability for occupied cells (0.5-1.0)
            prob_miss: Probability for free cells (0.0-0.5) 
            Note: prob_hit + prob_miss should be ≤ 1.0 
            clamping_thres_min: Minimum probability threshold
            clamping_thres_max: Maximum probability threshold
        """
        self.resolution = resolution
        self.sensor_origin = np.array(sensor_origin if sensor_origin is not None else [0.0, 0.0, 0.0])
        
        # Validate probability values
        if prob_hit + prob_miss > 1.0:
            print(f"⚠️  Warning: prob_hit ({prob_hit}) + prob_miss ({prob_miss}) = {prob_hit + prob_miss} > 1.0")
            print(f"    This may cause unexpected behavior. Consider adjusting values.")
        
        # Initialize OctoMap
        self.tree = pyoctomap.OcTree(resolution)
        
        # Set probabilities
        self.tree.setProbHit(prob_hit)
        self.tree.setProbMiss(prob_miss)
        self.tree.setClampingThresMin(clamping_thres_min)
        self.tree.setClampingThresMax(clamping_thres_max)
        
        # Visualization configuration
        self.viz_config = VisualizationConfig()
        
        # Statistics
        self.stats = MappingStats()
        
        # Cached visualization data
        self._cached_points = None
        self._cache_invalid = True
        
        print(f"��️  Initialized SequentialOccupancyGrid:")
        print(f"    Resolution: {resolution}m")
        print(f"    Sensor origin: {self.sensor_origin}")
        print(f"    Prob hit/miss: {prob_hit}/{prob_miss}")
    
    def update_sensor_origin(self, origin: Union[List[float], np.ndarray]):
        """Update the sensor origin for ray casting"""
        self.sensor_origin = np.array(origin)
        
    def add_point(self, 
                  point: Union[List[float], np.ndarray], 
                  sensor_origin: Optional[Union[List[float], np.ndarray]] = None,
                  update_inner_occupancy: bool = False) -> bool:
        """
        Add a single 3D point to update the occupancy grid
        
        Args:
            point: 3D point [x, y, z] in meters
            sensor_origin: Optional sensor origin for this measurement
            update_inner_occupancy: Whether to update inner node occupancy (expensive)
            
        Returns:
            bool: True if point was successfully added
        """
        start_time = time.time()
        
        try:
            point = np.array(point, dtype=np.float64)
            origin = np.array(sensor_origin if sensor_origin is not None else self.sensor_origin, dtype=np.float64)
            
            # Use the new integrated method from octomap
            success = self.tree.addPointWithRayCasting(point, origin, update_inner_occupancy)
            
            if success:
                # Update statistics
                self.stats.total_points_processed += 1
                self.stats.last_update_time = time.time() - start_time
                self.stats.processing_time += self.stats.last_update_time
                self.stats.tree_size = self.tree.size()
                
                # Invalidate cache
                self._cache_invalid = True
            
            return success
            
        except Exception as e:
            print(f"❌ Error adding point {point}: {e}")
            return False
    
    def _mark_free_space_along_ray(self, origin: np.ndarray, end_point: np.ndarray, 
                                   step_size: float = None) -> None:
        """
        Mark free space along a ray from origin to end_point using manual sampling.
        
        Args:
            origin: Ray start point
            end_point: Ray end point  
            step_size: Step size for ray sampling (defaults to resolution)
        """
        # Use the new integrated method from octomap
        self.tree.markFreeSpaceAlongRay(origin, end_point, step_size)
        
    def add_points_batch(self, 
                        points: Union[List, np.ndarray],
                        sensor_origins: Optional[Union[List, np.ndarray]] = None,
                        update_inner_occupancy: bool = True) -> int:
        """
        Add multiple 3D points in batch for better performance
        
        Args:
            points: Array of 3D points [[x,y,z], ...] 
            sensor_origins: Optional array of sensor origins for each point
            update_inner_occupancy: Whether to update inner node occupancy
            
        Returns:
            int: Number of points successfully added
        """
        start_time = time.time()
        points = np.asarray(points, dtype=np.float64)
        
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        
        if points.shape[1] != 3:
            print(f"❌ Invalid point shape: {points.shape}. Expected (N, 3)")
            return 0
        
        try:
            # Handle sensor origins
            if sensor_origins is not None:
                sensor_origins = np.asarray(sensor_origins, dtype=np.float64)
                if len(sensor_origins.shape) == 1:
                    sensor_origins = sensor_origins.reshape(1, -1)
            else:
                # Use default sensor origin for all points
                sensor_origins = np.tile(self.sensor_origin, (len(points), 1))
            
            # Use the new integrated batch method from octomap
            success_count = self.tree.addPointsBatch(points, sensor_origins, update_inner_occupancy)
            
            # Update statistics
            batch_time = time.time() - start_time
            self.stats.total_points_processed += success_count
            self.stats.processing_time += batch_time
            self.stats.tree_size = self.tree.size()
            
            # Invalidate cache
            self._cache_invalid = True
            
            print(f"✅ Added {success_count}/{len(points)} points in {batch_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            success_count = 0
        
        return success_count
    
    def add_point_cloud(self,
                       point_cloud: np.ndarray,
                       sensor_origin: Optional[Union[List[float], np.ndarray]] = None,
                       max_range: float = -1.0,
                       use_ray_casting: bool = True) -> bool:
        """
        Add a full point cloud using either ray casting or OctoMap's optimized insertion
        
        Args:
            point_cloud: Nx3 array of points
            sensor_origin: Sensor origin for the point cloud
            max_range: Maximum range for points (-1 = no limit)
            use_ray_casting: Whether to use ray casting for more accurate free space marking
            
        Returns:
            bool: True if successful
        """
        start_time = time.time()
        
        try:
            point_cloud = np.asarray(point_cloud, dtype=np.float64)
            origin = np.array(sensor_origin if sensor_origin is not None else self.sensor_origin, dtype=np.float64)
            
            if len(point_cloud.shape) != 2 or point_cloud.shape[1] != 3:
                print(f"❌ Invalid point cloud shape: {point_cloud.shape}")
                return False
            
            if use_ray_casting:
                # Use the new integrated ray casting method
                success_count = self.tree.addPointCloudWithRayCasting(point_cloud, origin, max_range, True)
                success = success_count > 0
            else:
                # Use OctoMap's optimized point cloud insertion
                self.tree.insertPointCloud(point_cloud, origin, max_range, lazy_eval=False)
                self.tree.updateInnerOccupancy()
                success = True
                success_count = len(point_cloud)
            
            # Update statistics
            process_time = time.time() - start_time
            self.stats.total_points_processed += success_count
            self.stats.processing_time += process_time
            self.stats.tree_size = self.tree.size()
            
            # Invalidate cache
            self._cache_invalid = True
            
            method_str = "ray casting" if use_ray_casting else "standard"
            print(f"✅ Inserted point cloud ({success_count} points) using {method_str} in {process_time:.3f}s")
            return success
            
        except Exception as e:
            print(f"❌ Point cloud insertion error: {e}")
            return False
    
    def get_points_for_open3d(self, 
                             include_free: bool = True,
                             include_uncertain: bool = True,
                             use_cache: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract points from occupancy grid for Open3D visualization
        
        Args:
            include_free: Include free space points
            include_uncertain: Include uncertain points
            use_cache: Use cached results if available
            
        Returns:
            Dict with 'occupied_points', 'occupied_colors', 'free_points', 'free_colors', etc.
        """
        # Use cache if valid and requested
        if use_cache and not self._cache_invalid and self._cached_points is not None:
            return self._cached_points
        
        start_time = time.time()
        
        occupied_points = []
        free_points = []
        uncertain_points = []
        
        try:
            # Iterate through all leaf nodes
            for leaf_it in self.tree.begin_leafs():
                coord = leaf_it.getCoordinate()
                point = np.array(coord, dtype=np.float64)
                
                try:
                    node = leaf_it.current_node
                    if node is None:
                        continue
                    
                    # Classify the node
                    if self.tree.isNodeAtThreshold(node):
                        if include_uncertain:
                            uncertain_points.append(point)
                    elif self.tree.isNodeOccupied(node):
                        occupied_points.append(point)
                    else:
                        if include_free:
                            free_points.append(point)
                            
                except Exception:
                    # Fallback classification
                    try:
                        occupancy = node.getOccupancy()
                        if occupancy > 0.6:
                            occupied_points.append(point)
                        elif occupancy < 0.4:
                            if include_free:
                                free_points.append(point)
                        else:
                            if include_uncertain:
                                uncertain_points.append(point)
                    except Exception:
                        # Default to occupied if we can't determine
                        occupied_points.append(point)
            
            # Convert to numpy arrays
            occupied_points = np.array(occupied_points) if occupied_points else np.zeros((0, 3))
            free_points = np.array(free_points) if free_points else np.zeros((0, 3))
            uncertain_points = np.array(uncertain_points) if uncertain_points else np.zeros((0, 3))
            
            # Generate colors
            result = {
                'occupied_points': occupied_points,
                'free_points': free_points, 
                'uncertain_points': uncertain_points,
                'occupied_colors': self._generate_colors(occupied_points, 'occupied'),
                'free_colors': self._generate_colors(free_points, 'free'),
                'uncertain_colors': self._generate_colors(uncertain_points, 'uncertain'),
            }
            
            # Update statistics
            self.stats.occupied_cells = len(occupied_points)
            self.stats.free_cells = len(free_points)
            self.stats.uncertain_cells = len(uncertain_points)
            
            # Cache results
            self._cached_points = result
            self._cache_invalid = False
            
            extraction_time = time.time() - start_time
            print(f"📊 Extracted points in {extraction_time:.3f}s:")
            print(f"    Occupied: {len(occupied_points)}")
            print(f"    Free: {len(free_points)}")
            print(f"    Uncertain: {len(uncertain_points)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error extracting points: {e}")
            return {
                'occupied_points': np.zeros((0, 3)),
                'free_points': np.zeros((0, 3)),
                'uncertain_points': np.zeros((0, 3)),
                'occupied_colors': np.zeros((0, 3)),
                'free_colors': np.zeros((0, 3)),
                'uncertain_colors': np.zeros((0, 3)),
            }
    
    def _generate_colors(self, points: np.ndarray, point_type: str) -> np.ndarray:
        """Generate colors for points based on type and configuration"""
        if len(points) == 0:
            return np.zeros((0, 3))
        
        if point_type == 'occupied':
            base_color = self.viz_config.occupied_color
        elif point_type == 'free':
            base_color = self.viz_config.free_color
        elif point_type == 'uncertain':
            base_color = self.viz_config.uncertain_color
        else:
            base_color = [0.5, 0.5, 0.5]  # Gray default
        
        colors = np.tile(base_color, (len(points), 1))
        
        # Optional height-based coloring
        if self.viz_config.use_height_coloring and len(points) > 0:
            z_coords = points[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            
            if z_max > z_min:
                # Normalize height to [0, 1]
                height_factor = (z_coords - z_min) / (z_max - z_min)
                
                # Modulate color intensity based on height
                for i in range(3):  # RGB channels
                    colors[:, i] = colors[:, i] * (0.5 + 0.5 * height_factor)
        
        return np.clip(colors, 0.0, 1.0)
    
    def create_open3d_visualization(self, 
                                   title: str = "Sequential Occupancy Grid",
                                   include_coordinate_frame: bool = True,
                                   window_size: Tuple[int, int] = (1200, 900)) -> Optional[List]:
        """
        Create Open3D geometries for visualization
        
        Args:
            title: Window title
            include_coordinate_frame: Add coordinate frame
            window_size: Window dimensions
            
        Returns:
            List of Open3D geometries or None if Open3D not available
        """
        if not OPEN3D_AVAILABLE:
            print("⚠️ Open3D not available - cannot create visualization")
            return None
        
        # Get points
        point_data = self.get_points_for_open3d(
            include_free=self.viz_config.show_free_space,
            include_uncertain=self.viz_config.show_uncertain
        )
        
        geometries = []
        
        # Add occupied points
        if len(point_data['occupied_points']) > 0:
            occupied_pcd = o3d.geometry.PointCloud()
            occupied_pcd.points = o3d.utility.Vector3dVector(point_data['occupied_points'])
            occupied_pcd.colors = o3d.utility.Vector3dVector(point_data['occupied_colors'])
            geometries.append(occupied_pcd)
        
        # Add free points
        if self.viz_config.show_free_space and len(point_data['free_points']) > 0:
            free_pcd = o3d.geometry.PointCloud()
            free_pcd.points = o3d.utility.Vector3dVector(point_data['free_points'])
            free_pcd.colors = o3d.utility.Vector3dVector(point_data['free_colors'])
            geometries.append(free_pcd)
        
        # Add uncertain points
        if self.viz_config.show_uncertain and len(point_data['uncertain_points']) > 0:
            uncertain_pcd = o3d.geometry.PointCloud()
            uncertain_pcd.points = o3d.utility.Vector3dVector(point_data['uncertain_points'])
            uncertain_pcd.colors = o3d.utility.Vector3dVector(point_data['uncertain_colors'])
            geometries.append(uncertain_pcd)
        
        # Add coordinate frame
        if include_coordinate_frame:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            geometries.append(frame)
        
        # Add sensor origin marker
        sensor_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sensor_sphere.translate(self.sensor_origin)
        sensor_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for better visibility
        geometries.append(sensor_sphere)
        
        return geometries
    
    def visualize_with_open3d(self,
                             title: str = "Sequential Occupancy Grid",
                             auto_view: bool = True,
                             background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)):
        """
        Show interactive Open3D visualization
        
        Args:
            title: Window title
            auto_view: Automatically set view parameters
            background_color: RGB background color
        """
        if not OPEN3D_AVAILABLE:
            print("⚠️ Open3D not available - skipping visualization")
            return
        
        geometries = self.create_open3d_visualization(title)
        
        if not geometries:
            print("⚠️ No geometries to visualize")
            return
        
        print(f"🎨 Opening Open3D visualization: {title}")
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1200, height=900)
        
        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Configure render options
        render_option = vis.get_render_option()
        render_option.point_size = self.viz_config.point_size
        render_option.background_color = np.array(background_color)
        
        # Auto-set view if requested
        if auto_view:
            view_control = vis.get_view_control()
            view_control.set_front([0.5, -0.5, -0.5])
            view_control.set_lookat([0.0, 0.0, 1.0])
            view_control.set_up([0, 0, 1])
            view_control.set_zoom(0.8)
        
        # Show visualization
        vis.run()
        vis.destroy_window()
    
    def get_statistics(self) -> MappingStats:
        """Get current mapping statistics"""
        self.stats.tree_size = self.tree.size()
        return self.stats
    
    def print_statistics(self):
        """Print detailed statistics"""
        stats = self.get_statistics()
        print(f"\n📈 Occupancy Grid Statistics:")
        print(f"    Total points processed: {stats.total_points_processed}")
        print(f"    Tree size (nodes): {stats.tree_size}")
        print(f"    Occupied cells: {stats.occupied_cells}")
        print(f"    Free cells: {stats.free_cells}")
        print(f"    Uncertain cells: {stats.uncertain_cells}")
        print(f"    Total processing time: {stats.processing_time:.3f}s")
        if stats.total_points_processed > 0:
            print(f"    Avg time per point: {stats.processing_time/stats.total_points_processed*1000:.2f}ms")
        print(f"    Resolution: {self.resolution}m")
        print(f"    Sensor origin: {self.sensor_origin}")
    
    def save_map(self, filename: str) -> bool:
        """Save the occupancy map to file"""
        try:
            success = self.tree.write(filename)
            if success:
                file_size = os.path.getsize(filename)
                print(f"✅ Map saved to {filename} ({file_size} bytes)")
                return True
            else:
                print(f"❌ Failed to save map to {filename}")
                return False
        except Exception as e:
            print(f"❌ Error saving map: {e}")
            return False
    
    def load_map(self, filename: str) -> bool:
        """Load occupancy map from file"""
        try:
            loaded_tree = self.tree.read(filename)
            if loaded_tree:
                self.tree = loaded_tree
                self._cache_invalid = True
                print(f"✅ Map loaded from {filename}")
                print(f"    Tree size: {self.tree.size()} nodes")
                return True
            else:
                print(f"❌ Failed to load map from {filename}")
                return False
        except Exception as e:
            print(f"❌ Error loading map: {e}")
            return False
    
    def clear_map(self):
        """Clear the occupancy map"""
        self.tree.clear()
        self._cache_invalid = True
        self.stats = MappingStats()
        print("🗑️  Map cleared")
    
    def get_occupancy_at_point(self, point: Union[List[float], np.ndarray]) -> Optional[float]:
        """
        Get occupancy probability at a specific point
        
        Args:
            point: 3D point [x, y, z]
            
        Returns:
            Occupancy probability (0-1) or None if unknown
        """
        try:
            point = np.array(point, dtype=np.float64)
            node = self.tree.search(point)
            if node:
                return node.getOccupancy()
            return None
        except Exception as e:
            print(f"❌ Error getting occupancy at {point}: {e}")
            return None
    
    def is_point_occupied(self, point: Union[List[float], np.ndarray], threshold: float = 0.5) -> bool:
        """
        Check if a point is considered occupied
        
        Args:
            point: 3D point [x, y, z]
            threshold: Occupancy threshold
            
        Returns:
            True if occupied
        """
        occupancy = self.get_occupancy_at_point(point)
        return occupancy is not None and occupancy > threshold


def demo_sequential_occupancy_grid():
    """Demonstration of the Sequential Occupancy Grid functionality"""
    print("🚀 Sequential Occupancy Grid Demo")
    print("=" * 50)
    
    # Create occupancy grid
    grid = SequentialOccupancyGrid(
        resolution=0.05,
        sensor_origin=[0, 0, 1.5],
        prob_hit=0.7,
        prob_miss=0.3
    )
    
    # Configure visualization
    grid.viz_config.use_height_coloring = True
    grid.viz_config.show_free_space = True
    grid.viz_config.point_size = 8.0
    
    print("\n📍 Adding individual points...")
    # Add some individual points
    test_points = [
        [1.0, 1.0, 1.0],
        [1.1, 1.0, 1.0], 
        [1.0, 1.1, 1.0],
        [2.0, 2.0, 0.5],
        [2.5, 2.0, 0.8],
        [3.0, 1.5, 1.2]
    ]
    
    for point in test_points:
        grid.add_point(point)
    
    print("\n📦 Adding batch of points...")
    # Create a small room structure
    room_points = []
    # Floor
    for x in np.arange(0, 4, 0.2):
        for y in np.arange(0, 3, 0.2):
            room_points.append([x, y, 0.0])
    
    # Walls
    for x in np.arange(0, 4, 0.2):
        for z in np.arange(0, 2, 0.2):
            room_points.append([x, 0, z])    # Front wall
            room_points.append([x, 3, z])    # Back wall
    
    for y in np.arange(0, 3, 0.2):
        for z in np.arange(0, 2, 0.2):
            room_points.append([0, y, z])    # Left wall
            room_points.append([4, y, z])    # Right wall
    
    # Add obstacle in the middle
    for x in np.arange(1.5, 2.5, 0.1):
        for y in np.arange(1.0, 2.0, 0.1):
            for z in np.arange(0, 1.0, 0.1):
                room_points.append([x, y, z])
    
    grid.add_points_batch(room_points)
    
    print("\n☁️  Adding cylindrical obstacle...")
    # Create a cylindrical obstacle without free space marking
    cylinder_points = []
    # Cylindrical obstacle - make it more dense and visible
    for angle in np.linspace(0, 2*np.pi, 40):  # More angles
        for z in np.arange(0.1, 1.5, 0.05):    # More z steps
            x = 3.5 + 0.3 * np.cos(angle)
            y = 1.5 + 0.3 * np.sin(angle)
            cylinder_points.append([x, y, z])
    
    # Add some points inside the cylinder to make it more solid
    for angle in np.linspace(0, 2*np.pi, 20):
        for z in np.arange(0.2, 1.4, 0.1):
            for r in np.arange(0.1, 0.3, 0.05):
                x = 3.5 + r * np.cos(angle)
                y = 1.5 + r * np.sin(angle)
                cylinder_points.append([x, y, z])
    
    # Add cylinder points directly without ray casting (no free space marking)
    for point in cylinder_points:
        grid.tree.updateNode(np.array(point, dtype=np.float64), True)
    
    print(f"✅ Added {len(cylinder_points)} cylinder points")
    
    # Print statistics
    grid.print_statistics()
    
    # Test point queries
    print("\n�� Testing point queries...")
    test_query_points = [
        [1.0, 1.0, 1.0],  # Should be occupied
        [0.5, 0.5, 0.5],  # Should be free
        [10, 10, 10]      # Should be unknown
    ]
    
    for point in test_query_points:
        occupancy = grid.get_occupancy_at_point(point)
        is_occupied = grid.is_point_occupied(point)
        occupancy_str = f"{occupancy:.3f}" if occupancy is not None else "None"
        print(f"    Point {point}: occupancy={occupancy_str}, occupied={is_occupied}")
    
    # Get points for visualization
    print("\n🎨 Extracting points for visualization...")
    viz_points = grid.get_points_for_open3d()
    
    print(f"    Extracted {len(viz_points['occupied_points'])} occupied points")
    print(f"    Extracted {len(viz_points['free_points'])} free points")
    print(f"    Extracted {len(viz_points['uncertain_points'])} uncertain points")
    
    # Show visualization if Open3D is available
    if OPEN3D_AVAILABLE:
        print("\n🖼️  Opening visualization...")
        grid.visualize_with_open3d("Sequential Occupancy Grid Demo")
    
    # Test file operations
    print("\n�� Testing file operations...")
    filename = "demo_sequential_map.bt"
    if grid.save_map(filename):
        # Test loading
        new_grid = SequentialOccupancyGrid(resolution=0.05)
        if new_grid.load_map(filename):
            print(f"✅ Successfully loaded map with {new_grid.tree.size()} nodes")
        
        # Clean up
        try:
            os.remove(filename)
            print(f"🗑️  Cleaned up {filename}")
        except:
            pass
    
    print("\n" + "=" * 50)
    print("🎉 Sequential Occupancy Grid Demo completed!")
    print(f"✅ Processed {grid.stats.total_points_processed} points")
    print(f"✅ Generated {grid.stats.tree_size} tree nodes")
    print(f"✅ Processing time: {grid.stats.processing_time:.3f}s")
    
    return grid


if __name__ == "__main__":
    # Run the demonstration
    demo_grid = demo_sequential_occupancy_grid()
    
