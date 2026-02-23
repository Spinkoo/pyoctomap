#!/usr/bin/env python3
"""
DecayOcTree Example - Demonstrating Temporal Decay in Dynamic Environments

This example shows how to use DecayOcTree to handle dynamic environments.
It simulates an environment with a static wall and a moving person.
By using `decayAndInsertPointCloud`, the old "ghost" points from the 
moving person will fade away over time.
"""

import numpy as np
import time
import math
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
import pyoctomap

def visualize_decay_process():
    """Visualize a sequence of scans to demonstrate decay."""
    
    if not OPEN3D_AVAILABLE:
        print("Open3D is required to run the visualization.")
        return

    # Initialize a DecayOcTree with 0.1m resolution
    tree = pyoctomap.DecayOcTree(0.1)
    
    # We will simulate a sequence of 20 frames
    num_frames = 20
    
    # Sensor origin is fixed for simplicity
    sensor_origin = np.array([0.0, 0.0, 0.5])
    
    # Set up Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="DecayOcTree Demo (Red = Moving, Blue = Static)", width=1024, height=768)
    
    # Point cloud geometry to hold the current occupancy map
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # Setup camera
    view_control = vis.get_view_control()
    view_control.set_front([0.5, -0.6, -0.6])
    view_control.set_lookat([3.0, 2.0, 1.0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(0.8)
    
    # Add coordinate frame
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    # Pre-generate the static environment (a simple wall)
    static_points = []
    for x in np.arange(4.0, 4.5, 0.1):
        for y in np.arange(-2.0, 4.0, 0.1):
            for z in np.arange(0.0, 2.0, 0.1):
                static_points.append([x, y, z])
    static_points = np.array(static_points)

    print("Simulating frames...")
    
    for frame in range(num_frames):
        # 1. Generate new dynamic data (a person walking)
        dynamic_points = []
        # Person moves along y axis
        y_pos = -1.0 + frame * 0.25 
        for dx in np.arange(-0.2, 0.3, 0.1):
            for dy in np.arange(-0.2, 0.3, 0.1):
                for z in np.arange(0.0, 1.8, 0.1):
                    dynamic_points.append([2.0 + dx, y_pos + dy, z])
        dynamic_points = np.array(dynamic_points)
        
        # Combine static and dynamic points
        scan_points = np.vstack([static_points, dynamic_points])
        
        # 2. Add some noise for realism
        noise = np.random.normal(0, 0.02, scan_points.shape)
        scan_points += noise

        # 3. Decay and Insert!
        # logodd_decay_value of -0.5 means a fully occupied voxel (logodd ~4.0) 
        # will disappear in about 4/0.5 = 8 frames if not seen again.
        tree.decayAndInsertPointCloud(
            point_cloud=scan_points,
            sensor_origin=sensor_origin,
            logodd_decay_value=-0.5,
            max_range=-1.0,
            update_inner_occupancy=False # Faster, wait until extraction
        )
        
        # Manually update inner occupancy before extraction
        tree.updateInnerOccupancy()

        # 4. Extract points for visualization
        voxels, _ = tree.extractPointCloud()
        
        if voxels is not None and len(voxels) > 0:
            pcd.points = o3d.utility.Vector3dVector(voxels)
            
            # Color points to distinguish ghosts (we color by X coordinate simply for visual effect)
            colors = np.zeros_like(voxels)
            for i, p in enumerate(voxels):
                if p[0] < 3.0: # Dynamic region
                    # The newer the point, the closer to the current frame
                    # We can't know the exact age easily, so just make them red
                    colors[i] = [1.0, 0.0, 0.0]
                else:          # Static region
                    colors[i] = [0.0, 0.5, 1.0]

            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Update Open3D window
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # Pause to see the effect
        time.sleep(0.2)
        
    print("Done simulating.")
    print("Close the Open3D window to exit.")
    
    # Keep window open until user closes it
    vis.run()
    vis.destroy_window()

def test_ray_casting_clear():
    """Demonstrate clearing free space with single point raycasting."""
    tree = pyoctomap.DecayOcTree(0.1)
    
    print("\n--- Raycasting Demo ---")
    
    # 1. Add an obstacle
    obstacle_point = np.array([2.0, 0.0, 1.0])
    sensor = np.array([0.0, 0.0, 1.0])
    
    print(f"Adding obstacle at {obstacle_point}")
    tree.updateNode(obstacle_point, True)
    
    # Check it exists
    node = tree.search(obstacle_point)
    print(f"Node occupied? {tree.isNodeOccupied(node) if node else 'No Node'}")
    
    # 2. Shoot a ray *past* the obstacle to clear it
    far_point = np.array([3.0, 0.0, 1.0])
    print(f"\nShooting ray to {far_point} from {sensor}")
    
    # This should clear the space between sensor and far_point
    tree.addPointWithRayCasting(far_point, sensor, update_inner_occupancy=True)
    
    # Check obstacle again
    node = tree.search(obstacle_point)
    # The node might exist but should be free (prob < 0.5)
    is_occ = tree.isNodeOccupied(node) if node else False
    print(f"Is old obstacle still occupied? {is_occ}")

def main():
    test_ray_casting_clear()
    
    # Run the visual demo
    visualize_decay_process()

if __name__ == "__main__":
    main()
