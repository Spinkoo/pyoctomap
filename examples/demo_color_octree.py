#!/usr/bin/env python3
"""
ColorOcTree Example - Demonstrating Color-Enabled 3D Occupancy Mapping

This example shows how to use ColorOcTree to create a 3D occupancy map
with color information. ColorOcTree extends OcTree with RGB color support,
allowing you to store color information for each occupied voxel.

Key Features Demonstrated:
- Creating and configuring ColorOcTree
- Adding nodes with color information
- Setting, averaging, and integrating colors
- Searching and querying colored nodes
- File I/O with color data
- Iterating through colored nodes
"""

import numpy as np
import pyoctomap

# Try to import Open3D for visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def visualize_color_octree(tree, title="ColorOcTree Visualization"):
    """
    Visualize ColorOcTree with actual RGB colors from nodes.
    
    Args:
        tree: ColorOcTree instance
        title: Title for the visualization window
    """
    if not OPEN3D_AVAILABLE:
        return
    
    # Extract colored points from the tree
    colored_points = []
    colored_rgb = []
    
    # Get bounding box to sample points
    bbx_min = tree.getBBXMin()
    bbx_max = tree.getBBXMax()
    
    # If bounding box is not set, use a reasonable default based on known structures
    if bbx_min[0] == bbx_max[0] and bbx_min[1] == bbx_max[1] and bbx_min[2] == bbx_max[2]:
        # Use a default bounding box that covers all our structures
        bbx_min = np.array([-3.0, -4.0, -0.5])
        bbx_max = np.array([7.5, 7.0, 3.5])
    
    resolution = tree.getResolution()
    
    # Sample points in the bounding box
    x_range = np.arange(bbx_min[0], bbx_max[0] + resolution, resolution)
    y_range = np.arange(bbx_min[1], bbx_max[1] + resolution, resolution)
    z_range = np.arange(bbx_min[2], bbx_max[2] + resolution, resolution)
    
    # Sample every point in the bounding box
    for x in x_range:
        for y in y_range:
            for z in z_range:
                coord = np.array([x, y, z])
                search_node = tree.search(coord)
                if search_node:
                    # Check if node is occupied
                    if tree.isNodeOccupied(search_node):
                        # Get color from ColorOcTreeNode
                        if isinstance(search_node, pyoctomap.ColorOcTreeNode):
                            color = search_node.getColor()
                            # Convert RGB from 0-255 to 0.0-1.0 for Open3D
                            rgb_normalized = [c / 255.0 for c in color]
                            colored_points.append(coord)
                            colored_rgb.append(rgb_normalized)
    
    if not colored_points:
        # Alternative: use known coordinates from the example
        known_coords = [
            np.array([2.0, 0.0, 1.0]),   # red_wall
            np.array([0.0, 2.0, 1.5]),   # green_tree
            np.array([-2.0, 0.0, 0.5]),  # blue_object
            np.array([0.0, -2.0, 2.0]),  # yellow_landmark
            np.array([1.0, 1.0, 1.0]),   # test_point
            np.array([-1.0, -1.0, 1.0]), # integrate_point
            np.array([5.0, 6.0, 1.0]),   # building walls
            np.array([6.0, 5.0, 1.0]),
            np.array([5.0, 4.0, 1.0]),
            np.array([4.0, 5.0, 1.0]),
        ]
        
        for coord in known_coords:
            search_node = tree.search(coord)
            if search_node and tree.isNodeOccupied(search_node):
                if isinstance(search_node, pyoctomap.ColorOcTreeNode):
                    color = search_node.getColor()
                    rgb_normalized = [c / 255.0 for c in color]
                    colored_points.append(coord)
                    colored_rgb.append(rgb_normalized)
    
    if not colored_points:
        return
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(colored_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colored_rgb))
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Visualize with better settings
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1024, height=768)
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    
    # Set up camera view to see all structures
    view_control = vis.get_view_control()
    view_control.set_front([0.3, -0.5, -0.7])
    view_control.set_lookat([2.5, 1.0, 1.0])  # Look at center of structures
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(0.5)  # Zoom out to see more
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = 5.0  # Smaller points for denser structures
    render_option.background_color = np.array([0.05, 0.05, 0.1])  # Dark blue background
    
    try:
        vis.run()
    except Exception:
        pass
    finally:
        try:
            vis.destroy_window()
        except:
            pass


def main():
    # ========================================================================
    # 1. Create a ColorOcTree
    # ========================================================================
    tree = pyoctomap.ColorOcTree(0.1)
    
    # ========================================================================
    # 2. Add obstacles with different colors (creating visible structures)
    # ========================================================================
    resolution = tree.getResolution()
    
    # Red wall (vertical surface)
    red_wall_center = np.array([2.0, 0.0, 1.0])
    wall_width = 0.3  # 30cm thick
    wall_height = 2.0  # 2m tall
    wall_depth = 0.2  # 20cm deep
    for x in np.arange(red_wall_center[0] - wall_width/2, red_wall_center[0] + wall_width/2, resolution):
        for y in np.arange(red_wall_center[1] - wall_depth/2, red_wall_center[1] + wall_depth/2, resolution):
            for z in np.arange(red_wall_center[2] - wall_height/2, red_wall_center[2] + wall_height/2, resolution):
                coord = np.array([x, y, z])
                tree.updateNode(coord, True)
                tree.setNodeColor(coord, 255, 0, 0)  # Red
    
    # Green vegetation (tree-like structure)
    green_tree_center = np.array([0.0, 2.0, 1.0])
    tree_radius = 0.4  # 40cm radius
    tree_height = 1.5  # 1.5m tall
    for x in np.arange(green_tree_center[0] - tree_radius, green_tree_center[0] + tree_radius, resolution):
        for y in np.arange(green_tree_center[1] - tree_radius, green_tree_center[1] + tree_radius, resolution):
            for z in np.arange(green_tree_center[2], green_tree_center[2] + tree_height, resolution):
                dist = np.sqrt((x - green_tree_center[0])**2 + (y - green_tree_center[1])**2)
                if dist <= tree_radius:
                    coord = np.array([x, y, z])
                    tree.updateNode(coord, True)
                    # Vary green intensity slightly for realism
                    green_intensity = int(200 + 55 * (1 - dist/tree_radius))
                    tree.setNodeColor(coord, 0, green_intensity, 0)
    
    # Blue object (cube)
    blue_object_center = np.array([-2.0, 0.0, 0.5])
    cube_size = 0.5  # 50cm cube
    for x in np.arange(blue_object_center[0] - cube_size/2, blue_object_center[0] + cube_size/2, resolution):
        for y in np.arange(blue_object_center[1] - cube_size/2, blue_object_center[1] + cube_size/2, resolution):
            for z in np.arange(blue_object_center[2] - cube_size/2, blue_object_center[2] + cube_size/2, resolution):
                coord = np.array([x, y, z])
                tree.updateNode(coord, True)
                tree.setNodeColor(coord, 0, 0, 255)  # Blue
    
    # Yellow landmark (tall pillar)
    yellow_landmark_center = np.array([0.0, -2.0, 1.0])
    pillar_radius = 0.15  # 15cm radius
    pillar_height = 2.5  # 2.5m tall
    for x in np.arange(yellow_landmark_center[0] - pillar_radius, yellow_landmark_center[0] + pillar_radius, resolution):
        for y in np.arange(yellow_landmark_center[1] - pillar_radius, yellow_landmark_center[1] + pillar_radius, resolution):
            for z in np.arange(yellow_landmark_center[2] - pillar_height/2, yellow_landmark_center[2] + pillar_height/2, resolution):
                dist = np.sqrt((x - yellow_landmark_center[0])**2 + (y - yellow_landmark_center[1])**2)
                if dist <= pillar_radius:
                    coord = np.array([x, y, z])
                    tree.updateNode(coord, True)
                    tree.setNodeColor(coord, 255, 255, 0)  # Yellow
    
    # Store centers for later reference
    red_wall = red_wall_center
    green_tree = green_tree_center
    blue_object = blue_object_center
    yellow_landmark = yellow_landmark_center
    
    # ========================================================================
    # 3. Search nodes and retrieve colors
    # ========================================================================
    node = tree.search(red_wall)
    if node:
        color = node.getColor()
    
    node = tree.search(green_tree)
    
    # ========================================================================
    # 4. Average colors (useful for multiple measurements)
    # ========================================================================
    # Create a small colored region for averaging demo
    test_center = np.array([1.0, 1.0, 1.0])
    test_size = 0.2  # 20cm cube
    resolution = tree.getResolution()
    
    # Set initial color
    for x in np.arange(test_center[0] - test_size/2, test_center[0] + test_size/2, resolution):
        for y in np.arange(test_center[1] - test_size/2, test_center[1] + test_size/2, resolution):
            for z in np.arange(test_center[2] - test_size/2, test_center[2] + test_size/2, resolution):
                coord = np.array([x, y, z])
                tree.updateNode(coord, True)
                tree.setNodeColor(coord, 100, 100, 100)  # Gray
    
    # Average with new measurement
    tree.averageNodeColor(test_center, 200, 200, 200)  # Lighter gray
    
    # ========================================================================
    # 5. Integrate colors (weighted by occupancy probability)
    # ========================================================================
    integrate_center = np.array([-1.0, -1.0, 1.0])
    integrate_size = 0.15  # 15cm cube
    
    # Set initial color
    for x in np.arange(integrate_center[0] - integrate_size/2, integrate_center[0] + integrate_size/2, resolution):
        for y in np.arange(integrate_center[1] - integrate_size/2, integrate_center[1] + integrate_size/2, resolution):
            for z in np.arange(integrate_center[2] - integrate_size/2, integrate_center[2] + integrate_size/2, resolution):
                coord = np.array([x, y, z])
                tree.updateNode(coord, True)
                tree.setNodeColor(coord, 128, 128, 128)  # Medium gray
    
    # Integrate new color measurement
    tree.integrateNodeColor(integrate_center, 64, 64, 64)  # Darker gray
    
    # ========================================================================
    # 6. Create a colored structure (building with colored walls)
    # ========================================================================
    # Create walls of a simple building (actual wall surfaces, not just points)
    building_center = np.array([5.0, 5.0, 1.0])
    wall_length = 2.0  # 2m long walls
    wall_height = 2.0  # 2m tall
    wall_thickness = 0.2  # 20cm thick
    
    wall_configs = [
        {"name": "North", "offset": np.array([0, 1, 0]), "color": (255, 200, 150)},  # Light orange
        {"name": "East", "offset": np.array([1, 0, 0]), "color": (200, 150, 255)},   # Light purple
        {"name": "South", "offset": np.array([0, -1, 0]), "color": (150, 255, 200)}, # Light green
        {"name": "West", "offset": np.array([-1, 0, 0]), "color": (255, 255, 150)},  # Light yellow
    ]
    
    for i, wall_cfg in enumerate(wall_configs):
        wall_center = building_center + wall_cfg["offset"]
        color = wall_cfg["color"]
        
        # Create wall surface
        if wall_cfg["name"] in ["North", "South"]:
            # North/South walls extend in X direction
            for x in np.arange(wall_center[0] - wall_length/2, wall_center[0] + wall_length/2, resolution):
                for y in np.arange(wall_center[1] - wall_thickness/2, wall_center[1] + wall_thickness/2, resolution):
                    for z in np.arange(wall_center[2] - wall_height/2, wall_center[2] + wall_height/2, resolution):
                        coord = np.array([x, y, z])
                        tree.updateNode(coord, True)
                        tree.setNodeColor(coord, color[0], color[1], color[2])
        else:
            # East/West walls extend in Y direction
            for x in np.arange(wall_center[0] - wall_thickness/2, wall_center[0] + wall_thickness/2, resolution):
                for y in np.arange(wall_center[1] - wall_length/2, wall_center[1] + wall_length/2, resolution):
                    for z in np.arange(wall_center[2] - wall_height/2, wall_center[2] + wall_height/2, resolution):
                        coord = np.array([x, y, z])
                        tree.updateNode(coord, True)
                        tree.setNodeColor(coord, color[0], color[1], color[2])
    
    # ========================================================================
    # 7. Iterate through colored nodes
    # ========================================================================
    colored_nodes_count = 0
    color_distribution = {}
    
    # Iterate through all leaf nodes
    # Note: SimpleLeafIterator returns OcTreeNode objects, not ColorOcTreeNode
    # We need to search for the ColorOcTreeNode to get color information
    for node in pyoctomap.SimpleLeafIterator(tree):
        if tree.isNodeOccupied(node):
            # Get node coordinate to search and get color
            try:
                coord = node.getCoordinate()
                search_node = tree.search(coord)
                if search_node and isinstance(search_node, pyoctomap.ColorOcTreeNode):
                    color = search_node.getColor()
                    # Count nodes by color category
                    if color == (255, 0, 0):
                        color_distribution['Red'] = color_distribution.get('Red', 0) + 1
                    elif color == (0, 255, 0):
                        color_distribution['Green'] = color_distribution.get('Green', 0) + 1
                    elif color == (0, 0, 255):
                        color_distribution['Blue'] = color_distribution.get('Blue', 0) + 1
                    elif color == (255, 255, 0):
                        color_distribution['Yellow'] = color_distribution.get('Yellow', 0) + 1
                    else:
                        color_distribution['Other'] = color_distribution.get('Other', 0) + 1
                    colored_nodes_count += 1
            except:
                pass  # Skip nodes that can't be processed
    
    # ========================================================================
    # 8. Ray casting with color information
    # ========================================================================
    # Cast ray towards the red wall
    origin = np.array([0.0, 0.0, 1.0])
    direction = red_wall - origin
    direction = direction / np.linalg.norm(direction)  # Normalize
    end_point = np.zeros(3)
    
    hit = tree.castRay(origin, direction, end_point, ignoreUnknownCells=True, maxRange=10.0)
    
    # ========================================================================
    # 9. File I/O with color data
    # ========================================================================
    import tempfile
    import os
    
    # Save to binary file
    with tempfile.NamedTemporaryFile(suffix='.ot', delete=False) as tmp_file:
        filename = tmp_file.name
    
    # Use write() method for standard OctoMap format
    success = tree.write(filename)
    if success:
        # Load from file - ColorOcTree constructor can read files directly
        try:
            loaded_tree = pyoctomap.ColorOcTree(filename)
        except Exception:
            pass
    
    # Clean up
    if os.path.exists(filename):
        os.unlink(filename)
    
    # ========================================================================
    # 10. Advanced: Color-based region query
    # ========================================================================
    # Find all red nodes in a bounding box
    bbx_min = np.array([-5.0, -5.0, -1.0])
    bbx_max = np.array([5.0, 5.0, 3.0])
    
    red_nodes = []
    for node in pyoctomap.SimpleLeafBBXIterator(tree, bbx_min, bbx_max):
        if tree.isNodeOccupied(node):
            try:
                coord = node.getCoordinate()
                search_node = tree.search(coord)
                if search_node and isinstance(search_node, pyoctomap.ColorOcTreeNode):
                    color = search_node.getColor()
                    if color == (255, 0, 0):  # Red
                        red_nodes.append(coord)
            except:
                pass
    
    # ========================================================================
    # 11. Visualize the colored map
    # ========================================================================
    visualize_color_octree(tree, title="ColorOcTree - Colored 3D Occupancy Map")


if __name__ == "__main__":
    main()

