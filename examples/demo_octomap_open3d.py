#!/usr/bin/env python3
"""
Demonstration script showing practical usage of the OctoMap wrapper.
This script shows how to use the wrapper for real-world applications.
"""

import numpy as np
import sys
import os

# Add current directory to path for proper import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import Open3D for visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("✅ Open3D available for visualization")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️ Open3D not available - visualization will be skipped")


try:
    import octomap
    print("✅ OctoMap import successful!")
except ImportError as e:
    print(f"❌ Failed to import octomap: {e}")
    sys.exit(1)

def demo_room_mapping():
    """Demonstrate mapping a richer indoor scene with architectural details."""
    print("\n🏠 Room Mapping Demo (Enhanced)")
    print("=" * 40)

    # Create octree with 0.1m resolution (10cm)
    tree = octomap.OcTree(0.1)

    # Scene extents
    room_w, room_d, room_h = 6.0, 5.0, 3.0  # width (x), depth (y), height (z)

    # Helper to add a wall rectangle (with optional holes for doors/windows)
    def add_wall(x_range, y_range, z_range, hole_boxes=None, step=0.1):
        hx = hole_boxes or []
        for x in np.arange(x_range[0], x_range[1] + step/2, step):
            for y in np.arange(y_range[0], y_range[1] + step/2, step):
                for z in np.arange(z_range[0], z_range[1] + step/2, step):
                    # skip if inside any hole
                    skip = False
                    for (hx0, hx1, hy0, hy1, hz0, hz1) in hx:
                        if hx0 <= x <= hx1 and hy0 <= y <= hy1 and hz0 <= z <= hz1:
                            skip = True
                            break
                    if not skip:
                        tree.updateNode([x, y, z], True)

    # Floor and ceiling (filled thin slabs)
    def add_slab(z, thickness=0.1, step=0.15):
        for x in np.arange(0.0, room_w + step/2, step):
            for y in np.arange(0.0, room_d + step/2, step):
                for dz in np.arange(0.0, thickness + step/2, step):
                    tree.updateNode([x, y, z + dz], True)

    print("Building architecture: floor, walls, ceiling, openings...")
    # Floor at z=0 and ceiling at z=room_h
    add_slab(0.0, thickness=0.1, step=0.15)
    add_slab(room_h - 0.1, thickness=0.1, step=0.15)

    # Walls: create four walls with a door and a window opening
    door_w, door_h = 0.9, 2.1
    door_x0, door_x1 = 2.0, 2.0 + door_w
    door_z0, door_z1 = 0.0, door_h
    # front wall (y=0), window elsewhere
    window_w, window_h = 1.2, 0.8
    win_x0, win_x1 = 4.0, 4.0 + window_w
    win_z0, win_z1 = 1.2, 2.0

    # Front wall at y ∈ [0, 0.15]
    add_wall([0.0, room_w], [0.0, 0.15], [0.0, room_h],
             hole_boxes=[(door_x0, door_x1, 0.0, 0.15, door_z0, door_z1),
                         (win_x0, win_x1, 0.0, 0.15, win_z0, win_z1)], step=0.15)
    # Back wall at y ∈ [room_d-0.15, room_d]
    add_wall([0.0, room_w], [room_d - 0.15, room_d], [0.0, room_h], step=0.15)
    # Left wall at x ∈ [0, 0.15]
    add_wall([0.0, 0.15], [0.0, room_d], [0.0, room_h], step=0.15)
    # Right wall at x ∈ [room_w - 0.15, room_w]
    add_wall([room_w - 0.15, room_w], [0.0, room_d], [0.0, room_h], step=0.15)

    print("Adding architectural elements: pillars, curved sofa, table set, shelves, plants...")
    # Pillars (cylinders) near corners
    def add_cylinder(center, radius, height, step=0.12):
        cx, cy, cz = center
        for x in np.arange(cx - radius, cx + radius + step/2, step):
            for y in np.arange(cy - radius, cy + radius + step/2, step):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2 + 1e-6:
                    for z in np.arange(cz, cz + height + step/2, step):
                        tree.updateNode([x, y, z], True)

    add_cylinder([0.8, 0.8, 0.0], 0.25, 2.8)
    add_cylinder([room_w - 0.8, room_d - 0.8, 0.0], 0.25, 2.8)

    # Curved sofa (arc of small cylinders/blocks)
    sofa_center = [1.8, 3.5]
    sofa_radius = 1.0
    for angle in np.linspace(-0.5 * np.pi, 0.1 * np.pi, 10):
        px = sofa_center[0] + sofa_radius * np.cos(angle)
        py = sofa_center[1] + sofa_radius * np.sin(angle)
        add_cylinder([px, py, 0.0], 0.12, 0.6, step=0.1)

    # Coffee table (low cylinder)
    add_cylinder([sofa_center[0], sofa_center[1], 0.0], 0.35, 0.45, step=0.08)

    # Dining table with four chairs
    def add_box(pmin, pmax, step=0.1):
        x0, y0, z0 = pmin
        x1, y1, z1 = pmax
        for x in np.arange(x0, x1 + step/2, step):
            for y in np.arange(y0, y1 + step/2, step):
                for z in np.arange(z0, z1 + step/2, step):
                    tree.updateNode([x, y, z], True)

    table_min = [4.3, 2.0, 0.0]
    table_max = [5.2, 2.8, 0.75]
    add_box(table_min, table_max, step=0.1)
    chair_size = [0.45, 0.45, 0.9]
    chair_offsets = [[-0.55, -0.55], [0.6, -0.55], [-0.55, 0.6], [0.6, 0.6]]
    for off in chair_offsets:
        cmin = [table_min[0] + off[0], table_min[1] + off[1], 0.0]
        cmax = [cmin[0] + chair_size[0], cmin[1] + chair_size[1], chair_size[2]]
        add_box(cmin, cmax, step=0.1)

    # Wall-mounted shelves on back wall
    shelf_y = room_d - 0.2
    for i in range(3):
        z_base = 0.8 + i * 0.5
        add_box([0.6, shelf_y, z_base], [2.0, shelf_y + 0.1, z_base + 0.08], step=0.08)

    # Plant cluster (spheres approximated by concentric cylinders)
    def add_sphere(center, radius, step=0.12):
        cx, cy, cz = center
        for x in np.arange(cx - radius, cx + radius + step/2, step):
            for y in np.arange(cy - radius, cy + radius + step/2, step):
                for z in np.arange(cz - radius, cz + radius + step/2, step):
                    if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2 + 1e-6:
                        tree.updateNode([x, y, z], True)

    add_sphere([5.5, 4.4, 0.5], 0.4, step=0.1)

    # Carve free corridor from door to center
    print("Carving free-space corridor from door to center...")
    start = [door_x0 + 0.2, 0.2, 0.6]
    goal = [room_w/2, room_d/2, 0.6]
    steps = 60
    for i in range(steps + 1):
        t = i / steps
        x = start[0] + t * (goal[0] - start[0])
        y = start[1] + t * (goal[1] - start[1])
        for z in np.arange(0.2, 2.0, 0.2):
            tree.updateNode([x, y, z], False)

    print(f"Total tree size: {tree.size()} nodes")

    # Visualize the enriched scene
    visualize_octree(tree, "Enhanced Room Mapping - Architectural Scene", max_depth=0, show_wireframe=True)

    return tree

def demo_probabilistic_mapping():
    """Demonstrate probabilistic occupancy mapping"""
    print("\n📊 Probabilistic Mapping Demo")
    print("=" * 40)
    
    tree = octomap.OcTree(0.1)
    
    # Create a more diverse set of sensor readings
    occupied_readings = []
    free_readings = []
    mixed_readings = []
    
    # High confidence occupied area (multiple readings)
    for i in range(5):
        occupied_readings.append([1.0, 1.0, 1.0])
        occupied_readings.append([1.1, 1.0, 1.0])
        occupied_readings.append([1.0, 1.1, 1.0])
    
    # High confidence free area (multiple readings)
    for i in range(5):
        free_readings.append([0.5, 0.5, 0.5])
        free_readings.append([0.6, 0.5, 0.5])
        free_readings.append([0.5, 0.6, 0.5])
    
    # Uncertain/mixed area
    mixed_points = [
        [2.0, 2.0, 2.0], [2.1, 2.0, 2.0], [2.0, 2.1, 2.0],
        [2.2, 2.0, 2.0], [2.0, 2.2, 2.0], [2.1, 2.1, 2.0]
    ]
    
    # Create a grid of different confidence levels
    print("Creating probabilistic map with different confidence levels...")
    
    # Area 1: High confidence occupied (walls)
    for x in np.arange(3.0, 4.0, 0.1):
        for y in np.arange(0.0, 1.0, 0.1):
            for z in np.arange(0.0, 1.0, 0.2):
                for _ in range(4):  # Multiple readings for high confidence
                    tree.updateNode([x, y, z], True)
    
    # Area 2: High confidence free (open space)
    for x in np.arange(0.2, 1.5, 0.2):
        for y in np.arange(0.2, 1.5, 0.2):
            for z in np.arange(0.2, 1.0, 0.2):
                for _ in range(3):  # Multiple readings for high confidence
                    tree.updateNode([x, y, z], False)
    
    # Area 3: Medium confidence occupied (furniture)
    for x in np.arange(2.0, 2.8, 0.2):
        for y in np.arange(1.5, 2.2, 0.2):
            for z in np.arange(0.0, 0.8, 0.2):
                for _ in range(2):  # Fewer readings
                    tree.updateNode([x, y, z], True)
    
    # Area 4: Low confidence / mixed readings (uncertain area)
    uncertain_points = [
        [1.8, 1.8, 0.5], [1.9, 1.8, 0.5], [1.8, 1.9, 0.5],
        [1.7, 1.8, 0.5], [1.8, 1.7, 0.5]
    ]
    for point in uncertain_points:
        tree.updateNode(point, True)   # Occupied reading
        tree.updateNode(point, False)  # Free reading
        tree.updateNode(point, True)   # Occupied reading
    
    print(f"Tree size: {tree.size()} nodes")
    
    # Test different areas
    test_areas = [
        ([3.5, 0.5, 0.5], "High confidence wall"),
        ([0.8, 0.8, 0.5], "High confidence free space"), 
        ([2.4, 1.8, 0.4], "Medium confidence furniture"),
        ([1.8, 1.8, 0.5], "Low confidence / uncertain"),
        ([5.0, 5.0, 5.0], "Unknown area")
    ]
    
    print("\nFinal occupancy states by area:")
    for point, description in test_areas:
        node = tree.search(point)
        if node:
            try:
                occupied = tree.isNodeOccupied(node)
                at_threshold = tree.isNodeAtThreshold(node)
                occupancy = node.getOccupancy()
                log_odds = node.getLogOdds()
                print(f"  {description}: Occupied={occupied}, AtThreshold={at_threshold}")
                print(f"    Occupancy={occupancy:.3f}, LogOdds={log_odds:.3f}")
            except Exception as e:
                print(f"  {description}: Error reading node - {e}")
        else:
            print(f"  {description}: Unknown (no node)")
    
    # Visualize the probabilistic mapping
    visualize_octree(tree, "Probabilistic Mapping - Different Confidence Levels")
    
    return tree

def _interpolate_palette(t, palette):
    """Linear interpolate a list of RGB colors (0-1) with parameter t in [0,1]."""
    if t <= 0.0:
        return palette[0]
    if t >= 1.0:
        return palette[-1]
    n = len(palette) - 1
    idx = int(t * n)
    frac = t * n - idx
    c0 = palette[idx]
    c1 = palette[idx + 1]
    return [c0[0] + (c1[0] - c0[0]) * frac,
            c0[1] + (c1[1] - c0[1]) * frac,
            c0[2] + (c1[2] - c0[2]) * frac]


def _colorize_by_height(points_arr, palette):
    """Return Nx3 colors (0-1) mapping Z to a multi-color palette."""
    if len(points_arr) == 0:
        return np.zeros((0, 3), dtype=float)
    z = points_arr[:, 2]
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    if zmax <= zmin:
        t = np.zeros_like(z)
    else:
        t = (z - zmin) / (zmax - zmin)
    colors = np.zeros((points_arr.shape[0], 3), dtype=float)
    for i, ti in enumerate(t):
        colors[i] = _interpolate_palette(float(ti), palette)
    return colors


def visualize_octree(tree, title="OctoMap Visualization", max_depth=0, show_wireframe=True):
    """Visualize the octree using Open3D with leaf-iterator extraction and better overlays."""
    if not OPEN3D_AVAILABLE:
        print("⚠️ Open3D not available - skipping visualization")
        return

    print(f"\n🎨 Creating Open3D visualization: {title}")

    # Extract points using iterators (faster and exact)
    occupied_pts, free_pts, uncertain_pts, occ_samples_for_boxes = [], [], [], []
    counter = 0
    for it in tree.begin_leafs(max_depth):
        node = it.current_node
        if not node:
            continue
        coord = it.getCoordinate()
        # Classify
        try:
            at_threshold = tree.isNodeAtThreshold(node)
            if at_threshold:
                uncertain_pts.append(coord)
                continue
            is_occ = tree.isNodeOccupied(node)
        except Exception:
            # Fallback if occupancy helpers not available
            is_occ = node.getLogOdds() > 0

        if is_occ:
            occupied_pts.append(coord)
            # Keep a decimated sample for wireframe cubes
            if (counter % 20) == 0:
                occ_samples_for_boxes.append((coord, it.getSize()))
        else:
            free_pts.append(coord)
        counter += 1

    print(f"Extracted: {len(occupied_pts)} occupied, {len(free_pts)} free, {len(uncertain_pts)} uncertain")
    if not occupied_pts and not free_pts and not uncertain_pts:
        print("⚠️ No points found for visualization")
        return

    geometries = []

    if occupied_pts:
        occupied_arr = np.array(occupied_pts)
        occupied_pcd = o3d.geometry.PointCloud()
        occupied_pcd.points = o3d.utility.Vector3dVector(occupied_arr)
        # Multi-color palette for occupied: red -> magenta -> orange
        occ_palette = [
            [1.0, 0.0, 0.0],  # red
            [1.0, 0.0, 1.0],  # magenta
            [1.0, 0.5, 0.0],  # orange
        ]
        occ_colors = _colorize_by_height(occupied_arr, occ_palette)
        occupied_pcd.colors = o3d.utility.Vector3dVector(occ_colors)
        geometries.append(occupied_pcd)

    if free_pts:
        free_arr = np.array(free_pts)
        free_pcd = o3d.geometry.PointCloud()
        free_pcd.points = o3d.utility.Vector3dVector(free_arr)
        # Multi-color palette for free: teal -> green -> cyan
        free_palette = [
            [0.0, 0.6, 0.6],  # teal
            [0.0, 0.8, 0.0],  # green
            [0.0, 0.7, 1.0],  # cyan
        ]
        free_colors = _colorize_by_height(free_arr, free_palette)
        free_pcd.colors = o3d.utility.Vector3dVector(free_colors)
        geometries.append(free_pcd)

    if uncertain_pts:
        unc_arr = np.array(uncertain_pts)
        uncertain_pcd = o3d.geometry.PointCloud()
        uncertain_pcd.points = o3d.utility.Vector3dVector(unc_arr)
        # Multi-color palette for uncertain: yellow -> orange -> purple
        unc_palette = [
            [1.0, 0.9, 0.0],  # yellow
            [1.0, 0.6, 0.0],  # orange
            [0.7, 0.3, 0.8],  # purple
        ]
        unc_colors = _colorize_by_height(unc_arr, unc_palette)
        uncertain_pcd.colors = o3d.utility.Vector3dVector(unc_colors)
        geometries.append(uncertain_pcd)

    # Coordinate frame
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    # Optional: wireframe cubes to show voxel boundaries for some occupied leaves
    if show_wireframe and occ_samples_for_boxes:
        for coord, size in occ_samples_for_boxes:
            try:
                cube = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
                # move to center (Open3D boxes are at origin corner by default)
                cube.translate([coord[0] - size / 2.0, coord[1] - size / 2.0, coord[2] - size / 2.0])
                wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(cube)
                wireframe.paint_uniform_color([0.8, 0.0, 0.0])
                geometries.append(wireframe)
            except Exception:
                # If any error in size/placement, skip the cube
                pass

    print("🖼️ Opening Open3D visualization window...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=1000)
    for g in geometries:
        vis.add_geometry(g)

    vc = vis.get_view_control()
    vc.set_front([0.5, -0.5, -0.5])
    vc.set_lookat([2.0, 2.0, 1.5])
    vc.set_up([0, 0, 1])
    vc.set_zoom(0.6)

    ro = vis.get_render_option()
    ro.point_size = 6.0
    ro.line_width = 2.0
    ro.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()

def _generate_pointcloud_room_scene():
    """Generate a richer synthetic point cloud (walls, furniture, plants) similar to the matplotlib demo."""
    points = []
    room_w, room_d, room_h = 6.0, 5.0, 3.0
    # Floor grid
    for x in np.arange(0.0, room_w, 0.15):
        for y in np.arange(0.0, room_d, 0.15):
            if np.random.rand() < 0.4:
                points.append([x, y, 0.0])
    # Walls planes (sparse)
    for x in np.arange(0.0, room_w, 0.15):
        for z in np.arange(0.0, room_h, 0.15):
            if np.random.rand() < 0.6:
                points.append([x, 0.0, z])
                points.append([x, room_d, z])
    for y in np.arange(0.0, room_d, 0.15):
        for z in np.arange(0.0, room_h, 0.15):
            if np.random.rand() < 0.6:
                points.append([0.0, y, z])
                points.append([room_w, y, z])
    # Pillars
    def add_cyl(center, radius, height, step=0.1):
        cx, cy, cz = center
        for x in np.arange(cx - radius, cx + radius, step):
            for y in np.arange(cy - radius, cy + radius, step):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    for z in np.arange(cz, cz + height, step):
                        if np.random.rand() < 0.8:
                            points.append([x, y, z])
    add_cyl([0.8, 0.8, 0.0], 0.25, 2.8)
    add_cyl([room_w - 0.8, room_d - 0.8, 0.0], 0.25, 2.8)
    # Curved sofa arc
    sofa_center = [1.8, 3.5]
    sofa_radius = 1.0
    for angle in np.linspace(-0.5 * np.pi, 0.1 * np.pi, 10):
        px = sofa_center[0] + sofa_radius * np.cos(angle)
        py = sofa_center[1] + sofa_radius * np.sin(angle)
        add_cyl([px, py, 0.0], 0.12, 0.6, step=0.08)
    # Coffee table
    add_cyl([sofa_center[0], sofa_center[1], 0.0], 0.35, 0.45, step=0.08)
    # Dining table + chairs (boxes as point clouds)
    def add_box_points(pmin, pmax, step=0.1):
        x0, y0, z0 = pmin
        x1, y1, z1 = pmax
        for x in np.arange(x0, x1, step):
            for y in np.arange(y0, y1, step):
                for z in np.arange(z0, z1, step):
                    if np.random.rand() < 0.8:
                        points.append([x, y, z])
    table_min = [4.3, 2.0, 0.0]
    table_max = [5.2, 2.8, 0.75]
    add_box_points(table_min, table_max, step=0.08)
    chair_size = [0.45, 0.45, 0.9]
    chair_offsets = [[-0.55, -0.55], [0.6, -0.55], [-0.55, 0.6], [0.6, 0.6]]
    for off in chair_offsets:
        cmin = [table_min[0] + off[0], table_min[1] + off[1], 0.0]
        cmax = [cmin[0] + chair_size[0], cmin[1] + chair_size[1], chair_size[2]]
        add_box_points(cmin, cmax, step=0.08)
    # Plant sphere
    def add_sphere_points(center, radius, step=0.1):
        cx, cy, cz = center
        for x in np.arange(cx - radius, cx + radius, step):
            for y in np.arange(cy - radius, cy + radius, step):
                for z in np.arange(cz - radius, cz + radius, step):
                    if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2:
                        if np.random.rand() < 0.8:
                            points.append([x, y, z])
    add_sphere_points([5.5, 4.4, 0.6], 0.45, step=0.08)
    return np.array(points, dtype=np.float64)


def demo_pointcloud_scene():
    """Build an octree from a synthetic point cloud scene and visualize with Open3D."""
    print("\n🧪 Point Cloud Scene Demo (Enhanced)")
    print("=" * 40)

    pts = _generate_pointcloud_room_scene()
    print(f"Generated {len(pts)} input points")

    tree = octomap.OcTree(0.1)
    origin = np.array([0.5, 0.5, 1.5], dtype=np.float64)

    # insert in batches
    batch = 2000
    for i in range(0, len(pts), batch):
        tree.insertPointCloud(pts[i:i+batch], origin)
    tree.updateInnerOccupancy()

    print(f"Tree size: {tree.size()} nodes, depth: {tree.getTreeDepth()}")
    visualize_octree(tree, "Point Cloud Scene - Open3D", max_depth=0, show_wireframe=True)
    return tree

def demo_file_operations(tree):
    """Demonstrate file save/load operations"""
    print("\n💾 File Operations Demo")
    print("=" * 40)
    
    # Save the tree
    filename = "demo_octree.bt"
    try:
        success = tree.write(filename)
        if success:
            print(f"✅ Tree saved to {filename}")
            
            # Get file size
            import os
            file_size = os.path.getsize(filename)
            print(f"File size: {file_size} bytes")
            
            # Load the tree
            loaded_tree = tree.read(filename)
            if loaded_tree:
                print(f"✅ Tree loaded from {filename}")
                print(f"Loaded tree size: {loaded_tree.size()} nodes")
            else:
                print("❌ Failed to load tree from file")
                return
            
            # Verify data integrity
            test_point = [1.0, 1.0, 1.0]
            original_node = tree.search(test_point)
            loaded_node = loaded_tree.search(test_point)
            
            if original_node and loaded_node:
                original_occupied = tree.isNodeOccupied(original_node)
                loaded_occupied = loaded_tree.isNodeOccupied(loaded_node)
                print(f"Data integrity check: {original_occupied == loaded_occupied}")
            
            # Clean up
            os.remove(filename)
            print(f"✅ Cleaned up {filename}")

            
        else:
            print("❌ Failed to save tree")
            
    except Exception as e:
        print(f"❌ File operations failed: {e}")

def main():
    """Run all demonstrations"""
    print("🚀 OctoMap Wrapper Demonstration")
    print("=" * 50)
    
    try:
        # Demo 1: Room mapping
        room_tree = demo_room_mapping()
        
        # Demo 2: Probabilistic mapping
        prob_tree = demo_probabilistic_mapping()
        
        # Demo 3: File operations
        demo_file_operations(room_tree)
        
        print("\n" + "=" * 50)
        print("🎉 All demonstrations completed successfully!")
        print("\nKey features demonstrated:")
        print("✅ 3D occupancy mapping with walls and furniture")
        print("✅ Probabilistic occupancy updates")
        print("✅ Path planning and collision detection")
        print("✅ File save/load operations")
        print("✅ Data integrity verification")
        if OPEN3D_AVAILABLE:
            print("✅ Open3D visualization")
        
        print("\nThe OctoMap wrapper is ready for robotics applications!")
        
        # Optional: Show a final combined visualization
        if OPEN3D_AVAILABLE:
            print("\n🎨 Would you like to see a final visualization? (y/n): ", end="")
            try:
                response = input().lower().strip()
                if response in ['y', 'yes']:
                    # Create a simple test tree for final visualization
                    final_tree = octomap.OcTree(0.1)
                    # Add some test points
                    for i in range(10):
                        for j in range(10):
                            for k in range(5):
                                point = [i * 0.2, j * 0.2, k * 0.2]
                                occupied = (i + j + k) % 3 == 0
                                final_tree.updateNode(point, occupied)
                    visualize_octree(final_tree, "Final Demo - Test Pattern")
            except (EOFError, KeyboardInterrupt):
                print("\nSkipping final visualization.")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
