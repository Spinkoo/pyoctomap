#!/usr/bin/env python3
"""
Test file specifically for creating and visualizing occupancy grids from OctoMap.
Shows the final occupancy state in 2D and 3D visualizations.
"""

import numpy as np
import sys
import os

# Add current directory to path for proper import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import octomap
    print("✅ OctoMap import successful!")
except ImportError as e:
    print(f"❌ Failed to import octomap: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
    print("✅ Matplotlib available for visualization")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib not available - visualization will be skipped")

def create_test_environment():
    """Create a test environment with solid 3D obstacles"""
    print("\n🏗️ Creating Test Environment with Solid Obstacles...")
    
    tree = octomap.OcTree(0.05)  # 5cm resolution
    
    # Create room boundaries (walls)
    print("  Adding room boundaries...")
    wall_points = []
    
    # Floor (solid)
    for x in np.arange(0, 4.0, 0.05):
        for y in np.arange(0, 4.0, 0.05):
            wall_points.append([x, y, 0])
    
    # Ceiling (solid)
    for x in np.arange(0, 4.0, 0.05):
        for y in np.arange(0, 4.0, 0.05):
            wall_points.append([x, y, 2.0])
    
    # Side walls (solid)
    for z in np.arange(0, 2.0, 0.05):
        for y in np.arange(0, 4.0, 0.05):
            wall_points.append([0, y, z])  # Left wall
            wall_points.append([4.0, y, z])  # Right wall
        for x in np.arange(0, 4.0, 0.05):
            wall_points.append([x, 0, z])  # Front wall
            wall_points.append([x, 4.0, z])  # Back wall
    
    # Add walls to octree
    for point in wall_points:
        tree.updateNode(point, True)
    
    print(f"  Added {len(wall_points)} boundary points")
    
    # Create solid 3D obstacles
    print("  Adding solid 3D obstacles...")
    obstacle_points = []
    
    # Obstacle 1: Large rectangular block (1.5m x 1.0m x 1.2m)
    print("    Adding rectangular block...")
    for x in np.arange(0.5, 2.0, 0.05):
        for y in np.arange(0.5, 1.5, 0.05):
            for z in np.arange(0.1, 1.3, 0.05):
                obstacle_points.append([x, y, z])
    
    # Obstacle 2: Cylindrical pillar (diameter 0.6m, height 1.8m)
    print("    Adding cylindrical pillar...")
    center_x, center_y = 3.0, 1.0
    radius = 0.3
    for x in np.arange(center_x - radius, center_x + radius + 0.05, 0.05):
        for y in np.arange(center_y - radius, center_y + radius + 0.05, 0.05):
            for z in np.arange(0.1, 1.9, 0.05):
                # Check if point is inside cylinder
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= radius:
                    obstacle_points.append([x, y, z])
    
    # Obstacle 3: L-shaped structure
    print("    Adding L-shaped structure...")
    # Horizontal part
    for x in np.arange(1.0, 2.5, 0.05):
        for y in np.arange(2.0, 2.3, 0.05):
            for z in np.arange(0.1, 0.8, 0.05):
                obstacle_points.append([x, y, z])
    # Vertical part
    for x in np.arange(2.0, 2.3, 0.05):
        for y in np.arange(2.0, 3.0, 0.05):
            for z in np.arange(0.1, 1.2, 0.05):
                obstacle_points.append([x, y, z])
    
    # Obstacle 4: Spherical obstacle (radius 0.4m)
    print("    Adding spherical obstacle...")
    sphere_center = [3.5, 3.0, 0.4]
    sphere_radius = 0.4
    for x in np.arange(sphere_center[0] - sphere_radius, sphere_center[0] + sphere_radius + 0.05, 0.05):
        for y in np.arange(sphere_center[1] - sphere_radius, sphere_center[1] + sphere_radius + 0.05, 0.05):
            for z in np.arange(0.1, 0.9, 0.05):
                # Check if point is inside sphere
                dist = np.sqrt((x - sphere_center[0])**2 + (y - sphere_center[1])**2 + (z - sphere_center[2])**2)
                if dist <= sphere_radius:
                    obstacle_points.append([x, y, z])
    
    # Obstacle 5: Staircase-like structure
    print("    Adding staircase structure...")
    for step in range(4):
        step_height = 0.2 + step * 0.2
        step_x_start = 0.2 + step * 0.3
        step_x_end = step_x_start + 0.25
        for x in np.arange(step_x_start, step_x_end, 0.05):
            for y in np.arange(3.2, 3.8, 0.05):
                for z in np.arange(0.1, step_height, 0.05):
                    obstacle_points.append([x, y, z])
    
    # Add obstacles to octree
    for point in obstacle_points:
        tree.updateNode(point, True)
    
    print(f"  Added {len(obstacle_points)} obstacle points")
    print(f"  Total tree size: {tree.size()} nodes")
    
    # Add some free space markers (optional - to show free areas)
    print("  Marking free space areas...")
    free_space_points = []
    
    # Mark some areas as explicitly free
    free_areas = [
        (0.1, 0.1, 0.1, 0.3, 0.3, 0.5),  # Small free area
        (2.2, 2.2, 0.1, 2.8, 2.8, 0.5),  # Another free area
    ]
    
    for x1, y1, z1, x2, y2, z2 in free_areas:
        for x in np.arange(x1, x2, 0.1):
            for y in np.arange(y1, y2, 0.1):
                for z in np.arange(z1, z2, 0.1):
                    point = [x, y, z]
                    # Only mark as free if not already occupied
                    node = tree.search(point)
                    if not node or not tree.isNodeOccupied(node):
                        tree.updateNode(point, False)
                        free_space_points.append(point)
    
    print(f"  Marked {len(free_space_points)} free space points")
    
    return tree

def create_2d_occupancy_grid(tree, resolution=0.1, max_coord=4.0, z_levels=None):
    """Create a 2D occupancy grid from the octree"""
    if z_levels is None:
        z_levels = [0.1, 0.5, 1.0, 1.5, 1.9]  # Sample at different heights
    
    print(f"\n📊 Creating 2D Occupancy Grid...")
    print(f"  Resolution: {resolution}m")
    print(f"  Z levels: {z_levels}")
    
    # Create grid dimensions
    grid_size = int(max_coord / resolution)
    occupancy_grid = np.zeros((grid_size, grid_size), dtype=float)
    
    for z in z_levels:
        print(f"  Sampling at Z = {z}m")
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * resolution
                y = j * resolution
                point = [x, y, z]
                
                node = tree.search(point)
                if node and tree.isNodeOccupied(node):
                    occupancy_grid[i, j] += 1.0
    
    # Normalize by number of Z levels
    occupancy_grid = occupancy_grid / len(z_levels)
    
    print(f"  Created {grid_size}x{grid_size} grid")
    print(f"  Max occupancy: {occupancy_grid.max():.2f}")
    print(f"  Min occupancy: {occupancy_grid.min():.2f}")
    print(f"  Mean occupancy: {occupancy_grid.mean():.2f}")
    
    return occupancy_grid, resolution

def visualize_2d_occupancy_grid(occupancy_grid, resolution, title="2D Occupancy Grid"):
    """Visualize the 2D occupancy grid with multiple views"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib not available - skipping visualization")
        return
    
    print(f"\n🎨 Visualizing {title}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Heatmap
    im1 = axes[0, 0].imshow(occupancy_grid, cmap='RdYlGn', aspect='equal', origin='lower')
    axes[0, 0].set_title('Heatmap View')
    axes[0, 0].set_xlabel('X (grid cells)')
    axes[0, 0].set_ylabel('Y (grid cells)')
    plt.colorbar(im1, ax=axes[0, 0], label='Occupancy Probability')
    
    # Contour plot
    x = np.arange(0, occupancy_grid.shape[1]) * resolution
    y = np.arange(0, occupancy_grid.shape[0]) * resolution
    X, Y = np.meshgrid(x, y)
    
    contour = axes[0, 1].contourf(X, Y, occupancy_grid, levels=20, cmap='RdYlGn')
    axes[0, 1].set_title('Contour Plot')
    axes[0, 1].set_xlabel('X (meters)')
    axes[0, 1].set_ylabel('Y (meters)')
    plt.colorbar(contour, ax=axes[0, 1], label='Occupancy Probability')
    
    # Binary occupancy (threshold at 0.5)
    binary_grid = (occupancy_grid > 0.5).astype(float)
    im3 = axes[1, 0].imshow(binary_grid, cmap='RdYlGn', aspect='equal', origin='lower')
    axes[1, 0].set_title('Binary Occupancy (threshold=0.5)')
    axes[1, 0].set_xlabel('X (grid cells)')
    axes[1, 0].set_ylabel('Y (grid cells)')
    plt.colorbar(im3, ax=axes[1, 0], label='Occupied (1) / Free (0)')
    
    # 3D surface plot
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    surf = ax4.plot_surface(X, Y, occupancy_grid, cmap='RdYlGn', alpha=0.8)
    ax4.set_title('3D Surface Plot')
    ax4.set_xlabel('X (meters)')
    ax4.set_ylabel('Y (meters)')
    ax4.set_zlabel('Occupancy Probability')
    plt.colorbar(surf, ax=ax4, label='Occupancy Probability')
    
    plt.suptitle(f'{title} - Resolution: {resolution}m')
    plt.tight_layout()
    print("  Close the window to continue...")
    plt.show()

def create_3d_occupancy_visualization(tree, resolution=0.1, max_coord=4.0):
    """Create a 3D occupancy visualization with solid obstacles"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib not available - skipping 3D visualization")
        return
    
    print(f"\n🎨 Creating 3D Occupancy Visualization...")
    
    occupied_points = []
    free_points = []
    
    # Sample the octree (every 2nd point for better visualization)
    grid_size = int(max_coord / resolution)
    sample_count = 0
    
    for i in range(0, grid_size, 2):
        for j in range(0, grid_size, 2):
            for k in range(0, int(2.0 / resolution), 2):
                x = i * resolution
                y = j * resolution
                z = k * resolution
                point = [x, y, z]
                
                node = tree.search(point)
                if node:
                    if tree.isNodeOccupied(node):
                        occupied_points.append(point)
                    else:
                        free_points.append(point)
                sample_count += 1
    
    print(f"  Sampled {sample_count} points:")
    print(f"    Occupied: {len(occupied_points)}")
    print(f"    Free: {len(free_points)}")
    
    # Create 3D visualization
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if occupied_points:
        occupied_array = np.array(occupied_points)
        # Color by height for better visualization
        colors = occupied_array[:, 2]  # Use Z coordinate for color
        scatter = ax.scatter(occupied_array[:, 0], occupied_array[:, 1], occupied_array[:, 2], 
                           c=colors, cmap='Reds', s=40, alpha=0.8, 
                           label=f'Occupied Obstacles ({len(occupied_points)})')
        plt.colorbar(scatter, ax=ax, label='Height (m)', shrink=0.8)
    
    if free_points:
        free_array = np.array(free_points)
        ax.scatter(free_array[:, 0], free_array[:, 1], free_array[:, 2], 
                  c='lightgreen', s=8, alpha=0.4, label=f'Free Space ({len(free_points)})')
    
    # Add coordinate frame
    ax.quiver(0, 0, 0, 1, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='X')
    ax.quiver(0, 0, 0, 0, 1, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Occupancy Map - Solid Obstacles')
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_xlim([0, max_coord])
    ax.set_ylim([0, max_coord])
    ax.set_zlim([0, 2.0])
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    print("  Close the window to continue...")
    plt.show()

def analyze_occupancy_statistics(occupancy_grid):
    """Analyze and print occupancy statistics"""
    print(f"\n📈 Occupancy Statistics:")
    print(f"  Grid size: {occupancy_grid.shape[0]}x{occupancy_grid.shape[1]}")
    print(f"  Total cells: {occupancy_grid.size}")
    print(f"  Occupied cells (>0.5): {np.sum(occupancy_grid > 0.5)}")
    print(f"  Free cells (<0.5): {np.sum(occupancy_grid < 0.5)}")
    print(f"  Unknown cells (=0.0): {np.sum(occupancy_grid == 0.0)}")
    print(f"  Max occupancy: {occupancy_grid.max():.3f}")
    print(f"  Min occupancy: {occupancy_grid.min():.3f}")
    print(f"  Mean occupancy: {occupancy_grid.mean():.3f}")
    print(f"  Std occupancy: {occupancy_grid.std():.3f}")

def main():
    """Run the occupancy grid test"""
    print("🚀 OctoMap Occupancy Grid Test")
    print("=" * 40)
    
    try:
        # Create test environment
        tree = create_test_environment()
        
        # Create 2D occupancy grid
        occupancy_grid, resolution = create_2d_occupancy_grid(tree)
        
        # Analyze statistics
        analyze_occupancy_statistics(occupancy_grid)
        
        # Visualize results
        if MATPLOTLIB_AVAILABLE:
            visualize_2d_occupancy_grid(occupancy_grid, resolution, "Final Occupancy Grid")
            create_3d_occupancy_visualization(tree)
        
        print("\n" + "=" * 40)
        print("🎉 Occupancy grid test completed!")
        print(f"✅ Tree size: {tree.size()} nodes")
        print(f"✅ Grid resolution: {resolution}m")
        print(f"✅ Grid size: {occupancy_grid.shape[0]}x{occupancy_grid.shape[1]}")
        if MATPLOTLIB_AVAILABLE:
            print("✅ Visualization completed")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
