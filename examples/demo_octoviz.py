#!/usr/bin/env python3
"""
OctoViz Visualization Example - Using the Official OctoMap Visualizer

This example demonstrates how to create an octree with PyOctoMap and visualize it
using octovis, the official OctoMap visualization tool.

OctoViz (octovis) is a powerful 3D visualization tool specifically designed for
OctoMap files. It provides interactive navigation, different rendering modes, and
advanced features like trajectory visualization.

Key Features Demonstrated:
- Creating a complex 3D scene with PyOctoMap
- Saving octree to .bt format (OctoMap binary format)
- Launching octovis for visualization
- Tips for using octovis effectively

Prerequisites:
- PyOctoMap installed
- octovis installed (part of OctoMap package)
  - On Ubuntu/Debian: sudo apt-get install octovis
  - On macOS: brew install octomap
  - Or build from source: https://github.com/OctoMap/octomap
"""

import numpy as np
import sys
import os
import subprocess
import shutil

# Add current directory to path for proper import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pyoctomap
    print("✅ PyOctoMap import successful!")
except ImportError as e:
    print(f"❌ Failed to import pyoctomap: {e}")
    sys.exit(1)

# Import Pointcloud if available
try:
    from pyoctomap import Pointcloud
    POINTCLOUD_AVAILABLE = True
except (ImportError, AttributeError):
    POINTCLOUD_AVAILABLE = False
    print("⚠️ Pointcloud not available - some features may be limited")


def check_octovis_available():
    """Check if octovis is available in the system PATH."""
    octovis_path = shutil.which('octovis')
    if octovis_path:
        print(f"✅ octovis found at: {octovis_path}")
        return True, octovis_path
    else:
        print("⚠️ octovis not found in PATH")
        print("   Please install octovis:")
        print("   - Ubuntu/Debian: sudo apt-get install octovis")
        print("   - macOS: brew install octomap")
        print("   - Or build from source: https://github.com/OctoMap/octomap")
        return False, None


def create_demo_scene():
    """
    Create an interesting 3D scene for visualization.
    This creates a room with furniture, walls, and architectural features.
    """
    print("\n🏗️ Creating Demo Scene")
    print("=" * 50)
    
    # Create octree with 0.1m resolution (10cm)
    tree = pyoctomap.OcTree(0.1)
    
    # Scene dimensions
    room_w, room_d, room_h = 8.0, 6.0, 3.5  # width, depth, height
    
    print(f"Room dimensions: {room_w}m x {room_d}m x {room_h}m")
    
    # Helper function to add a wall
    def add_wall(x_range, y_range, z_range, step=0.1):
        for x in np.arange(x_range[0], x_range[1] + step/2, step):
            for y in np.arange(y_range[0], y_range[1] + step/2, step):
                for z in np.arange(z_range[0], z_range[1] + step/2, step):
                    tree.updateNode([x, y, z], True)
    
    # Helper function to add a cylinder (pillar)
    def add_cylinder(center, radius, height, step=0.12):
        cx, cy, cz = center
        for x in np.arange(cx - radius, cx + radius + step/2, step):
            for y in np.arange(cy - radius, cy + radius + step/2, step):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2 + 1e-6:
                    for z in np.arange(cz, cz + height + step/2, step):
                        tree.updateNode([x, y, z], True)
    
    # Helper function to add a box
    def add_box(pmin, pmax, step=0.1):
        x0, y0, z0 = pmin
        x1, y1, z1 = pmax
        for x in np.arange(x0, x1 + step/2, step):
            for y in np.arange(y0, y1 + step/2, step):
                for z in np.arange(z0, z1 + step/2, step):
                    tree.updateNode([x, y, z], True)
    
    # Build the scene
    print("Building floor and ceiling...")
    # Floor
    add_wall([0.0, room_w], [0.0, room_d], [0.0, 0.1], step=0.15)
    # Ceiling
    add_wall([0.0, room_w], [0.0, room_d], [room_h - 0.1, room_h], step=0.15)
    
    print("Building walls...")
    # Four walls
    add_wall([0.0, room_w], [0.0, 0.15], [0.0, room_h], step=0.15)  # Front
    add_wall([0.0, room_w], [room_d - 0.15, room_d], [0.0, room_h], step=0.15)  # Back
    add_wall([0.0, 0.15], [0.0, room_d], [0.0, room_h], step=0.15)  # Left
    add_wall([room_w - 0.15, room_w], [0.0, room_d], [0.0, room_h], step=0.15)  # Right
    
    print("Adding architectural features...")
    # Pillars
    add_cylinder([1.0, 1.0, 0.0], 0.3, 3.0)
    add_cylinder([room_w - 1.0, room_d - 1.0, 0.0], 0.3, 3.0)
    add_cylinder([room_w - 1.0, 1.0, 0.0], 0.25, 2.5)
    
    print("Adding furniture...")
    # Large table in center
    add_box([3.0, 2.5, 0.0], [5.0, 3.5, 0.75], step=0.1)
    
    # Chairs around table
    chair_positions = [
        [2.5, 2.5, 0.0], [5.5, 2.5, 0.0],
        [3.0, 2.0, 0.0], [5.0, 4.0, 0.0]
    ]
    for pos in chair_positions:
        add_box([pos[0], pos[1], 0.0], 
                [pos[0] + 0.5, pos[1] + 0.5, 0.9], step=0.1)
    
    # Bookshelf on back wall
    add_box([1.0, room_d - 0.2, 0.3], [3.0, room_d - 0.1, 2.5], step=0.08)
    
    # Sofa
    sofa_length = 2.5
    sofa_width = 0.8
    sofa_height = 0.7
    sofa_x = 6.0
    sofa_y = 1.0
    add_box([sofa_x, sofa_y, 0.0], 
            [sofa_x + sofa_length, sofa_y + sofa_width, sofa_height], step=0.1)
    
    # Coffee table in front of sofa
    add_box([sofa_x + 0.3, sofa_y + sofa_width + 0.2, 0.0],
            [sofa_x + 0.3 + 1.0, sofa_y + sofa_width + 0.2 + 0.6, 0.4], step=0.1)
    
    print("Carving free space corridors...")
    # Create a free space corridor from door to center
    door_x = 4.0
    door_y = 0.0
    door_width = 1.0
    door_height = 2.2
    
    # Carve door opening
    for x in np.arange(door_x, door_x + door_width, 0.1):
        for z in np.arange(0.0, door_height, 0.1):
            tree.updateNode([x, 0.0, z], False)
    
    # Carve corridor from door to center
    corridor_path = [
        [door_x + door_width/2, 0.5, 0.6],
        [door_x + door_width/2, 1.5, 0.6],
        [4.0, 3.0, 0.6]
    ]
    for i in range(len(corridor_path) - 1):
        start = corridor_path[i]
        end = corridor_path[i + 1]
        steps = 30
        for j in range(steps + 1):
            t = j / steps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            z = start[2] + t * (end[2] - start[2])
            # Carve a corridor (free space)
            for dz in np.arange(-0.3, 1.5, 0.2):
                tree.updateNode([x, y, z + dz], False)
    
    # Update inner occupancy for proper tree structure
    tree.updateInnerOccupancy()
    
    print(f"✅ Scene created: {tree.size()} nodes")
    print(f"   Tree depth: {tree.getTreeDepth()}")
    print(f"   Resolution: {tree.getResolution()}m")
    
    return tree


def save_octree_for_octoviz(tree, filename="demo_octoviz.bt"):
    """Save the octree to a .bt file for visualization with octovis."""
    print(f"\n💾 Saving octree to {filename}...")
    
    try:
        # For .bt files, use writeBinary() instead of write()
        # write() creates .ot format, writeBinary() creates .bt format
        if filename.endswith('.bt'):
            success = tree.writeBinary(filename)
        else:
            success = tree.write(filename)
        
        if success:
            file_size = os.path.getsize(filename)
            print(f"✅ Octree saved successfully!")
            print(f"   File: {filename}")
            print(f"   Size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
            return True, filename
        else:
            print(f"❌ Failed to save octree")
            return False, None
    except Exception as e:
        print(f"❌ Error saving octree: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def launch_octoviz(octree_file, octovis_path=None):
    """
    Launch octovis to visualize the octree file.
    
    Args:
        octree_file: Path to the .bt file
        octovis_path: Optional path to octovis executable
    """
    print(f"\n🎨 Launching octovis...")
    print("=" * 50)
    
    if octovis_path:
        cmd = [octovis_path, octree_file]
    else:
        cmd = ['octovis', octree_file]
    
    print(f"Command: {' '.join(cmd)}")
    print("\n📖 OctoViz Usage Tips:")
    print("   - Mouse: Rotate view (left button), Pan (middle/right button)")
    print("   - Scroll: Zoom in/out")
    print("   - 'R': Reset view")
    print("   - 'F': Toggle free space visualization")
    print("   - 'O': Toggle occupied space visualization")
    print("   - 'U': Toggle unknown space visualization")
    print("   - 'T': Toggle tree structure visualization")
    print("   - 'C': Toggle color mode")
    print("   - 'H': Show/hide help")
    print("   - 'Q' or ESC: Quit")
    print("\n⏳ Starting octovis (this will open a new window)...")
    
    try:
        # Launch octovis (non-blocking)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ octovis launched (PID: {process.pid})")
        print("   Close the octovis window when done viewing.")
        return process
    except FileNotFoundError:
        print(f"❌ Could not find octovis executable")
        print(f"   Tried: {' '.join(cmd)}")
        return None
    except Exception as e:
        print(f"❌ Error launching octovis: {e}")
        return None


def print_octoviz_instructions():
    """Print detailed instructions for using octovis."""
    print("\n" + "=" * 70)
    print("📚 Complete OctoViz Guide")
    print("=" * 70)
    print("""
OctoViz (octovis) is the official 3D visualization tool for OctoMap files.
It provides interactive navigation and advanced visualization features.

INSTALLATION:
-------------
Ubuntu/Debian:
  sudo apt-get install octovis

macOS:
  brew install octomap

From Source:
  git clone https://github.com/OctoMap/octomap.git
  cd octomap
  mkdir build && cd build
  cmake ..
  make
  sudo make install

USAGE:
------
Command line:
  octovis <octree_file.bt>

Or from Python:
  import subprocess
  subprocess.Popen(['octovis', 'my_map.bt'])

KEYBOARD SHORTCUTS:
-------------------
Navigation:
  Mouse Left Button:    Rotate view
  Mouse Middle/Right:   Pan view
  Mouse Scroll:         Zoom in/out
  R:                    Reset view to default

Visualization Modes:
  F:                    Toggle free space (blue)
  O:                    Toggle occupied space (red)
  U:                    Toggle unknown space (gray)
  T:                    Toggle tree structure
  C:                    Toggle color mode

Other:
  H:                    Show/hide help overlay
  Q / ESC:             Quit application

FEATURES:
---------
- Interactive 3D navigation
- Multiple rendering modes
- Adjustable visualization parameters
- Support for trajectories and poses
- Export screenshots
- Multi-resolution viewing

FILE FORMAT:
------------
OctoViz reads .bt (binary tree) files created by PyOctoMap's write() method.
These files contain the complete octree structure with occupancy probabilities.
""")


def main():
    """Main demonstration function."""
    print("🚀 OctoViz Visualization Example")
    print("=" * 70)
    
    # Check if octovis is available
    octovis_available, octovis_path = check_octovis_available()
    
    # Create demo scene
    tree = create_demo_scene()
    
    # Save octree to file
    success, filename = save_octree_for_octoviz(tree)
    
    if not success:
        print("\n❌ Cannot proceed without saved octree file")
        return
    
    # Print instructions
    print_octoviz_instructions()
    
    # Launch octovis if available
    if octovis_available:
        print("\n" + "=" * 70)
        response = input("\n❓ Launch octovis now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            process = launch_octoviz(filename, octovis_path)
            if process:
                print("\n✅ octovis is running. Close the window when done.")
                print(f"   The octree file '{filename}' will remain for future use.")
            else:
                print("\n⚠️ Could not launch octovis automatically.")
                print(f"   You can manually launch it with: octovis {filename}")
        else:
            print(f"\n✅ Octree saved to '{filename}'")
            print(f"   Launch octovis manually with: octovis {filename}")
    else:
        print(f"\n✅ Octree saved to '{filename}'")
        print(f"   Install octovis and launch with: octovis {filename}")
    
    print("\n" + "=" * 70)
    print("🎉 Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
