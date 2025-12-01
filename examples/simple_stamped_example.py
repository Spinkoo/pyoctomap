#!/usr/bin/env python3
"""
Simple example demonstrating OcTreeStamped usage.

This example shows the basic workflow:
1. Create an OcTreeStamped
2. Update nodes (which updates timestamps)
3. Query node timestamps
4. Degrade outdated nodes
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for proper import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pyoctomap import OcTreeStamped, OcTreeNodeStamped
    print("✅ OcTreeStamped imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import OcTreeStamped: {e}")
    sys.exit(1)


def main():
    print("="*60)
    print("Simple OcTreeStamped Example")
    print("="*60)
    
    # 1. Create an OcTreeStamped with 0.1m resolution
    print("\n1. Creating OcTreeStamped...")
    tree = OcTreeStamped(0.1)
    print(f"   Resolution: {tree.getResolution()}m")
    print(f"   Tree type: {tree.getTreeType()}")
    
    # 2. Update nodes (this will set timestamps automatically)
    print("\n2. Updating nodes (timestamps are set automatically)...")
    coords = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    
    initial_time = time.time()
    for i, coord in enumerate(coords):
        # Update node as occupied - this creates the node and sets timestamp
        node = tree.updateNode(coord, True)  # True = occupied
        if node:
            print(f"   Updated node {i+1} at [{coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}]: "
                  f"timestamp = {node.getTimestamp()}")
        else:
            print(f"   Failed to update node at [{coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}]")
    
    # 3. Query timestamps
    print("\n3. Querying node timestamps...")
    for coord in coords:
        node = tree.search(coord)
        if node:
            timestamp = node.getTimestamp()
            print(f"   Location [{coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}]: "
                  f"timestamp = {timestamp}")
        else:
            print(f"   Location [{coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}]: Not found")
    
    # 4. Get last update time
    print("\n4. Tree statistics:")
    last_update = tree.getLastUpdateTime()
    print(f"   Last update time: {last_update}")
    print(f"   Current time: {int(time.time())}")
    print(f"   Total nodes: {tree.size()}")
    print(f"   Leaf nodes: {tree.getNumLeafNodes()}")
    
    # 5. Manual timestamp manipulation
    print("\n5. Manual timestamp manipulation example...")
    node = tree.search([1.0, 2.0, 3.0])
    if node:
        initial_ts = node.getTimestamp()
        print(f"   Initial timestamp: {initial_ts}")
        node.updateTimestamp()
        updated_ts = node.getTimestamp()
        print(f"   After updateTimestamp(): {updated_ts}")
        node.setTimestamp(1000)
        print(f"   After setTimestamp(1000): {node.getTimestamp()}")
    else:
        print("   Node not found")
    
    # 6. Degrade outdated nodes
    print("\n6. Degrading outdated nodes...")
    print("   (Nodes older than 1 second will be degraded)")
    tree.degradeOutdatedNodes(1)  # 1 second threshold
    print("   Degradation complete")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

