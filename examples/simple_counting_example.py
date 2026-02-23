#!/usr/bin/env python3
"""
Simple example demonstrating CountingOcTree usage.

This example shows the basic workflow:
1. Create a CountingOcTree
2. Update nodes (which increments their count)
3. Query node counts
4. Find frequently observed locations
"""

import numpy as np
import sys
import os

# Add parent directory to path for proper import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pyoctomap import CountingOcTree, CountingOcTreeNode
    print("✅ CountingOcTree imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import CountingOcTree: {e}")
    sys.exit(1)


def main():
    print("="*60)
    print("Simple CountingOcTree Example")
    print("="*60)
    
    # 1. Create a CountingOcTree with 0.1m resolution
    print("\n1. Creating CountingOcTree...")
    tree = CountingOcTree(0.1)
    print(f"   Resolution: {tree.getResolution()}m")
    
    # 2. Simulate sensor observations
    print("\n2. Simulating sensor observations...")
    observations = [
        [1.0, 2.0, 3.0],  # Location A - observed once
        [1.0, 2.0, 3.0],  # Location A - observed again (count = 2)
        [1.0, 2.0, 3.0],  # Location A - observed again (count = 3)
        [4.0, 5.0, 6.0],  # Location B - observed once
        [4.0, 5.0, 6.0],  # Location B - observed again (count = 2)
        [7.0, 8.0, 9.0],  # Location C - observed once
    ]
    
    for i, obs in enumerate(observations):
        node = tree.updateNode(obs)
        if node:
            print(f"   Observation {i+1} at [{obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}]: Count = {node.getCount()}")
    
    # 3. Query node counts
    print("\n3. Querying node counts...")
    locations = [
        [1.0, 2.0, 3.0],  # Location A - should have count = 3
        [4.0, 5.0, 6.0],  # Location B - should have count = 2
        [7.0, 8.0, 9.0],  # Location C - should have count = 1
        [10.0, 11.0, 12.0],  # Location D - not observed, should return None
    ]
    
    for loc in locations:
        node = tree.search(loc)
        if node:
            print(f"   Location [{loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f}]: Count = {node.getCount()}")
        else:
            print(f"   Location [{loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f}]: Not observed")
    
    # 4. Find frequently observed locations (count >= 2)
    print("\n4. Finding frequently observed locations (count >= 2)...")
    frequent_centers = tree.getCentersMinHits(2)
    print(f"   Found {len(frequent_centers)} locations with count >= 2:")
    for i, center in enumerate(frequent_centers):
        node = tree.search(center)
        if node:
            print(f"   [{i+1}] [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]: Count = {node.getCount()}")
    
    # 5. Manual count manipulation
    print("\n5. Manual count manipulation example...")
    node = tree.search([1.0, 2.0, 3.0])
    if node:
        print(f"   Initial count: {node.getCount()}")
        node.increaseCount()
        print(f"   After increaseCount(): {node.getCount()}")
        node.setCount(100)
        print(f"   After setCount(100): {node.getCount()}")
    
    # 6. Tree statistics
    print("\n6. Tree statistics:")
    print(f"   Total nodes: {tree.size()}")
    print(f"   Leaf nodes: {tree.getNumLeafNodes()}")
    print(f"   Memory usage: {tree.memoryUsage()} bytes")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

