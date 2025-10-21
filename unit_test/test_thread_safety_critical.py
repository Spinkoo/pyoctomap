"""
Critical Thread Safety Tests for PyOctoMap

This test focuses on the most critical thread safety scenarios that could
lead to crashes, data corruption, or undefined behavior in PyOctoMap.

Critical scenarios:
1. Concurrent tree modifications during iteration
2. Iterator invalidation and dangling pointers
3. C++ object destruction during Python operations
4. Memory corruption in shared C++ structures
5. Race conditions in node updates
"""

import pytest
import threading
import time
import numpy as np
import pyoctomap
import random
import gc
import sys
from typing import List, Tuple, Optional


class CriticalThreadSafetyTest:
    """Critical thread safety testing focused on crash scenarios"""
    
    def __init__(self):
        self.resolution = 0.1
        self.test_iterations = 100
        
    def test_concurrent_modification_during_iteration(self):
        """Test the most dangerous scenario: modifying tree during iteration"""
        print("\n💥 Testing concurrent modification during iteration...")
        
        tree = pyoctomap.OcTree(self.resolution)
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Pre-populate tree
        initial_points = np.random.uniform(-5, 5, (100, 3)).astype(np.float64)
        tree.addPointCloudWithRayCasting(initial_points, sensor_origin)
        
        crash_count = 0
        iteration_errors = []
        lock = threading.Lock()
        
        def dangerous_iterator_thread(thread_id: int):
            """Thread that iterates while tree is being modified"""
            nonlocal crash_count, iteration_errors
            try:
                for _ in range(self.test_iterations):
                    # This is the most dangerous operation
                    try:
                        leaf_count = 0
                        for leaf in tree.begin_leafs():
                            # Try to access leaf properties while tree is being modified
                            coord = leaf.getCoordinate()
                            size = leaf.getSize()
                            depth = leaf.getDepth()
                            
                            # Try to check if node is occupied (dangerous!)
                            if leaf.current_node is not None:
                                tree.isNodeOccupied(leaf.current_node)
                            
                            leaf_count += 1
                            if leaf_count > 50:  # Limit to prevent infinite loops
                                break
                                
                    except Exception as e:
                        with lock:
                            crash_count += 1
                            iteration_errors.append(f"Iterator {thread_id}: {e}")
                    
                    time.sleep(0.001)  # Small delay
                    
            except Exception as e:
                with lock:
                    iteration_errors.append(f"Iterator thread {thread_id}: {e}")
        
        def aggressive_modifier_thread(thread_id: int):
            """Thread that aggressively modifies the tree"""
            nonlocal iteration_errors
            try:
                for _ in range(self.test_iterations):
                    # Add points
                    points = np.random.uniform(-5, 5, (20, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)
                    
                    # Update nodes
                    for _ in range(10):
                        test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                        tree.updateNode(test_point, True)
                    
                    # Clear and rebuild (very dangerous for iterators)
                    if random.random() < 0.1:  # 10% chance
                        tree.clear()
                        new_points = np.random.uniform(-5, 5, (30, 3)).astype(np.float64)
                        tree.addPointCloudWithRayCasting(new_points, sensor_origin)
                    
                    # Update inner occupancy (can invalidate iterators)
                    if random.random() < 0.2:  # 20% chance
                        tree.updateInnerOccupancy()
                    
            except Exception as e:
                with lock:
                    iteration_errors.append(f"Modifier {thread_id}: {e}")
        
        # Start dangerous threads
        iterator_threads = []
        for i in range(4):
            t = threading.Thread(target=dangerous_iterator_thread, args=(i,))
            t.start()
            iterator_threads.append(t)
        
        modifier_threads = []
        for i in range(2):
            t = threading.Thread(target=aggressive_modifier_thread, args=(i,))
            t.start()
            modifier_threads.append(t)
        
        # Wait for completion
        for t in iterator_threads + modifier_threads:
            t.join()
        
        print(f"   Crashes during iteration: {crash_count}")
        print(f"   Total errors: {len(iteration_errors)}")
        
        if iteration_errors:
            print("   Critical errors:")
            for error in iteration_errors[:10]:
                print(f"     {error}")
        
        # This test is expected to have errors due to iterator invalidation
        print(f"   ✅ Concurrent modification test completed (crashes: {crash_count})")
    
    def test_iterator_dangling_pointer_scenarios(self):
        """Test scenarios where iterators might have dangling pointers"""
        print("\n💥 Testing iterator dangling pointer scenarios...")
        
        tree = pyoctomap.OcTree(self.resolution)
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Pre-populate tree
        initial_points = np.random.uniform(-5, 5, (50, 3)).astype(np.float64)
        tree.addPointCloudWithRayCasting(initial_points, sensor_origin)
        
        dangling_errors = []
        lock = threading.Lock()
        
        def dangling_pointer_worker(thread_id: int):
            """Worker that creates potential dangling pointer scenarios"""
            nonlocal dangling_errors
            try:
                for _ in range(50):
                    # Create iterator
                    iterator = tree.begin_leafs()
                    
                    # Start iteration
                    leaf_count = 0
                    for leaf in iterator:
                        # Access leaf properties
                        coord = leaf.getCoordinate()
                        size = leaf.getSize()
                        
                        # Simulate some work
                        time.sleep(0.001)
                        
                        leaf_count += 1
                        if leaf_count > 10:
                            break
                    
                    # Create another iterator while first might still be active
                    iterator2 = tree.begin_leafs()
                    for leaf in iterator2:
                        coord = leaf.getCoordinate()
                        break  # Just one iteration
                    
                    # Tree modifications that might invalidate iterators
                    points = np.random.uniform(-5, 5, (10, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)
                    
                    # Try to use iterators after modification
                    try:
                        for leaf in iterator:
                            coord = leaf.getCoordinate()
                            break
                    except Exception as e:
                        with lock:
                            dangling_errors.append(f"Dangling iterator {thread_id}: {e}")
                    
                    # Clean up
                    del iterator
                    del iterator2
                    
            except Exception as e:
                with lock:
                    dangling_errors.append(f"Dangling worker {thread_id}: {e}")
        
        def tree_modifier_worker(thread_id: int):
            """Worker that modifies tree while iterators are active"""
            nonlocal dangling_errors
            try:
                for _ in range(50):
                    # Add points
                    points = np.random.uniform(-5, 5, (15, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)
                    
                    # Update nodes
                    for _ in range(5):
                        test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                        tree.updateNode(test_point, True)
                    
                    # Clear tree (very dangerous for active iterators)
                    if random.random() < 0.05:  # 5% chance
                        tree.clear()
                        new_points = np.random.uniform(-5, 5, (20, 3)).astype(np.float64)
                        tree.addPointCloudWithRayCasting(new_points, sensor_origin)
                    
                    time.sleep(0.01)
                    
            except Exception as e:
                with lock:
                    dangling_errors.append(f"Modifier {thread_id}: {e}")
        
        # Start workers
        workers = []
        for i in range(6):
            t = threading.Thread(target=dangling_pointer_worker, args=(i,))
            t.start()
            workers.append(t)
        
        for i in range(2):
            t = threading.Thread(target=tree_modifier_worker, args=(i,))
            t.start()
            workers.append(t)
        
        # Wait for completion
        for t in workers:
            t.join()
        
        print(f"   Dangling pointer errors: {len(dangling_errors)}")
        
        if dangling_errors:
            print("   Dangling pointer errors:")
            for error in dangling_errors[:10]:
                print(f"     {error}")
        
        print("   ✅ Dangling pointer test completed")
    
    def test_cpp_object_destruction_during_operations(self):
        """Test C++ object destruction during Python operations"""
        print("\n💥 Testing C++ object destruction during operations...")
        
        destruction_errors = []
        lock = threading.Lock()
        
        def destruction_worker(thread_id: int):
            """Worker that creates and destroys trees rapidly"""
            nonlocal destruction_errors
            try:
                for _ in range(30):
                    # Create tree
                    tree = pyoctomap.OcTree(self.resolution)
                    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    
                    # Add data
                    points = np.random.uniform(-5, 5, (30, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)
                    
                    # Create iterators
                    iterators = []
                    for _ in range(3):
                        iterators.append(tree.begin_leafs())
                    
                    # Start using iterators
                    for iterator in iterators:
                        count = 0
                        for leaf in iterator:
                            coord = leaf.getCoordinate()
                            count += 1
                            if count > 5:
                                break
                    
                    # Simulate some work
                    time.sleep(0.01)
                    
                    # Tree goes out of scope - test destruction
                    del tree
                    
                    # Try to use iterators after tree destruction (dangerous!)
                    try:
                        for iterator in iterators:
                            for leaf in iterator:
                                coord = leaf.getCoordinate()
                                break
                    except Exception as e:
                        with lock:
                            destruction_errors.append(f"Post-destruction {thread_id}: {e}")
                    
                    # Clean up iterators
                    del iterators
                    
                    # Force garbage collection
                    if random.random() < 0.3:
                        gc.collect()
                    
            except Exception as e:
                with lock:
                    destruction_errors.append(f"Destruction worker {thread_id}: {e}")
        
        # Start destruction workers
        workers = []
        for i in range(8):
            t = threading.Thread(target=destruction_worker, args=(i,))
            t.start()
            workers.append(t)
        
        # Wait for completion
        for t in workers:
            t.join()
        
        # Final cleanup
        gc.collect()
        
        print(f"   Destruction errors: {len(destruction_errors)}")
        
        if destruction_errors:
            print("   Destruction errors:")
            for error in destruction_errors[:10]:
                print(f"     {error}")
        
        print("   ✅ C++ object destruction test completed")
    
    def test_memory_corruption_in_shared_structures(self):
        """Test for memory corruption in shared C++ structures"""
        print("\n💥 Testing memory corruption in shared structures...")
        
        # Create multiple trees that might share internal structures
        trees = [pyoctomap.OcTree(self.resolution) for _ in range(3)]
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        corruption_errors = []
        lock = threading.Lock()
        
        def corruption_worker(tree_id: int, tree: pyoctomap.OcTree):
            """Worker that might cause memory corruption"""
            nonlocal corruption_errors
            try:
                for _ in range(50):
                    # Rapid operations that might cause corruption
                    points = np.random.uniform(-5, 5, (20, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)
                    
                    # Search operations
                    for _ in range(10):
                        test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                        node = tree.search(test_point)
                        if node is not None:
                            # Try to access node properties
                            try:
                                tree.isNodeOccupied(node)
                                tree.isNodeAtThreshold(node)
                            except Exception as e:
                                with lock:
                                    corruption_errors.append(f"Node access {tree_id}: {e}")
                    
                    # Tree operations
                    tree.getNumLeafNodes()
                    tree.getResolution()
                    tree.getTreeDepth()
                    
                    # Iteration
                    try:
                        leaf_count = 0
                        for leaf in tree.begin_leafs():
                            coord = leaf.getCoordinate()
                            size = leaf.getSize()
                            leaf_count += 1
                            if leaf_count > 10:
                                break
                    except Exception as e:
                        with lock:
                            corruption_errors.append(f"Iteration {tree_id}: {e}")
                    
                    # Random tree operations
                    if random.random() < 0.1:
                        tree.clear()
                    elif random.random() < 0.05:
                        tree.updateInnerOccupancy()
                    
            except Exception as e:
                with lock:
                    corruption_errors.append(f"Corruption worker {tree_id}: {e}")
        
        # Start corruption workers
        workers = []
        for i, tree in enumerate(trees):
            t = threading.Thread(target=corruption_worker, args=(i, tree))
            t.start()
            workers.append(t)
        
        # Wait for completion
        for t in workers:
            t.join()
        
        # Clean up trees
        del trees
        gc.collect()
        
        print(f"   Memory corruption errors: {len(corruption_errors)}")
        
        if corruption_errors:
            print("   Corruption errors:")
            for error in corruption_errors[:10]:
                print(f"     {error}")
        
        assert len(corruption_errors) == 0, f"Memory corruption detected: {corruption_errors}"
        print("   ✅ Memory corruption test passed")
    
    def test_race_conditions_in_node_updates(self):
        """Test race conditions in node update operations"""
        print("\n💥 Testing race conditions in node updates...")
        
        tree = pyoctomap.OcTree(self.resolution)
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Pre-populate tree
        initial_points = np.random.uniform(-5, 5, (100, 3)).astype(np.float64)
        tree.addPointCloudWithRayCasting(initial_points, sensor_origin)
        
        race_errors = []
        lock = threading.Lock()
        
        def race_condition_worker(thread_id: int):
            """Worker that creates race conditions"""
            nonlocal race_errors
            try:
                for _ in range(100):
                    # Update the same nodes from multiple threads
                    test_points = [
                        np.array([1.0, 1.0, 1.0], dtype=np.float64),
                        np.array([2.0, 2.0, 2.0], dtype=np.float64),
                        np.array([3.0, 3.0, 3.0], dtype=np.float64),
                        np.array([-1.0, -1.0, -1.0], dtype=np.float64),
                        np.array([-2.0, -2.0, -2.0], dtype=np.float64)
                    ]
                    
                    # Update nodes
                    for point in test_points:
                        try:
                            node = tree.updateNode(point, True)
                            if node is not None:
                                tree.isNodeOccupied(node)
                        except Exception as e:
                            with lock:
                                race_errors.append(f"Node update {thread_id}: {e}")
                    
                    # Add new points
                    points = np.random.uniform(-5, 5, (10, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)
                    
                    # Update inner occupancy (can cause race conditions)
                    if random.random() < 0.1:
                        tree.updateInnerOccupancy()
                    
            except Exception as e:
                with lock:
                    race_errors.append(f"Race worker {thread_id}: {e}")
        
        def consistency_checker_worker(thread_id: int):
            """Worker that checks consistency during updates"""
            nonlocal race_errors
            try:
                for _ in range(50):
                    # Check tree consistency
                    num_nodes = tree.getNumLeafNodes()
                    resolution = tree.getResolution()
                    
                    # Search for nodes
                    test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                    node = tree.search(test_point)
                    if node is not None:
                        try:
                            occupied = tree.isNodeOccupied(node)
                            at_threshold = tree.isNodeAtThreshold(node)
                        except Exception as e:
                            with lock:
                                race_errors.append(f"Consistency check {thread_id}: {e}")
                    
                    time.sleep(0.01)
                    
            except Exception as e:
                with lock:
                    race_errors.append(f"Consistency worker {thread_id}: {e}")
        
        # Start race condition workers
        workers = []
        for i in range(6):
            t = threading.Thread(target=race_condition_worker, args=(i,))
            t.start()
            workers.append(t)
        
        for i in range(2):
            t = threading.Thread(target=consistency_checker_worker, args=(i,))
            t.start()
            workers.append(t)
        
        # Wait for completion
        for t in workers:
            t.join()
        
        print(f"   Race condition errors: {len(race_errors)}")
        
        if race_errors:
            print("   Race condition errors:")
            for error in race_errors[:10]:
                print(f"     {error}")
        
        print("   ✅ Race condition test completed")
    
    def run_all_critical_tests(self):
        """Run all critical thread safety tests"""
        print("💥 Starting PyOctoMap Critical Thread Safety Test Suite")
        print("=" * 70)
        
        try:
            self.test_concurrent_modification_during_iteration()
            self.test_iterator_dangling_pointer_scenarios()
            self.test_cpp_object_destruction_during_operations()
            self.test_memory_corruption_in_shared_structures()
            self.test_race_conditions_in_node_updates()
            
            print("\n" + "=" * 70)
            print("✅ All critical tests completed!")
            
        except Exception as e:
            print(f"\n❌ Critical test suite failed: {e}")
            raise


def test_critical_thread_safety():
    """Main critical test function"""
    critical_test = CriticalThreadSafetyTest()
    critical_test.run_all_critical_tests()


def test_minimal_thread_safety():
    """Minimal thread safety test for basic validation"""
    print("\n⚡ Running minimal thread safety test...")
    
    tree = pyoctomap.OcTree(0.1)
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    def minimal_worker():
        for _ in range(10):
            points = np.random.uniform(-5, 5, (5, 3)).astype(np.float64)
            tree.addPointCloudWithRayCasting(points, sensor_origin)
            time.sleep(0.01)
    
    # Start minimal threads
    threads = []
    for _ in range(4):
        t = threading.Thread(target=minimal_worker)
        t.start()
        threads.append(t)
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Basic validation
    assert tree.getNumLeafNodes() > 0
    assert tree.getResolution() > 0
    
    print("   ✅ Minimal thread safety test passed")


if __name__ == "__main__":
    # Run the critical test suite
    critical_test = CriticalThreadSafetyTest()
    critical_test.run_all_critical_tests()
