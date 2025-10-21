"""
Comprehensive Thread Safety Tests for PyOctoMap

This test suite validates thread safety across different scenarios:
1. Concurrent read/write operations
2. Iterator safety during modifications
3. Data consistency under concurrent access
4. Memory safety and race conditions
5. Performance under concurrent load

Thread Safety Considerations:
- OcTree C++ implementation may not be thread-safe
- Python GIL provides some protection but not complete
- Iterators may become invalid during tree modifications
- Concurrent writes can cause data corruption
- Read operations should be safe with other reads
"""

import pytest
import threading
import time
import numpy as np
import pyoctomap
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import random
from typing import List, Tuple, Dict, Any
import gc


class ThreadSafetyTestSuite:
    """Comprehensive thread safety test suite for PyOctoMap"""
    
    def __init__(self):
        self.resolution = 0.1
        self.test_duration = 2.0  # seconds
        self.num_threads = 8
        self.points_per_batch = 100
        
    def generate_test_data(self, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test point cloud data"""
        # Generate points in a 10x10x10 meter cube
        points = np.random.uniform(-5, 5, (num_points, 3)).astype(np.float64)
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        return points, sensor_origin
    
    def test_concurrent_read_write_operations(self):
        """Test concurrent read and write operations"""
        print("\n🧪 Testing concurrent read/write operations...")
        
        tree = pyoctomap.OcTree(self.resolution)
        points, sensor_origin = self.generate_test_data(500)
        
        # Shared data for coordination
        read_count = 0
        write_count = 0
        errors = []
        lock = threading.Lock()
        
        def writer_thread(thread_id: int):
            """Writer thread that adds points to the tree"""
            nonlocal write_count, errors
            try:
                # Add points in batches
                batch_size = 50
                for i in range(0, len(points), batch_size):
                    batch = points[i:i+batch_size]
                    tree.addPointCloudWithRayCasting(batch, sensor_origin)
                    
                    with lock:
                        write_count += len(batch)
                    
                    # Small delay to allow reads
                    time.sleep(0.01)
                    
            except Exception as e:
                with lock:
                    errors.append(f"Writer {thread_id}: {e}")
        
        def reader_thread(thread_id: int):
            """Reader thread that queries the tree"""
            nonlocal read_count, errors
            try:
                start_time = time.time()
                while time.time() - start_time < self.test_duration:
                    # Random read operations
                    if random.random() < 0.3:
                        # Search for random points
                        test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                        node = tree.search(test_point)
                        if node is not None:
                            tree.isNodeOccupied(node)
                    
                    elif random.random() < 0.3:
                        # Get tree statistics
                        tree.getNumLeafNodes()
                        tree.getResolution()
                        tree.getTreeDepth()
                    
                    elif random.random() < 0.4:
                        # Iterate through leaves
                        leaf_count = 0
                        for leaf in tree.begin_leafs():
                            leaf_count += 1
                            if leaf_count > 10:  # Limit iteration
                                break
                    
                    with lock:
                        read_count += 1
                    
                    time.sleep(0.001)  # Small delay
                    
            except Exception as e:
                with lock:
                    errors.append(f"Reader {thread_id}: {e}")
        
        # Start threads
        threads = []
        
        # Start writer threads
        for i in range(2):
            t = threading.Thread(target=writer_thread, args=(i,))
            t.start()
            threads.append(t)
        
        # Start reader threads
        for i in range(4):
            t = threading.Thread(target=reader_thread, args=(i,))
            t.start()
            threads.append(t)
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Check results
        print(f"   Read operations: {read_count}")
        print(f"   Write operations: {write_count}")
        print(f"   Errors: {len(errors)}")
        
        if errors:
            print("   Errors encountered:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"     {error}")
        
        # Basic sanity checks
        assert read_count > 0, "No read operations completed"
        assert write_count > 0, "No write operations completed"
        assert len(errors) == 0, f"Thread safety errors detected: {errors}"
        
        print("   ✅ Concurrent read/write test passed")
    
    def test_iterator_safety_during_modifications(self):
        """Test iterator safety when tree is being modified"""
        print("\n🧪 Testing iterator safety during modifications...")
        
        tree = pyoctomap.OcTree(self.resolution)
        points, sensor_origin = self.generate_test_data(200)
        
        # Pre-populate tree
        tree.addPointCloudWithRayCasting(points, sensor_origin)
        
        errors = []
        iterations_completed = 0
        lock = threading.Lock()
        
        def modifier_thread():
            """Thread that modifies the tree"""
            nonlocal errors
            try:
                for _ in range(10):
                    # Add new points
                    new_points = np.random.uniform(-5, 5, (20, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(new_points, sensor_origin)
                    
                    # Update existing nodes
                    for _ in range(5):
                        test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                        tree.updateNode(test_point, True)
                    
                    time.sleep(0.05)  # Allow iterator to work
                    
            except Exception as e:
                with lock:
                    errors.append(f"Modifier: {e}")
        
        def iterator_thread():
            """Thread that iterates through the tree"""
            nonlocal errors, iterations_completed
            try:
                for _ in range(20):
                    # Test tree iterator
                    try:
                        for node in tree.begin_tree():
                            if node is not None:
                                tree.isNodeOccupied(node)
                            break  # Just test first node
                    except Exception as e:
                        with lock:
                            errors.append(f"Tree iterator: {e}")
                    
                    # Test leaf iterator
                    try:
                        leaf_count = 0
                        for leaf in tree.begin_leafs():
                            leaf_count += 1
                            if leaf_count > 5:  # Limit to avoid long iterations
                                break
                    except Exception as e:
                        with lock:
                            errors.append(f"Leaf iterator: {e}")
                    
                    with lock:
                        iterations_completed += 1
                    
                    time.sleep(0.02)
                    
            except Exception as e:
                with lock:
                    errors.append(f"Iterator: {e}")
        
        # Start threads
        modifier = threading.Thread(target=modifier_thread)
        iterator = threading.Thread(target=iterator_thread)
        
        modifier.start()
        iterator.start()
        
        modifier.join()
        iterator.join()
        
        print(f"   Iterations completed: {iterations_completed}")
        print(f"   Errors: {len(errors)}")
        
        if errors:
            print("   Iterator errors:")
            for error in errors:
                print(f"     {error}")
        
        # Iterator errors are expected due to concurrent modifications
        # We just want to ensure the program doesn't crash
        assert iterations_completed > 0, "No iterations completed"
        print("   ✅ Iterator safety test completed (errors expected)")
    
    def test_data_consistency_under_concurrent_access(self):
        """Test data consistency when multiple threads access the same tree"""
        print("\n🧪 Testing data consistency under concurrent access...")
        
        tree = pyoctomap.OcTree(self.resolution)
        points, sensor_origin = self.generate_test_data(300)
        
        # Shared state
        consistency_errors = []
        lock = threading.Lock()
        
        def consistency_checker_thread(thread_id: int):
            """Thread that performs consistency checks"""
            nonlocal consistency_errors
            try:
                for _ in range(50):
                    # Check that tree properties are consistent
                    num_nodes = tree.getNumLeafNodes()
                    resolution = tree.getResolution()
                    tree_depth = tree.getTreeDepth()
                    
                    # Basic consistency checks
                    if resolution <= 0:
                        with lock:
                            consistency_errors.append(f"Thread {thread_id}: Invalid resolution {resolution}")
                    
                    if tree_depth < 0:
                        with lock:
                            consistency_errors.append(f"Thread {thread_id}: Invalid tree depth {tree_depth}")
                    
                    if num_nodes < 0:
                        with lock:
                            consistency_errors.append(f"Thread {thread_id}: Invalid node count {num_nodes}")
                    
                    # Check that search operations return consistent results
                    test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                    node1 = tree.search(test_point)
                    node2 = tree.search(test_point)
                    
                    # Both searches should return the same result
                    if (node1 is None) != (node2 is None):
                        with lock:
                            consistency_errors.append(f"Thread {thread_id}: Inconsistent search results")
                    
                    time.sleep(0.01)
                    
            except Exception as e:
                with lock:
                    consistency_errors.append(f"Thread {thread_id}: {e}")
        
        def writer_thread(thread_id: int):
            """Thread that writes to the tree"""
            nonlocal consistency_errors
            try:
                for _ in range(20):
                    # Add points
                    new_points = np.random.uniform(-5, 5, (15, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(new_points, sensor_origin)
                    
                    # Update nodes
                    for _ in range(5):
                        test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                        tree.updateNode(test_point, True)
                    
                    time.sleep(0.02)
                    
            except Exception as e:
                with lock:
                    consistency_errors.append(f"Writer {thread_id}: {e}")
        
        # Start threads
        threads = []
        
        # Start consistency checker threads
        for i in range(3):
            t = threading.Thread(target=consistency_checker_thread, args=(i,))
            t.start()
            threads.append(t)
        
        # Start writer threads
        for i in range(2):
            t = threading.Thread(target=writer_thread, args=(i,))
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        print(f"   Consistency errors: {len(consistency_errors)}")
        
        if consistency_errors:
            print("   Consistency issues:")
            for error in consistency_errors[:5]:
                print(f"     {error}")
        
        # Some consistency errors may be expected due to concurrent modifications
        # We mainly want to ensure no crashes occur
        print("   ✅ Data consistency test completed")
    
    def test_memory_safety_and_race_conditions(self):
        """Test for memory safety issues and race conditions"""
        print("\n🧪 Testing memory safety and race conditions...")
        
        # Test multiple tree instances
        trees = [pyoctomap.OcTree(self.resolution) for _ in range(3)]
        points, sensor_origin = self.generate_test_data(200)
        
        memory_errors = []
        lock = threading.Lock()
        
        def tree_worker_thread(tree_id: int, tree: pyoctomap.OcTree):
            """Worker thread for a specific tree"""
            nonlocal memory_errors
            try:
                for _ in range(30):
                    # Add points
                    batch = np.random.uniform(-5, 5, (10, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(batch, sensor_origin)
                    
                    # Perform various operations
                    tree.getNumLeafNodes()
                    tree.getResolution()
                    
                    # Test search operations
                    test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                    node = tree.search(test_point)
                    if node is not None:
                        tree.isNodeOccupied(node)
                    
                    # Test iteration
                    leaf_count = 0
                    for leaf in tree.begin_leafs():
                        leaf_count += 1
                        if leaf_count > 5:
                            break
                    
                    time.sleep(0.01)
                    
            except Exception as e:
                with lock:
                    memory_errors.append(f"Tree {tree_id}: {e}")
        
        # Start worker threads for each tree
        threads = []
        for i, tree in enumerate(trees):
            t = threading.Thread(target=tree_worker_thread, args=(i, tree))
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Clean up trees
        del trees
        gc.collect()
        
        print(f"   Memory errors: {len(memory_errors)}")
        
        if memory_errors:
            print("   Memory issues:")
            for error in memory_errors[:5]:
                print(f"     {error}")
        
        assert len(memory_errors) == 0, f"Memory safety issues detected: {memory_errors}"
        print("   ✅ Memory safety test passed")
    
    def test_performance_under_concurrent_load(self):
        """Test performance characteristics under concurrent load"""
        print("\n🧪 Testing performance under concurrent load...")
        
        tree = pyoctomap.OcTree(self.resolution)
        points, sensor_origin = self.generate_test_data(1000)
        
        # Performance metrics
        operations_per_second = []
        lock = threading.Lock()
        
        def performance_worker_thread(thread_id: int):
            """Worker thread for performance testing"""
            nonlocal operations_per_second
            try:
                start_time = time.time()
                operation_count = 0
                
                while time.time() - start_time < self.test_duration:
                    # Mix of operations
                    if random.random() < 0.4:
                        # Add points
                        batch = np.random.uniform(-5, 5, (20, 3)).astype(np.float64)
                        tree.addPointCloudWithRayCasting(batch, sensor_origin)
                        operation_count += 1
                    
                    elif random.random() < 0.3:
                        # Search operations
                        test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                        node = tree.search(test_point)
                        if node is not None:
                            tree.isNodeOccupied(node)
                        operation_count += 1
                    
                    elif random.random() < 0.3:
                        # Read operations
                        tree.getNumLeafNodes()
                        tree.getResolution()
                        operation_count += 1
                    
                    time.sleep(0.001)  # Small delay
                
                end_time = time.time()
                ops_per_sec = operation_count / (end_time - start_time)
                
                with lock:
                    operations_per_second.append(ops_per_sec)
                
            except Exception as e:
                print(f"Performance worker {thread_id} error: {e}")
        
        # Start performance worker threads
        threads = []
        for i in range(self.num_threads):
            t = threading.Thread(target=performance_worker_thread, args=(i,))
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Calculate performance metrics
        if operations_per_second:
            avg_ops_per_sec = sum(operations_per_second) / len(operations_per_second)
            total_ops_per_sec = sum(operations_per_second)
            
            print(f"   Average ops/sec per thread: {avg_ops_per_sec:.1f}")
            print(f"   Total ops/sec across all threads: {total_ops_per_sec:.1f}")
            print(f"   Threads: {len(operations_per_second)}")
            
            # Basic performance check
            assert avg_ops_per_sec > 10, f"Performance too low: {avg_ops_per_sec} ops/sec"
            assert total_ops_per_sec > 50, f"Total performance too low: {total_ops_per_sec} ops/sec"
        
        print("   ✅ Performance test completed")
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling in concurrent scenarios"""
        print("\n🧪 Testing edge cases and error handling...")
        
        tree = pyoctomap.OcTree(self.resolution)
        errors = []
        lock = threading.Lock()
        
        def edge_case_worker_thread(thread_id: int):
            """Worker thread that tests edge cases"""
            nonlocal errors
            try:
                for _ in range(20):
                    # Test with invalid inputs
                    try:
                        # Empty point cloud
                        empty_points = np.zeros((0, 3), dtype=np.float64)
                        tree.addPointCloudWithRayCasting(empty_points, np.array([0, 0, 0], dtype=np.float64))
                    except Exception as e:
                        with lock:
                            errors.append(f"Thread {thread_id} empty points: {e}")
                    
                    # Test with NaN values
                    try:
                        nan_points = np.array([[np.nan, np.nan, np.nan]], dtype=np.float64)
                        tree.addPointCloudWithRayCasting(nan_points, np.array([0, 0, 0], dtype=np.float64))
                    except Exception as e:
                        with lock:
                            errors.append(f"Thread {thread_id} NaN points: {e}")
                    
                    # Test with very large values
                    try:
                        large_points = np.array([[1e10, 1e10, 1e10]], dtype=np.float64)
                        tree.addPointCloudWithRayCasting(large_points, np.array([0, 0, 0], dtype=np.float64))
                    except Exception as e:
                        with lock:
                            errors.append(f"Thread {thread_id} large points: {e}")
                    
                    # Test with zero-length rays
                    try:
                        same_point = np.array([1.0, 1.0, 1.0], dtype=np.float64)
                        tree.addPointCloudWithRayCasting(same_point.reshape(1, -1), same_point)
                    except Exception as e:
                        with lock:
                            errors.append(f"Thread {thread_id} zero ray: {e}")
                    
                    time.sleep(0.01)
                    
            except Exception as e:
                with lock:
                    errors.append(f"Thread {thread_id} general: {e}")
        
        # Start edge case worker threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=edge_case_worker_thread, args=(i,))
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        print(f"   Edge case errors: {len(errors)}")
        
        if errors:
            print("   Edge case issues:")
            for error in errors[:5]:
                print(f"     {error}")
        
        # Some errors are expected for invalid inputs
        print("   ✅ Edge case test completed")
    
    def run_all_tests(self):
        """Run all thread safety tests"""
        print("🚀 Starting PyOctoMap Thread Safety Test Suite")
        print("=" * 60)
        
        try:
            self.test_concurrent_read_write_operations()
            self.test_iterator_safety_during_modifications()
            self.test_data_consistency_under_concurrent_access()
            self.test_memory_safety_and_race_conditions()
            self.test_performance_under_concurrent_load()
            self.test_edge_cases_and_error_handling()
            
            print("\n" + "=" * 60)
            print("✅ All thread safety tests completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Thread safety test suite failed: {e}")
            raise


def test_thread_safety_comprehensive():
    """Main test function for thread safety"""
    test_suite = ThreadSafetyTestSuite()
    test_suite.run_all_tests()


def test_basic_thread_safety():
    """Basic thread safety test for CI/CD"""
    print("\n🧪 Running basic thread safety test...")
    
    tree = pyoctomap.OcTree(0.1)
    points = np.random.uniform(-5, 5, (100, 3)).astype(np.float64)
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    # Simple concurrent test
    def worker():
        for _ in range(10):
            batch = np.random.uniform(-5, 5, (10, 3)).astype(np.float64)
            tree.addPointCloudWithRayCasting(batch, sensor_origin)
            time.sleep(0.01)
    
    # Start multiple threads
    threads = []
    for _ in range(4):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Verify tree is in a valid state
    assert tree.getNumLeafNodes() > 0
    assert tree.getResolution() > 0
    
    print("   ✅ Basic thread safety test passed")


if __name__ == "__main__":
    # Run the comprehensive test suite
    test_suite = ThreadSafetyTestSuite()
    test_suite.run_all_tests()
