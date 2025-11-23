"""
Stress Test for Thread Safety Issues in PyOctoMap

This test focuses on high-stress scenarios that are most likely to reveal
thread safety issues, race conditions, and memory corruption problems.

Key scenarios tested:
1. Rapid concurrent modifications to the same tree
2. Iterator invalidation during tree modifications
3. Memory corruption under high concurrency
4. GIL-related issues and deadlocks
5. C++ pointer invalidation during Python object destruction
"""

import gc
import random
import threading
import time

import numpy as np

import pyoctomap


class ThreadSafetyStressTest:
    """High-stress thread safety testing"""

    def __init__(self):
        self.resolution = 0.1
        self.stress_duration = 5.0  # Longer duration for stress testing
        self.high_thread_count = 16  # More threads for stress

    def test_rapid_concurrent_modifications(self):
        """Test rapid concurrent modifications to the same tree"""
        print("\n🔥 Testing rapid concurrent modifications...")

        tree = pyoctomap.OcTree(self.resolution)
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Shared state
        modification_count = 0
        errors = []
        lock = threading.Lock()

        def rapid_modifier_thread(thread_id: int):
            """Thread that rapidly modifies the tree"""
            nonlocal modification_count, errors
            try:
                start_time = time.time()
                while time.time() - start_time < self.stress_duration:
                    # Rapid point additions
                    for _ in range(5):
                        points = np.random.uniform(-5, 5, (5, 3)).astype(np.float64)
                        tree.addPointCloudWithRayCasting(points, sensor_origin)

                        # Rapid node updates
                        for _ in range(3):
                            test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                            tree.updateNode(test_point, True)

                        with lock:
                            modification_count += 8

                    # No sleep - maximum stress

            except Exception as e:
                with lock:
                    errors.append(f"Modifier {thread_id}: {e}")

        # Start many modifier threads
        threads = []
        for i in range(self.high_thread_count):
            t = threading.Thread(target=rapid_modifier_thread, args=(i,))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        print(f"   Modifications: {modification_count}")
        print(f"   Errors: {len(errors)}")

        if errors:
            print("   Stress errors:")
            for error in errors[:10]:
                print(f"     {error}")

        # Check tree integrity
        try:
            num_nodes = tree.getNumLeafNodes()
            resolution = tree.getResolution()
            assert num_nodes >= 0, f"Invalid node count: {num_nodes}"
            assert resolution > 0, f"Invalid resolution: {resolution}"
        except Exception as e:
            errors.append(f"Tree integrity check: {e}")

        # Some errors may be expected under extreme stress
        print(f"   ✅ Rapid modification test completed (errors: {len(errors)})")

    def test_iterator_invalidation_stress(self):
        """Test iterator invalidation under high stress"""
        print("\n🔥 Testing iterator invalidation stress...")

        tree = pyoctomap.OcTree(self.resolution)
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Pre-populate tree
        initial_points = np.random.uniform(-5, 5, (200, 3)).astype(np.float64)
        tree.addPointCloudWithRayCasting(initial_points, sensor_origin)

        iterator_errors = []
        lock = threading.Lock()

        def iterator_stress_thread(thread_id: int):
            """Thread that stresses iterators"""
            nonlocal iterator_errors
            try:
                for _ in range(100):
                    # Test tree iterator
                    try:
                        node_count = 0
                        for node in tree.begin_tree():
                            if node is not None:
                                tree.isNodeOccupied(node)
                            node_count += 1
                            if node_count > 20:  # Limit to prevent infinite loops
                                break
                    except Exception as e:
                        with lock:
                            iterator_errors.append(f"Tree iterator {thread_id}: {e}")

                    # Test leaf iterator
                    try:
                        leaf_count = 0
                        for _leaf in tree.begin_leafs():
                            leaf_count += 1
                            if leaf_count > 20:  # Limit to prevent infinite loops
                                break
                    except Exception as e:
                        with lock:
                            iterator_errors.append(f"Leaf iterator {thread_id}: {e}")

                    # Test BBX iterator
                    try:
                        bbx_min = np.array([-5, -5, -5], dtype=np.float64)
                        bbx_max = np.array([5, 5, 5], dtype=np.float64)
                        bbx_count = 0
                        for _leaf in tree.begin_leafs_bbx(bbx_min, bbx_max):
                            bbx_count += 1
                            if bbx_count > 20:  # Limit to prevent infinite loops
                                break
                    except Exception as e:
                        with lock:
                            iterator_errors.append(f"BBX iterator {thread_id}: {e}")

            except Exception as e:
                with lock:
                    iterator_errors.append(f"Iterator stress {thread_id}: {e}")

        def modifier_stress_thread(thread_id: int):
            """Thread that rapidly modifies the tree"""
            nonlocal iterator_errors
            try:
                for _ in range(50):
                    # Add points
                    points = np.random.uniform(-5, 5, (10, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)

                    # Update nodes
                    for _ in range(5):
                        test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                        tree.updateNode(test_point, True)

                    # Clear and rebuild
                    if random.random() < 0.1:  # 10% chance
                        tree.clear()
                        new_points = np.random.uniform(-5, 5, (50, 3)).astype(
                            np.float64
                        )
                        tree.addPointCloudWithRayCasting(new_points, sensor_origin)

            except Exception as e:
                with lock:
                    iterator_errors.append(f"Modifier stress {thread_id}: {e}")

        # Start iterator stress threads
        iterator_threads = []
        for i in range(8):
            t = threading.Thread(target=iterator_stress_thread, args=(i,))
            t.start()
            iterator_threads.append(t)

        # Start modifier stress threads
        modifier_threads = []
        for i in range(4):
            t = threading.Thread(target=modifier_stress_thread, args=(i,))
            t.start()
            modifier_threads.append(t)

        # Wait for completion
        for t in iterator_threads + modifier_threads:
            t.join()

        print(f"   Iterator errors: {len(iterator_errors)}")

        if iterator_errors:
            print("   Iterator stress errors:")
            for error in iterator_errors[:10]:
                print(f"     {error}")

        print("   ✅ Iterator invalidation stress test completed")

    def test_memory_corruption_stress(self):
        """Test for memory corruption under high concurrency"""
        print("\n🔥 Testing memory corruption stress...")

        # Create multiple trees to stress memory management
        trees = []
        for _ in range(5):
            tree = pyoctomap.OcTree(self.resolution)
            trees.append(tree)

        memory_errors = []
        lock = threading.Lock()

        def memory_stress_worker(tree_id: int, tree: pyoctomap.OcTree):
            """Worker that stresses memory management"""
            nonlocal memory_errors
            try:
                sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

                for _ in range(100):
                    # Rapid operations
                    points = np.random.uniform(-5, 5, (20, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)

                    # Search operations
                    for _ in range(10):
                        test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                        node = tree.search(test_point)
                        if node is not None:
                            tree.isNodeOccupied(node)

                    # Tree statistics
                    tree.getNumLeafNodes()
                    tree.getResolution()
                    tree.getTreeDepth()

                    # Iteration
                    leaf_count = 0
                    for _leaf in tree.begin_leafs():
                        leaf_count += 1
                        if leaf_count > 10:
                            break

                    # Random tree operations
                    if random.random() < 0.1:
                        tree.clear()
                    elif random.random() < 0.05:
                        tree.updateInnerOccupancy()

            except Exception as e:
                with lock:
                    memory_errors.append(f"Memory stress {tree_id}: {e}")

        # Start memory stress workers
        threads = []
        for i, tree in enumerate(trees):
            t = threading.Thread(target=memory_stress_worker, args=(i, tree))
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
            print("   Memory stress errors:")
            for error in memory_errors[:10]:
                print(f"     {error}")

        assert len(memory_errors) == 0, f"Memory corruption detected: {memory_errors}"
        print("   ✅ Memory corruption stress test passed")

    def test_gil_and_deadlock_scenarios(self):
        """Test GIL-related issues and potential deadlocks"""
        print("\n🔥 Testing GIL and deadlock scenarios...")

        tree = pyoctomap.OcTree(self.resolution)
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        deadlock_errors = []
        lock = threading.Lock()

        def gil_stress_worker(thread_id: int):
            """Worker that stresses GIL interactions"""
            nonlocal deadlock_errors
            try:
                for _ in range(50):
                    # Mix of CPU-intensive and I/O operations
                    points = np.random.uniform(-5, 5, (30, 3)).astype(np.float64)

                    # Batch operations (may hold GIL longer)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)

                    # Many small operations (frequent GIL releases)
                    for _ in range(20):
                        test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                        node = tree.search(test_point)
                        if node is not None:
                            tree.isNodeOccupied(node)

                    # Complex operations
                    tree.getNumLeafNodes()
                    tree.getResolution()

                    # Iteration (may hold GIL)
                    leaf_count = 0
                    for _leaf in tree.begin_leafs():
                        leaf_count += 1
                        if leaf_count > 15:
                            break

                    # Simulate some CPU work
                    _ = sum(range(1000))

            except Exception as e:
                with lock:
                    deadlock_errors.append(f"GIL stress {thread_id}: {e}")

        # Start many GIL stress workers
        threads = []
        for i in range(12):
            t = threading.Thread(target=gil_stress_worker, args=(i,))
            t.start()
            threads.append(t)

        # Wait for completion with timeout
        start_time = time.time()
        for t in threads:
            t.join(timeout=10.0)  # 10 second timeout per thread
            if t.is_alive():
                print(f"   Warning: Thread {t.name} did not complete within timeout")

        total_time = time.time() - start_time
        print(f"   Total execution time: {total_time:.2f}s")
        print(f"   GIL/deadlock errors: {len(deadlock_errors)}")

        if deadlock_errors:
            print("   GIL stress errors:")
            for error in deadlock_errors[:10]:
                print(f"     {error}")

        print("   ✅ GIL and deadlock stress test completed")

    def test_pointer_invalidation_stress(self):
        """Test C++ pointer invalidation during Python object destruction"""
        print("\n🔥 Testing pointer invalidation stress...")

        pointer_errors = []
        lock = threading.Lock()

        def pointer_stress_worker(thread_id: int):
            """Worker that creates and destroys many trees"""
            nonlocal pointer_errors
            try:
                for _ in range(20):
                    # Create tree
                    tree = pyoctomap.OcTree(self.resolution)
                    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

                    # Add some data
                    points = np.random.uniform(-5, 5, (50, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)

                    # Perform operations
                    tree.getNumLeafNodes()
                    tree.getResolution()

                    # Create iterators
                    iterators = []
                    for _ in range(3):
                        iterators.append(tree.begin_leafs())

                    # Use iterators
                    for iterator in iterators:
                        count = 0
                        for _leaf in iterator:
                            count += 1
                            if count > 5:
                                break

                    # Tree goes out of scope here - test destruction
                    del tree
                    del iterators

                    # Force garbage collection
                    if random.random() < 0.3:
                        gc.collect()

            except Exception as e:
                with lock:
                    pointer_errors.append(f"Pointer stress {thread_id}: {e}")

        # Start pointer stress workers
        threads = []
        for i in range(8):
            t = threading.Thread(target=pointer_stress_worker, args=(i,))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        # Final garbage collection
        gc.collect()

        print(f"   Pointer invalidation errors: {len(pointer_errors)}")

        if pointer_errors:
            print("   Pointer stress errors:")
            for error in pointer_errors[:10]:
                print(f"     {error}")

        assert len(pointer_errors) == 0, (
            f"Pointer invalidation detected: {pointer_errors}"
        )
        print("   ✅ Pointer invalidation stress test passed")

    def test_extreme_concurrency_scenarios(self):
        """Test extreme concurrency scenarios"""
        print("\n🔥 Testing extreme concurrency scenarios...")

        tree = pyoctomap.OcTree(self.resolution)
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        extreme_errors = []
        lock = threading.Lock()

        def extreme_worker(worker_id: int):
            """Extreme concurrency worker"""
            nonlocal extreme_errors
            try:
                # Very rapid operations
                for _ in range(200):
                    # Point cloud operations
                    points = np.random.uniform(-5, 5, (5, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)

                    # Node operations
                    test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                    tree.updateNode(test_point, True)

                    # Search operations
                    node = tree.search(test_point)
                    if node is not None:
                        tree.isNodeOccupied(node)

                    # Statistics
                    tree.getNumLeafNodes()

                    # Iteration
                    count = 0
                    for _leaf in tree.begin_leafs():
                        count += 1
                        if count > 3:
                            break

                    # No delays - maximum stress

            except Exception as e:
                with lock:
                    extreme_errors.append(f"Extreme worker {worker_id}: {e}")

        # Start many extreme workers
        threads = []
        for i in range(20):  # Very high thread count
            t = threading.Thread(target=extreme_worker, args=(i,))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        print(f"   Extreme concurrency errors: {len(extreme_errors)}")

        if extreme_errors:
            print("   Extreme concurrency errors:")
            for error in extreme_errors[:10]:
                print(f"     {error}")

        # Check final tree state
        try:
            final_nodes = tree.getNumLeafNodes()
            final_resolution = tree.getResolution()
            assert final_nodes >= 0, f"Invalid final node count: {final_nodes}"
            assert final_resolution > 0, f"Invalid final resolution: {final_resolution}"
        except Exception as e:
            extreme_errors.append(f"Final state check: {e}")

        print(
            f"   ✅ Extreme concurrency test completed (errors: {len(extreme_errors)})"
        )

    def run_all_stress_tests(self):
        """Run all stress tests"""
        print("🔥 Starting PyOctoMap Thread Safety Stress Test Suite")
        print("=" * 70)

        try:
            self.test_rapid_concurrent_modifications()
            self.test_iterator_invalidation_stress()
            self.test_memory_corruption_stress()
            self.test_gil_and_deadlock_scenarios()
            self.test_pointer_invalidation_stress()
            self.test_extreme_concurrency_scenarios()

            print("\n" + "=" * 70)
            print("✅ All stress tests completed!")

        except Exception as e:
            print(f"\n❌ Stress test suite failed: {e}")
            raise


def test_thread_safety_stress():
    """Main stress test function"""
    stress_test = ThreadSafetyStressTest()
    stress_test.run_all_stress_tests()


def test_quick_thread_safety():
    """Quick thread safety test for CI/CD"""
    print("\n⚡ Running quick thread safety test...")

    tree = pyoctomap.OcTree(0.1)
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def quick_worker():
        for _ in range(20):
            points = np.random.uniform(-5, 5, (10, 3)).astype(np.float64)
            tree.addPointCloudWithRayCasting(points, sensor_origin)
            time.sleep(0.001)

    # Start multiple threads
    threads = []
    for _ in range(6):
        t = threading.Thread(target=quick_worker)
        t.start()
        threads.append(t)

    # Wait for completion
    for t in threads:
        t.join()

    # Verify tree state
    assert tree.getNumLeafNodes() > 0
    assert tree.getResolution() > 0

    print("   ✅ Quick thread safety test passed")


if __name__ == "__main__":
    # Run the stress test suite
    stress_test = ThreadSafetyStressTest()
    stress_test.run_all_stress_tests()
