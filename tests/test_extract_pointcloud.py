#!/usr/bin/env python3
"""
Tests for extractPointCloud on CountingOcTree and ColorOcTree,
and for the simplified insertion methods.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pyoctomap import CountingOcTree, OcTree, ColorOcTree
    ALL_AVAILABLE = True
except ImportError:
    ALL_AVAILABLE = False
    pytest.skip("Required tree types not available", allow_module_level=True)


# ============================================================================
# CountingOcTree.extractPointCloud tests
# ============================================================================

class TestCountingExtractPointCloud:

    def test_empty_tree(self):
        tree = CountingOcTree(0.1)
        coords, counts = tree.extractPointCloud()
        assert coords.shape == (0, 3)
        assert counts.shape == (0,)
        assert coords.dtype == np.float64
        assert counts.dtype == np.uint32

    def test_single_point(self):
        tree = CountingOcTree(0.1)
        tree.updateNode([1.0, 2.0, 3.0])
        coords, counts = tree.extractPointCloud()
        assert coords.shape == (1, 3)
        assert counts.shape == (1,)
        assert counts[0] == 1
        # Coordinate should be near the input
        assert np.allclose(coords[0], [1.0, 2.0, 3.0], atol=0.1)

    def test_multiple_points(self):
        tree = CountingOcTree(0.1)
        input_pts = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        for pt in input_pts:
            tree.updateNode(pt)
        coords, counts = tree.extractPointCloud()
        assert len(coords) == 3
        assert all(c == 1 for c in counts)

    def test_repeated_updates_counted(self):
        tree = CountingOcTree(0.1)
        for _ in range(7):
            tree.updateNode([1.0, 2.0, 3.0])
        coords, counts = tree.extractPointCloud()
        assert len(coords) == 1
        assert counts[0] == 7

    def test_mixed_counts(self):
        tree = CountingOcTree(0.1)
        for _ in range(3):
            tree.updateNode([1.0, 0.0, 0.0])
        for _ in range(5):
            tree.updateNode([2.0, 0.0, 0.0])
        coords, counts = tree.extractPointCloud()
        assert len(coords) == 2
        count_set = set(counts)
        assert count_set == {3, 5}

    def test_large_batch(self):
        tree = CountingOcTree(0.1)
        np.random.seed(42)
        n = 200
        pts = np.random.rand(n, 3) * 10.0
        for pt in pts:
            tree.updateNode(pt.tolist())
        coords, counts = tree.extractPointCloud()
        # Should have at least 1 leaf and total counts == n
        assert len(coords) >= 1
        assert counts.sum() == n

    def test_consistency_with_getCentersMinHits(self):
        """extractPointCloud leaves should match getCentersMinHits(1)."""
        tree = CountingOcTree(0.1)
        for x in range(5):
            tree.updateNode([float(x), 0.0, 0.0])
        coords, counts = tree.extractPointCloud()
        centers = tree.getCentersMinHits(1)
        # Both should return the same number of leaves
        assert len(coords) == len(centers)


# ============================================================================
# ColorOcTree.extractPointCloud tests
# ============================================================================

class TestColorExtractPointCloud:

    def test_empty_tree(self):
        tree = ColorOcTree(0.1)
        occ, colors, free = tree.extractPointCloud()
        assert occ.shape == (0, 3)
        assert colors.shape == (0, 3)
        assert free.shape == (0, 3)

    def test_occupied_nodes_with_colors(self):
        tree = ColorOcTree(0.1)
        tree.updateNode([1.0, 2.0, 3.0], True)
        tree.setNodeColor([1.0, 2.0, 3.0], 255, 0, 0)
        tree.updateNode([4.0, 5.0, 6.0], True)
        tree.setNodeColor([4.0, 5.0, 6.0], 0, 255, 0)

        occ, colors, free = tree.extractPointCloud()
        assert len(occ) == 2
        assert len(colors) == 2
        assert colors.dtype == np.uint8
        # Should have red and green
        color_set = {tuple(c) for c in colors}
        assert (255, 0, 0) in color_set
        assert (0, 255, 0) in color_set

    def test_free_nodes(self):
        tree = ColorOcTree(0.1)
        tree.updateNode([1.0, 0.0, 0.0], True)   # occupied
        tree.updateNode([5.0, 0.0, 0.0], False)   # free

        occ, colors, free = tree.extractPointCloud()
        assert len(occ) >= 1
        assert len(free) >= 1
        assert len(occ) == len(colors)

    def test_return_types(self):
        tree = ColorOcTree(0.1)
        tree.updateNode([1.0, 0.0, 0.0], True)
        tree.setNodeColor([1.0, 0.0, 0.0], 128, 64, 32)

        occ, colors, free = tree.extractPointCloud()
        assert occ.dtype == np.float64
        assert colors.dtype == np.uint8
        assert free.dtype == np.float64


# ============================================================================
# OcTree.extractPointCloud (base) — ensure it still works
# ============================================================================

class TestBaseOcTreeExtract:

    def test_basic_extract(self):
        tree = OcTree(0.1)
        tree.updateNode([1.0, 0.0, 0.0], True)
        tree.updateNode([2.0, 0.0, 0.0], False)
        occ, empty = tree.extractPointCloud()
        assert isinstance(occ, np.ndarray)
        assert isinstance(empty, np.ndarray)
        assert occ.shape[1] == 3
        assert empty.shape[1] == 3


# ============================================================================
# CountingOcTree insertion method tests
# ============================================================================

class TestCountingInsertionMethods:

    def test_insertPointCloud_maxrange_truncation(self):
        """Points beyond maxrange should be truncated to maxrange distance."""
        tree = CountingOcTree(0.1)
        pts = np.array([[10.0, 0.0, 0.0]], dtype=np.float64)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tree.insertPointCloud(pts, origin, maxrange=3.0)

        # The point at 10.0 should be truncated to ~3.0 along x-axis
        node_at_10 = tree.search([10.0, 0.0, 0.0])
        assert node_at_10 is None, "Original far point should not exist"

        node_at_3 = tree.search([3.0, 0.0, 0.0])
        assert node_at_3 is not None, "Truncated point should exist near maxrange"
        assert node_at_3.getCount() >= 1

    def test_insertPointCloud_no_maxrange(self):
        """With maxrange=-1, points should be inserted as-is."""
        tree = CountingOcTree(0.1)
        pts = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float64)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tree.insertPointCloud(pts, origin, maxrange=-1.0)

        for pt in pts:
            node = tree.search(pt.tolist())
            assert node is not None
            assert node.getCount() >= 1

    def test_insertPointCloudRaysFast_delegates(self):
        """insertPointCloudRaysFast should raise DeprecationWarning and produce same result."""
        pts = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ], dtype=np.float64)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        tree1 = CountingOcTree(0.1)
        tree1.insertPointCloud(pts, origin, maxrange=3.0)

        tree2 = CountingOcTree(0.1)
        with pytest.warns(DeprecationWarning, match="insertPointCloudRaysFast is deprecated for CountingOcTree"):
            tree2.insertPointCloudRaysFast(pts, origin, max_range=3.0)

        # Both trees should have the same leaf counts
        coords1, counts1 = tree1.extractPointCloud()
        coords2, counts2 = tree2.extractPointCloud()
        assert len(coords1) == len(coords2)
        assert np.array_equal(np.sort(counts1), np.sort(counts2))

    def test_updateNodes_coords(self):
        tree = CountingOcTree(0.1)
        n = tree.updateNodes([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        assert n == 2
        coords, counts = tree.extractPointCloud()
        assert len(coords) == 2

    def test_updateNodes_empty(self):
        tree = CountingOcTree(0.1)
        assert tree.updateNodes([]) == 0
        assert tree.updateNodes(None) == 0

    def test_addPointWithRayCasting(self):
        tree = CountingOcTree(0.1)
        pt = np.array([3.0, 4.0, 5.0], dtype=np.float64)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        with pytest.warns(DeprecationWarning, match="addPointWithRayCasting is deprecated for CountingOcTree"):
            assert tree.addPointWithRayCasting(pt, origin) is True
            
        node = tree.search(pt.tolist())
        assert node is not None
        assert node.getCount() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
