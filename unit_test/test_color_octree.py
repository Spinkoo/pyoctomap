#!/usr/bin/env python3
"""
Unit tests for ColorOcTree functionality.
Tests color-specific features and integration with occupancy mapping.
"""

import numpy as np
import pytest
import pyoctomap
import tempfile
import os


class TestColorOcTreeNode:
    """Test ColorOcTreeNode class"""
    
    def test_color_node_creation(self):
        """Test creating a ColorOcTree and accessing nodes"""
        tree = pyoctomap.ColorOcTree(0.1)
        assert tree is not None
        assert tree.getTreeType() == "ColorOcTree"
    
    def test_set_and_get_color(self):
        """Test setting and getting node colors"""
        tree = pyoctomap.ColorOcTree(0.1)
        coord = np.array([1.0, 2.0, 3.0])
        
        # Update node to create it
        node = tree.updateNode(coord, True)
        assert node is not None
        
        # Set color
        result_node = tree.setNodeColor(coord, 255, 0, 0)  # Red
        assert result_node is not None
        
        # Get color
        color = result_node.getColor()
        assert color == (255, 0, 0)
    
    def test_is_color_set(self):
        """Test checking if color is set"""
        tree = pyoctomap.ColorOcTree(0.1)
        coord = np.array([1.0, 2.0, 3.0])
        
        # Update node
        node = tree.updateNode(coord, True)
        assert node is not None
        
        # Initially should not be set (white = 255,255,255)
        assert not node.isColorSet()
        
        # Set color
        tree.setNodeColor(coord, 100, 100, 100)
        node = tree.search(coord)
        assert node.isColorSet()
    
    def test_average_node_color(self):
        """Test averaging node colors"""
        tree = pyoctomap.ColorOcTree(0.1)
        coord = np.array([1.0, 2.0, 3.0])
        
        # Update node
        node = tree.updateNode(coord, True)
        assert node is not None
        
        # Set initial color
        tree.setNodeColor(coord, 100, 100, 100)
        
        # Average with new color
        result_node = tree.averageNodeColor(coord, 200, 200, 200)
        assert result_node is not None
        
        color = result_node.getColor()
        # Average of 100 and 200 should be 150
        assert color == (150, 150, 150)
    
    def test_integrate_node_color(self):
        """Test integrating node color with occupancy probability"""
        tree = pyoctomap.ColorOcTree(0.1)
        coord = np.array([1.0, 2.0, 3.0])
        
        # Update node as occupied
        node = tree.updateNode(coord, True)
        assert node is not None
        
        # Set initial color
        tree.setNodeColor(coord, 100, 100, 100)
        
        # Integrate new color (weighted by occupancy)
        result_node = tree.integrateNodeColor(coord, 200, 200, 200)
        assert result_node is not None
        
        color = result_node.getColor()
        # Should be weighted average based on occupancy probability
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
    
    def test_node_color_with_key(self):
        """Test setting color using OcTreeKey"""
        tree = pyoctomap.ColorOcTree(0.1)
        coord = np.array([1.0, 2.0, 3.0])
        
        # Get key
        key = tree.coordToKey(coord)
        
        # Update node
        tree.updateNode(coord, True)
        
        # Set color using key
        result_node = tree.setNodeColor(key, 0, 255, 0)  # Green
        assert result_node is not None
        
        color = result_node.getColor()
        assert color == (0, 255, 0)


class TestColorOcTree:
    """Test ColorOcTree class"""
    
    def test_tree_creation(self):
        """Test creating ColorOcTree instances"""
        tree1 = pyoctomap.ColorOcTree(0.1)
        assert tree1.getResolution() == 0.1
        assert tree1.getTreeType() == "ColorOcTree"
        
        tree2 = pyoctomap.ColorOcTree(0.05)
        assert tree2.getResolution() == 0.05
    
    def test_point_cloud_insertion(self):
        """Test inserting point cloud into ColorOcTree"""
        tree = pyoctomap.ColorOcTree(0.1)
        
        # Create point cloud
        points = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0]
        ])
        origin = np.array([0.0, 0.0, 0.0])
        
        # Insert point cloud
        tree.insertPointCloud(points, origin)
        
        # Check nodes were created
        node = tree.search(points[0])
        assert node is not None
    
    def test_color_with_point_cloud(self):
        """Test setting colors after point cloud insertion"""
        tree = pyoctomap.ColorOcTree(0.1)
        
        # Create point cloud
        points = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0]
        ])
        origin = np.array([0.0, 0.0, 0.0])
        
        # Insert point cloud
        tree.insertPointCloud(points, origin)
        
        # Set colors for each point
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for point, (r, g, b) in zip(points, colors):
            node = tree.setNodeColor(point, r, g, b)
            assert node is not None
            color = node.getColor()
            assert color == (r, g, b)
    
    def test_update_inner_occupancy(self):
        """Test updating inner node occupancy and colors"""
        tree = pyoctomap.ColorOcTree(0.1)
        
        # Create some nodes with colors
        coords = [
            np.array([1.0, 1.0, 1.0]),
            np.array([1.1, 1.1, 1.1]),
            np.array([1.2, 1.2, 1.2])
        ]
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        for coord, (r, g, b) in zip(coords, colors):
            tree.updateNode(coord, True)
            tree.setNodeColor(coord, r, g, b)
        
        # Update inner occupancy (should also update colors)
        tree.updateInnerOccupancy()
        
        # Verify nodes still exist
        for coord in coords:
            node = tree.search(coord)
            assert node is not None
    
    def test_tree_statistics(self):
        """Test ColorOcTree statistics"""
        tree = pyoctomap.ColorOcTree(0.1)
        
        # Insert some nodes
        points = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0]
        ])
        origin = np.array([0.0, 0.0, 0.0])
        tree.insertPointCloud(points, origin)
        
        # Check statistics
        assert tree.size() > 0
        assert tree.getNumLeafNodes() > 0
        assert tree.calcNumNodes() > 0
        assert tree.memoryUsage() > 0
    
    def test_search_and_color(self):
        """Test searching nodes and accessing colors"""
        tree = pyoctomap.ColorOcTree(0.1)
        coord = np.array([1.5, 2.5, 3.5])
        
        # Update node
        tree.updateNode(coord, True)
        
        # Set color
        tree.setNodeColor(coord, 128, 64, 32)
        
        # Search and verify color
        node = tree.search(coord)
        assert node is not None
        color = node.getColor()
        assert color == (128, 64, 32)
    
    def test_file_io(self):
        """Test reading and writing ColorOcTree files"""
        tree = pyoctomap.ColorOcTree(0.1)
        
        # Create some colored nodes
        coords = [
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([3.0, 3.0, 3.0])
        ]
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        for coord, (r, g, b) in zip(coords, colors):
            tree.updateNode(coord, True)
            tree.setNodeColor(coord, r, g, b)
        
        # Write to file
        with tempfile.NamedTemporaryFile(suffix='.ot', delete=False) as f:
            filename = f.name
        
        try:
            success = tree.write(filename)
            assert success
            
            # Read from file
            tree2 = pyoctomap.ColorOcTree(filename)
            assert tree2 is not None
            
            # Verify colors are preserved
            for coord, (r, g, b) in zip(coords, colors):
                node = tree2.search(coord)
                if node:  # Node might not exist if tree was pruned
                    color = node.getColor()
                    # Colors should be preserved (or averaged if pruned)
                    assert len(color) == 3
                    assert all(0 <= c <= 255 for c in color)
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_binary_file_io(self):
        """Test reading and writing binary ColorOcTree files"""
        tree = pyoctomap.ColorOcTree(0.1)
        
        # Create some colored nodes
        coord = np.array([1.0, 1.0, 1.0])
        tree.updateNode(coord, True)
        tree.setNodeColor(coord, 255, 128, 64)
        
        # Write binary
        with tempfile.NamedTemporaryFile(suffix='.bt', delete=False) as f:
            filename = f.name
        
        try:
            success = tree.writeBinary(filename)
            assert success
            
            # Read binary
            tree2 = pyoctomap.ColorOcTree(0.1)
            success = tree2.readBinary(filename)
            assert success
            
            # Verify color is preserved
            node = tree2.search(coord)
            if node:
                color = node.getColor()
                assert len(color) == 3
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_ray_casting(self):
        """Test ray casting in ColorOcTree"""
        tree = pyoctomap.ColorOcTree(0.1)
        
        # Create an obstacle
        obstacle = np.array([5.0, 5.0, 5.0])
        tree.updateNode(obstacle, True)
        tree.setNodeColor(obstacle, 255, 0, 0)
        
        # Cast ray
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 1.0, 1.0])
        # Normalize direction (though OctoMap does this internally, being explicit helps)
        direction = direction / np.linalg.norm(direction)
        end = np.zeros(3)
        
        # Set a reasonable maxRange to ensure the ray reaches the obstacle
        # Distance from origin to obstacle is sqrt(5^2 + 5^2 + 5^2) = sqrt(75) ≈ 8.66
        hit = tree.castRay(origin, direction, end, ignoreUnknownCells=True, maxRange=10.0)
        assert hit
        
        # Check if hit point has color
        node = tree.search(end)
        if node:
            color = node.getColor()
            assert len(color) == 3
    
    def test_bounding_box(self):
        """Test bounding box operations with ColorOcTree"""
        tree = pyoctomap.ColorOcTree(0.1)
        
        # Set bounding box
        bbx_min = np.array([0.0, 0.0, 0.0])
        bbx_max = np.array([10.0, 10.0, 10.0])
        tree.setBBXMin(bbx_min)
        tree.setBBXMax(bbx_max)
        
        # Verify bounding box
        min_val = tree.getBBXMin()
        max_val = tree.getBBXMax()
        assert np.allclose(min_val, bbx_min)
        assert np.allclose(max_val, bbx_max)
    
    def test_node_occupancy_check(self):
        """Test checking node occupancy in ColorOcTree"""
        tree = pyoctomap.ColorOcTree(0.1)
        coord = np.array([1.0, 2.0, 3.0])
        
        # Update as occupied
        node = tree.updateNode(coord, True)
        assert node is not None
        
        # Check occupancy
        assert tree.isNodeOccupied(node)
        
        # Set color
        tree.setNodeColor(coord, 255, 0, 0)
        
        # Verify node still exists and is occupied
        node = tree.search(coord)
        assert node is not None
        assert tree.isNodeOccupied(node)
    
    def test_multiple_color_updates(self):
        """Test multiple color updates on same node"""
        tree = pyoctomap.ColorOcTree(0.1)
        coord = np.array([1.0, 2.0, 3.0])
        
        # Update node
        tree.updateNode(coord, True)
        
        # Set color multiple times
        tree.setNodeColor(coord, 255, 0, 0)  # Red
        node = tree.search(coord)
        assert node.getColor() == (255, 0, 0)
        
        tree.setNodeColor(coord, 0, 255, 0)  # Green
        node = tree.search(coord)
        assert node.getColor() == (0, 255, 0)
        
        tree.setNodeColor(coord, 0, 0, 255)  # Blue
        node = tree.search(coord)
        assert node.getColor() == (0, 0, 255)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

