"""
Pytest configuration and fixtures for octomap tests.
"""

import pytest
import numpy as np
from pyoctomap import octomap


@pytest.fixture
def octree():
    """Create a basic octree for testing."""
    tree = octomap.OcTree(0.1)
    
    # Add some test data
    test_points = [
        ([1.0, 2.0, 3.0], True),
        ([1.1, 2.1, 3.1], True),
        ([0.5, 0.5, 0.5], False),
        ([2.0, 2.0, 2.0], True),
        ([-1.0, -1.0, -1.0], False),
    ]
    
    for point, occupied in test_points:
        tree.updateNode(point, occupied)
    
    return tree


@pytest.fixture
def sample_points():
    """Generate sample 3D points for testing."""
    return np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [0.5, 0.5, 0.5],
        [2.0, 2.0, 2.0],
        [-1.0, -1.0, -1.0],
    ])
