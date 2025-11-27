#!/usr/bin/env python3
"""
Simple script to test if the build works correctly.
Used by GitHub Actions CI.
"""

def test_import():
    """Test basic import functionality."""
    import pyoctomap
    assert pyoctomap.__version__ is not None

def test_basic_functionality():
    """Test basic OctoMap functionality."""
    import pyoctomap
    
    # Create octree
    tree = pyoctomap.OcTree(0.1)
    assert tree.getResolution() == 0.1
    
    # Add some nodes
    tree.updateNode([1.0, 2.0, 3.0], True)
    tree.updateNode([1.1, 2.1, 3.1], True)
    tree.updateNode([0.5, 0.5, 0.5], False)
    
    # Test search
    node = tree.search([1.0, 2.0, 3.0])
    assert node is not None, "Should find node at [1.0, 2.0, 3.0]"
    assert tree.isNodeOccupied(node), "Node should be occupied"
    assert tree.size() > 0

def test_github2pypi():
    """Test github2pypi URL conversion."""
    import sys
    import os
    
    # Add github2pypi to path
    github2pypi_path = os.path.join(os.path.dirname(__file__), '../github2pypi')
    sys.path.insert(0, github2pypi_path)
    
    from replace_url import replace_url
    
    # Test with sample content
    test_content = """# Test
![Image](images/test.png)
[Link](docs/guide.md)
"""
    
    result = replace_url('Spinkoo/pyoctomap', test_content)
    
    # Check conversions
    assert 'https://github.com/Spinkoo/pyoctomap/blob/main/images/test.png?raw=true' in result, "Image URL not converted correctly"
    assert 'https://github.com/Spinkoo/pyoctomap/blob/main/docs/guide.md' in result, "Link URL not converted correctly"

# This file is now a proper pytest module
# Run with: python -m pytest unit_test/test_build.py -v
