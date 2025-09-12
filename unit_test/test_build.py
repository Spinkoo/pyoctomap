#!/usr/bin/env python3
"""
Simple script to test if the build works correctly.
Used by GitHub Actions CI.
"""

def test_import():
    """Test basic import functionality."""
    import octomap
    print(f"✅ Successfully imported octomap version: {octomap.__version__}")
    # Just assert success - if import fails, pytest will catch the exception

def test_basic_functionality():
    """Test basic OctoMap functionality."""
    import octomap
    
    # Create octree
    tree = octomap.OcTree(0.1)
    print(f"✅ Created OcTree with resolution: {tree.getResolution()}")
    
    # Add some nodes
    tree.updateNode([1.0, 2.0, 3.0], True)
    tree.updateNode([1.1, 2.1, 3.1], True)
    tree.updateNode([0.5, 0.5, 0.5], False)
    
    # Test search
    node = tree.search([1.0, 2.0, 3.0])
    assert node is not None, "Should find node at [1.0, 2.0, 3.0]"
    assert tree.isNodeOccupied(node), "Node should be occupied"
    
    print("✅ Basic OctoMap functionality working!")
    print(f"✅ Tree has {tree.size()} nodes")

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
    
    result = replace_url('Spinkoo/octomap2python', test_content)
    
    # Check conversions
    assert 'https://github.com/Spinkoo/octomap2python/blob/main/images/test.png?raw=true' in result, "Image URL not converted correctly"
    assert 'https://github.com/Spinkoo/octomap2python/blob/main/docs/guide.md' in result, "Link URL not converted correctly"
    
    print("✅ github2pypi URL conversion working correctly")

def main():
    """Run all tests."""
    print("🧪 Running build tests...")
    
    tests = [
        ("Import test", test_import),
        ("Functionality test", test_basic_functionality),
        ("github2pypi test", test_github2pypi),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🔍 {name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {name} FAILED")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
