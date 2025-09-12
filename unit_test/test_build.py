#!/usr/bin/env python3
"""
Simple script to test if the build works correctly.
Used by GitHub Actions CI.
"""

def test_import():
    """Test basic import functionality."""
    try:
        import octomap
        print(f"✅ Successfully imported octomap version: {octomap.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import octomap: {e}")
        return False

def test_basic_functionality():
    """Test basic OctoMap functionality."""
    try:
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
        if node and tree.isNodeOccupied(node):
            print("✅ Basic OctoMap functionality working!")
        else:
            print("❌ Basic functionality test failed!")
            return False
        
        print(f"✅ Tree has {tree.size()} nodes")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_github2pypi():
    """Test github2pypi URL conversion."""
    try:
        # Import the github2pypi module
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
        if 'https://github.com/Spinkoo/octomap2python/blob/main/images/test.png?raw=true' in result:
            print("✅ github2pypi URL conversion working correctly")
            return True
        else:
            print("❌ github2pypi URL conversion failed")
            return False
            
    except Exception as e:
        print(f"❌ github2pypi test failed: {e}")
        return False

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
