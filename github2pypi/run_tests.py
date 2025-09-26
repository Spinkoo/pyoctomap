#!/usr/bin/env python3
"""
Test script for github2pypi URL replacement functionality.
Tests the replace_url function with various PyOctoMap scenarios.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replace_url import replace_url


def test_image_replacement():
    """Test image URL replacement"""
    content = "![Example](images/octomap_core.png)"
    result = replace_url("Spinkoo/pyoctomap", content)
    expected = "![Example](https://github.com/Spinkoo/pyoctomap/blob/main/images/octomap_core.png?raw=true)"
    
    print("Test: Image URL replacement")
    print(f"Input:    {content}")
    print(f"Output:   {result}")
    print(f"Expected: {expected}")
    print(f"Status:   {'✅ PASS' if result == expected else '❌ FAIL'}")
    print()
    
    return result == expected


def test_link_replacement():
    """Test link URL replacement"""
    content = "[Documentation](Readme.md)"
    result = replace_url("Spinkoo/pyoctomap", content)
    expected = "[Documentation](https://github.com/Spinkoo/pyoctomap/blob/main/Readme.md)"
    
    print("Test: Link URL replacement")
    print(f"Input:    {content}")
    print(f"Output:   {result}")
    print(f"Expected: {expected}")
    print(f"Status:   {'✅ PASS' if result == expected else '❌ FAIL'}")
    print()
    
    return result == expected


def test_absolute_url_unchanged():
    """Test that absolute URLs are not changed"""
    content = "![Example](https://github.com/Spinkoo/pyoctomap/blob/main/images/octomap_core.png?raw=true)"
    result = replace_url("Spinkoo/pyoctomap", content)
    expected = content  # Should remain unchanged
    
    print("Test: Absolute URL unchanged")
    print(f"Input:    {content}")
    print(f"Output:   {result}")
    print(f"Expected: {expected}")
    print(f"Status:   {'✅ PASS' if result == expected else '❌ FAIL'}")
    print()
    
    return result == expected


def test_multiple_replacements():
    """Test multiple URL replacements in one content"""
    content = """
# PyOctoMap

![Example](images/octomap_core.png)

See the [documentation](Readme.md) for more details.

![Another](images/occupancy_grid.png)
"""
    
    result = replace_url("Spinkoo/pyoctomap", content)
    
    # Check that all relative URLs were replaced
    success = (
        "https://github.com/Spinkoo/pyoctomap/blob/main/images/octomap_core.png?raw=true" in result and
        "https://github.com/Spinkoo/pyoctomap/blob/main/Readme.md" in result and
        "https://github.com/Spinkoo/pyoctomap/blob/main/images/occupancy_grid.png?raw=true" in result
    )
    
    print("Test: Multiple URL replacements")
    print(f"Input contains: images/octomap_core.png, Readme.md, images/occupancy_grid.png")
    print(f"All converted: {'✅ PASS' if success else '❌ FAIL'}")
    print()
    
    return success


def test_custom_branch():
    """Test custom branch specification"""
    content = "![Example](images/octomap_core.png)"
    result = replace_url("Spinkoo/pyoctomap", content, branch="develop")
    expected = "![Example](https://github.com/Spinkoo/pyoctomap/blob/develop/images/octomap_core.png?raw=true)"
    
    print("Test: Custom branch")
    print(f"Input:    {content}")
    print(f"Output:   {result}")
    print(f"Expected: {expected}")
    print(f"Status:   {'✅ PASS' if result == expected else '❌ FAIL'}")
    print()
    
    return result == expected


def main():
    """Run all tests"""
    print("Testing GitHub2PyPI URL replacement for PyOctoMap")
    print("=" * 60)
    
    tests = [
        test_image_replacement,
        test_link_replacement,
        test_absolute_url_unchanged,
        test_multiple_replacements,
        test_custom_branch,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("=" * 60)
    print(f"Tests run: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(tests) - sum(results)}")
    
    if all(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
