"""
OctoMap Python Wrapper with Bundled Libraries

This package provides Python bindings for the OctoMap library with bundled
shared libraries to avoid dependency issues.
"""

import sys
from pathlib import Path

# Version information
__version__ = "1.1.2"
__author__ = "Spinkoo"
__email__ = "lespinkoo@gmail.com"

# Note: Library paths are handled via rpath in compiled extensions
# No runtime setup required

# Import the main module first
try:
    from .octomap import *
    __all__ = [
        "OcTree", "OcTreeNode", "OcTreeKey",
        "SimpleTreeIterator", "SimpleLeafIterator",
        "NullPointerException"
    ]
except ImportError as e:
    print(f"Error importing octomap module: {e}")
    print("This might be due to missing shared libraries or compilation issues.")
    raise

# Import Pointcloud (after octomap to avoid conflicts)
_has_pointcloud = False
try:
    from .pointcloud import Pointcloud
    _has_pointcloud = True
    __all__.append("Pointcloud")
except ImportError as e:
    _has_pointcloud = False
    Pointcloud = None
    # Print warning to help with debugging
    import os
    verbose = os.environ.get('PYOCTOMAP_VERBOSE', '').lower() in ('1', 'true', 'yes')
    if verbose:
        import traceback
        print(f"⚠️ Pointcloud module not available: {e}")
        traceback.print_exc()
        print("Pointcloud features will not be available.")
        print("To compile pointcloud module, run: python setup.py build_ext --inplace")
except Exception as e:
    # Catch any other errors (compilation errors, etc.)
    _has_pointcloud = False
    Pointcloud = None
    import os
    verbose = os.environ.get('PYOCTOMAP_VERBOSE', '').lower() in ('1', 'true', 'yes')
    if verbose:
        import traceback
        print(f"⚠️ Error loading Pointcloud module: {e}")
        traceback.print_exc()
        print("Pointcloud features will not be available.")
        print("To compile pointcloud module, run: python setup.py build_ext --inplace")

# Add ColorOcTree if available
try:
    from .color_octree import ColorOcTree, ColorOcTreeNode
    __all__.extend(["ColorOcTree", "ColorOcTreeNode"])
except ImportError:
    pass

# Add CountingOcTree if available
try:
    from .counting_octree import CountingOcTree, CountingOcTreeNode
    __all__.extend(["CountingOcTree", "CountingOcTreeNode"])
except ImportError:
    pass

# Add OcTreeStamped if available
try:
    from .stamped_octree import OcTreeStamped, OcTreeNodeStamped
    __all__.extend(["OcTreeStamped", "OcTreeNodeStamped"])
except ImportError:
    pass

# Memory management is handled in the Cython code

# Package information
def get_package_info():
    """Get information about the package and its libraries"""
    package_dir = Path(__file__).parent.absolute()
    lib_dir = package_dir / "lib"
    
    info = {
        "version": __version__,
        "package_dir": str(package_dir),
        "lib_dir": str(lib_dir),
        "lib_dir_exists": lib_dir.exists(),
    }
    
    if lib_dir.exists():
        lib_files = list(lib_dir.glob("*"))
        info["lib_files"] = [f.name for f in lib_files]
        info["lib_count"] = len(lib_files)
    
    return info

# Example usage and testing
def test_installation():
    """Test if the installation is working correctly"""
    print("Testing OctoMap Python installation...")
    
    try:
        # Test basic import
        from .octomap import OcTree
        print("✅ OcTree import successful")
        
        # Test creating an octree
        tree = OcTree(0.1)
        print("✅ OcTree creation successful")
        
        # Test basic operations
        tree.updateNode(1.0, 2.0, 3.0, True)
        print("✅ Basic operations successful")
        
        print("🎉 Installation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

if __name__ == "__main__":
    # Print package information
    info = get_package_info()
    print("OctoMap Python Package Information:")
    print(f"  Version: {info['version']}")
    print(f"  Package Directory: {info['package_dir']}")
    print(f"  Library Directory: {info['lib_dir']}")
    print(f"  Library Directory Exists: {info['lib_dir_exists']}")
    
    if 'lib_files' in info:
        print(f"  Library Files ({info['lib_count']}):")
        for lib_file in info['lib_files']:
            print(f"    - {lib_file}")
    
    # Run installation test
    test_installation()