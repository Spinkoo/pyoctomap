"""
OctoMap Python Wrapper with Bundled Libraries

This package provides Python bindings for the OctoMap library with bundled
shared libraries to avoid dependency issues.
"""

import os
import sys
import platform
from pathlib import Path

# Version information
__version__ = "0.3.1"
__author__ = "Spinkoo"
__email__ = "lespinkoo@gmail.com"

def setup_library_paths():
    """No-op: runtime linking handled via rpath in extension"""
    return True

def check_libraries():
    """Check if required libraries are available"""
    package_dir = Path(__file__).parent.absolute()
    lib_dir = package_dir / "lib"
    
    # Try alternative paths for installed packages
    if not lib_dir.exists():
        alt_paths = [
            package_dir.parent / "octomap" / "lib",
            Path(sys.prefix) / "lib" / "python3.12" / "site-packages" / "octomap" / "lib",
            Path(sys.prefix) / "lib" / "site-packages" / "octomap" / "lib"
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                lib_dir = alt_path
                break
        else:
            return False, f"Library directory not found: {lib_dir}"
    
    # Check for required libraries
    required_libs = ["liboctomap", "libdynamicedt3d", "liboctomath"]
    missing_libs = []
    
    for lib_name in required_libs:
        lib_files = list(lib_dir.glob(f"{lib_name}*"))
        if not lib_files:
            missing_libs.append(lib_name)
    
    if missing_libs:
        return False, f"Missing required libraries: {missing_libs}"
    
    return True, "All required libraries found"

# Set up library paths automatically BEFORE importing
setup_success = setup_library_paths()

# Optional diagnostic can be added here if needed
lib_check_success, lib_message = True, "rpath used"

# Import the main module
try:
    from .octomap import *
    __all__ = [
        "OcTree", "OcTreeNode", "OcTreeKey", "Pointcloud",
        "SimpleTreeIterator", "SimpleLeafIterator",
        "NullPointerException"
    ]
except ImportError as e:
    print(f"Error importing octomap module: {e}")
    print("This might be due to missing shared libraries or compilation issues.")
    raise

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
        "setup_success": setup_success,
        "lib_check_success": lib_check_success,
        "lib_message": lib_message,
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
    print(f"  Setup Success: {info['setup_success']}")
    print(f"  Library Check Success: {info['lib_check_success']}")
    print(f"  Library Message: {info['lib_message']}")
    
    if 'lib_files' in info:
        print(f"  Library Files ({info['lib_count']}):")
        for lib_file in info['lib_files']:
            print(f"    - {lib_file}")
    
    # Run installation test
    test_installation()