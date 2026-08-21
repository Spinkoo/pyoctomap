"""
OctoMap Python Wrapper with Bundled Libraries

This package provides Python bindings for the OctoMap library with bundled
shared libraries to avoid dependency issues.
"""

import logging
import os
import sys
import traceback
from pathlib import Path

_logger = logging.getLogger(__name__)

# Bundled native deps: Linux/macOS use rpath on the extension; Windows needs an
# explicit DLL search path (Python 3.8+) before loading .pyd modules.
if sys.platform == "win32":
    _bundled_lib = Path(__file__).resolve().parent / "lib"
    if _bundled_lib.is_dir():
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(_bundled_lib))
        else:
            os.environ["PATH"] = str(_bundled_lib) + os.pathsep + os.environ.get("PATH", "")

# Version information
__version__ = "1.2.1"
__author__ = "Spinkoo"
__email__ = "lespinkoo@gmail.com"

from .octomap import *
__all__ = [
    "OcTree", "OcTreeNode", "OcTreeKey",
    "SimpleTreeIterator", "SimpleLeafIterator",
    "NullPointerException",
]

# Import Pointcloud (after octomap to avoid conflicts)
_has_pointcloud = False
try:
    from .pointcloud import Pointcloud
    _has_pointcloud = True
    __all__.append("Pointcloud")
except ImportError as e:
    _has_pointcloud = False
    Pointcloud = None
    if os.environ.get("PYOCTOMAP_VERBOSE", "").lower() in ("1", "true", "yes"):
        _logger.debug("Pointcloud not available: %s", e)
except Exception as e:
    _has_pointcloud = False
    Pointcloud = None
    if os.environ.get("PYOCTOMAP_VERBOSE", "").lower() in ("1", "true", "yes"):
        _logger.debug("Pointcloud load error: %s", e)
        traceback.print_exc()

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

def test_installation():
    """Return True if core OcTree operations work, else False."""
    try:
        from .octomap import OcTree

        tree = OcTree(0.1)
        tree.updateNode(1.0, 2.0, 3.0, True)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    info = get_package_info()
    for k in ("version", "package_dir", "lib_dir", "lib_dir_exists"):
        print(f"{k}: {info.get(k)}")
    if "lib_files" in info:
        print("lib_files:")
        for name in info["lib_files"]:
            print(f"  {name}")
    print("test_installation:", test_installation())