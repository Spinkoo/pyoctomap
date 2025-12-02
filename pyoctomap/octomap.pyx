# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Main wrapper module that imports all octree components.
This module maintains backward compatibility by re-exporting all classes.
"""

# Import all base classes
from .octree_base import (
    OcTreeKey,
    OcTreeNode,
    NullPointerException
)

# Import all iterator classes
from .octree_iterators import (
    SimpleTreeIterator,
    SimpleLeafIterator,
    SimpleLeafBBXIterator
)

# Import the main OcTree class and helper function
from .octree import (
    OcTree,
    _octree_read
)

# Import ColorOcTree classes (optional, may not be available)
try:
    from .color_octree import (
        ColorOcTree,
        ColorOcTreeNode
    )
    _has_color_octree = True
except ImportError:
    _has_color_octree = False

# Import CountingOcTree classes (optional, may not be available)
try:
    from .counting_octree import (
        CountingOcTree,
        CountingOcTreeNode
    )
    _has_counting_octree = True
except ImportError:
    _has_counting_octree = False

# Import OcTreeStamped classes (optional, may not be available)
try:
    from .stamped_octree import (
        OcTreeStamped,
        OcTreeNodeStamped
    )
    _has_stamped_octree = True
except ImportError:
    _has_stamped_octree = False

# Import Pointcloud class (optional, may not be available)
try:
    from .pointcloud import Pointcloud
    _has_pointcloud = True
except ImportError:
    _has_pointcloud = False

# Re-export everything for backward compatibility
__all__ = [
    "OcTreeKey",
    "OcTreeNode",
    "NullPointerException",
    "SimpleTreeIterator",
    "SimpleLeafIterator",
    "SimpleLeafBBXIterator",
    "OcTree",
    "_octree_read"
]

# Conditionally add ColorOcTree exports
if _has_color_octree:
    __all__.extend(["ColorOcTree", "ColorOcTreeNode"])

# Conditionally add CountingOcTree exports
if _has_counting_octree:
    __all__.extend(["CountingOcTree", "CountingOcTreeNode"])

# Conditionally add OcTreeStamped exports
if _has_stamped_octree:
    __all__.extend(["OcTreeStamped", "OcTreeNodeStamped"])

# Conditionally add Pointcloud exports
if _has_pointcloud:
    __all__.append("Pointcloud")
