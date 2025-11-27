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
