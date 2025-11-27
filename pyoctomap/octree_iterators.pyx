# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from cython.operator cimport dereference as deref, preincrement as inc
from libc.stddef cimport size_t
cimport octomap_defs as defs
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t

# Import base classes - use runtime import to avoid circular dependency
from .octree_base import NullPointerException
# Note: We can't cimport octree here because it creates a circular dependency
# Instead, we'll cast tree objects directly using isinstance checks
# The type casting will be done at runtime

# Fix NumPy API compatibility
np.import_array()

# NumPy compatibility for newer versions
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator* tree_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator* leaf_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator* leaf_bbx_iterator_ptr

# Simplified iterator classes that work around Cython template limitations

cdef class SimpleTreeIterator:
    """
    Robust wrapper around octomap C++ tree_iterator.
    Captures per-step state so methods refer to the item yielded by the last next().
    """
    cdef object _tree  # Python OcTree
    cdef tree_iterator_ptr _it
    cdef tree_iterator_ptr _end
    cdef bint _is_end
    # Snapshot state
    cdef object _current_node
    cdef list _current_coord
    cdef double _current_size
    cdef int _current_depth

    def __dealloc__(self):
        if self._it != NULL:
            del self._it
            self._it = NULL
        if self._end != NULL:
            del self._end
            self._end = NULL
        self._tree = None
        self._current_node = None

    def __cinit__(self):
        self._tree = None
        self._it = NULL
        self._end = NULL
        self._is_end = True
        self._current_node = None
        self._current_coord = None
        self._current_size = 0.0
        self._current_depth = 0

    def __init__(self, tree, maxDepth=0):
        # Runtime import to avoid circular dependency
        from .octree import OcTree
        if tree is None or not isinstance(tree, OcTree):
            self._is_end = True
            return
        # Access thisptr through the helper method (returns address as size_t)
        cdef size_t ptr_addr = tree._get_ptr_addr()
        cdef defs.OcTree* tree_ptr = <defs.OcTree*>ptr_addr
        if tree_ptr == NULL:
            self._is_end = True
            return
        self._tree = tree
        cdef unsigned char depth = <unsigned char?>maxDepth
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator tmp_it = tree_ptr.begin_tree(depth)
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator tmp_end = tree_ptr.end_tree()
        self._it = new defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator(tmp_it)
        self._end = new defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator(tmp_end)
        self._is_end = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._is_end or self._it == NULL or self._end == NULL:
            raise StopIteration
        if deref(self._it) == deref(self._end):
            self._is_end = True
            raise StopIteration

        # Snapshot current iterator state
        cdef defs.point3d p = deref(self._it).getCoordinate()
        self._current_coord = [p.x(), p.y(), p.z()]
        self._current_size = deref(self._it).getSize()
        self._current_depth = <int?>deref(self._it).getDepth()

        # Capture node by searching at current coordinate (robust approach)
        cdef np.ndarray[DOUBLE_t, ndim=1] _pt = np.array(self._current_coord, dtype=np.float64)
        self._current_node = self._tree.search(_pt)

        # Advance iterator
        inc(deref(self._it))
        return self

    def getCoordinate(self):
        if self._current_coord is not None:
            return self._current_coord
        return [0.0, 0.0, 0.0]

    def getSize(self):
        return self._current_size

    def getDepth(self):
        return self._current_depth

    def isLeaf(self):
        if self._current_node is None:
            return True
        return not self._tree.nodeHasChildren(self._current_node)

    
cdef class SimpleLeafIterator:
    """
    Robust wrapper around octomap C++ leaf_iterator.
    Captures per-step state so methods refer to the item yielded by the last next().
    """
    cdef object _tree
    cdef leaf_iterator_ptr _it
    cdef leaf_iterator_ptr _end
    cdef bint _is_end
    # Snapshot state
    cdef object _current_node
    cdef list _current_coord
    cdef double _current_size
    cdef int _current_depth

    def __dealloc__(self):
        if self._it != NULL:
            del self._it
            self._it = NULL
        if self._end != NULL:
            del self._end
            self._end = NULL
        self._tree = None
        self._current_node = None

    def __cinit__(self):
        self._tree = None
        self._it = NULL
        self._end = NULL
        self._is_end = True
        self._current_node = None
        self._current_coord = None
        self._current_size = 0.0
        self._current_depth = 0

    def __init__(self, tree, maxDepth=0):
        # Runtime import to avoid circular dependency
        from .octree import OcTree
        if tree is None or not isinstance(tree, OcTree):
            self._is_end = True
            return
        # Access thisptr through the helper method (returns address as size_t)
        cdef size_t ptr_addr = tree._get_ptr_addr()
        cdef defs.OcTree* tree_ptr = <defs.OcTree*>ptr_addr
        if tree_ptr == NULL:
            self._is_end = True
            return
        self._tree = tree
        cdef unsigned char depth = <unsigned char?>maxDepth
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator tmp_it = tree_ptr.begin_leafs(depth)
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator tmp_end = tree_ptr.end_leafs()
        self._it = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator(tmp_it)
        self._end = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator(tmp_end)
        self._is_end = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._is_end or self._it == NULL or self._end == NULL:
            raise StopIteration
        if deref(self._it) == deref(self._end):
            self._is_end = True
            raise StopIteration

        # Snapshot the current state
        cdef defs.point3d p = deref(self._it).getCoordinate()
        self._current_coord = [p.x(), p.y(), p.z()]
        self._current_size = deref(self._it).getSize()
        self._current_depth = <int?>deref(self._it).getDepth()

        # Capture node by searching at current coordinate (robust approach)
        cdef np.ndarray[DOUBLE_t, ndim=1] _pt = np.array(self._current_coord, dtype=np.float64)
        self._current_node = self._tree.search(_pt)

        # Advance
        inc(deref(self._it))
        return self

    def getCoordinate(self):
        if self._current_coord is not None:
            return self._current_coord
        return [0.0, 0.0, 0.0]

    def getSize(self):
        return self._current_size

    def getDepth(self):
        return self._current_depth

    def isLeaf(self):
        """Check if current node is a leaf"""
        if self._current_node is None:
            return True
        return not self._tree.nodeHasChildren(self._current_node)

    @property
    def current_node(self):
        return self._current_node

cdef class SimpleLeafBBXIterator:
    """
    Robust wrapper around octomap C++ leaf_bbx_iterator.
    Captures per-step state so methods refer to the item yielded by the last next().
    """
    cdef object _tree
    cdef leaf_bbx_iterator_ptr _it
    cdef leaf_bbx_iterator_ptr _end
    cdef bint _is_end
    # Snapshot state
    cdef object _current_node
    cdef list _current_coord
    cdef double _current_size
    cdef int _current_depth

    def __dealloc__(self):
        if self._it != NULL:
            del self._it
            self._it = NULL
        if self._end != NULL:
            del self._end
            self._end = NULL
        self._tree = None
        self._current_node = None

    def __cinit__(self):
        self._tree = None
        self._it = NULL
        self._end = NULL
        self._is_end = True
        self._current_node = None
        self._current_coord = None
        self._current_size = 0.0
        self._current_depth = 0

    def __init__(self, tree, np.ndarray[DOUBLE_t, ndim=1] bbx_min, np.ndarray[DOUBLE_t, ndim=1] bbx_max, maxDepth=0):
        # Runtime import to avoid circular dependency
        from .octree import OcTree
        if tree is None or not isinstance(tree, OcTree):
            self._is_end = True
            return
        # Access thisptr through the helper method (returns address as size_t)
        cdef size_t ptr_addr = tree._get_ptr_addr()
        cdef defs.OcTree* tree_ptr = <defs.OcTree*>ptr_addr
        if tree_ptr == NULL:
            self._is_end = True
            return
        self._tree = tree
        cdef defs.point3d pmin = defs.point3d(<float?>bbx_min[0], <float?>bbx_min[1], <float?>bbx_min[2])
        cdef defs.point3d pmax = defs.point3d(<float?>bbx_max[0], <float?>bbx_max[1], <float?>bbx_max[2])
        cdef unsigned char depth = <unsigned char?>maxDepth
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator tmp_it = tree_ptr.begin_leafs_bbx(pmin, pmax, depth)
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator tmp_end = tree_ptr.end_leafs_bbx()
        self._it = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator(tmp_it)
        self._end = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator(tmp_end)
        self._is_end = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._is_end or self._it == NULL or self._end == NULL:
            raise StopIteration
        if deref(self._it) == deref(self._end):
            self._is_end = True
            raise StopIteration

        # Snapshot
        cdef defs.point3d p = deref(self._it).getCoordinate()
        self._current_coord = [p.x(), p.y(), p.z()]
        self._current_size = deref(self._it).getSize()
        self._current_depth = <int?>deref(self._it).getDepth()

        # Capture node by searching at current coordinate (robust approach)
        cdef np.ndarray[DOUBLE_t, ndim=1] _pt = np.array(self._current_coord, dtype=np.float64)
        self._current_node = self._tree.search(_pt)

        inc(deref(self._it))
        return self

    def getCoordinate(self):
        if self._current_coord is not None:
            return self._current_coord
        return [0.0, 0.0, 0.0]

    def getSize(self):
        return self._current_size

    def getDepth(self):
        return self._current_depth

    def isLeaf(self):
        """Check if current node is a leaf"""
        if self._current_node is None:
            return True
        return not self._tree.nodeHasChildren(self._current_node)

    @property
    def current_node(self):
        return self._current_node

    @property
    def is_end(self):
        return self._is_end

