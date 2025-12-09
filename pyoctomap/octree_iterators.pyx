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
# ColorOcTree iterator types
ctypedef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_iterator* color_leaf_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_bbx_iterator* color_leaf_bbx_iterator_ptr
# OcTreeStamped iterator types
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_iterator* stamped_leaf_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_bbx_iterator* stamped_leaf_bbx_iterator_ptr

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
    Works with OcTree, ColorOcTree, and OcTreeStamped.
    Captures per-step state so methods refer to the item yielded by the last next().
    """
    cdef object _tree
    cdef leaf_iterator_ptr _it
    cdef leaf_iterator_ptr _end
    cdef color_leaf_iterator_ptr _color_it
    cdef color_leaf_iterator_ptr _color_end
    cdef stamped_leaf_iterator_ptr _stamped_it
    cdef stamped_leaf_iterator_ptr _stamped_end
    cdef int _tree_type  # 0=OcTree, 1=ColorOcTree, 2=OcTreeStamped
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
        if self._color_it != NULL:
            del self._color_it
            self._color_it = NULL
        if self._color_end != NULL:
            del self._color_end
            self._color_end = NULL
        if self._stamped_it != NULL:
            del self._stamped_it
            self._stamped_it = NULL
        if self._stamped_end != NULL:
            del self._stamped_end
            self._stamped_end = NULL
        self._tree = None
        self._current_node = None

    def __cinit__(self):
        self._tree = None
        self._it = NULL
        self._end = NULL
        self._color_it = NULL
        self._color_end = NULL
        self._stamped_it = NULL
        self._stamped_end = NULL
        self._tree_type = 0
        self._is_end = True
        self._current_node = None
        self._current_coord = None
        self._current_size = 0.0
        self._current_depth = 0

    def __init__(self, tree, maxDepth=0):
        # Runtime import to avoid circular dependency
        from .octree import OcTree
        from .color_octree import ColorOcTree
        from .stamped_octree import OcTreeStamped
        
        # Declare all C variables at the top (required by Cython)
        cdef size_t ptr_addr
        cdef unsigned char depth = <unsigned char?>maxDepth
        cdef defs.OcTree* tree_ptr_oc = NULL
        cdef defs.ColorOcTree* tree_ptr_color = NULL
        cdef defs.OcTreeStamped* tree_ptr_stamped = NULL
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator tmp_it_oc
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator tmp_end_oc
        cdef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_iterator tmp_it_color
        cdef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_iterator tmp_end_color
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_iterator tmp_it_stamped
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_iterator tmp_end_stamped
        
        if tree is None:
            self._is_end = True
            return
        
        if isinstance(tree, OcTree):
            # OcTree case
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_oc = <defs.OcTree*>ptr_addr
            if tree_ptr_oc == NULL:
                self._is_end = True
                return
            self._tree = tree
            self._tree_type = 0
            tmp_it_oc = tree_ptr_oc.begin_leafs(depth)
            tmp_end_oc = tree_ptr_oc.end_leafs()
            self._it = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator(tmp_it_oc)
            self._end = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator(tmp_end_oc)
            self._is_end = False
        elif isinstance(tree, ColorOcTree):
            # ColorOcTree case
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_color = <defs.ColorOcTree*>ptr_addr
            if tree_ptr_color == NULL:
                self._is_end = True
                return
            self._tree = tree
            self._tree_type = 1
            tmp_it_color = tree_ptr_color.begin_leafs(depth)
            tmp_end_color = tree_ptr_color.end_leafs()
            self._color_it = new defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_iterator(tmp_it_color)
            self._color_end = new defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_iterator(tmp_end_color)
            self._is_end = False
        elif isinstance(tree, OcTreeStamped):
            # OcTreeStamped case
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_stamped = <defs.OcTreeStamped*>ptr_addr
            if tree_ptr_stamped == NULL:
                self._is_end = True
                return
            self._tree = tree
            self._tree_type = 2
            tmp_it_stamped = tree_ptr_stamped.begin_leafs(depth)
            tmp_end_stamped = tree_ptr_stamped.end_leafs()
            self._stamped_it = new defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_iterator(tmp_it_stamped)
            self._stamped_end = new defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_iterator(tmp_end_stamped)
            self._is_end = False
        else:
            self._is_end = True

    def __iter__(self):
        return self

    def __next__(self):
        # Declare all C variables at the top (required by Cython)
        cdef defs.point3d p
        cdef np.ndarray[DOUBLE_t, ndim=1] _pt
        
        if self._is_end:
            raise StopIteration
        
        if self._tree_type == 1:  # ColorOcTree
            if self._color_it == NULL or self._color_end == NULL:
                raise StopIteration
            if deref(self._color_it) == deref(self._color_end):
                self._is_end = True
                raise StopIteration
            
            # Snapshot the current state
            p = deref(self._color_it).getCoordinate()
            self._current_coord = [p.x(), p.y(), p.z()]
            self._current_size = deref(self._color_it).getSize()
            self._current_depth = <int?>deref(self._color_it).getDepth()
            
            # Advance
            inc(deref(self._color_it))
        elif self._tree_type == 2:  # OcTreeStamped
            if self._stamped_it == NULL or self._stamped_end == NULL:
                raise StopIteration
            if deref(self._stamped_it) == deref(self._stamped_end):
                self._is_end = True
                raise StopIteration
            
            # Snapshot the current state
            p = deref(self._stamped_it).getCoordinate()
            self._current_coord = [p.x(), p.y(), p.z()]
            self._current_size = deref(self._stamped_it).getSize()
            self._current_depth = <int?>deref(self._stamped_it).getDepth()
            
            # Advance
            inc(deref(self._stamped_it))
        else:  # OcTree (default)
            if self._it == NULL or self._end == NULL:
                raise StopIteration
            if deref(self._it) == deref(self._end):
                self._is_end = True
                raise StopIteration
            
            # Snapshot the current state
            p = deref(self._it).getCoordinate()
            self._current_coord = [p.x(), p.y(), p.z()]
            self._current_size = deref(self._it).getSize()
            self._current_depth = <int?>deref(self._it).getDepth()
            
            # Advance
            inc(deref(self._it))
        
        # Capture node by searching at current coordinate (robust approach)
        _pt = np.array(self._current_coord, dtype=np.float64)
        self._current_node = self._tree.search(_pt)
        
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
    Works with OcTree, ColorOcTree, and OcTreeStamped.
    Captures per-step state so methods refer to the item yielded by the last next().
    """
    cdef object _tree
    cdef leaf_bbx_iterator_ptr _it
    cdef leaf_bbx_iterator_ptr _end
    cdef color_leaf_bbx_iterator_ptr _color_it
    cdef color_leaf_bbx_iterator_ptr _color_end
    cdef stamped_leaf_bbx_iterator_ptr _stamped_it
    cdef stamped_leaf_bbx_iterator_ptr _stamped_end
    cdef int _tree_type  # 0=OcTree, 1=ColorOcTree, 2=OcTreeStamped
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
        if self._color_it != NULL:
            del self._color_it
            self._color_it = NULL
        if self._color_end != NULL:
            del self._color_end
            self._color_end = NULL
        if self._stamped_it != NULL:
            del self._stamped_it
            self._stamped_it = NULL
        if self._stamped_end != NULL:
            del self._stamped_end
            self._stamped_end = NULL
        self._tree = None
        self._current_node = None

    def __cinit__(self):
        self._tree = None
        self._it = NULL
        self._end = NULL
        self._color_it = NULL
        self._color_end = NULL
        self._stamped_it = NULL
        self._stamped_end = NULL
        self._tree_type = 0
        self._is_end = True
        self._current_node = None
        self._current_coord = None
        self._current_size = 0.0
        self._current_depth = 0

    def __init__(self, tree, np.ndarray[DOUBLE_t, ndim=1] bbx_min, np.ndarray[DOUBLE_t, ndim=1] bbx_max, maxDepth=0):
        # Runtime import to avoid circular dependency
        from .octree import OcTree
        from .color_octree import ColorOcTree
        from .stamped_octree import OcTreeStamped
        
        # Declare all C variables at the top (required by Cython)
        cdef size_t ptr_addr
        cdef defs.point3d pmin = defs.point3d(<float?>bbx_min[0], <float?>bbx_min[1], <float?>bbx_min[2])
        cdef defs.point3d pmax = defs.point3d(<float?>bbx_max[0], <float?>bbx_max[1], <float?>bbx_max[2])
        cdef unsigned char depth = <unsigned char?>maxDepth
        cdef defs.OcTree* tree_ptr_oc = NULL
        cdef defs.ColorOcTree* tree_ptr_color = NULL
        cdef defs.OcTreeStamped* tree_ptr_stamped = NULL
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator tmp_it_oc
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator tmp_end_oc
        cdef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_bbx_iterator tmp_it_color
        cdef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_bbx_iterator tmp_end_color
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_bbx_iterator tmp_it_stamped
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_bbx_iterator tmp_end_stamped
        
        if tree is None:
            self._is_end = True
            return
        
        if isinstance(tree, OcTree):
            # OcTree case
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_oc = <defs.OcTree*>ptr_addr
            if tree_ptr_oc == NULL:
                self._is_end = True
                return
            self._tree = tree
            self._tree_type = 0
            tmp_it_oc = tree_ptr_oc.begin_leafs_bbx(pmin, pmax, depth)
            tmp_end_oc = tree_ptr_oc.end_leafs_bbx()
            self._it = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator(tmp_it_oc)
            self._end = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator(tmp_end_oc)
            self._is_end = False
        elif isinstance(tree, ColorOcTree):
            # ColorOcTree case
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_color = <defs.ColorOcTree*>ptr_addr
            if tree_ptr_color == NULL:
                self._is_end = True
                return
            self._tree = tree
            self._tree_type = 1
            tmp_it_color = tree_ptr_color.begin_leafs_bbx(pmin, pmax, depth)
            tmp_end_color = tree_ptr_color.end_leafs_bbx()
            self._color_it = new defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_bbx_iterator(tmp_it_color)
            self._color_end = new defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_bbx_iterator(tmp_end_color)
            self._is_end = False
        elif isinstance(tree, OcTreeStamped):
            # OcTreeStamped case
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_stamped = <defs.OcTreeStamped*>ptr_addr
            if tree_ptr_stamped == NULL:
                self._is_end = True
                return
            self._tree = tree
            self._tree_type = 2
            tmp_it_stamped = tree_ptr_stamped.begin_leafs_bbx(pmin, pmax, depth)
            tmp_end_stamped = tree_ptr_stamped.end_leafs_bbx()
            self._stamped_it = new defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_bbx_iterator(tmp_it_stamped)
            self._stamped_end = new defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_bbx_iterator(tmp_end_stamped)
            self._is_end = False
        else:
            self._is_end = True

    def __iter__(self):
        return self

    def __next__(self):
        # Declare all C variables at the top (required by Cython)
        cdef defs.point3d p
        cdef np.ndarray[DOUBLE_t, ndim=1] _pt
        
        if self._is_end:
            raise StopIteration
        
        if self._tree_type == 1:  # ColorOcTree
            if self._color_it == NULL or self._color_end == NULL:
                raise StopIteration
            if deref(self._color_it) == deref(self._color_end):
                self._is_end = True
                raise StopIteration
            
            # Snapshot
            p = deref(self._color_it).getCoordinate()
            self._current_coord = [p.x(), p.y(), p.z()]
            self._current_size = deref(self._color_it).getSize()
            self._current_depth = <int?>deref(self._color_it).getDepth()
            
            # Advance
            inc(deref(self._color_it))
        elif self._tree_type == 2:  # OcTreeStamped
            if self._stamped_it == NULL or self._stamped_end == NULL:
                raise StopIteration
            if deref(self._stamped_it) == deref(self._stamped_end):
                self._is_end = True
                raise StopIteration
            
            # Snapshot
            p = deref(self._stamped_it).getCoordinate()
            self._current_coord = [p.x(), p.y(), p.z()]
            self._current_size = deref(self._stamped_it).getSize()
            self._current_depth = <int?>deref(self._stamped_it).getDepth()
            
            # Advance
            inc(deref(self._stamped_it))
        else:  # OcTree (default)
            if self._it == NULL or self._end == NULL:
                raise StopIteration
            if deref(self._it) == deref(self._end):
                self._is_end = True
                raise StopIteration
            
            # Snapshot
            p = deref(self._it).getCoordinate()
            self._current_coord = [p.x(), p.y(), p.z()]
            self._current_size = deref(self._it).getSize()
            self._current_depth = <int?>deref(self._it).getDepth()
            
            # Advance
            inc(deref(self._it))
        
        # Capture node by searching at current coordinate (robust approach)
        _pt = np.array(self._current_coord, dtype=np.float64)
        self._current_node = self._tree.search(_pt)
        
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


