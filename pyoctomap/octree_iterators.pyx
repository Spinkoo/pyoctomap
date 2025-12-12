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

# Fix NumPy API compatibility
np.import_array()

# Iterator type definitions
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator* tree_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator* leaf_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator* leaf_bbx_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].tree_iterator* color_tree_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_iterator* color_leaf_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_bbx_iterator* color_leaf_bbx_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].tree_iterator* stamped_tree_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_iterator* stamped_leaf_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_bbx_iterator* stamped_leaf_bbx_iterator_ptr

# ============================================================================
# Iterator State Management - Unified abstraction for iterator operations
# ============================================================================

cdef struct IteratorState:
    void* it_ptr
    void* end_ptr
    int tree_type
    bint is_end

cdef inline bint _iterator_at_end(IteratorState* state):
    """Check if iterator has reached the end"""
    if state.is_end:
        return True
    
    if state.tree_type == 0:  # OcTree
        if state.it_ptr == NULL or state.end_ptr == NULL:
            return True
        return deref(<tree_iterator_ptr>state.it_ptr) == deref(<tree_iterator_ptr>state.end_ptr)
    elif state.tree_type == 1:  # ColorOcTree
        if state.it_ptr == NULL or state.end_ptr == NULL:
            return True
        return deref(<color_tree_iterator_ptr>state.it_ptr) == deref(<color_tree_iterator_ptr>state.end_ptr)
    elif state.tree_type == 2:  # OcTreeStamped
        if state.it_ptr == NULL or state.end_ptr == NULL:
            return True
        return deref(<stamped_tree_iterator_ptr>state.it_ptr) == deref(<stamped_tree_iterator_ptr>state.end_ptr)
    return True

cdef inline void _iterator_get_coordinate(IteratorState* state, defs.point3d* out):
    """Extract coordinate from current iterator position"""
    if state.tree_type == 0:  # OcTree
        out[0] = deref(<tree_iterator_ptr>state.it_ptr).getCoordinate()
    elif state.tree_type == 1:  # ColorOcTree
        out[0] = deref(<color_tree_iterator_ptr>state.it_ptr).getCoordinate()
    elif state.tree_type == 2:  # OcTreeStamped
        out[0] = deref(<stamped_tree_iterator_ptr>state.it_ptr).getCoordinate()

cdef inline double _iterator_get_size(IteratorState* state):
    """Extract size from current iterator position"""
    if state.tree_type == 0:  # OcTree
        return deref(<tree_iterator_ptr>state.it_ptr).getSize()
    elif state.tree_type == 1:  # ColorOcTree
        return deref(<color_tree_iterator_ptr>state.it_ptr).getSize()
    elif state.tree_type == 2:  # OcTreeStamped
        return deref(<stamped_tree_iterator_ptr>state.it_ptr).getSize()
    return 0.0

cdef inline int _iterator_get_depth(IteratorState* state):
    """Extract depth from current iterator position"""
    if state.tree_type == 0:  # OcTree
        return <int?>deref(<tree_iterator_ptr>state.it_ptr).getDepth()
    elif state.tree_type == 1:  # ColorOcTree
        return <int?>deref(<color_tree_iterator_ptr>state.it_ptr).getDepth()
    elif state.tree_type == 2:  # OcTreeStamped
        return <int?>deref(<stamped_tree_iterator_ptr>state.it_ptr).getDepth()
    return 0

cdef inline void _iterator_advance(IteratorState* state):
    """Advance iterator to next position"""
    if state.tree_type == 0:  # OcTree
        inc(deref(<tree_iterator_ptr>state.it_ptr))
    elif state.tree_type == 1:  # ColorOcTree
        inc(deref(<color_tree_iterator_ptr>state.it_ptr))
    elif state.tree_type == 2:  # OcTreeStamped
        inc(deref(<stamped_tree_iterator_ptr>state.it_ptr))

# Note: Cleanup must be done in __dealloc__ with typed pointers, not through void* casts

# ============================================================================
# Leaf Iterator State Management
# ============================================================================

cdef struct LeafIteratorState:
    void* it_ptr
    void* end_ptr
    int tree_type
    bint is_end

cdef inline bint _leaf_iterator_at_end(LeafIteratorState* state):
    """Check if leaf iterator has reached the end"""
    if state.is_end:
        return True
    
    if state.tree_type == 0:  # OcTree
        if state.it_ptr == NULL or state.end_ptr == NULL:
            return True
        return deref(<leaf_iterator_ptr>state.it_ptr) == deref(<leaf_iterator_ptr>state.end_ptr)
    elif state.tree_type == 1:  # ColorOcTree
        if state.it_ptr == NULL or state.end_ptr == NULL:
            return True
        return deref(<color_leaf_iterator_ptr>state.it_ptr) == deref(<color_leaf_iterator_ptr>state.end_ptr)
    elif state.tree_type == 2:  # OcTreeStamped
        if state.it_ptr == NULL or state.end_ptr == NULL:
            return True
        return deref(<stamped_leaf_iterator_ptr>state.it_ptr) == deref(<stamped_leaf_iterator_ptr>state.end_ptr)
    return True

cdef inline void _leaf_iterator_get_coordinate(LeafIteratorState* state, defs.point3d* out):
    """Extract coordinate from current leaf iterator position"""
    if state.tree_type == 0:  # OcTree
        out[0] = deref(<leaf_iterator_ptr>state.it_ptr).getCoordinate()
    elif state.tree_type == 1:  # ColorOcTree
        out[0] = deref(<color_leaf_iterator_ptr>state.it_ptr).getCoordinate()
    elif state.tree_type == 2:  # OcTreeStamped
        out[0] = deref(<stamped_leaf_iterator_ptr>state.it_ptr).getCoordinate()

cdef inline double _leaf_iterator_get_size(LeafIteratorState* state):
    """Extract size from current leaf iterator position"""
    if state.tree_type == 0:  # OcTree
        return deref(<leaf_iterator_ptr>state.it_ptr).getSize()
    elif state.tree_type == 1:  # ColorOcTree
        return deref(<color_leaf_iterator_ptr>state.it_ptr).getSize()
    elif state.tree_type == 2:  # OcTreeStamped
        return deref(<stamped_leaf_iterator_ptr>state.it_ptr).getSize()
    return 0.0

cdef inline int _leaf_iterator_get_depth(LeafIteratorState* state):
    """Extract depth from current leaf iterator position"""
    if state.tree_type == 0:  # OcTree
        return <int?>deref(<leaf_iterator_ptr>state.it_ptr).getDepth()
    elif state.tree_type == 1:  # ColorOcTree
        return <int?>deref(<color_leaf_iterator_ptr>state.it_ptr).getDepth()
    elif state.tree_type == 2:  # OcTreeStamped
        return <int?>deref(<stamped_leaf_iterator_ptr>state.it_ptr).getDepth()
    return 0

cdef inline void _leaf_iterator_advance(LeafIteratorState* state):
    """Advance leaf iterator to next position"""
    if state.tree_type == 0:  # OcTree
        inc(deref(<leaf_iterator_ptr>state.it_ptr))
    elif state.tree_type == 1:  # ColorOcTree
        inc(deref(<color_leaf_iterator_ptr>state.it_ptr))
    elif state.tree_type == 2:  # OcTreeStamped
        inc(deref(<stamped_leaf_iterator_ptr>state.it_ptr))

# Note: Cleanup must be done in __dealloc__ with typed pointers, not through void* casts

# ============================================================================
# Leaf BBX Iterator State Management (reuses LeafIteratorState structure)
# ============================================================================

cdef inline bint _leaf_bbx_iterator_at_end(LeafIteratorState* state):
    """Check if leaf BBX iterator has reached the end"""
    if state.is_end:
        return True
    
    if state.tree_type == 0:  # OcTree
        if state.it_ptr == NULL or state.end_ptr == NULL:
            return True
        return deref(<leaf_bbx_iterator_ptr>state.it_ptr) == deref(<leaf_bbx_iterator_ptr>state.end_ptr)
    elif state.tree_type == 1:  # ColorOcTree
        if state.it_ptr == NULL or state.end_ptr == NULL:
            return True
        return deref(<color_leaf_bbx_iterator_ptr>state.it_ptr) == deref(<color_leaf_bbx_iterator_ptr>state.end_ptr)
    elif state.tree_type == 2:  # OcTreeStamped
        if state.it_ptr == NULL or state.end_ptr == NULL:
            return True
        return deref(<stamped_leaf_bbx_iterator_ptr>state.it_ptr) == deref(<stamped_leaf_bbx_iterator_ptr>state.end_ptr)
    return True

cdef inline void _leaf_bbx_iterator_get_coordinate(LeafIteratorState* state, defs.point3d* out):
    """Extract coordinate from current leaf BBX iterator position"""
    if state.tree_type == 0:  # OcTree
        out[0] = deref(<leaf_bbx_iterator_ptr>state.it_ptr).getCoordinate()
    elif state.tree_type == 1:  # ColorOcTree
        out[0] = deref(<color_leaf_bbx_iterator_ptr>state.it_ptr).getCoordinate()
    elif state.tree_type == 2:  # OcTreeStamped
        out[0] = deref(<stamped_leaf_bbx_iterator_ptr>state.it_ptr).getCoordinate()

cdef inline double _leaf_bbx_iterator_get_size(LeafIteratorState* state):
    """Extract size from current leaf BBX iterator position"""
    if state.tree_type == 0:  # OcTree
        return deref(<leaf_bbx_iterator_ptr>state.it_ptr).getSize()
    elif state.tree_type == 1:  # ColorOcTree
        return deref(<color_leaf_bbx_iterator_ptr>state.it_ptr).getSize()
    elif state.tree_type == 2:  # OcTreeStamped
        return deref(<stamped_leaf_bbx_iterator_ptr>state.it_ptr).getSize()
    return 0.0

cdef inline int _leaf_bbx_iterator_get_depth(LeafIteratorState* state):
    """Extract depth from current leaf BBX iterator position"""
    if state.tree_type == 0:  # OcTree
        return <int?>deref(<leaf_bbx_iterator_ptr>state.it_ptr).getDepth()
    elif state.tree_type == 1:  # ColorOcTree
        return <int?>deref(<color_leaf_bbx_iterator_ptr>state.it_ptr).getDepth()
    elif state.tree_type == 2:  # OcTreeStamped
        return <int?>deref(<stamped_leaf_bbx_iterator_ptr>state.it_ptr).getDepth()
    return 0

cdef inline void _leaf_bbx_iterator_advance(LeafIteratorState* state):
    """Advance leaf BBX iterator to next position"""
    if state.tree_type == 0:  # OcTree
        inc(deref(<leaf_bbx_iterator_ptr>state.it_ptr))
    elif state.tree_type == 1:  # ColorOcTree
        inc(deref(<color_leaf_bbx_iterator_ptr>state.it_ptr))
    elif state.tree_type == 2:  # OcTreeStamped
        inc(deref(<stamped_leaf_bbx_iterator_ptr>state.it_ptr))

# Note: Cleanup must be done in __dealloc__ with typed pointers, not through void* casts

# ============================================================================
# Tree Iterator Implementation
# ============================================================================

cdef class SimpleTreeIterator:
    """
    Unified tree iterator supporting OcTree, ColorOcTree, and OcTreeStamped.
    
    Uses a clean abstraction layer that encapsulates type-specific iterator
    operations, making the code maintainable and extensible.
    """
    cdef object _tree
    cdef IteratorState _state
    # Typed pointers for cleanup (must match _state.tree_type)
    cdef tree_iterator_ptr _it_oc
    cdef tree_iterator_ptr _end_oc
    cdef color_tree_iterator_ptr _it_color
    cdef color_tree_iterator_ptr _end_color
    cdef stamped_tree_iterator_ptr _it_stamped
    cdef stamped_tree_iterator_ptr _end_stamped
    # Snapshot state for Python access
    cdef object _current_node
    cdef list _current_coord
    cdef double _current_size
    cdef int _current_depth
    cdef object _current_color
    cdef unsigned int _current_timestamp

    def __cinit__(self):
        self._tree = None
        self._state.it_ptr = NULL
        self._state.end_ptr = NULL
        self._state.tree_type = 0
        self._state.is_end = True
        self._it_oc = NULL
        self._end_oc = NULL
        self._it_color = NULL
        self._end_color = NULL
        self._it_stamped = NULL
        self._end_stamped = NULL
        self._current_node = None
        self._current_coord = None
        self._current_size = 0.0
        self._current_depth = 0
        self._current_color = None
        self._current_timestamp = 0

    def __dealloc__(self):
        # Clean up all iterator pointers that were allocated
        # Check each pointer individually to avoid leaks if initialization failed
        if self._it_oc != NULL:
            del self._it_oc
        if self._end_oc != NULL:
            del self._end_oc
        if self._it_color != NULL:
            del self._it_color
        if self._end_color != NULL:
            del self._end_color
        if self._it_stamped != NULL:
            del self._it_stamped
        if self._end_stamped != NULL:
            del self._end_stamped
        self._tree = None
        self._current_node = None

    def __init__(self, tree, maxDepth=0):
        """Initialize iterator for the given tree type"""
        from .octree import OcTree
        from .color_octree import ColorOcTree
        from .stamped_octree import OcTreeStamped
        
        cdef size_t ptr_addr
        cdef unsigned char depth = <unsigned char?>maxDepth
        cdef defs.OcTree* tree_ptr_oc = NULL
        cdef defs.ColorOcTree* tree_ptr_color = NULL
        cdef defs.OcTreeStamped* tree_ptr_stamped = NULL
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator tmp_it_oc
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator tmp_end_oc
        cdef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].tree_iterator tmp_it_color
        cdef defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].tree_iterator tmp_end_color
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].tree_iterator tmp_it_stamped
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].tree_iterator tmp_end_stamped
        
        if tree is None:
            self._state.is_end = True
            return
        
        if isinstance(tree, OcTree):
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_oc = <defs.OcTree*>ptr_addr
            if tree_ptr_oc == NULL:
                self._state.is_end = True
                return
            self._tree = tree
            self._state.tree_type = 0
            tmp_it_oc = tree_ptr_oc.begin_tree(depth)
            tmp_end_oc = tree_ptr_oc.end_tree()
            self._it_oc = new defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator(tmp_it_oc)
            self._end_oc = new defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator(tmp_end_oc)
            self._state.it_ptr = <void*>self._it_oc
            self._state.end_ptr = <void*>self._end_oc
            self._state.is_end = False
        elif isinstance(tree, ColorOcTree):
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_color = <defs.ColorOcTree*>ptr_addr
            if tree_ptr_color == NULL:
                self._state.is_end = True
                return
            self._tree = tree
            self._state.tree_type = 1
            tmp_it_color = tree_ptr_color.begin_tree(depth)
            tmp_end_color = tree_ptr_color.end_tree()
            self._it_color = new defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].tree_iterator(tmp_it_color)
            self._end_color = new defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].tree_iterator(tmp_end_color)
            self._state.it_ptr = <void*>self._it_color
            self._state.end_ptr = <void*>self._end_color
            self._state.is_end = False
        elif isinstance(tree, OcTreeStamped):
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_stamped = <defs.OcTreeStamped*>ptr_addr
            if tree_ptr_stamped == NULL:
                self._state.is_end = True
                return
            self._tree = tree
            self._state.tree_type = 2
            tmp_it_stamped = tree_ptr_stamped.begin_tree(depth)
            tmp_end_stamped = tree_ptr_stamped.end_tree()
            self._it_stamped = new defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].tree_iterator(tmp_it_stamped)
            self._end_stamped = new defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].tree_iterator(tmp_end_stamped)
            self._state.it_ptr = <void*>self._it_stamped
            self._state.end_ptr = <void*>self._end_stamped
            self._state.is_end = False
        else:
            self._state.is_end = True

    def __iter__(self):
        return self

    def __next__(self):
        """Advance iterator and return self"""
        cdef defs.point3d p
        cdef np.ndarray[DOUBLE_t, ndim=1] _pt
        
        if _iterator_at_end(&self._state):
            self._state.is_end = True
            raise StopIteration
        
        # Extract current state using unified abstraction
        _iterator_get_coordinate(&self._state, &p)
        self._current_coord = [p.x(), p.y(), p.z()]
        self._current_size = _iterator_get_size(&self._state)
        self._current_depth = _iterator_get_depth(&self._state)
        
        # Advance iterator
        _iterator_advance(&self._state)
        
        # Capture node by searching at current coordinate
        _pt = np.array(self._current_coord, dtype=np.float64)
        self._current_node = self._tree.search(_pt)
        
        return self

    def getCoordinate(self):
        """Get coordinate of current iterator position"""
        if self._current_coord is not None:
            return self._current_coord
        return [0.0, 0.0, 0.0]

    def getSize(self):
        """Get size of current node"""
        return self._current_size

    def getDepth(self):
        """Get depth of current node"""
        return self._current_depth
    
    def getColor(self):
        """
        Get color of current node (only for ColorOcTree). Returns (r, g, b).
        """
        if self._current_color is not None:
            return self._current_color
        return (255, 255, 255)

    def getTimestamp(self):
        """
        Get timestamp of current node (only for OcTreeStamped). Returns int.
        """
        return <unsigned int?>self._current_timestamp
    
    def _set_end(self):
        """Set iterator to end state (internal use)"""
        self._state.is_end = True

    def isLeaf(self):
        """Check if current node is a leaf"""
        if self._current_node is None:
            return True
        return not self._tree.nodeHasChildren(self._current_node)
    
    def _set_end(self):
        """Set iterator to end state (internal use)"""
        self._state.is_end = True

# ============================================================================
# Leaf Iterator Implementation
# ============================================================================

cdef class SimpleLeafIterator:
    """
    Unified leaf iterator supporting OcTree, ColorOcTree, and OcTreeStamped.

    Uses the same clean abstraction pattern as SimpleTreeIterator.
    """
    cdef object __weakref__  # Enable weak references
    cdef object _tree
    cdef LeafIteratorState _state
    # Typed pointers for cleanup (must match _state.tree_type)
    cdef leaf_iterator_ptr _it_oc
    cdef leaf_iterator_ptr _end_oc
    cdef color_leaf_iterator_ptr _it_color
    cdef color_leaf_iterator_ptr _end_color
    cdef stamped_leaf_iterator_ptr _it_stamped
    cdef stamped_leaf_iterator_ptr _end_stamped
    # Snapshot state for Python access
    cdef object _current_node
    cdef list _current_coord
    cdef double _current_size
    cdef int _current_depth
    cdef object _current_color
    cdef unsigned int _current_timestamp

    def __cinit__(self):
        self._tree = None
        self._state.it_ptr = NULL
        self._state.end_ptr = NULL
        self._state.tree_type = 0
        self._state.is_end = True
        self._it_oc = NULL
        self._end_oc = NULL
        self._it_color = NULL
        self._end_color = NULL
        self._it_stamped = NULL
        self._end_stamped = NULL
        self._current_node = None
        self._current_coord = None
        self._current_size = 0.0
        self._current_depth = 0
        self._current_color = None
        self._current_timestamp = 0

    def __dealloc__(self):
        # Clean up all iterator pointers that were allocated
        # Check each pointer individually to avoid leaks if initialization failed
        if self._it_oc != NULL:
            del self._it_oc
        if self._end_oc != NULL:
            del self._end_oc
        if self._it_color != NULL:
            del self._it_color
        if self._end_color != NULL:
            del self._end_color
        if self._it_stamped != NULL:
            del self._it_stamped
        if self._end_stamped != NULL:
            del self._end_stamped
        self._tree = None
        self._current_node = None

    def __init__(self, tree, maxDepth=0):
        """Initialize leaf iterator for the given tree type"""
        from .octree import OcTree
        from .color_octree import ColorOcTree
        from .stamped_octree import OcTreeStamped
        
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
            self._state.is_end = True
            return
        
        if isinstance(tree, OcTree):
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_oc = <defs.OcTree*>ptr_addr
            if tree_ptr_oc == NULL:
                self._state.is_end = True
                return
            # Store tree reference to enable node access (needed for current_node property)
            self._tree = tree
            self._state.tree_type = 0
            tmp_it_oc = tree_ptr_oc.begin_leafs(depth)
            tmp_end_oc = tree_ptr_oc.end_leafs()
            self._it_oc = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator(tmp_it_oc)
            self._end_oc = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator(tmp_end_oc)
            self._state.it_ptr = <void*>self._it_oc
            self._state.end_ptr = <void*>self._end_oc
            self._state.is_end = False
        elif isinstance(tree, ColorOcTree):
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_color = <defs.ColorOcTree*>ptr_addr
            if tree_ptr_color == NULL:
                self._state.is_end = True
                return
            # Store tree reference to enable node access (needed for current_node property)
            self._tree = tree
            self._state.tree_type = 1
            tmp_it_color = tree_ptr_color.begin_leafs(depth)
            tmp_end_color = tree_ptr_color.end_leafs()
            self._it_color = new defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_iterator(tmp_it_color)
            self._end_color = new defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_iterator(tmp_end_color)
            self._state.it_ptr = <void*>self._it_color
            self._state.end_ptr = <void*>self._end_color
            self._state.is_end = False
        elif isinstance(tree, OcTreeStamped):
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_stamped = <defs.OcTreeStamped*>ptr_addr
            if tree_ptr_stamped == NULL:
                self._state.is_end = True
                return
            # Store tree reference to enable node access (needed for current_node property)
            self._tree = tree
            self._state.tree_type = 2
            tmp_it_stamped = tree_ptr_stamped.begin_leafs(depth)
            tmp_end_stamped = tree_ptr_stamped.end_leafs()
            self._it_stamped = new defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_iterator(tmp_it_stamped)
            self._end_stamped = new defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_iterator(tmp_end_stamped)
            self._state.it_ptr = <void*>self._it_stamped
            self._state.end_ptr = <void*>self._end_stamped
            self._state.is_end = False
        else:
            self._state.is_end = True

    def __iter__(self):
        return self

    def __next__(self):
        """Advance iterator and return self"""
        cdef defs.point3d p
        cdef defs.ColorOcTreeNode* node_ptr_color
        cdef defs.OcTreeNodeStamped* node_ptr_stamped
        cdef defs.ColorOcTreeNode.Color c
        cdef np.ndarray[DOUBLE_t, ndim=1] _pt

        if _leaf_iterator_at_end(&self._state):
            self._state.is_end = True
            raise StopIteration

        # Extract current state using unified abstraction
        _leaf_iterator_get_coordinate(&self._state, &p)
        self._current_coord = [p.x(), p.y(), p.z()]
        self._current_size = _leaf_iterator_get_size(&self._state)
        self._current_depth = _leaf_iterator_get_depth(&self._state)

        # Initialize cached data
        self._current_node = None
        self._current_color = None
        self._current_timestamp = 0

        # Extract color/timestamp data based on tree type
        if self._state.tree_type == 1:  # ColorOcTree
            try:
                node_ptr_color = <defs.ColorOcTreeNode*>&deref(deref(<color_leaf_iterator_ptr>self._state.it_ptr))
                c = node_ptr_color.getColor()
                self._current_color = (c.r, c.g, c.b)
            except Exception:
                self._current_color = None
        elif self._state.tree_type == 2:  # OcTreeStamped
            try:
                node_ptr_stamped = <defs.OcTreeNodeStamped*>&deref(deref(<stamped_leaf_iterator_ptr>self._state.it_ptr))
                self._current_timestamp = node_ptr_stamped.getTimestamp()
            except Exception:
                self._current_timestamp = 0

        # Advance iterator
        _leaf_iterator_advance(&self._state)
        
        # Don't create node here - use lazy property instead for efficiency

        return self

    def getCoordinate(self):
        """Get coordinate of current iterator position"""
        if self._current_coord is not None:
            return self._current_coord
        return [0.0, 0.0, 0.0]

    def getSize(self):
        """Get size of current node"""
        return self._current_size

    def getDepth(self):
        """Get depth of current node"""
        return self._current_depth

    def getColor(self):
        """Get color of current node (only for ColorOcTree). Returns (r, g, b)."""
        if self._current_color is not None:
            return self._current_color
        return (255, 255, 255)  # Default white

    def getTimestamp(self):
        """Get timestamp of current node (only for OcTreeStamped). Returns int."""
        return self._current_timestamp
    
    @property
    def current_node(self):
        """Get the current node wrapper object (lazy evaluation - only searches when accessed)"""
        # Lazy evaluation: only search for node when property is accessed
        cdef np.ndarray[DOUBLE_t, ndim=1] _pt
        if self._current_node is None and self._tree is not None and self._current_coord is not None:
            try:
                _pt = np.array(self._current_coord, dtype=np.float64)
                self._current_node = self._tree.search(_pt)
            except Exception:
                self._current_node = None
        return self._current_node
    
    def _set_end(self):
        """Set iterator to end state (internal use)"""
        self._state.is_end = True

    def _set_end(self):
        """Set iterator to end state (internal use)"""
        self._state.is_end = True

# ============================================================================
# Leaf BBX Iterator Implementation
# ============================================================================

cdef class SimpleLeafBBXIterator:
    """
    Unified leaf bounding box iterator supporting OcTree, ColorOcTree, and OcTreeStamped.

    Uses the same clean abstraction pattern with additional bounding box filtering.
    """
    cdef object __weakref__  # Enable weak references
    cdef object _tree
    cdef LeafIteratorState _state
    # Typed pointers for cleanup (must match _state.tree_type)
    cdef leaf_bbx_iterator_ptr _it_oc
    cdef leaf_bbx_iterator_ptr _end_oc
    cdef color_leaf_bbx_iterator_ptr _it_color
    cdef color_leaf_bbx_iterator_ptr _end_color
    cdef stamped_leaf_bbx_iterator_ptr _it_stamped
    cdef stamped_leaf_bbx_iterator_ptr _end_stamped
    # Bounding box (use object instead of buffer type for class attributes)
    cdef object _bbx_min
    cdef object _bbx_max
    # Snapshot state for Python access
    cdef object _current_node
    cdef list _current_coord
    cdef double _current_size
    cdef int _current_depth
    cdef object _current_color
    cdef unsigned int _current_timestamp

    def __cinit__(self):
        self._tree = None
        self._state.it_ptr = NULL
        self._state.end_ptr = NULL
        self._state.tree_type = 0
        self._state.is_end = True
        self._it_oc = NULL
        self._end_oc = NULL
        self._it_color = NULL
        self._end_color = NULL
        self._it_stamped = NULL
        self._end_stamped = NULL
        self._current_node = None
        self._current_coord = None
        self._current_size = 0.0
        self._current_depth = 0
        self._current_color = None
        self._current_timestamp = 0

    def __dealloc__(self):
        # Clean up all iterator pointers that were allocated
        # Check each pointer individually to avoid leaks if initialization failed
        if self._it_oc != NULL:
            del self._it_oc
        if self._end_oc != NULL:
            del self._end_oc
        if self._it_color != NULL:
            del self._it_color
        if self._end_color != NULL:
            del self._end_color
        if self._it_stamped != NULL:
            del self._it_stamped
        if self._end_stamped != NULL:
            del self._end_stamped
        # Clear Python object references
        self._tree = None
        self._bbx_min = None
        self._bbx_max = None
        self._current_node = None

    def __init__(self, tree, np.ndarray[DOUBLE_t, ndim=1] bbx_min, 
                 np.ndarray[DOUBLE_t, ndim=1] bbx_max, maxDepth=0):
        """Initialize leaf BBX iterator for the given tree type"""
        from .octree import OcTree
        from .color_octree import ColorOcTree
        from .stamped_octree import OcTreeStamped
        
        cdef size_t ptr_addr
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
        cdef defs.point3d min_pt = defs.point3d(bbx_min[0], bbx_min[1], bbx_min[2])
        cdef defs.point3d max_pt = defs.point3d(bbx_max[0], bbx_max[1], bbx_max[2])
        
        self._bbx_min = bbx_min.copy()
        self._bbx_max = bbx_max.copy()
        
        if tree is None:
            self._state.is_end = True
            return
        
        if isinstance(tree, OcTree):
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_oc = <defs.OcTree*>ptr_addr
            if tree_ptr_oc == NULL:
                self._state.is_end = True
                return
            self._tree = tree
            self._state.tree_type = 0
            tmp_it_oc = tree_ptr_oc.begin_leafs_bbx(min_pt, max_pt, depth)
            tmp_end_oc = tree_ptr_oc.end_leafs_bbx()
            self._it_oc = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator(tmp_it_oc)
            self._end_oc = new defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator(tmp_end_oc)
            self._state.it_ptr = <void*>self._it_oc
            self._state.end_ptr = <void*>self._end_oc
            self._state.is_end = False
        elif isinstance(tree, ColorOcTree):
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_color = <defs.ColorOcTree*>ptr_addr
            if tree_ptr_color == NULL:
                self._state.is_end = True
                return
            self._tree = tree
            self._state.tree_type = 1
            tmp_it_color = tree_ptr_color.begin_leafs_bbx(min_pt, max_pt, depth)
            tmp_end_color = tree_ptr_color.end_leafs_bbx()
            self._it_color = new defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_bbx_iterator(tmp_it_color)
            self._end_color = new defs.OccupancyOcTreeBase[defs.ColorOcTreeNode].leaf_bbx_iterator(tmp_end_color)
            self._state.it_ptr = <void*>self._it_color
            self._state.end_ptr = <void*>self._end_color
            self._state.is_end = False
        elif isinstance(tree, OcTreeStamped):
            ptr_addr = tree._get_ptr_addr()
            tree_ptr_stamped = <defs.OcTreeStamped*>ptr_addr
            if tree_ptr_stamped == NULL:
                self._state.is_end = True
                return
            self._tree = tree
            self._state.tree_type = 2
            tmp_it_stamped = tree_ptr_stamped.begin_leafs_bbx(min_pt, max_pt, depth)
            tmp_end_stamped = tree_ptr_stamped.end_leafs_bbx()
            self._it_stamped = new defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_bbx_iterator(tmp_it_stamped)
            self._end_stamped = new defs.OccupancyOcTreeBase[defs.OcTreeNodeStamped].leaf_bbx_iterator(tmp_end_stamped)
            self._state.it_ptr = <void*>self._it_stamped
            self._state.end_ptr = <void*>self._end_stamped
            self._state.is_end = False
        else:
            self._state.is_end = True

    def __iter__(self):
        return self

    def __next__(self):
        """Advance iterator and return self"""
        cdef defs.point3d p
        cdef defs.ColorOcTreeNode* node_ptr_color
        cdef defs.OcTreeNodeStamped* node_ptr_stamped
        cdef defs.ColorOcTreeNode.Color c
        cdef np.ndarray[DOUBLE_t, ndim=1] _pt

        if _leaf_bbx_iterator_at_end(&self._state):
            self._state.is_end = True
            raise StopIteration

        # Extract current state using unified abstraction
        _leaf_bbx_iterator_get_coordinate(&self._state, &p)
        self._current_coord = [p.x(), p.y(), p.z()]
        self._current_size = _leaf_bbx_iterator_get_size(&self._state)
        self._current_depth = _leaf_bbx_iterator_get_depth(&self._state)

        # Initialize cached data
        self._current_node = None
        self._current_color = None
        self._current_timestamp = 0

        # Extract color/timestamp data based on tree type
        if self._state.tree_type == 1:  # ColorOcTree
            try:
                node_ptr_color = <defs.ColorOcTreeNode*>&deref(deref(<color_leaf_bbx_iterator_ptr>self._state.it_ptr))
                c = node_ptr_color.getColor()
                self._current_color = (c.r, c.g, c.b)
            except Exception:
                self._current_color = None
        elif self._state.tree_type == 2:  # OcTreeStamped
            try:
                node_ptr_stamped = <defs.OcTreeNodeStamped*>&deref(deref(<stamped_leaf_bbx_iterator_ptr>self._state.it_ptr))
                self._current_timestamp = node_ptr_stamped.getTimestamp()
            except Exception:
                self._current_timestamp = 0

        # Advance iterator
        _leaf_bbx_iterator_advance(&self._state)

        # Capture node by searching at current coordinate
        _pt = np.array(self._current_coord, dtype=np.float64)
        self._current_node = self._tree.search(_pt)

        return self

    def getCoordinate(self):
        """Get coordinate of current iterator position"""
        if self._current_coord is not None:
            return self._current_coord
        return [0.0, 0.0, 0.0]

    def getSize(self):
        """Get size of current node"""
        return self._current_size

    def getDepth(self):
        """Get depth of current node"""
        return self._current_depth

    def getColor(self):
        """Get color of current node (ColorOcTree only)"""
        if self._current_color is not None:
            return self._current_color
        return (255, 255, 255)  # Default white

    def getTimestamp(self):
        """Get timestamp of current node (OcTreeStamped only)"""
        return self._current_timestamp

    def _set_end(self):
        """Set iterator to end state (internal use)"""
        self._state.is_end = True
