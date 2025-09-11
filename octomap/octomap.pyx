from libcpp.string cimport string
from libcpp cimport bool as cppbool
from libc.string cimport memcpy
from cython.operator cimport dereference as deref, preincrement as inc, address
cimport octomap_defs as defs
cimport dynamicEDT3D_defs as edt
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator* tree_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator* leaf_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator* leaf_bbx_iterator_ptr

class NullPointerException(Exception):
    """
    Null pointer exception
    """
    def __init__(self):
        pass

cdef class OcTreeKey:
    """
    OcTreeKey is a container class for internal key addressing.
    The keys count the number of cells (voxels) from the origin as discrete address of a voxel.
    """
    cdef defs.OcTreeKey *thisptr
    def __cinit__(self, unsigned short int a=0, unsigned short int b=0, unsigned short int c=0):
        if a == 0 and b == 0 and c == 0:
            self.thisptr = new defs.OcTreeKey()
        else:
            self.thisptr = new defs.OcTreeKey(a, b, c)
    def __dealloc__(self):
        if self.thisptr != NULL:
            del self.thisptr
            self.thisptr = NULL
    def __richcmp__(self, other, int op):
        if op == 2:
            return (self.thisptr[0][0] == other[0] and \
                    self.thisptr[0][1] == other[1] and \
                    self.thisptr[0][2] == other[2])
        elif op == 3:
            return not (self.thisptr[0][0] == other[0] and \
                        self.thisptr[0][1] == other[1] and \
                        self.thisptr[0][2] == other[2])
    
    def __getitem__(self, unsigned int i):
        if self.thisptr:
            return self.thisptr[0][i]
        else:
            raise NullPointerException
    def __setitem__(self, unsigned int i, unsigned int value):
        if self.thisptr:
            self.thisptr[0][i] = value
        else:
            raise NullPointerException
            
    def __repr__(self):
        if self.thisptr:
            return f"OcTreeKey({self.thisptr[0][0]}, {self.thisptr[0][1]}, {self.thisptr[0][2]})"
        else:
            return "OcTreeKey(NULL)"
            
    def computeChildIdx(self, OcTreeKey key, int depth):
        cdef unsigned int result
        cdef defs.OcTreeKey key_in = defs.OcTreeKey()
        key_in[0] = key[0]
        key_in[1] = key[1]
        key_in[2] = key[2]
        if self.thisptr:
            result = defs.computeChildIdx(key_in, depth)
            return result
        else:
            raise NullPointerException
    def computeIndexKey(self, unsigned int level, OcTreeKey key):
        cdef defs.OcTreeKey key_in = defs.OcTreeKey()
        cdef defs.OcTreeKey result
        key_in[0] = key[0]
        key_in[1] = key[1]
        key_in[2] = key[2]
        if self.thisptr:
            result = defs.computeIndexKey(level, key_in)
            # Convert back to Python OcTreeKey
            return OcTreeKey(result[0], result[1], result[2])
        else:
            raise NullPointerException

cdef class OcTreeNode:
    """
    Nodes to be used in OcTree.
    They represent 3d occupancy grid cells. "value" stores their log-odds occupancy.
    """
    cdef defs.OcTreeNode *thisptr
    def __cinit__(self):
        pass
    def __dealloc__(self):
        pass
    def addValue(self, float p):
        """
        adds p to the node's logOdds value (with no boundary / threshold checking!)
        """
        if self.thisptr:
            self.thisptr.addValue(p)
        else:
            raise NullPointerException
    def childExists(self, unsigned int i):
        """
        Safe test to check of the i-th child exists,
        first tests if there are any children.
        """
        if self.thisptr:
            return self.thisptr.childExists(i)
        else:
            raise NullPointerException
    def getValue(self):
        if self.thisptr:
            return self.thisptr.getValue()
        else:
            raise NullPointerException
    def setValue(self, float v):
        if self.thisptr:
            self.thisptr.setValue(v)
        else:
            raise NullPointerException
    def getOccupancy(self):
        if self.thisptr:
            return self.thisptr.getOccupancy()
        else:
            raise NullPointerException
    def getLogOdds(self):
        if self.thisptr:
            return self.thisptr.getLogOdds()
        else:
            raise NullPointerException
    def setLogOdds(self, float l):
        if self.thisptr:
            self.thisptr.setLogOdds(l)
        else:
            raise NullPointerException
    def hasChildren(self):
        """
        Deprecated: Use tree.nodeHasChildren(node) instead.
        This method is kept for backward compatibility but will show deprecation warnings.
        """
        if self.thisptr:
            return self.thisptr.hasChildren()
        else:
            raise NullPointerException
    def getMaxChildLogOdds(self):
        if self.thisptr:
            return self.thisptr.getMaxChildLogOdds()
        else:
            raise NullPointerException
    def updateOccupancyChildren(self):
        if self.thisptr:
            self.thisptr.updateOccupancyChildren()
        else:
            raise NullPointerException

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

    def __init__(self, OcTree tree, maxDepth=0):
        if tree is None or tree.thisptr == NULL:
            self._is_end = True
            return
        self._tree = tree
        cdef unsigned char depth = <unsigned char?>maxDepth
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator tmp_it = tree.thisptr.begin_tree(depth)
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator tmp_end = tree.thisptr.end_tree()
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

        # Capture node by searching at current coordinate (robust, avoids operator* issues)
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

    def __init__(self, OcTree tree, maxDepth=0):
        if tree is None or tree.thisptr == NULL:
            self._is_end = True
            return
        self._tree = tree
        cdef unsigned char depth = <unsigned char?>maxDepth
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator tmp_it = tree.thisptr.begin_leafs(depth)
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator tmp_end = tree.thisptr.end_leafs()
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

        cdef np.ndarray[DOUBLE_t, ndim=1] _pt2 = np.array(self._current_coord, dtype=np.float64)
        self._current_node = self._tree.search(_pt2)

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

    @property
    def current_node(self):
        return self._current_node
    
    def getSize(self):
        """Get the size of the current node"""
        if self._current_node:
            # Return a more reasonable size based on the tree resolution
            return self._tree.getResolution() * 2.0  # Simplified: 2x resolution
        return 0.0
    
    def getDepth(self):
        """Get the depth of the current node"""
        if self._current_node:
            return 0  # This would need to be tracked properly
        return 0

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

    def __init__(self, OcTree tree, np.ndarray[DOUBLE_t, ndim=1] bbx_min, np.ndarray[DOUBLE_t, ndim=1] bbx_max, maxDepth=0):
        if tree is None or tree.thisptr == NULL:
            self._is_end = True
            return
        self._tree = tree
        cdef defs.point3d pmin = defs.point3d(<float?>bbx_min[0], <float?>bbx_min[1], <float?>bbx_min[2])
        cdef defs.point3d pmax = defs.point3d(<float?>bbx_max[0], <float?>bbx_max[1], <float?>bbx_max[2])
        cdef unsigned char depth = <unsigned char?>maxDepth
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator tmp_it = tree.thisptr.begin_leafs_bbx(pmin, pmax, depth)
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator tmp_end = tree.thisptr.end_leafs_bbx()
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

        cdef np.ndarray[DOUBLE_t, ndim=1] _pt3 = np.array(self._current_coord, dtype=np.float64)
        self._current_node = self._tree.search(_pt3)

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

    @property
    def current_node(self):
        return self._current_node

    @property
    def is_end(self):
        return self._is_end

def _octree_read(filename):
    """
    Read the file header, create the appropriate class and deserialize.
    This creates a new octree which you need to delete yourself.
    """
    cdef defs.istringstream iss
    cdef OcTree tree = OcTree(0.1)
    cdef string c_filename = filename.encode('utf-8')
    cdef defs.OcTree* new_tree = NULL
    
    if filename.startswith(b"# Octomap OcTree file"):
        iss.str(string(<char*?>filename, len(filename)))
        new_tree = <defs.OcTree*>tree.thisptr.read(<defs.istream&?>iss)
    else:
        new_tree = <defs.OcTree*>tree.thisptr.read(c_filename)
    
    if new_tree != NULL:
        # Clean up the original tree and replace with the loaded one
        if tree.thisptr != NULL:
            del tree.thisptr
        tree.thisptr = new_tree
        tree.owner = True
    
    return tree

cdef class OcTree:
    """
    octomap main map data structure, stores 3D occupancy grid map in an OcTree.
    """
    cdef defs.OcTree *thisptr
    cdef edt.DynamicEDTOctomap *edtptr
    cdef bint owner
    
    def __cinit__(self, arg):
        import numbers
        self.owner = True
        self.edtptr = NULL  # Initialize to NULL
        if isinstance(arg, numbers.Number):
            self.thisptr = new defs.OcTree(<double?>arg)
        else:
            self.thisptr = new defs.OcTree(string(<char*?>arg))

    def __dealloc__(self):
        # Clean up DynamicEDT first (it may reference the tree)
        if self.edtptr != NULL:
            del self.edtptr
            self.edtptr = NULL

        # Then clean up the OcTree itself
        if self.owner and self.thisptr != NULL:
            del self.thisptr
            self.thisptr = NULL

    def adjustKeyAtDepth(self, OcTreeKey key, depth):
        cdef defs.OcTreeKey key_in = defs.OcTreeKey()
        key_in[0] = key[0]
        key_in[1] = key[1]
        key_in[2] = key[2]
        cdef defs.OcTreeKey key_out = self.thisptr.adjustKeyAtDepth(key_in, <int?>depth)
        res = OcTreeKey()
        res[0] = key_out[0]
        res[1] = key_out[1]
        res[2] = key_out[2]
        return res

    def bbxSet(self):
        return self.thisptr.bbxSet()

    def calcNumNodes(self):
        return self.thisptr.calcNumNodes()

    def clear(self):
        self.thisptr.clear()

    def coordToKey(self, np.ndarray[DOUBLE_t, ndim=1] coord, depth=None):
        cdef defs.OcTreeKey key
        if depth is None:
            key = self.thisptr.coordToKey(defs.point3d(coord[0],
                                                       coord[1],
                                                       coord[2]))
        else:
            key = self.thisptr.coordToKey(defs.point3d(coord[0],
                                                       coord[1],
                                                       coord[2]),
                                          <unsigned int?>depth)
        res = OcTreeKey()
        res[0] = key[0]
        res[1] = key[1]
        res[2] = key[2]
        return res

    def coordToKeyChecked(self, np.ndarray[DOUBLE_t, ndim=1] coord, depth=None):
        cdef defs.OcTreeKey key
        cdef cppbool chk
        if depth is None:
            chk = self.thisptr.coordToKeyChecked(defs.point3d(coord[0],
                                                              coord[1],
                                                              coord[2]),
                                                 key)
        else:
            chk = self.thisptr.coordToKeyChecked(defs.point3d(coord[0],
                                                              coord[1],
                                                              coord[2]),
                                                 <unsigned int?>depth,
                                                 key)
        if chk:
            res = OcTreeKey()
            res[0] = key[0]
            res[1] = key[1]
            res[2] = key[2]
            return chk, res
        else:
            return chk, None

    def deleteNode(self, np.ndarray[DOUBLE_t, ndim=1] value, depth=1):
        return self.thisptr.deleteNode(defs.point3d(value[0],
                                                    value[1],
                                                    value[2]),
                                       <int?>depth)

    def castRay(self, np.ndarray[DOUBLE_t, ndim=1] origin,
                np.ndarray[DOUBLE_t, ndim=1] direction,
                np.ndarray[DOUBLE_t, ndim=1] end,
                ignoreUnknownCells=False,
                maxRange=-1.0):
        """
        A ray is cast from origin with a given direction,
        the first occupied cell is returned (as center coordinate).
        If the starting coordinate is already occupied in the tree,
        this coordinate will be returned as a hit.
        """
        cdef defs.point3d e
        cdef cppbool hit
        hit = self.thisptr.castRay(
            defs.point3d(origin[0], origin[1], origin[2]),
            defs.point3d(direction[0], direction[1], direction[2]),
            e,
            bool(ignoreUnknownCells),
            <double?>maxRange
        )
        if hit:
            end[0:3] = e.x(), e.y(), e.z()
        return hit
    
    def read(self, filename):
        cdef string c_filename = filename.encode('utf-8')
        cdef defs.OcTree* result
        result = <defs.OcTree*>self.thisptr.read(c_filename)
        if result != NULL:
            # Create new OcTree instance with the loaded data
            new_tree = OcTree(0.1)  # Temporary resolution, will be overwritten
            new_tree.thisptr = result
            new_tree.owner = True
            return new_tree
        return None
    
    def write(self, filename=None):
        """
        Write file header and complete tree to file/stream (serialization)
        """
        cdef defs.ostringstream oss
        cdef string c_filename
        if not filename is None:
            c_filename = filename.encode('utf-8')
            return self.thisptr.write(c_filename)
        else:
            ret = self.thisptr.write(<defs.ostream&?>oss)
            if ret:
                return oss.str().c_str()[:oss.str().length()]
            else:
                return False

    def readBinary(self, filename):
        # Treat input as a filesystem path; accept str or bytes
        cdef string c_filename
        if isinstance(filename, (bytes, bytearray)):
            c_filename = (<bytes>filename).decode('utf-8')
        else:
            c_filename = (<str>filename).encode('utf-8')
        return self.thisptr.readBinary(c_filename)

    def writeBinary(self, filename=None):
        cdef defs.ostringstream oss
        cdef string c_filename
        if not filename is None:
            c_filename = filename.encode('utf-8')
            return self.thisptr.writeBinary(c_filename)
        else:
            ret = self.thisptr.writeBinary(<defs.ostream&?>oss)
            if ret:
                return oss.str().c_str()[:oss.str().length()]
            else:
                return False

    def isNodeOccupied(self, node):
        if isinstance(node, OcTreeNode):
            if (<OcTreeNode>node).thisptr:
                return self.thisptr.isNodeOccupied(deref((<OcTreeNode>node).thisptr))
            else:
                raise NullPointerException
        elif isinstance(node, (SimpleTreeIterator, SimpleLeafIterator, SimpleLeafBBXIterator)):
            # Handle iterator case - get the current node from the iterator
            if hasattr(node, '_current_node') and node._current_node is not None:
                if isinstance(node._current_node, OcTreeNode) and (<OcTreeNode>node._current_node).thisptr:
                    return self.thisptr.isNodeOccupied(deref((<OcTreeNode>node._current_node).thisptr))
                else:
                    raise NullPointerException
            else:
                # Iterator doesn't have a valid current node
                raise RuntimeError("Iterator has no current node - ensure iterator is properly initialized and advanced")
        else:
            raise TypeError(f"Expected OcTreeNode or iterator, got {type(node)}")

    def isNodeAtThreshold(self, node):
        if isinstance(node, OcTreeNode):
            if (<OcTreeNode>node).thisptr:
                return self.thisptr.isNodeAtThreshold(deref((<OcTreeNode>node).thisptr))
            else:
                raise NullPointerException
        elif isinstance(node, (SimpleTreeIterator, SimpleLeafIterator, SimpleLeafBBXIterator)):
            # Handle iterator case - get the current node from the iterator
            if hasattr(node, '_current_node') and node._current_node is not None:
                if isinstance(node._current_node, OcTreeNode) and (<OcTreeNode>node._current_node).thisptr:
                    return self.thisptr.isNodeAtThreshold(deref((<OcTreeNode>node._current_node).thisptr))
                else:
                    raise NullPointerException
            else:
                # Iterator doesn't have a valid current node
                raise RuntimeError("Iterator has no current node - ensure iterator is properly initialized and advanced")
        else:
            raise TypeError(f"Expected OcTreeNode or iterator, got {type(node)}")

    def getLabels(self, np.ndarray[DOUBLE_t, ndim=2] points):
        cdef int i
        cdef np.ndarray[DOUBLE_t, ndim=1] pt
        cdef OcTreeKey key
        cdef OcTreeNode node
        # -1: unknown, 0: empty, 1: occupied
        cdef np.ndarray[np.int32_t, ndim=1] labels = \
            np.full((points.shape[0],), -1, dtype=np.int32)
        for i, pt in enumerate(points):
            key = self.coordToKey(pt)
            node = self.search(key)
            if node is None:
                labels[i] = -1
            else:
                try:
                    labels[i] = 1 if self.isNodeOccupied(node) else 0
                except Exception:
                    labels[i] = -1
        return labels
    
    def extractPointCloud(self):
        cdef float resolution = self.getResolution()

        cdef list occupied = []
        cdef list empty = []
        cdef SimpleLeafIterator it
        cdef float size
        cdef int is_occupied
        cdef np.ndarray[DOUBLE_t, ndim=1] center
        cdef np.ndarray[DOUBLE_t, ndim=1] origin
        cdef np.ndarray[np.int64_t, ndim=2] indices
        cdef np.ndarray[DOUBLE_t, ndim=2] points
        cdef np.ndarray keep
        cdef int dimension
        for it in self.begin_leafs():
            # Try to get occupancy status from the iterator
            try:
                is_occupied = self.isNodeOccupied(it)
            except:
                # Fallback: assume occupied if we can't determine status
                is_occupied = True
            size = it.getSize()
            center = np.array(it.getCoordinate(), dtype=np.float64)

            # Limit dimension to prevent memory issues
            raw_dimension = max(1, round(it.getSize() / resolution))
            dimension = min(raw_dimension, 100)  # Cap at 100 to prevent memory issues
            origin = center - (dimension / 2 - 0.5) * resolution
            indices = np.column_stack(np.nonzero(np.ones((dimension, dimension, dimension))))
            points = origin + indices * np.array(resolution)

            if is_occupied:
                occupied.append(points)
            else:
                empty.append(points)

        cdef np.ndarray[DOUBLE_t, ndim=2] occupied_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] empty_arr
        if len(occupied) == 0:
            occupied_arr = np.zeros((0, 3), dtype=float)
        else:
            occupied_arr = np.concatenate(occupied, axis=0)
        if len(empty) == 0:
            empty_arr = np.zeros((0, 3), dtype=float)
        else:
            empty_arr = np.concatenate(empty, axis=0)
        return occupied_arr, empty_arr
    
    def insertPointCloud(self,
                         np.ndarray[DOUBLE_t, ndim=2] pointcloud,
                         np.ndarray[DOUBLE_t, ndim=1] origin,
                         maxrange=-1.,
                         lazy_eval=False,
                         discretize=False):
        """
        Integrate a Pointcloud (in global reference frame), parallelized with OpenMP.

        Special care is taken that each voxel in the map is updated only once, and occupied
        nodes have a preference over free ones. This avoids holes in the floor from mutual
        deletion.
        :param pointcloud: Pointcloud (measurement endpoints), in global reference frame
        :param origin: measurement origin in global reference frame
        :param maxrange: maximum range for how long individual beams are inserted (default -1: complete beam)
        :param : whether update of inner nodes is omitted after the update (default: false).
        This speeds up the insertion, but you need to call updateInnerOccupancy() when done.
        """
        cdef defs.Pointcloud pc = defs.Pointcloud()
        for p in pointcloud:
            pc.push_back(<float>p[0],
                         <float>p[1],
                         <float>p[2])

        self.thisptr.insertPointCloud(pc,
                                      defs.Vector3(<float>origin[0],
                                                   <float>origin[1],
                                                   <float>origin[2]),
                                      <double?>maxrange,
                                      bool(lazy_eval),
                                      bool(discretize))

    def begin_tree(self, maxDepth=0):
        """Return a simplified tree iterator"""
        return SimpleTreeIterator(self, maxDepth)

    def begin_leafs(self, maxDepth=0):
        """Return a simplified leaf iterator"""
        return SimpleLeafIterator(self, maxDepth)

    def begin_leafs_bbx(self, np.ndarray[DOUBLE_t, ndim=1] bbx_min, np.ndarray[DOUBLE_t, ndim=1] bbx_max, maxDepth=0):
        """Return a simplified leaf iterator for a bounding box"""
        return SimpleLeafBBXIterator(self, bbx_min, bbx_max, maxDepth)

    def end_tree(self):
        """Return an end iterator for tree traversal"""
        itr = SimpleTreeIterator(self)
        itr._is_end = True
        return itr

    def end_leafs(self):
        """Return an end iterator for leaf traversal"""
        itr = SimpleLeafIterator(self)
        itr._is_end = True
        return itr

    def end_leafs_bbx(self):
        """Return an end iterator for leaf bounding box traversal"""
        itr = SimpleLeafBBXIterator(self, np.array([0.0, 0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0, 1.0], dtype=np.float64))
        itr._is_end = True
        itr._sampled_points = []  # Clear sampled points to ensure it's empty
        return itr

    def getBBXBounds(self):
        cdef defs.point3d p = self.thisptr.getBBXBounds()
        return np.array((p.x(), p.y(), p.z()))

    def getBBXCenter(self):
        cdef defs.point3d p = self.thisptr.getBBXCenter()
        return np.array((p.x(), p.y(), p.z()))

    def getBBXMax(self):
        cdef defs.point3d p = self.thisptr.getBBXMax()
        return np.array((p.x(), p.y(), p.z()))

    def getBBXMin(self):
        cdef defs.point3d p = self.thisptr.getBBXMin()
        return np.array((p.x(), p.y(), p.z()))

    def getRoot(self):
        node = OcTreeNode()
        node.thisptr = self.thisptr.getRoot()
        return node

    def getNumLeafNodes(self):
        return self.thisptr.getNumLeafNodes()

    def getResolution(self):
        return self.thisptr.getResolution()

    def getTreeDepth(self):
        return self.thisptr.getTreeDepth()

    def getTreeType(self):
        return self.thisptr.getTreeType().c_str()

    def inBBX(self, np.ndarray[DOUBLE_t, ndim=1] p):
        return self.thisptr.inBBX(defs.point3d(p[0], p[1], p[2]))

    def keyToCoord(self, OcTreeKey key, depth=None):
        cdef defs.OcTreeKey key_in = defs.OcTreeKey()
        cdef defs.point3d p = defs.point3d()
        key_in[0] = key[0]
        key_in[1] = key[1]
        key_in[2] = key[2]
        if depth is None:
            p = self.thisptr.keyToCoord(key_in)
        else:
            p = self.thisptr.keyToCoord(key_in, <int?>depth)
        return np.array((p.x(), p.y(), p.z()))

    def memoryFullGrid(self):
        return self.thisptr.memoryFullGrid()

    def memoryUsage(self):
        return self.thisptr.memoryUsage()

    def memoryUsageNode(self):
        return self.thisptr.memoryUsageNode()

    def resetChangeDetection(self):
        """
        Reset the set of changed keys. Call this after you obtained all changed nodes.
        """
        self.thisptr.resetChangeDetection()
    

        
    def search(self, value, depth=0):
        node = OcTreeNode()
        if isinstance(value, OcTreeKey):
            node.thisptr = self.thisptr.search(defs.OcTreeKey(<unsigned short int>value[0],
                                                              <unsigned short int>value[1],
                                                              <unsigned short int>value[2]),
                                               <unsigned int?>depth)
        else:
            node.thisptr = self.thisptr.search(<double>value[0],
                                               <double>value[1],
                                               <double>value[2],
                                               <unsigned int?>depth)
        # Return None if the search failed (thisptr is NULL)
        if node.thisptr == NULL:
            return None
        return node

    def setBBXMax(self, np.ndarray[DOUBLE_t, ndim=1] max):
        """
        sets the maximum for a query bounding box to use
        """
        self.thisptr.setBBXMax(defs.point3d(max[0], max[1], max[2]))

    def setBBXMin(self, np.ndarray[DOUBLE_t, ndim=1] min):
        """
        sets the minimum for a query bounding box to use
        """
        self.thisptr.setBBXMin(defs.point3d(min[0], min[1], min[2]))

    def setResolution(self, double r):
        """
        Change the resolution of the octree, scaling all voxels. This will not preserve the (metric) scale!
        """
        self.thisptr.setResolution(r)

    def size(self):
        return self.thisptr.size()

    def toMaxLikelihood(self):
        """
        Creates the maximum likelihood map by calling toMaxLikelihood on all tree nodes,
        setting their occupancy to the corresponding occupancy thresholds.
        """
        self.thisptr.toMaxLikelihood()

    def updateNodes(self, values, update, lazy_eval=False):
        """
        Integrate occupancy measurements and Manipulate log_odds value of voxel directly. 
        """
        if values is None or len(values) == 0:
            return
        if isinstance(values[0], OcTreeKey):
            if isinstance(update, bool):
                for v in values:
                    self.thisptr.updateNode(defs.OcTreeKey(<unsigned short int>v[0],
                                                           <unsigned short int>v[1],
                                                           <unsigned short int>v[2]),
                                            <cppbool>update,
                                            <cppbool?>lazy_eval)
            else:
                for v in values:
                    self.thisptr.updateNode(defs.OcTreeKey(<unsigned short int>v[0],
                                                           <unsigned short int>v[1],
                                                           <unsigned short int>v[2]),
                                            <float?>update,
                                            <cppbool?>lazy_eval)
        else:
            if isinstance(update, bool):
                for v in values:
                    self.thisptr.updateNode(<double?>v[0],
                                            <double?>v[1],
                                            <double?>v[2],
                                            <cppbool>update,
                                            <cppbool?>lazy_eval)
            else:
                for v in values:
                    self.thisptr.updateNode(<double?>v[0],
                                            <double?>v[1],
                                            <double?>v[2],
                                            <float?>update,
                                            <cppbool?>lazy_eval)

    def updateNode(self, value, update, lazy_eval=False):
        """
        Integrate occupancy measurement and Manipulate log_odds value of voxel directly. 
        """
        node = OcTreeNode()
        if isinstance(value, OcTreeKey):
            if isinstance(update, bool):
                node.thisptr = self.thisptr.updateNode(defs.OcTreeKey(<unsigned short int>value[0],
                                                                      <unsigned short int>value[1],
                                                                      <unsigned short int>value[2]),
                                                       <cppbool>update,
                                                       <cppbool?>lazy_eval)
            else:
                node.thisptr = self.thisptr.updateNode(defs.OcTreeKey(<unsigned short int>value[0],
                                                                      <unsigned short int>value[1],
                                                                      <unsigned short int>value[2]),
                                                       <float?>update,
                                                       <cppbool?>lazy_eval)
        else:
            if isinstance(update, bool):
                node.thisptr = self.thisptr.updateNode(<double?>value[0],
                                                       <double?>value[1],
                                                       <double?>value[2],
                                                       <cppbool>update,
                                                       <cppbool?>lazy_eval)
            else:
                node.thisptr = self.thisptr.updateNode(<double?>value[0],
                                                       <double?>value[1],
                                                       <double?>value[2],
                                                       <float?>update,
                                                       <cppbool?>lazy_eval)
        return node

    def updateInnerOccupancy(self):
        """
        Updates the occupancy of all inner nodes to reflect their children's occupancy.
        """
        self.thisptr.updateInnerOccupancy()

    def useBBXLimit(self, enable):
        """
        use or ignore BBX limit (default: ignore)
        """
        self.thisptr.useBBXLimit(bool(enable))

    def volume(self):
        return self.thisptr.volume()

    def getClampingThresMax(self):
        return self.thisptr.getClampingThresMax()

    def getClampingThresMaxLog(self):
        return self.thisptr.getClampingThresMaxLog()

    def getClampingThresMin(self):
        return self.thisptr.getClampingThresMin()

    def getClampingThresMinLog(self):
        return self.thisptr.getClampingThresMinLog()

    def getOccupancyThres(self):
        return self.thisptr.getOccupancyThres()

    def getOccupancyThresLog(self):
        return self.thisptr.getOccupancyThresLog()

    def getProbHit(self):
        return self.thisptr.getProbHit()

    def getProbHitLog(self):
        return self.thisptr.getProbHitLog()

    def getProbMiss(self):
        return self.thisptr.getProbMiss()

    def getProbMissLog(self):
        return self.thisptr.getProbMissLog()

    def setClampingThresMax(self, double thresProb):
        self.thisptr.setClampingThresMax(thresProb)

    def setClampingThresMin(self, double thresProb):
        self.thisptr.setClampingThresMin(thresProb)

    def setOccupancyThres(self, double prob):
        self.thisptr.setOccupancyThres(prob)

    def setProbHit(self, double prob):
        self.thisptr.setProbHit(prob)

    def setProbMiss(self, double prob):
        self.thisptr.setProbMiss(prob)

    def getMetricSize(self):
        cdef double x = 0
        cdef double y = 0
        cdef double z = 0
        self.thisptr.getMetricSize(x, y, z)
        return np.array([x, y, z], dtype=float)

    def getMetricMin(self):
        cdef double x = 0
        cdef double y = 0
        cdef double z = 0
        self.thisptr.getMetricMin(x, y, z)
        return np.array([x, y, z], dtype=float)

    def getMetricMax(self):
        cdef double x = 0
        cdef double y = 0
        cdef double z = 0
        self.thisptr.getMetricMax(x, y, z)
        return np.array([x, y, z], dtype=float)

    def expandNode(self, node):
        self.thisptr.expandNode((<OcTreeNode>node).thisptr)

    def createNodeChild(self, node, int idx):
        child = OcTreeNode()
        child.thisptr = self.thisptr.createNodeChild((<OcTreeNode>node).thisptr, idx)
        return child

    def getNodeChild(self, node, int idx):
        child = OcTreeNode()
        child.thisptr = self.thisptr.getNodeChild((<OcTreeNode>node).thisptr, idx)
        return child

    def isNodeCollapsible(self, node):
        return self.thisptr.isNodeCollapsible((<OcTreeNode>node).thisptr)

    def deleteNodeChild(self, node, int idx):
        self.thisptr.deleteNodeChild((<OcTreeNode>node).thisptr, idx)

    def pruneNode(self, node):
        return self.thisptr.pruneNode((<OcTreeNode>node).thisptr)
    
    def nodeHasChildren(self, node):
        """
        Check if a node has children (recommended replacement for node.hasChildren()).
        
        Args:
            node: OcTreeNode to check
            
        Returns:
            bool: True if node has children, False otherwise
        """
        if isinstance(node, OcTreeNode):
            if (<OcTreeNode>node).thisptr:
                return self.thisptr.nodeHasChildren((<OcTreeNode>node).thisptr)
            else:
                raise NullPointerException
        else:
            raise TypeError("Expected OcTreeNode")
    
    def dynamicEDT_generate(self, maxdist,
                            np.ndarray[DOUBLE_t, ndim=1] bbx_min,
                            np.ndarray[DOUBLE_t, ndim=1] bbx_max,
                            treatUnknownAsOccupied=False):
        # Clean up existing DynamicEDT if it exists
        if self.edtptr != NULL:
            del self.edtptr
            self.edtptr = NULL
        
        self.edtptr = new edt.DynamicEDTOctomap(<float?>maxdist,
                                                self.thisptr,
                                                defs.point3d(bbx_min[0], bbx_min[1], bbx_min[2]),
                                                defs.point3d(bbx_max[0], bbx_max[1], bbx_max[2]),
                                                <cppbool?>treatUnknownAsOccupied)

    def dynamicEDT_checkConsistency(self):
        if self.edtptr:
            return self.edtptr.checkConsistency()
        else:
            raise NullPointerException

    def dynamicEDT_update(self, updateRealDist):
        if self.edtptr:
            self.edtptr.update(<cppbool?>updateRealDist)
        else:
            raise NullPointerException

    def dynamicEDT_getMaxDist(self):
        if self.edtptr:
            return self.edtptr.getMaxDist()
        else:
            raise NullPointerException

    def dynamicEDT_getDistance(self, p):
        if self.edtptr:
            if isinstance(p, OcTreeKey):
                return self.edtptr.getDistance(edt.OcTreeKey(<unsigned short int>p[0],
                                                             <unsigned short int>p[1],
                                                             <unsigned short int>p[2]))
            else:
                return self.edtptr.getDistance(edt.point3d(<float?>p[0],
                                                           <float?>p[1],
                                                           <float?>p[2]))
        else:
            raise NullPointerException