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
    cdef defs.OcTreeKey thisptr
    def __cinit__(self, unsigned short int a=0, unsigned short int b=0, unsigned short int c=0):
        self.thisptr.k[0] = a
        self.thisptr.k[1] = b
        self.thisptr.k[2] = c
    def __richcmp__(self, other, int op):
        if op == 2:
            return (self.thisptr.k[0] == other[0] and \
                    self.thisptr.k[1] == other[1] and \
                    self.thisptr.k[2] == other[2])
        elif op == 3:
            return not (self.thisptr.k[0] == other[0] and \
                        self.thisptr.k[1] == other[1] and \
                        self.thisptr.k[2] == other[2])
    
    def __getitem__(self, unsigned int i):
        return self.thisptr.k[i]
    def __setitem__(self, unsigned int i, unsigned int value):
        self.thisptr.k[i] = value
            
    def __repr__(self):
        return f"OcTreeKey({self.thisptr.k[0]}, {self.thisptr.k[1]}, {self.thisptr.k[2]})"
            
    def computeChildIdx(self, OcTreeKey key, int depth):
        cdef unsigned int result
        cdef defs.OcTreeKey key_in
        key_in.k[0] = key[0]
        key_in.k[1] = key[1]
        key_in.k[2] = key[2]
        result = defs.computeChildIdx(key_in, depth)
        return result
    def computeIndexKey(self, unsigned int level, OcTreeKey key):
        cdef defs.OcTreeKey key_in
        cdef defs.OcTreeKey result
        key_in.k[0] = key[0]
        key_in.k[1] = key[1]
        key_in.k[2] = key[2]
        result = defs.computeIndexKey(level, key_in)
        # Convert back to Python OcTreeKey
        return OcTreeKey(result.k[0], result.k[1], result.k[2])

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
        key_in.k[0] = key[0]
        key_in.k[1] = key[1]
        key_in.k[2] = key[2]
        cdef defs.OcTreeKey key_out = self.thisptr.adjustKeyAtDepth(key_in, <int?>depth)
        res = OcTreeKey()
        res.thisptr.k[0] = key_out.k[0]
        res.thisptr.k[1] = key_out.k[1]
        res.thisptr.k[2] = key_out.k[2]
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
        res.thisptr.k[0] = key.k[0]
        res.thisptr.k[1] = key.k[1]
        res.thisptr.k[2] = key.k[2]
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
            res.thisptr.k[0] = key.k[0]
            res.thisptr.k[1] = key.k[1]
            res.thisptr.k[2] = key.k[2]
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
        cdef defs.point3d search_point
        cdef defs.OcTreeNode* found_node
        
        if isinstance(node, OcTreeNode):
            if (<OcTreeNode>node).thisptr:
                return self.thisptr.isNodeOccupied(deref((<OcTreeNode>node).thisptr))
            else:
                raise NullPointerException
        elif isinstance(node, (SimpleTreeIterator, SimpleLeafIterator, SimpleLeafBBXIterator)):
            # Handle iterator case - use coordinate to search for the node
            try:
                coord = node.getCoordinate()
                # Convert coordinate to point3d for search
                search_point = defs.point3d(coord[0], coord[1], coord[2])
                found_node = self.thisptr.search(<double>coord[0], <double>coord[1], <double>coord[2], <unsigned int>0)
                if found_node != NULL:
                    result = self.thisptr.isNodeOccupied(deref(found_node))
                    return result
                else:
                    return False
            except Exception:
                return False
        else:
            raise TypeError(f"Expected OcTreeNode or iterator, got {type(node)}")

    def isNodeAtThreshold(self, node):
        cdef defs.point3d search_point
        cdef defs.OcTreeNode* found_node
        
        if isinstance(node, OcTreeNode):
            if (<OcTreeNode>node).thisptr:
                return self.thisptr.isNodeAtThreshold(deref((<OcTreeNode>node).thisptr))
            else:
                raise NullPointerException
        elif isinstance(node, (SimpleTreeIterator, SimpleLeafIterator, SimpleLeafBBXIterator)):
            # Handle iterator case - use coordinate to search for the node
            try:
                coord = node.getCoordinate()
                # Convert coordinate to point3d for search
                search_point = defs.point3d(coord[0], coord[1], coord[2])
                found_node = self.thisptr.search(<double>coord[0], <double>coord[1], <double>coord[2], <unsigned int>0)
                if found_node != NULL:
                    return self.thisptr.isNodeAtThreshold(deref(found_node))
                else:
                    return False
            except Exception:
                return False
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
        key_in.k[0] = key[0]
        key_in.k[1] = key[1]
        key_in.k[2] = key[2]
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
        cdef defs.OcTreeKey search_key
        node = OcTreeNode()
        if isinstance(value, OcTreeKey):
            search_key.k[0] = value[0]
            search_key.k[1] = value[1]
            search_key.k[2] = value[2]
            node.thisptr = self.thisptr.search(search_key,
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
        cdef defs.OcTreeKey update_key
        if values is None or len(values) == 0:
            return
        if isinstance(values[0], OcTreeKey):
            if isinstance(update, bool):
                for v in values:
                    update_key.k[0] = v[0]
                    update_key.k[1] = v[1]
                    update_key.k[2] = v[2]
                    self.thisptr.updateNode(update_key,
                                            <cppbool>update,
                                            <cppbool?>lazy_eval)
            else:
                for v in values:
                    update_key.k[0] = v[0]
                    update_key.k[1] = v[1]
                    update_key.k[2] = v[2]
                    self.thisptr.updateNode(update_key,
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
        cdef defs.OcTreeKey update_key # Moved to top
        """
        Integrate occupancy measurement and Manipulate log_odds value of voxel directly. 
        """
        node = OcTreeNode()
        if isinstance(value, OcTreeKey):
            if isinstance(update, bool):
                update_key.k[0] = value[0]
                update_key.k[1] = value[1]
                update_key.k[2] = value[2]
                node.thisptr = self.thisptr.updateNode(update_key,
                                                       <cppbool>update,
                                                       <cppbool?>lazy_eval)
            else:
                update_key.k[0] = value[0]
                update_key.k[1] = value[1]
                update_key.k[2] = value[2]
                node.thisptr = self.thisptr.updateNode(update_key,
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

    def addPointWithRayCasting(self, 
                              np.ndarray[DOUBLE_t, ndim=1] point,
                              np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                              update_inner_occupancy=False):
        """
        Add a single 3D point to update the occupancy grid using ray casting.
        
        This method efficiently adds a point by:
        1. Casting a ray from sensor_origin to the target point
        2. If the ray hits an obstacle, marking the hit point as occupied
        3. If no hit, marking the target point as occupied
        4. Marking free space along the ray from origin to the occupied point
        
        Args:
            point: 3D point [x, y, z] in meters
            sensor_origin: Sensor origin [x, y, z] in meters
            update_inner_occupancy: Whether to update inner node occupancy (expensive)
            
        Returns:
            bool: True if point was successfully added
        """
        cdef cppbool success = self._add_single_point_optimized(point, sensor_origin)
        
        if success and update_inner_occupancy:
            self.updateInnerOccupancy()
        
        return success

    def markFreeSpaceAlongRay(self, 
                             np.ndarray[DOUBLE_t, ndim=1] origin, 
                             np.ndarray[DOUBLE_t, ndim=1] end_point, 
                             step_size=None):
        """
        Mark free space along a ray from origin to end_point using manual sampling.
        
        Args:
            origin: Ray start point [x, y, z]
            end_point: Ray end point [x, y, z]
            step_size: Step size for ray sampling (defaults to tree resolution)
        """
        if step_size is not None and step_size != self.getResolution():
            # Use custom step size - fall back to original implementation
            resolution = self.getResolution()
            step = step_size
            
            # Calculate ray direction and length
            direction = end_point - origin
            ray_length = np.linalg.norm(direction)
            
            if ray_length == 0:
                return
                
            direction = direction / ray_length
            
            # Sample points along the ray
            num_steps = int(ray_length / step) + 1
            
            for i in range(1, num_steps):  # Skip origin (i=0)
                t = (i * step) / ray_length
                if t >= 1.0:
                    break
                    
                sample_point = origin + t * direction
                self.updateNode(sample_point, False)  # Mark as free
        else:
            # Use optimized version with default resolution
            self._mark_free_space_optimized(origin, end_point)

    cdef cppbool _add_single_point_optimized(self, np.ndarray[DOUBLE_t, ndim=1] point, 
                                            np.ndarray[DOUBLE_t, ndim=1] sensor_origin):
        """Optimized single point addition with minimal overhead"""
        cdef np.ndarray[DOUBLE_t, ndim=1] direction
        cdef np.ndarray[DOUBLE_t, ndim=1] end_point
        cdef double ray_length
        cdef cppbool hit
        
        try:
            # Check if origin and point are the same
            if (point[0] == sensor_origin[0] and 
                point[1] == sensor_origin[1] and 
                point[2] == sensor_origin[2]):
                self.updateNode(point, True)
                return True
            
            # Calculate direction vector
            direction = point - sensor_origin
            ray_length = np.linalg.norm(direction)
            
            if ray_length > 0:
                # Normalize direction
                direction = direction / ray_length
                
                # Use castRay to find the first occupied cell along the ray
                end_point = np.zeros(3, dtype=np.float64)
                hit = self.castRay(sensor_origin, direction, end_point, 
                                  ignoreUnknownCells=True, 
                                  maxRange=ray_length)
                
                if hit:
                    # Ray hit an obstacle - mark the hit point as occupied
                    self.updateNode(end_point, True)
                    # Mark free space from origin to hit point
                    self._mark_free_space_optimized(sensor_origin, end_point)
                else:
                    # No hit - mark the target point as occupied
                    self.updateNode(point, True)
                    # Mark free space from origin to target point
                    self._mark_free_space_optimized(sensor_origin, point)
            else:
                # Zero-length ray - just mark the point as occupied
                self.updateNode(point, True)
            
            return True
            
        except Exception:
            return False

    cdef void _mark_free_space_optimized(self, np.ndarray[DOUBLE_t, ndim=1] origin, 
                                        np.ndarray[DOUBLE_t, ndim=1] end_point):
        """Optimized free space marking with pre-calculated step size"""
        cdef double resolution = self.getResolution()
        cdef np.ndarray[DOUBLE_t, ndim=1] direction
        cdef double ray_length
        
        direction = end_point - origin
        ray_length = np.linalg.norm(direction)
        
        if ray_length == 0:
            return
            
        direction = direction / ray_length
        
        # Sample points along the ray
        cdef int num_steps = int(ray_length / resolution) + 1
        cdef int i
        cdef double t
        cdef np.ndarray[DOUBLE_t, ndim=1] sample_point
        
        for i in range(1, num_steps):  # Skip origin (i=0)
            t = (i * resolution) / ray_length
            if t >= 1.0:
                break
                
            sample_point = origin + t * direction
            self.updateNode(sample_point, False)  # Mark as free

    def addPointCloudWithRayCasting(self,
                                   np.ndarray[DOUBLE_t, ndim=2] point_cloud,
                                   np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                                   max_range=-1.0,
                                   update_inner_occupancy=True,
                                   discretize=False):
        """
        Add a full point cloud using ray casting for each point.
        
        This method provides more accurate free space marking compared to 
        the standard insertPointCloud method by using ray casting for each point.
        
        Args:
            point_cloud: Nx3 array of points
            sensor_origin: Sensor origin for the point cloud
            max_range: Maximum range for points (-1 = no limit)
            update_inner_occupancy: Whether to update inner node occupancy
            discretize: If True, discretize to unique keys first to reduce rays (faster for dense clouds)
        
        Returns:
            int: Number of points successfully added
        """
        cdef int success_count = 0
        cdef int i
        cdef int num_points = point_cloud.shape[0]
        cdef np.ndarray[DOUBLE_t, ndim=1] point
        cdef np.ndarray[DOUBLE_t, ndim=2] filtered_points
        cdef cppbool success
        
        try:
            # Discretize if requested (reduces N for dense clouds)
            if discretize:
                point_cloud = self._discretizePointCloud(point_cloud)
                num_points = point_cloud.shape[0]
            
            if max_range > 0:
                # Filter points by range first - vectorized approach
                distances = np.linalg.norm(point_cloud - sensor_origin, axis=1)
                filtered_points = point_cloud[distances <= max_range]
                success_count = self._process_points_vectorized(filtered_points, sensor_origin, filtered_points.shape[0])
            else:
                # Process all points without range filtering
                success_count = self._process_points_vectorized(point_cloud, sensor_origin, num_points)
            
            # Update inner occupancy once for the batch
            if update_inner_occupancy and success_count > 0:
                self.updateInnerOccupancy()
            
            return success_count
            
        except Exception as e:
            print(f"Error in point cloud processing: {e}")
            return success_count

    cdef int _process_points_vectorized(self, np.ndarray[DOUBLE_t, ndim=2] points, 
                                        np.ndarray[DOUBLE_t, ndim=1] origin, 
                                        int num_points):
        """Optimized vectorized processing for points with same origin"""
        cdef int success_count = 0
        cdef int i
        cdef cppbool hit
        cdef double ray_length
        cdef np.ndarray[DOUBLE_t, ndim=1] end_point_py
        
        # Vectorized pre-computation (NumPy ops auto-manage GIL)
        cdef np.ndarray[DOUBLE_t, ndim=2] directions
        cdef np.ndarray[DOUBLE_t, ndim=1] distances
        cdef np.ndarray[DOUBLE_t, ndim=2] valid_points  # Filtered points
        cdef int valid_num_points
        
        directions = points - origin  # (N, 3) vectorized subtract
        distances = np.linalg.norm(directions, axis=1)  # O(N) norms
        
        # Handle zero-distance points separately (small loop, rare/edge case)
        zero_mask = distances == 0.0
        if np.any(zero_mask):
            for j in range(num_points):
                if zero_mask[j]:
                    self.updateNode(points[j], True)  # Mark occupied
                    success_count += 1
        
        # Filter non-zero points
        non_zero = distances > 0.0
        valid_num_points = np.sum(non_zero)
        if valid_num_points == 0:
            return success_count
        
        valid_points = points[non_zero]
        directions = directions[non_zero] / distances[non_zero, np.newaxis]  # Normalize
        distances = distances[non_zero]  # Filtered distances
        
        # Pre-allocate end_point_py once
        end_point_py = np.zeros(3, dtype=np.float64)
        
        # Main loop: C-style optimized, auto GIL for method calls
        for i in range(valid_num_points):
            ray_length = distances[i]  # Pre-computed scalar (fast typed access)
            
            # Cast ray and updates (Cython auto-acquires GIL for these Python-bound calls)
            hit = self.castRay(origin, directions[i], end_point_py, 
                               ignoreUnknownCells=True, maxRange=ray_length)
            
            if hit:
                self.updateNode(end_point_py, True)
                self._mark_free_space_optimized(origin, end_point_py)
            else:
                self.updateNode(valid_points[i], True)
                self._mark_free_space_optimized(origin, valid_points[i])
            
            success_count += 1  # Assume success (add checks if needed)
        
        return success_count

    def _discretizePointCloud(self, np.ndarray[DOUBLE_t, ndim=2] point_cloud, bint checked=True):
        """
        Discretize points to unique octree keys (reduces duplicates for batching).
        Internal helper for faster insertion.
        """
        cdef int i, num_points = point_cloud.shape[0]
        cdef np.ndarray[DOUBLE_t, ndim=1] point
        cdef set unique_keys = set()
        cdef list discrete_points = []
        cdef OcTreeKey key
        
        for i in range(num_points):
            point = point_cloud[i]
            if checked:
                key = self.coordToKeyChecked(point)[1]  # Returns key if in bounds
                if key is not None:
                    key_tuple = (key[0], key[1], key[2])
                    if key_tuple not in unique_keys:
                        unique_keys.add(key_tuple)
                        discrete_points.append(point)
            else:
                key = self.coordToKey(point)
                key_tuple = (key[0], key[1], key[2])
                if key_tuple not in unique_keys:
                    unique_keys.add(key_tuple)
                    discrete_points.append(point)
        
        return np.array(discrete_points, dtype=np.float64)

    cdef void _build_pointcloud_and_insert(self, np.ndarray[DOUBLE_t, ndim=2] point_cloud,
                                      np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                                      double max_range,
                                      bint discretize,
                                      bint lazy_eval):
        """Shared internal: Build Pointcloud, optional discretize, insert via C++."""
        cdef int i, num_points = point_cloud.shape[0]
        cdef np.ndarray[DOUBLE_t, ndim=1] point
        cdef defs.Pointcloud pc = defs.Pointcloud()
        cdef cppbool success = True
        
        # Discretize if requested (reduces N)
        if discretize:
            point_cloud = self._discretizePointCloud(point_cloud)
            num_points = point_cloud.shape[0]
        
        # Build C++ Pointcloud
        for i in range(num_points):
            point = point_cloud[i]
            pc.push_back(<float>point[0], <float>point[1], <float>point[2])
        
        # Call native batch
        self.thisptr.insertPointCloud(pc,
                                      defs.point3d(<float>sensor_origin[0],
                                                   <float>sensor_origin[1],
                                                   <float>sensor_origin[2]),
                                      <double>max_range,
                                      <cppbool>lazy_eval,
                                      <cppbool>discretize)
        
        if not lazy_eval:
            self.updateInnerOccupancy()
        
        # No return; assume success

    def insertPointCloudFast(self,
                         np.ndarray[DOUBLE_t, ndim=2] point_cloud,
                         np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                         double max_range=-1.0,
                         bint discretize=False,
                         bint lazy_eval=False):
        """
        Fast batch insertion using native C++ (parallelized with OpenMP, batched updates).
        
        Marks full rays from origin to endpoints as free, endpoints as occupied.
        Less accurate than ray-casting (doesn't stop at hits) but much faster.
        Uses discretization if enabled (groups points to reduce rays ~50% for dense clouds).
        
        Args:
            point_cloud: Nx3 array of points
            sensor_origin: Sensor origin [x, y, z]
            max_range: Max range per ray (-1 = unlimited)
            discretize: If True, discretize points to keys first (faster for dense clouds)
            lazy_eval: If True, defer updateInnerOccupancy (call manually later)
        
        Returns:
            int: Number of points processed
        """
        cdef int num_points = point_cloud.shape[0]
        cdef cppbool success = True
        
        self._build_pointcloud_and_insert(point_cloud, sensor_origin, max_range, discretize, lazy_eval)
        
        return num_points if success else 0

    def insertPointCloud(self,
                     np.ndarray[DOUBLE_t, ndim=2] point_cloud,
                     np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                     double max_range=-1.0,
                     bint lazy_eval=False,
                     bint discretize=False):
        """
        Original native C++ batch insertion (full rays, no Python-specific opts beyond params).
        
        Equivalent to insertPointCloudFast with wrapper logic shared.
        
        Args:
            point_cloud: Nx3 array of points
            sensor_origin: Sensor origin [x, y, z]
            max_range: Max range per ray (-1 = unlimited)
            lazy_eval: If True, defer updateInnerOccupancy (call manually later)
            discretize: If True, discretize points to keys first
        
        Returns:
            int: Number of points processed
        """
        cdef int num_points = point_cloud.shape[0]
        cdef cppbool success = True
        
        self._build_pointcloud_and_insert(point_cloud, sensor_origin, max_range, discretize, lazy_eval)
        
        return num_points if success else 0

    def insertPointCloudRaysFast(self,
                                np.ndarray[DOUBLE_t, ndim=2] point_cloud,
                                np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                                double max_range=-1.0,
                                bint lazy_eval=False):
        """
        Ultra-fast batch using native insertPointCloudRays (parallel rays, no key sets).
        Inserts full rays without deduplication—fastest but may over-update.
        """
        cdef defs.Pointcloud pc
        cdef int i, num_points = point_cloud.shape[0]
        cdef np.ndarray[DOUBLE_t, ndim=1] point
        cdef defs.point3d origin_c  # C++ type declaration
        
        pc = defs.Pointcloud()  # C++ constructor
        
        for i in range(num_points):
            point = point_cloud[i]
            pc.push_back(<float>point[0], <float>point[1], <float>point[2])
        
        # Create C++ origin without Python conversion
        origin_c = defs.point3d(<float>sensor_origin[0], <float>sensor_origin[1], <float>sensor_origin[2])
        
        self.thisptr.insertPointCloudRays(pc, origin_c, <double>max_range, <cppbool>lazy_eval)
        
        if not lazy_eval:
            self.updateInnerOccupancy()
        
        return num_points