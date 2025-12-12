# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libcpp.string cimport string
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from cython.operator cimport dereference as deref, preincrement as inc
cimport octomap_defs as defs
import numpy as np
cimport numpy as np
# Note: DOUBLE_t is declared in octree.pxd, not here

# Fix NumPy API compatibility
np.import_array()

# NumPy compatibility for newer versions
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].tree_iterator* tree_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_iterator* leaf_iterator_ptr
ctypedef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator* leaf_bbx_iterator_ptr

# Import base classes and iterators
from .octree_base import OcTreeKey, OcTreeNode, NullPointerException
from .octree_iterators import SimpleTreeIterator, SimpleLeafIterator, SimpleLeafBBXIterator
# Also cimport for type casting - needed for OcTreeKey.thisptr access
cimport pyoctomap.octree_base as octree_base

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
    # Note: thisptr, edtptr, and owner are declared in octree.pxd, not here
    
    # Helper method to get the C++ pointer address (for use in other modules)
    # Returns pointer as size_t so it can be called from Python code
    cpdef size_t _get_ptr_addr(self):
        return <size_t>self.thisptr
    
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

    def adjustKeyAtDepth(self, key, depth):
        cdef defs.OcTreeKey key_in
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
        # Create OcTreeKey using Cython type directly, then convert to Python object
        cdef octree_base.OcTreeKey res_cython = octree_base.OcTreeKey(key.k[0], key.k[1], key.k[2])
        # Convert to Python object by creating new one with same values
        res = OcTreeKey(res_cython[0], res_cython[1], res_cython[2])
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
    cdef void _fast_decay_in_bbx(self, 
                                  defs.point3d bbx_min, 
                                  defs.point3d bbx_max, 
                                  float logodd_decay_value):
        """
        (Internal C++-level worker)
        Iterates over all leaves in a BBX at C++ speed
        and applies a decay value.
        """
        
        # 1. Get the raw C++ iterators
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator it
        cdef defs.OccupancyOcTreeBase[defs.OcTreeNode].leaf_bbx_iterator end_it
        
        # Get all leaves (maxDepth = 0)
        it = self.thisptr.begin_leafs_bbx(bbx_min, bbx_max, 0) 
        end_it = self.thisptr.end_leafs_bbx()

        # 2. C-style loop (this is now extremely fast)
        while it != end_it:
            # 3. C-style checks and updates
            # We use deref() to get the OcTreeNode&
            if self.thisptr.isNodeOccupied(deref(it)):
                # Directly call the C++ addValue
                deref(it).addValue(logodd_decay_value)

            # 4. Advance iterator using the imported 'inc'
            inc(it)

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
        cdef defs.OcTreeNode* node_ptr
        cdef size_t ptr_addr
        if isinstance(node, OcTreeNode):
            # Access pointer through helper method that returns address
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.OcTreeNode*>ptr_addr
            if node_ptr != NULL:
                return self.thisptr.isNodeOccupied(deref(node_ptr))
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
        cdef defs.OcTreeNode* node_ptr
        cdef size_t ptr_addr
        if isinstance(node, OcTreeNode):
            # Access pointer through helper method that returns address
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.OcTreeNode*>ptr_addr
            if node_ptr != NULL:
                return self.thisptr.isNodeAtThreshold(deref(node_ptr))
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
        cdef object key_obj
        cdef object node_obj
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
        cdef object it
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
        itr._set_end()
        return itr

    def end_leafs(self):
        """Return an end iterator for leaf traversal"""
        itr = SimpleLeafIterator(self)
        itr._set_end()
        return itr

    def end_leafs_bbx(self):
        """Return an end iterator for leaf bounding box traversal"""
        itr = SimpleLeafBBXIterator(self, np.array([0.0, 0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0, 1.0], dtype=np.float64))
        itr._set_end()
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
        cdef defs.OcTreeNode* root_ptr = self.thisptr.getRoot()
        node = OcTreeNode()
        node._set_ptr(<size_t>root_ptr)
        return node

    def getNumLeafNodes(self):
        return self.thisptr.getNumLeafNodes()

    def getResolution(self):
        return self.thisptr.getResolution()

    def getTreeDepth(self):
        return self.thisptr.getTreeDepth()

    def getTreeType(self):
        return self.thisptr.getTreeType().c_str().decode('utf-8')

    def inBBX(self, np.ndarray[DOUBLE_t, ndim=1] p):
        return self.thisptr.inBBX(defs.point3d(p[0], p[1], p[2]))

    def keyToCoord(self, key, depth=None):
        cdef defs.OcTreeKey key_in
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
        cdef defs.OcTreeNode* node_ptr
        node = OcTreeNode()
        if isinstance(value, OcTreeKey):
            search_key.k[0] = value[0]
            search_key.k[1] = value[1]
            search_key.k[2] = value[2]
            node_ptr = self.thisptr.search(search_key,
                                          <unsigned int?>depth)
        else:
            node_ptr = self.thisptr.search(<double>value[0],
                                           <double>value[1],
                                           <double>value[2],
                                           <unsigned int?>depth)
        # Return None if the search failed (node_ptr is NULL)
        if node_ptr == NULL:
            return None
        node._set_ptr(<size_t>node_ptr)
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
        cdef defs.OcTreeKey update_key
        cdef defs.OcTreeNode* node_ptr
        """
        Integrate occupancy measurement and Manipulate log_odds value of voxel directly. 
        """
        node = OcTreeNode()
        if isinstance(value, OcTreeKey):
            if isinstance(update, bool):
                update_key.k[0] = value[0]
                update_key.k[1] = value[1]
                update_key.k[2] = value[2]
                node_ptr = self.thisptr.updateNode(update_key,
                                                   <cppbool>update,
                                                   <cppbool?>lazy_eval)
            else:
                update_key.k[0] = value[0]
                update_key.k[1] = value[1]
                update_key.k[2] = value[2]
                node_ptr = self.thisptr.updateNode(update_key,
                                                   <float?>update,
                                                   <cppbool?>lazy_eval)
        else:
            if isinstance(update, bool):
                node_ptr = self.thisptr.updateNode(<double?>value[0],
                                                   <double?>value[1],
                                                   <double?>value[2],
                                                   <cppbool>update,
                                                   <cppbool?>lazy_eval)
            else:
                node_ptr = self.thisptr.updateNode(<double?>value[0],
                                                   <double?>value[1],
                                                   <double?>value[2],
                                                   <float?>update,
                                                   <cppbool?>lazy_eval)
        node._set_ptr(<size_t>node_ptr)
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
        cdef defs.OcTreeNode* node_ptr
        cdef size_t ptr_addr
        if isinstance(node, OcTreeNode):
            # Access pointer through helper method that returns address
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.OcTreeNode*>ptr_addr
            if node_ptr != NULL:
                self.thisptr.expandNode(node_ptr)
            else:
                raise NullPointerException
        else:
            raise TypeError("Expected OcTreeNode")

    def createNodeChild(self, node, int idx):
        cdef defs.OcTreeNode* node_ptr
        cdef defs.OcTreeNode* child_ptr
        cdef size_t ptr_addr
        if isinstance(node, OcTreeNode):
            # Access pointer through helper method that returns address
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.OcTreeNode*>ptr_addr
            if node_ptr != NULL:
                child_ptr = self.thisptr.createNodeChild(node_ptr, idx)
                child = OcTreeNode()
                child._set_ptr(<size_t>child_ptr)
                return child
            else:
                raise NullPointerException
        else:
            raise TypeError("Expected OcTreeNode")

    def getNodeChild(self, node, int idx):
        cdef defs.OcTreeNode* node_ptr
        cdef defs.OcTreeNode* child_ptr
        cdef size_t ptr_addr
        if isinstance(node, OcTreeNode):
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.OcTreeNode*>ptr_addr
            if node_ptr == NULL:
                raise NullPointerException
            child_ptr = self.thisptr.getNodeChild(node_ptr, idx)
            child = OcTreeNode()
            child._set_ptr(<size_t>child_ptr)
            return child
        else:
            raise TypeError("Expected OcTreeNode")

    def isNodeCollapsible(self, node):
        cdef defs.OcTreeNode* node_ptr
        cdef size_t ptr_addr
        if isinstance(node, OcTreeNode):
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.OcTreeNode*>ptr_addr
            if node_ptr == NULL:
                raise NullPointerException
            return self.thisptr.isNodeCollapsible(node_ptr)
        else:
            raise TypeError("Expected OcTreeNode")

    def deleteNodeChild(self, node, int idx):
        cdef defs.OcTreeNode* node_ptr
        cdef size_t ptr_addr
        if isinstance(node, OcTreeNode):
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.OcTreeNode*>ptr_addr
            if node_ptr == NULL:
                raise NullPointerException
            self.thisptr.deleteNodeChild(node_ptr, idx)
        else:
            raise TypeError("Expected OcTreeNode")

    def pruneNode(self, node):
        cdef defs.OcTreeNode* node_ptr
        cdef size_t ptr_addr
        if isinstance(node, OcTreeNode):
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.OcTreeNode*>ptr_addr
            if node_ptr == NULL:
                raise NullPointerException
            return self.thisptr.pruneNode(node_ptr)
        else:
            raise TypeError("Expected OcTreeNode")
    
    def nodeHasChildren(self, node):
        """
        Check if a node has children (recommended replacement for node.hasChildren()).
        
        Args:
            node: OcTreeNode to check
            
        Returns:
            bool: True if node has children, False otherwise
        """
        cdef defs.OcTreeNode* node_ptr
        cdef size_t ptr_addr
        if isinstance(node, OcTreeNode):
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.OcTreeNode*>ptr_addr
            if node_ptr == NULL:
                raise NullPointerException
            return self.thisptr.nodeHasChildren(node_ptr)
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
        1. Clearing the entire ray as free space first (removes old occupied voxels)
        2. Casting a ray from sensor_origin to the target point
        3. If the ray hits an obstacle, marking the hit point as occupied
        4. If no hit, marking the target point as occupied
        
        Args:
            point: 3D point [x, y, z] in meters
            sensor_origin: Sensor origin [x, y, z] in meters
            update_inner_occupancy: Whether to update inner node occupancy (expensive)
            
        Returns:
            bool: True if point was successfully added
        """
        cdef cppbool success = self._add_single_point_optimized(point, sensor_origin, 0.1)
        
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

    #Debugging function to test the ray casting insertion
    cdef cppbool _add_single_point_optimized(self, np.ndarray[DOUBLE_t, ndim=1] point, 
                                            np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                                            double decay_factor):
        """Optimized single point addition with probabilistic decay along ray"""
        cdef np.ndarray[DOUBLE_t, ndim=1] direction
        cdef np.ndarray[DOUBLE_t, ndim=1] end_point
        cdef double ray_length
        cdef cppbool hit
        cdef float logodd_decay
        cdef float logodd_miss
        cdef double resolution
        cdef double hit_distance
        cdef int num_steps
        cdef int i
        cdef double t
        cdef np.ndarray[DOUBLE_t, ndim=1] sample_point
        
        try:
            # Check if origin and point are the same
            if (point[0] == sensor_origin[0] and 
                point[1] == sensor_origin[1] and 
                point[2] == sensor_origin[2]):
                self.updateNode(point, True)
                return True
            
            # Calculate decay values
            logodd_miss = self.thisptr.getProbMissLog()
            logodd_decay = logodd_miss * <float>decay_factor
            
            # Calculate direction vector
            direction = point - sensor_origin
            ray_length = np.linalg.norm(direction)
            
            if ray_length > 0:
                # Normalize direction
                direction = direction / ray_length
                
                # STEP 1: Use castRay to find the first occupied cell along the ray
                end_point = np.zeros(3, dtype=np.float64)
                hit = self.castRay(sensor_origin, direction, end_point, 
                                  ignoreUnknownCells=True, 
                                  maxRange=ray_length)
                
                if hit:
                    # Apply probabilistic decay to voxels along the ray up to the hit point
                    # This maintains the probabilistic nature of OctoMap
                    hit_distance = np.linalg.norm(end_point - sensor_origin)
                    resolution = self.getResolution()
                    num_steps = <int>(hit_distance / resolution) + 1
                    
                    for i in range(1, num_steps):  # Skip origin (i=0)
                        t = i * resolution
                        if t >= hit_distance:
                            break
                        
                        sample_point = sensor_origin + direction * t
                        # Apply decay to voxels along the ray (probabilistic update)
                        self.thisptr.updateNode(sample_point[0], sample_point[1], sample_point[2], 
                                              logodd_decay, <cppbool>True)
                    
                    # Decay the hit voxel as well
                    self.thisptr.updateNode(end_point[0], end_point[1], end_point[2], 
                                          logodd_decay, <cppbool>True)
                else:
                    # No hit - apply decay along the ray and mark target point as occupied
                    resolution = self.getResolution()
                    num_steps = <int>(ray_length / resolution) + 1
                    
                    for i in range(1, num_steps):  # Skip origin (i=0)
                        t = i * resolution
                        if t >= ray_length:
                            break
                        
                        sample_point = sensor_origin + direction * t
                        # Apply decay to voxels along the ray (probabilistic update)
                        self.thisptr.updateNode(sample_point[0], sample_point[1], sample_point[2], 
                                              logodd_decay, <cppbool>True)
                    
                    # Mark the target point as occupied
                    self.updateNode(point, True)
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


    def _discretizePointCloud(self, np.ndarray[DOUBLE_t, ndim=2] point_cloud, bint checked=True):
        """
        Discretize points to unique octree keys (reduces duplicates for batching).
        Internal helper for faster insertion.
        """
        cdef int i, num_points = point_cloud.shape[0]
        cdef np.ndarray[DOUBLE_t, ndim=1] point
        cdef set unique_keys = set()
        cdef list discrete_points = []
        cdef object key
        
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
        
        # Always call updateInnerOccupancy() to ensure tree consistency
        # Even when lazy_eval=False, this ensures all inner nodes are properly updated
        # and the tree is in a consistent state for queries
        if not lazy_eval:
            self.updateInnerOccupancy()
        
        # No return; assume success

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
    def decayOccupancyInBBX(self,
                              np.ndarray[DOUBLE_t, ndim=2] point_cloud,
                              np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                              float logodd_decay_value=-0.2):
        """
        Calculates the bounding box of a scan (points + origin) and
        applies a log-odds decay to all *occupied* leaf nodes within it.

        This function is a high-performance wrapper that calls an
        internal C++ iterator loop.
        """
        
        # --- 1. Bounding Box Calculation (NumPy) ---
        cdef np.ndarray[DOUBLE_t, ndim=1] bbx_min_np, bbx_max_np
        cdef np.ndarray[DOUBLE_t, ndim=1] pc_min = np.min(point_cloud, axis=0)
        cdef np.ndarray[DOUBLE_t, ndim=1] pc_max = np.max(point_cloud, axis=0)
        
        bbx_min_np = np.minimum(pc_min, sensor_origin)
        bbx_max_np = np.maximum(pc_max, sensor_origin)

        # --- 2. Convert to C++ types ---
        cdef defs.point3d bbx_min_cpp = defs.point3d(
            <float>bbx_min_np[0], <float>bbx_min_np[1], <float>bbx_min_np[2])
        cdef defs.point3d bbx_max_cpp = defs.point3d(
            <float>bbx_max_np[0], <float>bbx_max_np[1], <float>bbx_max_np[2])

        # Ensure decay is negative
        if logodd_decay_value > 0.0:
            logodd_decay_value = -logodd_decay_value

        # --- 3. Call the fast C++ worker function ---
        try:
            # This C++ call will be very fast
            self._fast_decay_in_bbx(bbx_min_cpp, bbx_max_cpp, logodd_decay_value)
        except Exception as e:
            # Catch C++ exceptions
            print(f"Warning: Error during fast decay step: {e}")

    def decayAndInsertPointCloud(self,
                                 np.ndarray[DOUBLE_t, ndim=2] point_cloud,
                                 np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                                 float logodd_decay_value=-0.2,
                                 double max_range=-1.0,
                                 bint update_inner_occupancy=True):
        """
        Solves the occluded-ghost problem by first applying a temporal
        decay to the scan's bounding box, and then
        inserting the new point cloud.
        
        This is the recommended function for inserting scans from a
        moving sensor in a dynamic environment.

        Args:
            point_cloud: The new point cloud to insert.
            sensor_origin: The origin of the sensor.
            logodd_decay_value: The negative log-odds to add to all
                occupied voxels in the region *before* inserting
                the new scan. This value *must* be negative.

                **Tuning Guide:**
                This value controls how fast the map "forgets" old data.
                A good way to tune it is to decide how many scans
                it should take for a ghost to disappear.

                A fully occupied voxel typically has a log-odds value
                of around +4.0.

                Formula: Scans_to_Forget ≈ 4.0 / abs(logodd_decay_value)

                - **Moderate (Default): -0.2**
                  Takes ~20 scans for a ghost to fade.
                - **Aggressive: -1.0 to -3.0**
                  Takes 2-4 scans to fade. Good for highly dynamic
                  environments.
                - **Weak: -0.05 to -0.1**
                  Takes 40-80 scans to fade. Good for mostly
                  static maps.

            max_range: (for insertion) Max range of the sensor.
            update_inner_occupancy: (for insertion) Whether to update
                                     the tree hierarchy.
        """
        
        # --- 1. DECAY STEP ---
        # The bounding box calculation is inside this function call
        self.decayOccupancyInBBX(point_cloud, sensor_origin, logodd_decay_value)

        # --- 2. INSERT STEP ---
        # Insert the new point cloud using the fast C++ method
        self.insertPointCloud(point_cloud, sensor_origin, max_range=max_range)
        if update_inner_occupancy:
            self.updateInnerOccupancy()