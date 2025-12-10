# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libcpp.string cimport string
from libcpp cimport bool as cppbool
from cython.operator cimport dereference as deref, preincrement as inc
cimport octomap_defs as defs
cimport pyoctomap.octree_base as octree_base
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t

# Fix NumPy API compatibility
np.import_array()

# Import NullPointerException from octree_base
from .octree_base import NullPointerException

# CountingOcTreeNode wrapper class
cdef class CountingOcTreeNode:
    """
    CountingOcTreeNode extends OcTreeDataNode with a counter.
    Each node stores an unsigned integer count value.
    """
    cdef defs.CountingOcTreeNode *thisptr
    
    def __cinit__(self):
        pass
    
    def __dealloc__(self):
        pass
    
    def getCount(self):
        """
        Get the count value of the node.
        Returns: unsigned int count
        """
        if self.thisptr:
            return self.thisptr.getCount()
        else:
            raise NullPointerException
    
    def increaseCount(self):
        """
        Increment the count value by 1.
        """
        if self.thisptr:
            self.thisptr.increaseCount()
        else:
            raise NullPointerException
    
    def setCount(self, unsigned int c):
        """
        Set the count value of the node.
        Args:
            c: Count value (unsigned int)
        """
        if self.thisptr:
            self.thisptr.setCount(c)
        else:
            raise NullPointerException
    
    def getValue(self):
        """
        Get the underlying value (same as getCount).
        Returns: unsigned int value
        """
        if self.thisptr:
            return self.thisptr.getValue()
        else:
            raise NullPointerException
    
    def setValue(self, unsigned int v):
        """
        Set the underlying value (same as setCount).
        Args:
            v: Value to set (unsigned int)
        """
        if self.thisptr:
            self.thisptr.setValue(v)
        else:
            raise NullPointerException

# CountingOcTree wrapper class
cdef class CountingOcTree:
    """
    CountingOcTree extends OcTreeBase with counting functionality.
    Each node stores a count value that can be incremented.
    """
    cdef defs.CountingOcTree *thisptr
    cdef bint owner
    
    def __cinit__(self, arg):
        import numbers
        cdef string c_filename
        cdef defs.CountingOcTree* result
        self.owner = True
        if isinstance(arg, numbers.Number):
            self.thisptr = new defs.CountingOcTree(<double?>arg)
        else:
            # CountingOcTree doesn't have a string constructor, so create with default resolution
            # and then read from file using the read() method
            if isinstance(arg, (bytes, bytearray)):
                c_filename = (<bytes>arg).decode('utf-8')
            else:
                c_filename = (<str>arg).encode('utf-8')
            # Create tree with default resolution
            self.thisptr = new defs.CountingOcTree(0.1)
            # Read from file - read() returns AbstractOcTree*, need to cast to CountingOcTree*
            result = <defs.CountingOcTree*>self.thisptr.read(c_filename)
            if result == NULL:
                # If read failed, clean up and raise error
                del self.thisptr
                self.thisptr = NULL
                raise IOError(f"Failed to read CountingOcTree from file: {arg}")
            # If read() returns a different pointer (new tree), use that instead
            if result != self.thisptr:
                del self.thisptr
                self.thisptr = result
    
    def __dealloc__(self):
        if self.owner and self.thisptr != NULL:
            del self.thisptr
            self.thisptr = NULL
    
    def updateNode(self, key_or_coord):
        """
        Update node count at given key or coordinate. Increments count.
        Args:
            key_or_coord: Either OcTreeKey or numpy array [x, y, z]
        Returns:
            CountingOcTreeNode if found/created, None otherwise
        """
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
        cdef defs.CountingOcTreeNode* node = NULL
        cdef defs.OcTreeKey key_in
        cdef np.ndarray[DOUBLE_t, ndim=1] coord
        if isinstance(key_or_coord, OcTreeKey):
            key_in.k[0] = key_or_coord[0]
            key_in.k[1] = key_or_coord[1]
            key_in.k[2] = key_or_coord[2]
            node = self.thisptr.updateNode(key_in)
        else:
            # Assume it's a coordinate array
            coord = np.array(key_or_coord, dtype=np.float64)
            node = self.thisptr.updateNode(defs.point3d(coord[0], coord[1], coord[2]))
        if node != NULL:
            result = CountingOcTreeNode()
            result.thisptr = node
            return result
        return None
    
    def getCentersMinHits(self, unsigned int min_hits):
        """
        Get centers of nodes with at least min_hits count.
        Args:
            min_hits: Minimum hit count threshold
        Returns:
            List of numpy arrays [x, y, z] representing node centers
        """
        # Check if tree is empty to avoid segfault
        if self.thisptr.size() == 0:
            return []
        
        cdef defs.list[defs.point3d] centers_list
        cdef defs.list[defs.point3d].iterator it
        cdef defs.list[defs.point3d].iterator end_it
        cdef defs.point3d p
        cdef list result = []
        
        self.thisptr.getCentersMinHits(centers_list, min_hits)
        
        it = centers_list.begin()
        end_it = centers_list.end()
        while it != end_it:
            p = deref(it)
            result.append(np.array([p.x(), p.y(), p.z()]))
            inc(it)
        
        return result
    
    # Inherit common OcTree methods
    def getResolution(self):
        return self.thisptr.getResolution()
    
    def getTreeDepth(self):
        return self.thisptr.getTreeDepth()
    
    def getTreeType(self):
        # CountingOcTree doesn't override getTreeType() in C++, so override it here
        return "CountingOcTree"
    
    def size(self):
        return self.thisptr.size()
    
    def getNumLeafNodes(self):
        return self.thisptr.getNumLeafNodes()
    
    def calcNumNodes(self):
        return self.thisptr.calcNumNodes()
    
    def clear(self):
        self.thisptr.clear()
    
    def coordToKey(self, np.ndarray[DOUBLE_t, ndim=1] coord, depth=None):
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
        cdef defs.OcTreeKey key
        if depth is None:
            key = self.thisptr.coordToKey(defs.point3d(coord[0], coord[1], coord[2]))
        else:
            key = self.thisptr.coordToKey(defs.point3d(coord[0], coord[1], coord[2]), <unsigned int?>depth)
        # Use the Cython type directly to avoid Python wrapper issues
        cdef octree_base.OcTreeKey res_cython = octree_base.OcTreeKey()
        res_cython.thisptr.k[0] = key.k[0]
        res_cython.thisptr.k[1] = key.k[1]
        res_cython.thisptr.k[2] = key.k[2]
        # Convert to Python object for return
        res = OcTreeKey()
        res[0] = res_cython[0]
        res[1] = res_cython[1]
        res[2] = res_cython[2]
        return res
    
    def keyToCoord(self, key, depth=None):
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
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
    
    def search(self, value, depth=0):
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
        cdef defs.OcTreeKey search_key
        node = CountingOcTreeNode()
        if isinstance(value, OcTreeKey):
            search_key.k[0] = value[0]
            search_key.k[1] = value[1]
            search_key.k[2] = value[2]
            node.thisptr = self.thisptr.search(search_key, <unsigned int?>depth)
        else:
            node.thisptr = self.thisptr.search(<double>value[0], <double>value[1], <double>value[2], <unsigned int?>depth)
        if node.thisptr == NULL:
            return None
        return node
    
    def isNodeOccupied(self, node):
        # CountingOcTree doesn't have occupancy concept, but we can check if node exists
        if isinstance(node, CountingOcTreeNode):
            if (<CountingOcTreeNode>node).thisptr:
                return (<CountingOcTreeNode>node).thisptr.getCount() > 0
            else:
                raise NullPointerException
        else:
            raise TypeError(f"Expected CountingOcTreeNode, got {type(node)}")
    
    def isNodeAtThreshold(self, node):
        # CountingOcTree doesn't have threshold concept, always return False
        if isinstance(node, CountingOcTreeNode):
            if (<CountingOcTreeNode>node).thisptr:
                return False
            else:
                raise NullPointerException
        else:
            raise TypeError(f"Expected CountingOcTreeNode, got {type(node)}")
    
    def castRay(self, np.ndarray[DOUBLE_t, ndim=1] origin,
                np.ndarray[DOUBLE_t, ndim=1] direction,
                np.ndarray[DOUBLE_t, ndim=1] end,
                ignoreUnknownCells=False, maxRange=-1.0):
        # CountingOcTree doesn't support castRay, but we can provide a stub
        # that returns False to maintain API compatibility
        return False
    
    def write(self, filename=None):
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
    
    def read(self, filename):
        cdef string c_filename = filename.encode('utf-8')
        cdef defs.CountingOcTree* result
        result = <defs.CountingOcTree*>self.thisptr.read(c_filename)
        if result != NULL:
            new_tree = CountingOcTree(0.1)
            new_tree.thisptr = result
            new_tree.owner = True
            return new_tree
        return None
    
    # Note: CountingOcTree doesn't have readBinary/writeBinary methods
    # (inherits from OcTreeBase, not OccupancyOcTreeBase)
    # Use read() and write() instead for file I/O
    
    def getRoot(self):
        cdef defs.CountingOcTreeNode* root_ptr = self.thisptr.getRoot()
        if root_ptr == NULL:
            return None
        node = CountingOcTreeNode()
        node.thisptr = root_ptr
        return node
    
    def nodeHasChildren(self, node):
        if isinstance(node, CountingOcTreeNode):
            if (<CountingOcTreeNode>node).thisptr:
                return self.thisptr.nodeHasChildren((<CountingOcTreeNode>node).thisptr)
            else:
                raise NullPointerException
        else:
            raise TypeError("Expected CountingOcTreeNode")
    
    def getNodeChild(self, node, int idx):
        child = CountingOcTreeNode()
        child.thisptr = self.thisptr.getNodeChild((<CountingOcTreeNode>node).thisptr, idx)
        return child
    
    def createNodeChild(self, node, int idx):
        child = CountingOcTreeNode()
        child.thisptr = self.thisptr.createNodeChild((<CountingOcTreeNode>node).thisptr, idx)
        return child
    
    def deleteNodeChild(self, node, int idx):
        self.thisptr.deleteNodeChild((<CountingOcTreeNode>node).thisptr, idx)
    
    def expandNode(self, node):
        self.thisptr.expandNode((<CountingOcTreeNode>node).thisptr)
    
    def isNodeCollapsible(self, node):
        return self.thisptr.isNodeCollapsible((<CountingOcTreeNode>node).thisptr)
    
    def pruneNode(self, node):
        return self.thisptr.pruneNode((<CountingOcTreeNode>node).thisptr)
    
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
    
    def memoryUsage(self):
        return self.thisptr.memoryUsage()
    
    def memoryUsageNode(self):
        return self.thisptr.memoryUsageNode()
    
    def volume(self):
        return self.thisptr.volume()

