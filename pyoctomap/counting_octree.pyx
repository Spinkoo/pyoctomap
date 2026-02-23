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
    
    def updateInnerCounts(self):
        """
        Update inner node counts to be the sum of their children's counts.
        This is useful if you've manually modified leaf node counts and need
        to propagate those changes upward.
        
        Note: CountingOcTree's updateNode() automatically maintains consistency
        by incrementing counts along the path. This method is only needed if
        you've directly modified node counts (e.g., using setCount()).
        
        Returns:
            None
        """
        cdef defs.CountingOcTreeNode* root_node = self.thisptr.getRoot()
        if root_node == NULL:
            return
        self._updateInnerCountsRecurs(root_node, 0)
    
    cdef void _updateInnerCountsRecurs(self, defs.CountingOcTreeNode* node, unsigned int depth):
        """
        Recursively update inner node counts to sum of children.
        Internal helper method.
        """
        if node == NULL:
            return
        cdef unsigned int i
        cdef defs.CountingOcTreeNode* child
        cdef unsigned int sum_count = 0
        
        # Only process inner nodes (nodes with children)
        if self.thisptr.nodeHasChildren(node):

            
            # Recurse into children first (bottom-up)
            if depth < self.thisptr.getTreeDepth():
                for i in range(8):
                    if self.thisptr.nodeChildExists(node, i):
                        child = self.thisptr.getNodeChild(node, i)
                        self._updateInnerCountsRecurs(child, depth + 1)
                        sum_count += child.getCount()
            
            # Update this node's count to sum of children
            node.setCount(sum_count)
    
    def updateInnerOccupancy(self):
        """
        Alias for updateInnerCounts() for API compatibility with other tree types.
        """
        self.updateInnerCounts()
    
    def extractPointCloud(self):
        """
        Extract all leaf node data from the tree.
        
        Uses the C++ getCentersMinHits traversal to collect leaf centers,
        then searches each to retrieve its count.
        
        Returns:
            tuple: (coords, counts) where:
                - coords: Nx3 float64 numpy array of voxel center coordinates
                - counts: N uint32 numpy array of observation counts
        """
        if self.thisptr.size() == 0:
            return (np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0,), dtype=np.uint32))
        
        # Use C++ getCentersMinHits to get all leaf centers with count >= 1
        cdef defs.list[defs.point3d] centers_list
        cdef defs.list[defs.point3d].iterator it
        cdef defs.list[defs.point3d].iterator end_it
        cdef defs.point3d p
        cdef defs.CountingOcTreeNode* node
        cdef list coord_list = []
        cdef list count_list = []
        
        self.thisptr.getCentersMinHits(centers_list, 1)
        
        it = centers_list.begin()
        end_it = centers_list.end()
        while it != end_it:
            p = deref(it)
            node = self.thisptr.search(p, 0)
            if node != NULL:
                coord_list.append([p.x(), p.y(), p.z()])
                count_list.append(node.getCount())
            inc(it)
        
        cdef int n = len(coord_list)
        if n == 0:
            return (np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0,), dtype=np.uint32))
        
        cdef np.ndarray[DOUBLE_t, ndim=2] coords = np.array(coord_list, dtype=np.float64)
        cdef np.ndarray[np.uint32_t, ndim=1] counts = np.array(count_list, dtype=np.uint32)
        return coords, counts
    

    def insertPointCloud(self,
                         np.ndarray[DOUBLE_t, ndim=2] pointcloud,
                         np.ndarray[DOUBLE_t, ndim=1] origin,
                         double maxrange=-1.0,
                         bint lazy_eval=False,
                         bint discretize=False):
        """
        Integrate a point cloud into the counting tree, incrementing counts
        for each point. Points beyond maxrange are truncated to maxrange
        distance from the origin.
        
        Note: CountingOcTree counts observation hits per voxel. Unlike
        probabilistic OcTrees, there is no free-space decrement along rays.
        
        Args:
            pointcloud: Nx3 numpy array of point coordinates
            origin: Sensor origin [x, y, z] (used for maxrange truncation)
            maxrange: Maximum range (-1 = unlimited). Points farther than
                      this are truncated to maxrange along the ray direction.
            lazy_eval: If True, skip updateInnerCounts after insertion
            discretize: If True, deduplicate points by voxel key first
        
        Returns:
            int: Number of points processed
        """
        cdef int i, num_points = pointcloud.shape[0]
        cdef np.ndarray[DOUBLE_t, ndim=1] point
        cdef defs.point3d origin_c = defs.point3d(<float>origin[0], <float>origin[1], <float>origin[2])
        cdef defs.point3d point_cpp
        cdef defs.OcTreeKey key
        cdef set unique_keys = set()
        cdef list discrete_points = []
        cdef np.ndarray[DOUBLE_t, ndim=1] ray_lengths
        cdef np.ndarray[DOUBLE_t, ndim=2] directions
        cdef defs.point3d direction_vec
        
        # Deduplicate by voxel key if requested
        if discretize:
            for i in range(num_points):
                point = pointcloud[i]
                key = self.thisptr.coordToKey(defs.point3d(point[0], point[1], point[2]))
                key_tuple = (key.k[0], key.k[1], key.k[2])
                if key_tuple not in unique_keys:
                    unique_keys.add(key_tuple)
                    discrete_points.append(point)
            pointcloud = np.array(discrete_points, dtype=np.float64)
            num_points = pointcloud.shape[0]
        
        if maxrange > 0.0:
            # Vectorized distance computation
            directions = pointcloud - origin
            ray_lengths = np.sqrt(np.sum(directions**2, axis=1))
            
            for i in range(num_points):
                point = pointcloud[i]
                point_cpp = defs.point3d(<float>point[0], <float>point[1], <float>point[2])
                
                if ray_lengths[i] > maxrange:
                    direction_vec = point_cpp - origin_c
                    direction_vec.normalize()
                    direction_vec *= maxrange
                    point_cpp = origin_c + direction_vec
                
                self.thisptr.updateNode(point_cpp)
        else:
            for i in range(num_points):
                point = pointcloud[i]
                self.thisptr.updateNode(defs.point3d(<float>point[0], <float>point[1], <float>point[2]))
        
        if not lazy_eval:
            self.updateInnerCounts()
        
        return num_points
    
    def insertPointCloudRaysFast(self,
                                 np.ndarray[DOUBLE_t, ndim=2] pointcloud,
                                 np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                                 double max_range=-1.0,
                                 bint lazy_eval=False):
        """
        DEPRECATED: Use insertPointCloud() instead.
        CountingOcTree does not use ray carving, so this method is identical to insertPointCloud.
        """
        import warnings
        warnings.warn("insertPointCloudRaysFast is deprecated for CountingOcTree. "
                      "Use insertPointCloud() instead, as CountingOcTree does not use ray carving.",
                      DeprecationWarning, stacklevel=2)
        return self.insertPointCloud(pointcloud, sensor_origin,
                                     maxrange=max_range, lazy_eval=lazy_eval)
    
    def updateNodes(self, values, lazy_eval=False):
        """
        Batch update: increment counts for multiple nodes.
        
        Args:
            values: List of OcTreeKey objects or numpy arrays [x, y, z]
            lazy_eval: If True, skip updateInnerCounts after updates
        
        Returns:
            int: Number of nodes updated
        """
        from .octomap import OcTreeKey
        cdef defs.OcTreeKey update_key
        cdef int count = 0
        cdef np.ndarray[DOUBLE_t, ndim=1] coord
        if values is None or len(values) == 0:
            return 0
        
        for v in values:
            if isinstance(v, OcTreeKey):
                update_key.k[0] = v[0]
                update_key.k[1] = v[1]
                update_key.k[2] = v[2]
                self.thisptr.updateNode(update_key)
            else:
                coord = np.array(v, dtype=np.float64)
                self.thisptr.updateNode(defs.point3d(<float>coord[0], <float>coord[1], <float>coord[2]))
            count += 1
        
        if not lazy_eval:
            self.updateInnerCounts()
        
        return count
    
    def addPointWithRayCasting(self,
                               np.ndarray[DOUBLE_t, ndim=1] point,
                               np.ndarray[DOUBLE_t, ndim=1] sensor_origin=None,
                               bint update_inner_occupancy=False):
        """
        DEPRECATED: Use updateNode() instead.
        CountingOcTree does not implement ray carving for single points.
        """
        import warnings
        warnings.warn("addPointWithRayCasting is deprecated for CountingOcTree. "
                      "Use updateNode() instead, as CountingOcTree does not perform ray carving.",
                      DeprecationWarning, stacklevel=2)
        cdef defs.point3d point_cpp = defs.point3d(<float>point[0],
                                                   <float>point[1],
                                                   <float>point[2])
        try:
            self.thisptr.updateNode(point_cpp)
            if update_inner_occupancy:
                self.updateInnerCounts()
            return True
        except Exception:
            return False

