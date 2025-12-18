# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libcpp.string cimport string
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
cimport octomap_defs as defs
cimport pyoctomap.octree_base as octree_base
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t

# Fix NumPy API compatibility
np.import_array()

# Import NullPointerException from octree_base
from .octree_base import NullPointerException

# OcTreeNodeStamped wrapper class
cdef class OcTreeNodeStamped:
    """
    OcTreeNodeStamped extends OcTreeNode with timestamp information.
    Each node stores a timestamp indicating when it was last updated.
    """
    cdef defs.OcTreeNodeStamped *thisptr
    
    def __cinit__(self):
        pass
    
    def __dealloc__(self):
        pass
    
    def getTimestamp(self):
        """
        Get the timestamp of the node.
        Returns: unsigned int timestamp
        """
        if self.thisptr:
            return self.thisptr.getTimestamp()
        else:
            raise NullPointerException
    
    def updateTimestamp(self):
        """
        Update the timestamp to current time.
        """
        if self.thisptr:
            self.thisptr.updateTimestamp()
        else:
            raise NullPointerException
    
    def setTimestamp(self, unsigned int t):
        """
        Set the timestamp of the node.
        Args:
            t: The timestamp value (unsigned int)
        """
        if self.thisptr:
            self.thisptr.setTimestamp(t)
        else:
            raise NullPointerException
    
    def updateOccupancyChildren(self):
        """
        Update occupancy from children and update timestamp.
        """
        if self.thisptr:
            self.thisptr.updateOccupancyChildren()
        else:
            raise NullPointerException
    
    # Inherit OcTreeNode methods
    def addValue(self, float p):
        if self.thisptr:
            self.thisptr.addValue(p)
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

# OcTreeStamped wrapper class
cdef class OcTreeStamped:
    """
    OcTreeStamped extends OcTree with timestamp information.
    Each node stores a timestamp indicating when it was last updated.
    Useful for temporal mapping and degrading outdated information.
    """
    cdef defs.OcTreeStamped *thisptr
    cdef bint owner
    
    def __cinit__(self, arg):
        import numbers
        cdef string c_filename
        cdef defs.OcTreeStamped* result
        self.owner = True
        if isinstance(arg, numbers.Number):
            self.thisptr = new defs.OcTreeStamped(<double?>arg)
        else:
            if isinstance(arg, (bytes, bytearray)):
                c_filename = (<bytes>arg).decode('utf-8')
            else:
                c_filename = (<str>arg).encode('utf-8')
            # Create tree with default resolution and then read from file
            self.thisptr = new defs.OcTreeStamped(0.1)
            result = <defs.OcTreeStamped*>self.thisptr.read(c_filename)
            if result == NULL:
                del self.thisptr
                self.thisptr = NULL
                raise IOError(f"Failed to read OcTreeStamped from file: {arg}")
            if result != self.thisptr:
                del self.thisptr
                self.thisptr = result
    
    def __dealloc__(self):
        if self.owner and self.thisptr != NULL:
            del self.thisptr
            self.thisptr = NULL
    
    def getLastUpdateTime(self):
        """
        Get the timestamp of the last update (root node timestamp).
        Returns: unsigned int timestamp, or 0 if tree is empty
        """
        if self.thisptr == NULL:
            return 0
        # Check if tree is empty - getLastUpdateTime() accesses root which may be NULL
        if self.thisptr.size() == 0:
            return 0
        return self.thisptr.getLastUpdateTime()
    
    def degradeOutdatedNodes(self, unsigned int time_thres):
        """
        Degrade nodes that haven't been updated within time_thres seconds.
        Args:
            time_thres: Time threshold in seconds (unsigned int)
        """
        self.thisptr.degradeOutdatedNodes(time_thres)
    
    def updateNodeLogOdds(self, node, float update):
        """
        Update node log odds and timestamp.
        Args:
            node: OcTreeNodeStamped instance
            update: Log odds update value (float)
        """
        if isinstance(node, OcTreeNodeStamped):
            if (<OcTreeNodeStamped>node).thisptr:
                self.thisptr.updateNodeLogOdds((<OcTreeNodeStamped>node).thisptr, update)
            else:
                raise NullPointerException
        else:
            raise TypeError(f"Expected OcTreeNodeStamped, got {type(node)}")
    
    def integrateMissNoTime(self, node):
        """
        Integrate miss observation without updating timestamp.
        Args:
            node: OcTreeNodeStamped instance
        """
        if isinstance(node, OcTreeNodeStamped):
            if (<OcTreeNodeStamped>node).thisptr:
                self.thisptr.integrateMissNoTime((<OcTreeNodeStamped>node).thisptr)
            else:
                raise NullPointerException
        else:
            raise TypeError(f"Expected OcTreeNodeStamped, got {type(node)}")
    
    def updateNode(self, x_or_key, y=None, z=None, occupied_or_logodds=None, lazy_eval=False):
        """
        Update node occupancy. Timestamp is automatically updated.
        Args:
            x_or_key: Either x coordinate (float) or OcTreeKey or coordinate array [x, y, z]
            y: y coordinate (if x is float) or log_odds_update/occupied (if x is coordinate)
            z: z coordinate (if x is float)
            occupied_or_logodds: bool (occupied) or float (log_odds_update) or None
            lazy_eval: Whether to defer inner node updates (default: False)
        Returns:
            OcTreeNodeStamped if found/created, None otherwise
        """
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
        cdef defs.OcTreeNodeStamped* node_ptr = NULL
        cdef defs.OcTreeKey key_in
        cdef np.ndarray[DOUBLE_t, ndim=1] coord
        
        # Handle different input formats
        if isinstance(x_or_key, OcTreeKey):
            # updateNode(key, occupied/logodds, lazy_eval)
            key_in.k[0] = x_or_key[0]
            key_in.k[1] = x_or_key[1]
            key_in.k[2] = x_or_key[2]
            if isinstance(y, bool):
                node_ptr = self.thisptr.updateNode(key_in, <cppbool>y, <cppbool>lazy_eval)
            elif isinstance(y, (int, float)):
                node_ptr = self.thisptr.updateNode(key_in, <float?>y, <cppbool>lazy_eval)
            else:
                raise TypeError("Expected bool (occupied) or float (log_odds_update) as second argument")
        elif isinstance(x_or_key, (list, tuple, np.ndarray)):
            # updateNode([x, y, z], occupied/logodds, lazy_eval)
            coord = np.array(x_or_key, dtype=np.float64)
            if len(coord) != 3:
                raise ValueError("Coordinate array must have 3 elements")
            if isinstance(y, bool):
                node_ptr = self.thisptr.updateNode(defs.point3d(coord[0], coord[1], coord[2]), <cppbool>y, <cppbool>lazy_eval)
            elif isinstance(y, (int, float)):
                node_ptr = self.thisptr.updateNode(defs.point3d(coord[0], coord[1], coord[2]), <float?>y, <cppbool>lazy_eval)
            else:
                raise TypeError("Expected bool (occupied) or float (log_odds_update) as second argument")
        elif isinstance(x_or_key, (int, float)) and y is not None and z is not None:
            # updateNode(x, y, z, occupied/logodds, lazy_eval)
            if isinstance(occupied_or_logodds, bool):
                node_ptr = self.thisptr.updateNode(<double?>x_or_key, <double?>y, <double?>z, <cppbool>occupied_or_logodds, <cppbool>lazy_eval)
            elif isinstance(occupied_or_logodds, (int, float)):
                node_ptr = self.thisptr.updateNode(<double?>x_or_key, <double?>y, <double?>z, <float?>occupied_or_logodds, <cppbool>lazy_eval)
            else:
                raise TypeError("Expected bool (occupied) or float (log_odds_update) as fourth argument")
        else:
            raise TypeError("Invalid arguments to updateNode")
        
        if node_ptr != NULL:
            result = OcTreeNodeStamped()
            result.thisptr = node_ptr
            return result
        return None
    
    # Inherit common OcTree methods
    def getResolution(self):
        return self.thisptr.getResolution()
    
    def getTreeDepth(self):
        return self.thisptr.getTreeDepth()
    
    def getTreeType(self):
        return self.thisptr.getTreeType().c_str()[:self.thisptr.getTreeType().length()].decode('utf-8')
    
    def size(self):
        return self.thisptr.size()
    
    def getNumLeafNodes(self):
        return self.thisptr.getNumLeafNodes()
    
    def calcNumNodes(self):
        return self.thisptr.calcNumNodes()
    
    def clear(self):
        self.thisptr.clear()
    
    def coordToKey(self, coord, depth=None):
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
        # Convert to numpy array if needed
        coord_arr = np.array(coord, dtype=np.float64)
        if len(coord_arr) != 3:
            raise ValueError("Coordinate must have 3 elements")
        cdef defs.OcTreeKey key
        if depth is None:
            key = self.thisptr.coordToKey(defs.point3d(coord_arr[0], coord_arr[1], coord_arr[2]))
        else:
            key = self.thisptr.coordToKey(defs.point3d(coord_arr[0], coord_arr[1], coord_arr[2]), <unsigned int?>depth)
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
        cdef defs.point3d p
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
        node = OcTreeNodeStamped()
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
        if isinstance(node, OcTreeNodeStamped):
            if (<OcTreeNodeStamped>node).thisptr:
                return self.thisptr.isNodeOccupied((<OcTreeNodeStamped>node).thisptr)
            else:
                raise NullPointerException
        else:
            raise TypeError(f"Expected OcTreeNodeStamped, got {type(node)}")
    
    def isNodeAtThreshold(self, node):
        if isinstance(node, OcTreeNodeStamped):
            if (<OcTreeNodeStamped>node).thisptr:
                return self.thisptr.isNodeAtThreshold((<OcTreeNodeStamped>node).thisptr)
            else:
                raise NullPointerException
        else:
            raise TypeError(f"Expected OcTreeNodeStamped, got {type(node)}")
    
    def castRay(self, np.ndarray[DOUBLE_t, ndim=1] origin,
                np.ndarray[DOUBLE_t, ndim=1] direction,
                np.ndarray[DOUBLE_t, ndim=1] end,
                ignoreUnknownCells=False, maxRange=-1.0):
        cdef defs.point3d e
        cdef cppbool hit
        hit = self.thisptr.castRay(
            defs.point3d(origin[0], origin[1], origin[2]),
            defs.point3d(direction[0], direction[1], direction[2]),
            e, <cppbool>ignoreUnknownCells, <double>maxRange)
        if hit:
            end[0:3] = e.x(), e.y(), e.z()
        return hit
    
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
        cdef defs.OcTreeStamped* result
        result = <defs.OcTreeStamped*>self.thisptr.read(c_filename)
        if result != NULL:
            new_tree = OcTreeStamped(0.1)
            new_tree.thisptr = result
            new_tree.owner = True
            return new_tree
        return None
    
    def writeBinary(self, filename=None):
        """
        Write file header and complete tree to binary file (.bt format) or stream.
        Persists occupancy via the core library and, when a filename is provided,
        appends a timestamp trailer understood by this binding.
        """
        import io, struct
        cdef defs.ostringstream oss
        cdef string c_filename
        if filename is not None:
            # Convert filename to string for file operations
            if isinstance(filename, (bytes, bytearray)):
                filename_str = (<bytes>filename).decode('utf-8')
            else:
                filename_str = <str>filename
            c_filename = filename_str.encode('utf-8')

            # Write core binary data
            ok = self.thisptr.writeBinary(c_filename)
            if not ok:
                return False

            # Append timestamp metadata trailer
            try:
                coords = []
                times = []
                for leaf in self.begin_leafs():
                    coord = leaf.getCoordinate()
                    timestamp = leaf.getTimestamp()
                    if timestamp != 0:
                        coords.append(coord)
                        times.append(timestamp)

                if not coords:
                    return True

                # Compress and write trailer
                coords_arr = np.asarray(coords, dtype=np.float64)
                times_arr = np.asarray(times, dtype=np.uint32)

                buf = io.BytesIO()
                np.savez_compressed(buf, coords=coords_arr, times=times_arr)
                payload = buf.getvalue()

                with open(filename_str, "ab") as f:
                    f.write(b"\n#PYOC_TIME_V1\n")
                    f.write(struct.pack("<I", len(payload)))
                    f.write(payload)

            except Exception:
                # Ignore metadata failures to remain compatible
                pass

            return True
        else:
            ret = self.thisptr.writeBinary(<defs.ostream&?>oss)
            if ret:
                return oss.str().c_str()[:oss.str().length()]
            else:
                return False
    
    def readBinary(self, filename):
        """
        Read tree from binary file (.bt format).
        Loads occupancy via the core library and restores timestamps if a trailer
        appended by this binding is present.
        """
        import io, struct
        cdef string c_filename = filename.encode('utf-8')
        ok = self.thisptr.readBinary(c_filename)
        if not ok:
            return False

        # Attempt to restore timestamp metadata from trailer
        try:
            marker = b"\n#PYOC_TIME_V1\n"
            with open(filename, "rb") as f:
                data = f.read()
            pos = data.rfind(marker)
            if pos != -1 and pos + len(marker) + 4 <= len(data):
                start_len = pos + len(marker)
                payload_len = struct.unpack("<I", data[start_len:start_len+4])[0]
                payload_start = start_len + 4
                payload_end = payload_start + payload_len
                if payload_end <= len(data):
                    payload = data[payload_start:payload_end]
                    buf = io.BytesIO(payload)
                    npz = np.load(buf, allow_pickle=False)
                    coords = npz["coords"]
                    times = npz["times"]
                    for coord, t in zip(coords, times):
                        node = self.updateNode(coord, True)
                        if node is not None:
                            node.setTimestamp(int(t))
        except Exception:
            pass

        return True
    
    def getRoot(self):
        cdef defs.OcTreeNodeStamped* root_ptr = self.thisptr.getRoot()
        if root_ptr == NULL:
            return None
        node = OcTreeNodeStamped()
        node.thisptr = root_ptr
        return node
    
    def nodeHasChildren(self, node):
        if isinstance(node, OcTreeNodeStamped):
            if (<OcTreeNodeStamped>node).thisptr:
                return self.thisptr.nodeHasChildren((<OcTreeNodeStamped>node).thisptr)
            else:
                raise NullPointerException
        else:
            raise TypeError("Expected OcTreeNodeStamped")
    
    def getNodeChild(self, node, int idx):
        child = OcTreeNodeStamped()
        child.thisptr = self.thisptr.getNodeChild((<OcTreeNodeStamped>node).thisptr, idx)
        return child
    
    def createNodeChild(self, node, int idx):
        child = OcTreeNodeStamped()
        child.thisptr = self.thisptr.createNodeChild((<OcTreeNodeStamped>node).thisptr, idx)
        return child
    
    def deleteNodeChild(self, node, int idx):
        self.thisptr.deleteNodeChild((<OcTreeNodeStamped>node).thisptr, idx)
    
    def expandNode(self, node):
        self.thisptr.expandNode((<OcTreeNodeStamped>node).thisptr)
    
    def isNodeCollapsible(self, node):
        return self.thisptr.isNodeCollapsible((<OcTreeNodeStamped>node).thisptr)
    
    def pruneNode(self, node):
        return self.thisptr.pruneNode((<OcTreeNodeStamped>node).thisptr)
    
    def updateInnerOccupancy(self):
        """
        Updates the occupancy of all inner nodes to reflect their children's occupancy.
        """
        self.thisptr.updateInnerOccupancy()
    
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
    
    def getBBXMin(self):
        cdef defs.point3d p = self.thisptr.getBBXMin()
        return np.array([p.x(), p.y(), p.z()])
    
    def getBBXMax(self):
        cdef defs.point3d p = self.thisptr.getBBXMax()
        return np.array([p.x(), p.y(), p.z()])
    
    def getBBXCenter(self):
        cdef defs.point3d p = self.thisptr.getBBXCenter()
        return np.array([p.x(), p.y(), p.z()])
    
    def getBBXBounds(self):
        cdef defs.point3d p = self.thisptr.getBBXBounds()
        return np.array([p.x(), p.y(), p.z()])
    
    def setBBXMin(self, np.ndarray[DOUBLE_t, ndim=1] value):
        self.thisptr.setBBXMin(defs.point3d(value[0], value[1], value[2]))
    
    def setBBXMax(self, np.ndarray[DOUBLE_t, ndim=1] value):
        self.thisptr.setBBXMax(defs.point3d(value[0], value[1], value[2]))
    
    def inBBX(self, np.ndarray[DOUBLE_t, ndim=1] value):
        return self.thisptr.inBBX(defs.point3d(value[0], value[1], value[2]))
    
    def insertPointCloud(self, np.ndarray[DOUBLE_t, ndim=2] pointcloud,
                         np.ndarray[DOUBLE_t, ndim=1] sensor_origin,
                         maxrange=-1., lazy_eval=False, discretize=False):
        """
        Integrate a Pointcloud into the octree.
        """
        cdef defs.Pointcloud pc = defs.Pointcloud()
        for p in pointcloud:
            pc.push_back(<float>p[0], <float>p[1], <float>p[2])
        self.thisptr.insertPointCloud(pc,
                                      defs.point3d(sensor_origin[0], sensor_origin[1], sensor_origin[2]),
                                      <double?>maxrange, bool(lazy_eval), bool(discretize))
        # Always call updateInnerOccupancy() when lazy_eval=False to ensure tree consistency
        if not lazy_eval:
            self.updateInnerOccupancy()
    
    def insertPointCloudWithTimestamp(self, double[:,::1] points, unsigned int timestamp,
                                      sensor_origin=None, double max_range=-1.0, bint lazy_eval=True):
        """
        Inserts points and updates their timestamp to the specified value.
        This method inserts geometry first, then updates timestamps using key-based search.
        
        Args:
            points: Nx3 array of point coordinates
            timestamp: Timestamp value to set for all nodes (unsigned int)
            sensor_origin: Optional sensor origin [x, y, z] for ray casting. If None, uses (0, 0, 0).
            max_range: Maximum range for ray casting (-1 = unlimited)
            lazy_eval: If True, defer updateInnerOccupancy (call manually later)
        
        Returns:
            int: Number of points processed
        """
        cdef int i
        cdef int n_points = points.shape[0]
        cdef double x, y, z
        cdef defs.OcTreeNodeStamped* node_ptr = NULL
        cdef defs.point3d origin_c
        
        # 1. Insert Geometry (Standard C++ Batch)
        # Use provided sensor origin or default to (0, 0, 0)
        if sensor_origin is None:
            origin_c = defs.point3d(0.0, 0.0, 0.0)
        else:
            origin_arr = np.array(sensor_origin, dtype=np.float64)
            if len(origin_arr) != 3:
                raise ValueError("sensor_origin must be a 3-element array [x, y, z]")
            origin_c = defs.point3d(<float>origin_arr[0], <float>origin_arr[1], <float>origin_arr[2])
        
        cdef defs.Pointcloud pc = defs.Pointcloud()
        for i in range(n_points):
            pc.push_back(<float>points[i, 0], <float>points[i, 1], <float>points[i, 2])
        self.thisptr.insertPointCloud(pc, origin_c, <double>max_range, <cppbool>lazy_eval, <cppbool>False)
        
        # 2. Update Timestamps (The "Missing" Batch Loop)
        # This loop runs at C speed, not Python speed
        # OPTIMIZATION: Convert coordinates to keys first, then use key-based search
        # Key-based operations are faster than coordinate-based, and search is fast since nodes exist
        cdef defs.OcTreeKey key
        cdef defs.point3d coord_pt
        for i in range(n_points):
            x = points[i, 0]
            y = points[i, 1]
            z = points[i, 2]

            # Convert coordinate to key once (key-based operations are faster)
            coord_pt = defs.point3d(<float>x, <float>y, <float>z)
            key = self.thisptr.coordToKey(coord_pt)
            
            # Search for node using key (fast since insertPointCloud just created it)
            # Then set timestamp directly on node pointer - avoids redundant updateNode call
            node_ptr = self.thisptr.search(key, 0)
            if node_ptr != NULL:
                node_ptr.setTimestamp(timestamp)

        if not lazy_eval:
            self.thisptr.updateInnerOccupancy()
        
        return n_points
    
    # Helper method to get the C++ pointer address (for use in other modules)
    cpdef size_t _get_ptr_addr(self):
        return <size_t>self.thisptr
    
    def begin_leafs(self, maxDepth=0):
        """Return a simplified leaf iterator"""
        from .octree_iterators import SimpleLeafIterator
        return SimpleLeafIterator(self, maxDepth)
    
    def begin_leafs_bbx(self, np.ndarray[DOUBLE_t, ndim=1] bbx_min, np.ndarray[DOUBLE_t, ndim=1] bbx_max, maxDepth=0):
        """Return a simplified leaf iterator for a bounding box"""
        from .octree_iterators import SimpleLeafBBXIterator
        return SimpleLeafBBXIterator(self, bbx_min, bbx_max, maxDepth)

