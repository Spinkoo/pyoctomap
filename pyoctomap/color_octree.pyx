# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libcpp.string cimport string
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from cython.operator cimport dereference as deref
cimport octomap_defs as defs
cimport pyoctomap.octree_base as octree_base
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t

# Fix NumPy API compatibility
np.import_array()

# Import NullPointerException from octree_base
from .octree_base import NullPointerException

# OcTreeKey will be imported at runtime from octomap module
# We'll use a runtime import for OcTreeKey since it's a Python class

# ColorOcTreeNode wrapper class
cdef class ColorOcTreeNode:
    """
    ColorOcTreeNode extends OcTreeNode with color information.
    Each node stores RGB color values (0-255).
    """
    cdef defs.ColorOcTreeNode *thisptr
    
    def __cinit__(self):
        pass
    
    def __dealloc__(self):
        pass
    
    def getColor(self):
        """
        Get the RGB color of the node.
        Returns: tuple (r, g, b) with values 0-255
        """
        cdef defs.ColorOcTreeNode.Color c
        cdef const defs.ColorOcTreeNode* const_node
        if self.thisptr:
            const_node = <const defs.ColorOcTreeNode*>self.thisptr
            c = const_node.getColor()
            return (c.r, c.g, c.b)
        else:
            raise NullPointerException
    
    def setColor(self, r, g, b):
        """
        Set the RGB color of the node.
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
        """
        if self.thisptr:
            self.thisptr.setColor(<unsigned char?>r, <unsigned char?>g, <unsigned char?>b)
        else:
            raise NullPointerException
    
    def isColorSet(self):
        """
        Check if color has been set (not pure white).
        Returns: True if color is set, False otherwise
        """
        if self.thisptr:
            return self.thisptr.isColorSet()
        else:
            raise NullPointerException
    
    def updateColorChildren(self):
        """
        Update color from children's average color.
        """
        if self.thisptr:
            self.thisptr.updateColorChildren()
        else:
            raise NullPointerException
    
    def getAverageChildColor(self):
        """
        Get the average color of child nodes.
        Returns: tuple (r, g, b) with values 0-255
        """
        cdef defs.ColorOcTreeNode.Color c
        if self.thisptr:
            c = self.thisptr.getAverageChildColor()
            return (c.r, c.g, c.b)
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
    
    def _get_ptr_addr(self):
        """Helper method to get the C++ pointer address (for use in other modules)"""
        return <size_t>self.thisptr

# ColorOcTree wrapper class
cdef class ColorOcTree:
    """
    ColorOcTree extends OcTree with color information.
    Each node can store RGB color values for visualization and color-based mapping.
    """
    cdef defs.ColorOcTree *thisptr
    cdef bint owner
    
    def __cinit__(self, arg):
        import numbers
        cdef string c_filename
        cdef defs.ColorOcTree* result
        self.owner = True
        if isinstance(arg, numbers.Number):
            self.thisptr = new defs.ColorOcTree(<double?>arg)
        else:
            # ColorOcTree doesn't have a string constructor, so create with default resolution
            # and then read from file using the read() method
            if isinstance(arg, (bytes, bytearray)):
                c_filename = (<bytes>arg).decode('utf-8')
            else:
                c_filename = (<str>arg).encode('utf-8')
            # Create tree with default resolution
            self.thisptr = new defs.ColorOcTree(0.1)
            # Read from file - read() returns AbstractOcTree*, need to cast to ColorOcTree*
            result = <defs.ColorOcTree*>self.thisptr.read(c_filename)
            if result == NULL:
                # If read failed, clean up and raise error
                del self.thisptr
                self.thisptr = NULL
                raise IOError(f"Failed to read ColorOcTree from file: {arg}")
            # If read() returns a different pointer (new tree), use that instead
            if result != self.thisptr:
                del self.thisptr
                self.thisptr = result
    
    def __dealloc__(self):
        if self.owner and self.thisptr != NULL:
            del self.thisptr
            self.thisptr = NULL
    
    def setNodeColor(self, key_or_coord, r, g, b):
        """
        Set node color at given key or coordinate. Replaces previous color.
        Args:
            key_or_coord: Either OcTreeKey or numpy array [x, y, z]
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
        Returns:
            ColorOcTreeNode if found, None otherwise
        """
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
        cdef defs.ColorOcTreeNode* node = NULL
        cdef defs.OcTreeKey key_in
        cdef np.ndarray[DOUBLE_t, ndim=1] coord
        if isinstance(key_or_coord, OcTreeKey):
            key_in.k[0] = key_or_coord[0]
            key_in.k[1] = key_or_coord[1]
            key_in.k[2] = key_or_coord[2]
            node = self.thisptr.setNodeColor(key_in, <unsigned char?>r, <unsigned char?>g, <unsigned char?>b)
        else:
            # Assume it's a coordinate array
            coord = np.array(key_or_coord, dtype=np.float64)
            node = self.thisptr.setNodeColor(<float?>coord[0], <float?>coord[1], <float?>coord[2],
                                             <unsigned char?>r, <unsigned char?>g, <unsigned char?>b)
        if node != NULL:
            result = ColorOcTreeNode()
            result.thisptr = node
            return result
        return None
    
    def averageNodeColor(self, key_or_coord, r, g, b):
        """
        Average node color with new measurement. Averages with previous color if set.
        Args:
            key_or_coord: Either OcTreeKey or numpy array [x, y, z]
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
        Returns:
            ColorOcTreeNode if found, None otherwise
        """
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
        cdef defs.ColorOcTreeNode* node = NULL
        cdef defs.OcTreeKey key_in
        cdef np.ndarray[DOUBLE_t, ndim=1] coord
        if isinstance(key_or_coord, OcTreeKey):
            key_in.k[0] = key_or_coord[0]
            key_in.k[1] = key_or_coord[1]
            key_in.k[2] = key_or_coord[2]
            node = self.thisptr.averageNodeColor(key_in, <unsigned char?>r, <unsigned char?>g, <unsigned char?>b)
        else:
            coord = np.array(key_or_coord, dtype=np.float64)
            node = self.thisptr.averageNodeColor(<float?>coord[0], <float?>coord[1], <float?>coord[2],
                                                 <unsigned char?>r, <unsigned char?>g, <unsigned char?>b)
        if node != NULL:
            result = ColorOcTreeNode()
            result.thisptr = node
            return result
        return None
    
    def integrateNodeColor(self, key_or_coord, r, g, b):
        """
        Integrate color measurement weighted by occupancy probability.
        Args:
            key_or_coord: Either OcTreeKey or numpy array [x, y, z]
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
        Returns:
            ColorOcTreeNode if found, None otherwise
        """
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
        cdef defs.ColorOcTreeNode* node = NULL
        cdef defs.OcTreeKey key_in
        cdef np.ndarray[DOUBLE_t, ndim=1] coord
        if isinstance(key_or_coord, OcTreeKey):
            key_in.k[0] = key_or_coord[0]
            key_in.k[1] = key_or_coord[1]
            key_in.k[2] = key_or_coord[2]
            node = self.thisptr.integrateNodeColor(key_in, <unsigned char?>r, <unsigned char?>g, <unsigned char?>b)
        else:
            coord = np.array(key_or_coord, dtype=np.float64)
            node = self.thisptr.integrateNodeColor(<float?>coord[0], <float?>coord[1], <float?>coord[2],
                                                   <unsigned char?>r, <unsigned char?>g, <unsigned char?>b)
        if node != NULL:
            result = ColorOcTreeNode()
            result.thisptr = node
            return result
        return None
    
    def updateInnerOccupancy(self):
        """
        Updates the occupancy and color of all inner nodes to reflect their children's values.
        """
        self.thisptr.updateInnerOccupancy()
    
    def writeColorHistogram(self, filename):
        """
        Write RGB color histogram to file using gnuplot (not supported on Windows).
        Args:
            filename: Output filename for histogram
        """
        cdef string c_filename = filename.encode('utf-8')
        self.thisptr.writeColorHistogram(c_filename)
    
    # Inherit common OcTree methods
    def getResolution(self):
        return self.thisptr.getResolution()
    
    def getTreeDepth(self):
        return self.thisptr.getTreeDepth()
    
    def getTreeType(self):
        return self.thisptr.getTreeType().c_str().decode('utf-8')
    
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
        node = ColorOcTreeNode()
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
    
    def updateNode(self, value, update, lazy_eval=False):
        """
        Update node occupancy and return ColorOcTreeNode.
        Args:
            value: Either OcTreeKey or numpy array [x, y, z]
            update: Either bool (occupied) or float (log_odds_update)
            lazy_eval: Whether to defer inner node updates
        Returns:
            ColorOcTreeNode if found/created, None otherwise
        """
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
        cdef defs.ColorOcTreeNode* node = NULL
        cdef defs.OcTreeKey update_key
        if isinstance(value, OcTreeKey):
            if isinstance(update, bool):
                update_key.k[0] = value[0]
                update_key.k[1] = value[1]
                update_key.k[2] = value[2]
                node = self.thisptr.updateNode(update_key, <cppbool>update, <cppbool?>lazy_eval)
            else:
                update_key.k[0] = value[0]
                update_key.k[1] = value[1]
                update_key.k[2] = value[2]
                node = self.thisptr.updateNode(update_key, <float?>update, <cppbool?>lazy_eval)
        else:
            if isinstance(update, bool):
                node = self.thisptr.updateNode(<double?>value[0], <double?>value[1], <double?>value[2],
                                              <cppbool>update, <cppbool?>lazy_eval)
            else:
                node = self.thisptr.updateNode(<double?>value[0], <double?>value[1], <double?>value[2],
                                              <float?>update, <cppbool?>lazy_eval)
        if node != NULL:
            result = ColorOcTreeNode()
            result.thisptr = node
            return result
        return None
    
    def insertPointCloud(self, np.ndarray[DOUBLE_t, ndim=2] pointcloud,
                         np.ndarray[DOUBLE_t, ndim=1] origin,
                         maxrange=-1., lazy_eval=False, discretize=False):
        """
        Integrate a Pointcloud with color information.
        """
        cdef defs.Pointcloud pc = defs.Pointcloud()
        for p in pointcloud:
            pc.push_back(<float>p[0], <float>p[1], <float>p[2])
        cdef defs.point3d origin_c = defs.point3d(<float>origin[0], <float>origin[1], <float>origin[2])
        self.thisptr.insertPointCloud(pc,
                                      origin_c,
                                      <double?>maxrange, bool(lazy_eval), bool(discretize))
        # Always call updateInnerOccupancy() when lazy_eval=False to ensure tree consistency
        if not lazy_eval:
            self.updateInnerOccupancy()
    
    def insertPointCloudWithColor(self, double[:,::1] points, double[:,::1] colors, 
                                   sensor_origin=None, double max_range=-1.0, bint lazy_eval=True):
        """
        Custom high-performance implementation that inserts point cloud geometry
        and then updates colors in a C++-speed loop.
        
        Args:
            points: Nx3 array of point coordinates
            colors: Nx3 array of color values (0-1 range, will be converted to 0-255)
            sensor_origin: Optional sensor origin [x, y, z] for ray casting. If None, uses (0, 0, 0).
            max_range: Maximum range for ray casting (-1 = unlimited)
            lazy_eval: If True, defer updateInnerOccupancy (call manually later)
        
        Returns:
            int: Number of points processed
        """
        cdef int i
        cdef int n_points = points.shape[0]
        cdef int n_colors = colors.shape[0]
        cdef int n_color_channels = colors.shape[1]
        cdef double x, y, z
        cdef unsigned char r, g, b
        cdef defs.point3d origin_c
        
        if n_points != n_colors:
            raise ValueError("Points and colors arrays must have the same number of rows")
        if n_color_channels != 3:
            raise ValueError("Colors array must have 3 columns (R, G, B)")
        
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
        
        # 2. Update Colors (The "Missing" Batch Loop)
        # This loop runs at C speed, not Python speed
        # OPTIMIZATION: Convert coordinates to keys first, then use key-based search
        # Key-based operations are faster than coordinate-based, and search is fast since nodes exist
        cdef defs.ColorOcTreeNode* node_ptr = NULL
        cdef defs.OcTreeKey key
        cdef defs.point3d coord_pt
        for i in range(n_points):
            x = points[i, 0]
            y = points[i, 1]
            z = points[i, 2]
            
            # Convert 0-1 float to 0-255 uint8
            r = <unsigned char>(colors[i, 0] * 255)
            g = <unsigned char>(colors[i, 1] * 255)
            b = <unsigned char>(colors[i, 2] * 255)

            # Convert coordinate to key once (key-based operations are faster)
            coord_pt = defs.point3d(<float>x, <float>y, <float>z)
            key = self.thisptr.coordToKey(coord_pt)
            
            # Search for node using key (fast since insertPointCloud just created it)
            # Then set color directly on node pointer - avoids redundant updateNode call
            node_ptr = self.thisptr.search(key, 0)
            if node_ptr != NULL:
                node_ptr.setColor(r, g, b)

        if not lazy_eval:
            self.thisptr.updateInnerOccupancy()
        
        return n_points
    
    def isNodeOccupied(self, node):
        """
        Check if a node is occupied. Accepts ColorOcTreeNode or iterator.
        """
        cdef defs.point3d search_point
        cdef defs.ColorOcTreeNode* found_node
        cdef defs.ColorOcTreeNode* node_ptr
        cdef size_t ptr_addr
        cdef bint result
        # Runtime import to avoid circular dependency
        from .octree_iterators import SimpleTreeIterator, SimpleLeafIterator, SimpleLeafBBXIterator
        
        if isinstance(node, ColorOcTreeNode):
            # Access pointer through helper method that returns address
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.ColorOcTreeNode*>ptr_addr
            if node_ptr != NULL:
                return self.thisptr.isNodeOccupied(deref(node_ptr))
            else:
                raise NullPointerException
        elif isinstance(node, (SimpleTreeIterator, SimpleLeafIterator, SimpleLeafBBXIterator)):
            # Handle iterator case - use coordinate to search for the node
            try:
                coord = node.getCoordinate()
                found_node = self.thisptr.search(<double>coord[0], <double>coord[1], <double>coord[2], <unsigned int>0)
                if found_node != NULL:
                    result = self.thisptr.isNodeOccupied(deref(found_node))
                    return result
                else:
                    return False
            except Exception:
                return False
        else:
            raise TypeError(f"Expected ColorOcTreeNode or iterator, got {type(node)}")
    
    def isNodeAtThreshold(self, node):
        """
        Check if a node is at occupancy threshold. Accepts ColorOcTreeNode or iterator.
        """
        cdef defs.point3d search_point
        cdef defs.ColorOcTreeNode* found_node
        cdef defs.ColorOcTreeNode* node_ptr
        cdef size_t ptr_addr
        cdef bint result
        # Runtime import to avoid circular dependency
        from .octree_iterators import SimpleTreeIterator, SimpleLeafIterator, SimpleLeafBBXIterator
        
        if isinstance(node, ColorOcTreeNode):
            # Access pointer through helper method that returns address
            ptr_addr = node._get_ptr_addr()
            node_ptr = <defs.ColorOcTreeNode*>ptr_addr
            if node_ptr != NULL:
                return self.thisptr.isNodeAtThreshold(deref(node_ptr))
            else:
                raise NullPointerException
        elif isinstance(node, (SimpleTreeIterator, SimpleLeafIterator, SimpleLeafBBXIterator)):
            # Handle iterator case - use coordinate to search for the node
            try:
                coord = node.getCoordinate()
                found_node = self.thisptr.search(<double>coord[0], <double>coord[1], <double>coord[2], <unsigned int>0)
                if found_node != NULL:
                    result = self.thisptr.isNodeAtThreshold(deref(found_node))
                    return result
                else:
                    return False
            except Exception:
                return False
        else:
            raise TypeError(f"Expected ColorOcTreeNode or iterator, got {type(node)}")
    
    def castRay(self, np.ndarray[DOUBLE_t, ndim=1] origin,
                np.ndarray[DOUBLE_t, ndim=1] direction,
                np.ndarray[DOUBLE_t, ndim=1] end,
                ignoreUnknownCells=False, maxRange=-1.0):
        cdef defs.point3d e
        cdef cppbool hit
        hit = self.thisptr.castRay(
            defs.point3d(origin[0], origin[1], origin[2]),
            defs.point3d(direction[0], direction[1], direction[2]),
            e, bool(ignoreUnknownCells), <double?>maxRange)
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
        cdef defs.ColorOcTree* result
        result = <defs.ColorOcTree*>self.thisptr.read(c_filename)
        if result != NULL:
            new_tree = ColorOcTree(0.1)
            new_tree.thisptr = result
            new_tree.owner = True
            return new_tree
        return None
    
    def readBinary(self, filename):
        """
        Read tree from binary file (.bt format).
        Loads occupancy from the core library and, if present, restores a
        color trailer appended by this binding.
        """
        import io, struct
        cdef string c_filename
        if isinstance(filename, (bytes, bytearray)):
            filename_str = (<bytes>filename).decode('utf-8')
        else:
            filename_str = <str>filename
        c_filename = filename_str.encode('utf-8')

        ok = self.thisptr.readBinary(c_filename)
        if not ok:
            return False

        # Attempt to restore color metadata from trailer if present
        try:
            marker = b"\n#PYOC_COLOR_V1\n"
            with open(filename_str, "rb") as f:
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
                    colors = npz["colors"]
                    for coord, color in zip(coords, colors):
                        self.setNodeColor(coord, int(color[0]), int(color[1]), int(color[2]))
        except Exception:
            # Ignore metadata errors to stay backward-compatible
            pass

        return True
    
    def writeBinary(self, filename):
        """
        Write file header and complete tree to binary file (.bt format).
        Persists occupancy via the core library and appends a color trailer
        understood by this binding. Safe to read with older readers (they
        will ignore the trailer).
        """
        import io, struct
        cdef string c_filename

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

        # Append color metadata trailer
        try:
            coords = []
            colors = []
            for leaf in self.begin_leafs():
                coord = leaf.getCoordinate()
                color = leaf.getColor()
                if color != (255, 255, 255):
                    coords.append(coord)
                    colors.append(color)

            if not coords:
                return True

            # Compress and write trailer
            coords_arr = np.asarray(coords, dtype=np.float64)
            colors_arr = np.asarray(colors, dtype=np.uint8)

            buf = io.BytesIO()
            np.savez_compressed(buf, coords=coords_arr, colors=colors_arr)
            payload = buf.getvalue()

            with open(filename_str, "ab") as f:
                f.write(b"\n#PYOC_COLOR_V1\n")
                f.write(struct.pack("<I", len(payload)))
                f.write(payload)

        except Exception:
            # Ignore metadata failures to remain compatible
            pass

        return True
    
    def getBBXMin(self):
        cdef defs.point3d p = self.thisptr.getBBXMin()
        return np.array((p.x(), p.y(), p.z()))
    
    def getBBXMax(self):
        cdef defs.point3d p = self.thisptr.getBBXMax()
        return np.array((p.x(), p.y(), p.z()))
    
    def getBBXCenter(self):
        cdef defs.point3d p = self.thisptr.getBBXCenter()
        return np.array((p.x(), p.y(), p.z()))
    
    def getBBXBounds(self):
        cdef defs.point3d p = self.thisptr.getBBXBounds()
        return np.array((p.x(), p.y(), p.z()))
    
    def setBBXMin(self, np.ndarray[DOUBLE_t, ndim=1] min):
        self.thisptr.setBBXMin(defs.point3d(min[0], min[1], min[2]))
    
    def setBBXMax(self, np.ndarray[DOUBLE_t, ndim=1] max):
        self.thisptr.setBBXMax(defs.point3d(max[0], max[1], max[2]))
    
    def inBBX(self, np.ndarray[DOUBLE_t, ndim=1] p):
        return self.thisptr.inBBX(defs.point3d(p[0], p[1], p[2]))
    
    def getRoot(self):
        node = ColorOcTreeNode()
        node.thisptr = self.thisptr.getRoot()
        return node
    
    def nodeHasChildren(self, node):
        if isinstance(node, ColorOcTreeNode):
            if (<ColorOcTreeNode>node).thisptr:
                return self.thisptr.nodeHasChildren((<ColorOcTreeNode>node).thisptr)
            else:
                raise NullPointerException
        else:
            raise TypeError("Expected ColorOcTreeNode")
    
    def getNodeChild(self, node, int idx):
        child = ColorOcTreeNode()
        child.thisptr = self.thisptr.getNodeChild((<ColorOcTreeNode>node).thisptr, idx)
        return child
    
    
    def isNodeCollapsible(self, node):
        return self.thisptr.isNodeCollapsible((<ColorOcTreeNode>node).thisptr)
    
    def pruneNode(self, node):
        return self.thisptr.pruneNode((<ColorOcTreeNode>node).thisptr)
    
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
    
    def setOccupancyThres(self, double prob):
        self.thisptr.setOccupancyThres(prob)
    
    def setProbHit(self, double prob):
        self.thisptr.setProbHit(prob)
    
    def setProbMiss(self, double prob):
        self.thisptr.setProbMiss(prob)
    
    def getClampingThresMax(self):
        return self.thisptr.getClampingThresMax()
    
    def getClampingThresMaxLog(self):
        return self.thisptr.getClampingThresMaxLog()
    
    def getClampingThresMin(self):
        return self.thisptr.getClampingThresMin()
    
    def getClampingThresMinLog(self):
        return self.thisptr.getClampingThresMinLog()
    
    def setClampingThresMax(self, double thresProb):
        self.thisptr.setClampingThresMax(thresProb)
    
    def setClampingThresMin(self, double thresProb):
        self.thisptr.setClampingThresMin(thresProb)
    
    def prune(self):
        """Prune the tree by removing collapsible nodes"""
        self.thisptr.prune()
    
    def expand(self):
        """Expand all nodes to maximum depth"""
        self.thisptr.expand()
    
    def useBBXLimit(self, enable):
        """Enable or disable bounding box limit for queries"""
        self.thisptr.useBBXLimit(bool(enable))
    
    def expandNode(self, node):
        """Expand a node to create children"""
        if isinstance(node, ColorOcTreeNode):
            if (<ColorOcTreeNode>node).thisptr:
                self.thisptr.expandNode((<ColorOcTreeNode>node).thisptr)
            else:
                raise NullPointerException
        else:
            raise TypeError("Expected ColorOcTreeNode")
    
    def createNodeChild(self, node, int idx):
        """Create a child node at the specified index"""
        child = ColorOcTreeNode()
        if isinstance(node, ColorOcTreeNode):
            if (<ColorOcTreeNode>node).thisptr:
                child.thisptr = self.thisptr.createNodeChild((<ColorOcTreeNode>node).thisptr, idx)
                return child
            else:
                raise NullPointerException
        else:
            raise TypeError("Expected ColorOcTreeNode")
    
    def deleteNodeChild(self, node, int idx):
        """Delete a child node at the specified index"""
        if isinstance(node, ColorOcTreeNode):
            if (<ColorOcTreeNode>node).thisptr:
                self.thisptr.deleteNodeChild((<ColorOcTreeNode>node).thisptr, idx)
            else:
                raise NullPointerException
        else:
            raise TypeError("Expected ColorOcTreeNode")
    
    def getLabels(self, np.ndarray[DOUBLE_t, ndim=2] points):
        """
        Get occupancy labels for a set of points.
        Returns: -1 for unknown, 0 for free, 1 for occupied
        """
        cdef int i
        cdef np.ndarray[DOUBLE_t, ndim=1] pt
        # Runtime import to avoid circular dependency
        from .octomap import OcTreeKey
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
        """
        Extract point clouds for occupied and free voxels.
        Returns: (occupied_points, empty_points) as numpy arrays
        """
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
    
    # Helper method to get the C++ pointer address (for use in other modules)
    cpdef size_t _get_ptr_addr(self):
        return <size_t>self.thisptr
    
    def begin_tree(self, maxDepth=0):
        """Return a simplified tree iterator"""
        from .octree_iterators import SimpleTreeIterator
        return SimpleTreeIterator(self, maxDepth)
    
    def begin_leafs(self, maxDepth=0):
        """Return a simplified leaf iterator"""
        from .octree_iterators import SimpleLeafIterator
        return SimpleLeafIterator(self, maxDepth)
    
    def begin_leafs_bbx(self, np.ndarray[DOUBLE_t, ndim=1] bbx_min, np.ndarray[DOUBLE_t, ndim=1] bbx_max, maxDepth=0):
        """Return a simplified leaf iterator for a bounding box"""
        from .octree_iterators import SimpleLeafBBXIterator
        return SimpleLeafBBXIterator(self, bbx_min, bbx_max, maxDepth)
    
    def end_tree(self):
        """Return an end iterator for tree traversal"""
        from .octree_iterators import SimpleTreeIterator
        itr = SimpleTreeIterator(self)
        itr._set_end()
        return itr
    
    def end_leafs(self):
        """Return an end iterator for leaf traversal"""
        from .octree_iterators import SimpleLeafIterator
        itr = SimpleLeafIterator(self)
        itr._set_end()
        return itr
    
    def end_leafs_bbx(self):
        """Return an end iterator for leaf bounding box traversal"""
        from .octree_iterators import SimpleLeafBBXIterator
        itr = SimpleLeafBBXIterator(self, np.array([0.0, 0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0, 1.0], dtype=np.float64))
        itr._set_end()
        return itr

