# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libcpp.string cimport string
from libcpp cimport bool as cppbool
from cython.operator cimport dereference as deref
cimport octomap_defs as defs
cimport pyoctomap.octree_base as octree_base
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t

# Fix NumPy API compatibility
np.import_array()

# Define NullPointerException locally (shared exception class)
class NullPointerException(Exception):
    """
    Null pointer exception
    """
    def __init__(self):
        pass

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
        self.thisptr.insertPointCloud(pc,
                                      defs.Vector3(<float>origin[0], <float>origin[1], <float>origin[2]),
                                      <double?>maxrange, bool(lazy_eval), bool(discretize))
    
    def isNodeOccupied(self, node):
        if isinstance(node, ColorOcTreeNode):
            if (<ColorOcTreeNode>node).thisptr:
                return self.thisptr.isNodeOccupied(deref((<ColorOcTreeNode>node).thisptr))
            else:
                raise NullPointerException
        else:
            raise TypeError(f"Expected ColorOcTreeNode, got {type(node)}")
    
    def isNodeAtThreshold(self, node):
        if isinstance(node, ColorOcTreeNode):
            if (<ColorOcTreeNode>node).thisptr:
                return self.thisptr.isNodeAtThreshold(deref((<ColorOcTreeNode>node).thisptr))
            else:
                raise NullPointerException
        else:
            raise TypeError(f"Expected ColorOcTreeNode, got {type(node)}")
    
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
        cdef string c_filename
        if isinstance(filename, (bytes, bytearray)):
            c_filename = (<bytes>filename).decode('utf-8')
        else:
            c_filename = (<str>filename).encode('utf-8')
        return self.thisptr.readBinary(c_filename)
    
    def writeBinary(self, filename):
        """
        Write file header and complete tree to binary file.
        Args:
            filename: Path to the output file (required)
        Returns:
            True if successful, False otherwise
        """
        cdef string c_filename
        if isinstance(filename, (bytes, bytearray)):
            c_filename = (<bytes>filename).decode('utf-8')
        else:
            c_filename = (<str>filename).encode('utf-8')
        return self.thisptr.writeBinary(c_filename)
    
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
    
    def createNodeChild(self, node, int idx):
        child = ColorOcTreeNode()
        child.thisptr = self.thisptr.createNodeChild((<ColorOcTreeNode>node).thisptr, idx)
        return child
    
    def deleteNodeChild(self, node, int idx):
        self.thisptr.deleteNodeChild((<ColorOcTreeNode>node).thisptr, idx)
    
    def expandNode(self, node):
        self.thisptr.expandNode((<ColorOcTreeNode>node).thisptr)
    
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

