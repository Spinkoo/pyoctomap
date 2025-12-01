# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libcpp.string cimport string
from libc.stddef cimport size_t
cimport octomap_defs as defs
import numpy as np
cimport numpy as np

# Fix NumPy API compatibility
np.import_array()

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
    
    def __getitem__(self, int i):
        # Handle negative indices (Python convention: -1 means last element)
        if i < 0:
            i = 3 + i
        if i < 0 or i >= 3:
            raise IndexError("OcTreeKey index out of range")
        return self.thisptr.k[i]
    
    def __setitem__(self, int i, unsigned int value):
        # Handle negative indices (Python convention: -1 means last element)
        if i < 0:
            i = 3 + i
        if i < 0 or i >= 3:
            raise IndexError("OcTreeKey index out of range")
        self.thisptr.k[i] = value
    
    def __len__(self):
        return 3
    
    def __iter__(self):
        """Make OcTreeKey iterable"""
        yield self.thisptr.k[0]
        yield self.thisptr.k[1]
        yield self.thisptr.k[2]
            
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
    # Note: thisptr is declared in octree_base.pxd, not here
    def __cinit__(self):
        pass
    def __dealloc__(self):
        pass
    
    # Helper method to get the C++ pointer address (for use in other modules)
    # Returns pointer as size_t so it can be called from Python code
    cpdef size_t _get_ptr_addr(self):
        return <size_t>self.thisptr
    
    # Helper method to set the C++ pointer (for use in other modules)
    # Takes pointer address as size_t so it can be called from Python code
    cpdef void _set_ptr(self, size_t ptr_addr):
        self.thisptr = <defs.OcTreeNode*>ptr_addr
    
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

