# Declaration file for octree_base module
# This allows cimporting OcTreeNode and OcTreeKey for type casting

from libc.stddef cimport size_t
cimport octomap_defs as defs

cdef class OcTreeNode:
    cdef defs.OcTreeNode *thisptr
    cpdef size_t _get_ptr_addr(self)
    cpdef void _set_ptr(self, size_t ptr_addr)

cdef class OcTreeKey:
    cdef defs.OcTreeKey thisptr

