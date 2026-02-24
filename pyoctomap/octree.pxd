# Declaration file for octree module
# This allows cimporting OcTree for type casting

from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
cimport octomap_defs as defs
cimport dynamicEDT3D_defs as edt
cimport numpy as np
ctypedef np.float64_t DOUBLE_t

cdef class OcTree:
    cdef defs.OcTree *thisptr
    cdef edt.DynamicEDTOctomap *edtptr
    cdef bint owner
    
    # Helper method to get the C++ pointer (for use in other modules)
    # Using cpdef so it can be called from Python code, but returns pointer address as size_t
    cpdef size_t _get_ptr_addr(self)
    
    # cdef method declarations

    cdef void _build_pointcloud_and_insert(self, np.ndarray[DOUBLE_t, ndim=2] point_cloud, np.ndarray[DOUBLE_t, ndim=1] sensor_origin, double max_range, bint discretize, bint lazy_eval)

