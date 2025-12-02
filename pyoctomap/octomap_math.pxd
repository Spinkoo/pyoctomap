# Math type declarations for OctoMap wrapper

cimport octomap_std
from octomap_std cimport vector

cdef extern from "math/Vector3.h" namespace "octomath":
    cdef cppclass Vector3:
        Vector3() except +
        Vector3(float, float, float) except +
        Vector3(Vector3& other) except +
        float& x()
        float& y()
        float& z()

cdef extern from "octomap_types.h" namespace "octomap":
    ctypedef Vector3 point3d
    ctypedef vector[Vector3] point3d_collection

cdef extern from "math/Quaternion.h" namespace "octomath":
    cdef cppclass Quaternion:
        Quaternion() except +
        Quaternion(const Quaternion& other) except +
        Quaternion(float u, float x, float y, float z) except +
        Quaternion(double roll, double pitch, double yaw) except +

cdef extern from "math/Pose6D.h" namespace "octomath":
    cdef cppclass Pose6D:
        Pose6D() except +
        Pose6D(float x, float y, float z, double roll, double pitch, double yaw) except +
        Pose6D(const Vector3& trans, const Quaternion& rot) except +
        Pose6D(const Pose6D& other) except +
        Vector3 transform(const Vector3& v) const
        Pose6D inv() const

cdef extern from "octomap_types.h" namespace "octomap":
    ctypedef Pose6D pose6d

