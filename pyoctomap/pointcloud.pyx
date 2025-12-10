# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libcpp.string cimport string
from libcpp cimport bool as cppbool
from cython.operator cimport dereference as deref
cimport octomap_defs as defs
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t

# Fix NumPy API compatibility
np.import_array()

# Import NullPointerException from octree_base
from .octree_base import NullPointerException

# Pointcloud wrapper class
cdef class Pointcloud:
    """
    A collection of 3D coordinates (point3d), which are regarded as endpoints of a 3D laser scan.
    """
    cdef defs.Pointcloud *thisptr
    cdef bint owner
    
    def __cinit__(self, arg=None):
        """
        Initialize Pointcloud.
        Args:
            arg: Optional. Can be:
                - None: Create empty pointcloud
                - Pointcloud: Copy from another pointcloud
                - numpy array (Nx3): Initialize from numpy array of points
        """
        self.owner = True
        if arg is None:
            self.thisptr = new defs.Pointcloud()
        elif isinstance(arg, Pointcloud):
            # Copy constructor
            self.thisptr = new defs.Pointcloud((<Pointcloud>arg).thisptr)
        elif isinstance(arg, np.ndarray):
            # Initialize from numpy array
            arr = np.asarray(arg, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError("Expected numpy array of shape (N, 3)")
            self.thisptr = new defs.Pointcloud()
            for i in range(arr.shape[0]):
                self.thisptr.push_back(<float>arr[i, 0], <float>arr[i, 1], <float>arr[i, 2])
        else:
            raise TypeError(f"Unsupported argument type: {type(arg)}")
    
    def __dealloc__(self):
        if self.owner and self.thisptr != NULL:
            del self.thisptr
            self.thisptr = NULL
    
    def __len__(self):
        """Return the number of points in the pointcloud."""
        return self.thisptr.size()
    
    def __getitem__(self, i):
        """
        Get point at index i.
        Supports negative indices (Python convention: -1 is last element).
        Returns: numpy array [x, y, z]
        """
        cdef size_t idx
        cdef size_t size = self.thisptr.size()
        
        # Handle negative indices (Python convention)
        if i < 0:
            # Check bounds before converting to size_t to avoid overflow
            if abs(i) > size:
                raise IndexError(f"Index {i} out of range for pointcloud of size {size}")
            idx = size + i  # Now safe: size + i >= 0
        else:
            idx = i
            if idx >= size:
                raise IndexError(f"Index {i} out of range for pointcloud of size {size}")
        
        cdef defs.point3d p = (deref(self.thisptr))[idx]
        return np.array([p.x(), p.y(), p.z()], dtype=np.float64)
    
    def __setitem__(self, i, value):
        """
        Set point at index i.
        Supports negative indices (Python convention: -1 is last element).
        Args:
            i: Index (can be negative)
            value: numpy array [x, y, z] or tuple/list of 3 floats
        """
        cdef size_t idx
        cdef size_t size = self.thisptr.size()
        
        # Handle negative indices (Python convention)
        if i < 0:
            # Check bounds before converting to size_t to avoid overflow
            if abs(i) > size:
                raise IndexError(f"Index {i} out of range for pointcloud of size {size}")
            idx = size + i  # Now safe: size + i >= 0
        else:
            idx = i
            if idx >= size:
                raise IndexError(f"Index {i} out of range for pointcloud of size {size}")
        
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError("Expected array of shape (3,)")
        # Assign directly to operator[] result (which returns a reference)
        (deref(self.thisptr))[idx] = defs.point3d(<float>arr[0], <float>arr[1], <float>arr[2])
    
    def push_back(self, x, y=None, z=None):
        """
        Add a point to the pointcloud.
        Args:
            x: Either x coordinate (if y, z provided) or numpy array [x, y, z]
            y: y coordinate (optional)
            z: z coordinate (optional)
        """
        if y is not None and z is not None:
            # Three separate arguments
            self.thisptr.push_back(<float?>x, <float?>y, <float?>z)
        else:
            # Single argument - assume it's a coordinate array
            arr = np.asarray(x, dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError("Expected array of shape (3,) or three separate coordinates")
            self.thisptr.push_back(<float>arr[0], <float>arr[1], <float>arr[2])
    
    def size(self):
        """
        Get the number of points in the pointcloud.
        Returns: size_t
        """
        return self.thisptr.size()
    
    def clear(self):
        """Clear all points from the pointcloud."""
        self.thisptr.clear()
    
    def reserve(self, size_t size):
        """
        Reserve memory for a given number of points.
        Args:
            size: Number of points to reserve space for
        """
        self.thisptr.reserve(size)
    
    def getPoint(self, unsigned int i):
        """
        Get a copy of the ith point in point cloud.
        Args:
            i: Index of the point
        Returns: numpy array [x, y, z]
        """
        cdef defs.point3d p = self.thisptr.getPoint(i)
        return np.array([p.x(), p.y(), p.z()], dtype=np.float64)
    
    def back(self):
        """
        Get the last point in the pointcloud.
        Returns: numpy array [x, y, z]
        Raises: IndexError if pointcloud is empty
        """
        if self.thisptr.size() == 0:
            raise IndexError("Cannot call back() on empty pointcloud")
        cdef defs.point3d p = self.thisptr.back()
        return np.array([p.x(), p.y(), p.z()], dtype=np.float64)
    
    def transform(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None):
        """
        Apply transform to each point.
        Args:
            Either provide:
            - x, y, z, roll, pitch, yaw: Translation and rotation (Euler angles in radians)
            Or provide a single argument:
            - transform: tuple/list of 6 values [x, y, z, roll, pitch, yaw]
        """
        cdef defs.pose6d transform_pose
        if roll is not None and pitch is not None and yaw is not None:
            # Six separate arguments - cast Pose6D to pose6d typedef
            transform_pose = <defs.pose6d>defs.Pose6D(<float?>x, <float?>y, <float?>z, 
                                         <double?>roll, <double?>pitch, <double?>yaw)
        elif x is not None:
            # Single argument - assume it's a list/tuple of 6 values
            arr = np.asarray(x, dtype=np.float64)
            if arr.shape != (6,):
                raise ValueError("Expected array of shape (6,) [x, y, z, roll, pitch, yaw] or six separate arguments")
            transform_pose = <defs.pose6d>defs.Pose6D(<float>arr[0], <float>arr[1], <float>arr[2],
                                        <double>arr[3], <double>arr[4], <double>arr[5])
        else:
            raise ValueError("Must provide either six arguments (x, y, z, roll, pitch, yaw) or a single array of 6 values")
        self.thisptr.transform(transform_pose)
    
    def transformAbsolute(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None):
        """
        Apply transform to each point, undoing previous transforms.
        Args:
            Either provide:
            - x, y, z, roll, pitch, yaw: Translation and rotation (Euler angles in radians)
            Or provide a single argument:
            - transform: tuple/list of 6 values [x, y, z, roll, pitch, yaw]
        """
        cdef defs.pose6d transform_pose
        if roll is not None and pitch is not None and yaw is not None:
            # Six separate arguments - cast Pose6D to pose6d typedef
            transform_pose = <defs.pose6d>defs.Pose6D(<float?>x, <float?>y, <float?>z,
                                         <double?>roll, <double?>pitch, <double?>yaw)
        elif x is not None:
            # Single argument - assume it's a list/tuple of 6 values
            arr = np.asarray(x, dtype=np.float64)
            if arr.shape != (6,):
                raise ValueError("Expected array of shape (6,) [x, y, z, roll, pitch, yaw] or six separate arguments")
            transform_pose = <defs.pose6d>defs.Pose6D(<float>arr[0], <float>arr[1], <float>arr[2],
                                        <double>arr[3], <double>arr[4], <double>arr[5])
        else:
            raise ValueError("Must provide either six arguments (x, y, z, roll, pitch, yaw) or a single array of 6 values")
        self.thisptr.transformAbsolute(transform_pose)
    
    def rotate(self, double roll, double pitch, double yaw):
        """
        Rotate each point in pointcloud.
        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
        """
        self.thisptr.rotate(roll, pitch, yaw)
    
    def calcBBX(self):
        """
        Calculate bounding box of Pointcloud.
        Returns: tuple (lowerBound, upperBound) where each is numpy array [x, y, z]
        """
        cdef defs.point3d lowerBound = defs.point3d()
        cdef defs.point3d upperBound = defs.point3d()
        self.thisptr.calcBBX(lowerBound, upperBound)
        return (np.array([lowerBound.x(), lowerBound.y(), lowerBound.z()], dtype=np.float64),
                np.array([upperBound.x(), upperBound.y(), upperBound.z()], dtype=np.float64))
    
    def crop(self, lowerBound, upperBound):
        """
        Crop Pointcloud to given bounding box.
        Args:
            lowerBound: numpy array [x, y, z] or tuple/list of 3 floats
            upperBound: numpy array [x, y, z] or tuple/list of 3 floats
        """
        lower_arr = np.asarray(lowerBound, dtype=np.float64)
        upper_arr = np.asarray(upperBound, dtype=np.float64)
        if lower_arr.shape != (3,) or upper_arr.shape != (3,):
            raise ValueError("Expected arrays of shape (3,)")
        # Use Vector3 constructor directly instead of assignment
        cdef defs.point3d lower = <defs.point3d>defs.Vector3(<float>lower_arr[0], <float>lower_arr[1], <float>lower_arr[2])
        cdef defs.point3d upper = <defs.point3d>defs.Vector3(<float>upper_arr[0], <float>upper_arr[1], <float>upper_arr[2])
        self.thisptr.crop(lower, upper)
    
    def minDist(self, double thres):
        """
        Remove any points closer than [thres] to (0,0,0).
        Args:
            thres: Minimum distance threshold
        """
        self.thisptr.minDist(thres)
    
    def subSampleRandom(self, unsigned int num_samples):
        """
        Randomly subsample the pointcloud.
        Args:
            num_samples: Number of samples to extract
        Returns: New Pointcloud with sampled points
        """
        cdef defs.Pointcloud sample_cloud = defs.Pointcloud()
        self.thisptr.subSampleRandom(num_samples, sample_cloud)
        cdef defs.point3d p
        result = Pointcloud()
        # Copy points from sample_cloud to result
        cdef size_t i
        for i in range(sample_cloud.size()):
            p = sample_cloud[i]
            result.thisptr.push_back(p.x(), p.y(), p.z())
        return result
    
    def to_numpy(self):
        """
        Convert pointcloud to numpy array.
        Returns: numpy array of shape (N, 3) where N is the number of points
        """
        cdef size_t n = self.thisptr.size()
        cdef np.ndarray[DOUBLE_t, ndim=2] result = np.zeros((n, 3), dtype=np.float64)
        cdef size_t i
        cdef defs.point3d p
        for i in range(n):
            p = (deref(self.thisptr))[i]
            result[i, 0] = p.x()
            result[i, 1] = p.y()
            result[i, 2] = p.z()
        return result
    
    def writeVrml(self, filename):
        """
        Export the Pointcloud to a VRML file.
        Args:
            filename: Output filename for VRML file
        """
        cdef string c_filename
        if isinstance(filename, (bytes, bytearray)):
            c_filename = (<bytes>filename).decode('utf-8')
        else:
            c_filename = (<str>filename).encode('utf-8')
        self.thisptr.writeVrml(c_filename)
    
    def readBinary(self, filename):
        """
        Read pointcloud from binary file.
        Args:
            filename: Input filename
        Returns: True if successful, False otherwise
        """
        cdef string c_filename
        if isinstance(filename, (bytes, bytearray)):
            c_filename = (<bytes>filename).decode('utf-8')
        else:
            c_filename = (<str>filename).encode('utf-8')
        cdef defs.ifstream ifs = defs.ifstream(c_filename.c_str())
        if not ifs.is_open():
            raise IOError(f"Failed to open file: {filename}")
        try:
            # Cast ifstream to istream& (ifstream inherits from istream)
            self.thisptr.readBinary(<defs.istream&>ifs)
            return True
        finally:
            ifs.close()
    
    def writeBinary(self, filename):
        """
        Write pointcloud to binary file.
        Args:
            filename: Output filename
        Returns: True if successful, False otherwise
        """
        cdef string c_filename
        if isinstance(filename, (bytes, bytearray)):
            c_filename = (<bytes>filename).decode('utf-8')
        else:
            c_filename = (<str>filename).encode('utf-8')
        cdef defs.ofstream ofs = defs.ofstream(c_filename.c_str())
        if not ofs.is_open():
            raise IOError(f"Failed to open file for writing: {filename}")
        try:
            # Cast ofstream to ostream& (ofstream inherits from ostream)
            self.thisptr.writeBinary(<defs.ostream&>ofs)
            return True
        finally:
            ofs.close()
    
    def to_open3d(self):
        """
        Convert pointcloud to Open3D PointCloud object.
        Requires Open3D to be installed.
        Returns: open3d.geometry.PointCloud object
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D is required for to_open3d(). Install it with: pip install open3d")
        
        # Convert to numpy array
        points = self.to_numpy()
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
    
    @staticmethod
    def from_open3d(pcd):
        """
        Create Pointcloud from Open3D PointCloud object.
        Args:
            pcd: open3d.geometry.PointCloud object
        Returns: Pointcloud instance
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D is required for from_open3d(). Install it with: pip install open3d")
        
        # Extract points from Open3D point cloud
        points = np.asarray(pcd.points, dtype=np.float64)
        
        # Create Pointcloud from numpy array
        return Pointcloud(points)
    
    def __repr__(self):
        return f"Pointcloud(size={self.thisptr.size()})"
    
    def __str__(self):
        return f"Pointcloud with {self.thisptr.size()} points"

