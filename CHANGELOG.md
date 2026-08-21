# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-02

### Added
- Initial stable release of PyOctoMap
- Complete Python bindings for OctoMap C++ library
- Support for multiple octree variants:
  - `OcTree`: Standard probabilistic occupancy tree
  - `ColorOcTree`: Occupancy with RGB color support
  - `CountingOcTree`: Integer hit counters per voxel
  - `OcTreeStamped`: Occupancy with per-node timestamps
- Bundled shared libraries for easy deployment (manylinux wheels)
- Comprehensive documentation:
  - API Reference
  - Build System Guide
  - Performance Guide
  - Troubleshooting Guide
  - File Format Guide
  - Wheel Technology Documentation
- Dynamic mapping support with `decayAndInsertPointCloud()` for temporal decay
- Batch operations for efficient point cloud processing:
  - `insertPointCloud()` with lazy evaluation
  - `insertPointCloudRaysFast()` for maximum performance
- Iterator support for tree traversal:
  - Tree iterator (all nodes)
  - Leaf iterator (leaf nodes only)
  - Bounding box iterator (spatial filtering)
- Ray casting for collision detection and path planning
- File I/O support for `.bt` and `.ot` formats
- Docker-based build system for cross-platform wheel generation
- Example scripts demonstrating core functionality
- Comprehensive test suite
- Python 3.8+ support (wheels for 3.9-3.14)
- NumPy integration for efficient array operations
- Open3D visualization examples

### Changed
- Modernized build system using `pyproject.toml` as single source of truth
- Simplified `setup.py` to focus only on build logic
- Consolidated configuration files
- Improved library path handling with dynamic Python version detection
- Updated documentation with absolute GitHub URLs for PyPI compatibility

### Fixed
- Hardcoded Python 3.12 path now uses dynamic version detection
- Removed duplicate pytest configuration
- Removed dead code from package initialization
- Fixed garbled text in example files
- Corrected copyright information in LICENSE
- Cleaned up redundant metadata definitions

### Deprecated
- None

### Removed
- Unused library setup functions (handled via rpath)
- Duplicate configuration files
- Redundant metadata from setup.py

### Security
- None

## [Unreleased]

### Planned
- ROS/ROS2 integration (in development on `ros` branch)
- Additional visualization tools
- Performance optimizations for large-scale maps
- Extended documentation and tutorials

---

## Version History

- **1.0.0** (2025-12-02): Initial stable release

[1.0.0]: https://github.com/Spinkoo/pyoctomap/releases/tag/v1.0.0

