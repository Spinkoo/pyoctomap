"""
Simplified setup.py that focuses only on custom build logic.
Metadata is now handled by pyproject.toml to avoid duplication.
"""

import sys
import os
import shutil
import platform
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop


def get_lib_files():
    """Get the appropriate library files for the current platform"""
    lib_dir = "src/octomap/lib"
    
    if not os.path.exists(lib_dir):
        return []
    
    lib_files = []
    
    # Get platform-specific library extensions
    if platform.system() == "Windows":
        lib_extensions = [".dll", ".lib"]
    elif platform.system() == "Darwin":  # macOS
        lib_extensions = [".dylib", ".a"]
    else:  # Linux and others
        lib_extensions = [".so", ".a"]
    
    # Find all library files (including versioned ones like .so.1.10, .so.1.10.0)
    for file in os.listdir(lib_dir):
        file_path = os.path.join(lib_dir, file)
        # Include files that match extensions or contain versioned patterns
        # Skip symlinks - we'll copy the actual files they point to
        if not os.path.islink(file_path):
            if (any(file.endswith(ext) for ext in lib_extensions) or 
                ('.so.' in file and platform.system() != "Windows")):
                lib_files.append(file_path)
    
    return lib_files


def copy_libraries_to_directory(lib_package_dir):
    """Copy libraries to a target directory, preserving symlink structure"""
    lib_dir = "src/octomap/lib"
    
    if not os.path.exists(lib_dir):
        return
    
    os.makedirs(lib_package_dir, exist_ok=True)
    
    # First, copy all actual files (not symlinks) - these are the .so.1.10.0 files
    actual_files = {}
    for file in os.listdir(lib_dir):
        lib_file = os.path.join(lib_dir, file)
        if os.path.isfile(lib_file) and not os.path.islink(lib_file):
            if file.endswith('.so') or file.endswith('.a') or '.so.' in file:
                dest_file = os.path.join(lib_package_dir, file)
                shutil.copy2(lib_file, dest_file)
                actual_files[file] = dest_file
    
    # Then, resolve and copy symlinks by copying their targets with the symlink name
    for file in os.listdir(lib_dir):
        lib_file = os.path.join(lib_dir, file)
        if os.path.islink(lib_file):
            target = os.readlink(lib_file)
            if os.path.isabs(target):
                target_name = os.path.basename(target)
                target_path = target
            else:
                target_name = target
                target_path = os.path.join(os.path.dirname(lib_file), target)
            
            # Resolve the symlink chain to find the actual file
            while os.path.islink(target_path):
                next_target = os.readlink(target_path)
                if os.path.isabs(next_target):
                    target_path = next_target
                else:
                    target_path = os.path.join(os.path.dirname(target_path), next_target)
            
            # Copy the actual file with the symlink's name
            if os.path.exists(target_path):
                dest_file = os.path.join(lib_package_dir, file)
                shutil.copy2(target_path, dest_file)


def copy_libraries_to_source():
    """Copy libraries to source pyoctomap/lib/ directory before build"""
    lib_package_dir = os.path.join("pyoctomap", "lib")
    copy_libraries_to_directory(lib_package_dir)


class CustomBuildExt(build_ext):
    """Custom build extension that copies libraries to the package"""
    
    def run(self):
        # Copy libraries to source directory first (for MANIFEST.in)
        copy_libraries_to_source()
        
        # Run the normal build
        super().run()
        
        # Copy libraries to the build directory
        self.copy_libraries()
    
    def copy_libraries(self):
        """Copy shared libraries to the build package directory"""
        package_dir = os.path.join(self.build_lib, "pyoctomap")
        lib_package_dir = os.path.join(package_dir, "lib")
        copy_libraries_to_directory(lib_package_dir)


class CustomInstall(install):
    """Custom install that sets up library paths"""
    
    def run(self):
        super().run()
        # Copy libraries to installed package
        self.copy_libraries_to_installed()
    
    def copy_libraries_to_installed(self):
        """Copy libraries to the installed package directory"""
        install_lib = self.install_lib
        package_dir = os.path.join(install_lib, "pyoctomap")
        lib_package_dir = os.path.join(package_dir, "lib")
        copy_libraries_to_directory(lib_package_dir)


class CustomDevelop(develop):
    """Custom develop install that sets up library paths"""
    
    def run(self):
        super().run()
        # Copy libraries to development package
        self.copy_libraries_to_installed()
    
    def copy_libraries_to_installed(self):
        """Copy libraries to the development package directory"""
        package_dir = "pyoctomap"
        lib_package_dir = os.path.join(package_dir, "lib")
        copy_libraries_to_directory(lib_package_dir)


def build_extensions():
    """Build the Cython extensions with proper configuration"""
    
    # Import required modules - these should be available as build dependencies
    try:
        import numpy
        from Cython.Build import cythonize
    except ImportError as e:
        print(f"Error: Required build dependency not found: {e}")
        print("Please install build dependencies with: pip install numpy cython")
        sys.exit(1)
    
    # Get numpy include directory at build time (not install time)
    numpy_include = numpy.get_include()

    # Compiler flags for better memory management and debugging
    extra_compile_args = []
    extra_link_args = []
    rpath_args = []
    
    if platform.system() == "Windows":
        extra_compile_args = ["/O2", "/DNDEBUG", "/wd4996"]  # Suppress deprecation warnings
        extra_link_args = []
    else:
        extra_compile_args = [
            "-O2", "-DNDEBUG", "-fPIC",
            "-Wno-deprecated-declarations",  # Suppress deprecation warnings
            "-Wno-deprecated",               # Suppress all deprecated warnings
            "-Wno-unused-function"           # Suppress unused function warnings
        ]
        extra_link_args = ["-fPIC"]
        # Ensure extension finds bundled libs at runtime without LD_LIBRARY_PATH
        if platform.system() == "Linux":
            rpath_args = ["-Wl,-rpath,$ORIGIN/lib"]
        elif platform.system() == "Darwin":
            rpath_args = ["-Wl,-rpath,@loader_path/lib"]

    # Find all .pyx files
    pyx_files = {
        "pyoctomap.octree_base": None,
        "pyoctomap.octree_iterators": None,
        "pyoctomap.octree": None,
        "pyoctomap.octomap": None,
        "pyoctomap.color_octree": None,
        "pyoctomap.counting_octree": None,
        "pyoctomap.stamped_octree": None,
        "pyoctomap.pointcloud": None,
    }
    
    possible_paths = {
        "pyoctomap.octree_base": ["pyoctomap/octree_base.pyx"],
        "pyoctomap.octree_iterators": ["pyoctomap/octree_iterators.pyx"],
        "pyoctomap.octree": ["pyoctomap/octree.pyx"],
        "pyoctomap.octomap": ["pyoctomap/octomap.pyx"],
        "pyoctomap.color_octree": ["pyoctomap/color_octree.pyx"],
        "pyoctomap.counting_octree": ["pyoctomap/counting_octree.pyx"],
        "pyoctomap.stamped_octree": ["pyoctomap/stamped_octree.pyx"],
        "pyoctomap.pointcloud": ["pyoctomap/pointcloud.pyx"],
    }
    
    for module_name, paths in possible_paths.items():
        for path in paths:
            if os.path.exists(path):
                pyx_files[module_name] = path
                break
    
    # Common extension configuration
    common_include_dirs = [
        "pyoctomap",
        "src/octomap/octomap/include",
        "src/octomap/octomap/include/octomap",
        "src/octomap/dynamicEDT3D/include",
        numpy_include,
    ]
    
    common_library_dirs = ["src/octomap/lib"]
    
    common_libraries = ["dynamicedt3d", "octomap", "octomath"]
    
    common_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    
    ext_modules = []
    
    # Build octree_base extension
    if pyx_files["pyoctomap.octree_base"]:
        ext_modules.append(
            Extension(
                "pyoctomap.octree_base",
                [pyx_files["pyoctomap.octree_base"]],
                include_dirs=common_include_dirs,
                library_dirs=common_library_dirs,
                libraries=common_libraries,
                define_macros=common_macros,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args + rpath_args,
            )
        )
    
    # Build octree_iterators extension
    if pyx_files["pyoctomap.octree_iterators"]:
        ext_modules.append(
            Extension(
                "pyoctomap.octree_iterators",
                [pyx_files["pyoctomap.octree_iterators"]],
                include_dirs=common_include_dirs,
                library_dirs=common_library_dirs,
                libraries=common_libraries,
                define_macros=common_macros,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args + rpath_args,
            )
        )
    
    # Build octree extension
    if pyx_files["pyoctomap.octree"]:
        ext_modules.append(
            Extension(
                "pyoctomap.octree",
                [pyx_files["pyoctomap.octree"]],
                include_dirs=common_include_dirs,
                library_dirs=common_library_dirs,
                libraries=common_libraries,
                define_macros=common_macros,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args + rpath_args,
            )
        )
    
    # Build octomap wrapper extension
    if pyx_files["pyoctomap.octomap"]:
        ext_modules.append(
            Extension(
                "pyoctomap.octomap",
                [pyx_files["pyoctomap.octomap"]],
                include_dirs=common_include_dirs,
                library_dirs=common_library_dirs,
                libraries=common_libraries,
                define_macros=common_macros,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args + rpath_args,
            )
        )
    
    # Build color_octree extension
    if pyx_files["pyoctomap.color_octree"]:
        ext_modules.append(
            Extension(
                "pyoctomap.color_octree",
                [pyx_files["pyoctomap.color_octree"]],
                include_dirs=common_include_dirs,
                library_dirs=common_library_dirs,
                libraries=common_libraries,
                define_macros=common_macros,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args + rpath_args,
            )
        )
    
    # Build counting_octree extension
    if pyx_files["pyoctomap.counting_octree"]:
        ext_modules.append(
            Extension(
                "pyoctomap.counting_octree",
                [pyx_files["pyoctomap.counting_octree"]],
                include_dirs=common_include_dirs,
                library_dirs=common_library_dirs,
                libraries=common_libraries,
                define_macros=common_macros,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args + rpath_args,
            )
        )
    
    # Build stamped_octree extension
    if pyx_files["pyoctomap.stamped_octree"]:
        ext_modules.append(
            Extension(
                "pyoctomap.stamped_octree",
                [pyx_files["pyoctomap.stamped_octree"]],
                include_dirs=common_include_dirs,
                library_dirs=common_library_dirs,
                libraries=common_libraries,
                define_macros=common_macros,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args + rpath_args,
            )
        )
    
    # Build pointcloud extension
    if pyx_files["pyoctomap.pointcloud"]:
        ext_modules.append(
            Extension(
                "pyoctomap.pointcloud",
                [pyx_files["pyoctomap.pointcloud"]],
                include_dirs=common_include_dirs,
                library_dirs=common_library_dirs,
                libraries=common_libraries,
                define_macros=common_macros,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args + rpath_args,
            )
        )
    
    return cythonize(
        ext_modules, 
        include_path=["pyoctomap"],
        compiler_directives={'language_level': 3}  # Ensure Python 3 syntax
    )


def main():
    """Main setup function - minimal since pyproject.toml handles metadata"""
    
    # Build extensions
    ext_modules = build_extensions()

    setup(
        # Metadata comes from pyproject.toml
        ext_modules=ext_modules,
        
        # Build configuration
        cmdclass={
            "build_ext": CustomBuildExt,
            "install": CustomInstall,
            "develop": CustomDevelop,
        },
        zip_safe=False,
    )


if __name__ == "__main__":
    main()