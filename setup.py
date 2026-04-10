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

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# Ensure all relative paths in setup/egg_info/cythonize are rooted at project.
os.chdir(ROOT_DIR)


def copy_libraries_to_directory(lib_package_dir):
    """Copy libraries to a target directory, preserving symlink structure"""
    lib_dir = "src/octomap/lib"
    
    if not os.path.exists(lib_dir):
        return
    
    os.makedirs(lib_package_dir, exist_ok=True)
    
    # First, copy all actual files (not symlinks)
    for file in os.listdir(lib_dir):
        lib_file = os.path.join(lib_dir, file)
        if os.path.isfile(lib_file) and not os.path.islink(lib_file):
            if platform.system() == "Windows":
                copy_this = file.endswith(".dll") or file.endswith(".lib")
            elif platform.system() == "Darwin":
                copy_this = file.endswith(".dylib") or file.endswith(".a")
            else:
                copy_this = file.endswith(".so") or file.endswith(".a") or ".so." in file
            if copy_this:
                dest_file = os.path.join(lib_package_dir, file)
                shutil.copy2(lib_file, dest_file)
    
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
        prev_cwd = os.getcwd()
        os.chdir(ROOT_DIR)
        try:
            # Copy libraries to source directory first (for MANIFEST.in)
            copy_libraries_to_source()
            
            # Run the normal build
            super().run()
            
            # Copy libraries to the build directory
            self.copy_libraries()
        finally:
            os.chdir(prev_cwd)
    
    def copy_libraries(self):
        """Copy shared libraries to the build package directory"""
        package_dir = os.path.join(self.build_lib, "pyoctomap")
        lib_package_dir = os.path.join(package_dir, "lib")
        copy_libraries_to_directory(lib_package_dir)

    def get_source_files(self):
        """
        setuptools egg_info/manifest generation calls this and rejects absolute
        project-local paths. Normalize to repo-relative here as a final guard.
        """
        files = super().get_source_files()
        out = []
        for f in files:
            try:
                p = os.fspath(f)
            except TypeError:
                out.append(f)
                continue
            if isinstance(p, str) and os.path.isabs(p):
                try:
                    common = os.path.commonpath([ROOT_DIR, p])
                except ValueError:
                    common = ""
                if os.path.normcase(common) == os.path.normcase(ROOT_DIR):
                    p = os.path.relpath(p, ROOT_DIR)
            out.append(p.replace("\\", "/") if isinstance(p, str) else p)
        return out


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
            "-std=c++14",                    # Required for std::move in Cython-generated code (macOS clang defaults to C++98)
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

    module_to_pyx = {
        "pyoctomap.octree_base": "pyoctomap/octree_base.pyx",
        "pyoctomap.octree_iterators": "pyoctomap/octree_iterators.pyx",
        "pyoctomap.octree": "pyoctomap/octree.pyx",
        "pyoctomap.octomap": "pyoctomap/octomap.pyx",
        "pyoctomap.color_octree": "pyoctomap/color_octree.pyx",
        "pyoctomap.counting_octree": "pyoctomap/counting_octree.pyx",
        "pyoctomap.stamped_octree": "pyoctomap/stamped_octree.pyx",
        "pyoctomap.pointcloud": "pyoctomap/pointcloud.pyx",
    }
    
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
    for module_name, pyx_path in module_to_pyx.items():
        if not os.path.exists(pyx_path):
            continue
        ext_modules.append(
            Extension(
                module_name,
                [pyx_path],
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