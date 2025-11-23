"""
Setup script for pyoctomap.
Most metadata is configured in pyproject.toml.
This script handles the compilation of Cython extensions and bundling of C++ libraries.
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
        print(f"Warning: Library directory {lib_dir} not found. Run compile_cpp_libs.sh first.")
        return []
    
    lib_files = []
    
    # Get platform-specific library extensions
    if platform.system() == "Windows":
        lib_extensions = [".dll", ".lib"]
    elif platform.system() == "Darwin":  # macOS
        lib_extensions = [".dylib", ".a"]
    else:  # Linux and others
        lib_extensions = [".so", ".a"]
    
    # Find all library files
    for file in os.listdir(lib_dir):
        if any(file.endswith(ext) for ext in lib_extensions):
            lib_files.append(os.path.join(lib_dir, file))
    
    return lib_files


class CustomBuildExt(build_ext):
    """Custom build extension that copies libraries to the package"""
    
    def run(self):
        # Run the normal build
        super().run()
        
        # Copy libraries to the package directory
        self.copy_libraries()
    
    def copy_libraries(self):
        """Copy shared libraries to the package directory and create versioned symlinks"""
        lib_files = get_lib_files()
        
        if not lib_files:
            print("No library files found to copy")
            return
        
        # Get the package directory
        package_dir = os.path.join(self.build_lib, "pyoctomap")
        os.makedirs(package_dir, exist_ok=True)
        
        # Create lib subdirectory in package
        lib_package_dir = os.path.join(package_dir, "lib")
        os.makedirs(lib_package_dir, exist_ok=True)
        
        # Copy library files and create versioned symlinks
        for lib_file in lib_files:
            if os.path.exists(lib_file):
                dest_file = os.path.join(lib_package_dir, os.path.basename(lib_file))
                shutil.copy2(lib_file, dest_file)
                print(f"Copied {lib_file} -> {dest_file}")
                
                # Create versioned symlinks for .so files (Linux)
                if lib_file.endswith('.so'):
                    lib_name = os.path.basename(lib_file)
                    versioned_names = []
                    
                    if 'liboctomap.so' in lib_name and not lib_name.endswith('.1.10.0'):
                        versioned_names = ['liboctomap.so.1.10', 'liboctomap.so.1.10.0']
                    elif 'libdynamicedt3d.so' in lib_name and not lib_name.endswith('.1.10.0'):
                        versioned_names = ['libdynamicedt3d.so.1.10', 'libdynamicedt3d.so.1.10.0']
                    elif 'liboctomath.so' in lib_name and not lib_name.endswith('.1.10.0'):
                        versioned_names = ['liboctomath.so.1.10', 'liboctomath.so.1.10.0']
                    
                    for versioned_name in versioned_names:
                        versioned_path = os.path.join(lib_package_dir, versioned_name)
                        if not os.path.exists(versioned_path):
                            try:
                                os.symlink(lib_name, versioned_path)
                                print(f"Created symlink {versioned_name} -> {lib_name}")
                            except OSError as e:
                                print(f"Failed to create symlink {versioned_name}: {e}")


class CustomInstall(install):
    """Custom install that sets up library paths"""
    def run(self):
        super().run()


class CustomDevelop(develop):
    """Custom develop install that sets up library paths"""
    def run(self):
        super().run()


def build_extensions():
    """Build the Cython extensions with proper configuration"""
    
    try:
        import numpy
        from Cython.Build import cythonize
    except ImportError as e:
        print(f"Error: Required build dependency not found: {e}")
        print("Please install build dependencies with: pip install numpy cython")
        sys.exit(1)
    
    numpy_include = numpy.get_include()
    
    extra_compile_args = []
    extra_link_args = []
    rpath_args = []
    
    if platform.system() == "Windows":
        extra_compile_args = ["/O2", "/DNDEBUG", "/wd4996"]
    else:
        extra_compile_args = [
            "-O2", "-DNDEBUG", "-fPIC",
            "-Wno-deprecated-declarations",
            "-Wno-deprecated",
            "-Wno-unused-function"
        ]
        extra_link_args = ["-fPIC"]
        if platform.system() == "Linux":
            rpath_args = ["-Wl,-rpath,$ORIGIN/lib"]
        elif platform.system() == "Darwin":
            rpath_args = ["-Wl,-rpath,@loader_path/lib"]

    # Locate .pyx file
    pyx_file = "pyoctomap/octomap.pyx"
    if not os.path.exists(pyx_file):
        print(f"Error: Could not find {pyx_file}")
        sys.exit(1)
    
    ext_modules = [
        Extension(
            "pyoctomap.octomap",
            [pyx_file],
            include_dirs=[
                "pyoctomap",
                "src/octomap/octomap/include",
                "src/octomap/octomap/include/octomap",
                "src/octomap/dynamicEDT3D/include",
                numpy_include,
            ],
            library_dirs=[
                "src/octomap/lib",
            ],
            libraries=[
                "dynamicedt3d",
                "octomap",
                "octomath",
            ],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args + rpath_args,
        )
    ]
    
    return cythonize(
        ext_modules, 
        include_path=["pyoctomap"],
        compiler_directives={'language_level': 3}
    )


def main():
    # Build extensions
    ext_modules = build_extensions()

    setup(
        ext_modules=ext_modules,
        cmdclass={
            "build_ext": CustomBuildExt,
            "install": CustomInstall,
            "develop": CustomDevelop,
        },
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
