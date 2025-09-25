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

# Import github2pypi for PyPI README conversion
try:
    from github2pypi import replace_url
    
    def get_long_description():
        """Get long description with GitHub URLs for PyPI compatibility."""
        with open("README.md", encoding="utf-8") as f:
            content = f.read()
        
        # Convert relative URLs to absolute GitHub URLs
        return replace_url(
            slug="Spinkoo/octomap2python", 
            content=content,
            branch="main"
        )
except ImportError:
    def get_long_description():
        """Fallback long description if github2pypi is not available."""
        with open("README.md", encoding="utf-8") as f:
            return f.read()


def get_version():
    """Get version from octomap/__init__.py without importing the module."""
    version_file = os.path.join(os.path.dirname(__file__), "octomap", "__init__.py")
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    except (FileNotFoundError, IOError) as e:
        print(f"Warning: Could not read version from {version_file}: {e}")
    return "0.0.0"


def get_lib_files():
    """Get the appropriate library files for the current platform"""
    lib_dir = "src/octomap/lib"
    
    if not os.path.exists(lib_dir):
        print(f"Warning: Library directory {lib_dir} not found")
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
    
    print(f"Found {len(lib_files)} library files: {lib_files}")
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
        package_dir = os.path.join(self.build_lib, "octomap")
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
                
                # Create versioned symlinks for .so files
                if lib_file.endswith('.so'):
                    lib_name = os.path.basename(lib_file)
                    # Create versioned symlinks (e.g., liboctomap.so.1.10 -> liboctomap.so)
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
    print(f"Using NumPy headers from: {numpy_include}")

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

    ext_modules = [
        Extension(
            "octomap.octomap",
            ["octomap/octomap.pyx"],
            include_dirs=[
                "octomap",  # Include the octomap directory for .pxd files
                "src/octomap/octomap/include",
                "src/octomap/octomap/include/octomap",
                "src/octomap/dynamicEDT3D/include",
                numpy_include,  # Use the variable instead of calling numpy.get_include() again
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
        include_path=["octomap"],
        compiler_directives={'language_level': 3}  # Ensure Python 3 syntax
    )


def main():
    """Main setup function - minimal since pyproject.toml handles metadata"""
    
    # Get library files for package data
    lib_files = get_lib_files()
    
    # Create package data structure
    package_data = {
        "octomap": [
            "lib/*",
            "lib/*.so*"  # Include versioned symlinks
        ],
        "": [
            "docs/*.md",
            "docs/**/*.md"
        ]
    }
    
    # Include library files in package
    data_files = []
    if lib_files:
        data_files.append(("octomap/lib", lib_files))

    # Build extensions
    ext_modules = build_extensions()

    setup(
        ext_modules=ext_modules,
        packages=["octomap"],
        package_data=package_data,
        data_files=data_files,
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        cmdclass={
            "build_ext": CustomBuildExt,
            "install": CustomInstall,
            "develop": CustomDevelop,
        },
        zip_safe=False,
    )


if __name__ == "__main__":
    main()