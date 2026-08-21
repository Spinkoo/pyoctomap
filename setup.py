"""
Simplified setup.py that focuses only on custom build logic.
Metadata is now handled by pyproject.toml to avoid duplication.
"""

import glob
import os
import platform
import shutil
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install

# Paths must be absolute: isolated / PEP 517 builds do not guarantee cwd == project root.
_ROOT = os.path.abspath(os.path.dirname(__file__))
_SRC_OCTO_LIB = os.path.join(_ROOT, "src", "octomap", "lib")
_PYOCTOMAP_LIB = os.path.join(_ROOT, "pyoctomap", "lib")

# Set by resolve_octomap(); CustomBuildExt skips bundling in system/conda mode.
USE_SYSTEM_OCTOMAP = False


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _unique(seq):
    seen = set()
    out = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _prefix_candidates():
    prefixes = []
    for key in ("PREFIX", "CONDA_PREFIX"):
        value = os.environ.get(key)
        if value:
            prefixes.append(value)
    prefixes.append(sys.prefix)
    if platform.system() == "Windows":
        for key in ("LIBRARY_PREFIX", "LIBRARY_INC"):
            value = os.environ.get(key)
            if value:
                prefixes.append(
                    os.path.dirname(value) if key == "LIBRARY_INC" else value
                )
    return _unique(prefixes)


def _include_dir(prefix):
    if platform.system() == "Windows":
        for candidate in (
            os.environ.get("LIBRARY_INC"),
            os.path.join(prefix, "include"),
            os.path.join(prefix, "Library", "include"),
        ):
            if candidate and os.path.isdir(candidate):
                return candidate
    return os.path.join(prefix, "include")


def _lib_dir(prefix):
    if platform.system() == "Windows":
        for candidate in (
            os.environ.get("LIBRARY_LIB"),
            os.path.join(prefix, "lib"),
            os.path.join(prefix, "Library", "lib"),
        ):
            if candidate and os.path.isdir(candidate):
                return candidate
        return os.path.join(prefix, "lib")
    for candidate in (os.path.join(prefix, "lib"), os.path.join(prefix, "lib64")):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(prefix, "lib")


def _has_library(lib_dir, name):
    if not os.path.isdir(lib_dir):
        return False
    patterns = [
        f"lib{name}.so",
        f"lib{name}.so.*",
        f"lib{name}.dylib",
        f"lib{name}.*.dylib",
        f"{name}.lib",
        f"lib{name}.dll.a",
        f"lib{name}.dll",
        f"{name}.dll",
    ]
    return any(glob.glob(os.path.join(lib_dir, pattern)) for pattern in patterns)


def find_system_octomap():
    """Locate a preinstalled OctoMap (conda-forge or system).

    Returns (include_dirs, library_dirs, libraries, has_dynamic_edt) or None.
    """
    for prefix in _prefix_candidates():
        include_root = _include_dir(prefix)
        header = os.path.join(include_root, "octomap", "octomap.h")
        if not os.path.exists(header):
            continue
        lib_dir = _lib_dir(prefix)
        if not _has_library(lib_dir, "octomap"):
            continue
        include_dirs = [include_root, os.path.join(include_root, "octomap")]
        libraries = ["octomap", "octomath"]
        edt_header = os.path.join(include_root, "dynamicEDT3D", "dynamicEDTOctomap.h")
        has_dynamic_edt = os.path.exists(edt_header) and _has_library(
            lib_dir, "dynamicedt3d"
        )
        if has_dynamic_edt:
            libraries.insert(0, "dynamicedt3d")
        return include_dirs, [lib_dir], libraries, has_dynamic_edt
    return None


def vendored_octomap_available():
    header = os.path.join(
        _ROOT, "src", "octomap", "octomap", "include", "octomap", "octomap.h"
    )
    return os.path.exists(header) and os.path.isdir(_SRC_OCTO_LIB) and os.listdir(
        _SRC_OCTO_LIB
    )


def _skip_native_build():
    """True for sdist / PEP 517 metadata collection (no compile, no OctoMap libs)."""
    setup_cmds = {arg for arg in sys.argv if not str(arg).startswith("-")}
    compiling = {
        "build_ext",
        "build",
        "bdist_wheel",
        "bdist_egg",
        "install",
        "develop",
        "editable_wheel",
    }
    return compiling.isdisjoint(setup_cmds)


def resolve_octomap():
    """Choose system/conda OctoMap vs the vendored tree."""
    global USE_SYSTEM_OCTOMAP
    want_system = _env_flag("PYOCTOMAP_SYSTEM_OCTOMAP")
    system = find_system_octomap()

    if want_system:
        if system is None:
            prefixes = ", ".join(_prefix_candidates()) or "(none)"
            raise RuntimeError(
                "PYOCTOMAP_SYSTEM_OCTOMAP is set, but no octomap install was found. "
                f"Searched prefixes: {prefixes}"
            )
        USE_SYSTEM_OCTOMAP = True
        return system

    if vendored_octomap_available():
        include_dirs = [
            "src/octomap/octomap/include",
            "src/octomap/octomap/include/octomap",
            "src/octomap/dynamicEDT3D/include",
        ]
        if platform.system() == "Windows":
            library_dirs = [
                d
                for d in ("src/octomap/lib", "pyoctomap/lib")
                if os.path.isdir(os.path.join(_ROOT, d))
            ]
        else:
            library_dirs = ["src/octomap/lib"]
        USE_SYSTEM_OCTOMAP = False
        return include_dirs, library_dirs, ["dynamicedt3d", "octomap", "octomath"], True

    if system is not None:
        print("Vendored OctoMap not found; falling back to system/conda octomap")
        USE_SYSTEM_OCTOMAP = True
        return system

    raise RuntimeError(
        "OctoMap headers/libraries not found. Either build the vendored copy "
        "(scripts/ci/build_octomap.sh) or install octomap and set "
        "PYOCTOMAP_SYSTEM_OCTOMAP=1."
    )


def get_lib_files():
    """Get the appropriate library files for the current platform"""
    lib_dir = _SRC_OCTO_LIB
    
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
            if USE_SYSTEM_OCTOMAP:
                print("System/conda octomap requested; not bundling shared libraries")
            else:
                # Copy libraries to source directory first (for MANIFEST.in)
                copy_libraries_to_source()

            # Run the normal build
            super().run()

            if not USE_SYSTEM_OCTOMAP:
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
        if USE_SYSTEM_OCTOMAP:
            return
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
        if USE_SYSTEM_OCTOMAP:
            return
        package_dir = "pyoctomap"
        lib_package_dir = os.path.join(package_dir, "lib")
        copy_libraries_to_directory(lib_package_dir)


def _extension_sources():
    """Map extension module names to .pyx paths that exist on disk."""
    possible_paths = {
        "pyoctomap.octree_base": "pyoctomap/octree_base.pyx",
        "pyoctomap.octree_iterators": "pyoctomap/octree_iterators.pyx",
        "pyoctomap.octree": "pyoctomap/octree.pyx",
        "pyoctomap.octomap": "pyoctomap/octomap.pyx",
        "pyoctomap.color_octree": "pyoctomap/color_octree.pyx",
        "pyoctomap.counting_octree": "pyoctomap/counting_octree.pyx",
        "pyoctomap.stamped_octree": "pyoctomap/stamped_octree.pyx",
        "pyoctomap.pointcloud": "pyoctomap/pointcloud.pyx",
    }
    found = {}
    for module_name, path in possible_paths.items():
        if os.path.exists(os.path.join(_ROOT, path)):
            found[module_name] = path
    return found


def build_extensions(require_native=True):
    """Build the Cython extensions with proper configuration"""

    pyx_files = _extension_sources()

    # sdist / egg_info only need to declare sources; do not locate or compile OctoMap.
    if not require_native:
        return [
            Extension(name, [path], language="c++")
            for name, path in pyx_files.items()
        ]

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
        # Bundled-lib rpath is added only in vendored mode below.

    octomap_includes, octomap_lib_dirs, octomap_libs, has_dynamic_edt = resolve_octomap()
    if USE_SYSTEM_OCTOMAP:
        print(
            "Linking against system/conda octomap "
            f"(dynamicEDT3D={'yes' if has_dynamic_edt else 'no'})"
        )
        rpath_args = []
    else:
        print("Linking against vendored OctoMap and bundling shared libraries")
        if platform.system() == "Linux":
            rpath_args = ["-Wl,-rpath,$ORIGIN/lib"]
        elif platform.system() == "Darwin":
            rpath_args = ["-Wl,-rpath,@loader_path/lib"]

    # Paths here must be relative to setup.py (setuptools / egg_info); use chdir in CustomBuildExt.run.
    common_include_dirs = ["pyoctomap", *octomap_includes, numpy_include]
    common_library_dirs = octomap_lib_dirs
    common_libraries = octomap_libs

    if not USE_SYSTEM_OCTOMAP and platform.system() == "Windows":
        win_import_libs = ("dynamicedt3d.lib", "octomap.lib", "octomath.lib")
        if not common_library_dirs:
            sys.exit(
                "Windows build: expected library directories missing. "
                f"Neither src/octomap/lib nor pyoctomap/lib exists under {_ROOT}."
            )
        if not any(
            all(os.path.isfile(os.path.join(_ROOT, d, f)) for f in win_import_libs)
            for d in common_library_dirs
        ):
            sys.exit(
                "Windows build: link against OctoMap requires these import libraries in "
                "src/octomap/lib (or a full set in pyoctomap/lib):\n  "
                + "\n  ".join(win_import_libs)
                + "\nBuild native libs first, e.g.:\n"
                '  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/ci/build_octomap_windows.ps1 '
                f'-ProjectRoot "{_ROOT}"'
            )

    common_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    
    ext_modules = []
    
    # Build octree_base extension
    if pyx_files.get("pyoctomap.octree_base"):
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
    if pyx_files.get("pyoctomap.octree_iterators"):
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
    if pyx_files.get("pyoctomap.octree"):
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
    if pyx_files.get("pyoctomap.octomap"):
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
    if pyx_files.get("pyoctomap.color_octree"):
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
    if pyx_files.get("pyoctomap.counting_octree"):
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
    if pyx_files.get("pyoctomap.stamped_octree"):
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
    if pyx_files.get("pyoctomap.pointcloud"):
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
        compiler_directives={'language_level': 3},
        compile_time_env={"HAS_DYNAMIC_EDT": bool(has_dynamic_edt)},
    )


def main():
    """Main setup function - minimal since pyproject.toml handles metadata"""

    ext_modules = build_extensions(require_native=not _skip_native_build())

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