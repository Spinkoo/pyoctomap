#!/usr/bin/env bash
set -euo pipefail

# Build OctoMap C++ libraries with CMake, then build/install Python package.
# conda-build sets SRC_DIR to the extracted source root and PREFIX to install prefix.

cd "${SRC_DIR}"

# Show toolchain info (best-effort)
echo "=== Build Environment ==="
cmake --version || true
${CXX:-g++} --version || true
echo "Python: ${PYTHON}"
echo "PREFIX: ${PREFIX}"
echo "SRC_DIR: ${SRC_DIR}"

# Verify source directory structure
if [ ! -d "src/octomap" ]; then
    echo "Error: src/octomap directory not found!"
    ls -la src/ 2>/dev/null || ls -la
    exit 1
fi

# Build and install OctoMap (C++)
echo "=== Building OctoMap C++ libraries ==="
pushd src/octomap
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
  -G "Ninja" || cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"

cmake --build . --parallel ${CPU_COUNT:-}
cmake --install .
popd

# Build and install Python package (will link to libs in ${PREFIX})
echo "=== Building Python package ==="
"${PYTHON}" -m pip install . -vv --no-deps --ignore-installed --no-build-isolation

