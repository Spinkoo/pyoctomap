#!/usr/bin/env bash
set -euo pipefail

# Build OctoMap C++ libraries with CMake, then build/install Python package.
# conda-build sets SRC_DIR to the extracted source root and PREFIX to install prefix.

cd "${SRC_DIR}"

# Show toolchain info (best-effort)
cmake --version || true
${CXX:-g++} --version || true

# Build and install OctoMap (C++)
pushd src/octomap
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build . --parallel
cmake --install .
popd

# Build and install Python package (will link to libs in ${PREFIX})
"${PYTHON}" -m pip install . -vv --no-deps --ignore-installed

