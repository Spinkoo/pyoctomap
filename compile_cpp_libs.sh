#!/bin/bash

# Script to compile the C++ OctoMap libraries locally
# This mimics the build process in docker/Dockerfile.wheel

set -e

echo "🚀 Compiling C++ OctoMap libraries..."

# Check for cmake
if ! command -v cmake &> /dev/null; then
    echo "❌ Error: cmake not found. Please install cmake."
    exit 1
fi

# Check for build tools
if ! command -v make &> /dev/null; then
    echo "❌ Error: make not found. Please install make/build-essential."
    exit 1
fi

# Directory setup
SRC_DIR="src/octomap"
BUILD_DIR="${SRC_DIR}/build"
LIB_DIR="${SRC_DIR}/lib"

echo "📂 Source directory: ${SRC_DIR}"

# Create build directory
mkdir -p "${BUILD_DIR}"
mkdir -p "${LIB_DIR}"

# Navigate to build directory
cd "${BUILD_DIR}"

# Check if we are in a subdirectory or if src/octomap is the root
if [ -f "../CMakeLists.txt" ]; then
    CMAKE_SOURCE_DIR=".."
    INSTALL_PREFIX=".."
elif [ -f "../../src/octomap/CMakeLists.txt" ]; then
    CMAKE_SOURCE_DIR="../../src/octomap"
    INSTALL_PREFIX="../../src/octomap"
else
    # Fallback - assume we are in src/octomap/build and parent has CMakeLists.txt
    CMAKE_SOURCE_DIR=".."
    INSTALL_PREFIX=".."
fi

# Configure CMake
echo "🔧 Configuring CMake with source ${CMAKE_SOURCE_DIR}..."
cmake "${CMAKE_SOURCE_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"

# Build
echo "🔨 Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Install (to src/octomap/lib via CMAKE_INSTALL_PREFIX)
echo "📦 Installing..."
make install

# Copy libraries if they ended up in a different lib folder (CMake install sometimes uses lib or lib64)
# We need them in src/octomap/lib
if [ -d "${INSTALL_PREFIX}/lib64" ]; then
    echo "Copying from lib64 to lib..."
    cp -r "${INSTALL_PREFIX}/lib64/"* "${INSTALL_PREFIX}/lib/" 2>/dev/null || true
fi

# Check if libraries exist in lib folder relative to source
echo "✅ Build complete. Checking libraries..."
ls -la "${INSTALL_PREFIX}/lib/"

echo "🎉 C++ libraries compiled successfully!"

