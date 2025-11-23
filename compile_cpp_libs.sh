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

# Configure CMake
echo "🔧 Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..

# Build
echo "🔨 Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Install (to src/octomap/lib via CMAKE_INSTALL_PREFIX)
echo "📦 Installing to ${LIB_DIR}..."
make install

# Copy libraries to the expected location if needed
# CMake install should handle this with CMAKE_INSTALL_PREFIX/lib, but let's ensure
# they are in src/octomap/lib where setup.py expects them.

# Check if libraries exist in lib folder relative to source
echo "✅ Build complete. Checking libraries..."
ls -la ../lib/

echo "🎉 C++ libraries compiled successfully!"

