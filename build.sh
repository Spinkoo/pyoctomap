#!/bin/bash

# PyOctoMap Build Script
# This script automates the build and installation process

set -e  # Exit on any error

echo "🚀 Building PyOctoMap..."

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: setup.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 Python version: $python_version"

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import setuptools" 2>/dev/null || {
    echo "❌ setuptools not found. Installing..."
    pip install setuptools
}

python3 -c "import numpy" 2>/dev/null || {
    echo "❌ NumPy not found. Installing..."
    pip install numpy
}

python3 -c "import Cython" 2>/dev/null || {
    echo "❌ Cython not found. Installing..."
    pip install Cython
}

echo "ℹ️  Note: Libraries are bundled in the wheel under octomap/lib, and rpath points to $ORIGIN/lib."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
# Also remove any locally-built extensions that could shadow installed package
rm -f octomap/*.so octomap/*.so.* octomap.cpython-*.so 2>/dev/null || true

# Build the wheel with bundled libraries
echo "🔨 Building wheel package with bundled libraries..."
python3 setup.py bdist_wheel

# Check if wheel was created
wheel_file=$(ls dist/*.whl 2>/dev/null | head -1)
if [ -z "$wheel_file" ]; then
    echo "❌ Error: Wheel file not created!"
    exit 1
fi

echo "✅ Wheel created: $wheel_file"

# Install the wheel
echo "📦 Installing wheel..."
pip install "$wheel_file" --force-reinstall

# Test the installation (no env vars)
echo "🧪 Testing installation (no env vars required)..."
# Change to a different directory to avoid source directory conflicts
cd /tmp
python3 - <<'PY'
import pyoctomap
print('✅ OctoMap import successful!')
# Basic functionality
t = pyoctomap.OcTree(0.1)
t.updateNode([1.0, 2.0, 3.0], True)
n = t.search([1.0, 2.0, 3.0])
if n and t.isNodeOccupied(n):
    print('✅ Basic functionality working!')
else:
    print('❌ Basic functionality test failed!')
    raise SystemExit(1)
PY

echo ""
echo "🎉 Build and installation completed successfully!"
echo "📁 Wheel file: $wheel_file"
echo "📦 Libraries bundled: liboctomap, libdynamicedt3d, liboctomath"
echo "📖 Demos available in the examples/ folder (e.g., examples/basic_test.py)"
echo "Usage:"
echo "  import pyoctomap"
echo "  tree = pyoctomap.OcTree(0.1)"
echo "  tree.updateNode([1.0, 2.0, 3.0], True)"
