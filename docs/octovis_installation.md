# Installing OctoViz (octovis)

OctoViz (octovis) is the official 3D visualization tool for OctoMap files. This guide provides installation instructions for different platforms.

## Linux (Ubuntu/Debian)

### Option 1: Install from Package Manager (Recommended)

```bash
sudo apt-get update
sudo apt-get install octovis
```

This installs octovis along with all its dependencies (Qt, libQGLViewer, etc.).

### Option 2: Build from Source

If the package manager version is outdated or unavailable:

```bash
# Install dependencies
sudo apt-get install build-essential cmake libqt4-dev libqglviewer-dev

# Clone OctoMap repository
git clone https://github.com/OctoMap/octomap.git
cd octomap

# Build octovis
cd octovis
mkdir build
cd build
cmake ..
make

# Install (optional)
sudo make install
```

**Note:** If you encounter Qt qmake errors, you may need to configure Qt alternatives:
```bash
sudo update-alternatives --config qmake
```
Select Qt4 if available.

## macOS

### Option 1: Using Homebrew (Recommended)

```bash
brew install octomap
```

This installs the complete OctoMap package including octovis.

### Option 2: Build from Source

```bash
# Install dependencies via Homebrew
brew install qt@4 cmake

# Clone and build
git clone https://github.com/OctoMap/octomap.git
cd octomap/octovis
mkdir build
cd build
cmake ..
make
```

## Windows

### Option 1: Using WSL (Windows Subsystem for Linux)

If you're using WSL (recommended for this project):

```bash
# In WSL terminal
sudo apt-get update
sudo apt-get install octovis
```

### Option 2: Build from Source (Native Windows)

Building on native Windows is more complex and requires:

1. **Install dependencies:**
   - CMake: https://cmake.org/download/
   - Qt4 or Qt5: https://www.qt.io/download
   - Visual Studio or MinGW

2. **Build process:**
   ```cmd
   git clone https://github.com/OctoMap/octomap.git
   cd octomap\octovis
   mkdir build
   cd build
   cmake .. -G "Visual Studio 16 2019"  # or your generator
   cmake --build . --config Release
   ```

## Verifying Installation

After installation, verify octovis is available:

```bash
octovis --version
# or
which octovis
```

## Usage

Once installed, you can visualize OctoMap files:

```bash
octovis demo_octoviz.bt
```

Or use it from Python:

```python
import subprocess
subprocess.Popen(['octovis', 'demo_octoviz.bt'])
```

## Troubleshooting

### Qt/qmake Not Found

If CMake can't find Qt:
```bash
# Set Qt path explicitly
export Qt5_DIR=/path/to/qt5/lib/cmake/Qt5
# or for Qt4
export Qt4_DIR=/usr/lib/x86_64-linux-gnu/cmake/Qt4
```

### libQGLViewer Not Found

Install the development package:
```bash
sudo apt-get install libqglviewer-dev
```

### Runtime Library Errors

If you get library loading errors:
```bash
# Linux: Add library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# macOS: Add library path
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

## Quick Test

Run the PyOctoMap example to generate a test file:

```bash
cd examples
python demo_octoviz.py
```

This will create `demo_octoviz.bt` which you can then visualize with octovis.

## Additional Resources

- OctoMap GitHub: https://github.com/OctoMap/octomap
- OctoMap Documentation: https://octomap.github.io/
- OctoViz README: https://github.com/OctoMap/octomap/blob/devel/octovis/README.md
