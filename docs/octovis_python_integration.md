# OctoViz Python Integration Options

## Overview

OctoViz (octovis) is a Qt-based GUI application for visualizing OctoMap files. This document explains the options for using it from Python.

## Current Approach: External Process (Recommended)

**Status:** ✅ Currently implemented in `examples/demo_octoviz.py`

**How it works:**
```python
import subprocess
subprocess.Popen(['octovis', 'demo_octoviz.bt'])
```

**Pros:**
- ✅ Simple and straightforward
- ✅ No licensing conflicts (octovis is GPL, PyOctoMap is BSD)
- ✅ No additional dependencies
- ✅ Standard workflow in robotics
- ✅ Works across all platforms
- ✅ No GUI event loop conflicts

**Cons:**
- ❌ Separate process (not embedded)
- ❌ Limited programmatic control
- ❌ Requires octovis to be installed separately

**When to use:** This is the recommended approach for most use cases.

---

## Option 1: Cython Bindings (Complex, Not Recommended)

### Technical Feasibility

**Can it be done?** Yes, but with significant challenges:

1. **License Issues:**
   - Octovis is **GPL v2** licensed
   - PyOctoMap core is **BSD** licensed
   - Creating bindings that link against octovis would require the bindings to be GPL
   - This could affect PyOctoMap's licensing if included

2. **Technical Challenges:**
   - Qt GUI integration (requires PyQt/PySide)
   - QGLViewer dependencies
   - GUI event loop management
   - Threading complexities
   - Heavy dependencies (Qt, OpenGL, QGLViewer)

3. **Architecture:**
   - Octovis is designed as a standalone application (`main.cpp` creates `QApplication`)
   - Not designed as a library API
   - Would need to wrap `ViewerWidget`, `OcTreeDrawer`, etc.

### What Would Be Required

```cython
# Hypothetical pyoctomap/octovis.pyx
cdef extern from "octovis/ViewerWidget.h":
    cdef cppclass ViewerWidget:
        ViewerWidget()
        void addSceneObject(SceneObject* obj)
        # ... more methods

# Would need Qt Python bindings (PyQt/PySide)
from PyQt5.QtWidgets import QApplication
from PyQt5.QtOpenGL import QGLWidget
```

**Dependencies needed:**
- PyQt5/PySide2 (Qt Python bindings)
- Qt development libraries
- QGLViewer development libraries
- Complex build configuration

**Estimated effort:** 2-4 weeks of development + ongoing maintenance

---

## Option 2: Python Qt Wrapper (Alternative Approach)

Instead of wrapping octovis directly, create a Python visualization using Qt:

```python
# Hypothetical approach
from PyQt5.QtWidgets import QApplication
from PyQt5.QtOpenGL import QGLWidget
import pyoctomap

class OctreeViewer(QGLWidget):
    def __init__(self, tree):
        super().__init__()
        self.tree = tree
    
    def paintGL(self):
        # Render octree using OpenGL
        # Extract points from tree and render
        pass
```

**Pros:**
- ✅ Full Python control
- ✅ Can use BSD license (no GPL dependency)
- ✅ Lighter weight than full octovis

**Cons:**
- ❌ Need to reimplement visualization logic
- ❌ Significant development effort
- ❌ Still requires Qt dependencies

---

## Option 3: Use Existing Python Visualization Libraries

**Current approach in examples:** Open3D visualization

```python
# examples/demo_octomap_open3d.py
import open3d as o3d
# Extract points from octree
# Visualize with Open3D
```

**Pros:**
- ✅ Pure Python
- ✅ No licensing issues
- ✅ Good performance
- ✅ Easy to integrate
- ✅ Already implemented

**Cons:**
- ❌ Different from octovis (different UI/features)
- ❌ Doesn't use official OctoMap visualizer

---

## Recommendation

### For Most Users: Use External Process ✅

```python
# examples/demo_octoviz.py - Current implementation
import subprocess
subprocess.Popen(['octovis', 'my_map.bt'])
```

This is the standard approach and works well for visualization needs.

### For Advanced Users: Use Open3D ✅

```python
# examples/demo_octomap_open3d.py - Already implemented
import open3d as o3d
# Full Python control, no external dependencies
```

### For Embedding Needs: Consider Cython Bindings (Not Recommended)

Only if you absolutely need:
- Embedded visualization in Python GUI
- Programmatic control over octovis
- Real-time updates

**But be aware:**
- License implications (GPL)
- Complex build requirements
- Significant development effort
- Maintenance burden

---

## License Considerations

| Component | License | Can Include? |
|-----------|---------|--------------|
| octomap core | BSD | ✅ Yes |
| octovis | GPL v2 | ⚠️ Only if PyOctoMap becomes GPL |
| PyOctoMap (current) | BSD | ✅ Yes |

**Key Point:** Linking against GPL code requires your code to be GPL. This is why octovis is kept as an external dependency.

---

## Summary

| Approach | Complexity | Licensing | Recommended |
|----------|------------|------------|-------------|
| External Process | Low | ✅ Clean | ✅ **Yes** |
| Open3D Visualization | Low | ✅ Clean | ✅ **Yes** |
| Cython Bindings | High | ⚠️ GPL | ❌ No |
| Python Qt Wrapper | Medium | ✅ Clean | ⚠️ Maybe |

**Bottom Line:** The current approach (external process + Open3D) provides the best balance of functionality, simplicity, and licensing compatibility.
