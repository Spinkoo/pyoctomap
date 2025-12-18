## API Reference

This page summarizes the public Python API of **PyOctoMap**, with a focus on
the different **tree types**, how they differ, and when to use each one.

- For installation & quick-start examples, see the main `README.md`.
- For build details, see `docs/build_system.md`.
- For file formats, see `docs/file_format.md`.

---

## Tree Families Overview

PyOctoMap exposes several octree variants, all accessible from the top‑level
`pyoctomap` package:

| Tree family        | Class name          | Stores                    | Typical use‑case                                                |
|--------------------|---------------------|---------------------------|-----------------------------------------------------------------|
| Occupancy tree     | `OcTree`           | Occupancy probability     | Standard 3D mapping, collision checking, path planning         |
| Colored occupancy  | `ColorOcTree`      | Occupancy + RGB color     | Semantic / visual mapping, colored point clouds                |
| Counting tree      | `CountingOcTree`   | Integer hit count         | Hit statistics, sensor coverage, multi‑hit filtering           |
| Stamped occupancy  | `OcTreeStamped`    | Occupancy + timestamps    | Temporal maps, degrading outdated information over time        |

All trees share the same **octree layout**, similar construction pattern, and
basic notions of resolution and nodes. Some methods (e.g. ray casting) are
only meaningful for true occupancy trees.

---

## Choosing the Right Tree

- **Use `OcTree`** when you just need a fast, standard occupancy map.
- **Use `ColorOcTree`** when you want both occupancy and color per voxel.
- **Use `CountingOcTree`** when you care about how many times a voxel was hit,
  or when you want to threshold points based on minimum hits.
- **Use `OcTreeStamped`** when you need time‑aware maps that can **degrade**
  or down‑weight old measurements.

### Feature Comparison

| Feature                        | `OcTree` | `ColorOcTree` | `CountingOcTree` | `OcTreeStamped` |
|--------------------------------|:--------:|:-------------:|:----------------:|:---------------:|
| Occupancy log‑odds             |   ✔      |      ✔        |        ✖         |       ✔         |
| RGB color per leaf             |   ✖      |      ✔        |        ✖         |       ✖         |
| Integer hit counter            |   ✖      |      ✖        |        ✔         |       ✖         |
| Timestamp per node             |   ✖      |      ✖        |        ✖         |       ✔         |
| Ray casting (`castRay`)        |   ✔      |      ✔        |  stub / not used |       ✔         |
| Dynamic decay helpers          |   ✔      |      ✔        |        ✖         |       ✔ (time)  |
| Standard `.bt`/`.ot` I/O       |   ✔      |      ✔        |        ✔         |       ✔         |

---

## Common Concepts and Types

These types are shared across most operations:

- **`OcTreeKey`** – discrete 3D index inside the tree.
- **Node classes** – `OcTreeNode`, `ColorOcTreeNode`, `CountingOcTreeNode`,
  `OcTreeNodeStamped`.
- **Iterators** – `SimpleTreeIterator`, `SimpleLeafIterator`,
  `SimpleLeafBBXIterator` (iterate all nodes, just leaf nodes, or only leaf
  nodes in a bounding box).

### Core Usage Pattern

```python
import numpy as np
import pyoctomap

# Create a tree with 0.1 m resolution
tree = pyoctomap.OcTree(0.1)

# Update occupancy at coordinates
tree.updateNode([1.0, 2.0, 3.0], True)   # occupied
tree.updateNode([1.0, 2.0, 3.0], False)  # free

# Query
node = tree.search([1.0, 2.0, 3.0])
if node and tree.isNodeOccupied(node):
    print("Occupied voxel")
```

---

## `OcTree` – Standard Occupancy Tree

The canonical octree for probabilistic 3D occupancy mapping.

### Construction

```python
from pyoctomap import OcTree

tree = OcTree(resolution=0.1)  # 10 cm voxels
```

- **`resolution` (float)** – voxel size in meters.

### Node Operations

- **`updateNode(point, occupied, lazy_eval=False)`**
  - `point`: list / `np.ndarray` of `[x, y, z]` (meters).
  - `occupied`: `bool`, `True` = occupied, `False` = free.
  - `lazy_eval`: if `True`, inner nodes are not updated immediately; call
    `updateInnerOccupancy()` later for maximum performance.

- **`search(point, depth=0)`**
  - Returns an `OcTreeNode` or `None` if the voxel is unknown.

- **`isNodeOccupied(node)` / `isNodeAtThreshold(node)`**
  - Convenience predicates for `OcTreeNode` state.

### Tree Information

- **`size()`** – total number of nodes in the tree.
- **`getResolution()`** – tree resolution (meters).
- **`getTreeDepth()`** – maximum depth of the tree.
- **`getNumLeafNodes()`** – number of leaf nodes.

### File I/O

```python
tree.write("map.bt")        # Save to standard OctoMap .bt file
loaded = tree.read("map.bt")

blob = tree.writeBinary()   # Serialize to bytes
tree.readBinary("map.bt")   # Read from file path
```

See `docs/file_format.md` for more detail.

### Ray Casting

```python
import numpy as np
origin = np.array([0.0, 0.0, 1.5])
direction = np.array([1.0, 0.0, 0.0])
end = np.zeros(3, dtype=np.float64)

hit = tree.castRay(origin, direction, end,
                   ignoreUnknownCells=True,
                   maxRange=50.0)
if hit:
    print("Hit at", end)
```

### Batch Insertion & Dynamic Mapping

These helpers are implemented in C++ for performance, and should be preferred
over Python loops:

- **`insertPointCloud(point_cloud, sensor_origin, max_range=-1.0, lazy_eval=False, discretize=False)`**
  - Batch insertion with optional discretization and lazy inner‑node updates.

- **`insertPointCloudRaysFast(point_cloud, sensor_origin, max_range=-1.0, lazy_eval=False)`**
  - Ultra‑fast insertion using independent rays (no deduplication).

- **`decayOccupancyInBBX(point_cloud, sensor_origin, logodd_decay_value=-0.2)`**
  - Apply temporal decay to occupied voxels in the scan’s bounding box.

- **`decayAndInsertPointCloud(point_cloud, sensor_origin, logodd_decay_value=-0.2, max_range=-1.0, update_inner_occupancy=True)`**
  - Recommended one‑shot helper for moving sensors in dynamic scenes:
    1. Decays old occupancy inside scan’s bounding box.
    2. Inserts the new point cloud.

**Decay tuning (rule of thumb)**  
Let \(d = \lvert\text{logodd\_decay\_value}\rvert\). A fully occupied voxel is
around log‑odds \(+4.0\).

\[
\text{Scans to forget} \approx \frac{4.0}{d}
\]

- `-0.2` (default): ≈ 20 scans to fade a ghost.
- `-1.0` to `-3.0`: ≈ 2–4 scans (very dynamic).
- `-0.05` to `-0.1`: ≈ 40–80 scans (mostly static).

---

## `ColorOcTree` – Occupancy + Color

`ColorOcTree` extends `OcTree` with RGB color information per voxel.

### Construction

```python
from pyoctomap import ColorOcTree

tree = ColorOcTree(0.1)
```

Methods match `OcTree` for occupancy, plus color‑specific helpers.

### Color Operations

- **`setNodeColor(point, r, g, b)`**
  - Set RGB color for a voxel at `point` (`0–255` ints).

- **`averageNodeColor(point, r, g, b)`**
  - Average new RGB measurement into existing color.

- **`integrateNodeColor(point, r, g, b)`**
  - Integrate color weighted by occupancy updates (useful when coupling with
    occupancy updates from sensors).

- **`insertPointCloudWithColor(points, colors, sensor_origin=None, max_range=-1.0, lazy_eval=True)`**
  - Insert a point cloud and set colors for all points in a single operation.
  - `points`: N×3 numpy array of point coordinates.
  - `colors`: N×3 numpy array of color values in [0, 1] range (converted to 0–255 internally).
  - `sensor_origin`: Optional sensor origin [x, y, z] for ray casting. If `None` (default), uses (0, 0, 0). Providing a proper sensor origin enables correct free-space carving along rays from sensor to points.
  - First inserts geometry using batch `insertPointCloud` with ray casting, then updates colors using key-based search for efficiency.
  - Returns the number of points processed.

### Example

```python
from pyoctomap import ColorOcTree
import numpy as np

tree = ColorOcTree(0.1)
coord = [1.0, 1.0, 1.0]

tree.updateNode(coord, True)
tree.setNodeColor(coord, 255, 0, 0)

node = tree.search(coord)
if node:
    print("Color:", node.getColor())  # (255, 0, 0)
```

---

## `CountingOcTree` – Hit Counting Tree

`CountingOcTree` records **integer hit counts** instead of probabilities. It
is useful for:

- Estimating sensor coverage or confidence.
- Filtering points/voxels based on minimum observations.
- Building occupancy maps downstream from stable hit statistics.

### Construction

```python
from pyoctomap import CountingOcTree

tree = CountingOcTree(0.1)        # resolution in meters
tree_from_file = CountingOcTree("counts.bt")
```

### Node Operations

- **`updateNode(key_or_coord)`**
  - Accepts either an `OcTreeKey` or `[x, y, z]` coordinates.
  - Increments the node’s **count** and returns a `CountingOcTreeNode`.

- `CountingOcTreeNode` methods:
  - `getCount()`, `increaseCount()`, `setCount(value)`.

### Queries

- **`getCentersMinHits(min_hits)`**
  - Returns a list of `[x, y, z]` centers for nodes whose count is at least
    `min_hits`.

- Occupancy helpers for compatibility:
  - `isNodeOccupied(node)` → `True` if `count > 0`.
  - `isNodeAtThreshold(node)` → always `False` (no probabilistic threshold).

### I/O and Geometry Helpers

`CountingOcTree` provides:

- `getResolution()`, `getTreeDepth()`, `size()`, `getNumLeafNodes()`,
  `calcNumNodes()`, `clear()`.
- `coordToKey(coord, depth=None)` → `OcTreeKey`.
- `keyToCoord(key, depth=None)` → `[x, y, z]`.

Note: `castRay` is implemented as a **stub** (always returns `False`), since
pure counts do not encode occupancy.

---

## `OcTreeStamped` – Time‑Stamped Occupancy

`OcTreeStamped` extends `OcTree` with per‑node timestamps. This allows building
maps that can **degrade outdated information** over time.

### Construction

```python
from pyoctomap import OcTreeStamped

tree = OcTreeStamped(0.1)
tree_from_file = OcTreeStamped("stamped.ot")
```

### Time‑Aware Operations

- **`getLastUpdateTime()`**
  - Timestamp of the last update (root node); `0` if the tree is empty.

- **`degradeOutdatedNodes(time_thres)`**
  - Decreases confidence of nodes that haven’t been updated for at least
    `time_thres` seconds.

- **`updateNodeLogOdds(node, update)`**
  - Update the log‑odds of an `OcTreeNodeStamped` and its timestamp together.

- **`integrateMissNoTime(node)`**
  - Integrate a miss **without updating** the timestamp (useful when you want
    to mark free space but keep the “last seen” time).

### Occupancy Updates

`OcTreeStamped.updateNode(...)` mirrors `OcTree.updateNode(...)` but always
updates timestamps internally:

```python
node = tree.updateNode([x, y, z], True)       # occupied
node = tree.updateNode([x, y, z], -0.4)       # log‑odds update
node = tree.updateNode(x, y, z, True)        # coordinate triplet
```

- **`insertPointCloudWithTimestamp(points, timestamp, sensor_origin=None, max_range=-1.0, lazy_eval=True)`**
  - Insert a point cloud and set timestamps for all points in a single operation.
  - `points`: N×3 numpy array of point coordinates.
  - `timestamp`: Unsigned integer timestamp value to set for all nodes.
  - `sensor_origin`: Optional sensor origin [x, y, z] for ray casting. If `None` (default), uses (0, 0, 0). Providing a proper sensor origin enables correct free-space carving along rays from sensor to points.
  - First inserts geometry using batch `insertPointCloud` with ray casting, then updates timestamps using key-based search for efficiency.
  - Returns the number of points processed.

Other common helpers are inherited:

- `coordToKey(coord, depth=None)` / `keyToCoord(key, depth=None)`.
- `search(value, depth=0)` → `OcTreeNodeStamped` or `None`.
- `isNodeOccupied(node)`, `isNodeAtThreshold(node)`.
- `castRay(...)` – same semantics as for `OcTree`.

---

## Node and Key Types

### `OcTreeNode`

Represents a single node in an occupancy tree (`OcTree`, `ColorOcTree`,
`OcTreeStamped`).

- `getOccupancy()` → probability in `[0.0, 1.0]`.
- `getValue()` / `setValue(value)` → low‑level log‑odds value.
- `getLogOdds()` / `setLogOdds(value)` → explicit log‑odds access.
- `hasChildren()`, `childExists(i)`, `addValue(p)`, `getMaxChildLogOdds()`,
  `updateOccupancyChildren()`.

### `ColorOcTreeNode`

Extends `OcTreeNode` with color:

- `getColor()` → `(r, g, b)` tuple.
- `setColor(r, g, b)`.
- `isColorSet()` – `True` if color differs from default.
- `getAverageChildColor()` – averaged children color.

### `CountingOcTreeNode`

See the `CountingOcTree` section above:

- `getCount()`, `setCount(value)`, `increaseCount()`.

### `OcTreeNodeStamped`

Extends `OcTreeNode` with:

- `getTimestamp()`, `setTimestamp(t)`, `updateTimestamp()`.
- `updateOccupancyChildren()` (updates both occupancy and timestamp).

### `OcTreeKey`

Discrete 3D key for internal indexing:

```python
from pyoctomap import OcTreeKey

key = OcTreeKey()
key[0], key[1], key[2]  # integer coordinates
```

Typical conversions:

- `coordToKey(coord, depth=None)` → `OcTreeKey`.
- `keyToCoord(key, depth=None)` → `[x, y, z]`.
- `coordToKeyChecked(coord, depth=None)` → `(success: bool, key)`.

---

## Iterators

Iterators allow efficient traversal of the tree. They are available on all
occupancy trees.

- **`begin_tree(maxDepth=0)`**
  - All nodes (inner + leaf).
  - Use for structure analysis or debugging.

- **`begin_leafs(maxDepth=0)`**
  - Leaf nodes only – best choice for most occupancy operations.

- **`begin_leafs_bbx(bbx_min, bbx_max, maxDepth=0)`**
  - Leaf nodes within a bounding box.

Example (leaf iteration):

```python
for leaf in tree.begin_leafs():
    coord = leaf.getCoordinate()
    if tree.isNodeOccupied(leaf):
        print("Occupied at", coord)
```

For more complete examples (path planning, dynamic mapping, visualization),
see the **Examples** section of `README.md` and the scripts in `examples/`.


