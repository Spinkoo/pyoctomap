# API Reference

Complete API documentation for OctoMap2Python.

## OcTree Class

The main class for 3D occupancy mapping using octrees.

### Constructor

```python
OcTree(resolution)
```

- `resolution` (float): Tree resolution in meters

### Core Methods

#### Node Operations

```python
updateNode(point, occupied, lazy_eval=False)
```
- `point` (list/np.array): 3D coordinates [x, y, z]
- `occupied` (bool): True for occupied, False for free
- `lazy_eval` (bool): Skip inner node updates for performance

```python
search(point, depth=0)
```
- `point` (list/np.array): 3D coordinates to search
- `depth` (int): Maximum search depth
- Returns: `OcTreeNode` or `None`

```python
isNodeOccupied(node)
```
- `node` (OcTreeNode): Node to check
- Returns: `bool`

```python
isNodeAtThreshold(node)
```
- `node` (OcTreeNode): Node to check
- Returns: `bool`

#### Tree Information

```python
size()
```
- Returns: Number of nodes in tree

```python
getResolution()
```
- Returns: Tree resolution in meters

```python
getTreeDepth()
```
- Returns: Maximum tree depth

```python
getNumLeafNodes()
```
- Returns: Number of leaf nodes

#### File Operations

```python
write(filename=None)
```
- `filename` (str): Output file path
- Returns: `bool` or `str` (if filename=None)

```python
read(filename)
```
- `filename` (str): Input file path
- Returns: New `OcTree` instance

```python
writeBinary(filename=None)
```
- `filename` (str): Output file path
- Returns: `bool` or `str` (if filename=None)

```python
readBinary(filename)
```
- `filename` (str): Input file path
- Returns: `bool`

#### Ray Casting

```python
castRay(origin, direction, end, ignoreUnknownCells=False, maxRange=-1.0)
```
- `origin` (list/np.array): Ray start point
- `direction` (list/np.array): Ray direction vector
- `end` (list/np.array): Output hit point
- `ignoreUnknownCells` (bool): Ignore unknown cells
- `maxRange` (float): Maximum ray range
- Returns: `bool` (hit/no hit)

#### Advanced Methods

```python
addPointWithRayCasting(point, sensor_origin, update_inner_occupancy=False)
```
- `point` (np.array): 3D point to add
- `sensor_origin` (np.array): Sensor origin for ray casting
- `update_inner_occupancy` (bool): Update inner node occupancy
- Returns: `bool` (success)

```python
markFreeSpaceAlongRay(origin, end_point, step_size=None)
```
- `origin` (np.array): Ray start point
- `end_point` (np.array): Ray end point
- `step_size` (float): Ray sampling step size

```python
addPointsBatch(points, sensor_origins=None, update_inner_occupancy=True)
```
- `points` (np.array): Nx3 array of points
- `sensor_origins` (np.array): Nx3 array of sensor origins
- `update_inner_occupancy` (bool): Update inner node occupancy
- Returns: `int` (number of points added)

```python
addPointCloudWithRayCasting(point_cloud, sensor_origin, max_range=-1.0, update_inner_occupancy=True)
```
- `point_cloud` (np.array): Nx3 array of points
- `sensor_origin` (np.array): Sensor origin for ray casting
- `max_range` (float): Maximum range for points
- `update_inner_occupancy` (bool): Update inner node occupancy
- Returns: `int` (number of points added)

#### Iterators

```python
begin_tree(maxDepth=0)
```
- Returns: `SimpleTreeIterator`

```python
begin_leafs(maxDepth=0)
```
- Returns: `SimpleLeafIterator`

```python
begin_leafs_bbx(bbx_min, bbx_max, maxDepth=0)
```
- `bbx_min` (np.array): Bounding box minimum
- `bbx_max` (np.array): Bounding box maximum
- Returns: `SimpleLeafBBXIterator`

## OcTreeNode Class

Represents a single node in the octree.

### Methods

```python
getOccupancy()
```
- Returns: Occupancy probability (0.0-1.0)

```python
getValue()
```
- Returns: Log-odds value

```python
setValue(value)
```
- `value` (float): Log-odds value

```python
getLogOdds()
```
- Returns: Log-odds value

```python
setLogOdds(value)
```
- `value` (float): Log-odds value

```python
hasChildren()
```
- Returns: `bool` (deprecated, use `tree.nodeHasChildren(node)`)

```python
childExists(i)
```
- `i` (int): Child index (0-7)
- Returns: `bool`

```python
addValue(p)
```
- `p` (float): Value to add to log-odds

```python
getMaxChildLogOdds()
```
- Returns: Maximum child log-odds value

```python
updateOccupancyChildren()
```
- Updates occupancy based on children

## Iterator Classes

### SimpleTreeIterator

Iterates over all nodes in the tree.

```python
for node_it in tree.begin_tree():
    coord = node_it.getCoordinate()
    depth = node_it.getDepth()
    size = node_it.getSize()
    is_leaf = node_it.isLeaf()
```

### SimpleLeafIterator

Iterates over leaf nodes only.

```python
for leaf_it in tree.begin_leafs():
    coord = leaf_it.getCoordinate()
    depth = leaf_it.getDepth()
    size = leaf_it.getSize()
    is_leaf = leaf_it.isLeaf()
    node = leaf_it.current_node
```

### SimpleLeafBBXIterator

Iterates over leaf nodes within a bounding box.

```python
bbx_min = np.array([0.0, 0.0, 0.0])
bbx_max = np.array([10.0, 10.0, 10.0])
for bbx_it in tree.begin_leafs_bbx(bbx_min, bbx_max):
    coord = bbx_it.getCoordinate()
    depth = bbx_it.getDepth()
    size = bbx_it.getSize()
    is_leaf = bbx_it.isLeaf()
    node = bbx_it.current_node
```

## OcTreeKey Class

Represents internal octree coordinates.

### Constructor

```python
OcTreeKey(a=0, b=0, c=0)
```

### Methods

```python
computeChildIdx(key, depth)
```
- `key` (OcTreeKey): Key to compute
- `depth` (int): Tree depth
- Returns: Child index

```python
computeIndexKey(level, key)
```
- `level` (int): Tree level
- `key` (OcTreeKey): Key to compute
- Returns: New `OcTreeKey`

### Properties

```python
key[0]  # X coordinate
key[1]  # Y coordinate
key[2]  # Z coordinate
```

## Utility Functions

### Coordinate Conversion

```python
coordToKey(coord, depth=None)
```
- `coord` (list/np.array): 3D coordinates
- `depth` (int): Optional depth
- Returns: `OcTreeKey`

```python
keyToCoord(key, depth=None)
```
- `key` (OcTreeKey): Octree key
- `depth` (int): Optional depth
- Returns: 3D coordinates

```python
coordToKeyChecked(coord, depth=None)
```
- `coord` (list/np.array): 3D coordinates
- `depth` (int): Optional depth
- Returns: `(bool, OcTreeKey)` (success, key)

## Error Handling

### Exceptions

- `NullPointerException`: Raised when accessing null pointers
- `RuntimeError`: Raised for iterator errors
- `TypeError`: Raised for incorrect argument types

### Best Practices

1. Always check if `search()` returns `None` before using nodes
2. Use try-catch blocks around iterator operations
3. Check `isNodeOccupied()` before accessing node properties
4. Use `updateInnerOccupancy()` after batch operations for consistency
