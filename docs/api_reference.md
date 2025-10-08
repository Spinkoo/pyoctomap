# API Reference

Complete API documentation for PyOctoMap.

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

**Note on Consolidation**: `insertPointCloud` and `insertPointCloudFast` are functionally equivalent (shared internal logic via a common helper) – use either. Both support the same parameters and produce identical results.

```python
addPointWithRayCasting(point, sensor_origin, update_inner_occupancy=False)
```

- `point` (np.array): 3D point to add
- `sensor_origin` (np.array): Sensor origin for ray casting
- `update_inner_occupancy` (bool): Update inner node occupancy
- Returns: `bool` (success)

```python
addPointCloudWithRayCasting(point_cloud, sensor_origin, max_range=-1.0, update_inner_occupancy=True, discretize=False)
```

- `point_cloud` (np.array): Nx3 array of points
- `sensor_origin` (np.array): Sensor origin for ray casting
- `max_range` (float): Maximum range for points
- `update_inner_occupancy` (bool): Update inner node occupancy
- `discretize` (bool): Reduce duplicates for dense clouds
- Returns: `int` (points added)

```python
insertPointCloudFast(point_cloud, sensor_origin, max_range=-1.0, discretize=False, lazy_eval=False)
```

- `point_cloud` (np.array): Nx3 array of points
- `sensor_origin` (np.array): Sensor origin for ray casting
- `max_range` (float): Maximum range for points
- `discretize` (bool): Reduce duplicates for dense clouds
- `lazy_eval` (bool): Defer inner node occupancy updates (call `updateInnerOccupancy` manually later)
- Returns: `int` (points processed)

```python
insertPointCloudRaysFast(point_cloud, sensor_origin, max_range=-1.0, lazy_eval=False)
```

- `point_cloud` (np.array): Nx3 array of points
- `sensor_origin` (np.array): Sensor origin for ray casting
- `max_range` (float): Maximum range for points
- `lazy_eval` (bool): Defer inner node occupancy updates (call `updateInnerOccupancy` manually later)
- Returns: `int` (points processed)

```python
markFreeSpaceAlongRay(origin, end_point, max_range=-1.0)
```

- `origin` (np.array): Ray start point
- `end_point` (np.array): Ray end point
- `max_range` (float): Maximum ray range
- No return (void)

```python
insertPointCloud(point_cloud, sensor_origin, max_range=-1.0, lazy_eval=False, discretize=False)
```

- `point_cloud` (np.array): Nx3 array of points
- `sensor_origin` (np.array): Sensor origin for ray casting
- `max_range` (float): Maximum range for points
- `lazy_eval` (bool): Defer inner node occupancy updates (call `updateInnerOccupancy` manually later)
- `discretize` (bool): Reduce duplicates for dense clouds
- Returns: `int` (points processed)
- Note: Equivalent to `insertPointCloudFast` (shared logic).

#### Iterators

PyOctoMap provides three iterator types for different traversal needs:

```python
begin_tree(maxDepth=0)
```

- **Purpose**: Iterate over ALL nodes in the tree (inner nodes + leaf nodes)
- **Use cases**: Tree structure analysis, debugging, inner node operations, performance profiling
- **Performance**: ~2-3x slower than `begin_leafs` (visits more nodes)
- **Returns**: `SimpleTreeIterator`

```python
begin_leafs(maxDepth=0)
```

- **Purpose**: Iterate over leaf nodes only (actual occupancy data)
- **Use cases**: Standard occupancy queries, fast iteration, most common operations
- **Performance**: Fastest iterator (visits only data nodes)
- **Returns**: `SimpleLeafIterator`

```python
begin_leafs_bbx(bbx_min, bbx_max, maxDepth=0)
```

- **Purpose**: Iterate over leaf nodes within a spatial bounding box
- **Use cases**: Region-specific analysis, spatial queries, filtered iteration
- **Performance**: Fast (filtered by spatial bounds)
- **Parameters**:
  - `bbx_min` (np.array): Bounding box minimum corner
  - `bbx_max` (np.array): Bounding box maximum corner
- **Returns**: `SimpleLeafBBXIterator`

#### Iterator Performance Comparison

| Iterator | Nodes Visited | Use Case | Performance |
|----------|---------------|----------|-------------|
| `begin_tree()` | ALL nodes (inner + leaves) | Tree analysis, debugging | ~2-3x slower |
| `begin_leafs()` | Leaf nodes only | Occupancy queries | Fastest |
| `begin_leafs_bbx()` | Leaves in bounding box | Spatial queries | Fast (filtered) |

**Iterator Selection Guide:**
- **99% of cases**: Use `begin_leafs()` for standard occupancy operations
- **Tree analysis**: Use `begin_tree()` for structure debugging and inner node access
- **Spatial queries**: Use `begin_leafs_bbx()` for region-specific operations

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

Iterates over ALL nodes in the tree (inner nodes + leaf nodes). This provides complete access to the octree structure but is slower than leaf-only iterators.

**Key characteristics:**
- Visits both inner nodes (tree structure) and leaf nodes (occupancy data)
- ~2-3x slower than `SimpleLeafIterator` (visits more nodes)
- Essential for tree structure analysis and debugging
- Allows access to inner node properties and hierarchy

```python
# Example: Analyze tree structure
for node_it in tree.begin_tree():
    coord = node_it.getCoordinate()
    depth = node_it.getDepth()
    size = node_it.getSize()
    is_leaf = node_it.isLeaf()
    
    if not is_leaf:
        print(f"Inner node at depth {depth}, size {size:.2f}m")
    else:
        print(f"Leaf node at {coord}, depth {depth}")

# Example: Count nodes by depth
depth_counts = {}
for node_it in tree.begin_tree():
    depth = node_it.getDepth()
    depth_counts[depth] = depth_counts.get(depth, 0) + 1
```

### SimpleLeafIterator

Iterates over leaf nodes only (actual occupancy data). This is the fastest iterator and most commonly used for standard occupancy operations.

**Key characteristics:**
- Visits only leaf nodes (contain actual occupancy data)
- Fastest iterator (visits only data nodes, not structure)
- Most common choice for occupancy queries
- Cannot access inner node properties

```python
# Example: Standard occupancy iteration
for leaf_it in tree.begin_leafs():
    coord = leaf_it.getCoordinate()
    depth = leaf_it.getDepth()
    size = leaf_it.getSize()
    is_leaf = leaf_it.isLeaf()  # Always True for leaf iterator
    node = leaf_it.current_node
    
    # Check occupancy
    if tree.isNodeOccupied(leaf_it):
        print(f"Occupied leaf at {coord}")

# Example: Count occupied vs free nodes
occupied_count = 0
free_count = 0
for leaf_it in tree.begin_leafs():
    if tree.isNodeOccupied(leaf_it):
        occupied_count += 1
    else:
        free_count += 1
```

### SimpleLeafBBXIterator

Iterates over leaf nodes within a spatial bounding box. This provides spatial filtering for region-specific analysis.

**Key characteristics:**
- Visits only leaf nodes within the specified bounding box
- Fast performance (spatially filtered)
- Useful for region-specific queries and analysis
- Same API as `SimpleLeafIterator` but with spatial filtering

```python
# Example: Analyze specific region
bbx_min = np.array([0.0, 0.0, 0.0])
bbx_max = np.array([10.0, 10.0, 10.0])
for bbx_it in tree.begin_leafs_bbx(bbx_min, bbx_max):
    coord = bbx_it.getCoordinate()
    depth = bbx_it.getDepth()
    size = bbx_it.getSize()
    is_leaf = bbx_it.isLeaf()  # Always True
    node = bbx_it.current_node
    
    # Process only nodes in the bounding box
    if tree.isNodeOccupied(bbx_it):
        print(f"Occupied node in region: {coord}")

# Example: Count nodes in different regions
regions = [
    (np.array([0, 0, 0]), np.array([5, 5, 5])),    # Bottom-left
    (np.array([5, 5, 5]), np.array([10, 10, 10]))  # Top-right
]

for i, (min_pt, max_pt) in enumerate(regions):
    count = 0
    for bbx_it in tree.begin_leafs_bbx(min_pt, max_pt):
        count += 1
    print(f"Region {i}: {count} nodes")
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
