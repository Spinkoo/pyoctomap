# CountingOcTree Logic Explanation

## Overview

`CountingOcTree` is a specialized octree that stores an **unsigned integer counter** in each node. Unlike `OcTree` (which stores occupancy probabilities), `CountingOcTree` simply counts how many times each voxel/volume has been observed.

## Key Concepts

### 1. **Recursive Counting**
The most important concept: **Counts are recursive** - parent nodes contain the **sum** of all their children's counts.

```
        Root (count = 5)
       /                  \
   Child 1 (count = 3)  Child 2 (count = 2)
   /      |      \       /      |      \
  ...    ...    ...    ...    ...    ...
```

### 2. **Node Structure**

Each `CountingOcTreeNode`:
- Inherits from `OcTreeDataNode<unsigned int>`
- Stores a single unsigned integer value (the count)
- Provides methods: `getCount()`, `increaseCount()`, `setCount()`

## How `updateNode()` Works

When you call `updateNode([x, y, z])` or `updateNode(key)`:

### Step-by-Step Process:

1. **Convert coordinate to key** (if needed)
   ```cpp
   coordToKeyChecked(value, key)  // Maps 3D coordinate → discrete voxel key
   ```

2. **Create root if it doesn't exist**
   ```cpp
   if (root == NULL) {
     root = new CountingOcTreeNode();
     tree_size++;
   }
   ```

3. **Increment root count**
   ```cpp
   curNode = root;
   curNode->increaseCount();  // Root count = total observations in entire tree
   ```

4. **Traverse down the tree** (from root to leaf)
   ```cpp
   for (int i = tree_depth-1; i >= 0; i--) {
     unsigned int pos = computeChildIdx(k, i);  // Which child (0-7)?
     
     if (!nodeChildExists(curNode, pos)) {
       createNodeChild(curNode, pos);  // Create child if doesn't exist
     }
     
     curNode = getNodeChild(curNode, pos);
     curNode->increaseCount();  // Increment EVERY node along the path
   }
   ```

### Important Behavior:

- **Every node along the path gets incremented** - from root to the target leaf
- This maintains the recursive property: parent count = sum of children counts
- The leaf node at the target location gets incremented
- All parent nodes also get incremented

### Example:

```
Initial state (empty tree):
Root: count = 0

After updateNode([1.0, 2.0, 3.0]):
Root: count = 1
  └─ Child[5]: count = 1
      └─ Grandchild[3]: count = 1
          └─ Leaf: count = 1

After updateNode([1.0, 2.0, 3.0]) again:
Root: count = 2
  └─ Child[5]: count = 2
      └─ Grandchild[3]: count = 2
          └─ Leaf: count = 2

After updateNode([4.0, 5.0, 6.0]):
Root: count = 3
  ├─ Child[5]: count = 2  (unchanged)
  │   └─ Grandchild[3]: count = 2
  │       └─ Leaf: count = 2
  └─ Child[2]: count = 1  (new branch)
      └─ Grandchild[1]: count = 1
          └─ Leaf: count = 1
```

## How `getCentersMinHits()` Works

This method finds all leaf nodes (voxels) that have been observed at least `min_hits` times.

### Algorithm:

1. **Recursive traversal** starting from root
2. **If node has children**: Recursively visit all children
3. **If node is a leaf** (max depth reached):
   - Check if `node->getCount() >= min_hits`
   - If yes, add the voxel center coordinate to the result list

```cpp
void getCentersMinHitsRecurs(...) {
  if (depth < max_depth && nodeHasChildren(node)) {
    // Not a leaf - recurse into children
    for (unsigned int i=0; i<8; ++i) {
      if (nodeChildExists(node, i)) {
        getCentersMinHitsRecurs(...);  // Recursive call
      }
    }
  } else {
    // Leaf node reached
    if (node->getCount() >= min_hits) {
      node_centers.push_back(keyToCoord(parent_key, depth));
    }
  }
}
```

### Why Leaf Nodes?

- Only leaf nodes represent actual voxels (volumes in space)
- Parent nodes are just organizational structure
- The count at a leaf node represents observations in that specific voxel
- Parent counts are sums, so they're always >= any child count

## Key Differences from OcTree

| Feature | OcTree | CountingOcTree |
|---------|--------|----------------|
| **Data stored** | Occupancy probability (log-odds) | Observation count (unsigned int) |
| **Update semantics** | Probabilistic update (Bayes' rule) | Simple increment |
| **Parent value** | Maximum child probability | Sum of child counts |
| **Use case** | Mapping with uncertainty | Counting observations |
| **Base class** | `OccupancyOcTreeBase` | `OcTreeBase` |

## Practical Example

```python
tree = CountingOcTree(0.1)  # 0.1m resolution

# Observe location [1.0, 2.0, 3.0] three times
tree.updateNode([1.0, 2.0, 3.0])  # Count = 1
tree.updateNode([1.0, 2.0, 3.0])  # Count = 2
tree.updateNode([1.0, 2.0, 3.0])  # Count = 3

# Query the count
node = tree.search([1.0, 2.0, 3.0])
print(node.getCount())  # Output: 3

# Find frequently observed locations (count >= 2)
frequent = tree.getCentersMinHits(2)
# Returns: [[1.05, 2.05, 3.05]]  (voxel center coordinates)
```

## Memory Efficiency

- Uses octree structure: only stores nodes that have been observed
- Unobserved regions don't consume memory
- Counts propagate upward, so you can query counts at any level
- Efficient for sparse observation patterns

## Use Cases

1. **Sensor observation counting**: Count how many times each location was observed
2. **Frequency analysis**: Find frequently observed locations
3. **Data collection tracking**: Track observation density
4. **Temporal analysis**: Count observations over time (if you clear/reset periodically)

## Important Notes

- **Counts are cumulative**: Once incremented, they stay incremented (unless manually changed)
- **No automatic decay**: Unlike occupancy probabilities, counts don't decay over time
- **Integer precision**: Counts are unsigned integers (0 to 4,294,967,295)
- **Voxel centers**: `getCentersMinHits()` returns voxel center coordinates, not exact input coordinates
- **Resolution matters**: All coordinates are discretized to voxel resolution

