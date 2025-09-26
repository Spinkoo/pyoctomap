# File Format Documentation

PyOctoMap supports several file formats for saving and loading octree data.

## Supported Formats

### Binary Tree Format (.bt)

The primary format used by OctoMap for efficient storage and loading.

**Characteristics:**
- **Efficient**: Compressed binary format
- **Portable**: Cross-platform compatible
- **Standard**: Compatible with OctoMap tools
- **Fast**: Optimized for quick loading/saving

**Usage:**
```python
# Save tree
tree.write("my_map.bt")

# Load tree
loaded_tree = tree.read("my_map.bt")
```

**File Structure:**
- Header with metadata (resolution, tree type, etc.)
- Compressed octree data
- Node occupancy information
- Log-odds values

### Binary Format (.bt) - Alternative Method

Direct binary serialization without file I/O.

**Usage:**
```python
# Save to binary data
binary_data = tree.writeBinary()

# Load from binary data
tree.readBinary("my_map.bt")
```

## File Format Details

### Header Information

Each .bt file contains:
- **Resolution**: Tree resolution in meters
- **Tree Type**: Octree implementation type
- **Version**: File format version
- **Metadata**: Additional tree properties

### Data Compression

The octree data is stored using:
- **Spatial compression**: Only stores non-empty nodes
- **Hierarchical structure**: Parent-child relationships
- **Log-odds encoding**: Probabilistic occupancy values

### Compatibility

**OctoMap Tools:**
- Compatible with all standard OctoMap tools
- Can be loaded in C++ OctoMap applications
- Supports visualization in octovis

**Cross-Platform:**
- Works on Linux, Windows (via WSL), macOS
- Endian-independent
- Architecture-independent

## Performance Considerations

### File Size

File size depends on:
- **Resolution**: Higher resolution = larger files
- **Occupied volume**: More occupied space = larger files
- **Tree depth**: Deeper trees = larger files

**Typical sizes:**
- Small room (5x5x3m, 0.1m resolution): ~100KB
- Large environment (50x50x10m, 0.05m resolution): ~10MB
- High-resolution scan (0.01m resolution): ~100MB+

### Loading Performance

**Factors affecting load time:**
- File size
- Disk I/O speed
- Memory allocation
- Tree reconstruction

**Optimization tips:**
- Use appropriate resolution for your application
- Consider lazy loading for large files
- Pre-allocate memory when possible

## Best Practices

### File Naming

```python
# Good naming conventions
tree.write("map_2024_01_15_10cm.bt")
tree.write("office_floor1_v2.bt")
tree.write("scan_room_001.bt")
```

### Error Handling

```python
try:
    tree.write("my_map.bt")
    print("Map saved successfully")
except Exception as e:
    print(f"Failed to save map: {e}")

try:
    loaded_tree = tree.read("my_map.bt")
    if loaded_tree:
        print("Map loaded successfully")
    else:
        print("Failed to load map")
except Exception as e:
    print(f"Error loading map: {e}")
```

### Backup Strategy

```python
import shutil
from datetime import datetime

def save_with_backup(tree, filename):
    # Create backup if file exists
    if os.path.exists(filename):
        backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filename, backup_name)
    
    # Save new file
    tree.write(filename)
    print(f"Map saved: {filename}")

# Usage
save_with_backup(tree, "my_map.bt")
```

## Troubleshooting

### Common Issues

**File not found:**
```python
import os
if not os.path.exists("my_map.bt"):
    print("File does not exist")
```

**Permission errors:**
```python
try:
    tree.write("/path/to/map.bt")
except PermissionError:
    print("Permission denied - check file permissions")
```

**Corrupted files:**
```python
try:
    loaded_tree = tree.read("corrupted.bt")
except Exception as e:
    print(f"File appears to be corrupted: {e}")
    # Try to recover or use backup
```

### File Validation

```python
def validate_bt_file(filename):
    """Validate a .bt file before loading"""
    if not os.path.exists(filename):
        return False, "File does not exist"
    
    if os.path.getsize(filename) == 0:
        return False, "File is empty"
    
    try:
        test_tree = octomap.OcTree(0.1)
        loaded_tree = test_tree.read(filename)
        if loaded_tree and loaded_tree.size() > 0:
            return True, "File is valid"
        else:
            return False, "File appears to be empty or corrupted"
    except Exception as e:
        return False, f"Error reading file: {e}"

# Usage
is_valid, message = validate_bt_file("my_map.bt")
print(f"File validation: {message}")
```

## Integration with Other Tools

### OctoMap Tools

```bash
# Convert to other formats using OctoMap tools
bt2vrml my_map.bt my_map.wrl
octree2pointcloud my_map.bt my_map.pcd
```

### ROS Integration

```python
# Save in ROS-compatible format
tree.write("/tmp/ros_map.bt")

# Load from ROS bag
import rosbag
bag = rosbag.Bag('map.bag')
for topic, msg, t in bag.read_messages(topics=['/octomap']):
    # Process ROS message
    pass
```

### Cloud Storage

```python
import boto3

def upload_to_s3(tree, bucket, key):
    """Upload octree to S3"""
    tree.write("/tmp/temp_map.bt")
    s3 = boto3.client('s3')
    s3.upload_file("/tmp/temp_map.bt", bucket, key)
    os.remove("/tmp/temp_map.bt")

def download_from_s3(bucket, key, local_path):
    """Download octree from S3"""
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_path)
    return octomap.OcTree(0.1).read(local_path)
```
