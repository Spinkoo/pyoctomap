## Performance Guide

This page focuses on **practical tuning guidelines** rather than exhaustive
benchmarks. For end‑to‑end usage patterns, see the examples in `README.md`
and the `examples/` directory.

---

## High‑Level Principles

- **Use batch operations** instead of per‑point Python loops.
- **Pick a realistic resolution** – higher detail costs memory and time.
- **Defer inner‑node updates** when inserting many points.
- **Filter your data early** (e.g., remove obvious outliers or downsample).

---

## Recommended Patterns

### 1. Batch Insertion

Avoid:

```python
for p in points:                # slow
    tree.updateNode(p, True)
```

Prefer:

```python
tree.insertPointCloud(points, sensor_origin,
                      max_range=50.0,
                      lazy_eval=True)
tree.updateInnerOccupancy()     # once per batch
```

For very large clouds where duplicated rays are acceptable, consider
`insertPointCloudRaysFast(...)` for maximum throughput.

### 2. Resolution vs. Memory

Rough intuition:

- **0.01–0.05 m** – fine detail, small local scenes, higher memory and CPU.
- **0.1–0.2 m** – general‑purpose robotics mapping (good compromise).
- **0.5–1.0 m** – large‑scale environment overviews, low memory.

You can often start with `0.1` and adjust after profiling.

### 3. Iteration Strategy

Use the narrowest iterator that still answers your question:

- `begin_leafs()` – default choice for occupancy queries.
- `begin_leafs_bbx(min, max)` – if you are only interested in a region.
- `begin_tree()` – only when you really need internal structure.

Limiting `maxDepth` can further reduce work when you only care about coarse
structure.

---

## Dynamic and Streaming Scenarios

### 1. Moving Sensors (LIDAR / Depth Cameras)

The helper `decayAndInsertPointCloud(...)` is designed for dynamic environments:

1. Decays old occupancy inside the scan’s bounding box.
2. Inserts the new point cloud.

This greatly reduces “ghost” artifacts as obstacles move or disappear.

Key parameter: `logodd_decay_value` (must be negative). See the decay tuning
formula in `docs/api_reference.md` for guidance.

### 2. Incremental Batching

When data arrives at high frequency:

```python
buffer = []

def add_scan(points):
    buffer.extend(points)
    if len(buffer) >= 10_000:
        arr = np.asarray(buffer, dtype=np.float64).reshape(-1, 3)
        tree.insertPointCloud(arr, sensor_origin, lazy_eval=True)
        buffer.clear()
        tree.updateInnerOccupancy()
```

This keeps Python overhead low while still updating the map frequently.

---

## Practical Profiling

### 1. Timing Critical Sections

```python
import time

start = time.time()
tree.insertPointCloud(points, sensor_origin, lazy_eval=True)
tree.updateInnerOccupancy()
elapsed = time.time() - start
print(f"Update took {elapsed:.3f} s")
```

Use this to compare different resolutions, batch sizes, or decay settings.

### 2. Monitoring Memory

On CPython you can use `psutil`:

```python
import os, psutil

process = psutil.Process(os.getpid())
print(f"RSS: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

Check before and after large insertions to understand footprint.

---

## Checklist for Slow Workloads

- Are you **looping in Python** instead of calling batch methods?
- Is the **resolution unnecessarily small** for your task?
- Are you calling `updateInnerOccupancy()` too often?
- Can you **limit the region** you iterate over with `begin_leafs_bbx()`?
- Is your input data **pre‑filtered and downsampled**?

If performance is still an issue after following this guide, consider opening
an issue with:

- A short code snippet,
- Approximate point counts and resolutions,
- Platform details (Python version, OS, CPU/GPU),
so that we can provide more targeted advice.


