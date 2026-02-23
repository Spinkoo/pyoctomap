#!/usr/bin/env python3
"""
Profiling: Python-loop extraction vs Cythonized extractPointCloud.

Compares two approaches for extracting point data from a CountingOcTree:
  1. Python loop: iterate with getCentersMinHits + search per node
  2. extractPointCloud: single call wrapping C++ getCentersMinHits + batch search

Also benchmarks ColorOcTree and OcTree extraction.
"""

import time
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyoctomap import CountingOcTree, ColorOcTree, OcTree


def build_counting_tree(n_points, resolution=0.05):
    tree = CountingOcTree(resolution)
    np.random.seed(42)
    pts = np.random.rand(n_points, 3) * 20.0 - 10.0
    for pt in pts:
        tree.updateNode(pt.tolist())
    tree.updateInnerCounts()
    return tree


def build_color_tree(n_points, resolution=0.05):
    tree = ColorOcTree(resolution)
    np.random.seed(42)
    pts = np.random.rand(n_points, 3) * 20.0 - 10.0
    for pt in pts:
        tree.updateNode(pt.tolist(), True)
        r, g, b = np.random.randint(0, 256, 3)
        tree.setNodeColor(pt.tolist(), int(r), int(g), int(b))
    return tree


def build_octree(n_points, resolution=0.05):
    tree = OcTree(resolution)
    np.random.seed(42)
    pts = np.random.rand(n_points, 3) * 20.0 - 10.0
    for pt in pts:
        tree.updateNode(pt.tolist(), True)
    return tree


# CountingOcTree benchmarks
def counting_python_loop(tree):
    centers = tree.getCentersMinHits(1)
    coords = []
    counts = []
    for c in centers:
        coords.append(c)
        node = tree.search(c)
        counts.append(node.getCount() if node else 0)
    return np.array(coords, dtype=np.float64), np.array(counts, dtype=np.uint32)


def counting_cython_extract(tree):
    return tree.extractPointCloud()


# ColorOcTree benchmarks
def color_python_loop(tree):
    occ_coords = []
    occ_colors = []
    free_coords = []
    for leaf in tree.begin_leafs():
        try:
            is_occ = tree.isNodeOccupied(leaf)
        except:
            is_occ = True
        if is_occ:
            occ_coords.append(leaf.getCoordinate())
            occ_colors.append(leaf.getColor())
        else:
            free_coords.append(leaf.getCoordinate())
    occ = np.array(occ_coords, dtype=np.float64) if occ_coords else np.zeros((0, 3))
    col = np.array(occ_colors, dtype=np.uint8) if occ_colors else np.zeros((0, 3), dtype=np.uint8)
    fre = np.array(free_coords, dtype=np.float64) if free_coords else np.zeros((0, 3))
    return occ, col, fre


def color_cython_extract(tree):
    return tree.extractPointCloud()


# OcTree benchmarks
def octree_python_loop(tree):
    resolution = tree.getResolution()
    occupied = []
    empty = []
    for leaf in tree.begin_leafs():
        try:
            is_occ = tree.isNodeOccupied(leaf)
        except:
            is_occ = True
        center = np.array(leaf.getCoordinate(), dtype=np.float64)
        raw_dim = max(1, round(leaf.getSize() / resolution))
        dim = min(raw_dim, 100)
        origin = center - (dim / 2 - 0.5) * resolution
        indices = np.column_stack(np.nonzero(np.ones((dim, dim, dim))))
        points = origin + indices * np.array(resolution)
        if is_occ:
            occupied.append(points)
        else:
            empty.append(points)
    occ = np.concatenate(occupied) if occupied else np.zeros((0, 3))
    emp = np.concatenate(empty) if empty else np.zeros((0, 3))
    return occ, emp


def octree_cython_extract(tree):
    return tree.extractPointCloud()


def bench(fn, tree, repeats=3):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(tree)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times) / len(times)
    return avg, result


def main():
    sizes = [500, 2000, 10000]
    repeats = 3

    print("=" * 72)
    print("EXTRACTION PROFILING: Python loop vs extractPointCloud")
    print("=" * 72)

    # CountingOcTree
    print("\n  CountingOcTree")
    print("  {:>10s}  {:>8s}  {:>12s}  {:>12s}  {:>8s}".format(
        "N points", "Leaves", "Python loop", "extractPC", "Speedup"))
    print("  " + "-" * 60)
    for n in sizes:
        tree = build_counting_tree(n)
        n_leaves = tree.getNumLeafNodes()
        t_py, (c_py, cnt_py) = bench(counting_python_loop, tree, repeats)
        t_cy, (c_cy, cnt_cy) = bench(counting_cython_extract, tree, repeats)
        speedup = t_py / t_cy if t_cy > 0 else float("inf")
        print("  {:>10d}  {:>8d}  {:>10.2f}ms  {:>10.2f}ms  {:>7.1f}x".format(
            n, n_leaves, t_py * 1000, t_cy * 1000, speedup))
        assert len(c_py) == len(c_cy), f"Mismatch: python={len(c_py)} vs cython={len(c_cy)}"

    # ColorOcTree
    print("\n  ColorOcTree")
    print("  {:>10s}  {:>8s}  {:>12s}  {:>12s}  {:>8s}".format(
        "N points", "Leaves", "Python loop", "extractPC", "Speedup"))
    print("  " + "-" * 60)
    for n in sizes:
        tree = build_color_tree(n)
        n_leaves = tree.size()
        t_py, _ = bench(color_python_loop, tree, repeats)
        t_cy, _ = bench(color_cython_extract, tree, repeats)
        speedup = t_py / t_cy if t_cy > 0 else float("inf")
        print("  {:>10d}  {:>8d}  {:>10.2f}ms  {:>10.2f}ms  {:>7.1f}x".format(
            n, n_leaves, t_py * 1000, t_cy * 1000, speedup))

    # OcTree (sub-voxel grid)
    print("\n  OcTree (sub-voxel grid expansion)")
    print("  {:>10s}  {:>8s}  {:>12s}  {:>12s}  {:>8s}".format(
        "N points", "Leaves", "Python loop", "extractPC", "Speedup"))
    print("  " + "-" * 60)
    for n in sizes:
        tree = build_octree(n)
        n_leaves = tree.size()
        t_py, _ = bench(octree_python_loop, tree, repeats)
        t_cy, _ = bench(octree_cython_extract, tree, repeats)
        speedup = t_py / t_cy if t_cy > 0 else float("inf")
        print("  {:>10d}  {:>8d}  {:>10.2f}ms  {:>10.2f}ms  {:>7.1f}x".format(
            n, n_leaves, t_py * 1000, t_cy * 1000, speedup))

    print("\nDone.")


if __name__ == "__main__":
    main()
