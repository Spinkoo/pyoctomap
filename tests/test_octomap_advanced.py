#!/usr/bin/env python3
"""
OctoMap Advanced Feature Test

Exercises a wide variety of operations and prints concise results.
Safe-guards are added so missing bindings won't abort the run.
Exit code is 0 if all critical checks pass, non-zero otherwise.
"""

import os
import sys
import math
import tempfile
from typing import List, Tuple

import numpy as np
from itertools import product


def h(title: str) -> None:
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


def try_call(desc: str, fn, *args, critical=False, **kwargs):
    try:
        out = fn(*args, **kwargs)
        print(f"{desc}: OK", ("-> " + str(out)) if out is not None else "")
        return True, out
    except Exception as e:
        print(f"{desc}: ERROR -> {e}")
        return (not critical), None


def generate_room_points(size=4.0, step=0.1) -> Tuple[np.ndarray, np.ndarray]:
    # Simple room: walls + a pillar
    walls = []
    # floor/ceiling
    grid = np.arange(0, size, step)
    for x in grid:
        for y in grid:
            walls.append([x, y, 0.0])
            walls.append([x, y, 2.0])
    # four walls
    for z in np.arange(0.0, 2.0, step):
        for y in grid:
            walls.append([0.0, y, z])
            walls.append([size, y, z])
        for x in grid:
            walls.append([x, 0.0, z])
            walls.append([x, size, z])

    pillar = []
    cx, cy, r = 2.0, 2.0, 0.3
    for x in np.arange(cx - r, cx + r + step, step):
        for y in np.arange(cy - r, cy + r + step, step):
            for z in np.arange(0.1, 1.8, step):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    pillar.append([x, y, z])
    return np.array(walls, dtype=np.float64), np.array(pillar, dtype=np.float64)


def main() -> int:
    import pyoctomap
    failures = 0

    h("Import")
    print("pyoctomap from:", getattr(pyoctomap, "__file__", None))
    if hasattr(pyoctomap, "get_package_info"):
        print(pyoctomap.get_package_info())

    # Basic tree
    h("Create Tree")
    ok, tree = try_call("OcTree(0.1)", pyoctomap.OcTree, 0.1, critical=True)
    if not ok:
        return 1

    # Parameter getters/setters
    h("Parameters & Thresholds")
    for getter in [
        tree.getClampingThresMax,
        tree.getClampingThresMin,
        tree.getOccupancyThres,
        tree.getProbHit,
        tree.getProbMiss,
    ]:
        try_call(getter.__name__, getter)
    try_call("setClampingThresMax(0.97)", tree.setClampingThresMax, 0.97)
    try_call("setClampingThresMin(0.12)", tree.setClampingThresMin, 0.12)
    try_call("setOccupancyThres(0.5)", tree.setOccupancyThres, 0.5)
    try_call("setProbHit(0.7)", tree.setProbHit, 0.7)
    try_call("setProbMiss(0.4)", tree.setProbMiss, 0.4)

    # Basic updates
    h("Node Updates & Queries")
    pts = np.array(
        [
            [0.2, 0.2, 0.2],
            [0.5, 0.2, 0.2],
            [0.8, 0.2, 0.2],
        ],
        dtype=np.float64,
    )
    for p in pts:
        try_call(f"updateNode({p.tolist()}, True)", tree.updateNode, p, True)
    try_call("size", tree.size)
    ok, node = try_call("search([0.2,0.2,0.2])", tree.search, [0.2, 0.2, 0.2])
    if node is not None:
        try_call("isNodeOccupied(node)", tree.isNodeOccupied, node)

    # coordToKey / keyToCoord roundtrip
    h("coordToKey / keyToCoord")
    coord = np.array([0.25, 0.25, 0.25], dtype=np.float64)
    ok, key = try_call("coordToKey", tree.coordToKey, coord)
    if key is not None:
        try_call("keyToCoord", tree.keyToCoord, key)

    # Insert a small synthetic point cloud
    h("Insert Point Cloud")
    pc = np.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.2, 0.0]], dtype=np.float64)
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    try_call("insertPointCloud", tree.insertPointCloud, pc, origin)
    try_call("updateInnerOccupancy", tree.updateInnerOccupancy)
    try_call("toMaxLikelihood", tree.toMaxLikelihood)
    # Ray casting in several directions
    h("Ray Casting")
    for ang in np.linspace(0, math.pi / 2, 4):
        dir_vec = np.array([math.cos(ang), math.sin(ang), 0.0], dtype=np.float64)
        end = np.zeros(3, dtype=np.float64)
        ok, hit = try_call(f"castRay dir={dir_vec.tolist()}", tree.castRay, origin, dir_vec, end)
        if hit:
            print("  -> hit at", end.tolist())

    # Region building: walls + pillar, and query labels
    h("Room Build & Labels")
    walls, pillar = generate_room_points(size=2.0, step=0.2)
    for p in walls[:200]:
        tree.updateNode(p, True)
    for p in pillar[:200]:
        tree.updateNode(p, True)
    samples = np.array([[0.2, 0.2, 0.1], [1.0, 1.0, 0.5], [3.0, 3.0, 3.0]], dtype=np.float64)

    # 1) Direct labels at provided coordinates
    direct_labels = None
    try:
        direct_labels = tree.getLabels(samples).tolist()
    except Exception as e:
        print("getLabels(direct) error ->", e)
        direct_labels = []
        for p in samples:
            n = tree.search(p)
            if not n:
                direct_labels.append(-1)
            else:
                direct_labels.append(1 if tree.isNodeOccupied(n) else 0)
    print("Direct labels (-1 unknown, 0 free, 1 occupied):", direct_labels)

    # 2) Snap samples to voxel centers
    snapped = []
    for p in samples:
        key = tree.coordToKey(p)
        center = tree.keyToCoord(key)
        snapped.append(center)
    snapped = np.array(snapped, dtype=np.float64)
    snapped_labels = tree.getLabels(snapped).tolist()
    print("Snapped centers:", [c.tolist() for c in snapped])
    print("Snapped labels:", snapped_labels)

    # 3) Nearest-voxel probing in 3x3x3 neighborhood around key
    nearest_labels = []
    for p in samples:
        key = tree.coordToKey(p)
        found = -1
        for dx, dy, dz in product([-1, 0, 1], repeat=3):
            try:
                nkey = pyoctomap.OcTreeKey()
                nkey[0] = key[0] + dx
                nkey[1] = key[1] + dy
                nkey[2] = key[2] + dz
                node = tree.search(nkey)
                if node:
                    found = 1 if tree.isNodeOccupied(node) else 0
                    break
            except Exception:
                continue
        nearest_labels.append(found)
    print("Nearest-voxel labels (3x3x3):", nearest_labels)

    # Label counts summary
    def counts(arr: list) -> dict:
        out = {"occupied": 0, "free": 0, "unknown": 0}
        for v in arr:
            if v == 1:
                out["occupied"] += 1
            elif v == 0:
                out["free"] += 1
            else:
                out["unknown"] += 1
        return out

    print("Counts -> direct:", counts(direct_labels),
          "snapped:", counts(snapped_labels),
          "nearest:", counts(nearest_labels))

    # Iterators if available
    h("Iterators")
    if hasattr(tree, "begin_tree"):
        cnt = 0
        try:
            for it in tree.begin_tree(0):
                cnt += 1
                if cnt >= 20:
                    break
            print("begin_tree ->", cnt)
        except Exception as e:
            print("begin_tree error:", e)
    else:
        print("begin_tree not available")

    if hasattr(tree, "begin_leafs"):
        cnt = 0
        try:
            for it in tree.begin_leafs(0):
                cnt += 1
                if cnt >= 20:
                    break
            print("begin_leafs ->", cnt)
        except Exception as e:
            print("begin_leafs error:", e)
    else:
        print("begin_leafs not available")

    if hasattr(tree, "begin_leafs_bbx"):
        cnt = 0
        try:
            bbx_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            bbx_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            for it in tree.begin_leafs_bbx(bbx_min, bbx_max, 0):
                cnt += 1
                if cnt >= 20:
                    break
            print("begin_leafs_bbx ->", cnt)
        except Exception as e:
            print("begin_leafs_bbx error:", e)
    else:
        print("begin_leafs_bbx not available")

    # BBX operations & iterators
    h("BBX Operations & Iterators")
    bbx_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    bbx_max = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    try_call("setBBXMin", tree.setBBXMin, bbx_min)
    try_call("setBBXMax", tree.setBBXMax, bbx_max)
    try_call("useBBXLimit(True)", tree.useBBXLimit, True)
    # Query BBX getters
    for getter in [tree.getBBXMin, tree.getBBXMax, tree.getBBXCenter, tree.getBBXBounds]:
        try_call(getter.__name__, getter)
    # Iterate using BBX iterator if available and verify points are inside BBX via inBBX
    if hasattr(tree, "begin_leafs_bbx"):
        cnt_bbx = 0
        inside_all = True
        try:
            for it in tree.begin_leafs_bbx(bbx_min, bbx_max, 0):
                cnt_bbx += 1
                coord = np.array(it.getCoordinate(), dtype=np.float64)
                if not bool(tree.inBBX(coord)):
                    inside_all = False
                    break
                if cnt_bbx >= 50:
                    break
            print(f"BBX leafs count(sampled up to 50): {cnt_bbx}, all inBBX: {inside_all}")
        except Exception as e:
            print("BBX iterator error:", e)
    else:
        print("begin_leafs_bbx not available")
    try_call("useBBXLimit(False)", tree.useBBXLimit, False)

    # File save/load
    h("Save / Load")
    with tempfile.TemporaryDirectory() as td:
        bt = os.path.join(td, "map.bt")
        ot = os.path.join(td, "map.ot")
        try_call("writeBinary(.bt)", tree.writeBinary, bt)
        try:
            ok_rb, _ = try_call("readBinary(.bt)", tree.readBinary, bt)
        except Exception as e:
            print("readBinary(.bt) error:", e)
        ok_wt, _ = try_call("write(.ot)", tree.write, ot)
        ok_rt, new_tree = try_call("read(.ot)", tree.read, ot)
        if new_tree is not None:
            try_call("loaded.size", new_tree.size)

    # DynamicEDT (if present)
    h("DynamicEDT")
    try:
        bbx_min = np.array([-0.2, -0.2, -0.2], dtype=np.float64)
        bbx_max = np.array([1.2, 1.2, 1.2], dtype=np.float64)
        if hasattr(tree, "dynamicEDT_generate"):
            try_call("dynamicEDT_generate", tree.dynamicEDT_generate, 2.0, bbx_min, bbx_max, False)
            try_call("dynamicEDT_checkConsistency", tree.dynamicEDT_checkConsistency)
            try_call("dynamicEDT_update(False)", tree.dynamicEDT_update, False)
            if hasattr(tree, "dynamicEDT_getMaxDist"):
                try_call("dynamicEDT_getMaxDist", tree.dynamicEDT_getMaxDist)
            if hasattr(tree, "dynamicEDT_getDistance"):
                try_call("dynamicEDT_getDistance([0.2,0.2,0.2])", tree.dynamicEDT_getDistance, np.array([0.2, 0.2, 0.2], dtype=np.float64))
        else:
            print("DynamicEDT not available; skipping")
    except Exception as e:
        print("DynamicEDT error:", e)

    h("Stats")
    for getter in [tree.memoryUsage, tree.memoryUsageNode, tree.volume, tree.getTreeDepth, tree.getNumLeafNodes]:
        try_call(getter.__name__, getter)

    print("\n✅ Advanced feature test complete")
    return failures


if __name__ == "__main__":
    rc = 0
    try:
        rc = main()
    except Exception as exc:
        print("❌ Fatal:", exc)
        import traceback
        traceback.print_exc()
        rc = 1
    sys.exit(rc)


