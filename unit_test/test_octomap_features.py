#!/usr/bin/env python3
"""
Comprehensive OctoMap feature test script.

Covers:
- Basic node updates and queries
- coordToKey / keyToCoord roundtrip
- Insert point cloud
- Ray casting
- Iterators (tree/leaf/BBX) if available
- Labels extraction
- File save/load (binary and text)
- DynamicEDT distance queries (if available)
"""

import os
import sys
import tempfile
import numpy as np


def section(title: str):
    print("\n" + "=" * 8 + f" {title} " + "=" * 8)


def main():
    section("Import")
    import pyoctomap
    print("pyoctomap imported from:", getattr(pyoctomap, "__file__", None))
    if hasattr(pyoctomap, "get_package_info"):
        info = pyoctomap.get_package_info()
        print("package info:", info)

    # Create a tree
    section("Create OcTree")
    tree = pyoctomap.OcTree(0.1)
    print("Resolution:", tree.getResolution())

    # Basic updates
    section("Basic Updates & Queries")
    pts = [
        [0.2, 0.2, 0.2],
        [0.5, 0.2, 0.2],
        [0.8, 0.2, 0.2],
    ]
    for p in pts:
        tree.updateNode(p, True)
    print("Updated:", len(pts), "nodes -> size:", tree.size())
    n = tree.search(pts[0])
    print("search -> node exists:", bool(n), "occupied:", (tree.isNodeOccupied(n) if n else None))

    # coordToKey / keyToCoord
    section("coordToKey / keyToCoord")
    coord = np.array([0.25, 0.25, 0.25], dtype=np.float64)
    key = tree.coordToKey(coord)
    back = tree.keyToCoord(key)
    print("coord:", coord.tolist(), "-> key:", [key[0], key[1], key[2]], "-> back:", back.tolist())

    # Insert point cloud
    section("Insert Point Cloud")
    pc = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [1.0, 0.2, 0.0],
        [1.0, 0.3, 0.0],
    ], dtype=np.float64)
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    tree.insertPointCloud(pc, origin)
    print("pointcloud inserted -> size:", tree.size())

    # Ray casting
    section("Ray Casting")
    end = np.zeros(3, dtype=np.float64)
    hit = tree.castRay(np.array([0.0, 0.0, 0.0], dtype=np.float64),
                       np.array([1.0, 0.0, 0.0], dtype=np.float64), end,
                       ignoreUnknownCells=False, maxRange=-1.0)
    print("castRay hit:", bool(hit), "end:", end.tolist())

    # Iterators (if exposed)
    section("Iterators")
    if hasattr(tree, "begin_tree"):
        try:
            count = 0
            for it in tree.begin_tree(0):
                count += 1
                if count >= 10:
                    break
            print("begin_tree available -> iterated:", count)
        except Exception as e:
            print("begin_tree error:", e)
    else:
        print("begin_tree not available; skipping")

    if hasattr(tree, "begin_leafs"):
        try:
            count = 0
            for it in tree.begin_leafs(0):
                count += 1
                if count >= 10:
                    break
            print("begin_leafs available -> iterated:", count)
        except Exception as e:
            print("begin_leafs error:", e)
    else:
        print("begin_leafs not available; skipping")

    if hasattr(tree, "begin_leafs_bbx"):
        try:
            bbx_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            bbx_max = np.array([1.0, 0.5, 0.5], dtype=np.float64)
            count = 0
            for it in tree.begin_leafs_bbx(bbx_min, bbx_max, 0):
                count += 1
                if count >= 10:
                    break
            print("begin_leafs_bbx available -> iterated:", count)
        except Exception as e:
            print("begin_leafs_bbx error:", e)
    else:
        print("begin_leafs_bbx not available; skipping")

    # Labels
    section("Labels")
    samples = np.array([
        [0.2, 0.2, 0.2],      # should exist
        [1.0, 0.0, 0.0],      # should exist after point cloud insertion
        [2.0, 2.0, 2.0],      # likely unknown
    ], dtype=np.float64)
    # Preferred: use provided API
    try:
        labels = tree.getLabels(samples)
        print("getLabels:", labels.tolist(), "(-1 unknown, 0 free, 1 occupied)")
    except Exception as e:
        print("getLabels error:", e)
        # Fallback manual labeling
        manual = []
        for p in samples:
            node = tree.search(p)
            if not node:
                manual.append(-1)
            else:
                manual.append(1 if tree.isNodeOccupied(node) else 0)
        print("manual labels:", manual)

    # Save / Load
    section("Save / Load")
    with tempfile.TemporaryDirectory() as td:
        bt_path = os.path.join(td, "test.bt")
        ot_path = os.path.join(td, "test.ot")
        ok_bt = tree.writeBinary(bt_path)
        ok_ot = tree.write(ot_path)
        print("writeBinary:", bool(ok_bt), "write text:", bool(ok_ot))
        # Read back using matching APIs
        try:
            loaded_bt_ok = tree.readBinary(bt_path)
            print("readBinary(bt) ->", bool(loaded_bt_ok))
        except Exception as e:
            print("readBinary error:", e)
        try:
            loaded_text = tree.read(ot_path)
            print("read(text) -> loaded size:", (loaded_text.size() if loaded_text else None))
        except Exception as e:
            print("read(text) error:", e)

    # Dynamic EDT (if exposed)
    section("DynamicEDT")
    try:
        # bounding box around the points we've set
        bbx_min = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
        bbx_max = np.array([1.5, 1.0, 0.5], dtype=np.float64)
        if hasattr(tree, "dynamicEDT_generate"):
            tree.dynamicEDT_generate(2.0, bbx_min, bbx_max, False)
            if hasattr(tree, "dynamicEDT_getDistance"):
                d1 = tree.dynamicEDT_getDistance(np.array([0.2, 0.2, 0.2], dtype=np.float64))
                d2 = tree.dynamicEDT_getDistance(np.array([0.9, 0.0, 0.0], dtype=np.float64))
                print("EDT distances:", float(d1), float(d2))
            else:
                print("dynamicEDT_getDistance not available")
        else:
            print("dynamicEDT_generate not available; skipping")
    except Exception as e:
        print("DynamicEDT error:", e)

    print("\nAll feature tests finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n❌ Test failed:", exc)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)


