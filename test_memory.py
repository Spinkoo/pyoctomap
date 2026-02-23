#!/usr/bin/env python3
"""
Quick test for iterator memory cleanup
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyoctomap
import gc
import weakref

# Test basic iterator creation and cleanup
tree = pyoctomap.ColorOcTree(0.1)
tree.updateNode([1.0, 2.0, 3.0], True)
tree.setNodeColor([1.0, 2.0, 3.0], 255, 0, 0)

print('Creating iterator...')
iterator = tree.begin_leafs()
iterator_ref = weakref.ref(iterator)

print('Using iterator...')
count = 0
for leaf in iterator:
    color = leaf.getColor()
    print(f'Found color: {color}')
    count += 1

print(f'Found {count} leaves')

print('Deleting iterator...')
del iterator
gc.collect()

print('Checking if iterator was collected...')
result = iterator_ref()
if result is None:
    print('✅ Iterator was properly garbage collected')
else:
    print('❌ Iterator was NOT garbage collected')
    print(f'Iterator still exists: {result}')
    # Check if it has a tree reference
    if hasattr(result, '_tree'):
        print(f'Iterator._tree = {result._tree}')
    else:
        print('Iterator has no _tree attribute')




