#!/usr/bin/env python3
"""
ROS package setup for pyoctomap
"""

from setuptools import setup
import os
from glob import glob

package_name = 'pyoctomap'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/examples', glob('examples/*.py')),
    ],
    install_requires=[
        'numpy>=1.16.0',
        'rclpy>=3.0.0',
        'sensor_msgs>=4.0.0',
        'geometry_msgs>=4.0.0',
        'nav_msgs>=4.0.0',
        'tf2_ros>=0.25.0',
    ],
    zip_safe=True,
    maintainer='Spinkoo',
    maintainer_email='lespinkoo@gmail.com',
    description='Python binding of the OctoMap library with bundled shared libraries for ROS integration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'octomap_node = pyoctomap.ros2_octomap_node:main',
        ],
    },
)
