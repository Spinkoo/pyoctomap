# flake8: noqa
"""
GitHub2PyPI utility for PyOctoMap

This module provides URL replacement functionality to convert relative URLs
in README.md files to absolute GitHub URLs for proper display on PyPI.

Used in PyOctoMap setup.py to ensure images and links work correctly
when the package is uploaded to PyPI.
"""

__version__ = "1.0.0"
__author__ = "Spinkoo"
__project__ = "pyoctomap"

from .replace_url import replace_url

__all__ = ["replace_url"]
