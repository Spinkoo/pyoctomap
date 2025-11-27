#!/usr/bin/env python3
"""
Script to build and publish pyoctomap to PyPI.
Uses github2pypi to ensure images work correctly on PyPI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and print output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    return result

def main():
    """Main build and publish workflow."""
    # Ensure we're in the project root
    if not Path("setup.py").exists():
        print("setup.py not found. Please run from project root.")
        sys.exit(1)
    
    # Clean previous builds
    for path in ["build", "dist", "*.egg-info"]:
        if Path(path).exists():
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    # Test github2pypi conversion
    try:
        from github2pypi import replace_url
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
        replace_url("Spinkoo/pyoctomap", content)
    except Exception as e:
        print(f"README conversion failed: {e}")
        sys.exit(1)
    
    # Build wheel
    run_command("python setup.py bdist_wheel")
    
    # Build source distribution
    run_command("python setup.py sdist")
    
    # List built files
    dist_files = list(Path("dist").glob("*"))
    for file in dist_files:
        print(f"Built: {file}")
    
    # Check with twine
    run_command("twine check dist/*")
    
    print("Build completed successfully!")

if __name__ == "__main__":
    main()
