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
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    return result

def main():
    """Main build and publish workflow."""
    print("🚀 Building pyoctomap for PyPI...")
    
    # Ensure we're in the project root
    if not Path("setup.py").exists():
        print("❌ setup.py not found. Please run from project root.")
        sys.exit(1)
    
    # Clean previous builds
    print("🧹 Cleaning previous builds...")
    for path in ["build", "dist", "*.egg-info"]:
        if Path(path).exists():
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    # Test github2pypi conversion
    print("🔄 Testing README conversion...")
    try:
        from github2pypi import replace_url
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
        converted = replace_url("Spinkoo/pyoctomap", content)
        print("✅ README conversion successful")
    except Exception as e:
        print(f"❌ README conversion failed: {e}")
        sys.exit(1)
    
    # Build wheel
    print("🔨 Building wheel...")
    run_command("python setup.py bdist_wheel")
    
    # Build source distribution
    print("📦 Building source distribution...")
    run_command("python setup.py sdist")
    
    # List built files
    print("📁 Built files:")
    dist_files = list(Path("dist").glob("*"))
    for file in dist_files:
        print(f"  - {file}")
    
    # Check with twine
    print("🔍 Checking with twine...")
    run_command("twine check dist/*")
    
    print("\n✅ Build completed successfully!")
    print("\n📋 Next steps:")
    print("  1. Test install: pip install dist/*.whl")
    print("  2. Upload to test PyPI: twine upload --repository-url https://test.pypi.org/legacy/ dist/*")
    print("  3. Upload to PyPI: twine upload dist/*")
    print("\n🔗 Images and links will work correctly on PyPI thanks to github2pypi!")

if __name__ == "__main__":
    main()
