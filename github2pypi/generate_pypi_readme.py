#!/usr/bin/env python3
"""
Script to generate PyPI README from main README.md using github2pypi.
This ensures that all relative URLs are converted to absolute GitHub URLs.
"""

import os
import sys
from pathlib import Path

# Add the github2pypi module to the path
sys.path.insert(0, str(Path(__file__).parent))

from replace_url import replace_url

def main():
    """Generate PyPI README from main README.md"""
    project_root = Path(__file__).parent.parent
    readme_path = project_root / "README.md"
    pypi_readme_path = project_root / "README_pypi_preview.md"
    
    if not readme_path.exists():
        print(f"❌ README.md not found at {readme_path}")
        sys.exit(1)
    
    print("🔄 Reading main README.md...")
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔄 Converting URLs for PyPI...")
    # Convert relative URLs to absolute GitHub URLs
    converted_content = replace_url(
        slug="Spinkoo/pyoctomap",
        content=content,
        branch="main"
    )
    
    # Make some PyPI-specific adjustments
    pypi_content = converted_content.replace(
        "On PyPI (Linux):\n```bash\npip install pyoctomap\n```",
        "**PyPI Installation (Recommended):**\n```bash\npip install pyoctomap\n```"
    )
    
    # Update installation section to prioritize PyPI
    pypi_content = pypi_content.replace(
        "**Linux / WSL (Windows Subsystem for Linux):**",
        "**From Source (Linux / WSL):**"
    )
    
    print(f"💾 Writing PyPI README to {pypi_readme_path}...")
    with open(pypi_readme_path, 'w', encoding='utf-8') as f:
        f.write(pypi_content)
    
    print("✅ PyPI README generated successfully!")
    print(f"📁 File: {pypi_readme_path}")
    print("\n🔗 All images and links are now PyPI-compatible with absolute GitHub URLs")
    
    # Verify the file was created
    if pypi_readme_path.exists():
        file_size = pypi_readme_path.stat().st_size
        print(f"📊 File size: {file_size:,} bytes")
    else:
        print("❌ Error: PyPI README file was not created")
        sys.exit(1)

if __name__ == "__main__":
    main()
