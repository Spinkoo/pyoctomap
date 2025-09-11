# GitHub2PyPI - OctoMap2Python

Utility to convert relative URLs in README.md to absolute GitHub URLs for PyPI distribution.

## Usage

```python
import github2pypi

def get_long_description():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    
    return github2pypi.replace_url(
        slug="Spinkoo/octomap2python", content=content
    )
```

## Example

**Before:**
```markdown
![Example](examples/octree_visualization.png)
```

**After:**
```markdown
![Example](https://github.com/Spinkoo/octomap2python/blob/main/examples/octree_visualization.png?raw=true)
```