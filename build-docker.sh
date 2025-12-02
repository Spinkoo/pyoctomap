#!/bin/bash

# Build and run tests in Docker container
echo "🐳 Building with Docker..."

# Build the Docker image
docker build -t pyoctomap .

# Run the build and tests in the container
docker run --rm -v $(pwd):/home/octomap -w /home/octomap pyoctomap bash -c "
    echo '�� Building PyOctoMap in Docker...'
    python3 setup.py bdist_wheel
    pip3 install dist/*.whl --force-reinstall
    
    echo '�� Running tests...'
    mv octomap _octomap_source
    python3 -m pytest tests/ -v
    mv _octomap_source octomap
    
    echo '✅ Build and tests completed!'
"
