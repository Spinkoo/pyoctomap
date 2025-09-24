#!/bin/bash

# Build and run tests in Docker container
echo "🐳 Building with Docker..."

# Build the Docker image
docker build -t octomap2python .

# Run the build and tests in the container
docker run --rm -v $(pwd):/home/octomap -w /home/octomap octomap2python bash -c "
    echo '�� Building OctoMap2Python in Docker...'
    python3 setup.py bdist_wheel
    pip3 install dist/*.whl --force-reinstall
    
    echo '�� Running tests...'
    mv octomap _octomap_source
    python3 -m pytest unit_test/ -v
    mv _octomap_source octomap
    
    echo '✅ Build and tests completed!'
"
