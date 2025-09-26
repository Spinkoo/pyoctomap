#!/bin/bash
# Local script to build wheels using Docker

set -e

echo "🐳 Building pyoctomap wheel using Docker..."

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.wheel -t pyoctomap-wheel .

# Extract the wheel from the container
echo "Extracting wheel from container..."
docker run --name temp-container pyoctomap-wheel
docker cp temp-container:/wheels/ ./dist/
docker rm temp-container

echo "✅ Built wheels:"
ls -la dist/

echo "🎉 Wheel building complete!"
echo "You can now test the wheel with: pip install dist/*.whl"
