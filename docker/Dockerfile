# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    libqt5opengl5-dev \
    libqglviewer-dev-qt5 \
    libeigen3-dev \
    libboost-all-dev \
    liboctomap-dev \
    liboctomap2.4 \
    && rm -rf /var/lib/apt/lists/*

# Install Python build dependencies
RUN pip3 install --upgrade pip setuptools wheel numpy cython pytest

# Create a non-root user
RUN useradd -m -s /bin/bash octomap
USER octomap
WORKDIR /home/octomap

# Set up the environment
ENV PYTHONPATH=/home/octomap:$PYTHONPATH
