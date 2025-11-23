# Contributing to PyOctoMap

Thank you for your interest in contributing to PyOctoMap! This guide will help you get started with the development environment.

## Development Setup

1.  **Clone the repository**

    ```bash
    git clone https://github.com/Spinkoo/pyoctomap.git
    cd pyoctomap
    ```

2.  **Install dependencies**

    You will need Python 3.9+, CMake, and a C++ compiler.

    ```bash
    pip install -r requirements-dev.txt
    # Or install manually
    pip install numpy cython pytest ruff build
    ```

3.  **Compile C++ Libraries**

    PyOctoMap wraps C++ libraries that must be compiled first. We provide a script for this:

    ```bash
    ./compile_cpp_libs.sh
    ```

    This script will:
    - Create a build directory in `src/octomap/build`
    - Run CMake and Make
    - Install the libraries to `src/octomap/lib`

4.  **Build Python Extensions**

    Once the C++ libraries are ready, you can build the Python extensions:

    ```bash
    pip install -e .
    ```

## Running Tests

We use `pytest` for testing. Ensure you have built the extensions first.

```bash
pytest unit_test/
```

## Code Style

We use `ruff` for linting and formatting.

```bash
# Check for issues
ruff check .

# Format code
ruff format .
```

## Building Wheels

To build a wheel for distribution:

```bash
python -m build
```

The wheels will be in the `dist/` directory.

## CI/CD

We use GitHub Actions for continuous integration. The workflow is defined in `.github/workflows/ci.yml`. It automatically builds and tests the package on Ubuntu.

