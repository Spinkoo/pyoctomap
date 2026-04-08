#!/usr/bin/env bash
# Build OctoMap + dynamicEDT3D from the distribution CMake under src/octomap,
# install to a local prefix, stage shared libs into src/octomap/lib for pyoctomap.
set -euo pipefail

PROJECT_ROOT="${1:?Usage: build_octomap.sh <project_root>}"

OCTO="${PROJECT_ROOT}/src/octomap"
LIB_STAGING="${OCTO}/lib"
INSTALL_PREFIX="${OCTO}/install"

CMAKE_BIN="cmake"
if command -v cmake3 >/dev/null 2>&1; then
  CMAKE_BIN="cmake3"
fi

echo "==> OctoMap CI build (Unix)"
echo "    PROJECT_ROOT=${PROJECT_ROOT}"
echo "    OCTO=${OCTO}"
echo "    cmake=${CMAKE_BIN}"

rm -rf "${OCTO}/build" "${OCTO}/install"
mkdir -p "${LIB_STAGING}" "${OCTO}/build"

(
  cd "${OCTO}/build"
  "${CMAKE_BIN}" .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_CXX_STANDARD=14 \
    -DBUILD_OCTOVIS_SUBPROJECT=OFF \
    -DBUILD_DYNAMICETD3D_SUBPROJECT=ON
  "${CMAKE_BIN}" --build . --parallel "$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  "${CMAKE_BIN}" --install .
)

# Stage for setuptools / cibuildwheel (predictable layout)
mkdir -p "${LIB_STAGING}"
for sub in lib lib64; do
  d="${INSTALL_PREFIX}/${sub}"
  [[ -d "${d}" ]] || continue
  while IFS= read -r -d '' f; do
    cp -av "${f}" "${LIB_STAGING}/"
  done < <(find "${d}" -maxdepth 1 -type f \( \
    -name 'liboctomap*' -o -name 'liboctomath*' -o -name 'libdynamicedt3d*' \) -print0 2>/dev/null)
done

echo "==> Staged under ${LIB_STAGING}:"
ls -la "${LIB_STAGING}"

# Verify we have the three shared libs (.so / .so.* on Linux, .dylib on macOS)
missing=0
for base in liboctomap liboctomath libdynamicedt3d; do
  shopt -s nullglob
  candidates=(
    "${LIB_STAGING}/${base}.so"
    "${LIB_STAGING}/${base}.so."*
    "${LIB_STAGING}/${base}"*.dylib
  )
  shopt -u nullglob
  ok=0
  for f in "${candidates[@]}"; do
    if [[ -f "${f}" ]]; then
      ok=1
      break
    fi
  done
  if [[ "${ok}" -ne 1 ]]; then
    echo "ERROR: missing ${base} (.so / .so.* / .dylib) under ${LIB_STAGING}"
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

echo "==> OctoMap CI build OK"
