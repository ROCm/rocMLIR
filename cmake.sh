#!/bin/bash

# The first time you set up the project, make sure you:
# 1. Download triton dependency
# $ git submodule update --init --recursive
# 2. Build triton's LLVM:
# $ cd external/triton/scripts/
# $ bash build-llvm-project.sh

rm -rf build
mkdir build
cd build

cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_FAT_LIBROCKCOMPILER=ON \
  -DLLD_BUILD_TOOLS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld"

ninja libconv-validation-wrappers.so; ninja check-rocmlir-build-only ci-performance-scripts
