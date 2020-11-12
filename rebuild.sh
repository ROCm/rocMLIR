mkdir build
cd build
cmake -G Ninja ../llvm    -DLLVM_ENABLE_PROJECTS=mlir    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"    -DCMAKE_BUILD_TYPE=Debug    -DLLVM_ENABLE_ASSERTIONS=ON    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
cmake --build . --target check-mlir
