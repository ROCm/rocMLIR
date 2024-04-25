# Installing Composable Kernel library

Perform the following steps outside `rocMLIR` source tree. You can compile the CK
library for multiple target devices. However, we recommend to compile the library
only for the target which you're going to use to run the benchmarks. This will
help you to save compilation-time.

```bash
CK_INSTALL_DIR=$(realpath .)/usr/ck
mkdir -p ${CK_INSTALL_DIR}

git clone https://github.com/ROCm/composable_kernel.git
cd composable_kernel
mkdir -p build && cd build

# specify your target here
TARGET="gfx908" 

CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ cmake .. \
-DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
-DCMAKE_C_COMPILER_LAUNCHER=ccache \
-DCMAKE_LINKER_TYPE=LLD \
-DCMAKE_PREFIX_PATH=/opt/rocm \
-DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
-DCMAKE_BUILD_TYPE=Release \
-DGPU_TARGETS="${TARGET}" \
-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
-DCMAKE_INSTALL_PREFIX="${CK_INSTALL_DIR}"

make -j<num_proc>
make install
```

Be patient, the CK library takes long time to compile. Try to use as many cores
as possible. Don't try to compile the CK library with the CMake Build Type other
than `Release` because, in this case, you will end up with lot's linking errors
related to symbols relocation.

Come back to the `rocMLIR` source directory and configure the project as follows:

```bash
CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ cmake .. \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DLLVM_CCACHE_BUILD=ON \
-DLLD_BUILD_TOOLS=ON \
-DROCMLIR_ENABLE_BENCHMARKS="ck" \
-DCMAKE_PREFIX_PATH="${CK_INSTALL_DIR}"
```

Install the CK benchmarks as follows

```bash
cmake --build . --target ck-benchmark-driver -j10
```
