# MLIR-based convolution and GEMM kernel generator for ROCm

This is the repository for a MLIR-based convolution and GEMM kernel generator
targetting AMD hardware. This generator is mainly used from
[MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX),
but it can be used on a standalone basis. (The ability to use this code via
`torch-mlir` is being investigated as well.)

## Building (and testing)
To build the system

```sh
mkdir build
cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=/opt/rocm/lllvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja check-rocmlir
```

Note that we require building against a relatively recent clang.
The above commands specify the ROCm clang release in order to match our
standard development practice.

To not actually run the tests, use `check-rocmlir-build-only`.

To build the static library that is used by MIGraphX
```sh
mkdir build
cd build
cmake -G Ninja .. -DBUILD_FAT_LIBROCKCOMPILER=On -DCMAKE_BUILD_TYPE=Release -
ninja
```


and to install it so MIGraphX can find it
```
cmake --install . --prefix [your MIOpen deps]
```

## Standalone usage

For usage examples, see `mlir/test/rocmlir-driver`, especiallly the files
`sanity.mlir` and the contents of the `e2e_for_pr` directory.

This project also includes code that translates from TOSA to kernels, see
`mlir/test/fusion` for examples of how to invoke it.

In general (with all invocations given from the build directory)
- `./bin/rocmlir-gen` generates high-level convolution operations and
  host code. Many of the options control data layout, size, etc, but some other
  useful flags are:
    - `-mfma=on` (which enables mfma usage) (or `-wmma=on` for gfx11 targets)
    - `-mfma=off` (which disables mfma usage) (or `-wmma=off` for gfx11 targets)
    - `-ph` (which causes host code to be generated)
    - `-pv` (which makes the host code validtae the results against a reference)
    - `-pv_with_gpu` (which uses a GPU validator instead)
    - `-pr` (which prints kkrnel results)
- `./bin/rocmlir-driver` is a wrapper around the kernel generation pipeline.
  Use `-c` (or `--kernel-pipeline=full --host-pipeline=runner`) to run the
  default pipeline


The result of this pipeline should, most simply, be passed to the `rocm-run`
script in `mlir/utils/widgets//rocm-run`, which calls `mlir-cpu-runner` with
the appropriate flags and infers the pathnames for libraries correctly.

In more detail, the result of the above pipeline can be passed to
`./external/llvm-project/llvm/bin/mlir-cpu-runner` .

`mlir-cpu-runner` needs to link the generated host code against libraries that
map from MLIR operations to the HIP runtime.
The required command-line arguments (if running from `build/`) are

```sh
./external/llvm-project/llvm/bin/mlir-cpu-runner --shared-libs=./external/llvm-project/llvm/lib/libmlir_rocm_runtime.so,./lib/libconv-validation-wrappers.so,./external/llvm-project/llvm/lib/libmlir_runner_utils.so --entry-point-result=void
```

Adding `--debug-only=serialize-to-blob` to the `rocmlir-driver` invocation
will cause the GCN assembly code for the kernels being executed to be dumped to
standard error.

### Disabling MFMA/WMMA in tests
By default, we infer the use of GPU-specific acceleration instructions,
like MFMA or WMMA, based on the features of the currently available GPU.

To disable this, add `-DROCMLIR_GEN_FLAGS="-mfma=off -wmma=off"` to
the `cmake` invocations given above. Note that this will **not** affect behavior
in production/static library builds, which do not use `rocmlir-gen`.
