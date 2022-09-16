# MLIR-based convolution and GEMM kernel generator for ROCm

This is the repository for a MLIR-based convolution and GEMM kernel generator
targetting AMD hardware. This generator is mainly used from
[Rock](https://github.com/ROCmSoftwarePlatform/Rock/)
and [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX),
but it can be used on a standalone basis. (The ability to use this code via
`torch-mlir` is being investigated as well.)

## Building (and testing)
To build the system

```sh
mkdir build
cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja check-mlir-miopen
```

If you will be targetting a MI-100, MI-200, or other system that supports
mfma instructions, add the flag `-DMLIR_ROCK_DRIVER_XDLOPS_TEST_ENABLED=1 `
to the `cmake` invocation above.

To not actually run the tests, use `check-mlir-miopen-build-only`.

To build the static library that is used by Rock
```sh
mkdir build
cd build
cmake -G Ninja .. -DBUILD_FAT_LIBROCKCOMPILER -DCMAKE_BUILD_TYPE=Release
ninja librockCompiler
```


and to install it so Rock can find it
```
cmake --install . --component librockCompiler --prefix [your Rock deps]
```

## Standalone usage

For usage examples, see `mlir/test/mlir-miopen-driver`, especiallly the files
`sanity.mlir` and the contents of the `e2e_for_pr` directory.

This project also includes cod that translates from TOSA to kernels, see
`mlir/test/fusion` for examples of how to invoke it.

In general (with all invocations given from the build directory)
- `./bin/miopen-gen` generates high-level convolution operations and
  host code. Many of the options control data layout, size, etc, but some other
  useful flags are:
    - `-x2` (which enables mfma usage)
    - `-ph` (which causes host code to be generated)
    - `-pv` (which makes the host code validtae the results against a reference)
    - `-pv_with_gpu` (which uses a GPU validator instead)
    - `-pr` (which prints kkrnel results)
- `./bin/mlir-miopen-driver` is a wrapper around the kernel generation pipeline.
  Use `-c` (or `--kernel-pipeline=gpu`) to run the default pipeline
- `./bin/mlir-rocm-runner` runs kernels by compiling the GPU code and
  JIT-compiling (as in `mlir-cpu-runner`) the host code.

`mlir-rocm-runner` needs to be told to link the generated kernels against utility
libraries that map from MLIR operations to the HIP runtime.
The required command-line arguments (if running from `build/`) are

```sh
./bin/mlir-rocm-runner --shared-libs=./external/llvm-project/llvm/lib/libmlir_rocm_runtime.so,./lib/libconv-validation-wrappers.so,./external/llvm-project/llvm/lib/libmlir_runner_utils.so --entry-point-result=void
```

Adding `--debug-only=serialize-to-blob` will cause the GCN assembly code for the
kernels being executed to be dumped to standard error.
