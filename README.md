[![License](https://img.shields.io/badge/License-Apache_2.0_with_LLVM_Exceptions-blue.svg)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/ROCm/rocMLIR.svg?style=flat)](https://github.com/ROCm/rocMLIR/graphs/contributors)
[![Build Status](https://github.com/ROCm/rocMLIR/actions/workflows/ci.yml/badge.svg)](https://github.com/ROCm/rocMLIR/actions/workflows/ci.yml)
<!-- Uncomment when an OpenSSF Best Practices project ID has been registered: -->
<!-- [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/YOUR-ID/badge)](https://www.bestpractices.dev/projects/YOUR-ID) -->

# rocMLIR

> MLIR-based GEMM, convolution, attention, GEMM+GEMM, and CONV+GEMM kernel generator for AMD GPUs.

rocMLIR is an MLIR-based GPU kernel generator targeting AMD GPUs. The high-level lowering is `migraphx` -> `tosa` / `linalg` -> `rock`, which then continues through MLIR's `amdgpu` and `rocdl` dialects to HSACO via the LLVM AMDGPU backend (vendored under `external/llvm-project/`).

It targets AMD CDNA and RDNA GPUs (gfx9xx / gfx10xx / gfx11xx / gfx12xx), and is primarily consumed as the static `librockCompiler` library by [MIGraphX](https://github.com/ROCm/AMDMIGraphX), though it can also be driven standalone for kernel generation, validation, and performance tuning.

## Prerequisites

- An AMD GPU and a working [ROCm](https://rocm.docs.amd.com/) installation (with `rocminfo` on `PATH`).
- A reasonably recent `clang` / `clang++` (the ROCm-shipped compiler at `/opt/rocm/llvm/bin/clang++` is the standard development toolchain).
- `lld`, `ninja`, and CMake >= 3.20.
- Python 3 (>= 3.8 if you build with `LLVM_INCLUDE_TESTS=ON`, the default; >= 3.0 otherwise). Required at configure time for the vendored LLVM build, plus in-tree development scripts and the LIT test runner. Not needed by downstream consumers (e.g. MIGraphX) that only link against the prebuilt `librockCompiler`.
- Git.

## Installation

```sh
git clone https://github.com/ROCm/rocMLIR.git
cd rocMLIR
mkdir -p build && cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja check-rocmlir
```

To not actually run the tests, use `check-rocmlir-build-only`.

To build the static `librockCompiler` library used by MIGraphX:

```sh
mkdir -p build && cd build
cmake -G Ninja .. -DBUILD_FAT_LIBROCKCOMPILER=On -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja
```

To install `librockCompiler` so MIGraphX can find it:

```sh
cmake --install . --prefix /path/to/MIGraphX/deps
```

Additional developer documentation lives under [`mlir/docs/`](mlir/docs/).

## Usage

A typical standalone pipeline generates a kernel with `rocmlir-gen`, lowers it with `rocmlir-driver -c`, and runs it via `rocm-run` -- a wrapper around `mlir-runner` that auto-locates the rocMLIR build and the LLVM build directory under `external/llvm-project/`, and links the right runtime libraries (`libmlir_rocm_runtime`, `libconv-validation-wrappers`, runner utils, etc.):

```sh
# Run from the repo root, with `build/` containing the build above.
ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]*' | head -1)

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f32 \
    --arch "$ARCH" -g 1 -m 64 -k 256 -n 128 \
  | build/bin/rocmlir-driver -c \
  | mlir/utils/widgets/rocm-run
```

Useful `rocmlir-gen` flags:

- `--arch` -- target AMDGPU architecture (e.g. `gfx942`, `gfx950`, `gfx1100`); MFMA/WMMA support is inferred from the chosen architecture.
- `-t` / `--dtype` -- data type selector (e.g. `f16`, `f32`, `bf16`, `i8`, `fp8_fp8`).
- `-out_datatype` / `--out_dtype` / `-tc` -- override the output data type independently of `-t` (e.g. f16 input with f32 output).
- `--perf_config` -- supply a serialized tuning configuration.
- `-ph` -- emit host code alongside the kernel.
- `-pv` -- validate kernel results against a CPU reference.
- `-pv_with_gpu` -- validate against a GPU reference instead.
- `-pr` -- print kernel results.
- `-mfma=on|off` -- explicitly enable/disable MFMA (or `-wmma=on|off` on WMMA targets).

Run `build/bin/rocmlir-gen --help` for the full, current option list.

`rocmlir-driver` is a wrapper around the kernel generation pipeline. Use `-c` (or `--kernel-pipeline=full --host-pipeline=runner`) to run the default pipeline. Adding `--debug-only=serialize-to-isa` will dump the GCN assembly for the executed kernels to standard error.

More examples live under `mlir/test/rocmlir-driver/` (notably `sanity.mlir`), with end-to-end PR tests under `mlir/test/fusion/pr-e2e/` (including the MIGraphX-dialect `mixr-*` tests) and `mlir/test/fusion/e2e/`. To build and run the full in-tree test suite (from the build directory):

```sh
cd build && ninja check-rocmlir
```

### Disabling MFMA/WMMA in tests

By default, we infer the use of GPU-specific acceleration instructions (MFMA or WMMA) based on the features of the currently available GPU. To disable this, add `-DROCMLIR_GEN_FLAGS="-mfma=off -wmma=off"` to the `cmake` invocation above. Note that this will **not** affect behavior in production/static library builds, which do not use `rocmlir-gen`.

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the issue-reporting and pull-request workflow.

For bugs and feature requests, open a [GitHub Issue](../../issues).

---

## Security

To report a security vulnerability, **do not open a public GitHub issue**.
See [SECURITY.md](SECURITY.md) for our responsible disclosure policy.

---

## Contact

For questions, [open a GitHub issue](../../issues/new).

See [CODEOWNERS](.github/CODEOWNERS) for the full ownership list.

---

## License

This project is licensed under the [Apache License 2.0 with LLVM Exceptions](LICENSE).
