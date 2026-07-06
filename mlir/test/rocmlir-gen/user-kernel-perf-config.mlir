// RUN: rocmlir-gen --arch gfx942 --operation gemm -g 1 -m 128 -k 128 -n 128 -t f16 -p \
// RUN:   | rocmlir-gen --arch gfx942 --perf_config="v3:16,32,4,16,16,4,4,1,2,1,1" -ph - \
// RUN:   | FileCheck %s

// COM: Exercises the readTestFile() path in mlir/tools/rocmlir-gen/rocmlir-gen.cpp,
// COM: where rocmlir-gen consumes a pre-generated kernel module (from stdin `-`)
// COM: instead of generating one from flags, and applies --perf_config to the
// COM: contained gemm before wrapping it in a host harness (-ph). The existing
// COM: rocmlir-gen tests always generate the kernel from flags, so this file
// COM: was never covered.

// CHECK: func.func @rock_gemm
// CHECK: rock.gemm{{.*}}perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"
