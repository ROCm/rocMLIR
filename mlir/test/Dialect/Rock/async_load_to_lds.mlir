// Check that we are generating async_load_to_lds

// RUN: rocmlir-gen -g 3 -m 1024 -k 768 -n 1024 --transA=true --transB=true --transC=false -t f16 -perf_config=v3:64,64,8,32,32,8,1,4,2,1,1  --operation gemm --arch gfx1250 | rocmlir-driver -c --mlir-print-ir-after=rock-sugar-to-loops -o /dev/null 2>&1 | FileCheck %s
// CHECK: amdgpu.async_load_to_lds
