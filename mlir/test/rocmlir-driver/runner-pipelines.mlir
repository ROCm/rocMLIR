// REQUIRES: rocm-runner
// RUN: rocmlir-driver -dump-pipelines -host-pipeline=runner -arch=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s --check-prefix=RUNNER

// RUNNER: Host runner pipeline:
// RUNNER-NEXT: {{^}}builtin.module(func.func(mhal-select-targets{archs=amdgcn-amd-amdhsa:gfx90a target-types=GPU}),
// RUNNER-SAME: func.func(convert-linalg-to-affine-loops,
// RUNNER-SAME: lower-affine,
// RUNNER-SAME: expand-strided-metadata,
// RUNNER-SAME: convert-scf-to-cf),
// RUNNER-SAME: func.func(gpu-async-region),
// RUNNER-SAME: convert-mhal-to-gpu,
// RUNNER-SAME: convert-mhal-to-cpu,
// RUNNER-SAME: async-parallel-for{async-dispatch=true min-task-size=1000 num-workers=8},
// RUNNER-SAME: func.func(arith-expand{include-bf16=false},
// RUNNER-SAME: convert-arith-to-llvm{index-bitwidth=0},
// RUNNER-SAME: convert-math-to-llvm{approximate-log1p=true}),
// RUNNER-SAME: convert-math-to-libm,
// RUNNER-SAME: convert-vector-to-llvm{enable-amx=false enable-arm-neon=false enable-arm-sve=false enable-x86vector=false force-32bit-vector-indices=true reassociate-fp-reductions=false},
// RUNNER-SAME: finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false},
// RUNNER-SAME: async-to-async-runtime,
// RUNNER-SAME: func.func(async-runtime-ref-counting,
// RUNNER-SAME: async-runtime-ref-counting-opt),
// RUNNER-SAME: symbol-dce,
// RUNNER-SAME: convert-async-to-llvm,
// RUNNER-SAME: gpu-to-llvm{use-bare-pointers-for-host=false use-bare-pointers-for-kernels=true},
// RUNNER-SAME: convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},
// RUNNER-SAME: reconcile-unrealized-casts){{$}}
