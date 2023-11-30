// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=migraphx -arch=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s --check-prefix=MIGRAPHX
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=gpu -arch=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s --check-prefix=GPU
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=binary -arch=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s --check-prefix=BINARY
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=binary -arch=gfx940 /dev/null -o /dev/null 2>&1 | FileCheck %s --check-prefix=BINARY_MI300
// RUN: rocmlir-driver -dump-pipelines -host-pipeline=partition -targets=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s --check-prefix=PARTITION
// RUN: rocmlir-driver -dump-pipelines -host-pipeline=mhal -targets=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s --check-prefix=MHAL
// RUN: rocmlir-driver -dump-pipelines -host-pipeline=highlevel -arch=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s --check-prefix=HIGHLEVEL

// MIGRAPHX: Kernel pipeline
// MIGRAPHX-NEXT: {{^}}builtin.module(func.func(migraphx-transform),
// MIGRAPHX-SAME: func.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}),
// MIGRAPHX-SAME: func.func(migraphx-to-tosa)){{$}}

// GPU: Kernel pipeline:
// GPU-NEXT: {{^}}builtin.module(func.func(
// GPU-SANE: rock-affix-params{block_size=0 fallback=false grid_size=0},
// GPU-SAME: rock-conv-to-gemm,
// GPU-SAME: rock-gemm-to-gridwise,
// GPU-SAME: rock-regularize,
// GPU-SAME: rock-gridwise-gemm-to-blockwise,
// GPU-SAME: rock-blockwise-gemm-to-threadwise,
// GPU-SAME: rock-pipeline{rock-pipeline-remove-stages=true},
// GPU-SAME: canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
// GPU-SAME: rock-linalg-align,
// GPU-SAME: canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
// GPU-SAME: convert-linalg-to-affine-loops,
// GPU-SAME: rock-lower-reduce,
// GPU-SAME: rock-threadwise-gemm-lowering,
// GPU-SAME: rock-analyze-memory-use,
// GPU-SAME: rock-sugar-to-loops,
// GPU-SAME: rock-clean-math,
// GPU-SAME: rock-buffer-load-merge,
// GPU-SAME: rock-transform-to-memref,
// GPU-SAME: rock-loops-to-cf),
// GPU-SAME: convert-rock-to-gpu){{$}}

// BINARY: Kernel pipeline:
// BINARY-NEXT: {{^}}builtin.module(strip-debuginfo,
// BINARY-SAME: gpu.module(amdgpu-emulate-atomics{chipset=gfx90a},
// BINARY-SAME: arith-emulate-unsupported-floats{source-types=bf16,f8E4M3FNUZ,f8E5M2FNUZ target-type=f32},
// BINARY-SAME: emulate-fp8-ext-trunc,
// BINARY-SAME: expand-strided-metadata,
// BINARY-SAME: convert-gpu-to-rocdl{chipset=gfx90a index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true use-opaque-pointers=true},
// BINARY-SAME: llvm.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
// BINARY-SAME: cse,
// BINARY-SAME: rock-prepare-llvm),
// BINARY-SAME: gpu-to-hsaco{chip=gfx90a dump-ptx=false features= gpu-binary-annotation=gpu.binary opt-level=3 rocm-path= triple=amdgcn-amd-amdhsa},
// BINARY-SAME: rock-check-residency),
// BINARY-SAME: emulate-fp8-ext-trunc){{$}}

// BINARY_MI300: Kernel pipeline:
// BINARY_MI300-NEXT: {{^}}builtin.module(strip-debuginfo,
// BINARY_MI300-SAME: gpu.module(amdgpu-emulate-atomics{chipset=gfx940},
// BINARY_MI300-SAME: arith-emulate-unsupported-floats{source-types=bf16,f8E4M3FNUZ,f8E5M2FNUZ target-type=f32},
// BINARY_MI300-SAME: convert-arith-to-amdgpu{saturate-fp8-truncf=true},
// BINARY_MI300-SAME: expand-strided-metadata,
// BINARY_MI300-SAME: convert-gpu-to-rocdl{chipset=gfx940 index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true use-opaque-pointers=true},
// BINARY_MI300-SAME: llvm.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
// BINARY_MI300-SAME: cse,
// BINARY_MI300-SAME: rock-prepare-llvm),
// BINARY_MI300-SAME: gpu-to-hsaco{chip=gfx940 dump-ptx=false features= gpu-binary-annotation=gpu.binary opt-level=3 rocm-path= triple=amdgcn-amd-amdhsa},
// BINARY_MI300-SAME: rock-check-residency),
// BINARY_MI300-SAME: emulate-fp8-ext-trunc){{$}}

// PARTITION: Partitioner pipeline:
// PARTITION-NEXT: {{^}}builtin.module(func.func(tosa-make-broadcastable),
// PARTITION-SAME: func.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}),
// PARTITION-SAME: tosa-partition{anchor-ops=tosa.conv2d,tosa.depthwise_conv2d,tosa.matmul partition-tag=kernel  trailing-only=true},
// PARTITION-SAME: duplicate-function-elimination,
// PARTITION-SAME: func.func(mhal-infer-graph),
// PARTITION-SAME: mhal-target-kernels{targets=amdgcn-amd-amdhsa:gfx90a}){{$}}

// MHAL: MHAL package pipeline:
// MHAL-NEXT: {{^}}any(mhal-package-targets){{$}}

// HIGHLEVEL: Kernel pipeline:
// HIGHLEVEL-NEXT: {{^}}builtin.module(func.func(tosa-to-tensor,
// HIGHLEVEL-SAME: tosa-to-rock,
// HIGHLEVEL-SAME: rock-view-to-transform),
// HIGHLEVEL-SAME: func.func(tosa-optional-decompositions),
// HIGHLEVEL-SAME: func.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}),
// HIGHLEVEL-SAME: func.func(tosa-infer-shapes),
// HIGHLEVEL-SAME: func.func(tosa-make-broadcastable),
// HIGHLEVEL-SAME: func.func(tosa-to-linalg-named),
// HIGHLEVEL-SAME: func.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}),
// HIGHLEVEL-SAME: func.func(tosa-layerwise-constant-fold),
// HIGHLEVEL-SAME: func.func(tosa-make-broadcastable),
// HIGHLEVEL-SAME: func.func(tosa-validate{level=8k profile=undefined strict-op-spec-alignment=false}),
// HIGHLEVEL-SAME: func.func(tosa-to-linalg),
// HIGHLEVEL-SAME: func.func(tosa-to-tensor,
// HIGHLEVEL-SAME: tosa-to-scf,
// HIGHLEVEL-SAME: tosa-to-arith{include-apply-rescale=false use-32-bit=false},
// HIGHLEVEL-SAME: linalg-fuse-elementwise-ops,
// HIGHLEVEL-SAME: linalg-fold-unit-extent-dims{use-rank-reducing-slices=false},
// HIGHLEVEL-SAME: rock-view-to-transform,
// HIGHLEVEL-SAME: canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
// HIGHLEVEL-SAME: cse),
// HIGHLEVEL-SAME: convert-tensor-to-linalg,
// HIGHLEVEL-SAME: func.func(empty-tensor-to-alloc-tensor,
// HIGHLEVEL-SAME: linalg-fold-unit-extent-dims{use-rank-reducing-slices=false}),
// HIGHLEVEL-SAME: one-shot-bufferize{allow-return-allocs=false allow-unknown-ops=false analysis-fuzzer-seed=0 analysis-heuristic=bottom-up bufferize-function-boundaries=false copy-before-write=false create-deallocs=true  dump-alias-sets=false function-boundary-type-conversion=infer-layout-map must-infer-memory-space=false  print-conflicts=false test-analysis-only=false unknown-type-conversion=fully-dynamic-layout-map},
// HIGHLEVEL-SAME: buffer-results-to-out-params){{$}}
