// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=migraphx -arch=gfx90a /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=MIGRAPHX --match-full-lines --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=gpu -arch=gfx90a /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=GPU --match-full-lines --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=binary -arch=gfx90a /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=BINARY --match-full-lines --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=binary -arch=gfx940 /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=BINARY_MI300 --match-full-lines --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -host-pipeline=partition -targets=gfx90a /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=PARTITION --match-full-lines --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -host-pipeline=mhal -targets=gfx90a /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=MHAL --match-full-lines --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -host-pipeline=highlevel -arch=gfx90a /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=HIGHLEVEL --match-full-lines --strict-whitespace

// COM: Do not put a leading space between the colon and the pass you're looking for
// MIGRAPHX:Kernel pipeline:
// MIGRAPHX-NEXT:builtin.module(func.func(migraphx-realize-int4,
// MIGRAPHX-NEXT:migraphx-transform,
// MIGRAPHX-NEXT:canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
// MIGRAPHX-NEXT:migraphx-to-tosa))

// GPU:Kernel pipeline:
// GPU-NEXT:builtin.module(func.func(rock-affix-params{fallback=false},
// GPU-NEXT:rock-conv-to-gemm,
// GPU-NEXT:rock-gemm-to-gridwise,
// GPU-NEXT:rock-regularize,
// GPU-NEXT:rock-gridwise-gemm-to-blockwise,
// GPU-NEXT:rock-blockwise-gemm-to-threadwise,
// GPU-NEXT:rock-linalg-align,
// GPU-NEXT:rock-pipeline{rock-pipeline-remove-stages=true},
// GPU-NEXT:canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
// GPU-NEXT:convert-linalg-to-affine-loops,
// GPU-NEXT:rock-vectorize-fusions,
// GPU-NEXT:rock-reuse-lds,
// GPU-NEXT:rock-output-swizzle,
// GPU-NEXT:rock-reuse-lds,
// GPU-NEXT:rock-lower-reduce,
// GPU-NEXT:rock-threadwise-gemm-lowering,
// GPU-NEXT:rock-analyze-memory-use,
// GPU-NEXT:rock-sugar-to-loops,
// GPU-NEXT:rock-clean-math,
// GPU-NEXT:math-extend-to-supported-types{extra-types={f16} target-type=f32},
// GPU-NEXT:rock-buffer-load-merge,
// GPU-NEXT:rock-transform-to-memref,
// GPU-NEXT:rock-emulate-narrow-type,
// GPU-NEXT:rock-loops-to-cf),
// GPU-NEXT:convert-rock-to-gpu)

// BINARY:Kernel pipeline:
// BINARY-NEXT:builtin.module(strip-debuginfo,
// BINARY-NEXT:gpu.module(amdgpu-emulate-atomics{chipset=gfx90a},
// BINARY-NEXT:arith-emulate-unsupported-floats{source-types={f8E4M3FNUZ,
// BINARY-NEXT:f8E5M2FNUZ,
// BINARY-NEXT:f8E4M3FN,
// BINARY-NEXT:f8E5M2} target-type=f32},
// BINARY-NEXT:convert-arith-to-amdgpu{allow-packed-f16-round-to-zero=true chipset=gfx90a saturate-fp8-truncf=true},
// BINARY-NEXT:emulate-fp8-ext-trunc,
// BINARY-NEXT:expand-strided-metadata,
// BINARY-NEXT:lower-affine,
// BINARY-NEXT:convert-gpu-to-rocdl{chipset=gfx90a index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true},
// BINARY-NEXT:llvm.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
// BINARY-NEXT:cse,
// BINARY-NEXT:rock-prepare-llvm)),
// BINARY-NEXT:rocdl-attach-target{O=3 abi=500 chip=gfx90a correct-sqrt=true daz=false fast=false features= finite-only=false  module= triple=amdgcn-amd-amdhsa unsafe-math=false wave64=true},
// BINARY-NEXT:gpu-module-to-binary{format=fatbin  opts= toolkit=},
// BINARY-NEXT:rock-check-residency,
// BINARY-NEXT:emulate-fp8-ext-trunc)

// BINARY_MI300:Kernel pipeline:
// BINARY_MI300-NEXT:builtin.module(strip-debuginfo,
// BINARY_MI300-NEXT:gpu.module(amdgpu-emulate-atomics{chipset=gfx940},
// BINARY_MI300-NEXT:arith-emulate-unsupported-floats{source-types={f8E4M3FNUZ,
// BINARY_MI300-NEXT:f8E5M2FNUZ,
// BINARY_MI300-NEXT:f8E4M3FN,
// BINARY_MI300-NEXT:f8E5M2} target-type=f32},
// BINARY_MI300-NEXT:convert-arith-to-amdgpu{allow-packed-f16-round-to-zero=true chipset=gfx940 saturate-fp8-truncf=true},
// BINARY_MI300-NEXT:expand-strided-metadata,
// BINARY_MI300-NEXT:lower-affine,
// BINARY_MI300-NEXT:convert-gpu-to-rocdl{chipset=gfx940 index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true},
// BINARY_MI300-NEXT:llvm.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
// BINARY_MI300-NEXT:cse,
// BINARY_MI300-NEXT:rock-prepare-llvm)),
// BINARY_MI300-NEXT:rocdl-attach-target{O=3 abi=500 chip=gfx940 correct-sqrt=true daz=false fast=false features= finite-only=false  module= triple=amdgcn-amd-amdhsa unsafe-math=false wave64=true},
// BINARY_MI300-NEXT:gpu-module-to-binary{format=fatbin  opts= toolkit=},
// BINARY_MI300-NEXT:rock-check-residency,
// BINARY_MI300-NEXT:emulate-fp8-ext-trunc)

// PARTITION:Partitioner pipeline:
// PARTITION-NEXT:builtin.module(func.func(tosa-make-broadcastable),
// PARTITION-NEXT:func.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}),
// PARTITION-NEXT:tosa-partition{anchor-ops={tosa.conv2d,
// PARTITION-NEXT:tosa.depthwise_conv2d,
// PARTITION-NEXT:tosa.matmul} partition-tag=kernel  trailing-only=true},
// PARTITION-NEXT:func.func(mhal-annotate-access-kinds),
// PARTITION-NEXT:duplicate-function-elimination,
// PARTITION-NEXT:func.func(mhal-infer-graph),
// PARTITION-NEXT:mhal-target-kernels{targets={amdgcn-amd-amdhsa:gfx90a}})

// MHAL:MHAL package pipeline:
// MHAL-NEXT:any(mhal-package-targets)

// HIGHLEVEL:Kernel pipeline:
// HIGHLEVEL-NEXT:builtin.module(func.func(tosa-to-tensor,
// HIGHLEVEL-NEXT:tosa-to-rock,
// HIGHLEVEL-NEXT:rock-view-to-transform,
// HIGHLEVEL-NEXT:rocmlir-custom-tosa-to-linalg),
// HIGHLEVEL-NEXT:func.func(tosa-optional-decompositions),
// HIGHLEVEL-NEXT:func.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}),
// HIGHLEVEL-NEXT:func.func(tosa-infer-shapes),
// HIGHLEVEL-NEXT:func.func(tosa-make-broadcastable),
// HIGHLEVEL-NEXT:func.func(tosa-to-linalg-named{prefer-conv2d-kernel-layout-hwcf=false}),
// HIGHLEVEL-NEXT:func.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}),
// HIGHLEVEL-NEXT:func.func(tosa-layerwise-constant-fold{aggressive-reduce-constant=false}),
// HIGHLEVEL-NEXT:func.func(tosa-make-broadcastable),
// HIGHLEVEL-NEXT:tosa-validate{level=none profile=undefined strict-op-spec-alignment=false},
// HIGHLEVEL-NEXT:func.func(tosa-to-linalg{aggressive-reduce-constant=false disable-tosa-decompositions=false}),
// HIGHLEVEL-NEXT:func.func(tosa-to-tensor,
// HIGHLEVEL-NEXT:tosa-to-scf,
// HIGHLEVEL-NEXT:tosa-to-arith{include-apply-rescale=false use-32-bit=false},
// HIGHLEVEL-NEXT:linalg-fuse-elementwise-ops,
// HIGHLEVEL-NEXT:linalg-fold-unit-extent-dims{use-rank-reducing-slices=false},
// HIGHLEVEL-NEXT:rock-view-to-transform,
// HIGHLEVEL-NEXT:rock-fold-broadcast,
// HIGHLEVEL-NEXT:canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
// HIGHLEVEL-NEXT:empty-tensor-to-alloc-tensor,
// HIGHLEVEL-NEXT:cse),
// HIGHLEVEL-NEXT:convert-tensor-to-linalg,
// HIGHLEVEL-NEXT:func.func(empty-tensor-to-alloc-tensor,
// HIGHLEVEL-NEXT:linalg-fold-unit-extent-dims{use-rank-reducing-slices=false}),
// HIGHLEVEL-NEXT:one-shot-bufferize{allow-return-allocs-from-loops=false allow-unknown-ops=false analysis-fuzzer-seed=0 analysis-heuristic=bottom-up bufferize-function-boundaries=false check-parallel-regions=true copy-before-write=false  dump-alias-sets=false function-boundary-type-conversion=infer-layout-map must-infer-memory-space=false  print-conflicts=false test-analysis-only=false unknown-type-conversion=fully-dynamic-layout-map},
// HIGHLEVEL-NEXT:buffer-results-to-out-params{add-result-attr=false hoist-static-allocs=false})
