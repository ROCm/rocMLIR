// RUN: rocmlir-gen --arch gfx90a --operation gemm -t f32  -g 1 -m 8192 -k 512  -n 8192 --emit-split-k-selection-likelihood | FileCheck %s --check-prefixes=TEST_GFX90A_1
// RUN: rocmlir-gen --arch gfx90a --operation gemm -t f32  -g 1 -m 1024 -k 256  -n 1024 --emit-split-k-selection-likelihood | FileCheck %s --check-prefixes=TEST_GFX90A_2
// RUN: rocmlir-gen --arch gfx90a --operation gemm -t f32  -g 1 -m 32   -k 1024 -n 64   --emit-split-k-selection-likelihood | FileCheck %s --check-prefixes=TEST_GFX90A_3

// RUN: rocmlir-gen --arch gfx1032 --operation gemm -t f32  -g 1 -m 8192 -k 512  -n 8192 --emit-split-k-selection-likelihood | FileCheck %s --check-prefixes=TEST_GFX1032_1
// RUN: rocmlir-gen --arch gfx1032 --operation gemm -t f32  -g 1 -m 1024 -k 256  -n 1024 --emit-split-k-selection-likelihood | FileCheck %s --check-prefixes=TEST_GFX1032_2
// RUN: rocmlir-gen --arch gfx1032 --operation gemm -t f32  -g 1 -m 32   -k 1024 -n 64   --emit-split-k-selection-likelihood | FileCheck %s --check-prefixes=TEST_GFX1032_3

// TEST_GFX90A_1: never
// TEST_GFX90A_2: maybe
// TEST_GFX90A_3: always

// TEST_GFX1032_1: never
// TEST_GFX1032_2: maybe
// TEST_GFX1032_3: maybe
