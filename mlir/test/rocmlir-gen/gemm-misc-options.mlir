// Arch is fixed here because not all architectures have atomic_add
// RUN: rocmlir-gen --arch gfx908 --operation gemm -p --store-method atomic_add | FileCheck %s --check-prefix=ATOMIC_ADD
// ATOMIC_ADD: rock.gemm
// ATOMIC_ADD-SAME: storeMethod = atomic_add
// RUN: ./bin/rocmlir-gen --emit-tuning-key -p --arch gfx900
// CHECK: rock.conv2d#memref<1x128x8x3x3xf32>#memref<128x1x8x32x32xf32>#[0 : i32, 0 : i32, 0 : i32, 0 : i32]#[1 : i32, 1 : i32]#[1 : i32, 1 : i32]#gkcyxnigicihiwinogokohowo#amdgcn-amd-amdhsa:gfx900#
