// Arch is fixed here because not all architectures have atomic_add
// RUN: rocmlir-gen --arch gfx908 --operation gemm -p --store-method atomic_add | FileCheck %s --check-prefix=ATOMIC_ADD
// ATOMIC_ADD: rock.gemm
// ATOMIC_ADD-SAME: storeMethod = atomic_add
// RUN: rocmlir-gen --emit-tuning-key -p --arch gfx900
// CHECK: amdgcn-amd-amdhsa:gfx900        conv -F 1 -f NCHW -I NCHW -O NCHW -n 128 -c 8 -h 32 -w 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -t f32
// RUN: rocmlir-gen --arch gfx908 --operation gemm -p --emit-tuning-key
// CHECK: amdgcn-amd-amdhsa:gfx908        gemm -transA false -transB false -g 1 -m 1024 -n 512 -k 769 -t f32
