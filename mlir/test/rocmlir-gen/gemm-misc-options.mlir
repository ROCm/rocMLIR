// Arch is fixed here because not all architectures have atomic_add
// RUN: rocmlir-gen --arch gfx908 --operation gemm -p --store-method atomic_add | FileCheck %s --check-prefix=ATOMIC_ADD
// ATOMIC_ADD: rock.gemm
// ATOMIC_ADD-SAME: storeMethod = atomic_add
// RUN: rocmlir-gen --emit-tuning-key -p --arch gfx900 | FileCheck %s --check-prefix=CONV
// CONV: amdgcn-amd-amdhsa:gfx900        conv -F 1 -f GNCHW -I NGCHW -O NGCHW -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1
// RUN: rocmlir-gen --arch gfx908 --operation gemm -p --emit-tuning-key | FileCheck %s --check-prefix=GEMM
// GEMM: amdgcn-amd-amdhsa:gfx908        -t f32 -out_datatype f32 -transA false -transB false -g 1 -m 1024 -n 512 -k 769
