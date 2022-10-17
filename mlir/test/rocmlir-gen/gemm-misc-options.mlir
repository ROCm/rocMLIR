// Arch is fixed here because not all architectures have atomic_add
// RUN: rocmlir-gen --arch gfx908 --operation gemm -p --store-method atomic_add | FileCheck %s --check-prefix=ATOMIC_ADD
// ATOMIC_ADD: rock.gemm
// ATOMIC_ADD-SAME: storeMethod = atomic_add
