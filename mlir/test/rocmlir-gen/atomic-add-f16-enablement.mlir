// RUN: rocmlir-gen --arch gfx1201 --operation gemm --store-method=atomic_add -t f16 -p | grep '|atomic_add_f16' | count 1
// RUN: rocmlir-gen --arch gfx1100 --operation gemm --store-method=atomic_add -t f16 -p | not grep '|atomic_add_f16'

// RUN: rocmlir-gen --arch gfx950 --operation gemm --store-method=atomic_add -t f16 -p | grep '|atomic_add_f16' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm --store-method=atomic_add -t f16 -p | grep '|atomic_add_f16' | count 1

// YES: rock.gemm
// YES-SAME: features = {{[^ ]*}}atomic_add_f16
// NO: rock.gemm
// NO-NOT: atomic_add_f16
