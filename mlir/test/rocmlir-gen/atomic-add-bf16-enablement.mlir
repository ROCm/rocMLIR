// RUN: rocmlir-gen --arch gfx1201 --operation gemm --store-method=atomic_add -t f16 -p | grep '|atomic_add_bf16' | count 1
// RUN: rocmlir-gen --arch gfx1100 --operation gemm --store-method=atomic_add -t f16 -p | not grep '|atomic_add_bf16'

// RUN: rocmlir-gen --arch gfx942 --operation gemm --store-method=atomic_add -t bf16 -p | not grep '|atomic_add_bf16'

// YES: rock.gemm
// YES-SAME: features = {{[^ ]*}}atomic_add_bf16
// NO: rock.gemm
// NO-NOT: atomic_add_bf16
