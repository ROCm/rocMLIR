// RUN: rocmlir-gen --arch gfx1201 --operation gemm -t f16 -p | not grep '|direct_to_lds_32b'
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t f16 -p | not grep '|direct_to_lds_32b'

// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p | grep '|direct_to_lds_32b' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p | grep '|direct_to_lds_32b' | count 1

// YES: rock.gemm
// YES-SAME: features = {{[^ ]*}}direct_to_lds_32b
// NO: rock.gemm
// NO-NOT: direct_to_lds_32b
