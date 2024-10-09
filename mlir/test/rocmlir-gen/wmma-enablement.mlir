// RUN: rocmlir-gen --arch gfx1201 --operation gemm --operation gemm -wmma infer -t f16 -p | grep '|wmma' | count 1
// RUN: rocmlir-gen --arch gfx1201 --operation gemm -wmma infer -t fp8_fp8 -p | grep '|wmma' | count 1
// RUN: rocmlir-gen --arch gfx1201 --operation gemm -wmma infer -t bf8_bf8 -p | grep '|wmma' | count 1
// RUN: rocmlir-gen --arch gfx1201 --operation gemm -wmma infer -t fp8_fp8 -force-f8-types=fnuz -p | not grep '|wmma'

// RUN: rocmlir-gen --arch gfx1100 --operation gemm -wmma infer -t f16 -p | grep '|wmma' | count 1
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -wmma infer -t fp8_fp8 -p | not grep '|wmma'

// YES: rock.gemm
// YES-SAME: features = {{[^ ]*}}wmma
// NO: rock.gemm
// NO-NOT: wmma
