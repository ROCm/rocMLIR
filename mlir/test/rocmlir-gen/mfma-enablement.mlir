// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t fp8 -p | grep '|mfma' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t bf8 -p | grep '|mfma' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t bf8_fp8 -p | grep '|mfma' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t fp8_bf8 -p | grep '|mfma' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t bf8_i8 -p | not grep '|mfma'
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t i8_bf8 -p | not grep '|mfma'
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t fp8_i8 -p | not grep '|mfma'
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t i8_fp8 -p | not grep '|mfma'

// YES: rock.gemm
// YES-SAME: features = {{[^ ]*}}mfma
// NO: rock.gemm
// NO-NOT: mfma
