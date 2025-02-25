// RUN: rocmlir-gen --arch gfx942 --operation gemm --operation gemm -mfma infer -t f32 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm --operation gemm -mfma infer -t f32 -p | grep 'mfma|' | count 1

// RUN: rocmlir-gen --arch gfx942 --operation gemm --operation gemm -mfma infer -t f16 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm --operation gemm -mfma infer -t f16 -p | grep 'mfma|' | count 1

// RUN: rocmlir-gen --arch gfx942 --operation gemm --operation gemm -mfma infer -t bf16 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm --operation gemm -mfma infer -t bf16 -p | grep 'mfma|' | count 1

// RUN: rocmlir-gen --arch gfx942 --operation gemm --operation gemm -mfma infer -t i8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm --operation gemm -mfma infer -t i8 -p | grep 'mfma|' | count 1

// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t fp8_fp8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t bf8_bf8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t fp8_fp8 -force-f8-types=fnuz -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t fp8_fp8 -force-f8-types=ocp -p | not grep 'mfma|'
// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t fp8_fp8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t bf8_bf8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t fp8_fp8 -force-f8-types=ocp -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t fp8_fp8 -force-f8-types=fnuz -p | not grep 'mfma|'

// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t bf8_fp8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t bf8_fp8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t fp8_bf8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t fp8_bf8 -p | grep 'mfma|' | count 1

// YES: rock.gemm
// YES-SAME: features = {{[^ ]*}}mfma
// NO: rock.gemm
// NO-NOT: mfma
