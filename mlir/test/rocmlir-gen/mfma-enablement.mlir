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

// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t fp8_fp8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t bf8_bf8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t fp8_fp8 -force-f8-types=ocp -p | grep 'mfma|' | count 1

// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t bf8_fp8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t bf8_fp8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -mfma infer -t fp8_bf8 -p | grep 'mfma|' | count 1
// RUN: rocmlir-gen --arch gfx950 --operation gemm -mfma infer -t fp8_bf8 -p | grep 'mfma|' | count 1

// Test type remapping for conv_bwd_data: inputElemType comes from outputDataType 
// When filter type (f32) has different bit width than remapped input type (f16 from output), MFMA should be disabled
// RUN: rocmlir-gen --arch gfx942 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --operation=conv_bwd_data -fil_dtype f32 -in_dtype f32 -out_dtype f16 -mfma infer -p | not grep 'mfma|'

// When filter type (f16) matches remapped input type (f16 from output), MFMA should be enabled
// RUN: rocmlir-gen --arch gfx942 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --operation=conv_bwd_data -fil_dtype f16 -in_dtype f32 -out_dtype f16 -mfma infer -p | grep 'mfma|' | count 1

// Test type remapping for conv_bwd_weight: filterElemType comes from outputDataType 
// When remapped filter type (f16 from output) matches input type (f16), MFMA should be enabled
// RUN: rocmlir-gen --arch gfx942 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --operation=conv_bwd_weight -fil_dtype f32 -in_dtype f16 -out_dtype f16 -mfma infer -p | grep 'mfma|' | count 1

// When remapped filter type (f16 from output) has different bit width than input type (f32), MFMA should be disabled
// RUN: rocmlir-gen --arch gfx942 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --operation=conv_bwd_weight -fil_dtype f32 -in_dtype f32 -out_dtype f16 -mfma infer -p | not grep 'mfma|'
