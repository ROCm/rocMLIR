// Winograd convolution E2E verification for all eligible tier1-conv-configs.
// Auto-generated from mlir/utils/performance/configs/tier1-conv-configs.
// Total eligible configs: 447
//
// Each test verifies E2E numerical correctness through the full pipeline.
// The Winograd path is selected automatically by the rock-conv-to-winograd pass.
// All internal computation uses f32 accumulation (even for f16 inputs),
// producing bit-identical results to the CPU reference. Tight thresholds used.

// f32 N=1 G=1 C=64 K=64 H=56 W=56 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T0
// T0: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=14 W=14 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T1
// T1: [1 1 1]

// f32 N=1 G=1 C=512 K=512 H=7 W=7 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T2
// T2: [1 1 1]

// f32 N=1 G=1 C=128 K=128 H=28 W=28 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T3
// T3: [1 1 1]

// f16 N=1 G=1 C=128 K=128 H=28 W=28 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T4
// T4: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=14 W=14 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T5
// T5: [1 1 1]

// f16 N=1 G=1 C=64 K=64 H=56 W=56 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T6
// T6: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=7 W=7 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T7
// T7: [1 1 1]

// f16 N=2 G=1 C=1280 K=1280 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=1280 -out_channels=1280 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T8
// T8: [1 1 1]

// f16 N=2 G=1 C=320 K=320 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=320 -out_channels=320 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T9
// T9: [1 1 1]

// f16 N=2 G=1 C=1280 K=1280 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=1280 -out_channels=1280 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T10
// T10: [1 1 1]

// f16 N=2 G=1 C=640 K=640 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=640 -out_channels=640 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T11
// T11: [1 1 1]

// f16 N=2 G=1 C=320 K=640 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=320 -out_channels=640 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T12
// T12: [1 1 1]

// f16 N=2 G=1 C=640 K=1280 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=640 -out_channels=1280 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T13
// T13: [1 1 1]

// f16 N=2 G=1 C=2560 K=1280 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=2560 -out_channels=1280 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T14
// T14: [1 1 1]

// f16 N=2 G=1 C=1920 K=1280 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=1920 -out_channels=1280 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T15
// T15: [1 1 1]

// f16 N=2 G=1 C=1280 K=1280 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=1280 -out_channels=1280 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T16
// T16: [1 1 1]

// f16 N=2 G=1 C=2560 K=1280 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=2560 -out_channels=1280 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T17
// T17: [1 1 1]

// f16 N=2 G=1 C=1920 K=640 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=1920 -out_channels=640 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T18
// T18: [1 1 1]

// f16 N=2 G=1 C=1280 K=640 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=1280 -out_channels=640 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T19
// T19: [1 1 1]

// f16 N=2 G=1 C=960 K=640 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=960 -out_channels=640 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T20
// T20: [1 1 1]

// f16 N=2 G=1 C=640 K=320 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=640 -out_channels=320 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T21
// T21: [1 1 1]

// f16 N=2 G=1 C=960 K=320 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=960 -out_channels=320 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T22
// T22: [1 1 1]

// f16 N=2 G=1 C=640 K=640 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=2 -in_channels=640 -out_channels=640 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T23
// T23: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T24
// T24: [1 1 1]

// f16 N=1 G=1 C=4 K=512 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=4 -out_channels=512 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T25
// T25: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T26
// T26: [1 1 1]

// f32 N=1 G=1 C=128 K=128 H=100 W=100 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=100 -in_w=100 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T27
// T27: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=200 W=200 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=200 -in_w=200 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T28
// T28: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=25 W=25 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=25 -in_w=25 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T29
// T29: [1 1 1]

// f32 N=1 G=1 C=512 K=512 H=25 W=25 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=25 -in_w=25 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T30
// T30: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=50 W=50 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=50 -in_w=50 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T31
// T31: [1 1 1]

// f32 N=100 G=1 C=256 K=256 H=14 W=14 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=100 -in_channels=256 -out_channels=256 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T32
// T32: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=200 W=200 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=200 -in_w=200 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T33
// T33: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=100 W=100 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=100 -in_w=100 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T34
// T34: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=13 W=13 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=13 -in_w=13 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T35
// T35: [1 1 1]

// f16 N=1 G=1 C=128 K=128 H=100 W=100 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=100 -in_w=100 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T36
// T36: [1 1 1]

// f16 N=1 G=1 C=64 K=64 H=200 W=200 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=200 -in_w=200 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T37
// T37: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=25 W=25 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=25 -in_w=25 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T38
// T38: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=50 W=50 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=50 -in_w=50 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T39
// T39: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=25 W=25 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=25 -in_w=25 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T40
// T40: [1 1 1]

// f16 N=100 G=1 C=256 K=256 H=14 W=14 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=100 -in_channels=256 -out_channels=256 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T41
// T41: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=200 W=200 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=200 -in_w=200 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T42
// T42: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=100 W=100 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=100 -in_w=100 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T43
// T43: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=13 W=13 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=13 -in_w=13 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T44
// T44: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=40 W=40 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=40 -in_w=40 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T45
// T45: [1 1 1]

// f32 N=1 G=1 C=128 K=128 H=20 W=20 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=20 -in_w=20 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T46
// T46: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=80 W=80 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=80 -in_w=80 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T47
// T47: [1 1 1]

// f32 N=1 G=1 C=128 K=64 H=40 W=40 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=64 -in_h=40 -in_w=40 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T48
// T48: [1 1 1]

// f32 N=1 G=1 C=256 K=64 H=20 W=20 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=64 -in_h=20 -in_w=20 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T49
// T49: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=20 W=20 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=20 -in_w=20 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T50
// T50: [1 1 1]

// f16 N=1 G=1 C=64 K=64 H=40 W=40 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=40 -in_w=40 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T51
// T51: [1 1 1]

// f16 N=1 G=1 C=128 K=128 H=20 W=20 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=20 -in_w=20 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T52
// T52: [1 1 1]

// f16 N=1 G=1 C=64 K=64 H=80 W=80 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=80 -in_w=80 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T53
// T53: [1 1 1]

// f16 N=1 G=1 C=128 K=64 H=40 W=40 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=64 -in_h=40 -in_w=40 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T54
// T54: [1 1 1]

// f16 N=1 G=1 C=256 K=64 H=20 W=20 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=64 -in_h=20 -in_w=20 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T55
// T55: [1 1 1]

// f16 N=1 G=1 C=64 K=64 H=20 W=20 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=20 -in_w=20 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T56
// T56: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=26 W=26 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=26 -in_w=26 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T57
// T57: [1 1 1]

// f16 N=1 G=1 C=128 K=128 H=52 W=52 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=52 -in_w=52 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T58
// T58: [1 1 1]

// f16 N=1 G=1 C=32 K=64 H=208 W=208 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=32 -out_channels=64 -in_h=208 -in_w=208 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T59
// T59: [1 1 1]

// f16 N=1 G=1 C=64 K=64 H=104 W=104 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=104 -in_w=104 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T60
// T60: [1 1 1]

// f16 N=1 G=1 C=256 K=512 H=26 W=26 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=512 -in_h=26 -in_w=26 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T61
// T61: [1 1 1]

// f16 N=1 G=1 C=128 K=256 H=52 W=52 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=256 -in_h=52 -in_w=52 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T62
// T62: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=13 W=13 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=13 -in_w=13 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T63
// T63: [1 1 1]

// f16 N=1 G=1 C=512 K=1024 H=13 W=13 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=1024 -in_h=13 -in_w=13 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T64
// T64: [1 1 1]

// f32 N=1 G=1 C=512 K=512 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T65
// T65: [1 1 1]

// f32 N=1 G=1 C=128 K=128 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T66
// T66: [1 1 1]

// f32 N=1 G=1 C=128 K=256 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=256 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T67
// T67: [1 1 1]

// f32 N=1 G=1 C=512 K=512 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T68
// T68: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T69
// T69: [1 1 1]

// f32 N=1 G=1 C=256 K=512 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=512 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T70
// T70: [1 1 1]

// f32 N=1 G=1 C=512 K=32 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=32 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T71
// T71: [1 1 1]

// f32 N=16 G=1 C=64 K=96 H=35 W=35 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=16 -in_channels=64 -out_channels=96 -in_h=35 -in_w=35 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T72
// T72: [1 1 1]

// f32 N=16 G=1 C=96 K=96 H=35 W=35 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=16 -in_channels=96 -out_channels=96 -in_h=35 -in_w=35 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T73
// T73: [1 1 1]

// f32 N=16 G=1 C=32 K=64 H=147 W=147 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=16 -in_channels=32 -out_channels=64 -in_h=147 -in_w=147 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T74
// T74: [1 1 1]

// f32 N=16 G=1 C=448 K=384 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=16 -in_channels=448 -out_channels=384 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T75
// T75: [1 1 1]

// f32 N=16 G=1 C=80 K=192 H=73 W=73 pad=0,0
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=16 -in_channels=80 -out_channels=192 -in_h=73 -in_w=73 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=0 -padding_h_r=0 -padding_w_l=0 -padding_w_r=0 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T76
// T76: [1 1 1]

// f32 N=32 G=1 C=96 K=96 H=35 W=35 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=32 -in_channels=96 -out_channels=96 -in_h=35 -in_w=35 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T77
// T77: [1 1 1]

// f32 N=32 G=1 C=64 K=96 H=35 W=35 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=32 -in_channels=64 -out_channels=96 -in_h=35 -in_w=35 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T78
// T78: [1 1 1]

// f32 N=32 G=1 C=80 K=192 H=73 W=73 pad=0,0
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=32 -in_channels=80 -out_channels=192 -in_h=73 -in_w=73 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=0 -padding_h_r=0 -padding_w_l=0 -padding_w_r=0 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T79
// T79: [1 1 1]

// f32 N=32 G=1 C=32 K=64 H=147 W=147 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=32 -in_channels=32 -out_channels=64 -in_h=147 -in_w=147 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T80
// T80: [1 1 1]

// f32 N=32 G=1 C=448 K=384 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=32 -in_channels=448 -out_channels=384 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T81
// T81: [1 1 1]

// f32 N=64 G=1 C=96 K=96 H=35 W=35 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=64 -in_channels=96 -out_channels=96 -in_h=35 -in_w=35 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T82
// T82: [1 1 1]

// f32 N=64 G=1 C=80 K=192 H=73 W=73 pad=0,0
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=64 -in_channels=80 -out_channels=192 -in_h=73 -in_w=73 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=0 -padding_h_r=0 -padding_w_l=0 -padding_w_r=0 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T83
// T83: [1 1 1]

// f32 N=64 G=1 C=448 K=384 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=64 -in_channels=448 -out_channels=384 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T84
// T84: [1 1 1]

// f32 N=64 G=1 C=64 K=96 H=35 W=35 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=64 -in_channels=64 -out_channels=96 -in_h=35 -in_w=35 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T85
// T85: [1 1 1]

// f32 N=64 G=1 C=32 K=64 H=147 W=147 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=64 -in_channels=32 -out_channels=64 -in_h=147 -in_w=147 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T86
// T86: [1 1 1]

// f32 N=128 G=1 C=96 K=96 H=35 W=35 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=128 -in_channels=96 -out_channels=96 -in_h=35 -in_w=35 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T87
// T87: [1 1 1]

// f32 N=128 G=1 C=80 K=192 H=73 W=73 pad=0,0
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=128 -in_channels=80 -out_channels=192 -in_h=73 -in_w=73 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=0 -padding_h_r=0 -padding_w_l=0 -padding_w_r=0 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T88
// T88: [1 1 1]

// f32 N=128 G=1 C=64 K=96 H=35 W=35 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=128 -in_channels=64 -out_channels=96 -in_h=35 -in_w=35 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T89
// T89: [1 1 1]

// f32 N=128 G=1 C=448 K=384 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=128 -in_channels=448 -out_channels=384 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T90
// T90: [1 1 1]

// f32 N=128 G=1 C=32 K=64 H=147 W=147 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=128 -in_channels=32 -out_channels=64 -in_h=147 -in_w=147 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T91
// T91: [1 1 1]

// f16 N=64 G=1 C=64 K=64 H=56 W=56 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=64 -in_channels=64 -out_channels=64 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T92
// T92: [1 1 1]

// f16 N=64 G=1 C=128 K=128 H=28 W=28 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=64 -in_channels=128 -out_channels=128 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T93
// T93: [1 1 1]

// f16 N=64 G=1 C=256 K=256 H=14 W=14 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=64 -in_channels=256 -out_channels=256 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T94
// T94: [1 1 1]

// f16 N=64 G=1 C=512 K=512 H=7 W=7 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=64 -in_channels=512 -out_channels=512 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T95
// T95: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T96
// T96: [1 1 1]

// f16 N=1 G=1 C=50 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=50 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T97
// T97: [1 1 1]

// f16 N=1 G=1 C=515 K=512 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T98
// T98: [1 1 1]

// f16 N=1 G=1 C=192 K=191 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=191 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T99
// T99: [1 1 1]

// f16 N=1 G=1 C=96 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T100
// T100: [1 1 1]

// f16 N=1 G=1 C=48 K=47 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=47 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T101
// T101: [1 1 1]

// f16 N=1 G=1 C=387 K=384 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=387 -out_channels=384 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T102
// T102: [1 1 1]

// f16 N=1 G=1 C=192 K=192 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=192 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T103
// T103: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T104
// T104: [1 1 1]

// f16 N=1 G=1 C=768 K=383 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=768 -out_channels=383 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T105
// T105: [1 1 1]

// f16 N=1 G=1 C=384 K=384 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=384 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T106
// T106: [1 1 1]

// f16 N=1 G=1 C=384 K=383 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=383 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T107
// T107: [1 1 1]

// f16 N=1 G=1 C=48 K=48 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=48 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T108
// T108: [1 1 1]

// f16 N=1 G=1 C=192 K=95 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=95 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T109
// T109: [1 1 1]

// f16 N=1 G=1 C=195 K=192 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=195 -out_channels=192 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T110
// T110: [1 1 1]

// f16 N=1 G=1 C=1024 K=511 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=511 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T111
// T111: [1 1 1]

// f16 N=1 G=1 C=96 K=95 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=95 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T112
// T112: [1 1 1]

// f16 N=1 G=1 C=515 K=512 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T113
// T113: [1 1 1]

// f16 N=1 G=1 C=512 K=511 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=511 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T114
// T114: [1 1 1]

// f16 N=1 G=1 C=384 K=191 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=191 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T115
// T115: [1 1 1]

// f32 N=1 G=1 C=50 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=50 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T116
// T116: [1 1 1]

// f32 N=1 G=1 C=192 K=192 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=192 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T117
// T117: [1 1 1]

// f32 N=1 G=1 C=96 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T118
// T118: [1 1 1]

// f32 N=1 G=1 C=48 K=47 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=47 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T119
// T119: [1 1 1]

// f32 N=1 G=1 C=768 K=383 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=768 -out_channels=383 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T120
// T120: [1 1 1]

// f32 N=1 G=1 C=387 K=384 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=387 -out_channels=384 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T121
// T121: [1 1 1]

// f32 N=1 G=1 C=384 K=383 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=383 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T122
// T122: [1 1 1]

// f32 N=1 G=1 C=384 K=384 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=384 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T123
// T123: [1 1 1]

// f32 N=1 G=1 C=192 K=95 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=95 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T124
// T124: [1 1 1]

// f32 N=1 G=1 C=195 K=192 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=195 -out_channels=192 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T125
// T125: [1 1 1]

// f32 N=1 G=1 C=48 K=48 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=48 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T126
// T126: [1 1 1]

// f32 N=1 G=1 C=1024 K=511 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=511 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T127
// T127: [1 1 1]

// f32 N=1 G=1 C=515 K=512 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T128
// T128: [1 1 1]

// f32 N=1 G=1 C=96 K=95 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=95 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T129
// T129: [1 1 1]

// f32 N=1 G=1 C=512 K=511 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=511 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T130
// T130: [1 1 1]

// f32 N=1 G=1 C=384 K=191 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=191 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T131
// T131: [1 1 1]

// f32 N=1 G=1 C=515 K=512 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T132
// T132: [1 1 1]

// f32 N=1 G=1 C=192 K=191 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=191 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T133
// T133: [1 1 1]

// f16 N=1 G=1 C=96 K=160 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=160 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T134
// T134: [1 1 1]

// f16 N=1 G=1 C=80 K=32 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=80 -out_channels=32 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T135
// T135: [1 1 1]

// f16 N=1 G=1 C=816 K=304 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=816 -out_channels=304 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T136
// T136: [1 1 1]

// f16 N=1 G=1 C=464 K=160 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=464 -out_channels=160 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T137
// T137: [1 1 1]

// f16 N=1 G=1 C=304 K=512 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=304 -out_channels=512 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T138
// T138: [1 1 1]

// f16 N=1 G=1 C=48 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T139
// T139: [1 1 1]

// f16 N=1 G=1 C=160 K=304 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=160 -out_channels=304 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T140
// T140: [1 1 1]

// f16 N=1 G=1 C=256 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T141
// T141: [1 1 1]

// f16 N=1 G=1 C=160 K=160 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=160 -out_channels=160 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T142
// T142: [1 1 1]

// f16 N=1 G=1 C=304 K=304 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=304 -out_channels=304 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T143
// T143: [1 1 1]

// f16 N=1 G=1 C=144 K=48 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=144 -out_channels=48 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T144
// T144: [1 1 1]

// f32 N=1 G=1 C=48 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T145
// T145: [1 1 1]

// f32 N=1 G=1 C=816 K=304 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=816 -out_channels=304 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T146
// T146: [1 1 1]

// f32 N=1 G=1 C=144 K=48 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=144 -out_channels=48 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T147
// T147: [1 1 1]

// f32 N=1 G=1 C=256 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T148
// T148: [1 1 1]

// f32 N=1 G=1 C=464 K=160 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=464 -out_channels=160 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T149
// T149: [1 1 1]

// f32 N=1 G=1 C=304 K=512 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=304 -out_channels=512 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T150
// T150: [1 1 1]

// f32 N=1 G=1 C=96 K=160 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=160 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T151
// T151: [1 1 1]

// f32 N=1 G=1 C=160 K=304 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=160 -out_channels=304 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T152
// T152: [1 1 1]

// f32 N=1 G=1 C=304 K=304 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=304 -out_channels=304 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T153
// T153: [1 1 1]

// f32 N=1 G=1 C=160 K=160 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=160 -out_channels=160 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T154
// T154: [1 1 1]

// f32 N=1 G=1 C=80 K=32 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=80 -out_channels=32 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T155
// T155: [1 1 1]

// f32 N=1 G=1 C=48 K=47 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=47 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T156
// T156: [1 1 1]

// f32 N=1 G=1 C=1024 K=511 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=511 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T157
// T157: [1 1 1]

// f32 N=1 G=1 C=96 K=96 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=96 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T158
// T158: [1 1 1]

// f32 N=1 G=1 C=192 K=95 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=95 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T159
// T159: [1 1 1]

// f32 N=1 G=1 C=768 K=383 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=768 -out_channels=383 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T160
// T160: [1 1 1]

// f32 N=1 G=1 C=384 K=191 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=191 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T161
// T161: [1 1 1]

// f32 N=1 G=1 C=195 K=192 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=195 -out_channels=192 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T162
// T162: [1 1 1]

// f32 N=1 G=1 C=387 K=384 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=387 -out_channels=384 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T163
// T163: [1 1 1]

// f32 N=1 G=1 C=48 K=48 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=48 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T164
// T164: [1 1 1]

// f32 N=1 G=1 C=515 K=512 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T165
// T165: [1 1 1]

// f32 N=1 G=1 C=512 K=511 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=511 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T166
// T166: [1 1 1]

// f32 N=1 G=1 C=96 K=95 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=95 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T167
// T167: [1 1 1]

// f32 N=1 G=1 C=384 K=384 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=384 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T168
// T168: [1 1 1]

// f32 N=1 G=1 C=192 K=191 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=191 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T169
// T169: [1 1 1]

// f32 N=1 G=1 C=384 K=383 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=383 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T170
// T170: [1 1 1]

// f32 N=1 G=1 C=192 K=192 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=192 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T171
// T171: [1 1 1]

// f32 N=1 G=1 C=512 K=512 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T172
// T172: [1 1 1]

// f32 N=1 G=1 C=128 K=64 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T173
// T173: [1 1 1]

// f32 N=1 G=1 C=32 K=64 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=32 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T174
// T174: [1 1 1]

// f32 N=1 G=1 C=64 K=32 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=32 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T175
// T175: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T176
// T176: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T177
// T177: [1 1 1]

// f32 N=1 G=1 C=256 K=512 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=512 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T178
// T178: [1 1 1]

// f32 N=1 G=1 C=128 K=256 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=256 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T179
// T179: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=4 W=4 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=4 -in_w=4 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T180
// T180: [1 1 1]

// f32 N=1 G=1 C=256 K=512 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=512 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T181
// T181: [1 1 1]

// f32 N=1 G=1 C=128 K=64 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=64 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T182
// T182: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T183
// T183: [1 1 1]

// f32 N=1 G=1 C=512 K=256 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=256 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T184
// T184: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T185
// T185: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T186
// T186: [1 1 1]

// f32 N=1 G=1 C=128 K=128 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T187
// T187: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T188
// T188: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T189
// T189: [1 1 1]

// f32 N=1 G=1 C=128 K=256 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=256 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T190
// T190: [1 1 1]

// f32 N=1 G=1 C=64 K=32 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=32 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T191
// T191: [1 1 1]

// f32 N=1 G=1 C=32 K=64 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=32 -out_channels=64 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T192
// T192: [1 1 1]

// f32 N=1 G=1 C=256 K=128 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=128 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T193
// T193: [1 1 1]

// f32 N=1 G=1 C=256 K=512 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=512 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T194
// T194: [1 1 1]

// f32 N=1 G=1 C=64 K=128 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=128 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T195
// T195: [1 1 1]

// f32 N=1 G=1 C=256 K=256 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T196
// T196: [1 1 1]

// f32 N=1 G=1 C=32 K=64 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=32 -out_channels=64 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T197
// T197: [1 1 1]

// f32 N=1 G=1 C=256 K=128 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=128 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T198
// T198: [1 1 1]

// f32 N=1 G=1 C=512 K=512 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T199
// T199: [1 1 1]

// f32 N=1 G=1 C=64 K=128 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=128 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T200
// T200: [1 1 1]

// f32 N=1 G=1 C=512 K=512 H=4 W=4 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=4 -in_w=4 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T201
// T201: [1 1 1]

// f32 N=1 G=1 C=128 K=128 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T202
// T202: [1 1 1]

// f32 N=1 G=1 C=128 K=64 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=64 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T203
// T203: [1 1 1]

// f16 N=1 G=1 C=704 K=288 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=704 -out_channels=288 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T204
// T204: [1 1 1]

// f16 N=1 G=1 C=400 K=160 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=400 -out_channels=160 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T205
// T205: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=6 W=6 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T206
// T206: [1 1 1]

// f16 N=1 G=1 C=96 K=48 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=48 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T207
// T207: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T208
// T208: [1 1 1]

// f16 N=1 G=1 C=336 K=512 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=336 -out_channels=512 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T209
// T209: [1 1 1]

// f16 N=1 G=1 C=64 K=64 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T210
// T210: [1 1 1]

// f16 N=1 G=1 C=224 K=96 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=224 -out_channels=96 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T211
// T211: [1 1 1]

// f16 N=1 G=1 C=96 K=96 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=96 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T212
// T212: [1 1 1]

// f16 N=1 G=1 C=112 K=192 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=112 -out_channels=192 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T213
// T213: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T214
// T214: [1 1 1]

// f16 N=1 G=1 C=192 K=336 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=336 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T215
// T215: [1 1 1]

// f16 N=1 G=1 C=1024 K=512 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=512 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T216
// T216: [1 1 1]

// f16 N=1 G=1 C=160 K=160 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=160 -out_channels=160 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T217
// T217: [1 1 1]

// f16 N=1 G=1 C=96 K=48 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=48 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T218
// T218: [1 1 1]

// f16 N=1 G=1 C=288 K=288 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=288 -out_channels=288 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T219
// T219: [1 1 1]

// f16 N=1 G=1 C=848 K=512 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=848 -out_channels=512 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T220
// T220: [1 1 1]

// f16 N=1 G=1 C=64 K=112 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=112 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T221
// T221: [1 1 1]

// f16 N=1 G=1 C=192 K=192 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=192 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T222
// T222: [1 1 1]

// f16 N=1 G=1 C=336 K=336 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=336 -out_channels=336 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T223
// T223: [1 1 1]

// f16 N=1 G=1 C=112 K=112 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=112 -out_channels=112 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T224
// T224: [1 1 1]

// f16 N=1 G=1 C=72 K=36 H=768 W=768 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=72 -out_channels=36 -in_h=768 -in_w=768 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T225
// T225: [1 1 1]

// f16 N=1 G=1 C=96 K=72 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=72 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T226
// T226: [1 1 1]

// f16 N=1 G=1 C=72 K=72 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=72 -out_channels=72 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T227
// T227: [1 1 1]

// f32 N=1 G=1 C=96 K=96 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=96 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T228
// T228: [1 1 1]

// f32 N=1 G=1 C=512 K=512 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T229
// T229: [1 1 1]

// f32 N=1 G=1 C=192 K=336 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=336 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T230
// T230: [1 1 1]

// f32 N=1 G=1 C=112 K=192 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=112 -out_channels=192 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T231
// T231: [1 1 1]

// f32 N=1 G=1 C=1024 K=512 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=512 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T232
// T232: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T233
// T233: [1 1 1]

// f32 N=1 G=1 C=224 K=96 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=224 -out_channels=96 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T234
// T234: [1 1 1]

// f32 N=1 G=1 C=336 K=512 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=336 -out_channels=512 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T235
// T235: [1 1 1]

// f32 N=1 G=1 C=96 K=48 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=48 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T236
// T236: [1 1 1]

// f32 N=1 G=1 C=512 K=512 H=6 W=6 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T237
// T237: [1 1 1]

// f32 N=1 G=1 C=512 K=512 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T238
// T238: [1 1 1]

// f32 N=1 G=1 C=704 K=288 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=704 -out_channels=288 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T239
// T239: [1 1 1]

// f32 N=1 G=1 C=400 K=160 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=400 -out_channels=160 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T240
// T240: [1 1 1]

// f32 N=1 G=1 C=336 K=336 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=336 -out_channels=336 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T241
// T241: [1 1 1]

// f32 N=1 G=1 C=192 K=192 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=192 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T242
// T242: [1 1 1]

// f32 N=1 G=1 C=64 K=112 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=112 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T243
// T243: [1 1 1]

// f32 N=1 G=1 C=288 K=288 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=288 -out_channels=288 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T244
// T244: [1 1 1]

// f32 N=1 G=1 C=848 K=512 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=848 -out_channels=512 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T245
// T245: [1 1 1]

// f32 N=1 G=1 C=160 K=160 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=160 -out_channels=160 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T246
// T246: [1 1 1]

// f32 N=1 G=1 C=96 K=48 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=48 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T247
// T247: [1 1 1]

// f32 N=1 G=1 C=112 K=112 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=112 -out_channels=112 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T248
// T248: [1 1 1]

// f32 N=1 G=1 C=96 K=72 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=72 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T249
// T249: [1 1 1]

// f32 N=1 G=1 C=72 K=36 H=768 W=768 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=72 -out_channels=36 -in_h=768 -in_w=768 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T250
// T250: [1 1 1]

// f32 N=1 G=1 C=72 K=72 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=72 -out_channels=72 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T251
// T251: [1 1 1]

// f16 N=1 G=1 C=515 K=512 H=6 W=6 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T252
// T252: [1 1 1]

// f16 N=1 G=1 C=384 K=384 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=384 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T253
// T253: [1 1 1]

// f16 N=1 G=1 C=512 K=511 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=511 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T254
// T254: [1 1 1]

// f16 N=1 G=1 C=192 K=95 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=95 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T255
// T255: [1 1 1]

// f16 N=1 G=1 C=384 K=383 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=383 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T256
// T256: [1 1 1]

// f16 N=1 G=1 C=96 K=96 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=96 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T257
// T257: [1 1 1]

// f16 N=1 G=1 C=384 K=191 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=191 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T258
// T258: [1 1 1]

// f16 N=1 G=1 C=195 K=192 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=195 -out_channels=192 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T259
// T259: [1 1 1]

// f16 N=1 G=1 C=48 K=47 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=47 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T260
// T260: [1 1 1]

// f16 N=1 G=1 C=96 K=95 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=95 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T261
// T261: [1 1 1]

// f16 N=1 G=1 C=192 K=191 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=191 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T262
// T262: [1 1 1]

// f16 N=1 G=1 C=387 K=384 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=387 -out_channels=384 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T263
// T263: [1 1 1]

// f16 N=1 G=1 C=515 K=512 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T264
// T264: [1 1 1]

// f16 N=1 G=1 C=1024 K=511 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=511 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T265
// T265: [1 1 1]

// f16 N=1 G=1 C=48 K=48 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=48 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T266
// T266: [1 1 1]

// f16 N=1 G=1 C=768 K=383 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=768 -out_channels=383 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T267
// T267: [1 1 1]

// f32 N=1 G=1 C=195 K=192 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=195 -out_channels=192 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T268
// T268: [1 1 1]

// f32 N=1 G=1 C=96 K=96 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=96 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T269
// T269: [1 1 1]

// f32 N=1 G=1 C=192 K=95 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=95 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T270
// T270: [1 1 1]

// f32 N=1 G=1 C=384 K=383 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=383 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T271
// T271: [1 1 1]

// f32 N=1 G=1 C=515 K=512 H=6 W=6 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T272
// T272: [1 1 1]

// f32 N=1 G=1 C=384 K=191 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=191 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T273
// T273: [1 1 1]

// f32 N=1 G=1 C=384 K=384 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=384 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T274
// T274: [1 1 1]

// f32 N=1 G=1 C=512 K=511 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=511 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T275
// T275: [1 1 1]

// f32 N=1 G=1 C=96 K=95 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=95 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T276
// T276: [1 1 1]

// f32 N=1 G=1 C=48 K=47 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=47 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T277
// T277: [1 1 1]

// f32 N=1 G=1 C=192 K=191 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=191 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T278
// T278: [1 1 1]

// f32 N=1 G=1 C=387 K=384 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=387 -out_channels=384 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T279
// T279: [1 1 1]

// f32 N=1 G=1 C=1024 K=511 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=511 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T280
// T280: [1 1 1]

// f32 N=1 G=1 C=515 K=512 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T281
// T281: [1 1 1]

// f32 N=1 G=1 C=48 K=48 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=48 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T282
// T282: [1 1 1]

// f32 N=1 G=1 C=768 K=383 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=768 -out_channels=383 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T283
// T283: [1 1 1]

// f32 N=1 G=1 C=512 K=128 H=102 W=102 pad=0,0
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=128 -in_h=102 -in_w=102 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=0 -padding_h_r=0 -padding_w_l=0 -padding_w_r=0 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T284
// T284: [1 1 1]

// f32 N=1 G=1 C=128 K=384 H=102 W=102 pad=0,0
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=384 -in_h=102 -in_w=102 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=0 -padding_h_r=0 -padding_w_l=0 -padding_w_r=0 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T285
// T285: [1 1 1]

// f16 N=1 G=1 C=51 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=51 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T286
// T286: [1 1 1]

// f32 N=1 G=1 C=51 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=51 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T287
// T287: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T288
// T288: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T289
// T289: [1 1 1]

// f32 N=1 G=1 C=48 K=64 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=64 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T290
// T290: [1 1 1]

// f16 N=1 G=1 C=528 K=192 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=528 -out_channels=192 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T291
// T291: [1 1 1]

// f16 N=1 G=1 C=848 K=336 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=848 -out_channels=336 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T292
// T292: [1 1 1]

// f16 N=1 G=1 C=1024 K=512 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=512 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T293
// T293: [1 1 1]

// f16 N=1 G=1 C=192 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T294
// T294: [1 1 1]

// f16 N=1 G=1 C=64 K=64 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T295
// T295: [1 1 1]

// f16 N=1 G=1 C=336 K=512 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=336 -out_channels=512 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T296
// T296: [1 1 1]

// f16 N=1 G=1 C=192 K=336 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=336 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T297
// T297: [1 1 1]

// f16 N=1 G=1 C=112 K=192 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=112 -out_channels=192 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T298
// T298: [1 1 1]

// f16 N=1 G=1 C=192 K=192 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=192 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T299
// T299: [1 1 1]

// f16 N=1 G=1 C=1024 K=512 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=512 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T300
// T300: [1 1 1]

// f16 N=1 G=1 C=112 K=112 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=112 -out_channels=112 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T301
// T301: [1 1 1]

// f16 N=1 G=1 C=176 K=64 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=176 -out_channels=64 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T302
// T302: [1 1 1]

// f16 N=1 G=1 C=336 K=336 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=336 -out_channels=336 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T303
// T303: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T304
// T304: [1 1 1]

// f16 N=1 G=1 C=64 K=112 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=112 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T305
// T305: [1 1 1]

// f16 N=1 G=1 C=304 K=112 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=304 -out_channels=112 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T306
// T306: [1 1 1]

// f16 N=1 G=1 C=112 K=56 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=112 -out_channels=56 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T307
// T307: [1 1 1]

// f32 N=1 G=1 C=192 K=96 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=96 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T308
// T308: [1 1 1]

// f32 N=1 G=1 C=112 K=112 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=112 -out_channels=112 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T309
// T309: [1 1 1]

// f32 N=1 G=1 C=528 K=192 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=528 -out_channels=192 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T310
// T310: [1 1 1]

// f32 N=1 G=1 C=176 K=64 H=512 W=512 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=176 -out_channels=64 -in_h=512 -in_w=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T311
// T311: [1 1 1]

// f32 N=1 G=1 C=848 K=336 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=848 -out_channels=336 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T312
// T312: [1 1 1]

// f32 N=1 G=1 C=192 K=336 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=336 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T313
// T313: [1 1 1]

// f32 N=1 G=1 C=1024 K=512 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=512 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T314
// T314: [1 1 1]

// f32 N=1 G=1 C=336 K=512 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=336 -out_channels=512 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T315
// T315: [1 1 1]

// f32 N=1 G=1 C=112 K=192 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=112 -out_channels=192 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T316
// T316: [1 1 1]

// f32 N=1 G=1 C=1024 K=512 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=512 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T317
// T317: [1 1 1]

// f32 N=1 G=1 C=304 K=112 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=304 -out_channels=112 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T318
// T318: [1 1 1]

// f32 N=1 G=1 C=64 K=112 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=112 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T319
// T319: [1 1 1]

// f32 N=1 G=1 C=336 K=336 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=336 -out_channels=336 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T320
// T320: [1 1 1]

// f32 N=1 G=1 C=112 K=56 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=112 -out_channels=56 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T321
// T321: [1 1 1]

// f16 N=1 G=1 C=96 K=32 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=32 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T322
// T322: [1 1 1]

// f16 N=1 G=1 C=128 K=32 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=32 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T323
// T323: [1 1 1]

// f16 N=1 G=1 C=160 K=32 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=160 -out_channels=32 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T324
// T324: [1 1 1]

// f16 N=1 G=1 C=64 K=32 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=32 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T325
// T325: [1 1 1]

// f16 N=1 G=1 C=192 K=64 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=64 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T326
// T326: [1 1 1]

// f16 N=1 G=1 C=48 K=64 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=64 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T327
// T327: [1 1 1]

// f16 N=1 G=1 C=64 K=64 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T328
// T328: [1 1 1]

// f16 N=1 G=1 C=64 K=64 H=256 W=256 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=256 -in_w=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T329
// T329: [1 1 1]

// f32 N=1 G=1 C=96 K=32 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=32 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T330
// T330: [1 1 1]

// f32 N=1 G=1 C=128 K=32 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=32 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T331
// T331: [1 1 1]

// f32 N=1 G=1 C=160 K=32 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=160 -out_channels=32 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T332
// T332: [1 1 1]

// f32 N=1 G=1 C=64 K=32 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=32 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T333
// T333: [1 1 1]

// f32 N=1 G=1 C=192 K=64 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=64 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T334
// T334: [1 1 1]

// f32 N=1 G=1 C=48 K=64 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=64 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T335
// T335: [1 1 1]

// f32 N=1 G=1 C=64 K=64 H=128 W=128 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=64 -out_channels=64 -in_h=128 -in_w=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T336
// T336: [1 1 1]

// f16 N=1 G=1 C=96 K=48 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=48 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T337
// T337: [1 1 1]

// f16 N=1 G=1 C=48 K=48 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=48 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T338
// T338: [1 1 1]

// f16 N=1 G=1 C=192 K=96 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=96 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T339
// T339: [1 1 1]

// f16 N=1 G=1 C=384 K=192 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=192 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T340
// T340: [1 1 1]

// f16 N=1 G=1 C=96 K=192 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=192 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T341
// T341: [1 1 1]

// f16 N=1 G=1 C=192 K=192 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=192 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T342
// T342: [1 1 1]

// f16 N=1 G=1 C=48 K=96 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=96 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T343
// T343: [1 1 1]

// f16 N=1 G=1 C=96 K=96 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=96 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T344
// T344: [1 1 1]

// f16 N=1 G=1 C=192 K=384 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=384 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T345
// T345: [1 1 1]

// f16 N=1 G=1 C=384 K=1152 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=1152 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T346
// T346: [1 1 1]

// f32 N=1 G=1 C=48 K=48 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=48 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T347
// T347: [1 1 1]

// f32 N=1 G=1 C=192 K=96 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=96 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T348
// T348: [1 1 1]

// f32 N=1 G=1 C=384 K=192 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=192 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T349
// T349: [1 1 1]

// f32 N=1 G=1 C=192 K=192 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=192 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T350
// T350: [1 1 1]

// f32 N=1 G=1 C=96 K=48 H=64 W=64 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=48 -in_h=64 -in_w=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T351
// T351: [1 1 1]

// f32 N=1 G=1 C=96 K=192 H=16 W=16 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=192 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T352
// T352: [1 1 1]

// f32 N=1 G=1 C=48 K=96 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=96 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T353
// T353: [1 1 1]

// f32 N=1 G=1 C=96 K=96 H=32 W=32 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=96 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T354
// T354: [1 1 1]

// f32 N=1 G=1 C=192 K=384 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=384 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T355
// T355: [1 1 1]

// f32 N=1 G=1 C=384 K=1152 H=8 W=8 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=1152 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T356
// T356: [1 1 1]

// f16 N=1 G=1 C=195 K=192 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=195 -out_channels=192 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T357
// T357: [1 1 1]

// f16 N=1 G=1 C=512 K=511 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=511 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T358
// T358: [1 1 1]

// f16 N=1 G=1 C=384 K=191 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=191 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T359
// T359: [1 1 1]

// f16 N=1 G=1 C=48 K=48 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=48 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T360
// T360: [1 1 1]

// f16 N=1 G=1 C=384 K=384 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=384 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T361
// T361: [1 1 1]

// f16 N=1 G=1 C=768 K=383 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=768 -out_channels=383 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T362
// T362: [1 1 1]

// f16 N=1 G=1 C=48 K=47 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=47 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T363
// T363: [1 1 1]

// f16 N=1 G=1 C=192 K=192 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=192 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T364
// T364: [1 1 1]

// f16 N=1 G=1 C=387 K=384 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=387 -out_channels=384 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T365
// T365: [1 1 1]

// f16 N=1 G=1 C=192 K=95 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=95 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T366
// T366: [1 1 1]

// f16 N=1 G=1 C=192 K=191 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=191 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T367
// T367: [1 1 1]

// f16 N=1 G=1 C=384 K=383 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=383 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T368
// T368: [1 1 1]

// f16 N=1 G=1 C=96 K=95 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=95 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T369
// T369: [1 1 1]

// f16 N=1 G=1 C=515 K=512 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T370
// T370: [1 1 1]

// f16 N=1 G=1 C=1024 K=511 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=511 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T371
// T371: [1 1 1]

// f32 N=1 G=1 C=384 K=384 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=384 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T372
// T372: [1 1 1]

// f32 N=1 G=1 C=384 K=383 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=383 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T373
// T373: [1 1 1]

// f32 N=1 G=1 C=512 K=511 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=511 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T374
// T374: [1 1 1]

// f32 N=1 G=1 C=384 K=191 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=191 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T375
// T375: [1 1 1]

// f32 N=1 G=1 C=195 K=192 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=195 -out_channels=192 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T376
// T376: [1 1 1]

// f32 N=1 G=1 C=192 K=95 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=95 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T377
// T377: [1 1 1]

// f32 N=1 G=1 C=387 K=384 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=387 -out_channels=384 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T378
// T378: [1 1 1]

// f32 N=1 G=1 C=192 K=192 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=192 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T379
// T379: [1 1 1]

// f32 N=1 G=1 C=192 K=191 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=192 -out_channels=191 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T380
// T380: [1 1 1]

// f32 N=1 G=1 C=48 K=47 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=47 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T381
// T381: [1 1 1]

// f32 N=1 G=1 C=96 K=95 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=95 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T382
// T382: [1 1 1]

// f32 N=1 G=1 C=1024 K=511 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1024 -out_channels=511 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T383
// T383: [1 1 1]

// f32 N=1 G=1 C=515 K=512 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=515 -out_channels=512 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T384
// T384: [1 1 1]

// f32 N=1 G=1 C=768 K=383 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=768 -out_channels=383 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T385
// T385: [1 1 1]

// f32 N=1 G=1 C=48 K=48 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=48 -out_channels=48 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f32 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T386
// T386: [1 1 1]

// f16 N=1 G=1 C=512 K=256 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=256 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T387
// T387: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T388
// T388: [1 1 1]

// f16 N=1 G=1 C=128 K=128 H=768 W=768 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=768 -in_w=768 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T389
// T389: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T390
// T390: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T391
// T391: [1 1 1]

// f16 N=1 G=1 C=256 K=128 H=768 W=768 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=128 -in_h=768 -in_w=768 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T392
// T392: [1 1 1]

// f16 N=1 G=1 C=4 K=512 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=4 -out_channels=512 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T393
// T393: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=768 W=768 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=768 -in_w=768 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T394
// T394: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T395
// T395: [1 1 1]

// f16 N=1 G=1 C=128 K=128 H=960 W=960 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=128 -in_h=960 -in_w=960 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T396
// T396: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=480 W=480 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=480 -in_w=480 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T397
// T397: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=120 W=120 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=120 -in_w=120 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T398
// T398: [1 1 1]

// f16 N=1 G=1 C=256 K=128 H=960 W=960 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=128 -in_h=960 -in_w=960 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T399
// T399: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=240 W=240 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=240 -in_w=240 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T400
// T400: [1 1 1]

// f16 N=1 G=1 C=512 K=256 H=480 W=480 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=256 -in_h=480 -in_w=480 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T401
// T401: [1 1 1]

// f16 N=1 G=1 C=4 K=512 H=120 W=120 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=4 -out_channels=512 -in_h=120 -in_w=120 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T402
// T402: [1 1 1]

// f16 N=1 G=1 C=256 K=256 H=960 W=960 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=256 -in_h=960 -in_w=960 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T403
// T403: [1 1 1]

// f16 N=1 G=1 C=512 K=512 H=480 W=480 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=512 -in_h=480 -in_w=480 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T404
// T404: [1 1 1]

// f16 N=1 G=1 C=128 K=256 H=384 W=384 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=256 -in_h=384 -in_w=384 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T405
// T405: [1 1 1]

// f16 N=1 G=1 C=512 K=8 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=8 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T406
// T406: [1 1 1]

// f16 N=1 G=1 C=256 K=512 H=192 W=192 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=512 -in_h=192 -in_w=192 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T407
// T407: [1 1 1]

// f16 N=1 G=1 C=128 K=256 H=480 W=480 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=128 -out_channels=256 -in_h=480 -in_w=480 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T408
// T408: [1 1 1]

// f16 N=1 G=1 C=512 K=8 H=120 W=120 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=512 -out_channels=8 -in_h=120 -in_w=120 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T409
// T409: [1 1 1]

// f16 N=1 G=1 C=256 K=512 H=240 W=240 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=512 -in_h=240 -in_w=240 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T410
// T410: [1 1 1]

// f16 N=1 G=1 C=1920 K=1280 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1920 -out_channels=1280 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T411
// T411: [1 1 1]

// f16 N=1 G=1 C=1280 K=1280 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1280 -out_channels=1280 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T412
// T412: [1 1 1]

// f16 N=1 G=1 C=640 K=640 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=640 -out_channels=640 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T413
// T413: [1 1 1]

// f16 N=1 G=1 C=8 K=320 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=8 -out_channels=320 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T414
// T414: [1 1 1]

// f16 N=1 G=1 C=320 K=320 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=320 -out_channels=320 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T415
// T415: [1 1 1]

// f16 N=1 G=1 C=960 K=640 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=960 -out_channels=640 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T416
// T416: [1 1 1]

// f16 N=1 G=1 C=1280 K=1280 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1280 -out_channels=1280 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T417
// T417: [1 1 1]

// f16 N=1 G=1 C=640 K=320 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=640 -out_channels=320 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T418
// T418: [1 1 1]

// f16 N=1 G=1 C=2560 K=1280 H=12 W=12 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=2560 -out_channels=1280 -in_h=12 -in_w=12 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T419
// T419: [1 1 1]

// f16 N=1 G=1 C=1280 K=640 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1280 -out_channels=640 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T420
// T420: [1 1 1]

// f16 N=1 G=1 C=2560 K=1280 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=2560 -out_channels=1280 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T421
// T421: [1 1 1]

// f16 N=1 G=1 C=320 K=640 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=320 -out_channels=640 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T422
// T422: [1 1 1]

// f16 N=1 G=1 C=1280 K=1280 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1280 -out_channels=1280 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T423
// T423: [1 1 1]

// f16 N=1 G=1 C=640 K=640 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=640 -out_channels=640 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T424
// T424: [1 1 1]

// f16 N=1 G=1 C=640 K=1280 H=24 W=24 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=640 -out_channels=1280 -in_h=24 -in_w=24 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T425
// T425: [1 1 1]

// f16 N=1 G=1 C=1920 K=640 H=48 W=48 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1920 -out_channels=640 -in_h=48 -in_w=48 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T426
// T426: [1 1 1]

// f16 N=1 G=1 C=960 K=320 H=96 W=96 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=960 -out_channels=320 -in_h=96 -in_w=96 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T427
// T427: [1 1 1]

// f16 N=1 G=1 C=1280 K=1280 H=15 W=15 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1280 -out_channels=1280 -in_h=15 -in_w=15 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T428
// T428: [1 1 1]

// f16 N=1 G=1 C=2560 K=1280 H=30 W=30 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=2560 -out_channels=1280 -in_h=30 -in_w=30 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T429
// T429: [1 1 1]

// f16 N=1 G=1 C=320 K=320 H=120 W=120 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=320 -out_channels=320 -in_h=120 -in_w=120 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T430
// T430: [1 1 1]

// f16 N=1 G=1 C=1920 K=640 H=60 W=60 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1920 -out_channels=640 -in_h=60 -in_w=60 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T431
// T431: [1 1 1]

// f16 N=1 G=1 C=4 K=640 H=120 W=120 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=4 -out_channels=640 -in_h=120 -in_w=120 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T432
// T432: [1 1 1]

// f16 N=1 G=1 C=960 K=320 H=120 W=120 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=960 -out_channels=320 -in_h=120 -in_w=120 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T433
// T433: [1 1 1]

// f16 N=1 G=1 C=1280 K=1280 H=30 W=30 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1280 -out_channels=1280 -in_h=30 -in_w=30 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T434
// T434: [1 1 1]

// f16 N=1 G=1 C=640 K=640 H=60 W=60 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=640 -out_channels=640 -in_h=60 -in_w=60 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T435
// T435: [1 1 1]

// f16 N=1 G=1 C=96 K=96 H=240 W=240 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=96 -out_channels=96 -in_h=240 -in_w=240 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T436
// T436: [1 1 1]

// f16 N=1 G=1 C=1920 K=1280 H=30 W=30 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1920 -out_channels=1280 -in_h=30 -in_w=30 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T437
// T437: [1 1 1]

// f16 N=1 G=1 C=2560 K=1280 H=15 W=15 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=2560 -out_channels=1280 -in_h=15 -in_w=15 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T438
// T438: [1 1 1]

// f16 N=1 G=1 C=960 K=640 H=60 W=60 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=960 -out_channels=640 -in_h=60 -in_w=60 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T439
// T439: [1 1 1]

// f16 N=1 G=1 C=640 K=320 H=120 W=120 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=640 -out_channels=320 -in_h=120 -in_w=120 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T440
// T440: [1 1 1]

// f16 N=1 G=1 C=256 K=320 H=120 W=120 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=256 -out_channels=320 -in_h=120 -in_w=120 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T441
// T441: [1 1 1]

// f16 N=1 G=1 C=1280 K=640 H=60 W=60 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1280 -out_channels=640 -in_h=60 -in_w=60 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T442
// T442: [1 1 1]

// f16 N=1 G=1 C=320 K=640 H=60 W=60 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=320 -out_channels=640 -in_h=60 -in_w=60 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T443
// T443: [1 1 1]

// f16 N=1 G=1 C=640 K=1280 H=30 W=30 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=640 -out_channels=1280 -in_h=30 -in_w=30 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T444
// T444: [1 1 1]

// f16 N=1 G=1 C=1280 K=1280 H=60 W=60 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=1280 -out_channels=1280 -in_h=60 -in_w=60 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T445
// T445: [1 1 1]

// f16 N=1 G=1 C=640 K=640 H=120 W=120 pad=1,1
// RUN: rocmlir-gen --arch %arch --operation conv -groupsize=1 -batchsize=1 -in_channels=640 -out_channels=640 -in_h=120 -in_w=120 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -t f16 %pv -rand 1 -rand_type float -RMS_threshold 0.0001 -absDiff_threshold 0.01 -relDiff_threshold 0.000001 %rocmlir_gen_flags | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=T446
// T446: [1 1 1]
