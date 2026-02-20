// Test various GEMM configurations with double buffering (scheduleVersion=2).
// This verifies correctness of the pipelining with double buffering enabled,
// including 8-wave configurations (blockSize = 8 * waveSize).
//
// When blockSize == 8 * waveSize AND initiationInterval == 1 (double buffering),
// the "compute-first" schedule is used. This schedule reorders stages so that
// MMA (compute) executes first in the main loop, followed by a single barrier,
// then LDSRead, LDSWrite, and GlobalRead. This enables "ping-pong" scheduling
// where different wave groups can overlap compute and memory operations.
// See RockPipeline.cpp createComputeFirstSchedule() for details.

// Test 1: Basic f16 GEMM 256x256x256 with schedule_version=2
// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -g 1 -m 256 -k 256 -n 256 --schedule_version=2 -pv | rocmlir-driver -c | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Test 2: f32 GEMM with schedule_version=2
// RUN: rocmlir-gen --arch %arch --operation gemm -t f32 -g 1 -m 128 -k 128 -n 128 --schedule_version=2 -pv | rocmlir-driver -c | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Test 3: bf16 GEMM with schedule_version=2
// RUN: rocmlir-gen --arch %arch --operation gemm -t bf16 -g 1 -m 128 -k 128 -n 128 --schedule_version=2 -pv | rocmlir-driver -c | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Test 4: 8-wave config (blockSize=512) with scheduleVersion=2 (double buffer)
// v4:mPerBlock,nPerBlock,kpackPerBlock,mPerWave,nPerWave,mnPerXdl,kpack,splitKFactor,scheduleVersion,...
// v4:128,64,8,32,32,32,8,1,2,2,0,0,1,1 gives 4*2=8 waves, blockSize=512, double buffered
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -g 1 -m 512 -k 512 -n 512 --perf_config="v4:128,64,8,32,32,32,8,1,2,2,0,0,1,1" -pv | rocmlir-driver -c | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Test 5: Another 8-wave config (blockSize=512) with scheduleVersion=2
// v4:64,128,8,32,32,32,8,1,2,2,0,0,1,1 gives 2*4=8 waves, blockSize=512, double buffered
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -g 1 -m 512 -k 512 -n 512 --perf_config="v4:64,128,8,32,32,32,8,1,2,2,0,0,1,1" -pv | rocmlir-driver -c | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Test 6: 8-wave config with single buffer for comparison
// v4:128,64,8,32,32,32,8,1,1,2,0,0,1,1 gives 4*2=8 waves, blockSize=512, single buffered
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -g 1 -m 512 -k 512 -n 512 --perf_config="v4:128,64,8,32,32,32,8,1,1,2,0,0,1,1" -pv | rocmlir-driver -c | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Test 7: Larger GEMM with schedule_version=2
// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -g 1 -m 512 -k 512 -n 512 --schedule_version=2 -pv | rocmlir-driver -c | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]
