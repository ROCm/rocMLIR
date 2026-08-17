// Arch is fixed here because not all architectures have atomic_add
// RUN: rocmlir-gen --arch gfx908 --operation gemm -p --store-method atomic_add | FileCheck %s --check-prefix=ATOMIC_ADD
// ATOMIC_ADD: rock.gemm
// ATOMIC_ADD-SAME: storeMethod = atomic_add
// RUN: rocmlir-gen --emit-tuning-key -p --arch gfx900 | FileCheck %s --check-prefix=CONV
// CONV: amdgcn-amd-amdhsa:gfx900   {{.*}}     conv -F 1 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1
// RUN: rocmlir-gen --emit-tuning-key -p -t fp8_fp8 --arch gfx900 | FileCheck %s --check-prefix=CONVFP8
// CONVFP8: amdgcn-amd-amdhsa:gfx900   {{.*}}     convfp8_fp8 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1
// RUN: rocmlir-gen --emit-tuning-key -p -t fp8_fp8 --arch gfx1201 | FileCheck %s --check-prefix=CONVOCPFP8
// CONVOCPFP8: amdgcn-amd-amdhsa:gfx1201   {{.*}}     convfp8_fp8 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1
// RUN: rocmlir-gen --arch gfx908 --operation gemm -p --emit-tuning-key | FileCheck %s --check-prefix=GEMM
// GEMM: amdgcn-amd-amdhsa:gfx908   {{.*}}     -t f32 -out_datatype f32 -transA false -transB false -g 1 -m 1024 -n 512 -k 769 -supportsSplitK true
// RUN: rocmlir-gen --emit-tuning-key -p -t fp8_fp8 --arch gfx950 | FileCheck %s --check-prefix=CONVOCPFP8_GFX950
// CONVOCPFP8_GFX950: amdgcn-amd-amdhsa:gfx950   {{.*}}     convfp8_fp8 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -p --num_cu 40 --num_chiplets 20 | FileCheck %s --check-prefix=NUM_CHIPLETS
// NUM_CHIPLETS: rock.num_chiplets = 20 : i64, rock.num_cu = 40 : i64

// `--emit-tuning-key` for backward-data / backward-weight: same conv key
// payload but with `-F 2` / `-F 4` instead of `-F 1`.
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_data -p -t f32 --emit-tuning-key | FileCheck %s --check-prefix=BWD_DATA_KEY
// BWD_DATA_KEY: amdgcn-amd-amdhsa:gfx942   {{.*}}     conv -F 2 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_weight -p -t f32 --emit-tuning-key | FileCheck %s --check-prefix=BWD_WRW_KEY
// BWD_WRW_KEY: amdgcn-amd-amdhsa:gfx942   {{.*}}     conv -F 4 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1

// `-pi` (`--print-inputs`) prints every input tensor of the host harness
// (all kernel args except the output). For a 3-arg GEMM that is A and B,
// emitted as two `printMemrefF32` calls. `-pr` and `-pvr` are already
// covered by populate_host_print*.mlir and the fusion E2E tests.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph -pi | FileCheck %s --check-prefix=PRINT_INPUTS
// PRINT_INPUTS-LABEL: func.func @main()
// PRINT_INPUTS-COUNT-2: call @printMemrefF32
// PRINT_INPUTS-NOT: call @printMemrefF32

// `--print-verify-results=<level>` is forwarded to `mcpuVerifyFloat` as the
// trailing `i8` constant in the call argument list (off=0, summary=1,
// failure=2, always=3). Summary is the default.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -pv | FileCheck %s --check-prefix=VERIFY_SUMMARY
// VERIFY_SUMMARY: %[[lvl:.*]] = arith.constant 1 : i8
// VERIFY_SUMMARY: call @mcpuVerifyFloat({{.*}}, %[[lvl]], %{{.*}}) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1, i1) -> ()
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -pv --print-verify-results=always | FileCheck %s --check-prefix=VERIFY_ALWAYS
// VERIFY_ALWAYS: %[[lvl:.*]] = arith.constant 3 : i8
// VERIFY_ALWAYS: call @mcpuVerifyFloat({{.*}}, %[[lvl]], %{{.*}}) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1, i1) -> ()
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -pv --print-verify-results=off | FileCheck %s --check-prefix=VERIFY_OFF
// VERIFY_OFF: %[[lvl:.*]] = arith.constant 0 : i8
// VERIFY_OFF: call @mcpuVerifyFloat({{.*}}, %[[lvl]], %{{.*}}) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1, i1) -> ()
// Same flag for integer kernels. With an i32 output and an i64 CPU reference
// the verifier helper is `mcpuVerifyInt32Int64`.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t i8 -out_datatype i32 -g 1 -m 32 -n 32 -k 32 -pv --print-verify-results=failure | FileCheck %s --check-prefix=VERIFY_INT
// VERIFY_INT: %[[lvl:.*]] = arith.constant 2 : i8
// VERIFY_INT: call @mcpuVerifyInt32Int64({{.*}}, %[[lvl]]) : (memref<?xi32>, memref<?xi64>, i8) -> ()

// `--device <N>` (and its `-dev` alias) registers a global constructor that
// calls `gpu.set_default_device` with the requested device index. The
// constructor is only emitted when the flag is actually passed.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph | FileCheck %s --check-prefix=NO_DEVICE
// NO_DEVICE-NOT: llvm.func @setDeviceCtor
// NO_DEVICE-NOT: gpu.set_default_device
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph --device 1 | FileCheck %s --check-prefix=DEVICE_1
// DEVICE_1: llvm.func @setDeviceCtor()
// DEVICE_1: %[[idx:.*]] = arith.constant 1 : i32
// DEVICE_1: gpu.set_default_device %[[idx]]
// DEVICE_1: llvm.mlir.global_ctors ctors = [@setDeviceCtor], priorities = [122 : i32], data = [#llvm.zero]
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph -dev 2 | FileCheck %s --check-prefix=DEVICE_2
// DEVICE_2: llvm.func @setDeviceCtor()
// DEVICE_2: %[[idx:.*]] = arith.constant 2 : i32
// DEVICE_2: gpu.set_default_device %[[idx]]
