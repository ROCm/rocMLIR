// I picked an arch that doesn't have MFMA to test --mfma=on. If we start doing
// stricter checks of MFMA availability, then change this to gfx908

// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 --mfma on -n 128 -k 8 -m 256 --perf_config "v3:128,64,4,64,32,1,1,1,2,1,1" | FileCheck %s --check-prefix=GEN
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 --mfma on -n 128 -k 8 -m 256 --perf_config "v3:128,64,4,64,32,1,1,1,2,1,1" | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=AFFIX
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 --mfma on -n 128 -k 8 -m 256 --perf_config "v3:128,64,4,64,32,1,1,1,2,1,1" | rocmlir-opt --rock-affix-params --rock-gemm-to-gridwise | FileCheck %s --check-prefix=GRIDWISE

// GEN: rock.gemm
// CHECK-SAME: features = mfma|dot
// CHECK-SAME: arch = "amdgcn-amd-amdhsa:gfx1030"
// CHECK-SAME: perf_config = "v3:128,64,4,64,64,1,1,1,2,1,1"
// AFFIX: #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 64, kpack = 1, mPerWave = 64, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>
// GRIDWISE: rock.gridwise_gemm_accel

// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 --mfma on -n 128 -k 8 -m 256 --perf_config "v3:128,64,4,64,32,1,1,2,2,1,1" | FileCheck %s --check-prefix=GEN_V2
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 --mfma on -n 128 -k 8 -m 256 --perf_config "v3:128,64,4,64,32,1,1,2,2,1,1" | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=AFFIX_V2
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 --mfma on -n 128 -k 8 -m 256 --perf_config "v3:128,64,4,64,32,1,1,2,2,1,1" | rocmlir-opt --rock-affix-params --rock-gemm-to-gridwise | FileCheck %s --check-prefix=GRIDWISE_V2

// GEN_V2: rock.gemm
// CHECK-SAME: features = mfma|dot
// CHECK-SAME: arch = "amdgcn-amd-amdhsa:gfx1030"
// CHECK-SAME: perf_config = "v3:128,64,4,64,64,1,1,2,2,1,1"
// AFFIX_V2: #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 64, kpack = 1, mPerWave = 64, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, forceUnroll = true>
// GRIDWISE_V2: rock.gridwise_gemm_accel
