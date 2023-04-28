// I picked an arch that doesn't have MFMA to test --mfma=on. If we start doing
// stricter checks of MFMA availability, then change this to gfx908

// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 --mfma on -n 128 -k 8 -m 256 --perf_config 128,64,4,64,64,1,1,1 | FileCheck %s --check-prefix=GEN
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 --mfma on -n 128 -k 8 -m 256 --perf_config 128,64,4,64,64,1,1,1 | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=AFFIX
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 --mfma on -n 128 -k 8 -m 256 --perf_config 128,64,4,64,64,1,1,1 | rocmlir-opt --rock-affix-params --rock-gemm-to-gridwise | FileCheck %s --check-prefix=GRIDWISE

// GEN: rock.gemm
// CHECK-SAME: features = mfma|dot
// CHECK-SAME: arch = "amdgcn-amd-amdhsa:gfx1030"
// CHECK-SAME: perf_config = 128,64,4,64,64,1,1,1
// AFFIX: #rock.xdlops_gemm_params<kPerBlock = 4, mPerBlock = 128, nPerBlock = 64, kpack = 1, mPerWave = 64, nPerWave = 64, forceUnroll = true>
// GRIDWISE: rock.gridwise_gemm_v2
