// Loop unrolling stubbed out by hotfix
// XFAIL: *

// RUN: rocmlir-gen --operation conv2d -t f32 --arch gfx908 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 1024 --in_h 14 --in_w 14 --out_channels 2048 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --perf_config="16,16,64,16,16,1,1,1" | rocmlir-driver  -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=UNROLL

// UNROLL: #xdlops_gemm_params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 64, mPerBlock = 16, nPerBlock = 16, kpack = 1, mPerWave = 16, nPerWave = 16, mnPerXdl = 32, forceUnroll = true>
// UNROLL-COUNT-14: rock.transforming_for {forceUnroll, useIndexDiffs}
// UNROLL-NOT: rock.transforming_for {useIndexDiffs}

// RUN: rocmlir-gen --operation conv2d -t f32 --arch gfx908 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 1024 --in_h 14 --in_w 14 --out_channels 2048 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --perf_config="16,16,64,16,16,1,0,1" | rocmlir-driver  -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=NOTUNROLL

// NOTUNROLL: #xdlops_gemm_params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 64, mPerBlock = 16, nPerBlock = 16, kpack = 1, mPerWave = 16, nPerWave = 16, mnPerXdl = 32, forceUnroll = false>
// NOTUNROLL-COUNT-14: rock.transforming_for {useIndexDiffs}
// NOTUNROLL-NOT: rock.transforming_for {forceUnroll, useIndexDiffs}

// RUN: rocmlir-gen --operation conv2d -t f32 --arch gfx908 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 1024 --in_h 14 --in_w 14 --out_channels 2048 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 | rocmlir-driver  -rock-affix-params | FileCheck %s --check-prefix=HEURISTIC

// HEURISTIC: #xdlops_gemm_params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, forceUnroll = true>
