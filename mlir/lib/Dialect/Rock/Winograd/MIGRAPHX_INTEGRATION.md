# MIGraphX Integration Guide for Winograd Assembly Kernels

## Overview

This document describes what MIGraphX needs to do to launch Winograd assembly
kernels produced by rocMLIR. The rocMLIR side is complete -- Winograd kernels
are assembled to HSACO and exposed through the existing C API. MIGraphX needs
to handle the different kernel argument layout.

## Background

rocMLIR's GEMM-compiled kernels take 3 bare pointers as kernel arguments:
```
kernarg (24 bytes): [filter_ptr, input_ptr, output_ptr]
```

Winograd assembly kernels take a 232-byte packed argument struct (V2 ABI):
```
kernarg (232 bytes): [N, C, H, W, K, n_groups, flags64, data_ptr, filter_ptr,
                      output_ptr, ..., strides, ..., activation, ...]
```

The HSACO binary and kernel metadata are returned through the same existing
C API functions. MIGraphX only needs to change how it packs the kernel
arguments at launch time.

## Detection: How MIGraphX Knows It's a Winograd Kernel

After calling `mlirMIGraphXAddBackendPipeline` or `miirLowerBin`, MIGraphX
can detect a Winograd kernel by checking the kernel name:

```cpp
// After compilation, get kernel info
uint32_t attrs[2];
mlirGetKernelAttrs(module, attrs);
uint32_t blockSize = attrs[0], gridSize = attrs[1];

size_t binSize;
mlirGetBinary(module, &binSize, nullptr);
char* binary = malloc(binSize);
mlirGetBinary(module, nullptr, binary);

// Get kernel name from the gpu.binary metadata
// Winograd kernel names start with "miopenSp3Asm"
bool isWinograd = kernelName.starts_with("miopenSp3Asm");
```

Alternatively, check if the perf_config starts with `winograd:`:
```cpp
// The perf_config that was set on the conv op
bool isWinograd = perfConfig.starts_with("winograd:");
```

## Launching: Packing the V2 ABI Argument Buffer

When launching a Winograd kernel, MIGraphX must construct the 232-byte
argument buffer and use `HIP_LAUNCH_PARAM_BUFFER_POINTER` instead of the
individual-arg `void** params` approach.

### V2 ABI Struct Layout (232 bytes)

```
Offset  Size  Type     Name           Source
------  ----  ----     ----           ------
  0      4    uint32   N              from conv problem
  4      4    uint32   C              from conv problem (per group)
  8      4    uint32   H              from conv problem
 12      4    uint32   W              from conv problem
 16      4    uint32   K              from conv problem (per group)
 20      4    uint32   n_groups       from kernel metadata (grid_size)
 24      8    uint64   flags64        see below
 32      8    ptr      data_addr      runtime: input GPU pointer
 40      8    ptr      filter_addr    runtime: filter GPU pointer
 48      8    ptr      output_addr    runtime: output GPU pointer
 56      8    uint64   reserved       0
 64      4    uint32   R              from conv problem (filter height)
 68      4    uint32   S              from conv problem (filter width)
 72      4    int32    pad_h          from conv problem
 76      4    int32    pad_w          from conv problem
 80      4    uint32   out_h          from conv problem
 84      4    uint32   out_w          from conv problem
 88      8    ptr      bias_addr      null (no bias fusion)
 96      4    float    alpha          1.0
100      4    float    beta           0.0
104      8    uint64   d_offset       0
112      8    uint64   f_offset       0
120      8    uint64   o_offset       0
128      8    uint64   b_offset       0
136      4    uint32   d_N_stride     C * H * W (elements)
140      4    uint32   d_C_stride     H * W (elements)
144      4    uint32   d_H_stride     W (elements)
148      4    uint32   (reserved)     0
152      4    uint32   f_K_stride     C * R * S (elements)
156      4    uint32   f_C_stride     R * S (elements)
160      4    uint32   f_R_stride     S (elements)
164      4    uint32   (reserved)     0
168      4    uint32   o_N_stride     K * out_h * out_w (elements)
172      4    uint32   o_K_stride     out_h * out_w (elements)
176      4    uint32   o_H_stride     out_w (elements)
180      4    uint32   (reserved)     0
184      4    uint32   G              group count (1 for non-grouped)
188      4    uint32   d_G_stride     C * H * W (elements)
192      4    uint32   f_G_stride     K * C * R * S (elements)
196      4    uint32   o_G_stride     K * out_h * out_w (elements)
200      1    uint8    activation     0 (identity, no activation)
201      1    uint8    sync_limit     0
202      1    uint8    sync_period    0
203      1    uint8    (reserved)     0
204      4    uint32   (reserved)     0
208      8    ptr      sync_addr      null
216      8    ptr      acc_addr       null
224      8    uint64   a_offset       0
```

All strides are in **elements** (not bytes). The kernel handles
element-to-byte conversion internally based on the data type.

### flags64 Value

For forward convolution:
```cpp
uint64_t flags64 = (1ULL << 3)    // F_DENORMS_RND_ENABLE
                 | (1ULL << 9)    // F_NKCHR_STRIDES
                 | (1ULL << 13)   // F_TENSOR_OFFSETS
                 | (1ULL << 14)   // F_USE_ACTIVATION_MODE
                 | (1ULL << 15);  // F_USE_EXTENDED_FLAGS_64
```

For backward data, add:
```cpp
flags64 |= (1ULL << 0)   // F_REVERSE_R
         | (1ULL << 1);  // F_REVERSE_S
```

### C++ Example: Packing and Launching

```cpp
#include <hip/hip_runtime.h>
#include <cstring>

struct __attribute__((packed)) WinoV2Args {
    uint32_t N, C, H, W, K, n_groups;
    uint64_t flags64;
    void *data_addr, *filter_addr, *output_addr;
    uint64_t reserved0;
    uint32_t R, S;
    int32_t pad_h, pad_w;
    uint32_t out_h, out_w;
    void *bias_addr;
    float alpha, beta;
    uint64_t d_offset, f_offset, o_offset, b_offset;
    uint32_t d_N_stride, d_C_stride, d_H_stride, d_W_stride;
    uint32_t f_K_stride, f_C_stride, f_R_stride, f_S_stride;
    uint32_t o_N_stride, o_K_stride, o_H_stride, o_W_stride;
    uint32_t G, d_G_stride, f_G_stride, o_G_stride;
    uint8_t activation_mode, sync_limit, sync_period, reserved1;
    uint32_t reserved2;
    void *sync_addr, *acc_addr;
    uint64_t a_offset;
};
static_assert(sizeof(WinoV2Args) == 232);

void launchWinogradKernel(
    hipFunction_t kernel,
    // Conv problem dimensions (compile-time known)
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int out_h, int out_w,
    int n_groups, int group,
    bool isForward,
    // Runtime GPU pointers
    void* input_gpu, void* filter_gpu, void* output_gpu,
    // Launch config
    int gridSize, int blockSize,
    hipStream_t stream)
{
    WinoV2Args args = {};
    args.N = N; args.C = C; args.H = H; args.W = W;
    args.K = K; args.n_groups = n_groups;

    args.flags64 = (1ULL<<3) | (1ULL<<9) | (1ULL<<13) | (1ULL<<14) | (1ULL<<15);
    if (!isForward)
        args.flags64 |= (1ULL<<0) | (1ULL<<1);

    args.data_addr = input_gpu;
    args.filter_addr = filter_gpu;
    args.output_addr = output_gpu;

    args.R = R; args.S = S;
    args.pad_h = pad_h; args.pad_w = pad_w;
    args.out_h = out_h; args.out_w = out_w;
    args.alpha = 1.0f; args.beta = 0.0f;

    // Element strides (NCHW layout)
    args.d_H_stride = W;
    args.d_C_stride = H * W;
    args.d_N_stride = C * H * W;
    args.d_G_stride = C * H * W;
    args.f_R_stride = S;
    args.f_C_stride = R * S;
    args.f_K_stride = C * R * S;
    args.f_G_stride = K * C * R * S;
    args.o_H_stride = out_w;
    args.o_K_stride = out_h * out_w;
    args.o_N_stride = K * out_h * out_w;
    args.o_G_stride = K * out_h * out_w;
    args.G = group;

    // Launch with packed buffer
    size_t argSize = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
        HIP_LAUNCH_PARAM_END
    };

    hipModuleLaunchKernel(kernel,
        gridSize, 1, 1,    // grid
        blockSize, 1, 1,   // block
        0,                 // shared mem (kernel descriptor handles LDS)
        stream,
        nullptr,           // params = null (using extra instead)
        config);           // extra = packed buffer
}
```

### Key Notes

1. `sharedMem` parameter to `hipModuleLaunchKernel` must be **0**. The kernel
   descriptor already declares `group_segment_fixed_size = 65536` (64KB LDS).
   The HIP runtime allocates this from the kernel descriptor, not from the
   launch parameter.

2. Strides are in **elements**, not bytes. The kernel multiplies by element
   size internally.

3. `n_groups` should equal `gridSize` (one workgroup per group).

4. Block sizes: Rage kernels use 768 on gfx942, 384 on gfx12. Fury kernels
   use 384. V30 kernels use 512.

## What MIGraphX Does NOT Need to Change

- `mlirMIGraphXAddHighLevelPipeline` -- unchanged
- `mlirMIGraphXAddApplicabilityPipeline` -- unchanged (Winograd configs
  correctly report as not-applicable through this pipeline; the assembly
  path is taken later)
- `mlirMIGraphXAddBackendPipeline` -- unchanged (WinogradInterceptPass
  runs automatically when perf_config starts with "winograd:")
- `mlirGetKernelAttrs` -- returns correct block_size and grid_size
- `mlirGetBinary` -- returns the assembled HSACO
- `mlirGetKernelInfo` -- returns the 3-memref func.func signature
  (same as GEMM). MIGraphX uses this to know how many buffer args to
  pass. For Winograd, the buffers are the same (filter, input, output)
  but the kernel launch packs them differently.
- `mlirRockTuningSpaceCreate` -- unchanged (Winograd entries will appear
  once tuning integration is enabled)
- `mlirIsModuleFusible` -- returns false for Winograd configs
  (fusions fall back to GEMM automatically)

## What MIGraphX DOES Need to Change

1. **Detect Winograd kernel** after compilation (check kernel name prefix
   or perf_config prefix)

2. **Pack the 232-byte V2 ABI argument buffer** using the conv problem
   dimensions (known at compile time) and the 3 runtime GPU pointers

3. **Launch with `HIP_LAUNCH_PARAM_BUFFER_POINTER`** via the `extra`
   parameter of `hipModuleLaunchKernel`, not the `params` parameter

4. **Set `sharedMem = 0`** in the launch call (kernel descriptor handles LDS)

## Supported Configurations

| Arch | Kernel Family | Data Types | Filter | Direction |
|------|---------------|------------|--------|-----------|
| gfx942 | Rage v4.9 | fp16, fp32, bf16 | 3x3, stride 1 | Fwd, BwdData |
| gfx942 | Rage v4.6 | fp16 | 3x3, stride 1 | Fwd, BwdData |
| gfx9xx | V30 | fp16, fp32 | 3x3, stride 1-2 | Fwd, BwdData, WrW |
| gfx11xx | Fury v2 | fp16 | 3x3, stride 1 | Fwd, BwdData |
| gfx12xx | V40 | fp16, fp32 | 3x3, stride 1-2 | Fwd, BwdData, WrW |
| gfx12xx | Fury v4 | fp16 | 3x3, stride 1 | Fwd, BwdData |
| gfx12xx | Rage v4.6/v4.9 | fp16 | 3x3, stride 1 | Fwd, BwdData |

Layout: NCHW only. NHWC convolutions fall back to GEMM automatically.
Fusions: Not supported. Fused modules fall back to GEMM automatically.

## Performance

Benchmarked on AMD Instinct MI300X (gfx942, 304 CUs):

- Geometric mean speedup vs greedy-tuned GEMM: **2.39x** on 61 fp16 3x3 configs
- Winograd wins **78%** of configs (48/61)
- Strongest on large spatial dims (28x28+): **1.5x-4x**
- GEMM wins on small spatial with large channels (7x7 C=512+): **1.5-2x**
