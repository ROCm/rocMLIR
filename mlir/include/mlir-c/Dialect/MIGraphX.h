//===-- mlir-c/Dialect/MIGraphX.h - C API for MIGraphX dialect --------*- C
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_MIGRAPHX_H
#define MLIR_C_DIALECT_MIGRAPHX_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

// Version 2: Use bare pointer ABI (kernels take just a pointer to the data
// buffer, not an entire memref struct). Also introduces this constant.
// Version 3: mlirMIGraphXAddBackendPipeline() to get full arch name instead of
// split strings of triple/chip/features
// Version 4: The MLIR shaped type is added to better represent MIGRaphX's
// native type
// Version 5: Breaking changes:
//   - mlirMIGraphXAddBackendPipeline() now takes a MlirMIGraphXBackendOptions*
//     (arch, perfConfig, optLevel) instead of a separate arch arg.
//   - mlirGetKernelAttrs() returns uint32_t[3] {block_size, grid_size,
//     cluster_size} instead of uint32_t[2] {block_size, grid_size}.
//   - Removed: mlirGetKernelInfo(), mlirMIGraphXAddApplicabilityPipeline().
//   - Added: rocmlirMIGraphXAttentionCreate() for building migraphx.attention
//     ops with variadic inputs, optional LSE, softmaxType, preSoftmaxBody,
//     feature flags (kvcache, causal, prefix_offset, sliding_window, splitkv),
//     currentSeqLen, prefixOffset, splitKV, and slidingWindowSize.
#define MLIR_MIGRAPHX_DIALECT_API_VERSION 5

typedef struct MlirMIGraphXBackendOptions {
  const char *arch;
  const char *perfConfig;
  int optLevel;
} MlirMIGraphXBackendOptions;

#define MLIR_MIGRAPHX_ATTENTION_NONE 0
#define MLIR_MIGRAPHX_ATTENTION_KVCACHE (1 << 0)
#define MLIR_MIGRAPHX_ATTENTION_CAUSAL (1 << 1)
#define MLIR_MIGRAPHX_ATTENTION_PREFIX_OFFSET (1 << 2)
#define MLIR_MIGRAPHX_ATTENTION_SLIDING_WINDOW (1 << 3)
#define MLIR_MIGRAPHX_ATTENTION_SPLITKV (1 << 4)

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx);

// Types

MLIR_CAPI_EXPORTED MlirTypeID rocmlirMIXRShapedTypeGetTypeId(void);

MLIR_CAPI_EXPORTED bool rocmlirIsAMIXRShapedType(MlirType type);

MLIR_CAPI_EXPORTED MlirType rocmlirMIXRShapedTypeGet(intptr_t rank,
                                                     const int64_t *shape,
                                                     const int64_t *strides,
                                                     MlirType elementType);

MLIR_CAPI_EXPORTED MlirType rocmlirMIXRShapedTypeAsTensor(MlirType type);

// Returns block_size, grid_size and cluster_size as uint32_t[3]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, uint32_t *attrs);

// Returns the size of compiled binary if called with null ptr
// and return the compiled binary when buffer is provided
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, size_t *size,
                                      char *bin);

// pipelines

/// Add the high-level pipeline that creates something that can be tuned.
/// Architecture, num_cu and num_chiplets information should be set on the
/// kernel function being compiled.
MLIR_CAPI_EXPORTED void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm);

/// Adds a full compile pipeline to the pass manager. This pipeline may either
/// receive the results of the high-level pipeline.
MLIR_CAPI_EXPORTED bool
mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                               const MlirMIGraphXBackendOptions *opts);

// Op creation helpers

/// Creates a `migraphx.attention` operation.
///
/// \p queries, \p keys, \p values are the required Q, K, V operands.
/// \p preSoftmaxElemWiseInputs is an array of \p numPreSoftmaxInputs additional
///    operands for element-wise fusion before softmax (can be NULL if 0).
/// \p resultType is the MIXRShaped type of the attention result (required).
/// \p lseType is the MIXRShaped type of the optional log-sum-exp output; pass
///    a null type (via mlirTypeIsNull) to omit.
/// \p softmaxType is the optional element type for softmax computation; pass
///    a null type to omit.
/// \p preSoftmaxBody is a caller-created region for pre-softmax element-wise
///    ops. Pass an empty region (mlirRegionCreate()) for a no-op body.
///    Ownership of the region transfers to the created operation.
/// \p features is the bitwise-OR of MLIR_MIGRAPHX_ATTENTION_* flags (0 = none).
/// \p currentSeqLen is required when kvcache is set; pass null value to omit.
/// \p prefixOffset is required when prefix_offset is set; pass null to omit.
/// \p splitKV is the number of KV splits (0 or 1 = omit attribute).
/// \p slidingWindowSize is the window size (0 = omit attribute).
///
/// Contract violations are rejected with a stderr diagnostic and a null
/// MlirOperation return (check via mlirOperationIsNull). The same contract
/// is enforced in both debug and release builds. Specifically the function
/// returns a null op (and writes a "rocmlirMIGraphXAttentionCreate: ..."
/// line to stderr) if any of \p queries, \p keys, \p values is null, if
/// \p numPreSoftmaxInputs is negative or \p preSoftmaxElemWiseInputs is
/// NULL when the count is positive, if \p splitKV or \p slidingWindowSize
/// is negative, if \p resultType is null, or if \p preSoftmaxBody is null
/// (use mlirRegionCreate() for the no-body case rather than a
/// default-initialized struct). All other invariants (operand element
/// types, shape compatibility, feature/operand consistency, etc.) are
/// still left to the AttentionOp verifier.
MLIR_CAPI_EXPORTED MlirOperation rocmlirMIGraphXAttentionCreate(
    MlirLocation location, MlirValue queries, MlirValue keys, MlirValue values,
    intptr_t numPreSoftmaxInputs, const MlirValue *preSoftmaxElemWiseInputs,
    MlirType resultType, MlirType lseType, MlirType softmaxType,
    MlirRegion preSoftmaxBody, uint32_t features, MlirValue currentSeqLen,
    MlirValue prefixOffset, int32_t splitKV, int32_t slidingWindowSize);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_MIGRAPHX_H
