
//===- AccelEmitter.cpp - MLIR helper to emit acceleration intrinsics
//---------------===//
//
// Copyright 2020 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This class tries to abstract away the code-generation details needed to
// generated calls to matrix multiply accelerator intrinsics (wmma, mfma).
//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ACCEL_EMITTER_INSN_GROUP_H
#define MLIR_ACCEL_EMITTER_INSN_GROUP_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/IR/WmmaInsnGroup.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include <memory>

namespace mlir {
namespace rock {
namespace accel {
struct AccelEmitter {

  // Select the right accelerator based on the set of features and architecture
  static std::unique_ptr<AccelEmitter>
  select(GemmFeatures features, Type dataTypeA, Type dataTypeB, StringRef arch,
         XdlopsGemmParamsAttr tuningParams);

  AccelEmitter(StringRef arch, XdlopsGemmParamsAttr tuningParams);
  /// Emit the actual intrinsic in the threadwise operation
  virtual void emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA,
                                  Value argB, Value bufferC,
                                  Value regCOffset) = 0;

  // Compute the correct lds source offset when loading data from shared memory
  // into registers
  virtual Value computeLdsSourceOffset(OpBuilder &kb, Value k_i, OpBuilder &mnb,
                                       Value mn_i, OpBuilder &b, Value MN,
                                       Location loc, Value sourceOffset,
                                       Value laneId) = 0;
  // Compute the output transform map to be used to store the result of the
  // matrix multiplication tile
  virtual ArrayAttr computeOutputTransforms(PatternRewriter &b, Location loc,
                                            int64_t M, int64_t N,
                                            int64_t blockSize, int64_t gridSize,
                                            Value regCAllocOp,
                                            Value convertedC) = 0;

  // Convert from memref<?xvector<?xT>> to memref<?xD> where the source T
  // is the accumulator type and D is the destination type
  virtual Value computeOutputConversion(PatternRewriter &b, Location loc,
                                        int64_t M, int64_t N, int64_t blockSize,
                                        int64_t gridSize, Value regCAllocOp,
                                        Value convertedC, bool forceUnroll) = 0;

  // Validate the accelerator structure
  void validateAcceleratorProperties();

  virtual ~AccelEmitter() {}

public:
  // Tuning parameters
  int64_t mPerBlock;
  int64_t nPerBlock;
  int64_t kPerBlock;
  int64_t mPerWave;
  int64_t nPerWave;
  int64_t kPack;

  // Accelerator parameters
  int64_t mRepeats;
  int64_t nRepeats;
  int64_t nResultVectors;
  int64_t mPerAccel;
  int64_t nPerAccel;
  int64_t kPerThread;
  int64_t kBase;
  int64_t kBasePerThread;
  int64_t inputBufferSize;
  int64_t waveSize;
  int64_t numOutputVectorElements;

  // Accelerator data types
  Type argTypeA;            // Type of the arguments (might be scalar or vector)
  Type argTypeB;            // Type of the arguments (might be scalar or vector)
  VectorType accVectorType; // Accumulator vector type (always vector type)
};

// Accel emitter implementation for mfma
struct MfmaEmitter : public AccelEmitter {

  MfmaEmitter(MfmaInsnGroup mfmaGroup, StringRef arch,
              XdlopsGemmParamsAttr tuningParams);

  void emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA, Value argB,
                          Value bufferC, Value regCOffset) override;

  Value computeLdsSourceOffset(OpBuilder &kb, Value k_i, OpBuilder &mnb,
                               Value mn_i, OpBuilder &b, Value MN, Location loc,
                               Value sourceOffset, Value laneId) override;

  ArrayAttr computeOutputTransforms(PatternRewriter &b, Location loc, int64_t M,
                                    int64_t N, int64_t blockSize,
                                    int64_t gridSize, Value regCAllocOp,
                                    Value convertedC) override;

  Value computeOutputConversion(PatternRewriter &b, Location loc, int64_t M,
                                int64_t N, int64_t blockSize, int64_t gridSize,
                                Value regCAllocOp, Value convertedC,
                                bool forceUnroll) override;

private:
  // Specifc mfma parameters
  bool isKReduction;
  int64_t inputSpansPerMfmaIn;
  int64_t inputSpanLen;
  MfmaInsnGroup mfmaGroup;
};

// Accel emitter implementation for wmma
struct WmmaEmitter : public AccelEmitter {

  WmmaEmitter(WmmaInsn wmmaInsn, StringRef arch,
              XdlopsGemmParamsAttr tuningParams);

  void emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA, Value argB,
                          Value bufferC, Value regCOffset) override;
  Value computeLdsSourceOffset(OpBuilder &kb, Value k_i, OpBuilder &mnb,
                               Value mn_i, OpBuilder &b, Value MN, Location loc,
                               Value sourceOffset, Value laneId) override;
  ArrayAttr computeOutputTransforms(PatternRewriter &b, Location loc, int64_t M,
                                    int64_t N, int64_t blockSize,
                                    int64_t gridSize, Value regCAllocOp,
                                    Value convertedC) override;

  Value computeOutputConversion(PatternRewriter &b, Location loc, int64_t M,
                                int64_t N, int64_t blockSize, int64_t gridSize,
                                Value regCAllocOp, Value convertedC,
                                bool forceUnroll) override;

private:
  // Specifc wmma parameters
  WmmaInsn wmmaInsn;
  VectorType reducedVectorType;
};
} // namespace accel
} // namespace rock
} // namespace mlir

#endif // MLIR_ACCEL_EMITTER_INSN_GROUP_H
