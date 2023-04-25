
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

#ifndef MLIR_LIB_DIALECT_ROCK_TRANSFORMS_MLIR_ACCEL_EMITTER_H
#define MLIR_LIB_DIALECT_ROCK_TRANSFORMS_MLIR_ACCEL_EMITTER_H

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
  // into registers. The pseudo-code of the lds-to-register loops is as follows
  // for(index_t m_i = 0; m_i < mRepeats; ++m_i)
  //   for(index_t k_i = 0; k_i < KPerThread; ++k_i)
  //       sourceOffset = computeLdsSourceOffset(d_i, k_i, dPerBlock,
  //       baseOffset)
  //       ...
  // In the above loop `d` can be either `m` or `n`.
  virtual Value computeLdsSourceOffset(OpBuilder &kBuilder, Value k_i,
                                       OpBuilder &dBuilder, Value d_i,
                                       OpBuilder &builder, Value dPerBlock,
                                       Location loc, Value baseOffset,
                                       Value laneId) = 0;

  // Compute the output transform map to be used to store the result of the
  // matrix multiplication tile
  virtual ArrayAttr computeOutputTransforms(PatternRewriter &b, Location loc,
                                            int64_t matrixM, int64_t matrixN,
                                            int64_t blockSize, int64_t gridSize,
                                            Value regC) = 0;

  // Convert from memref<?xvector<?xT>> to memref<?xD> where the source T
  // is the accumulator type and D is the destination type
  Value computeOutputConversion(PatternRewriter &b, Location loc,
                                int64_t matrixM, int64_t matrixN,
                                int64_t blockSize, int64_t gridSize,
                                Value regDest, Value convertedC,
                                bool forceUnroll);

  // Validate the accelerator structure
  void validateAcceleratorProperties();

  virtual ~AccelEmitter() {}

public:
  //
  // Tuning parameters
  //
  int64_t mPerBlock;
  int64_t nPerBlock;
  int64_t kPerBlock;
  int64_t mPerWave;
  int64_t nPerWave;
  int64_t kPack;
  int64_t waveSize;

  //
  // Accelerator parameters
  //
  // `mPerAccel`/`nPerAccel` represent how many rows an accelerator intrinsic
  // will compute, while mRepeats and nRepeats represent how many times a given
  // wave needs to iterate to compute the `mPerWave` x `nPerWave` tile. E.g., if
  // `mPerWave=64`, `nPerWave=64`, `mPerAccel=16` and `nPerAccel=16` we have
  // what `mRepeats=64/16=4` and `nRepeats=64/16=4`
  int64_t mRepeats;
  int64_t nRepeats;
  int64_t mPerAccel;
  int64_t nPerAccel;

  // A total of `kPerThread` k values are read by each worktiem.
  // Workitems need to read vectors of length `kBase` to compute the correct
  // output tile, but if `kPack>kBase` each thread will read multiple `kBase`
  // vectors.
  int64_t kBase;
  int64_t kPerThread;
  int64_t kBasePerThread;

  // This takes into account the fact that we might invoke accelerators back to
  // back and generate multiple sets of mRepeats*nRepeats vectors
  int64_t nResultVectors;

  // Each workitem invoking an accelerator receives as a result a given number
  // of elements stored in VGPR
  //
  // numOutputVectorElements =
  // nResultVectors*mRepeats*nRepeats*accVectorType.getNumElements()
  int64_t numOutputVectorElements;

  //
  // Accelerator data types
  //
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

  Value computeLdsSourceOffset(OpBuilder &kBuilder, Value k_i,
                               OpBuilder &dBuilder, Value d_i,
                               OpBuilder &builder, Value dPerBlock,
                               Location loc, Value baseOffset,
                               Value laneId) override;

  ArrayAttr computeOutputTransforms(PatternRewriter &b, Location loc,
                                    int64_t matrixM, int64_t matrixN,
                                    int64_t blockSize, int64_t gridSize,
                                    Value regC) override;

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

  Value computeLdsSourceOffset(OpBuilder &kBuilder, Value k_i,
                               OpBuilder &dBuilder, Value d_i,
                               OpBuilder &builder, Value dPerBlock,
                               Location loc, Value baseOffset,
                               Value laneId) override;

  ArrayAttr computeOutputTransforms(PatternRewriter &b, Location loc,
                                    int64_t matrixM, int64_t matrixN,
                                    int64_t blockSize, int64_t gridSize,
                                    Value regC) override;

private:
  // Specifc wmma parameters
  WmmaInsn wmmaInsn;
  VectorType reducedVectorType;
};
} // namespace accel
} // namespace rock
} // namespace mlir

#endif //  MLIR_LIB_DIALECT_ROCK_TRANSFORMS_MLIR_ACCEL_EMITTER_H
