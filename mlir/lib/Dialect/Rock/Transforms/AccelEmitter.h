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
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include <memory>

namespace mlir {
namespace rock {
namespace accel {

//
// Accelerator parameters used throughout the GEMM lowering pipeline
//
struct AccelEmitterParams {
  // `mPerAccel`/`nPerAccel` represent how many rows an accelerator intrinsic
  // will compute, while mRepeats and nRepeats represent how many times a given
  // wave needs to iterate to compute the `mPerWave` x `nPerWave` tile. E.g., if
  // `mPerWave=64`, `nPerWave=64`, `mPerAccel=16` and `nPerAccel=16` we have
  // what `mRepeats=64/16=4` and `nRepeats=64/16=4`
  int64_t mRepeats;
  int64_t nRepeats;
  int64_t mPerAccel;
  int64_t nPerAccel;

  // Each workitem reads a total of `kpackPerThread` vectors of length kpack.
  // Workitems need to read vectors of length `kBase` to compute the correct
  // output tile, but if `kPack>kBase` each thread will read multiple `kBase`
  // vectors.
  int64_t kBase;
  int64_t kpackPerThread;
  int64_t kBasePerThread;

  // This takes into account the fact that we might invoke accelerators back to
  // back and generate multiple sets of mRepeats*nRepeats vectors
  int64_t nResultVectors;

  Type argTypeA;            // Type of the arguments (might be scalar or vector)
  Type argTypeB;            // Type of the arguments (might be scalar or vector)
  VectorType accVectorType; // Accumulator vector type (always vector type)

  // Each workitem invoking an accelerator receives as a result a given number
  // of elements stored in VGPR
  int64_t numOutputVectorElements() const {
    return accVectorType.getNumElements() * nResultVectors * mRepeats *
           nRepeats;
  }
};

//
// Accelerator emitter strategy providing helpers to lower GEMM passes using an
// accelerator
//
struct AccelEmitter {

  /// Select the right accelerator based on the set of features and architecture
  static std::unique_ptr<AccelEmitter>
  select(GemmFeatures features, Type dataTypeA, Type dataTypeB, StringRef arch,
         RockAccelTuningParamAttrInterface tuningParams);

  AccelEmitter(StringRef arch, RockAccelTuningParamAttrInterface tuningParams,
               AccelEmitterParams accelEmitterParams);

  /// Emit the actual intrinsic in the threadwise operation
  virtual void emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA,
                                  Value argB, Value bufferC,
                                  ValueRange regCOffset) = 0;

  /// Return a wrapped view of the LDS buffer tailored for the accelerator
  /// load pattern. This is similar to wrapLDSBufferForStore, but while storing
  /// in LDS follows a similar pattern among accelerators, loading from LDS
  /// is dependent on the type of accelerator we are targeting
  virtual Value wrapLDSBufferForLoad(OpBuilder &b, Location loc, Value buffer,
                                     int64_t blockSize,
                                     int64_t dInCopyPerThread, StringRef dName,
                                     bool rotateDWithK, bool doSplitKAcrossThreadsFirst = false) const = 0;

  /// This functions creates the subtile views that is :
  /// 1) gridSubTileView :
  /// kloop x gblock x mblock x nblock x tid x iter --> ... --> [G, K, D]
  /// 2) blockSubTileView :
  /// tid x iter --> ... --> [KPerBlock, DPerBlock]
  /// 3) threadSubTileView :
  /// iter --> ... --> [KPerThread, DPerThread]
  /// for each operand tile to be used with gemm accelerators.
  virtual RegsAsMatrixSubTiles
  createAccelGemmOperandTransforms(OpBuilder &b, Location loc, int64_t kIters,
                                   ArrayRef<int64_t> bidGridLengths,
                                   int64_t blockSize, int64_t dInCopyPerThread,
                                   StringRef dName, bool isKContigousDim,
                                   bool rotateDWithK, bool doSplitKAcrossThreadsFirst = false) const = 0;

  /// Validate the accelerator structure
  virtual LogicalResult validateAcceleratorProperties() { return success(); };

  /// Compute the output transform map to be used to store the result of the
  /// matrix multiplication tile.
  virtual RegsAsMatrixSubTiles computeOutputTransforms(
      PatternRewriter &b, Location loc, int64_t mLen, int64_t nLen,
      int64_t blockSize, ArrayRef<int64_t> bidGridLengths, int64_t inMPerThread,
      int64_t inNPerThread, bool doSwapThreadIterSubDimsForM = false,
      bool doSwapThreadIterSubDimsForN = false) = 0;

  /// Convert from memref<?xvector<?xT>> to memref<?xD> where the source T
  /// is the accumulator type and D is the destination type
  void computeOutputConversion(PatternRewriter &b, Location loc, Value regDest,
                               Value convertedC, bool forceUnroll);

  // A view: A buffer is [0, K] so we can ignore `i`
  Value generateThreadwiseViewBufferA(PatternRewriter &b, Location loc,
                                      Value rawBufferA);
  // B view: B buffer is [0, K] so we can ignore `j`
  Value generateThreadwiseViewBufferB(PatternRewriter &b, Location loc,
                                      Value rawBufferB);
  // C view: C buffer is [mRepeats,nRepeats] and we need to write in
  // [i,j]. So we "freeze" the `i` and `j` indices and provide the value
  // of `i` and `j` as extra indices.
  Value generateThreadwiseViewBufferC(PatternRewriter &b, Location loc,
                                      Value rawBufferC);

  /// Return the accelerator parameters
  AccelEmitterParams getParams() const { return accelEmitterParams; }

  virtual ~AccelEmitter() {}

protected:
  RockAccelTuningParamAttrInterface tuningParams;
  AccelEmitterParams accelEmitterParams;
  int64_t waveSize;
};

// Accel emitter implementation for mfma
struct MfmaEmitter : public AccelEmitter {

  MfmaEmitter(MfmaInsnGroup mfmaGroup, StringRef arch,
              RockAccelTuningParamAttrInterface tuningParams);

  void emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA, Value argB,
                          Value bufferC, ValueRange regCOffset) override;

  virtual Value wrapLDSBufferForLoad(OpBuilder &b, Location loc, Value buffer,
                                     int64_t blockSize,
                                     int64_t dInCopyPerThread, StringRef dName,
                                     bool rotateDWithK, bool doSplitKAcrossThreadsFirst = false) const override;

  virtual RegsAsMatrixSubTiles
  createAccelGemmOperandTransforms(OpBuilder &b, Location loc, int64_t kIters,
                                   ArrayRef<int64_t> bidGridLengths,
                                   int64_t blockSize, int64_t dInCopyPerThread,
                                   StringRef dName, bool isKContigousDim,
                                   bool rotateDWithK, bool doSplitKAcrossThreadsFirst = false) const override;

  RegsAsMatrixSubTiles computeOutputTransforms(
      PatternRewriter &b, Location loc, int64_t mLen, int64_t nLen,
      int64_t blockSize, ArrayRef<int64_t> bidGridLengths, int64_t inMPerThread,
      int64_t inNPerThread, bool doSwapThreadIterSubDimsForM = false,
      bool doSwapThreadIterSubDimsForN = false) override;

  LogicalResult validateAcceleratorProperties() override;

  bool isKReduction() const;

private:
  /// Initialize the emitter parameters for mfma
  AccelEmitterParams
  initAccelEmitterParams(MfmaInsnGroup mfmaGroup,
                         RockAccelTuningParamAttrInterface tuningParams);

  MfmaInsnGroup mfmaGroup;
};

// Accel emitter implementation for wmma
struct WmmaEmitter : public AccelEmitter {

  WmmaEmitter(WmmaInsn wmmaInsn, StringRef arch,
              RockAccelTuningParamAttrInterface tuningParams);

  void emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA, Value argB,
                          Value bufferC, ValueRange regCOffset) override;

  virtual Value wrapLDSBufferForLoad(OpBuilder &b, Location loc, Value buffer,
                                     int64_t blockSize,
                                     int64_t dInCopyPerThread, StringRef dName,
                                     bool rotateDWithK, bool doSplitKAcrossThreadsFirst = false) const override;

  virtual RegsAsMatrixSubTiles
  createAccelGemmOperandTransforms(OpBuilder &b, Location loc, int64_t kIters,
                                   ArrayRef<int64_t> bidGridLengths,
                                   int64_t blockSize, int64_t dInCopyPerThread,
                                   StringRef dName, bool isKContigousDim,
                                   bool rotateDWithK, bool doSplitKAcrossThreadsFirst = false) const override;

  RegsAsMatrixSubTiles computeOutputTransforms(
      PatternRewriter &b, Location loc, int64_t mLen, int64_t nLen,
      int64_t blockSize, ArrayRef<int64_t> bidGridLengths, int64_t inMPerThread,
      int64_t inNPerThread, bool doSwapThreadIterSubDimsForM = false,
      bool doSwapThreadIterSubDimsForN = false) override;

private:
  /// Initialize the emitter parameters for wmma
  AccelEmitterParams
  initAccelEmitterParams(WmmaInsn wmmaInsn,
                         RockAccelTuningParamAttrInterface tuningParams);

  // Specifc wmma parameters
  WmmaInsn wmmaInsn;
};
} // namespace accel
} // namespace rock
} // namespace mlir

#endif //  MLIR_LIB_DIALECT_ROCK_TRANSFORMS_MLIR_ACCEL_EMITTER_H
