
#include "mlir/Dialect/Rock/IR/WmmaInsnGroup.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#define DEBUG_TYPE "rock-wmma-insn-group"

using namespace mlir;
using namespace mlir::rock;

static Type getRetType(Type inputType) {
  Builder b(inputType.getContext());
  if (inputType.isInteger(8))
    return b.getI32Type();

  return b.getF32Type();
}

bool WmmaInsn::isCoherentWithK(int64_t kpack, int64_t kPerBlock) {
  int64_t inputVectorLen = argTypeA.getNumElements();
  if (kPerBlock * kpack < inputVectorLen) {
    LLVM_DEBUG(llvm::dbgs()
               << "kPerBlock*kpack needs to be a multiple of inputLen "
               << inputVectorLen << "\n");
    return false;
  }
  return true;
}

FailureOr<WmmaInsn> WmmaInsn::select(mlir::Type elementTypeA,
                                     mlir::Type elementTypeB, int64_t waveSize,
                                     StringRef arch, int64_t mPerWave,
                                     int64_t nPerWave) {
  LLVM_DEBUG(llvm::dbgs() << "Invoke Wmma group selection:\n"
                          << "elementTypeA: " << elementTypeA << "\n"
                          << "elementTypeB: " << elementTypeB << "\n"
                          << "mPerWave: " << mPerWave << "\n"
                          << "nPerWave: " << nPerWave << "\n");

  if (elementTypeA != elementTypeB)
    return failure();

  if (waveSize != 32)
    return failure();

  // Length of the input vectors that need to be passed to the WMMA
  int64_t inputVectorLen = 8;
  // Length of the ouput vector that needs to be passed to the WMMA
  int64_t outputVectorLen = 8;
  // Number of rows/cols a given wmma instruction computes
  int64_t dPerAccel = 16;

  int64_t outStride = 2;
  if (arch.contains("gfx11")) {
    inputVectorLen = 16;
  }

  if (mPerWave % inputVectorLen != 0)
    return failure();
  if (nPerWave % inputVectorLen != 0)
    return failure();

  int64_t mRepeats = mPerWave / dPerAccel;
  int64_t nRepeats = nPerWave / dPerAccel;

  VectorType argTypeA = VectorType::get({inputVectorLen}, elementTypeA);
  VectorType argTypeB = VectorType::get({inputVectorLen}, elementTypeB);
  VectorType retType =
      VectorType::get({outputVectorLen}, getRetType(elementTypeA));

  StringRef insn;
  if (elementTypeA.isF16()) {
    insn = ROCDL::wmma_f32_16x16x16_f16::getOperationName();
  } else if (elementTypeA.isBF16()) {
    insn = ROCDL::wmma_f32_16x16x16_bf16::getOperationName();
  } else if (elementTypeA.isInteger(8)) {
    insn = ROCDL::wmma_i32_16x16x16_iu8::getOperationName();
  } else if (elementTypeA.isFloat8E4M3FN()) {
    // insn = ROCDL::wmma_
  } else if (elementTypeA.isFloat8E5M2()) {
    // insn = ROCDL::wmma_i32_16x16x16_iu8::getOperationName();
  } else {
    return failure();
  }

  return WmmaInsn{insn,     dPerAccel, outStride, mRepeats,
                  nRepeats, argTypeA,  argTypeB,  retType};
}
