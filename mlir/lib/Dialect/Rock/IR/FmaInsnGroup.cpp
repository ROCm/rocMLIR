#include "mlir/Dialect/Rock/IR/FmaInsnGroup.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#define DEBUG_TYPE "rock-fma-insn-group"

using namespace mlir;
using namespace mlir::rock;

static Type getRetType(Type inputType) {
  Builder b(inputType.getContext());
  if (inputType.isInteger(8))
    return b.getI32Type();

  return b.getF32Type();
}

FailureOr<FmaInsn> FmaInsn::select(mlir::Type elementTypeA,
                                   mlir::Type elementTypeB, int64_t blockSize,
                                   int64_t waveSize, StringRef arch,
                                   int64_t kPack, int64_t mPerBlock,
                                   int64_t nPerBlock) {
  LLVM_DEBUG(llvm::dbgs() << "Invoke FMA group selection:\n"
                          << "elementTypeA: " << elementTypeA << "\n"
                          << "elementTypeB: " << elementTypeB << "\n"
                          << "blockSize: " << blockSize << "\n"
                          << "waveSize: " << waveSize << "\n"
                          << "arch: " << arch << "\n"
                          << "kPack: " << kPack << "\n"
                          << "nPerBlock: " << nPerBlock << "\n"
                          << "mPerBlock: " << mPerBlock << "\n");

  Type argTypeA =
      (kPack == 1) ? elementTypeA : VectorType::get({kPack}, elementTypeA);
  Type argTypeB =
      (kPack == 1) ? elementTypeB : VectorType::get({kPack}, elementTypeB);
  VectorType retType = VectorType::get({1}, getRetType(elementTypeA));
  VectorType blockReductionType =
      VectorType::get({blockSize / waveSize}, getRetType(elementTypeA));

  return FmaInsn{argTypeA, argTypeB,  retType,  blockReductionType,
                 1,        nPerBlock, mPerBlock};
}
