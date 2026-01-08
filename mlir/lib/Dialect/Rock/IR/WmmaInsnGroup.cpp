
#include "mlir/Dialect/Rock/IR/WmmaInsnGroup.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#define DEBUG_TYPE "rock-wmma-insn-group"

using namespace mlir;
using namespace mlir::rock;

static Type getRetType(Type inputType, WmmaTypeId typeId) {
  Builder b(inputType.getContext());

  // Integer types always output I32
  if (typeId == WmmaTypeId::I8_To_I32_TyId ||
      typeId == WmmaTypeId::I4_To_I32_TyId)
    return b.getI32Type();

  // F16 output variants
  if (typeId == WmmaTypeId::F16_To_F16_TyId ||
      typeId == WmmaTypeId::Fp8Fp8_To_F16_TyId ||
      typeId == WmmaTypeId::Fp8Bf8_To_F16_TyId ||
      typeId == WmmaTypeId::Bf8Fp8_To_F16_TyId ||
      typeId == WmmaTypeId::Bf8Bf8_To_F16_TyId)
    return b.getF16Type();

  // BF16 output variants
  if (typeId == WmmaTypeId::Bf16_To_Bf16_TyId ||
      typeId == WmmaTypeId::Bf16_To_Bf16F32_TyId)
    return b.getBF16Type();

  // Small float scaled WMMA always outputs F32
  if (typeId == WmmaTypeId::SmallFloat_To_F32_TyId)
    return b.getF32Type();

  // Default: F32 output
  return b.getF32Type();
}

// Convert input types to type ID. This handles both same-type and mixed-type
// inputs and always uses F32/I32 outputs.
// If forScaledOp is true, FP8/BF8 types will use the scaled WMMA instruction.
static WmmaTypeId convertTypesToId(Type dataTypeA, Type dataTypeB,
                                   bool forScaledOp = false) {
  // Same type inputs
  if (dataTypeA == dataTypeB) {
    if (dataTypeA.isF32())
      return WmmaTypeId::F32_To_F32_TyId;
    if (dataTypeA.isF16())
      return WmmaTypeId::F16_To_F32_TyId;
    if (dataTypeA.isBF16())
      return WmmaTypeId::Bf16_To_F32_TyId;
    if (dataTypeA.isInteger(8))
      return WmmaTypeId::I8_To_I32_TyId;
    if (dataTypeA.isInteger(4))
      return WmmaTypeId::I4_To_I32_TyId;
  }

  // Check for small float types
  bool aIsFp8 = isa<Float8E4M3FNType>(dataTypeA);
  bool aIsBf8 = isa<Float8E5M2Type>(dataTypeA);
  bool bIsFp8 = isa<Float8E4M3FNType>(dataTypeB);
  bool bIsBf8 = isa<Float8E5M2Type>(dataTypeB);
  bool aIsFp6 = isa<Float6E2M3FNType, Float6E3M2FNType>(dataTypeA);
  bool bIsFp6 = isa<Float6E2M3FNType, Float6E3M2FNType>(dataTypeB);
  bool aIsFp4 = isa<Float4E2M1FNType>(dataTypeA);
  bool bIsFp4 = isa<Float4E2M1FNType>(dataTypeB);

  // For scaled operations with FP8/BF8 types, use the scaled WMMA instruction
  if (forScaledOp && ((aIsFp8 || aIsBf8) && (bIsFp8 || bIsBf8)))
    return WmmaTypeId::SmallFloat_To_F32_TyId;

  // Pure FP8/BF8 combinations (prefer regular intrinsics when available)
  if ((aIsFp8 || aIsBf8) && (bIsFp8 || bIsBf8)) {
    if (aIsFp8 && bIsFp8)
      return WmmaTypeId::Fp8Fp8_To_F32_TyId;
    if (aIsFp8 && bIsBf8)
      return WmmaTypeId::Fp8Bf8_To_F32_TyId;
    if (aIsBf8 && bIsFp8)
      return WmmaTypeId::Bf8Fp8_To_F32_TyId;
    if (aIsBf8 && bIsBf8)
      return WmmaTypeId::Bf8Bf8_To_F32_TyId;
  }

  // Any combination involving FP4 or FP6 (with or without FP8/BF8)
  if (aIsFp4 || bIsFp4 || aIsFp6 || bIsFp6)
    return WmmaTypeId::SmallFloat_To_F32_TyId;

  llvm_unreachable("Unsupported WMMA input type combination");
}

// WMMA instructions available on gfx11
static const llvm::DenseMap<WmmaInsnKey, WmmaInsnInfo> &getWmmaInsnMapGfx11() {
  static llvm::DenseMap<WmmaInsnKey, WmmaInsnInfo> insnMap{
      {{WmmaTypeId::F16_To_F32_TyId, 16},
       {ROCDL::wmma_f32_16x16x16_f16::getOperationName(), 16, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::Bf16_To_F32_TyId, 16},
       {ROCDL::wmma_f32_16x16x16_bf16::getOperationName(), 16, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::I8_To_I32_TyId, 16},
       {ROCDL::wmma_i32_16x16x16_iu8::getOperationName(), 16, 8, 16, 16,
        /*isScaled=*/false}},
  };
  return insnMap;
}

// WMMA instructions available on gfx12
static const llvm::DenseMap<WmmaInsnKey, WmmaInsnInfo> &getWmmaInsnMapGfx12() {
  static llvm::DenseMap<WmmaInsnKey, WmmaInsnInfo> insnMap{
      {{WmmaTypeId::F16_To_F32_TyId, 16},
       {ROCDL::wmma_f32_16x16x16_f16::getOperationName(), 8, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::Bf16_To_F32_TyId, 16},
       {ROCDL::wmma_f32_16x16x16_bf16::getOperationName(), 8, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::I8_To_I32_TyId, 16},
       {ROCDL::wmma_i32_16x16x16_iu8::getOperationName(), 8, 8, 16, 16,
        /*isScaled=*/false}},

      // FP8/BF8
      {{WmmaTypeId::Fp8Fp8_To_F32_TyId, 16},
       {ROCDL::wmma_f32_16x16x16_fp8_fp8::getOperationName(), 8, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::Bf8Bf8_To_F32_TyId, 16},
       {ROCDL::wmma_f32_16x16x16_bf8_bf8::getOperationName(), 8, 8, 16, 16,
        /*isScaled=*/false}},
  };
  return insnMap;
}

// WMMA instructions available on gfx1250
static const llvm::DenseMap<WmmaInsnKey, WmmaInsnInfo> &
getWmmaInsnMapGfx1250() {
  static llvm::DenseMap<WmmaInsnKey, WmmaInsnInfo> insnMap{
      // F32
      {{WmmaTypeId::F32_To_F32_TyId, 4},
       {ROCDL::wmma_f32_16x16x4_f32::getOperationName(), 2, 8, 16, 16,
        /*isScaled=*/false}},

      // F16/BF16
      {{WmmaTypeId::F16_To_F32_TyId, 32},
       {ROCDL::wmma_f32_16x16x32_f16::getOperationName(), 16, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::Bf16_To_F32_TyId, 32},
       {ROCDL::wmma_f32_16x16x32_bf16::getOperationName(), 16, 8, 16, 16,
        /*isScaled=*/false}},

      // FP8/BF8 (k=64)
      {{WmmaTypeId::Fp8Fp8_To_F32_TyId, 64},
       {ROCDL::wmma_f32_16x16x64_fp8_fp8::getOperationName(), 32, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::Fp8Bf8_To_F32_TyId, 64},
       {ROCDL::wmma_f32_16x16x64_fp8_bf8::getOperationName(), 32, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::Bf8Fp8_To_F32_TyId, 64},
       {ROCDL::wmma_f32_16x16x64_bf8_fp8::getOperationName(), 32, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::Bf8Bf8_To_F32_TyId, 64},
       {ROCDL::wmma_f32_16x16x64_bf8_bf8::getOperationName(), 32, 8, 16, 16,
        /*isScaled=*/false}},

      // FP8/BF8 (k=128)
      {{WmmaTypeId::Fp8Fp8_To_F32_TyId, 128},
       {ROCDL::wmma_f32_16x16x128_fp8_fp8::getOperationName(), 64, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::Fp8Bf8_To_F32_TyId, 128},
       {ROCDL::wmma_f32_16x16x128_fp8_bf8::getOperationName(), 64, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::Bf8Fp8_To_F32_TyId, 128},
       {ROCDL::wmma_f32_16x16x128_bf8_fp8::getOperationName(), 64, 8, 16, 16,
        /*isScaled=*/false}},
      {{WmmaTypeId::Bf8Bf8_To_F32_TyId, 128},
       {ROCDL::wmma_f32_16x16x128_bf8_bf8::getOperationName(), 64, 8, 16, 16,
        /*isScaled=*/false}},

      // Small float scaled WMMA (k=128)
      // This covers FP4, and mixed combinations (FP8+FP4, etc.)
      {{WmmaTypeId::SmallFloat_To_F32_TyId, 128},
       {ROCDL::wmma_scale_f32_16x16x128_f8f6f4::getOperationName(), 64, 8, 16,
        16, /*isScaled=*/true}},

      // I8
      {{WmmaTypeId::I8_To_I32_TyId, 64},
       {ROCDL::wmma_i32_16x16x64_iu8::getOperationName(), 32, 8, 16, 16,
        /*isScaled=*/false}},
  };
  return insnMap;
}

// Helper function to validate K coherence
// Returns true if kPerBlock * kPack is sufficient for the given inputVectorLen
static bool isKCoherent(int64_t inputVectorLen, int64_t kPack,
                        int64_t kPackPerBlock) {
  if (((kPackPerBlock * kPack) % inputVectorLen) != 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "kPerBlock*kpack needs to be a multiple of inputLen: "
               << kPackPerBlock << " * " << kPack << " = "
               << (kPackPerBlock * kPack) << " % " << inputVectorLen << "\n");
    return false;
  }
  return true;
}

bool WmmaInsn::isCoherentWithK(int64_t kpack, int64_t kPerBlock) {
  int64_t inputVectorLen = argTypeA.getNumElements();
  return isKCoherent(inputVectorLen, kpack, kPerBlock);
}

FailureOr<WmmaInsn> WmmaInsn::select(mlir::Type elementTypeA,
                                     mlir::Type elementTypeB, int64_t waveSize,
                                     StringRef arch, int64_t mPerWave,
                                     int64_t nPerWave, int64_t kPack,
                                     int64_t kPackPerBlock, bool forScaledOp) {
  LLVM_DEBUG(llvm::dbgs() << "Invoke Wmma instruction selection:\n"
                          << "elementTypeA: " << elementTypeA << "\n"
                          << "elementTypeB: " << elementTypeB << "\n"
                          << "arch: " << arch << "\n"
                          << "mPerWave: " << mPerWave << "\n"
                          << "nPerWave: " << nPerWave << "\n"
                          << "kPack: " << kPack << "\n"
                          << "kPackPerBlock: " << kPackPerBlock << "\n"
                          << "forScaledOp: " << forScaledOp << "\n");

  // WMMA only supports wave32
  if (waveSize != 32)
    return failure();

  // Architecture detection
  bool isGfx11 = arch.contains("gfx11");
  bool isGfx1250 = arch.contains("gfx1250");

  // Convert element types to ID for map lookup. Handles both same-type and
  // mixed-type inputs. For scaled operations, FP8 types use the scaled WMMA.
  WmmaTypeId typeId = convertTypesToId(elementTypeA, elementTypeB, forScaledOp);

  // Select instruction based on architecture priority: gfx1250 > gfx12 > gfx11
  const WmmaInsnInfo *insnInfo = nullptr;
  int64_t selectedKDim = 0; // Track the selected K dimension from the map key

  if (isGfx1250) {
    auto &gfx1250Map = getWmmaInsnMapGfx1250();

    // FP8/BF8 types have multiple K options and we need to select the best one.
    // We will always try to select the largest K value that is coherent with
    // the KPack and KPackPerBlock, and then fall back to the smaller values
    // if need be.
    if (typeId >= WmmaTypeId::Fp8Fp8_To_F32_TyId &&
        typeId <= WmmaTypeId::Bf8Bf8_To_F32_TyId) {
      for (int64_t k : {128, 64}) {
        auto it = gfx1250Map.find({typeId, k});
        if (it != gfx1250Map.end()) {
          const WmmaInsnInfo *info = &it->second;
          if (isKCoherent(info->inputVectorLen, kPack, kPackPerBlock)) {
            insnInfo = info;
            selectedKDim = k; // Extract K from the key
            LLVM_DEBUG(llvm::dbgs() << "Selected gfx1250 instruction: "
                                    << insnInfo->insn << "\n");
            break;
          }
        }
      }
    } else {
      // All other types have only one K value, so we can just directly look
      // that up
      int64_t k;
      if (typeId == WmmaTypeId::F32_To_F32_TyId) {
        k = 4;
      } else if (typeId == WmmaTypeId::I8_To_I32_TyId) {
        k = 64;
      } else if (typeId == WmmaTypeId::SmallFloat_To_F32_TyId) {
        k = 128;
      } else {
        // F16/BF16 types
        k = 32;
      }

      auto it = gfx1250Map.find({typeId, k});
      if (it != gfx1250Map.end()) {
        insnInfo = &it->second;
        selectedKDim = k;
        LLVM_DEBUG(llvm::dbgs() << "Selected gfx1250 instruction: "
                                << insnInfo->insn << "\n");
      }
    }
  }

  // Use gfx12 only if we don't have a selected instruction and not gfx11
  if (!insnInfo && !isGfx11) {
    auto &gfx12Map = getWmmaInsnMapGfx12();
    auto it = gfx12Map.find({typeId, 16});
    if (it != gfx12Map.end()) {
      insnInfo = &it->second;
      selectedKDim = 16;
      LLVM_DEBUG(llvm::dbgs()
                 << "Selected gfx12 instruction: " << insnInfo->insn << "\n");
    }
  }

  // Fall back to gfx11 if we are using gfx12, or if we explicitly ask for gfx11
  if (!insnInfo && (!isGfx1250 || isGfx11)) {
    auto &gfx11Map = getWmmaInsnMapGfx11();
    auto it = gfx11Map.find({typeId, 16});
    if (it != gfx11Map.end()) {
      insnInfo = &it->second;
      selectedKDim = 16;
      LLVM_DEBUG(llvm::dbgs()
                 << "Selected gfx11 instruction: " << insnInfo->insn << "\n");
    }
  }

  if (!insnInfo)
    return failure();

  // Extract instruction info
  int64_t inputVectorLen = insnInfo->inputVectorLen;
  int64_t outputVectorLen = insnInfo->outputVectorLen;
  int64_t kDim = selectedKDim;
  StringRef insn = insnInfo->insn;
  int64_t mPerAccel = insnInfo->mPerAccel;
  int64_t nPerAccel = insnInfo->nPerAccel;
  bool isScaled = insnInfo->isScaled;

  // Architecture-specific outStride
  // gfx11: full-wave K cooperation -> outStride=2
  // gfx12/gfx1250: half-wave K cooperation -> outStride=outputVectorLen
  int64_t outStride = isGfx11 ? 2 : outputVectorLen;

  // Validate dimensions
  if (mPerWave % inputVectorLen != 0 || mPerWave % mPerAccel != 0)
    return failure();
  if (nPerWave % inputVectorLen != 0 || nPerWave % nPerAccel != 0)
    return failure();

  int64_t mRepeats = mPerWave / mPerAccel;
  int64_t nRepeats = nPerWave / nPerAccel;

  VectorType argTypeA = VectorType::get({inputVectorLen}, elementTypeA);
  VectorType argTypeB = VectorType::get({inputVectorLen}, elementTypeB);
  VectorType retType =
      VectorType::get({outputVectorLen}, getRetType(elementTypeA, typeId));

  return WmmaInsn{insn,     mPerAccel, nPerAccel, kDim,    outStride, mRepeats,
                  nRepeats, argTypeA,  argTypeB,  retType, isScaled};
}
