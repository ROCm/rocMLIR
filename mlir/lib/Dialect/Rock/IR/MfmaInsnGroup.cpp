
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#define DEBUG_TYPE "rock-mfma-insn-group"

using namespace mlir;
using namespace mlir::rock;

// The static initialization will follow the defined ordering
// of the below lambdas
auto getMfmaInsnInfoMap = []() -> const llvm::StringMap<MfmaInsnInfo> & {
  static llvm::StringMap<MfmaInsnInfo> insnInfo{
      // fp32
      {ROCDL::mfma_f32_32x32x1f32::getOperationName(),
       {MfmaTypeId::Fp32TyId, 32, 1, 2}},
      {ROCDL::mfma_f32_32x32x2f32::getOperationName(),
       {MfmaTypeId::Fp32TyId, 32, 2, 1}},
      {ROCDL::mfma_f32_16x16x1f32::getOperationName(),
       {MfmaTypeId::Fp32TyId, 16, 1, 4}},
      {ROCDL::mfma_f32_16x16x4f32::getOperationName(),
       {MfmaTypeId::Fp32TyId, 16, 4, 1}},
      {ROCDL::mfma_f32_4x4x1f32::getOperationName(),
       {MfmaTypeId::Fp32TyId, 4, 1, 16}},

      // fp16
      {ROCDL::mfma_f32_32x32x4f16::getOperationName(),
       {MfmaTypeId::Fp16TyId, 32, 4, 2}},
      {ROCDL::mfma_f32_32x32x8f16::getOperationName(),
       {MfmaTypeId::Fp16TyId, 32, 8, 1}},
      {ROCDL::mfma_f32_16x16x4f16::getOperationName(),
       {MfmaTypeId::Fp16TyId, 16, 4, 4}},
      {ROCDL::mfma_f32_16x16x16f16::getOperationName(),
       {MfmaTypeId::Fp16TyId, 16, 16, 1}},
      {ROCDL::mfma_f32_4x4x4f16::getOperationName(),
       {MfmaTypeId::Fp16TyId, 4, 4, 16}},

      // bf16
      {ROCDL::mfma_f32_32x32x2bf16::getOperationName(),
       {MfmaTypeId::Bf16TyId, 32, 2, 2}},
      {ROCDL::mfma_f32_32x32x4bf16::getOperationName(),
       {MfmaTypeId::Bf16TyId, 32, 4, 1}},
      {ROCDL::mfma_f32_16x16x2bf16::getOperationName(),
       {MfmaTypeId::Bf16TyId, 16, 2, 4}},
      {ROCDL::mfma_f32_16x16x8bf16::getOperationName(),
       {MfmaTypeId::Bf16TyId, 16, 8, 1}},
      {ROCDL::mfma_f32_4x4x2bf16::getOperationName(),
       {MfmaTypeId::Bf16TyId, 4, 2, 16}},

      // bf16 (new)
      {ROCDL::mfma_f32_32x32x4bf16_1k::getOperationName(),
       {MfmaTypeId::Bf16TyId, 32, 4, 2}},
      {ROCDL::mfma_f32_16x16x4bf16_1k::getOperationName(),
       {MfmaTypeId::Bf16TyId, 16, 4, 4}},
      {ROCDL::mfma_f32_4x4x4bf16_1k::getOperationName(),
       {MfmaTypeId::Bf16TyId, 4, 4, 16}},
      {ROCDL::mfma_f32_32x32x8bf16_1k::getOperationName(),
       {MfmaTypeId::Bf16TyId, 32, 8, 1}},
      {ROCDL::mfma_f32_16x16x16bf16_1k::getOperationName(),
       {MfmaTypeId::Bf16TyId, 16, 16, 1}},

      // i8
      {ROCDL::mfma_i32_32x32x8i8::getOperationName(),
       {MfmaTypeId::I8TyId, 32, 8, 1}},
      {ROCDL::mfma_i32_16x16x16i8::getOperationName(),
       {MfmaTypeId::I8TyId, 16, 16, 1}},

      // i8 (new)
      {ROCDL::mfma_i32_32x32x16_i8::getOperationName(),
       {MfmaTypeId::I8TyId, 32, 16, 1}},
      {ROCDL::mfma_i32_16x16x32_i8::getOperationName(),
       {MfmaTypeId::I8TyId, 16, 32, 1}}};
  return insnInfo;
};

static MfmaInsnAttr deriveAttr(MfmaInsnInfo info) {
  int64_t mfmaNonKDim = info.mfmaNonKDim;
  int64_t k = info.k;
  int64_t blocksMfma = info.blocksMfma;

  constexpr int64_t waveSize = 64;
  // Derived properties of the individual MFMA. These are computed here
  // and used in places throughout the code and may not all be needed.
  int64_t kPerMfmaInput =
      math_util::integer_divide_ceil(waveSize, mfmaNonKDim * blocksMfma);
  // k_base is the number of times you need to step in the k dimension on each
  // lane in a wave.
  int64_t k_base = k / kPerMfmaInput;

  // Number of logical values each thread needs to pass in to the MFMA in
  // order for the correct number of input values to be passed to the MFMA.
  int64_t nInputsToMfma = (mfmaNonKDim * blocksMfma * k) / waveSize;
  int64_t nOutputsOfMfma = (mfmaNonKDim * mfmaNonKDim * blocksMfma) / waveSize;

  constexpr int64_t rowGroupSize = 4;
  // The number of rows in each MFMA output item (usually a VGPR, except in
  // the case of double-precision). Note, a "row" is a complete output row of
  // one of blocks produced by the MFMA.

  // For most MFMAs, this is bounded by the number of retitions of n_mfma
  // that you can fit into a wave (ex. a 32x32x2 mfma can fit two rows per
  // result). However, in the case of 4x4xk (16 blocks) MFMAs, counting that
  // number would produce the inaccurate result 64 / 4 = 16 rows per output,
  // even though there is only one row available to place in each output
  // (remembering that the rows are first tiled into groups of group_size
  // outputs). Therefore, we need to impose the bound m_mfma / group_size on
  // the number of rows per output.
  int64_t rowsPerMfmaOutput = std::min(waveSize / /*n_mfma=*/mfmaNonKDim,
                                       /*m_mfma=*/mfmaNonKDim / rowGroupSize);
  // The number of blocks in each MFMA output. If rowsPerMfmaOutput followed
  // the typical case and was computed using waveSize / n_mfma, this will
  // be 1. However, in the 4x4 case, where we do have 16 blocks packed into
  // each output blocksPerOutput will be > 1 (namely 16).
  int64_t blocksPerMfmaOutput = math_util::integer_divide_ceil(
      waveSize, rowsPerMfmaOutput * /*n_mfma=*/mfmaNonKDim);
  // The number of register groups (of four rows) per block of output
  // Note that the inclusion of blocksPerOutput forces this value to be 1 in
  // the 4x4 case, as it should be.
  int64_t rowGroupsPerBlock = math_util::integer_divide_ceil(
      /*m_mfma=*/mfmaNonKDim,
      rowGroupSize * rowsPerMfmaOutput * blocksPerMfmaOutput);
  // Number of output blocks that can be accessed by going through the
  // registers on any given lane.
  int64_t blocksInOutRegs = blocksMfma / blocksPerMfmaOutput;

  // Because the 4x4xk instructions are the only ones with blocksPerOutput > 1
  // and because they're only used for broadcast operations, we have
  // historically represented them as 4x64 operations that have one large
  // "block" instead of 16 tiny ones. So, the length of an input span
  // (row for A, column for B) is usually equal to mfmaNonKDim, but is
  // more generally equal to mfmaNonKDim * blocksPerOutput in order to
  // enable the math throughout our code to note break.
  int64_t inputSpanLen = mfmaNonKDim * blocksPerMfmaOutput;
  int64_t inputSpansPerMfmaIn = waveSize / inputSpanLen;

  return {
      mfmaNonKDim,
      k,
      blocksMfma,
      nInputsToMfma,
      k_base,
      inputSpanLen,
      inputSpansPerMfmaIn,
      nOutputsOfMfma,
      rowGroupSize,
      rowsPerMfmaOutput,
      blocksPerMfmaOutput,
      rowGroupsPerBlock,
      blocksInOutRegs,
  };
}

auto getMfmaInsnAttrMap = []() -> const llvm::StringMap<MfmaInsnAttr> & {
  static llvm::StringMap<MfmaInsnAttr> insnDb;
  static std::once_flag once;
  std::call_once(once, [&]() {
    for (const auto &insn : getMfmaInsnInfoMap()) {
      StringRef key = insn.getKey();
      MfmaInsnInfo info = insn.getValue();
      insnDb.insert(std::make_pair(key, deriveAttr(info)));
    };
  });
  return insnDb;
};

using MfmaInsnGroupMap =
    llvm::DenseMap<MfmaInsnGroupSelectKey, MfmaInsnGroupAttr,
                   MfmaInsnGroupSelectKeyInfo>;
auto getMfmaInsnGroupAttrMapAllArch = []() -> const MfmaInsnGroupMap & {
  using amdgpu::MFMAPermB;
  static MfmaInsnGroupMap
      // f32
      groupAttrMap{{{MfmaTypeId::Fp32TyId, 64, 64},
                    {ROCDL::mfma_f32_32x32x2f32::getOperationName()}},
                   {{MfmaTypeId::Fp32TyId, 64, 32},
                    {ROCDL::mfma_f32_32x32x2f32::getOperationName()}},
                   {{MfmaTypeId::Fp32TyId, 32, 64},
                    {ROCDL::mfma_f32_32x32x2f32::getOperationName()}},
                   {{MfmaTypeId::Fp32TyId, 64, 16},
                    {ROCDL::mfma_f32_16x16x4f32::getOperationName()}},
                   {{MfmaTypeId::Fp32TyId, 16, 64},
                    {ROCDL::mfma_f32_16x16x4f32::getOperationName()}},
                   {{MfmaTypeId::Fp32TyId, 8, 64},
                    {ROCDL::mfma_f32_4x4x1f32::getOperationName(),
                     {{4, 0, MFMAPermB::none}, {4, 1, MFMAPermB::none}}}},
                   {{MfmaTypeId::Fp32TyId, 4, 64},
                    {ROCDL::mfma_f32_4x4x1f32::getOperationName(),
                     {{4, 0, MFMAPermB::none}}}},
                   {{MfmaTypeId::Fp32TyId, 32, 32},
                    {ROCDL::mfma_f32_32x32x2f32::getOperationName()}},
                   {{MfmaTypeId::Fp32TyId, 16, 16},
                    {ROCDL::mfma_f32_16x16x4f32::getOperationName()}},
                   // f16
                   {{MfmaTypeId::Fp16TyId, 64, 64},
                    {ROCDL::mfma_f32_32x32x8f16::getOperationName()}},
                   {{MfmaTypeId::Fp16TyId, 64, 32},
                    {ROCDL::mfma_f32_32x32x8f16::getOperationName()}},
                   {{MfmaTypeId::Fp16TyId, 64, 16},
                    {ROCDL::mfma_f32_16x16x16f16::getOperationName()}},
                   {{MfmaTypeId::Fp16TyId, 16, 64},
                    {ROCDL::mfma_f32_16x16x16f16::getOperationName()}},
                   {{MfmaTypeId::Fp16TyId, 8, 64},
                    {ROCDL::mfma_f32_4x4x4f16::getOperationName(),
                     {{4, 0, MFMAPermB::none}, {4, 1, MFMAPermB::none}}}},
                   {{MfmaTypeId::Fp16TyId, 4, 64},
                    {ROCDL::mfma_f32_4x4x4f16::getOperationName(),
                     {{4, 0, MFMAPermB::none}}}},
                   {{MfmaTypeId::Fp16TyId, 32, 32},
                    {ROCDL::mfma_f32_32x32x8f16::getOperationName()}},
                   {{MfmaTypeId::Fp16TyId, 32, 64},
                    {ROCDL::mfma_f32_32x32x8f16::getOperationName()}},
                   {{MfmaTypeId::Fp16TyId, 16, 16},
                    {ROCDL::mfma_f32_16x16x16f16::getOperationName()}}};
  return groupAttrMap;
};

auto getMfmaInsnGroupAttrMapGfx908Bf16 = []() -> const MfmaInsnGroupMap & {
  using amdgpu::MFMAPermB;
  static MfmaInsnGroupMap
      // bf16
      groupAttrMap{{{MfmaTypeId::Bf16TyId, 64, 64},
                    {ROCDL::mfma_f32_32x32x4bf16::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 64, 32},
                    {ROCDL::mfma_f32_32x32x4bf16::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 32, 64},
                    {ROCDL::mfma_f32_32x32x4bf16::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 64, 16},
                    {ROCDL::mfma_f32_16x16x8bf16::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 16, 64},
                    {ROCDL::mfma_f32_16x16x8bf16::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 8, 64},
                    {ROCDL::mfma_f32_4x4x2bf16::getOperationName(),
                     {{4, 0, MFMAPermB::none}, {4, 1, MFMAPermB::none}}}},
                   {{MfmaTypeId::Bf16TyId, 4, 64},
                    {ROCDL::mfma_f32_4x4x2bf16::getOperationName(),
                     {{4, 0, MFMAPermB::none}}}},
                   {{MfmaTypeId::Bf16TyId, 32, 32},
                    {ROCDL::mfma_f32_32x32x4bf16::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 16, 16},
                    {ROCDL::mfma_f32_16x16x8bf16::getOperationName()}}};
  return groupAttrMap;
};

auto getMfmaInsnGroupAttrMapGfx90aPlusBf16 = []() {
  using amdgpu::MFMAPermB;
  static llvm::DenseMap<MfmaInsnGroupSelectKey, MfmaInsnGroupAttr,
                        MfmaInsnGroupSelectKeyInfo>
      // bf16
      groupAttrMap{{{MfmaTypeId::Bf16TyId, 64, 64},
                    {ROCDL::mfma_f32_32x32x8bf16_1k::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 64, 32},
                    {ROCDL::mfma_f32_32x32x8bf16_1k::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 32, 64},
                    {ROCDL::mfma_f32_32x32x8bf16_1k::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 64, 16},
                    {ROCDL::mfma_f32_16x16x16bf16_1k::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 16, 64},
                    {ROCDL::mfma_f32_16x16x16bf16_1k::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 8, 64},
                    {ROCDL::mfma_f32_4x4x4bf16_1k::getOperationName(),
                     {{4, 0, MFMAPermB::none}, {4, 1, MFMAPermB::none}}}},
                   {{MfmaTypeId::Bf16TyId, 4, 64},
                    {ROCDL::mfma_f32_4x4x4bf16_1k::getOperationName(),
                     {{4, 0, MFMAPermB::none}}}},
                   {{MfmaTypeId::Bf16TyId, 32, 32},
                    {ROCDL::mfma_f32_32x32x8bf16_1k::getOperationName()}},
                   {{MfmaTypeId::Bf16TyId, 16, 16},
                    {ROCDL::mfma_f32_16x16x16bf16_1k::getOperationName()}}};
  return groupAttrMap;
};

auto getMfmaInsnGroupAttrMapPreGfx940Int8 = []() {
  using amdgpu::MFMAPermB;
  static llvm::DenseMap<MfmaInsnGroupSelectKey, MfmaInsnGroupAttr,
                        MfmaInsnGroupSelectKeyInfo>
      // Int8
      groupAttrMap{{{MfmaTypeId::I8TyId, 64, 64},
                    {ROCDL::mfma_i32_32x32x8i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 64, 32},
                    {ROCDL::mfma_i32_32x32x8i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 32, 64},
                    {ROCDL::mfma_i32_32x32x8i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 32, 32},
                    {ROCDL::mfma_i32_32x32x8i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 64, 16},
                    {ROCDL::mfma_i32_16x16x16i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 16, 64},
                    {ROCDL::mfma_i32_16x16x16i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 16, 16},
                    {ROCDL::mfma_i32_16x16x16i8::getOperationName()}}};
  ;
  return groupAttrMap;
};

// New I8 and all Float8
auto getMfmaInsnGroupAttrMapGfx940Plus = []() {
  using amdgpu::MFMAPermB;
  static MfmaInsnGroupMap
      // Int8
      groupAttrMap{{{MfmaTypeId::I8TyId, 64, 64},
                    {ROCDL::mfma_i32_32x32x16_i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 64, 32},
                    {ROCDL::mfma_i32_32x32x16_i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 32, 64},
                    {ROCDL::mfma_i32_32x32x16_i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 32, 32},
                    {ROCDL::mfma_i32_32x32x16_i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 64, 16},
                    {ROCDL::mfma_i32_16x16x32_i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 16, 64},
                    {ROCDL::mfma_i32_16x16x32_i8::getOperationName()}},
                   {{MfmaTypeId::I8TyId, 16, 16},
                    {ROCDL::mfma_i32_16x16x32_i8::getOperationName()}}};
  ;
  return groupAttrMap;
};

FailureOr<MfmaInsn> MfmaInsn::select(StringRef mfmaInsn) {
  auto mfmaInsnAttrMap = getMfmaInsnAttrMap();
  auto it = mfmaInsnAttrMap.find(mfmaInsn);
  if (it == mfmaInsnAttrMap.end())
    return failure();
  return MfmaInsn((*it).getValue());
}

MfmaInsn::MfmaInsn(const MfmaInsnAttr &mfmaInsnAttr) : attr(mfmaInsnAttr) {}

MfmaInsnAttr MfmaInsn::getAttr() { return attr; }

Type MfmaInsn::getArgType(Type elementType) {
  return attr.nInputsToMfma == 1
             ? elementType
             : VectorType::get({attr.nInputsToMfma}, elementType);
}

VectorType MfmaInsn::getRetType(Type elementType) {
  Builder b(elementType.getContext());
  Type vectorElem;
  if (elementType.isa<IntegerType>())
    vectorElem = b.getI32Type();
  else
    vectorElem = b.getF32Type();
  return VectorType::get({attr.nOutputsOfMfma}, vectorElem);
}

bool MfmaInsn::isCoherentWithK(int64_t kpack, int64_t kPerBlock) {
  bool isKReduction =
      (attr.blocksInOutRegs == 1) && (attr.inputSpansPerMfmaIn > 1);
  if (kpack > 1) {
    if (kpack < attr.k_base) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Should pack at least k_base elements and avoid waste "
                    "xdlopsgemm cycles\n");
      return false;
    }
    if (isKReduction && kPerBlock < attr.inputSpansPerMfmaIn) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "When reduction, KPerBlock must be at least num_input_blks\n");
      return false;
    }
    return true;
  } else {
    if (!isKReduction && kPerBlock < attr.k_base) {
      LLVM_DEBUG(llvm::dbgs()
                 << "When non-reduction, KPerBlock must be at least k_base\n");
      return false;
    }
    if (isKReduction && kPerBlock < attr.k_base * attr.inputSpansPerMfmaIn) {
      LLVM_DEBUG(llvm::dbgs()
                 << "When reduction, KPerBlock must be at least k_base * "
                    "num_input_blks\n");
      return false;
    }
    return true;
  }
}

MfmaTypeId convertTypeToId(mlir::Type dataType) {
  if (dataType.isF32()) {
    return MfmaTypeId::Fp32TyId;
  }
  if (dataType.isF16()) {
    return MfmaTypeId::Fp16TyId;
  }
  if (dataType.isBF16()) {
    return MfmaTypeId::Bf16TyId;
  }
  if (dataType.isInteger(8)) {
    return MfmaTypeId::I8TyId;
  }
  llvm_unreachable("Unsupported input argument type.");
}

FailureOr<MfmaInsnGroup> MfmaInsnGroup::select(mlir::Type elementType,
                                               StringRef arch, int64_t mPerWave,
                                               int64_t nPerWave) {
  LLVM_DEBUG(llvm::dbgs() << "Invoke Mfma group selection:\n"
                          << "elementType: " << elementType << "\n"
                          << "arch: " << arch << "\n"
                          << "mPerWave: " << mPerWave << "\n"
                          << "nPerWave: " << nPerWave << "\n");

  // Use 64x64 as base unit in large waves
  int64_t mPerMfmaGroup = getLenPerMfmaGroup(mPerWave);
  int64_t nPerMfmaGroup = getLenPerMfmaGroup(nPerWave);

  MfmaInsnGroupSelectKey key = {convertTypeToId(elementType), mPerMfmaGroup,
                                nPerMfmaGroup};

  FailureOr<MfmaInsnGroup> result = failure();
  auto selectFrom = [&](const MfmaInsnGroupMap &groupMap) {
    // No point in overriding our good work
    if (succeeded(result))
      return;
    auto it = groupMap.find(key);
    if (it != groupMap.end()) {
      MfmaInsnGroupAttr groupAttr = (*it).second;
      auto maybeInsn = MfmaInsn::select(groupAttr.insn);
      if (failed(maybeInsn)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unsupported instruction: " << groupAttr.insn << "\n");
        result = failure();
        return;
      }
      result = MfmaInsnGroup(elementType, *maybeInsn, groupAttr);
    }
  };
  bool hasOldBf16 = arch.contains("gfx908");
  bool isPreGfx940 = arch.contains("gfx908") || arch.contains("gfx90a");
  if (elementType.isBF16())
    selectFrom(hasOldBf16 ? getMfmaInsnGroupAttrMapGfx908Bf16()
                          : getMfmaInsnGroupAttrMapGfx90aPlusBf16());
  selectFrom(isPreGfx940 ? getMfmaInsnGroupAttrMapPreGfx940Int8()
                         : getMfmaInsnGroupAttrMapGfx940Plus());
  selectFrom(getMfmaInsnGroupAttrMapAllArch());
  if (failed(result)) {
    LLVM_DEBUG(llvm::dbgs() << "No match found in MFMA database\n");
  }
  return result;
}

MfmaInsnGroup::MfmaInsnGroup(Type elementType, const MfmaInsn &mfmaInsn,
                             const MfmaInsnGroupAttr &groupAttr)
    : elementType(elementType), insn(mfmaInsn), groupAttr(groupAttr) {}

int64_t MfmaInsnGroup::getMRepeats(int64_t mPerWave) {
  auto mfmaInsnAttr = getInsnAttr();
  // mnPerXdl is how many row/columns a single Xdlops instruction will compute
  int64_t mnPerXdl = (mfmaInsnAttr.mfmaNonKDim * mfmaInsnAttr.blocksMfma);
  return std::max(int64_t(1), mPerWave / mnPerXdl);
}

int64_t MfmaInsnGroup::getNRepeats(int64_t nPerWave) {
  auto mfmaInsnAttr = getInsnAttr();
  // mnPerXdl is how many row/columns a single Xdlops instruction will compute
  int64_t mnPerXdl = (mfmaInsnAttr.mfmaNonKDim * mfmaInsnAttr.blocksMfma);
  return std::max(int64_t(1), nPerWave / mnPerXdl);
}

int64_t MfmaInsnGroup::getLenPerMfmaGroup(int64_t lenPerWave) {
  return (lenPerWave > 64) ? 64 : lenPerWave;
}

MfmaInsnAttr MfmaInsnGroup::getInsnAttr() { return insn.getAttr(); }

Type MfmaInsnGroup::getArgType() { return insn.getArgType(elementType); }

VectorType MfmaInsnGroup::getRetType() { return insn.getRetType(elementType); }

SmallVector<mlir::rock::MFMAParams, 2> MfmaInsnGroup::getImms() {
  return groupAttr.imms;
}

bool MfmaInsnGroup::isCoherentWithK(int64_t kpack, int64_t kPerBlock) {
  return insn.isCoherentWithK(kpack, kPerBlock);
}
