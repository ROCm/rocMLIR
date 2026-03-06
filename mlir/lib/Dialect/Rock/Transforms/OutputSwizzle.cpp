//===- OutputSwizzle - MLIR Rock ops lowering passes -----===//
//
// Copyright 2024 The MLIR Authors.
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
// ============================================================
//
// This pass changes ThreadwiseWriteAllOp to swizzle on LDS before writing to
// GPU memory. It first tries a cross-lane permute approach using ds_bpermute
// (via gpu.shuffle idx), which avoids LDS memory and barriers. If that's not
// applicable, it falls back to the LDS-based swizzle.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKOUTPUTSWIZZLEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-output-swizzle"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
using mlir::gpu::AddressSpace;

/// Tuning parameter values for controlling output swizzle behavior.
/// These map to the `output_swizzle` integer attribute on the kernel function.
///   0 = DISABLED:  Never apply any output swizzle.
///   1 = ENABLED:   Always try to apply output swizzle (ds_bpermute first,
///                  then LDS-based fallback if feasible).
///   2 = HEURISTIC: Try ds_bpermute first, then LDS-based only when extra
///                  LDS fits within already-allocated LDS budget.
enum OutputSwizzleTuningParam { DISABLED = 0, ENABLED = 1, HEURISTIC = 2 };

namespace {
struct RockOutputSwizzlePass
    : public rock::impl::RockOutputSwizzlePassBase<RockOutputSwizzlePass> {
  void runOnOperation() override;
};
} // end anonymous namespace

/// Returns true if the given memref type has GPU private (register) address
/// space. Private memory is per-thread and maps to VGPRs.
static bool hasPrivateMemoryAddressSpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  if (!memorySpace)
    return false;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace)) {

    return gpuAttr.getValue() == AddressSpace::Private;
  }
  return false;
}

/// Returns true if the given memref type targets global (device) memory.
/// Global memory is neither workgroup (LDS) nor private (registers).
static bool hasGlobalMemoryAddressSpace(MemRefType type) {
  return !gpu::GPUDialect::hasWorkgroupMemoryAddressSpace(type) &&
         !hasPrivateMemoryAddressSpace(type);
}

/// Computes the total LDS (shared memory) bytes allocated in the kernel.
/// Walks all GpuAllocOp operations and sums up the sizes of workgroup-
/// address-space allocations. Used by the heuristic to decide whether
/// the LDS-based swizzle can reuse existing LDS budget.
static int64_t getLDSTotalSize(func::FuncOp &func) {
  int64_t totalSize = 0;
  func.walk([&](GpuAllocOp gpuAlloc) {
    mlir::MemRefType type = gpuAlloc.getOutput().getType();
    auto memSpaceValue =
        dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace())
            .getValue();
    if (memSpaceValue == gpu::GPUDialect::getWorkgroupAddressSpace()) {
      totalSize +=
          getPackedByteSize(type.getNumElements(), type.getElementType());
    }
  });
  return totalSize;
}

/// Checks whether `ldsBytes` bytes of LDS fit within the architecture's
/// maximum LDS per workgroup. Returns success if they fit.
static LogicalResult checkLDSSize(Operation *op, int64_t ldsBytes) {
  StringAttr arch = getArchValue(op);
  const int64_t ldsSize = rock::lookupArchInfo(arch).maxSharedMemPerWG;
  return success(ldsBytes <= ldsSize);
}

/// Extracts the per-block output tile dimensions and the intra-block indexing
/// transforms from a ThreadwiseWriteAllOp.
///
/// The extraViewsAttr on the op contains a chain of TransformMapAttrs that
/// map (tid, item) -> (g_block, m_block, n_block, gemmBlockM, gemmBlockN).
/// This function strips the block-level dimensions (g_block, m_block,
/// n_block) to obtain the intra-block mapping (tid, item) -> (dim0, dim1),
/// where dim0 and dim1 are the per-block M and N tile sizes.
///
/// Returns {dim0PerBlock, dim1PerBlock, idToLDSTransforms} or nullopt on
/// failure (e.g., if the transform chain has an unexpected structure).
static std::optional<std::tuple<int64_t, int64_t, ArrayAttr>>
getIdToLDS(ThreadwiseWriteAllOp &op, OpBuilder &b) {
  ArrayAttr srcTransform = op.getExtraViewsAttr();
  if (srcTransform.empty())
    return std::nullopt;
  StringSet<> dimensionsToRemove{"g_block", "m_block", "n_block"};
  FailureOr<ArrayAttr> maybeIdToLDS =
      removeUpperDims(b, srcTransform, dimensionsToRemove);
  if (failed(maybeIdToLDS)) {
    LLVM_DEBUG(llvm::dbgs() << "getIdToLDS failed\n");
    return std::nullopt;
  }
  ArrayAttr idToLDS = maybeIdToLDS.value();

  ArrayRef<int64_t> shape = getLowerShape(idToLDS);
  if (shape.size() != 2) {
    LLVM_DEBUG(llvm::dbgs()
               << "Currently, this pass works only for two dimensions\n");
    return std::nullopt;
  }
  int64_t dim0PerBlock = shape[0];
  int64_t dim1PerBlock = shape[1];

  return std::make_tuple(dim0PerBlock, dim1PerBlock, idToLDS);
}

/// Extracts the m_tid and n_tid sub-dimension sizes from the output store's
/// transform chain.
///
/// After MFMA/WMMA, the hardware maps each lane to a fixed (m_row, n_col)
/// position in the output tile. The output transform chain encodes this
/// mapping in its first TransformMapAttr, which contains a Merge operation
/// that decomposes `tid` into sub-dimensions:
///
///   MFMA:  tid -> {wave, m_tid, n_tid}
///          where m_tid = lane / inputSpanLen  (row group index)
///                n_tid = lane % inputSpanLen  (column within MFMA tile)
///
///   WMMA:  tid -> {wave_m, wave_n, m_tid, n_tid}
///          where m_tid = lane / mPerAccel, n_tid = lane % mPerAccel
///
/// The sizes of m_tid and n_tid determine the thread-to-output mapping:
///   - n_tid (fast-varying in tid) has `inputSpanLen` values (16 or 32)
///   - m_tid (slow-varying in tid) has `waveSize/inputSpanLen` values (2 or 4)
///
/// For coalesced stores, we want the fast-varying tid sub-dimension to align
/// with the contiguous memory dimension. When it doesn't (e.g., transposed
/// output), swapping m_tid and n_tid via cross-lane permute can help.
///
/// Returns {m_tid_size, n_tid_size} or nullopt if the pattern is not
/// recognized.
static std::optional<std::pair<int64_t, int64_t>>
getTidSubDimSizes(ThreadwiseWriteAllOp &op) {
  ArrayAttr srcTransform = op.getExtraViewsAttr();
  if (srcTransform.empty())
    return std::nullopt;

  const auto firstTr = cast<TransformMapAttr>(srcTransform[0]);
  for (auto trOp : firstTr.getOps()) {
    ArrayRef<StringRef> upperNames = trOp.getUpperNames();
    // Look for the Merge that produces "tid" from {wave, m_tid, n_tid}
    // or {wave_m, wave_n, m_tid, n_tid}
    bool hasTid = false;
    for (auto name : upperNames) {
      if (name == "tid") {
        hasTid = true;
        break;
      }
    }
    if (!hasTid)
      continue;

    if (trOp.getType() != TransformType::Merge)
      continue;

    ArrayRef<StringRef> lowerNames = trOp.getLowerNames();
    ArrayRef<int64_t> params = trOp.getParams();

    // MFMA pattern: tid -> {wave, m_tid, n_tid} with params {waveCount,
    // m_tid_size, n_tid_size}
    if (lowerNames.size() == 3) {
      bool hasWave = lowerNames[0] == "wave";
      bool hasMTid = lowerNames[1] == "m_tid";
      bool hasNTid = lowerNames[2] == "n_tid";
      if (hasWave && hasMTid && hasNTid && params.size() == 3) {
        return std::make_pair(params[1], params[2]);
      }
    }

    // WMMA pattern: tid -> {wave_m, wave_n, m_tid, n_tid} with params
    // {mWaves, nWaves, m_tid_size, n_tid_size}
    if (lowerNames.size() == 4) {
      bool hasWaveM = lowerNames[0] == "wave_m";
      bool hasWaveN = lowerNames[1] == "wave_n";
      bool hasMTid = lowerNames[2] == "m_tid";
      bool hasNTid = lowerNames[3] == "n_tid";
      if (hasWaveM && hasWaveN && hasMTid && hasNTid && params.size() == 4) {
        return std::make_pair(params[2], params[3]);
      }
    }
  }
  return std::nullopt;
}

/// Cross-lane permute output swizzle pattern.
///
/// This pattern improves global store coalescing by using ds_bpermute (via
/// gpu.shuffle idx) to exchange MFMA/WMMA output data between lanes within
/// a wavefront. After the permute, the thread-to-output mapping is adjusted
/// so that consecutive threads write to consecutive memory addresses.
///
/// The key insight: MFMA/WMMA hardware fixes each lane's output position.
/// Lane L always outputs at (m_row, n_col) determined by the instruction.
/// The `n_tid = L % inputSpanLen` sub-dimension is fast-varying in tid.
/// When n_tid maps to the non-contiguous memory dimension (e.g., transposed
/// output or column-major storage), consecutive threads access different
/// cache lines — poor coalescing.
///
/// The permute swaps the roles of m_tid and n_tid: data is physically
/// exchanged between lanes so that what was in lane L moves to lane
/// `(L % n_tid_size) * m_tid_size + (L / n_tid_size)`. The output
/// transforms are then adjusted to reflect the new mapping.
///
/// Advantages over the LDS-based swizzle (ThreadwiseWriteAllRewritePattern):
///   - No LDS memory allocation needed
///   - No LDS barrier needed (intra-wave operation)
///   - Lower latency (~1 cycle per dword via ds_bpermute)
///
/// The pattern is tried first when output_swizzle is 1 or 2 (not disabled).
/// It is automatically applied when:
///   - m_tid_size > 1 (something to swap)
///   - m_tid_size > originalVectorLen (swap would improve vectorization)
///   - Element type is >= 16 bits (8-bit and smaller fall back to LDS)
///   - The full transform chain can be correctly reconstructed
///
/// When the cross-lane permute cannot be applied, the LDS-based swizzle
/// is tried as a fallback (subject to LDS size/heuristic checks).
///
/// Supported element types:
///   - 32-bit (i32, f32): direct shuffle, 1 ds_bpermute per element
///   - 16-bit (f16, bf16, i16): pack 2 elements into i32, shuffle, unpack
///   - 8-bit (i8, f8, bf8): pack 4 elements into i32, shuffle, unpack
///   - 4-bit (i4, fp4): pack 8 elements into i32, shuffle, unpack
struct CrossLanePermuteSwizzlePattern
    : public OpRewritePattern<ThreadwiseWriteAllOp> {
  using OpRewritePattern<ThreadwiseWriteAllOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadwiseWriteAllOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    MemRefType destMemRefType = cast<MemRefType>(op.getDest().getType());
    if (!hasGlobalMemoryAddressSpace(destMemRefType))
      return failure();

    // Extract m_tid and n_tid sizes from the transform chain.
    auto maybeTidSizes = getTidSubDimSizes(op);
    if (!maybeTidSizes.has_value()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CrossLanePermute: could not extract m_tid/n_tid sizes\n");
      return failure();
    }
    auto [mTidSize, nTidSize] = *maybeTidSizes;
    LLVM_DEBUG(llvm::dbgs() << "CrossLanePermute: m_tid_size=" << mTidSize
                            << " n_tid_size=" << nTidSize << "\n");

    // No benefit if m_tid has only 1 value (nothing to swap).
    if (mTidSize <= 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CrossLanePermute: m_tid_size <= 1, skipping\n");
      return failure();
    }

    // Check if swapping would improve vectorization.
    Value matC = op.getDest();
    Value destView = transform(b, matC, op.getExtraViews());
    auto destElemType = cast<MemRefType>(destView.getType()).getElementType();
    if (auto elemVecType = dyn_cast<VectorType>(destElemType)) {
      LLVM_DEBUG(llvm::dbgs() << "CrossLanePermute: vector element type, "
                              << "skipping\n");
      return failure();
    }

    size_t extraIdxCount = op.getExtraIndices().size();
    VectorizationResult originalVecRes =
        getMaxVectorization(destView, extraIdxCount);
    int64_t originalVectorLen = originalVecRes.max;

    // Check if the output transform chain contains swapThreadIterSubDims
    // patterns. These are Merge/Unmerge pairs that split gemmBlockM or
    // gemmBlockN into {m_iter, m_tid} or {n_iter, n_tid} and recombine
    // with swapped ordering to create adjacent element pairs.
    //
    // When these transforms exist, the downstream maps (Maps 4-5) reference
    // "n_tid"/"m_tid"/"n_iter"/"m_iter" by name with sizes that match the
    // original m_tid/n_tid decomposition. If we swap m_tid/n_tid sizes in
    // Map 1, the downstream size assumptions break and addresses become wrong.
    //
    // Therefore, we skip the cross-lane permute when swapThreadIterSubDims
    // is present — those cases already get vectorization from the existing
    // split-and-swap mechanism (e.g., dwordx2 via adjacent pairs).
    ArrayAttr srcTransformCheck = op.getExtraViewsAttr();
    bool hasSwapThreadIterSubDims = false;
    for (auto trAttr : srcTransformCheck) {
      auto trMap = cast<TransformMapAttr>(trAttr);
      for (auto trOp : trMap.getOps()) {
        for (auto ln : trOp.getLowerNames()) {
          if (ln == "n_iter" || ln == "m_iter") {
            hasSwapThreadIterSubDims = true;
            break;
          }
        }
        if (hasSwapThreadIterSubDims)
          break;
      }
      if (hasSwapThreadIterSubDims)
        break;
    }

    // Skip atomic store methods (e.g., splitk uses AtomicAdd for
    // accumulation). Permuting the tid would write partial results to
    // wrong positions, corrupting the atomic accumulation.
    if (op.getStoreMethod() != StoreMethod::Set) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CrossLanePermute: non-Set store method ("
                 << static_cast<int>(op.getStoreMethod())
                 << "), skipping\n");
      return failure();
    }

    if (hasSwapThreadIterSubDims) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CrossLanePermute: swapThreadIterSubDims detected, "
                 << "skipping (already has vectorization via "
                 << "split-and-swap)\n");
      return failure();
    }

    if (mTidSize <= originalVectorLen) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CrossLanePermute: m_tid_size=" << mTidSize
                 << " <= originalVectorLen=" << originalVectorLen
                 << ", no improvement from swap\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "CrossLanePermute: will swap m_tid/n_tid "
                            << "(m_tid_size=" << mTidSize
                            << " > originalVectorLen=" << originalVectorLen
                            << ")\n");

    // Get architecture info for potential instruction selection.
    StringAttr arch = getArchValue(op);
    AmdArchInfo archInfo = lookupArchInfo(arch);
    int64_t waveSize = archInfo.waveSize;

    // Get source buffer info.
    Value srcBuffer = op.getSource();
    auto srcMemRefType = cast<MemRefType>(srcBuffer.getType());
    int64_t numElements = srcMemRefType.getNumElements();
    Type srcElemType = srcMemRefType.getElementType();

    // The accumulator elements are typically i32 (MFMA) or may be f16/f32
    // (WMMA). ds_bpermute operates on i32, so we need to handle type
    // conversions.
    int64_t elemBitWidth = srcElemType.getIntOrFloatBitWidth();
    if (elemBitWidth > 32) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CrossLanePermute: element bit width " << elemBitWidth
                 << " > 32, not yet supported\n");
      return failure();
    }

    // Get thread ID.
    auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());
    Value tidI32 = arith::IndexCastOp::create(b, loc, b.getI32Type(), tid);

    // Compute lane_id = tid % waveSize.
    Value waveSizeVal =
        arith::ConstantIntOp::create(b, loc, waveSize, 32);
    Value laneId = arith::RemUIOp::create(b, loc, tidI32, waveSizeVal);

    // Compute the cross-lane permutation index.
    //
    // The permutation swaps m_tid and n_tid within each wave:
    //   Original:  lane = wave_offset + m_tid * n_tid_size + n_tid
    //   After:     lane holds data from source lane =
    //                wave_offset + (n_tid_orig * m_tid_size + m_tid_orig)
    //
    // Within each wave (ignoring wave_offset):
    //   source_lane = (lane % n_tid_size) * m_tid_size + (lane / n_tid_size)
    //
    // Example for 16x16 MFMA (n_tid_size=16, m_tid_size=4):
    //   lane 0 reads from lane 0    (m=0, n=0 -> src: n=0*4+m=0 = 0)
    //   lane 1 reads from lane 4    (m=0, n=1 -> src: n=1*4+m=0 = 4)
    //   lane 16 reads from lane 1   (m=1, n=0 -> src: n=0*4+m=1 = 1)
    //
    // After this permute, consecutive lanes (0,1,2,3) hold data from
    // lanes (0,4,8,12) which correspond to the same n_tid but different
    // m_tid values — adjacent in the M dimension.
    Value nTidSizeVal =
        arith::ConstantIntOp::create(b, loc, nTidSize, 32);
    Value mTidSizeVal =
        arith::ConstantIntOp::create(b, loc, mTidSize, 32);

    Value laneModN = arith::RemUIOp::create(b, loc, laneId, nTidSizeVal);
    Value laneDivN = arith::DivUIOp::create(b, loc, laneId, nTidSizeVal);
    Value permSrc =
        arith::AddIOp::create(b, loc,
                              arith::MulIOp::create(b, loc, laneModN,
                                                    mTidSizeVal),
                              laneDivN);

    // gpu.shuffle idx uses the source lane ID directly (not byte-addressed).
    // The width parameter controls how many lanes participate.
    Value shuffleWidth =
        arith::ConstantIntOp::create(b, loc, waveSize, 32);

    // Allocate new private buffer for permuted results.
    auto privateMemSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    auto permutedBufferType =
        MemRefType::get({numElements}, srcElemType, AffineMap{}, privateMemSpace);
    auto permutedBuffer = GpuAllocOp::create(b, loc, permutedBufferType);

    // For each element, load -> cast to i32 -> shuffle -> cast back -> store.
    // When elements are smaller than 32 bits, we pack multiple elements per
    // shuffle. We process in dword-sized chunks.
    Type i32Type = b.getI32Type();

    if (elemBitWidth == 32) {
      for (int64_t i = 0; i < numElements; ++i) {
        Value idx = arith::ConstantIndexOp::create(b, loc, i);
        Value elem = memref::LoadOp::create(b, loc, srcBuffer, idx);

        Value elemI32;
        if (srcElemType.isInteger(32)) {
          elemI32 = elem;
        } else {
          elemI32 = arith::BitcastOp::create(b, loc, i32Type, elem);
        }

        auto shuffleResult = gpu::ShuffleOp::create(
            b, loc, elemI32, permSrc, shuffleWidth, gpu::ShuffleMode::IDX);
        Value shuffled = shuffleResult.getShuffleResult();

        Value result;
        if (srcElemType.isInteger(32)) {
          result = shuffled;
        } else {
          result = arith::BitcastOp::create(b, loc, srcElemType, shuffled);
        }

        memref::StoreOp::create(b, loc, result, permutedBuffer, idx);
      }
    } else if (elemBitWidth == 16) {
      // f16/bf16/i16: pack 2 elements into i32, shuffle, unpack.
      if (numElements % 2 != 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "CrossLanePermute: odd number of 16-bit elements\n");
        return failure();
      }
      int64_t numPairs = numElements / 2;
      for (int64_t i = 0; i < numPairs; ++i) {
        Value idx0 = arith::ConstantIndexOp::create(b, loc, i * 2);
        Value idx1 = arith::ConstantIndexOp::create(b, loc, i * 2 + 1);
        Value e0 = memref::LoadOp::create(b, loc, srcBuffer, idx0);
        Value e1 = memref::LoadOp::create(b, loc, srcBuffer, idx1);

        Value e0i16 = arith::BitcastOp::create(b, loc, b.getI16Type(), e0);
        Value e1i16 = arith::BitcastOp::create(b, loc, b.getI16Type(), e1);
        Value e0i32 =
            arith::ExtUIOp::create(b, loc, i32Type, e0i16);
        Value e1i32 =
            arith::ExtUIOp::create(b, loc, i32Type, e1i16);
        Value shifted = arith::ShLIOp::create(
            b, loc, e1i32,
            arith::ConstantIntOp::create(b, loc, 16, 32));
        Value packed = arith::OrIOp::create(b, loc, e0i32, shifted);

        auto shuffleResult = gpu::ShuffleOp::create(
            b, loc, packed, permSrc, shuffleWidth, gpu::ShuffleMode::IDX);
        Value shuffled = shuffleResult.getShuffleResult();

        Value lo = arith::TruncIOp::create(b, loc, b.getI16Type(), shuffled);
        Value hiShifted = arith::ShRUIOp::create(
            b, loc, shuffled,
            arith::ConstantIntOp::create(b, loc, 16, 32));
        Value hi =
            arith::TruncIOp::create(b, loc, b.getI16Type(), hiShifted);
        Value r0 = arith::BitcastOp::create(b, loc, srcElemType, lo);
        Value r1 = arith::BitcastOp::create(b, loc, srcElemType, hi);
        memref::StoreOp::create(b, loc, r0, permutedBuffer, idx0);
        memref::StoreOp::create(b, loc, r1, permutedBuffer, idx1);
      }
    } else if (elemBitWidth == 8) {
      // i8/f8: pack 4 elements into i32, shuffle, unpack.
      if (numElements % 4 != 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "CrossLanePermute: element count not multiple of 4 "
                   << "for 8-bit type\n");
        return failure();
      }
      Type i8Type = b.getI8Type();
      Value c8 = arith::ConstantIntOp::create(b, loc, 8, 32);
      Value c16 = arith::ConstantIntOp::create(b, loc, 16, 32);
      Value c24 = arith::ConstantIntOp::create(b, loc, 24, 32);
      Value cFF = arith::ConstantIntOp::create(b, loc, 0xFF, 32);
      int64_t numQuads = numElements / 4;
      for (int64_t i = 0; i < numQuads; ++i) {
        SmallVector<Value, 4> elems;
        SmallVector<Value, 4> indices;
        for (int j = 0; j < 4; ++j) {
          indices.push_back(
              arith::ConstantIndexOp::create(b, loc, i * 4 + j));
          elems.push_back(
              memref::LoadOp::create(b, loc, srcBuffer, indices.back()));
        }

        // Pack: i32 = e0 | (e1 << 8) | (e2 << 16) | (e3 << 24)
        SmallVector<Value, 4> shifts = {
            arith::ConstantIntOp::create(b, loc, 0, 32), c8, c16, c24};
        Value packed = arith::ConstantIntOp::create(b, loc, 0, 32);
        for (int j = 0; j < 4; ++j) {
          Value ej = arith::BitcastOp::create(b, loc, i8Type, elems[j]);
          Value ext = arith::ExtUIOp::create(b, loc, i32Type, ej);
          Value shifted = arith::ShLIOp::create(b, loc, ext, shifts[j]);
          packed = arith::OrIOp::create(b, loc, packed, shifted);
        }

        auto shuffleResult = gpu::ShuffleOp::create(
            b, loc, packed, permSrc, shuffleWidth, gpu::ShuffleMode::IDX);
        Value shuffled = shuffleResult.getShuffleResult();

        // Unpack: extract each byte
        for (int j = 0; j < 4; ++j) {
          Value byte = arith::ShRUIOp::create(b, loc, shuffled, shifts[j]);
          byte = arith::AndIOp::create(b, loc, byte, cFF);
          Value trunc = arith::TruncIOp::create(b, loc, i8Type, byte);
          Value result =
              arith::BitcastOp::create(b, loc, srcElemType, trunc);
          memref::StoreOp::create(b, loc, result, permutedBuffer,
                                  indices[j]);
        }
      }
    } else if (elemBitWidth == 4) {
      // i4/fp4: pack 8 elements into i32, shuffle, unpack.
      if (numElements % 8 != 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "CrossLanePermute: element count not multiple of 8 "
                   << "for 4-bit type\n");
        return failure();
      }
      Type i4Type = b.getIntegerType(4);
      Value cF = arith::ConstantIntOp::create(b, loc, 0xF, 32);
      int64_t numOctets = numElements / 8;
      for (int64_t i = 0; i < numOctets; ++i) {
        SmallVector<Value, 8> elems;
        SmallVector<Value, 8> indices;
        for (int j = 0; j < 8; ++j) {
          indices.push_back(
              arith::ConstantIndexOp::create(b, loc, i * 8 + j));
          elems.push_back(
              memref::LoadOp::create(b, loc, srcBuffer, indices.back()));
        }

        // Pack: each nibble at position j*4
        Value packed = arith::ConstantIntOp::create(b, loc, 0, 32);
        for (int j = 0; j < 8; ++j) {
          Value shift =
              arith::ConstantIntOp::create(b, loc, j * 4, 32);
          Value ej = arith::BitcastOp::create(b, loc, i4Type, elems[j]);
          Value ext = arith::ExtUIOp::create(b, loc, i32Type, ej);
          Value shifted = arith::ShLIOp::create(b, loc, ext, shift);
          packed = arith::OrIOp::create(b, loc, packed, shifted);
        }

        auto shuffleResult = gpu::ShuffleOp::create(
            b, loc, packed, permSrc, shuffleWidth, gpu::ShuffleMode::IDX);
        Value shuffled = shuffleResult.getShuffleResult();

        // Unpack: extract each nibble
        for (int j = 0; j < 8; ++j) {
          Value shift =
              arith::ConstantIntOp::create(b, loc, j * 4, 32);
          Value nibble = arith::ShRUIOp::create(b, loc, shuffled, shift);
          nibble = arith::AndIOp::create(b, loc, nibble, cF);
          Value trunc = arith::TruncIOp::create(b, loc, i4Type, nibble);
          Value result =
              arith::BitcastOp::create(b, loc, srcElemType, trunc);
          memref::StoreOp::create(b, loc, result, permutedBuffer,
                                  indices[j]);
        }
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "CrossLanePermute: unsupported " << elemBitWidth
                 << "-bit element type\n");
      return failure();
    }

    // Reconstruct the output transform chain with m_tid and n_tid swapped
    // consistently through ALL maps.
    //
    // We create new TransformAttr objects with swapped names and, for the
    // tid Merge in Map 1, swapped sizes. Then we use
    // TransformMapAttr::get(ops, upperBounds, lowerBounds) which
    // automatically recomputes the affine map from the transform ops.
    //
    // The affine map is determined by the transform types, params, and
    // dimension indices — not by names. Names are labels for downstream
    // lookups. By swapping names consistently, downstream maps that
    // reference "m_tid" or "n_tid" pick up the swapped values.
    // Instead of modifying the transform chain (which has complex
    // invariants), we use a simpler approach:
    //
    // 1. Shuffle the data so lane L holds perm(L)'s result (done above).
    // 2. Compute perm_tid: the tid value that corresponds to the source
    //    lane. This makes the original transform chain compute the
    //    correct output address for perm(L)'s data.
    // 3. Replace tid in extraIndices with perm_tid.
    // 4. Keep the original transform chain unchanged.
    //
    // This works because: the transforms compute address(tid, item).
    // After the shuffle, lane L holds data from perm(L). By using
    // perm_tid = perm(tid) as the index, the transforms compute
    // address(perm(tid), item) = the correct destination for perm(L)'s
    // result. Each lane writes its shuffled data to the correct position.

    // Compute perm_tid as an index value.
    // perm formula: perm(lane) = (lane % n_tid_size) * m_tid_size + (lane / n_tid_size)
    // Applied to tid (within the wave): perm_tid = wave_offset + perm(lane_id)
    // where wave_offset = tid - lane_id = (tid / waveSize) * waveSize
    Value waveOffset =
        arith::SubIOp::create(b, loc, tidI32, laneId);
    Value permTid =
        arith::AddIOp::create(b, loc, waveOffset, permSrc);
    Value permTidIdx =
        arith::IndexCastOp::create(b, loc, b.getIndexType(), permTid);

    // Build new extraIndices with perm_tid replacing the original tid.
    // The tid is the last extra index.
    SmallVector<Value> newExtraIndices(op.getExtraIndices());
    newExtraIndices.back() = permTidIdx;

    b.replaceOpWithNewOp<ThreadwiseWriteAllOp>(
        op, permutedBuffer, op.getDest(), op.getExtraViewsAttr(),
        newExtraIndices, op.getStoreMethod(),
        op.getForceUnroll(), op.getUseIndexDiffs());

    return success();
  }
};

/// LDS-based output swizzle pattern.
///
/// Rewrites a ThreadwiseWriteAllOp that stores directly to global memory
/// into a three-step sequence:
///   1. Write MFMA/WMMA results from registers to LDS (using the existing
///      tid-to-block transform for addressing).
///   2. LDS barrier (ensure all threads have written before any reads).
///   3. Read back from LDS with a new linearized layout that places
///      consecutive elements in consecutive threads, then write to global
///      memory with improved vectorization.
///
/// The linearized layout maps:
///   flatIndex = iter * blockSize * vectorLen + tid * vectorLen + vecIdx
/// which gives each thread `vectorLen` contiguous elements per iteration,
/// enabling vector stores (e.g., global_store_dwordx4).
///
/// The pass decides whether to apply based on:
///   - Whether the achievable vectorization (`elementsWrittenPerThread`)
///     exceeds the original vectorization (`originalVectorLen`).
///   - The LDS memory budget (controlled by the tuning parameter).
///   - The output element type (vector element types are skipped).
///
/// The new output transforms map the linearized (tid, iter) coordinates
/// back to the global matrix dimensions (gemmG, gemmM, gemmN).
struct ThreadwiseWriteAllRewritePattern
    : public OpRewritePattern<ThreadwiseWriteAllOp> {
  using OpRewritePattern<ThreadwiseWriteAllOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadwiseWriteAllOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Prepare some useful constants.
    Value convertedC = op.getSource();
    Value matC = op.getDest();
    Type destType = op.getDest().getType().getElementType();

    // Convert from reg -> memory transform to reg -> block
    int64_t dim0PerBlock, dim1PerBlock;
    ArrayAttr idToLDS;
    std::optional<std::tuple<int64_t, int64_t, ArrayAttr>> maybeBlockInfo =
        getIdToLDS(op, b);
    if (!maybeBlockInfo.has_value()) {
      return failure();
    }
    std::tie(dim0PerBlock, dim1PerBlock, idToLDS) = maybeBlockInfo.value();

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> cShape;
    cShape = op.getDest().getType().getShape();
    if (cShape.size() != 3) {
      LLVM_DEBUG(llvm::dbgs()
                 << " getDest() number of dimensions is not 3, it's "
                 << cShape.size() << "\n");
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "dim1PerBlock: " << dim1PerBlock
                            << " dim0PerBlock: " << dim0PerBlock << "\n");
    FailureOr<IntegerAttr> maybeGridSize = getGridSize(op);
    if (failed(maybeGridSize)) {
      return failure();
    }
    FailureOr<IntegerAttr> maybeBlockSize = getBlockSize(op);
    if (failed(maybeBlockSize)) {
      return failure();
    }
    int64_t blockSize = maybeBlockSize.value().getValue().getSExtValue();
    bool useIndexDiffs = true;
    bool forceUnroll = true;
    int64_t ldsRequiredBytes =
        getPackedByteSize(dim0PerBlock * dim1PerBlock, destType);

    // Decide register vectorization.
    constexpr int64_t dimensionM = 1;
    constexpr int64_t dimensionN = 2;
    int64_t dataPerThread = (dim1PerBlock * dim0PerBlock) / blockSize;
    if ((dim1PerBlock * dim0PerBlock) % blockSize != 0) {
      return failure();
    }

    VectorizationResult mVectorRes =
        getMaxVectorization(matC, dimensionM, /*inputDimLen=*/
                            std::nullopt, matC.getDefiningOp());
    int64_t mVectorLen = mVectorRes.max;
    VectorizationResult nVectorRes =
        getMaxVectorization(matC, dimensionN, /*inputDimLen=*/
                            std::nullopt, matC.getDefiningOp());
    int64_t nVectorLen = nVectorRes.max;
    int64_t dim = (mVectorLen > nVectorLen) ? dimensionM : dimensionN;
    int64_t vectorLen = std::max(mVectorLen, nVectorLen);

    // check vectorization of iter in the original map to decide if we run the
    // pass
    Value destView = transform(b, matC, op.getExtraViews());
    auto destElemType = cast<MemRefType>(destView.getType()).getElementType();
    if (auto elemVecType = dyn_cast<VectorType>(destElemType)) {
      LLVM_DEBUG(llvm::dbgs() << "ThreadwiseWriteAllOp saves a vector type"
                              << ", skipping swizzle\n");
      return failure();
    }
    size_t extraIdxCount = op.getExtraIndices().size();
    VectorizationResult vectorRes =
        getMaxVectorization(destView, extraIdxCount);
    int64_t originalVectorLen = vectorRes.max;
    int64_t elementsWrittenPerThread = math_util::gcd(dataPerThread, vectorLen);

    if (elementsWrittenPerThread <= originalVectorLen) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Original vectorization of 'iter' is " << originalVectorLen
                 << ", the output swizzle could achieve "
                 << elementsWrittenPerThread << ", skipping swizzle\n");
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Original vectorization of 'iter' is " << originalVectorLen
               << ", the output swizzle could achieve "
               << elementsWrittenPerThread << ", performing swizzle\n");

    // Get current workitem ID.
    auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());

    // Allocate LDS for output.
    auto workgroupMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsMemRefOutputType =
        MemRefType::get({ldsRequiredBytes}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsBufferOutput = GpuAllocOp::create(b, loc, ldsMemRefOutputType);
    auto typedBuffer = viewBufferAs(b, ldsBufferOutput, destType);

    // Convert from raw -> dim0PerBlock, dim1PerBlock
    TopDownTMBuilder mnToRaw(b, {"gemmM", "gemmN"},
                             {dim0PerBlock, dim1PerBlock});
    if (dim == dimensionN) {
      mnToRaw.unmerge("flatten", 0, {"gemmM", "gemmN"},
                      {dim0PerBlock, dim1PerBlock});
    } else {
      mnToRaw.unmerge("flatten", 0, {"gemmN", "gemmM"},
                      {dim1PerBlock, dim0PerBlock});
    }
    auto mnToRawAttr = mnToRaw.get();

    SmallVector<Attribute> transformMNToRawAttrs;
    transformMNToRawAttrs.push_back(mnToRawAttr);
    ArrayAttr transformMNToRaw = b.getArrayAttr(transformMNToRawAttrs);

    auto ldsBufferMNToRaw = transform(b, typedBuffer, transformMNToRaw);

    // Store C results to LDS.
    ThreadwiseWriteAllOp::create(b, loc, convertedC, ldsBufferMNToRaw,
                                 /*extraViews=*/idToLDS,
                                 /*extraIndices=*/ValueRange{tid},
                                 StoreMethod::Set,
                                 /*forceUnroll=*/forceUnroll,
                                 /*useIndexDiffs=*/useIndexDiffs);

    // Load from LDS to registers.
    int64_t iter = dataPerThread / elementsWrittenPerThread;
    LLVM_DEBUG(llvm::dbgs()
               << "blockSize: " << blockSize
               << " dataPerThread: " << dataPerThread
               << " elementsWrittenPerThread: " << elementsWrittenPerThread
               << " iter: " << iter << "\n");
    if (dim == dimensionM) {
      LLVM_DEBUG(llvm::dbgs() << "dim = M\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "dim = N\n");
    }

    Value finalC =
        gpuAlloc(b, loc, dataPerThread, destType, AddressSpace::Private);

    TopDownTMBuilder tidIterMerge(b, {"tid", "iter"},
                                  {blockSize, dataPerThread});
    tidIterMerge.passThrough(ArrayRef<StringRef>{"tid"});
    tidIterMerge.merge({"iter", "numElements"}, {1, 2}, "iter",
                       {iter, elementsWrittenPerThread});
    auto tidIterMergeAttr = tidIterMerge.get();

    auto tidIterFlatten =
        TopDownTMBuilder::below(tidIterMerge, tidIterMergeAttr);
    tidIterFlatten.unmerge("flattenBlock", 0, {"iter", "tid", "numElements"},
                           {iter, blockSize, elementsWrittenPerThread});
    auto tidIterFlattenAttr = tidIterFlatten.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(tidIterMergeAttr);
    transformAttrs.push_back(tidIterFlattenAttr);

    ArrayAttr ldsRead = b.getArrayAttr(transformAttrs);
    auto ldsBufferForLoad = transform(b, typedBuffer, ldsRead);

    // LDS barrier.
    LDSBarrierOp::create(b, loc);

    ThreadwiseReadIntoOp::create(b, loc, ldsBufferForLoad, finalC,
                                 b.getArrayAttr({}), ValueRange{tid},
                                 forceUnroll, useIndexDiffs);

    SmallVector<int64_t, 5> bidGridLengths;
    SmallVector<StringRef, 5> bidGridOrder;

    llvm::SmallVector<uint32_t> passThrough, passThroughWOTid;
    // drop last dimension (tid)
    passThroughWOTid.reserve(op.getExtraIndices().size() - 1);
    passThrough.reserve(op.getExtraIndices().size());
    for (size_t dim = 0; dim < op.getExtraIndices().size() - 1; ++dim) {
      passThroughWOTid.push_back(dim);
      passThrough.push_back(dim);
    }
    passThrough.push_back(op.getExtraIndices().size() - 1);
    uint32_t idx = op.getExtraIndices().size();
    if (idx < 3) {
      return failure();
    }

    bool isAttention = idx == 3;
    LLVM_DEBUG(llvm::dbgs() << "isAttention: " << isAttention << "\n");

    // Save to memory
    ArrayAttr srcTransform = op.getExtraViewsAttr();
    const auto upperTr = cast<TransformMapAttr>(srcTransform[0]);
    ArrayRef<int64_t> startShape = upperTr.getUpperBounds().asArrayRef();
    llvm::SmallVector<StringRef> startNames;
    startNames.reserve(startShape.size());
    for (auto tr : upperTr.getOps()) {
      ArrayRef<StringRef> upperNames = tr.getUpperNames();
      startNames.append(upperNames.begin(), upperNames.end());
    }
    assert(startNames.size() == startShape.size());

    // get dimension names
    StringRef gDimName = startNames[0];
    StringRef nDimName = startNames[startNames.size() - 3];
    StringRef tidDimName = startNames[startNames.size() - 2];
    StringRef itemDimName = startNames[startNames.size() - 1];

    // get mBlocks and nBlocks
    int64_t nBlocks = startShape[startShape.size() - 3];
    // only valid for !isAttention
    int64_t mBlocks = startShape[startShape.size() - 4];

    TopDownTMBuilder tidIterMergeMem(b, startNames, startShape, loc);
    tidIterMergeMem.passThrough(passThrough, passThrough);
    tidIterMergeMem.merge({"iter", "numElements"}, {idx, idx + 1}, itemDimName,
                          {iter, elementsWrittenPerThread});
    auto tidIterMergeMemAttr = tidIterMergeMem.get();

    TopDownTMBuilder tidIterFlattenMem =
        TopDownTMBuilder::below(tidIterMergeMem, tidIterMergeMemAttr);
    tidIterFlattenMem.passThrough(passThroughWOTid, passThroughWOTid);
    tidIterFlattenMem.unmerge("flattenBlock", idx - 1,
                              {"iter", tidDimName, "numElements"},
                              {iter, blockSize, elementsWrittenPerThread});
    auto tidIterFlattenMemAttr = tidIterFlattenMem.get();

    auto flattenToBlockCoord =
        TopDownTMBuilder::below(tidIterFlattenMem, tidIterFlattenMemAttr);
    flattenToBlockCoord.passThrough(passThroughWOTid, passThroughWOTid);
    if (dim == dimensionN) {
      flattenToBlockCoord.merge({"block_m", "block_n"}, {idx - 1, idx},
                                "flattenBlock", {dim0PerBlock, dim1PerBlock});
    } else {
      flattenToBlockCoord.merge({"block_n", "block_m"}, {idx - 1, idx},
                                "flattenBlock", {dim1PerBlock, dim0PerBlock});
    }
    TransformMapAttr flattenToBlockCoordAttr = flattenToBlockCoord.get();

    auto toMatrixC =
        TopDownTMBuilder::below(flattenToBlockCoord, flattenToBlockCoordAttr);
    toMatrixC.passThrough({"gemmG"}, {0}, {gDimName});
    if (isAttention) {
      toMatrixC.passThrough({"gemmM"}, {1}, {"block_m"});
    } else {
      toMatrixC.unmerge("gemmM", 1, {startNames[1], "block_m"},
                        {mBlocks, dim0PerBlock});
    }
    toMatrixC.unmerge("gemmN", 2, {nDimName, "block_n"},
                      {nBlocks, dim1PerBlock});
    TransformMapAttr toMatrixCAttr = toMatrixC.get();

    SmallVector<Attribute> transformAttrsStore{
        tidIterMergeMemAttr, tidIterFlattenMemAttr, flattenToBlockCoordAttr,
        toMatrixCAttr};
    ArrayAttr idToMatrixCMaps = b.getArrayAttr(transformAttrsStore);

    b.replaceOpWithNewOp<ThreadwiseWriteAllOp>(
        op, finalC, matC, idToMatrixCMaps,
        /*extraIndices=*/
        op.getExtraIndices(), op.getStoreMethod(), forceUnroll, useIndexDiffs);
    return success();
  }
};

/// Main pass entry point.
///
/// The pass operates in two phases on ThreadwiseWriteAllOps that target
/// global memory:
///
/// Phase 1 — Cross-lane permute (when output_swizzle != 0):
///   Tries to swap m_tid/n_tid using ds_bpermute within the wavefront.
///   No LDS memory or barriers needed. Applied to ALL global writes
///   regardless of LDS budget. Falls back gracefully when the transform
///   chain reconstruction is not yet supported.
///
/// Phase 2 — LDS-based swizzle (default, controlled by output_swizzle tuning):
///   For any remaining global writes not handled by Phase 1, tries the
///   LDS-based approach: write to LDS, barrier, read back in coalesced
///   order, write to global. Subject to LDS size checks and heuristics.
///
/// The two-phase approach ensures that the cheaper cross-lane permute is
/// tried first, falling back to the more expensive LDS-based swizzle only
/// when the permute is not applicable or not enabled.
void RockOutputSwizzlePass::runOnOperation() {
  func::FuncOp func = getOperation();
  IRRewriter rewriter(func->getContext());

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;

  // Get total LDS memory allocated
  int64_t ldsAllocated = getLDSTotalSize(func);

  OutputSwizzleTuningParam tuning = OutputSwizzleTuningParam::HEURISTIC;
  if (func->hasAttrOfType<IntegerAttr>(
          rock::OutputSwizzleAttr::getMnemonic())) {
    // 0 -> disabled, 1 -> enabled, 2 -> heuristic
    int64_t outputSwizzleTuning =
        func->getAttrOfType<IntegerAttr>(rock::OutputSwizzleAttr::getMnemonic())
            .getInt();
    tuning = static_cast<OutputSwizzleTuningParam>(outputSwizzleTuning);
  }

  // Phase 1: Try cross-lane permute on all global writes (no LDS needed).
  // Only attempt when output swizzle is not disabled.
  if (tuning != OutputSwizzleTuningParam::DISABLED) {
    SmallVector<Operation *, 4> allGlobalWrites;
    func.walk([&allGlobalWrites](ThreadwiseWriteAllOp threadwiseWriteAll) {
      MemRefType destMemRefType =
          cast<MemRefType>(threadwiseWriteAll.getDest().getType());
      if (hasGlobalMemoryAddressSpace(destMemRefType)) {
        allGlobalWrites.push_back(threadwiseWriteAll);
      }
    });

    if (!allGlobalWrites.empty()) {
      GreedyRewriteConfig config;
      config.setStrictness(GreedyRewriteStrictness::ExistingOps);

      // Record which ops exist before the permute pass.
      DenseSet<Operation *> opsBeforePermute;
      for (auto *op : allGlobalWrites)
        opsBeforePermute.insert(op);

      RewritePatternSet permutePatterns(&getContext());
      permutePatterns.add<CrossLanePermuteSwizzlePattern>(&getContext());
      if (failed(applyOpPatternsGreedily(allGlobalWrites,
                                         std::move(permutePatterns), config))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "CrossLanePermute pattern application failed\n");
      }

      // Track which original ops were replaced (successfully permuted).
      // Ops created by the permute pattern should NOT get the LDS swizzle.
      func.walk([&opsBeforePermute](ThreadwiseWriteAllOp writeOp) {
        MemRefType destType = cast<MemRefType>(writeOp.getDest().getType());
        if (hasGlobalMemoryAddressSpace(destType) &&
            !opsBeforePermute.contains(writeOp.getOperation())) {
          // This is a new op created by the permute. Mark it to skip LDS.
          writeOp->setAttr("rock.permute_swizzled",
                           UnitAttr::get(writeOp->getContext()));
        }
      });
    }
  }

  // Now collect remaining global writes for the LDS-based swizzle,
  // applying the original LDS size/heuristic checks.
  // Skip ops that were already handled by the cross-lane permute.
  SmallVector<Operation *, 4> writes;
  func.walk([&writes, &rewriter, ldsAllocated,
             tuning](ThreadwiseWriteAllOp threadwiseWriteAll) {
    // Skip ops already handled by cross-lane permute.
    if (threadwiseWriteAll->hasAttr("rock.permute_swizzled"))
      return;

    MemRefType destMemRefType =
        cast<MemRefType>(threadwiseWriteAll.getDest().getType());

    if (hasGlobalMemoryAddressSpace(destMemRefType)) {
      int64_t dim0PerBlock, dim1PerBlock;
      ArrayAttr idToLDS;
      rewriter.setInsertionPoint(threadwiseWriteAll);
      std::optional<std::tuple<int64_t, int64_t, ArrayAttr>> maybeBlockInfo =
          getIdToLDS(threadwiseWriteAll, rewriter);
      if (!maybeBlockInfo.has_value()) {
        LLVM_DEBUG(llvm::dbgs() << "OutputSwizzle skipped due to getIdToLDS\n");
        return;
      }
      std::tie(dim0PerBlock, dim1PerBlock, idToLDS) = maybeBlockInfo.value();

      Type destType = threadwiseWriteAll.getDest().getType().getElementType();
      int64_t ldsRequiredBytes =
          getPackedByteSize(dim0PerBlock * dim1PerBlock, destType);

      if (failed(checkLDSSize(threadwiseWriteAll, ldsRequiredBytes))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "OutputSwizzle requires too much LDS memory: "
                   << ldsRequiredBytes << " bytes, skipping pass\n");
        return;
      }
      if (tuning == OutputSwizzleTuningParam::HEURISTIC) {
        LLVM_DEBUG(llvm::dbgs() << "Using heuristic\n");
        if (ldsRequiredBytes > ldsAllocated) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "OutputSwizzle requires more LDS memory, current usage: "
              << ldsAllocated << " bytes, required: " << ldsRequiredBytes
              << " bytes, skipping pass\n");
          return;
        }
      } else if (tuning == OutputSwizzleTuningParam::DISABLED) {
        LLVM_DEBUG(llvm::dbgs()
                   << "OutputSwizzle disabled using tuning params\n");
        return;
      }
      writes.push_back(threadwiseWriteAll);
    }
  });
  if (writes.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No ThreadwiseWriteAllOp writes eligible for "
                               "LDS-based swizzle\n");
  } else {
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    RewritePatternSet ldsPatterns(&getContext());
    ldsPatterns.add<ThreadwiseWriteAllRewritePattern>(&getContext());
    if (failed(
            applyOpPatternsGreedily(writes, std::move(ldsPatterns), config))) {
      return signalPassFailure();
    }
  }
}
