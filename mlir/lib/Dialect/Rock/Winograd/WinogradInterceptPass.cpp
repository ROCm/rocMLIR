//===-- WinogradInterceptPass.cpp - Assemble Winograd before MLIR ---------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This pass detects rock.conv ops with a "winograd:" perf_config and replaces
// them with pre-assembled Winograd kernels. The func.func keeps its original
// 3-memref signature so downstream consumers (MIGraphX) see no difference
// compared to a GEMM kernel.
//
// The assembled kernel HSACO is placed in a gpu.binary, and the rock.conv is
// replaced with a gpu.launch_func that packs all scalar constants plus the
// three runtime pointers into the kernel's V2 ABI argument layout.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Winograd/WinogradInterceptPass.h"
#include "mlir/Dialect/Rock/Winograd/WinogradArgLayout.h"
#include "mlir/Dialect/Rock/Winograd/WinogradAssembler.h"
#include "mlir/Dialect/Rock/Winograd/WinogradConvProblem.h"
#include "mlir/Dialect/Rock/Winograd/WinogradSolver.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockConvInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-winograd-intercept"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::rock::winograd;

struct WinogradInterceptPass
    : public PassWrapper<WinogradInterceptPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WinogradInterceptPass)

  WinogradInterceptPass() = default;
  WinogradInterceptPass(const WinogradInterceptPassOptions &opts)
      : triple(opts.triple), chip(opts.chip), features(opts.features) {}

  StringRef getArgument() const override { return "rock-winograd-intercept"; }
  StringRef getDescription() const override {
    return "Replace rock.conv with Winograd assembly kernel when perf_config "
           "requests it";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, LLVM::LLVMDialect, ROCDL::ROCDLDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Find the winograd perf_config and the conv op
    std::string winoPerfConfig;
    Operation *convOp = nullptr;
    module.walk([&](Operation *op) -> WalkResult {
      if (auto attr = op->getAttrOfType<StringAttr>("perf_config")) {
        if (attr.getValue().starts_with("winograd:")) {
          winoPerfConfig = attr.getValue().str();
          convOp = op;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    if (winoPerfConfig.empty())
      return; // Not a Winograd config, let normal pipeline handle it.

    // Only rock.conv ops implement RockConvInterface; bail if the walk
    // found a non-conv op with a winograd: perf_config (shouldn't happen,
    // but prevents a crash from cast<RockConvInterface>).
    if (!isa<RockConvInterface>(convOp)) {
      convOp->emitError("winograd: perf_config on non-conv op");
      return signalPassFailure();
    }

    // Winograd assembly kernels don't support fusions (conv+relu, conv+bias,
    // etc.). If the module has fused ops beyond the rock.conv, bail out and
    // let the GEMM pipeline handle it. We remove the perf_config so that
    // AffixTuningParameters uses its normal heuristic GEMM selection.
    bool hasFusedOps = false;
    module.walk([&](func::FuncOp f) {
      int otherKernelOps = 0;
      f.walk([&](Operation *op) {
        if (isa<linalg::GenericOp>(op) || isa<rock::ReduceOp>(op))
          ++otherKernelOps;
      });
      if (otherKernelOps > 0)
        hasFusedOps = true;
    });
    if (hasFusedOps) {
      convOp->emitRemark("Winograd: skipping fused module, falling back to "
                         "GEMM pipeline");
      convOp->removeAttr("perf_config");
      return;
    }

    // Validate layout: Winograd assembly kernels only support NCHW data layout.
    // Check that spatial dims ("0","1") come after channel dims ("c"/"k") in
    // the layout strings. Reject NHWC, NDHWC, and other channel-last layouts.
    if (auto convIF = dyn_cast<RockConvInterface>(convOp)) {
      auto checkNCHW = [](ArrayAttr layoutAttr,
                          StringRef /*name*/) -> LogicalResult {
        if (!layoutAttr)
          return success(); // no layout attr, assume OK
        int channelIdx = -1, spatialIdx = -1;
        for (int i = 0, e = layoutAttr.size(); i < e; ++i) {
          StringRef dim = cast<StringAttr>(layoutAttr[i]).getValue();
          // Channel dims: "c", "ci", "co", "k", "ki", "ko"
          if (dim == "c" || dim == "ci" || dim == "co" || dim == "k" ||
              dim == "ki" || dim == "ko")
            channelIdx = i;
          // Spatial dims: "0", "1", "0i", "1i", "0o", "1o"
          if (dim == "0" || dim == "0i" || dim == "0o")
            spatialIdx = i;
        }
        // In NCHW, channel comes before spatial (lower index)
        if (channelIdx >= 0 && spatialIdx >= 0 && channelIdx > spatialIdx)
          return failure();
        return success();
      };

      auto filterLayout = convOp->getAttrOfType<ArrayAttr>("filter_layout");
      auto inputLayout = convOp->getAttrOfType<ArrayAttr>("input_layout");
      auto outputLayout = convOp->getAttrOfType<ArrayAttr>("output_layout");

      if (failed(checkNCHW(filterLayout, "filter")) ||
          failed(checkNCHW(inputLayout, "input")) ||
          failed(checkNCHW(outputLayout, "output"))) {
        convOp->emitRemark(
            "Winograd: unsupported data layout (requires NCHW, got "
            "channel-last). Falling back to GEMM pipeline. "
            "Filter layout: ")
            << filterLayout << ", Input layout: " << inputLayout
            << ", Output layout: " << outputLayout;
        convOp->removeAttr("perf_config");
        return;
      }
    }

    // Get arch from module or pass options
    std::string archChip = chip;
    if (archChip.empty()) {
      if (auto archAttr = module->getAttrOfType<StringAttr>("mhal.arch")) {
        StringRef arch = archAttr.getValue();
        // Handle "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-" formats
        if (arch.contains("gfx")) {
          for (auto part : llvm::split(arch, ':')) {
            if (part.starts_with("gfx")) {
              archChip = part.str();
              break;
            }
          }
        }
        if (archChip.empty())
          archChip = arch.str();
      }
    }
    if (archChip.empty()) {
      convOp->emitError("Cannot determine GPU architecture for Winograd");
      return signalPassFailure();
    }

    // Build minimal problem for resolveFromPerfConfig
    WinogradConvProblem problem;
    problem.arch = archChip;
    problem.isFp16 = winoPerfConfig.find("fp16") != std::string::npos;
    problem.isBf16 = winoPerfConfig.find("bf16") != std::string::npos;
    problem.isFp32 = !problem.isFp16 && !problem.isBf16;
    problem.isXnackEnabled = false;
    problem.direction = WinogradDirection::Forward;
    // Defaults for resolve (not used by resolveFromPerfConfig)
    problem.N = 1;
    problem.C = 64;
    problem.H = 56;
    problem.W = 56;
    problem.K = 64;
    problem.R = 3;
    problem.S = 3;
    problem.outH = 56;
    problem.outW = 56;
    problem.padH = 1;
    problem.padW = 1;
    problem.strideH = 1;
    problem.strideW = 1;
    problem.dilationH = 1;
    problem.dilationW = 1;
    problem.groupCount = 1;
    problem.numCU = 120;

    // Extract real dims from the conv op
    if (auto convIF = dyn_cast<RockConvInterface>(convOp)) {
      auto gemmOp = cast<RockGemmWrapperInterface>(convOp);
      auto strides = extractFromIntegerArrayAttr<int64_t>(convIF.getStrides());
      auto dilations =
          extractFromIntegerArrayAttr<int64_t>(convIF.getDilations());
      auto padding = extractFromIntegerArrayAttr<int64_t>(convIF.getPadding());
      problem.strideH = strides.size() > 0 ? strides[0] : 1;
      problem.strideW = strides.size() > 1 ? strides[1] : 1;
      problem.dilationH = dilations.size() > 0 ? dilations[0] : 1;
      problem.dilationW = dilations.size() > 1 ? dilations[1] : 1;
      problem.padH = padding.size() > 0 ? padding[0] : 0;
      problem.padW = padding.size() > 2 ? padding[2] : 0;
      problem.numCU = getNumCUValue(gemmOp);

      mlir::Value filterVal = convIF.getFilter();
      mlir::Value inputVal = convIF.getInput();
      mlir::Value outputVal = convIF.getOutput();
      // Walk up through transform ops to get logical shapes
      auto getShape = [](mlir::Value v) {
        while (auto xform = v.getDefiningOp<rock::TransformOp>())
          v = xform.getInput();
        return cast<ShapedType>(v.getType());
      };
      auto fType = getShape(filterVal);

      // Determine element type
      mlir::Type elemType = fType.getElementType();
      problem.isFp16 = elemType.isF16();
      problem.isFp32 = elemType.isF32();
      problem.isBf16 = elemType.isBF16();

      // Shape: flattened 1-D memref, sizes encoded in transforms
      // Use the logical shape from the transformed type
      auto fLogical = cast<ShapedType>(filterVal.getType());
      auto iLogical = cast<ShapedType>(inputVal.getType());
      auto oLogical = cast<ShapedType>(outputVal.getType());
      if (fLogical.getRank() >= 5) {
        problem.K = fLogical.getDimSize(1);
        problem.R = fLogical.getDimSize(2);
        problem.S = fLogical.getDimSize(3);
        problem.C = fLogical.getDimSize(4);
        problem.groupCount = fLogical.getDimSize(0);
      }
      // Input layout: [ni, gi, ci, 0i, 1i] -> N at 0, H at 3, W at 4
      if (iLogical.getRank() >= 5) {
        problem.N = iLogical.getDimSize(0);
        problem.H = iLogical.getDimSize(3);
        problem.W = iLogical.getDimSize(4);
      }
      // Output layout: [no, go, ko, 0o, 1o] -> outH at 3, outW at 4
      if (oLogical.getRank() >= 5) {
        problem.outH = oLogical.getDimSize(3);
        problem.outW = oLogical.getDimSize(4);
      }

      auto kernelType = gemmOp.getKernelType();
      problem.direction = (kernelType == KernelType::ConvBwdData)
                              ? WinogradDirection::BackwardData
                              : WinogradDirection::Forward;
    }

    // Resolve kernel selection
    auto selection =
        WinogradSolver::resolveFromPerfConfig(problem, winoPerfConfig);
    if (!selection) {
      convOp->emitError("Failed to resolve Winograd perf_config: ")
          << winoPerfConfig;
      return signalPassFailure();
    }

    // Assemble
    std::string tripleStr = triple.empty() ? "amdgcn-amd-amdhsa" : triple;
    auto hsaco =
        assembleWinogradKernel(*selection, archChip, tripleStr, features);
    if (!hsaco) {
      convOp->emitError("Winograd assembly failed for ") << archChip;
      return signalPassFailure();
    }

    // === Build the replacement IR ===
    MLIRContext *ctx = module.getContext();
    OpBuilder builder(ctx);
    Location loc = convOp->getLoc();
    ctx->getOrLoadDialect<ROCDL::ROCDLDialect>();

    module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    UnitAttr::get(ctx));

    // Compute strides and flags
    auto tensorStrides = WinogradArgLayout::computeStrides(problem);
    uint64_t flags64 = WinogradArgLayout::computeFlagsV2(
        problem.direction == WinogradDirection::Forward, false,
        problem.groupCount > 1, false);

    // LLVM types used for building the arg buffer and launch calls
    auto i32 = builder.getI32Type();
    auto i64 = builder.getI64Type();
    auto f32 = builder.getF32Type();
    auto ptr = LLVM::LLVMPointerType::get(ctx);
    auto i8 = builder.getI8Type();

    // Replace the rock.conv with a direct mgpu kernel launch
    auto funcOp = convOp->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      convOp->emitError("rock.conv not inside func.func");
      return signalPassFailure();
    }

    auto convIF = cast<RockConvInterface>(convOp);
    OpBuilder fb(convOp);

    // Get base memrefs (walk through transforms)
    auto getBase = [](mlir::Value v) -> mlir::Value {
      while (auto xform = v.getDefiningOp<rock::TransformOp>())
        v = xform.getInput();
      return v;
    };
    mlir::Value filterBase = getBase(convIF.getFilter());
    mlir::Value inputBase = getBase(convIF.getInput());
    mlir::Value outputBase = getBase(convIF.getOutput());

    // Extract raw pointers
    auto extractPtr = [&](mlir::Value memref) -> mlir::Value {
      auto idx =
          memref::ExtractAlignedPointerAsIndexOp::create(fb, loc, memref);
      auto i64Val = arith::IndexCastOp::create(fb, loc, i64, idx);
      return LLVM::IntToPtrOp::create(fb, loc, ptr, i64Val, nullptr);
    };
    mlir::Value inputPtr = extractPtr(inputBase);
    mlir::Value filterPtr = extractPtr(filterBase);
    mlir::Value outputPtr = extractPtr(outputBase);

    // Scalar constant helpers
    auto ci32 = [&](int32_t v) -> mlir::Value {
      return arith::ConstantIntOp::create(fb, loc, i32, v);
    };
    auto ci64 = [&](int64_t v) -> mlir::Value {
      return arith::ConstantIntOp::create(fb, loc, i64, v);
    };
    auto cf32 = [&](float v) -> mlir::Value {
      return arith::ConstantOp::create(
          fb, loc, f32, builder.getFloatAttr(f32, static_cast<double>(v)));
    };
    auto ci8 = [&](int8_t v) -> mlir::Value {
      return arith::ConstantIntOp::create(fb, loc, i8, v);
    };
    auto nullPtr = [&]() -> mlir::Value {
      return LLVM::ZeroOp::create(fb, loc, ptr);
    };

    // Build a packed 232-byte argument buffer on the stack and launch
    // via mgpuLaunchKernel with HIP_LAUNCH_PARAM_BUFFER_POINTER.
    // This matches the exact byte layout the Winograd kernel expects.

    // Allocate 232 bytes on stack (as array of i8)
    auto arrTy = LLVM::LLVMArrayType::get(i8, 232);
    auto argBuf =
        LLVM::AllocaOp::create(fb, loc, ptr, arrTy, ci32(1), /*alignment=*/8);

    // Zero-initialize
    auto cst0_i8 = ci8(0);
    auto cst232 = ci32(232);
    LLVM::MemsetOp::create(fb, loc, argBuf, cst0_i8, cst232,
                           /*isVolatile=*/false);

    // Helper to store a value at a byte offset in the buffer
    auto storeAt = [&](mlir::Value val, int64_t offset) {
      auto gepIdx = ci64(offset);
      auto fieldPtr = LLVM::GEPOp::create(fb, loc, ptr, i8, argBuf,
                                          mlir::ValueRange{gepIdx});
      LLVM::StoreOp::create(fb, loc, val, fieldPtr);
    };

    auto &p = problem;
    auto &s = tensorStrides;

    // V2 ABI layout: write each field at its exact byte offset
    storeAt(ci32(p.N), 0);
    storeAt(ci32(p.C), 4);
    storeAt(ci32(p.H), 8);
    storeAt(ci32(p.W), 12);
    storeAt(ci32(p.K), 16);
    storeAt(ci32(selection->nGroups), 20);
    storeAt(ci64(flags64), 24);
    storeAt(inputPtr, 32);
    storeAt(filterPtr, 40);
    storeAt(outputPtr, 48);
    // 56: reserved (already zero)
    storeAt(ci32(p.R), 64);
    storeAt(ci32(p.S), 68);
    storeAt(ci32(p.padH), 72);
    storeAt(ci32(p.padW), 76);
    storeAt(ci32(p.outH), 80);
    storeAt(ci32(p.outW), 84);
    // 88: bias_addr (null, already zero)
    storeAt(cf32(1.0f), 96);  // alpha
    storeAt(cf32(0.0f), 100); // beta
    // 104-135: offsets (all zero, already zero)
    storeAt(ci32(s.d_N), 136);
    storeAt(ci32(s.d_C), 140);
    storeAt(ci32(s.d_H), 144);
    // 148: d_W reserved
    storeAt(ci32(s.f_K), 152);
    storeAt(ci32(s.f_C), 156);
    storeAt(ci32(s.f_R), 160);
    // 164: f_S reserved
    storeAt(ci32(s.o_N), 168);
    storeAt(ci32(s.o_K), 172);
    storeAt(ci32(s.o_H), 176);
    // 180: o_W reserved
    storeAt(ci32(p.groupCount), 184);
    storeAt(ci32(s.d_G), 188);
    storeAt(ci32(s.f_G), 192);
    storeAt(ci32(s.o_G), 196);
    // 200-203: activation(0), sync_limit(0), sync_period(0), reserved(0)
    // already zero

    // Build HIP_LAUNCH_PARAM extra array:
    // [BUFFER_POINTER(0x01), &argBuf, BUFFER_SIZE(0x02), &size, END(0x03)]
    auto extraArrTy = LLVM::LLVMArrayType::get(ptr, 5);
    auto extraArr = LLVM::AllocaOp::create(fb, loc, ptr, extraArrTy, ci32(1),
                                           /*alignment=*/8);

    // HIP_LAUNCH_PARAM_BUFFER_POINTER = (void*)0x01
    auto bufferPtrTag =
        LLVM::IntToPtrOp::create(fb, loc, ptr, ci64(0x01), nullptr);
    // HIP_LAUNCH_PARAM_BUFFER_SIZE = (void*)0x02
    auto bufferSizeTag =
        LLVM::IntToPtrOp::create(fb, loc, ptr, ci64(0x02), nullptr);
    // HIP_LAUNCH_PARAM_END = (void*)0x03
    auto endTag = LLVM::IntToPtrOp::create(fb, loc, ptr, ci64(0x03), nullptr);

    // Allocate a size_t for the buffer size
    auto sizeAlloc =
        LLVM::AllocaOp::create(fb, loc, ptr, i64, ci32(1), /*alignment=*/8);
    LLVM::StoreOp::create(fb, loc, ci64(232), sizeAlloc);

    // Fill extra array: [tag, buf, tag, &size, end]
    auto storeExtra = [&](mlir::Value val, int idx) {
      auto gepIdx = ci64(idx);
      auto elemPtr = LLVM::GEPOp::create(fb, loc, ptr, ptr, extraArr,
                                         mlir::ValueRange{gepIdx});
      LLVM::StoreOp::create(fb, loc, val, elemPtr);
    };
    storeExtra(bufferPtrTag, 0);
    storeExtra(argBuf, 1);
    storeExtra(bufferSizeTag, 2);
    storeExtra(sizeAlloc, 3);
    storeExtra(endTag, 4);

    // Emit the kernel launch directly via mgpu* runtime calls,
    // bypassing gpu.launch_func which can't handle packed kernel args.

    // Declare mgpu runtime functions
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto idxTy = i64; // 64-bit pointers on AMD GPUs

    auto getOrInsertFn = [&](StringRef name, mlir::Type resTy,
                             ArrayRef<mlir::Type> argTys) {
      auto fnTy = LLVM::LLVMFunctionType::get(resTy, argTys);
      if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
        return fn;
      OpBuilder::InsertionGuard guard(fb);
      fb.setInsertionPointToEnd(module.getBody());
      return LLVM::LLVMFuncOp::create(fb, loc, name, fnTy);
    };

    auto mgpuModuleLoadFn =
        getOrInsertFn("mgpuModuleLoad", ptr, {ptr, /*size=*/i64});
    auto mgpuModuleGetFunctionFn =
        getOrInsertFn("mgpuModuleGetFunction", ptr, {ptr, ptr});
    auto mgpuStreamCreateFn = getOrInsertFn("mgpuStreamCreate", ptr, {});
    auto mgpuStreamSyncFn =
        getOrInsertFn("mgpuStreamSynchronize", voidTy, {ptr});
    auto mgpuStreamDestroyFn =
        getOrInsertFn("mgpuStreamDestroy", voidTy, {ptr});
    auto mgpuLaunchKernelFn =
        getOrInsertFn("mgpuLaunchKernel", voidTy,
                      {ptr, idxTy, idxTy, idxTy, idxTy, idxTy, idxTy, i32, ptr,
                       ptr, ptr, i64});

    // Embed the HSACO binary as a global constant
    auto hsacoGlobal = [&]() -> mlir::Value {
      std::string globalName = selection->kernelName + "_binary";
      auto globalOp = module.lookupSymbol<LLVM::GlobalOp>(globalName);
      if (!globalOp) {
        OpBuilder::InsertionGuard guard(fb);
        fb.setInsertionPointToEnd(module.getBody());
        auto arrTy = LLVM::LLVMArrayType::get(i8, hsaco->size());
        globalOp = LLVM::GlobalOp::create(
            fb, loc, arrTy, /*isConstant=*/true, LLVM::Linkage::Internal,
            globalName,
            builder.getStringAttr(StringRef(hsaco->data(), hsaco->size())));
      }
      return LLVM::AddressOfOp::create(fb, loc, ptr, globalName);
    }();

    // Embed the kernel name as a global string
    auto kernelNameGlobal = [&]() -> mlir::Value {
      std::string globalName = selection->kernelName + "_name";
      auto globalOp = module.lookupSymbol<LLVM::GlobalOp>(globalName);
      if (!globalOp) {
        OpBuilder::InsertionGuard guard(fb);
        fb.setInsertionPointToEnd(module.getBody());
        std::string nameWithNull = selection->kernelName + '\0';
        auto arrTy = LLVM::LLVMArrayType::get(i8, nameWithNull.size());
        globalOp = LLVM::GlobalOp::create(fb, loc, arrTy, /*isConstant=*/true,
                                          LLVM::Linkage::Internal, globalName,
                                          builder.getStringAttr(nameWithNull));
      }
      return LLVM::AddressOfOp::create(fb, loc, ptr, globalName);
    }();

    // Load module, get function, create stream
    auto gpuModule = LLVM::CallOp::create(
        fb, loc, mgpuModuleLoadFn,
        mlir::ValueRange{hsacoGlobal, ci64(hsaco->size())});
    auto gpuFunc = LLVM::CallOp::create(
        fb, loc, mgpuModuleGetFunctionFn,
        mlir::ValueRange{gpuModule.getResult(), kernelNameGlobal});
    auto stream =
        LLVM::CallOp::create(fb, loc, mgpuStreamCreateFn, mlir::ValueRange{});

    // Launch with packed buffer via extra
    auto cstIdx = [&](int64_t v) -> mlir::Value {
      return LLVM::ConstantOp::create(fb, loc, idxTy,
                                      builder.getIntegerAttr(idxTy, v));
    };

    LLVM::CallOp::create(fb, loc, mgpuLaunchKernelFn,
                         mlir::ValueRange{
                             gpuFunc.getResult(), cstIdx(selection->gridSize),
                             cstIdx(1), cstIdx(1), cstIdx(selection->blockSize),
                             cstIdx(1), cstIdx(1),
                             ci32(0), // sharedMem = 0, KD handles LDS
                             stream.getResult(),
                             nullPtr(), // params = null
                             extraArr,  // extra = packed buffer config
                             ci64(0)    // paramsCount
                         });

    // Sync and cleanup
    LLVM::CallOp::create(fb, loc, mgpuStreamSyncFn,
                         mlir::ValueRange{stream.getResult()});
    LLVM::CallOp::create(fb, loc, mgpuStreamDestroyFn,
                         mlir::ValueRange{stream.getResult()});

    // Erase the rock.conv (already replaced by direct launch above)
    convOp->erase();

    // Remove kernel attr
    funcOp->removeAttr("kernel");
    funcOp->setAttr("block_size",
                    builder.getI32IntegerAttr(selection->blockSize));
    funcOp->setAttr("grid_size",
                    builder.getI32IntegerAttr(selection->gridSize));

    LLVM_DEBUG(llvm::dbgs()
               << "Winograd intercept: " << selection->kernelFile << " ("
               << hsaco->size() << " bytes) for " << archChip << "\n");
    return; // Done - Winograd kernel launched directly via mgpu runtime
  }

private:
  std::string triple;
  std::string chip;
  std::string features;
};

std::unique_ptr<OperationPass<ModuleOp>>
mlir::rock::createWinogradInterceptPass(
    const WinogradInterceptPassOptions &opts) {
  return std::make_unique<WinogradInterceptPass>(opts);
}
