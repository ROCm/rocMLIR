//===- InitRocMLIRDialects.h - rocMLIR Dialects Registration ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all our custom
// dialects and passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITROCMLIRDIALECTS_H_
#define MLIR_INITROCMLIRDIALECTS_H_

// rocMLIR includes
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/InitRocMLIRTarget.h"

// MLIR includes
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Dialect.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

// Below headers will allow registration to ROCm passes
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/TritonGPUConversion.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/InitAllPasses.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"

#include "triton/Tools/PluginUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"

// TODO: Re-enable when MHAL is updated for newer LLVM
// #include "mlir/InitMHALDialects.h"

namespace mlir {

inline void registerUpstreamDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<affine::AffineDialect,
                  arith::ArithDialect,
                  async::AsyncDialect,
                  bufferization::BufferizationDialect,
                  cf::ControlFlowDialect,
                  DLTIDialect,
                  index::IndexDialect,
                  func::FuncDialect,
                  LLVM::LLVMDialect,
                  linalg::LinalgDialect,
                  math::MathDialect,
                  memref::MemRefDialect,
                  scf::SCFDialect,
                  ub::UBDialect,
                  vector::VectorDialect,
                  tensor::TensorDialect,
                  tosa::TosaDialect>();
  // clang-format on

  // TODO: Re-enable GPU/AMDGPU/ROCDL dialects for Triton backend
  // These are handled by Triton now
  // registry.insert<amdgpu::AMDGPUDialect,
  //                 gpu::GPUDialect,
  //                 ROCDL::ROCDLDialect>();

  // Register bufferization hooks for rock interfaces
  rock::registerBufferizableOpInterfaceExternalModels(registry);

  // Register all dialect extensions.
  bufferization::registerTransformDialectExtension(registry);
  func::registerInlinerExtension(registry);

  // Register all conversions to LLVM extensions.
  arith::registerConvertArithToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);

  // Register all external models.
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerTilingInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerConvertVectorToLLVMInterface(registry);
  index::registerConvertIndexToLLVMInterface(registry);
}

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  // mlir::registerAllPasses();
  mlir::triton::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::triton::nvidia_gpu::registerTritonNvidiaGPUPasses();
  mlir::triton::instrument::registerTritonInstrumentPasses();
  mlir::triton::gluon::registerGluonPasses();
  // mlir::test::registerTestAliasPass();
  // mlir::test::registerTestAlignmentPass();
  // mlir::test::registerAMDTestAlignmentPass();
  // mlir::test::registerTestAllocationPass();
  // mlir::test::registerTestBufferRegionPass();
  // mlir::test::registerTestMembarPass();
  // mlir::test::registerTestLoopPeelingPass();
  // mlir::test::registerTestAMDGPUMembarPass();
  // mlir::test::registerTestTritonAMDGPURangeAnalysis();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerRelayoutTritonGPUPass();
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
  mlir::triton::registerConvertWarpSpecializeToLLVM();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::triton::registerAllocateSharedMemoryNvPass();
  mlir::registerLLVMDIScope();
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::NVVM::registerInlinerInterface(registry);
  mlir::registerLLVMDILocalVariable();

  // TritonAMDGPUToLLVM passes
  mlir::triton::registerAllocateAMDGPUSharedMemory();
  mlir::triton::registerTritonAMDGPUConvertWarpSpecializeToLLVM();
  mlir::triton::registerConvertTritonAMDGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  mlir::triton::registerConvertWarpPipeline();

  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);

  // TritonAMDGPUTransforms passes
  mlir::registerTritonAMDGPUAccelerateMatmul();
  mlir::registerTritonAMDGPUOptimizeEpilogue();
  mlir::registerTritonAMDGPUHoistLayoutConversions();
  mlir::registerTritonAMDGPUSinkLayoutConversions();
  mlir::registerTritonAMDGPUReorderInstructions();
  mlir::registerTritonAMDGPUBlockPingpong();
  mlir::registerTritonAMDGPUPipeline();
  mlir::registerTritonAMDGPUScheduleLoops();
  mlir::registerTritonAMDGPUCanonicalizePointers();
  mlir::registerTritonAMDGPUConvertToBufferOps();
  mlir::registerTritonAMDGPUInThreadTranspose();
  mlir::registerTritonAMDGPUCoalesceAsyncCopy();
  mlir::registerTritonAMDGPUUpdateAsyncWaitCount();
  mlir::registerTritonAMDGPUWarpPipeline();
  mlir::triton::registerTritonAMDGPUInsertInstructionSchedHints();
  mlir::triton::registerTritonAMDGPULowerInstructionSchedHints();
  mlir::registerTritonAMDFoldTrueCmpI();
  mlir::triton::amdgpu::registerTritonAMDGPUOptimizeDotOperands();

  // NVWS passes
  mlir::triton::registerNVWSTransformsPasses();

  // NVGPU transform passes
  mlir::registerNVHopperTransformsPasses();

  // Proton passes
  // mlir::test::proton::registerTestScopeIdAllocationPass();
  mlir::triton::proton::registerConvertProtonToProtonGPU();
  mlir::triton::proton::gpu::registerConvertProtonNvidiaGPUToLLVM();
  mlir::triton::proton::gpu::registerConvertProtonAMDGPUToLLVM();
  mlir::triton::proton::gpu::registerAllocateProtonSharedMemoryPass();
  mlir::triton::proton::gpu::registerAllocateProtonGlobalScratchBufferPass();
  mlir::triton::proton::gpu::registerScheduleBufferStorePass();
  mlir::triton::proton::gpu::registerAddSchedBarriersPass();

  // Plugin passes
  if (std::string filename =
          mlir::triton::tools::getStrEnv("TRITON_PASS_PLUGIN_PATH");
      !filename.empty()) {

    TritonPlugin TP(filename);
    std::vector<const char *> passNames;
    if (auto result = TP.getPassHandles(passNames); !result)
      llvm::report_fatal_error(result.takeError());

    for (const char *passName : passNames)
      if (auto result = TP.registerPass(passName); !result)
        llvm::report_fatal_error(result.takeError());
  }

  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
      mlir::triton::gpu::TritonGPUDialect,
      mlir::triton::instrument::TritonInstrumentDialect,
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
      mlir::triton::nvgpu::NVGPUDialect, mlir::triton::nvws::NVWSDialect,
      mlir::triton::amdgpu::TritonAMDGPUDialect,
      mlir::triton::proton::ProtonDialect,
      mlir::triton::proton::gpu::ProtonGPUDialect, mlir::ROCDL::ROCDLDialect,
      mlir::triton::gluon::GluonDialect>();
}

// Add all the MLIR dialects to the provided registry.
inline void registerRocMLIRDialects(DialectRegistry &registry) {
  // Register rocMLIR specific dialects
  registry.insert<rock::RockDialect, migraphx::MIGraphXDialect>();

  // TODO: Re-enable when MHAL is updated for newer LLVM
  // registerMHALDialects(registry);

  // Register auxiliary Upstream dialects
  registerUpstreamDialects(registry);

  // Register Triton dialects
  registerTritonDialects(registry);

  // Register the target serialization interface
  registerRocTarget(registry);
}

} // namespace mlir

#endif // MLIR_INITROCMLIRDIALECTS_H_
