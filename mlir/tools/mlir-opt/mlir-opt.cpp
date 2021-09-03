//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Conversion/MIOpenPasses.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

// Defined in the test directory, no public header.
namespace mlir {
void registerConvertToTargetEnvPass();
void registerPassManagerTestPass();
void registerPrintOpAvailabilityPass();
void registerShapeFunctionTestPasses();
void registerSideEffectTestPasses();
void registerSliceAnalysisTestPass();
void registerSymbolTestPasses();
void registerTestAffineDataCopyPass();
void registerTestAffineLoopUnswitchingPass();
void registerTestAllReduceLoweringPass();
void registerTestFunc();
void registerTestGpuMemoryPromotionPass();
void registerTestLoopPermutationPass();
void registerTestMatchers();
void registerTestOperationEqualPass();
void registerTestPrintDefUsePass();
void registerTestPrintNestingPass();
void registerTestReducer();
void registerTestSpirvEntryPointABIPass();
void registerTestSpirvGLSLCanonicalizationPass();
void registerTestSpirvModuleCombinerPass();
void registerTestTraitsPass();
void registerTosaTestQuantUtilAPIPass();
void registerVectorizerTestPass();

namespace test {
void registerConvertCallOpPass();
void registerInliner();
void registerMemRefBoundCheck();
void registerPatternsTestPass();
void registerSimpleParametricTilingPass();
void registerTestAffineLoopParametricTilingPass();
void registerTestAliasAnalysisPass();
void registerTestCallGraphPass();
void registerTestConstantFold();
void registerTestConvVectorization();
void registerTestGpuSerializeToCubinPass();
void registerTestGpuSerializeToHsacoPass();
void registerTestConvertGPUKernelToCubinPass();
void registerTestConvertGPUKernelToHsacoPass();
void registerTestDataLayoutQuery();
void registerTestDecomposeCallGraphTypes();
void registerTestDiagnosticsPass();
void registerTestDominancePass();
void registerTestDynamicPipelinePass();
void registerTestExpandTanhPass();
void registerTestComposeSubView();
void registerTestGpuParallelLoopMappingPass();
void registerTestIRVisitorsPass();
void registerTestInterfaces();
void registerTestLinalgCodegenStrategy();
void registerTestLinalgControlFuseByExpansion();
void registerTestLinalgDistribution();
void registerTestLinalgElementwiseFusion();
void registerTestPushExpandingReshape();
void registerTestLinalgFusionTransforms();
void registerTestLinalgTensorFusionTransforms();
void registerTestLinalgTiledLoopFusionTransforms();
void registerTestLinalgGreedyFusion();
void registerTestLinalgHoisting();
void registerTestLinalgTileAndFuseSequencePass();
void registerTestLinalgTransforms();
void registerTestLivenessPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestLoopUnrollingPass();
void registerTestMathAlgebraicSimplificationPass();
void registerTestMathPolynomialApproximationPass();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestNumberOfBlockExecutionsPass();
void registerTestNumberOfOperationExecutionsPass();
void registerTestOpaqueLoc();
void registerTestPDLByteCodePass();
void registerTestPreparationPassWithAllowedMemrefResults();
void registerTestRecursiveTypesPass();
void registerTestSCFUtilsPass();
void registerTestVectorConversions();
void registerTosaPartitionPass();
} // namespace test
} // namespace mlir

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

#ifdef MLIR_INCLUDE_TESTS
void registerTestPasses() {
  registerConvertToTargetEnvPass();
  registerPassManagerTestPass();
  registerPrintOpAvailabilityPass();
  registerShapeFunctionTestPasses();
  registerSideEffectTestPasses();
  registerSliceAnalysisTestPass();
  registerSymbolTestPasses();
  registerTestAffineDataCopyPass();
  registerTestAffineLoopUnswitchingPass();
  registerTestAllReduceLoweringPass();
  registerTestFunc();
  registerTestGpuMemoryPromotionPass();
  registerTestLoopPermutationPass();
  registerTestMatchers();
  registerTestOperationEqualPass();
  registerTestPrintDefUsePass();
  registerTestPrintNestingPass();
  registerTestReducer();
  registerTestSpirvEntryPointABIPass();
  registerTestSpirvGLSLCanonicalizationPass();
  registerTestSpirvModuleCombinerPass();
  registerTestTraitsPass();
  registerVectorizerTestPass();
  registerTosaTestQuantUtilAPIPass();

  mlir::test::registerConvertCallOpPass();
  mlir::test::registerInliner();
  mlir::test::registerMemRefBoundCheck();
  mlir::test::registerPatternsTestPass();
  mlir::test::registerSimpleParametricTilingPass();
  mlir::test::registerTestAffineLoopParametricTilingPass();
  mlir::test::registerTestAliasAnalysisPass();
  mlir::test::registerTestCallGraphPass();
  mlir::test::registerTestConstantFold();
  mlir::test::registerTestDiagnosticsPass();
#if MLIR_CUDA_CONVERSIONS_ENABLED
  mlir::test::registerTestGpuSerializeToCubinPass();
#endif
#if MLIR_ROCM_CONVERSIONS_ENABLED
  mlir::test::registerTestGpuSerializeToHsacoPass();
#endif
  test::registerTestConvVectorization();
  test::registerTestDecomposeCallGraphTypes();
  test::registerTestDataLayoutQuery();
  test::registerTestDominancePass();
  test::registerTestDynamicPipelinePass();
  test::registerTestExpandTanhPass();
  test::registerTestComposeSubView();
  test::registerTestGpuParallelLoopMappingPass();
  test::registerTestIRVisitorsPass();
  test::registerTestInterfaces();
  test::registerTestLinalgCodegenStrategy();
  test::registerTestLinalgDistribution();
  test::registerTestLinalgElementwiseFusion();
  test::registerTestPushExpandingReshape();
  test::registerTestLinalgFusionTransforms();
  test::registerTestLinalgTensorFusionTransforms();
  test::registerTestLinalgTiledLoopFusionTransforms();
  test::registerTestLinalgGreedyFusion();
  test::registerTestLinalgHoisting();
  test::registerTestLinalgTileAndFuseSequencePass();
  test::registerTestLinalgTransforms();
  test::registerTestLivenessPass();
  test::registerTestLoopFusion();
  test::registerTestLoopMappingPass();
  test::registerTestLoopUnrollingPass();
  test::registerTestMathAlgebraicSimplificationPass();
  test::registerTestMathPolynomialApproximationPass();
  test::registerTestMemRefDependenceCheck();
  test::registerTestMemRefStrideCalculation();
  test::registerTestNumberOfBlockExecutionsPass();
  test::registerTestNumberOfOperationExecutionsPass();
  test::registerTestOpaqueLoc();
  test::registerTestPDLByteCodePass();
  test::registerTestRecursiveTypesPass();
  test::registerTestSCFUtilsPass();
  test::registerTestVectorConversions();
  test::registerTosaPartitionPass();
}
#endif

int main(int argc, char **argv) {
  registerAllPasses();
  mlir::registerMIOpenConversionPasses();
  miopen::registerPasses();
#ifdef MLIR_INCLUDE_TESTS
  registerTestPasses();
#endif
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<miopen::MIOpenDialect>();
  registry.insert<migraphx::MIGraphXDialect>();
#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
#endif
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry,
                        /*preloadDialectsInContext=*/false));
}
