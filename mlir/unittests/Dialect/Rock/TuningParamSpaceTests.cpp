//===- TuningParamSpaceTests.cpp - Tests for Rock tuning space generation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {

static constexpr llvm::StringLiteral kNonAccelGemmModule = R"mlir(
module {
  func.func @non_accel_gemm(%a: memref<1x384x3072xf16>, %b: memref<1x3072x768xf16>, %c: memref<1x384x768xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx900"} {
    rock.gemm %c = %a * %b features = none storeMethod = set {arch = "amdgcn-amd-amdhsa:gfx900", perf_config = "v3:32,64,8,32,16,8,1,1,2,1,1"} : memref<1x384x768xf16> = memref<1x384x3072xf16> * memref<1x3072x768xf16>
    return
  }
}
)mlir";

static constexpr llvm::StringLiteral kAccelGemmModule = R"mlir(
module {
  func.func @accel_gemm(%a: memref<1x384x3072xf16>, %b: memref<1x3072x768xf16>, %c: memref<1x384x768xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    rock.gemm %c = %a * %b features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", perf_config = "v4:64,64,8,32,32,16,4,4,1,2,0,0,1,1"} : memref<1x384x768xf16> = memref<1x384x3072xf16> * memref<1x3072x768xf16>
    return
  }
}
)mlir";

class TuningParamSpaceTest : public ::testing::Test {
public:
  TuningParamSpaceTest() {
    DialectRegistry registry;
    registry.insert<func::FuncDialect, memref::MemRefDialect, rock::RockDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

protected:
  MLIRContext context;
};

TEST_F(TuningParamSpaceTest, GreedyFallsBackToExhaustiveForNonAccelGemm) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kNonAccelGemmModule, &context);
  ASSERT_TRUE(module) << "Failed to parse non-accel test module";

  TuningParamSpaceSettings settings{0, ""};
  std::unique_ptr<TuningParamSet> tuningSpace(
      createTunableParamSpace(*module, TuningParamSetKind::Greedy, settings));
  ASSERT_TRUE(tuningSpace);
  EXPECT_FALSE(tuningSpace->tuningRange.empty());
  EXPECT_EQ(tuningSpace->effectiveKind, TuningParamSetKind::Exhaustive);
  EXPECT_EQ(getNumberOfIterations(tuningSpace->effectiveKind), 1u);
  EXPECT_FALSE(needToUpdateBest(tuningSpace->effectiveKind));
}

TEST_F(TuningParamSpaceTest, GreedyRemainsGreedyForAccelGemm) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kAccelGemmModule, &context);
  ASSERT_TRUE(module) << "Failed to parse accel test module";

  TuningParamSpaceSettings settings{0, ""};
  std::unique_ptr<TuningParamSet> tuningSpace(
      createTunableParamSpace(*module, TuningParamSetKind::Greedy, settings));
  ASSERT_TRUE(tuningSpace);
  EXPECT_FALSE(tuningSpace->tuningRange.empty());
  EXPECT_EQ(tuningSpace->effectiveKind, TuningParamSetKind::Greedy);
  EXPECT_EQ(getNumberOfIterations(tuningSpace->effectiveKind), 3u);
  EXPECT_TRUE(needToUpdateBest(tuningSpace->effectiveKind));
}

} // namespace
