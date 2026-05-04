//===- QuickTuningDbTests.cpp - Tests for QuickTuningDb -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/QuickTuningDb.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/xxhash.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::rock;

//===----------------------------------------------------------------------===//
// Database invariants
//
// Guard against data regressions in the on-disk tables. Both invariants are
// preconditions for the binary searches in lookup() and resolveKey().
//===----------------------------------------------------------------------===//

TEST(QuickTuningDbInvariantsTest, IsSortedByKey) {
  EXPECT_TRUE(QuickTuningDb::isSortedByKey());
}

TEST(QuickTuningDbInvariantsTest, ProblemMapsAreSortedByHash) {
  EXPECT_TRUE(QuickTuningDb::problemMapsAreSortedByHash());
}

namespace {
class QuickTuningDbTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  Builder b{&ctx};
};

using ResolveKeyTest = QuickTuningDbTest;
} // namespace

//===----------------------------------------------------------------------===//
// resolveKey: fallback policy
//===----------------------------------------------------------------------===//

TEST_F(ResolveKeyTest, ExactMatch) {
  // Exact match returns itself.
  EXPECT_EQ("gfx942_conv_f16",
            QuickTuningDb::resolveKey("gfx942", KernelType::Conv,
                                      b.getF16Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, OldestRelative) {
  // gfx908 is the oldest available relative for gfx900.
  EXPECT_EQ("gfx908_conv_f16",
            QuickTuningDb::resolveKey("gfx900", KernelType::Conv,
                                      b.getF16Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, YoungestRelative) {
  // gfx1201 is the youngest available relative for gfx1900.
  EXPECT_EQ("gfx1201_conv_f16",
            QuickTuningDb::resolveKey("gfx1900", KernelType::Conv,
                                      b.getF16Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, OlderRelativeIsCloser) {
  // gfx949 is closer to gfx942 than to gfx950.
  EXPECT_EQ("gfx942_conv_f16",
            QuickTuningDb::resolveKey("gfx949", KernelType::Conv,
                                      b.getF16Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, YoungerRelativeIsCloser) {
  // gfx940 is closer to gfx942 than to gfx90a.
  EXPECT_EQ("gfx942_conv_f16",
            QuickTuningDb::resolveKey("gfx940", KernelType::Conv,
                                      b.getF16Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, PreferYoungerWhenEquidistant) {
  // gfx90a and gfx908 are equidistant from gfx909; the tiebreaker picks
  // the lexicographically larger one.
  EXPECT_EQ("gfx90a_conv_f16",
            QuickTuningDb::resolveKey("gfx909", KernelType::Conv,
                                      b.getF16Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, NoRelativesByPrefix) {
  // No relatives with matching prefix.
  EXPECT_EQ("", QuickTuningDb::resolveKey("gfx800", KernelType::Conv,
                                          b.getF16Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, NonAccelRestrictsToGfx1Family) {
  // Non-accel queries must yield a gfx1*+f32 sibling. gfx1201_gemm_f32 is
  // the lexicographically largest gfx1* candidate, so any target above
  // it (e.g. a hypothetical gfx2500) falls back to it.
  EXPECT_EQ("gfx1201_gemm_f32",
            QuickTuningDb::resolveKey("gfx2500", KernelType::Gemm,
                                      b.getF32Type(), /*isAccel=*/false));
}

TEST_F(ResolveKeyTest, NonAccelIgnoresInputDtype) {
  // The non-accel path forces the f32 suffix; querying with a different
  // dtype must still land on a gfx1*_<op>_f32 entry.
  StringRef key =
      QuickTuningDb::resolveKey("gfx2500", KernelType::Gemm, b.getF16Type(),
                                /*isAccel=*/false);
  EXPECT_TRUE(key.starts_with("gfx1"));
  EXPECT_TRUE(key.ends_with("_gemm_f32"));
}

TEST_F(ResolveKeyTest, AccelStaysWithinFamily) {
  // When isAccel is true the search is restricted to the same gfx family,
  // so a missing family yields no match instead of crossing over.
  EXPECT_EQ("", QuickTuningDb::resolveKey("gfx2500", KernelType::Gemm,
                                          b.getF32Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, Bf16FallsBackToF16) {
  // gfx942 has gfx942_conv_f16 but no gfx942_conv_bf16, and no other gfx9
  // arch carries _conv_bf16 either, so resolveKey must reuse the f16 entry.
  EXPECT_EQ("gfx942_conv_f16",
            QuickTuningDb::resolveKey("gfx942", KernelType::Conv,
                                      b.getBF16Type(), /*isAccel=*/true));
}

//===----------------------------------------------------------------------===//
// computeProblemKeyHash: byte-identity with the Python generator.
//
// The same goldens are pinned in pytest, so drift fails both suites.
//===----------------------------------------------------------------------===//

namespace {
constexpr uint64_t kHashGemmDefault = 0x8BB71834A431CBDDULL;
constexpr uint64_t kHashGemmTransposed = 0xBFD6CCD314CA3040ULL;
constexpr uint64_t kHashConvFwd = 0xA6F3626951158D16ULL;
constexpr uint64_t kHashAttentionPlain = 0xC75F3355EE11CD53ULL;

class ProblemKeyTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<rock::RockDialect, func::FuncDialect>();
  }

  // Returns the first rock op in the module parsed from `src`.
  Operation *parseFirstRockOp(StringRef src, OwningOpRef<ModuleOp> &moduleRef) {
    moduleRef = parseSourceString<ModuleOp>(src, &ctx);
    if (!moduleRef)
      return nullptr;
    Operation *found = nullptr;
    moduleRef->walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "rock") {
        found = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  }

  MLIRContext ctx;
};
} // namespace

TEST(ProblemKeyHashTest, XXH3MatchesPython) {
  // Confirms llvm::xxh3_64bits agrees with Python's xxh3_64_intdigest.
  EXPECT_EQ(0x9555E8555C62DCFDULL, llvm::xxh3_64bits("hello"));
}

TEST_F(ProblemKeyTest, GemmKeyDefault) {
  // TransA=False, TransB=False, G=1, M=1024, K=1024, N=1024.
  static constexpr StringLiteral src = R"MLIR(
    func.func @gemm(%a: memref<1x1024x1024xf32>, %b: memref<1x1024x1024xf32>,
                    %c: memref<1x1024x1024xf32>)
        attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
      rock.gemm %c = %a * %b features = none storeMethod = set
        : memref<1x1024x1024xf32> = memref<1x1024x1024xf32> *
                                    memref<1x1024x1024xf32>
      func.return
    }
  )MLIR";
  OwningOpRef<ModuleOp> module;
  Operation *op = parseFirstRockOp(src, module);
  ASSERT_NE(op, nullptr);
  auto h = QuickTuningDb::computeProblemKeyHash(op);
  ASSERT_TRUE(succeeded(h));
  EXPECT_EQ(*h, kHashGemmDefault);
}

TEST_F(ProblemKeyTest, GemmKeyTransposed) {
  // TransA=True, TransB=True, G=1, M=64, K=128, N=256.
  static constexpr StringLiteral src = R"MLIR(
    func.func @gemm(%a: memref<1x128x64xf32>, %b: memref<1x256x128xf32>,
                    %c: memref<1x64x256xf32>)
        attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
      rock.gemm %c = tr %a * tr %b features = none storeMethod = set
        : memref<1x64x256xf32> = memref<1x128x64xf32> * memref<1x256x128xf32>
      func.return
    }
  )MLIR";
  OwningOpRef<ModuleOp> module;
  Operation *op = parseFirstRockOp(src, module);
  ASSERT_NE(op, nullptr);
  auto h = QuickTuningDb::computeProblemKeyHash(op);
  ASSERT_TRUE(succeeded(h));
  EXPECT_EQ(*h, kHashGemmTransposed);
}

TEST_F(ProblemKeyTest, ConvFwdKey) {
  // fwd, kcyx + nchw + nkhw, N=1, C=64, H=14, W=14, K=128, Y=3, X=3,
  // dilations = strides = paddings = 1.
  static constexpr StringLiteral src = R"MLIR(
    func.func @conv(%filter: memref<1x128x64x3x3xf32>,
                    %input: memref<1x1x64x14x14xf32>,
                    %output: memref<1x1x128x12x12xf32>)
        attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
      rock.conv(%filter, %input, %output) features = none {
        filter_layout = ["g", "k", "c", "0", "1"],
        input_layout = ["ni", "gi", "ci", "0i", "1i"],
        output_layout = ["no", "go", "ko", "0o", "1o"],
        dilations = [1 : index, 1 : index],
        strides = [1 : index, 1 : index],
        padding = [1 : index, 1 : index, 1 : index, 1 : index]
      } : memref<1x128x64x3x3xf32>, memref<1x1x64x14x14xf32>,
          memref<1x1x128x12x12xf32>
      func.return
    }
  )MLIR";
  OwningOpRef<ModuleOp> module;
  Operation *op = parseFirstRockOp(src, module);
  ASSERT_NE(op, nullptr);
  auto h = QuickTuningDb::computeProblemKeyHash(op);
  ASSERT_TRUE(succeeded(h));
  EXPECT_EQ(*h, kHashConvFwd);
}

TEST_F(ProblemKeyTest, AttentionKeyPlain) {
  // G=1, seqLenQ=seqLenK=256, numHeadsQ=numHeadsKV=8, headDim=64; no
  // transposes / causal / LSE / scale / bias.
  static constexpr StringLiteral src = R"MLIR(
    func.func @attn(%q: memref<8x256x64xf32>, %k: memref<8x64x256xf32>,
                    %v: memref<8x256x64xf32>, %o: memref<8x256x64xf32>)
        attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
      rock.attention {
        qk = %q * %k : memref<8x256x64xf32>, memref<8x64x256xf32>
        %o = softmax(qk) * %v : memref<8x256x64xf32> -> memref<8x256x64xf32>
      } {
        features = #rock<GemmFeatures mfma>,
        firstGemmIndices = array<i64: 0>,
        numHeadsQ = 8 : i32,
        numHeadsKV = 8 : i32,
        splitKV = 1 : i32,
        storeMethod = #rock<StoreMethod set>
      }
      func.return
    }
  )MLIR";
  OwningOpRef<ModuleOp> module;
  Operation *op = parseFirstRockOp(src, module);
  ASSERT_NE(op, nullptr);
  auto h = QuickTuningDb::computeProblemKeyHash(op);
  ASSERT_TRUE(succeeded(h));
  EXPECT_EQ(*h, kHashAttentionPlain);
}

TEST_F(ProblemKeyTest, NullOpFails) {
  EXPECT_TRUE(failed(QuickTuningDb::computeProblemKeyHash(nullptr)));
}

TEST_F(ProblemKeyTest, UnsupportedOpFails) {
  static constexpr StringLiteral src = R"MLIR(
    func.func @nope() { func.return }
  )MLIR";
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(src, &ctx);
  ASSERT_TRUE(module);
  Operation *funcOp = &*module->getBody()->begin();
  EXPECT_TRUE(failed(QuickTuningDb::computeProblemKeyHash(funcOp)));
}
