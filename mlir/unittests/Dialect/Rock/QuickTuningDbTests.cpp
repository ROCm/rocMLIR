//===- QuickTuningDbTests.cpp - Tests for QuickTuningDb -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
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

#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

//===- Database invariants ------------------------------------------------===//
// Preconditions for the binary searches in lookup() and resolveKey().

TEST(QuickTuningDb, IsSortedByKey) {
  EXPECT_TRUE(QuickTuningDb::isSortedByKey());
}

TEST(QuickTuningDb, ProblemMapsAreSortedByHash) {
  EXPECT_TRUE(QuickTuningDb::problemMapsAreSortedByHash());
}

namespace {
class QuickTuningDbTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<rock::RockDialect, func::FuncDialect>();
  }

  // Parses `ir` (without op verification -- we only need shape metadata) and
  // returns the first op inside the first func.func body, or nullptr.
  Operation *parse(StringRef ir) {
    module = parseSourceString<ModuleOp>(
        ir, ParserConfig(&ctx, /*verifyAfterParse=*/false));
    if (!module || module->getBody()->empty())
      return nullptr;
    auto func = dyn_cast<func::FuncOp>(&module->getBody()->front());
    if (!func || func.getBody().empty() || func.getBody().front().empty())
      return nullptr;
    return &func.getBody().front().front();
  }

  // Parses `ir`, hashes the first rock op, and checks the result matches
  // `expected`. Goldens are mirrored in test_quickTuningGen.py.
  void expectKeyHash(StringRef ir, uint64_t expected) {
    Operation *op = parse(ir);
    ASSERT_NE(op, nullptr);
    auto h = QuickTuningDb::computeProblemKeyHash(op);
    ASSERT_TRUE(succeeded(h));
    EXPECT_EQ(*h, expected);
  }

  MLIRContext ctx;
  Builder b{&ctx};
  OwningOpRef<ModuleOp> module;
};

using ResolveKeyTest = QuickTuningDbTest;
using ProblemKeyTest = QuickTuningDbTest;
} // namespace

//===- resolveKey: fallback policy ---------------------------------------===//

TEST_F(ResolveKeyTest, ExactMatch) {
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
  // gfx90a and gfx908 are equidistant from gfx909; tiebreak picks the
  // lexicographically larger one.
  EXPECT_EQ("gfx90a_conv_f16",
            QuickTuningDb::resolveKey("gfx909", KernelType::Conv,
                                      b.getF16Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, NoRelativesByPrefix) {
  EXPECT_EQ("", QuickTuningDb::resolveKey("gfx800", KernelType::Conv,
                                          b.getF16Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, NonAccelRestrictsToGfx1Family) {
  // Non-accel queries must yield a gfx1*+f32 sibling. gfx1201_gemm_f32 is the
  // lex-largest gfx1* candidate, so any target above it falls back to it.
  EXPECT_EQ("gfx1201_gemm_f32",
            QuickTuningDb::resolveKey("gfx2500", KernelType::Gemm,
                                      b.getF32Type(), /*isAccel=*/false));
}

TEST_F(ResolveKeyTest, NonAccelIgnoresInputDtype) {
  // Non-accel forces the f32 suffix; another input dtype must still land on
  // a gfx1*_<op>_f32 entry.
  StringRef key =
      QuickTuningDb::resolveKey("gfx2500", KernelType::Gemm, b.getF16Type(),
                                /*isAccel=*/false);
  EXPECT_TRUE(key.starts_with("gfx1"));
  EXPECT_TRUE(key.ends_with("_gemm_f32"));
}

TEST_F(ResolveKeyTest, AccelStaysWithinFamily) {
  // isAccel restricts the search to the same gfx family; a missing family
  // yields no match instead of crossing over.
  EXPECT_EQ("", QuickTuningDb::resolveKey("gfx2500", KernelType::Gemm,
                                          b.getF32Type(), /*isAccel=*/true));
}

TEST_F(ResolveKeyTest, Bf16FallsBackToF16) {
  // gfx942 has gfx942_conv_f16 but no gfx942_conv_bf16, and no other gfx9
  // arch carries _conv_bf16, so resolveKey must reuse the f16 entry.
  EXPECT_EQ("gfx942_conv_f16",
            QuickTuningDb::resolveKey("gfx942", KernelType::Conv,
                                      b.getBF16Type(), /*isAccel=*/true));
}

//===- computeProblemKeyHash: byte-identity with the Python generator -----===//

TEST_F(ProblemKeyTest, GemmKeyDefault) {
  // TransA=False, TransB=False, G=1, M=1024, K=1024, N=1024.
  expectKeyHash(R"MLIR(
    func.func @gemm(%a: memref<1x1024x1024xf32>, %b: memref<1x1024x1024xf32>,
                    %c: memref<1x1024x1024xf32>)
        attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
      rock.gemm %c = %a * %b features = none storeMethod = set
        : memref<1x1024x1024xf32> = memref<1x1024x1024xf32> *
                                    memref<1x1024x1024xf32>
      func.return
    }
  )MLIR",
                0x8BB71834A431CBDDULL);
}

TEST_F(ProblemKeyTest, GemmKeyTransposed) {
  // TransA=True, TransB=True, G=1, M=64, K=128, N=256.
  expectKeyHash(R"MLIR(
    func.func @gemm(%a: memref<1x128x64xf32>, %b: memref<1x256x128xf32>,
                    %c: memref<1x64x256xf32>)
        attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
      rock.gemm %c = tr %a * tr %b features = none storeMethod = set
        : memref<1x64x256xf32> = memref<1x128x64xf32> * memref<1x256x128xf32>
      func.return
    }
  )MLIR",
                0xBFD6CCD314CA3040ULL);
}

TEST_F(ProblemKeyTest, ConvFwdKey) {
  // fwd, kcyx + nchw + nkhw, N=1, C=64, H=14, W=14, K=128, Y=3, X=3,
  // dilations = strides = paddings = 1.
  expectKeyHash(R"MLIR(
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
  )MLIR",
                0xA6F3626951158D16ULL);
}

TEST_F(ProblemKeyTest, ConvBwdDataKey) {
  // Same problem signature as fwd, but direction = "bwd".
  expectKeyHash(R"MLIR(
    func.func @conv(%filter: memref<1x128x64x3x3xf32>,
                    %input: memref<1x1x64x14x14xf32>,
                    %output: memref<1x1x128x12x12xf32>)
        attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
      rock.conv_bwd_data(%filter, %input, %output) features = none {
        filter_layout = ["g", "k", "c", "0", "1"],
        kernelId = 0 : index,
        usesV4R1 = false,
        input_layout = ["ni", "gi", "ci", "0i", "1i"],
        output_layout = ["no", "go", "ko", "0o", "1o"],
        dilations = [1 : index, 1 : index],
        strides = [1 : index, 1 : index],
        padding = [1 : index, 1 : index, 1 : index, 1 : index]
      } : memref<1x128x64x3x3xf32>, memref<1x1x64x14x14xf32>,
          memref<1x1x128x12x12xf32>
      func.return
    }
  )MLIR",
                0x61DCF2C43198890DULL);
}

TEST_F(ProblemKeyTest, ConvBwdWeightKey) {
  expectKeyHash(R"MLIR(
    func.func @conv(%filter: memref<1x128x64x3x3xf32>,
                    %input: memref<1x1x64x14x14xf32>,
                    %output: memref<1x1x128x12x12xf32>)
        attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
      rock.conv_bwd_weight(%filter, %input, %output) features = none {
        filter_layout = ["g", "k", "c", "0", "1"],
        numCU = 64 : i32,
        input_layout = ["ni", "gi", "ci", "0i", "1i"],
        output_layout = ["no", "go", "ko", "0o", "1o"],
        dilations = [1 : index, 1 : index],
        strides = [1 : index, 1 : index],
        padding = [1 : index, 1 : index, 1 : index, 1 : index]
      } : memref<1x128x64x3x3xf32>, memref<1x1x64x14x14xf32>,
          memref<1x1x128x12x12xf32>
      func.return
    }
  )MLIR",
                0xDB5C922433670198ULL);
}

TEST_F(ProblemKeyTest, AttentionKeyPlain) {
  // G=1, seqLenQ=seqLenK=256, numHeadsQ=numHeadsKV=8, headDim=64; no
  // transposes / causal / LSE / scale / bias.
  expectKeyHash(R"MLIR(
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
  )MLIR",
                0xC75F3355EE11CD53ULL);
}

TEST_F(ProblemKeyTest, AttentionKeyI8WithScale) {
  // i8 queries/keys with 3 preSoftmaxElemWiseInputs: the first 2 are the
  // implicit quantization-scale prefix (Q-scale, K-scale), the 3rd is the
  // attention scale. Exercises quantPrefix=2 and scaling=1, so the resulting
  // key carries WithAttnScale=1, WithAttnBias=0.
  expectKeyHash(R"MLIR(
    func.func @attn(%q: memref<8x256x64xi8>, %k: memref<8x64x256xi8>,
                    %v: memref<8x256x64xf16>, %o: memref<8x256x64xf16>,
                    %qs: memref<1xf32>, %ks: memref<1xf32>,
                    %sc: memref<1xf32>)
        attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
      rock.attention {
        qk = %q * %k : memref<8x256x64xi8>, memref<8x64x256xi8>
        qk = elementwise otherIns(%qs, %ks, %sc :
                                   memref<1xf32>, memref<1xf32>, memref<1xf32>) {
        ^bb0(%qScale: memref<1xf32>, %kScale: memref<1xf32>,
             %scale: memref<1xf32>, %gemm0_out: memref<8x256x256xf32>,
             %out: memref<8x256x256xf32>):
          rock.yield
        }
        %o = softmax(qk) * %v : memref<8x256x64xf16> -> memref<8x256x64xf16>
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
  )MLIR",
                0x24A5C1C837083EABULL);
}

TEST_F(ProblemKeyTest, UnsupportedOpFails) {
  // nullptr and non-rock ops both miss every dispatch arm.
  EXPECT_TRUE(failed(QuickTuningDb::computeProblemKeyHash(nullptr)));
  Operation *op = parse("func.func @nope() { func.return }");
  ASSERT_NE(op, nullptr);
  EXPECT_TRUE(failed(QuickTuningDb::computeProblemKeyHash(op)));
}
