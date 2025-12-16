//===- ParamLookupTableTests.cpp - Tests for Tuning Params Lookup ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

TEST(FindFallbackTest, ExactMatch) {
  // Exact match should return itself
  EXPECT_EQ(
      "gfx942_conv_f16",
      ParamLookupTable<AccelGemmParamsAttr>::findFallback("gfx942_conv_f16"));
}

TEST(FindFallbackTest, OldestRelative) {
  // gfx908 is the oldest available relative for gfx900
  EXPECT_EQ(
      "gfx908_conv_f16",
      ParamLookupTable<AccelGemmParamsAttr>::findFallback("gfx900_conv_f16"));
}

TEST(FindFallbackTest, YoungestRelative) {
  // gfx1201 is the youngest available relative for gfx1900
  EXPECT_EQ(
      "gfx1201_conv_f16",
      ParamLookupTable<AccelGemmParamsAttr>::findFallback("gfx1900_conv_f16"));
}

TEST(FindFallbackTest, OlderRelativeIsCloser) {
  // gfx949 is closer to gfx942 than gfx950
  EXPECT_EQ(
      "gfx942_conv_f16",
      ParamLookupTable<AccelGemmParamsAttr>::findFallback("gfx949_conv_f16"));
}

TEST(FindFallbackTest, YoungerRelativeIsCloser) {
  // gfx940 is closer to gfx942 than gfx90a
  EXPECT_EQ(
      "gfx942_conv_f16",
      ParamLookupTable<AccelGemmParamsAttr>::findFallback("gfx940_conv_f16"));
}

TEST(FindFallbackTest, PreferYoungerWhenEquidistant) {
  // gfx90a and gfx908 are equidistant to gfx909, prefer younger gfx90a
  EXPECT_EQ(
      "gfx90a_conv_f16",
      ParamLookupTable<AccelGemmParamsAttr>::findFallback("gfx909_conv_f16"));
}

TEST(FindFallbackTest, NoRelativesByPrefix) {
  // No relatives with matching prefix
  EXPECT_EQ("", ParamLookupTable<AccelGemmParamsAttr>::findFallback(
                    "gfx800_conv_f16"));
}

TEST(FindFallbackTest, NoRelativesBySuffix) {
  // No relatives with matching suffix
  EXPECT_EQ("", ParamLookupTable<AccelGemmParamsAttr>::findFallback(
                    "gfx942_op_type"));
}

TEST(FindFallbackTest, AnyGfxForNonAccel) {
  // Any gfx version is acceptable for non-accelerated operations
  EXPECT_EQ(
      "gfx1201_gemm_f32",
      ParamLookupTable<GeneralGemmParamsAttr>::findFallback("gfx942_gemm_f32"));
}
