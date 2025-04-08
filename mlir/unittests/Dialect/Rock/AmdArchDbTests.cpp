//===- AmdArchDbTests.cpp - Tests for the AMD arch database
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/AmdArchDb.h"

#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"

using namespace mlir::rock;

TEST(AmdArchDbTest, NativeArch) {
  auto info = lookupArchInfo("native:1");
  EXPECT_EQ(info.totalVGPRPerEU, 512);
}
