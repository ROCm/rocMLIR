//===- RockTuningTests.cpp - Tests for the tuning space API ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

TEST(TuningGetParamTest, EmptyTuningRange) {
  // An empty tuning range is expected whenever an operation has no quick-tuning
  // entries
  TuningParamSet tuningSpace;
  ParamEntry paramEntry;

  EXPECT_FALSE(tuningGetParam(&tuningSpace, 0, &paramEntry));
  EXPECT_FALSE(tuningGetParam(&tuningSpace, 1, &paramEntry));
}
