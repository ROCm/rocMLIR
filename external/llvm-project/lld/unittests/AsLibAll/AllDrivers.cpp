//===- AllDrivers.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "gmock/gmock.h"

bool lldInvoke(std::vector<const char *> args) {
  args.push_back("--version");
  lld::SafeReturn s = lld::safeLldMain(args, llvm::outs(), llvm::errs());
  return !s.ret && s.canRunAgain;
}

TEST(AsLib, AllDrivers) {
  EXPECT_TRUE(lldInvoke({"ld.lld"}));
  EXPECT_TRUE(lldInvoke({"ld64.lld"}));
  EXPECT_TRUE(lldInvoke({"ld", "-m", "i386pe"})); // MinGW
  EXPECT_TRUE(lldInvoke({"lld-link"}));
  EXPECT_TRUE(lldInvoke({"wasm-ld"}));
}
