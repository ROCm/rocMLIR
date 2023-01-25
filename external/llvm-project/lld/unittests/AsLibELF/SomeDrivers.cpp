//===- SomeDrivers.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "gmock/gmock.h"

bool lldInvoke(const char *lldExe) {
  std::vector<const char *> args{lldExe, "--version"};
  lld::SafeReturn s = lld::safeLldMain(args, llvm::outs(), llvm::errs());
  return !s.ret && s.canRunAgain;
}

TEST(AsLib, AllDrivers) {
  EXPECT_TRUE(lldInvoke("ld.lld")); // ELF
  // These drivers are not linked in this unit test.
  EXPECT_FALSE(lldInvoke("ld64.lld")); // ELF
  EXPECT_FALSE(lldInvoke("lld-link")); // COFF
  EXPECT_FALSE(lldInvoke("wasm-ld"));  // Wasm
}
