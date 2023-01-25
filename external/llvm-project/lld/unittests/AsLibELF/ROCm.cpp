//===- ROCm.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "gmock/gmock.h"
#include <algorithm>

// Implement drivers that we don't link/need in this test.
LLD_IMPLEMENT_SHALLOW_DRIVER(coff)
LLD_IMPLEMENT_SHALLOW_DRIVER(mingw)
LLD_IMPLEMENT_SHALLOW_DRIVER(macho)
LLD_IMPLEMENT_SHALLOW_DRIVER(wasm)

static std::string expand(const char *path) {
  llvm::StringRef thisFile = llvm::sys::path::parent_path(__FILE__);
  std::string expanded = path;
  if (llvm::StringRef(path).contains("%")) {
    expanded.replace(expanded.find("%S"), 2, thisFile.data(), thisFile.size());
  }
  return expanded;
}

bool lldInvoke(const char *inPath, const char *outPath) {
  std::vector<const char *> args{"ld.lld", "-shared", inPath, "-o", outPath};
  lld::SafeReturn s = lld::safeLldMain(args, llvm::outs(), llvm::errs());
  return !s.ret && s.canRunAgain;
}

static bool runLinker(const char *path) {
  // Create a temp file for HSA code object.
  int tempHsacoFD = -1;
  llvm::SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFD,
                                         tempHsacoFilename)) {
    return false;
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);
  // Invoke lld. Expect a true return value from lld.
  std::string expandedPath = expand(path);
  if (!lldInvoke(expandedPath.data(), tempHsacoFilename.c_str())) {
    llvm::errs() << "Failed to link: " << expandedPath << "\n";
    return false;
  }
  return true;
}

TEST(AsLib, ROCm) {
  EXPECT_TRUE(runLinker("%S/Inputs/kernel1.o"));
  EXPECT_TRUE(runLinker("%S/Inputs/kernel2.o"));
  EXPECT_TRUE(runLinker("%S/Inputs/kernel1.o"));
  EXPECT_TRUE(runLinker("%S/Inputs/kernel2.o"));
}
