//===- RocmDeviceName.cpp - MLIR to C++ option parsing ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements isa name string splitter
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "llvm/Support/Error.h"

using namespace mlir;

static constexpr const char kGcnDefaultTriple[] = "amdgcn-amd-amdhsa";

// This function converts the arch name to target features:
// sramecc+:xnack- to +sramecc,-xnack
static LogicalResult parseTargetFeatures(ArrayRef<StringRef> tokens,
                                         SmallString<32> &gcnArchFeatures) {
  // Convert each feature and append to feature result
  bool first = true;
  SmallVector<SmallString<8>, 2> featureTokens;
  for (auto &token : tokens) {
    if (!first)
      gcnArchFeatures += ",";
    first = false;
    char modifier = token.back();
    if (modifier != '+' && modifier != '-') {
      llvm::errs() << "Malformed token: must end with +/-.\n";
      return failure();
    }
    gcnArchFeatures += modifier;
    gcnArchFeatures += token.substr(0, token.size() - 1);
  }
  return success();
}

RocmDeviceName::RocmDeviceName(StringRef devName)
    : status(failure()), triple(kGcnDefaultTriple) {
  // tokenize on :
  SmallVector<StringRef, 8> tokens;
  devName.split(tokens, ':');

  // check for triple
  if (tokens.size() > 1) {
    SmallVector<StringRef, 8> tripleTokens;
    tokens.front().split(tripleTokens, '-');
    if (tripleTokens.size() == 3 && tripleTokens[0] == "amdgcn") {
      triple = tokens.front();
      tokens.erase(tokens.begin());
    }
  }

  // get chip name
  if (tokens.size()) {
    chip = tokens.front();
    tokens.erase(tokens.begin());
    if (!chip.startswith("gfx"))
      return;
    // get features
    status = parseTargetFeatures(tokens, features);
  }
}

SmallString<256> RocmDeviceName::getFullName() const {
  SmallString<256> fullName = chip.str();
  if (features.size()) {
    fullName += ":";
    fullName += features;
  }
  return fullName;
}
