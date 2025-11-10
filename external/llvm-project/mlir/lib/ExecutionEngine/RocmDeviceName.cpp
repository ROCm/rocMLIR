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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMapEntry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static constexpr const char kGcnDefaultTriple[] = "amdgcn-amd-amdhsa";

// This function converts the arch name to target features:
// sramecc+:xnack- to +sramecc,-xnack
static LogicalResult parseTargetFeatures(llvm::ArrayRef<llvm::StringRef> tokens,
                                         llvm::StringMap<bool> &features) {
  // Convert each feature and append to feature result
  for (auto &token : tokens) {
    char modifier = token.back();
    if (modifier != '+' && modifier != '-') {
      llvm::errs() << "Malformed token: must end with +/-.\n";
      return failure();
    }
    features.insert_or_assign(token.drop_back(), modifier == '+');
  }
  return success();
}

static void sortFeatures(const llvm::StringMap<bool> &features,
                         llvm::SmallVectorImpl<std::pair<llvm::StringRef, bool>> &out) {
  for (const llvm::StringMapEntry<bool> &feature : features) {
    out.emplace_back(feature.getKey(), feature.getValue());
  }
  std::sort(out.begin(), out.end());
}

LogicalResult RocmDeviceName::parse(llvm::StringRef devName) {
  triple = kGcnDefaultTriple;

  // tokenize on :
  llvm::SmallVector<llvm::StringRef, 8> tokens;
  devName.split(tokens, ':');

  // check for triple
  if (tokens.size() > 1) {
    llvm::SmallVector<llvm::StringRef, 8> tripleTokens;
    tokens.front().split(tripleTokens, '-');
    if (tripleTokens.size() == 3 && tripleTokens[0] == "amdgcn") {
      triple = tokens.front();
      tokens.erase(tokens.begin());
    }
  }

  // get chip name
  if (!tokens.empty()) {
    chip = tokens.front();
    tokens.erase(tokens.begin());
    if (chip.starts_with("gfx")) {
      // get features
      return parseTargetFeatures(tokens, features);
    }
  }

  return failure();
}

std::string RocmDeviceName::getFeaturesForBackend() const {
  std::string result;
  llvm::raw_string_ostream stream(result);
  // We don't sort these earlier to prevent the references into the map keys
  // from moving out from under us. They need to be sorted to ensure we have
  // the arch name in canonical form.
  llvm::SmallVector<std::pair<llvm::StringRef, bool>, 2> sortedFeatures;
  sortFeatures(features, sortedFeatures);
  llvm::interleave(
      sortedFeatures, stream,
      [&stream](const auto &pair) {
        stream << (pair.second ? "+" : "-") << pair.first;
      },
      ",");
  return result;
}

void RocmDeviceName::getFullName(llvm::SmallVectorImpl<char> &out) const {
  (llvm::Twine(triple) + ":" + llvm::Twine(chip)).toVector(out);
  llvm::raw_svector_ostream outStream(out);
  llvm::SmallVector<std::pair<llvm::StringRef, bool>, 2> sortedFeatures;
  sortFeatures(features, sortedFeatures);
  for (const auto &pair : sortedFeatures) {
    outStream << ':' << pair.first << (pair.second ? "+" : "-");
  }
}
