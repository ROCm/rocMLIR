//===- IsaNameSplitter.cpp - MLIR to C++ option parsing ---------------===//
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

#include "mlir/Dialect/MIOpen/utility/IsaNameSplitter.h"
#include "llvm/Support/Error.h"

#include <cstring>
#include <numeric>

using namespace mlir;

static constexpr const char kGcnArchDelimiter[] = ":";

static LogicalResult getTripleFromIsaName(const std::string &isaName,
                                          std::string &triple) {
  std::size_t firstSeperatorLoc = isaName.find(kGcnArchDelimiter);
  if (firstSeperatorLoc == std::string::npos) {
    return failure();
  }
  triple = isaName.substr(0, firstSeperatorLoc);
  return success();
}

static std::string getChipFromArchName(const std::string &gcnArchName) {
  std::size_t firstSeperatorLoc = gcnArchName.find(kGcnArchDelimiter);
  if (firstSeperatorLoc == std::string::npos) {
    return gcnArchName;
  }

  return gcnArchName.substr(0, firstSeperatorLoc);
}

// This function converts the arch name to target features:
// sramecc+:xnack- to +sramecc,-xnack
static LogicalResult parseTargetFeatures(std::string &gcnArchFeatures) {
  // First step: put each feature name to the vector
  std::string token;
  SmallVector<std::string, 2> featureTokens;
  auto convertFeatureToken =
      [](std::string &token) -> llvm::Expected<std::string> {
    if (token.back() != '+' && token.back() != '-') {
      return llvm::make_error<llvm::StringError>(
          "Malformed token: must end with +/-.",
          llvm::inconvertibleErrorCode());
    }
    token.insert(token.begin(), token.back());
    token.pop_back();
    return token;
  };

  size_t len = strlen(kGcnArchDelimiter);
  size_t featureStart = 0;
  size_t featureEnd = 0;
  for (size_t found = 0; featureStart < gcnArchFeatures.size(); ++found) {
    found = gcnArchFeatures.find(kGcnArchDelimiter, found);
    if (found == std::string::npos) {
      featureEnd = gcnArchFeatures.size() - 1;
    } else {
      featureEnd = found - len;
    }
    if (featureStart <= featureEnd) {
      std::string token =
          gcnArchFeatures.substr(featureStart, featureEnd - featureStart + 1);
      auto tokenOrErr = convertFeatureToken(token);
      if (tokenOrErr) {
        featureTokens.push_back(*tokenOrErr);
      } else {
        return failure();
      }
    }

    featureStart = featureEnd + len + 1;
  }

  // Second step: join processed token back to feature string
  const static std::string gcnFeatureDelimiter = ",";
  gcnArchFeatures = std::accumulate(
      featureTokens.begin(), featureTokens.end(), std::string(),
      [](const std::string &features, const std::string &token) {
        return features.empty() ? token
                                : features + gcnFeatureDelimiter + token;
      });
  return success();
}

IsaNameSplitter::IsaNameSplitter(const std::string &isa) : isaName(isa) {}

LogicalResult IsaNameSplitter::parseIsaName(std::string &chip,
                                            std::string &triple,
                                            std::string &features) {
  size_t len = strlen(kGcnArchDelimiter);
  auto status = getTripleFromIsaName(isaName, triple);
  if (status.failed()) {
    return failure();
  }
  std::string archName = isaName.substr(triple.size() + len);
  return parseArchName(archName, chip, features);
}

LogicalResult IsaNameSplitter::parseArchName(const std::string &archName,
                                             std::string &chip,
                                             std::string &features) {
  size_t len = strlen(kGcnArchDelimiter);
  chip = getChipFromArchName(archName);

  if (archName == chip) {
    features = "";
  } else {
    features = archName.substr(chip.size() + len);
    auto status = parseTargetFeatures(features);
    if (status.failed()) {
      return failure();
    }
  }

  return success();
}
