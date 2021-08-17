//===- IsaNameParser.h - MLIR to C++ option parsing ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares isa name string parser
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_ROCM_ISANAMEPARSER_H_
#define MLIR_EXECUTIONENGINE_ROCM_ISANAMEPARSER_H_

#include "mlir/Support/LogicalResult.h"

#include <string>

namespace mlir {
class IsaNameParser {
public:
  IsaNameParser(const std::string &isaName);
  LogicalResult parseIsaName(std::string &chip, std::string &triple,
                             std::string &features);
  static LogicalResult parseArchName(const std::string &archName,
                                     std::string &chip, std::string &features);

private:
  std::string isaName;
};
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_ROCM_ISANAMEPARSER_H_
