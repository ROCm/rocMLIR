//===- MHAL.h - MHAL MLIR Dialect ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MHAL attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MHAL_IR_MHAL_H_
#define MLIR_MHAL_IR_MHAL_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
//  MHAL Dialect
//===----------------------------------------------------------------------===//

namespace mlir {
namespace mhal {
template <typename T>
class iter_one
    : public llvm::iterator_facade_base<iter_one<T>, std::forward_iterator_tag,
                                        T> {
  T &obj;
  int idx = 0;

public:
  iter_one(T &_s, int _idx = 0) : obj(_s), idx(_idx) {}
  iter_one(const iter_one &) = default;
  T &operator*() const {
    assert(idx == 0);
    return obj;
  }
  iter_one &operator++() {
    ++idx;
    return *this;
  }
  bool operator==(const iter_one &that) const {
    return obj == that.obj && idx == that.idx;
  }
};
} // namespace mhal
} // namespace mlir

#include "mlir/Dialect/MHAL/IR/MHALDialect.h.inc"
#include "mlir/Dialect/MHAL/IR/MHALTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MHAL/IR/MHALAttrDefs.h.inc"

#endif // MLIR_MHAL_IR_MHAL_H_
