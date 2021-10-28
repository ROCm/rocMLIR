//===- RegionUtils.h - Region-related transformation utilities --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_REGIONUTILS_H_
#define MLIR_TRANSFORMS_REGIONUTILS_H_

#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SetVector.h"

namespace mlir {
class RewriterBase;

/// This class contains the information for comparing the equivalencies of two
/// blocks. Blocks are considered equivalent if they contain the same operations
/// in the same order. The only allowed divergence is for operands that come
/// from sources outside of the parent block, i.e. the uses of values produced
/// within the block must be equivalent.
///   e.g.,
/// Equivalent:
///  ^bb1(%arg0: i32)
///    return %arg0, %foo : i32, i32
///  ^bb2(%arg1: i32)
///    return %arg1, %bar : i32, i32
/// Not Equivalent:
///  ^bb1(%arg0: i32)
///    return %foo, %arg0 : i32, i32
///  ^bb2(%arg1: i32)
///    return %arg1, %bar : i32, i32
struct BlockEquivalenceData {
  BlockEquivalenceData(Block *block);

  /// Return the order index for the given value that is within the block of
  /// this data.
  unsigned getOrderOf(Value value) const;

  /// The block this data refers to.
  Block *block;
  /// A hash value for this block.
  llvm::hash_code hash;
  /// A map of result producing operations to their relative orders within this
  /// block. The order of an operation is the number of defined values that are
  /// produced within the block before this operation.
  DenseMap<Operation *, unsigned> opOrderIndex;
};


/// Check if all values in the provided range are defined above the `limit`
/// region.  That is, if they are defined in a region that is a proper ancestor
/// of `limit`.
template <typename Range>
bool areValuesDefinedAbove(Range values, Region &limit) {
  for (Value v : values)
    if (!v.getParentRegion()->isProperAncestor(&limit))
      return false;
  return true;
}

/// Replace all uses of `orig` within the given region with `replacement`.
void replaceAllUsesInRegionWith(Value orig, Value replacement, Region &region);

/// Calls `callback` for each use of a value within `region` or its descendants
/// that was defined at the ancestors of the `limit`.
void visitUsedValuesDefinedAbove(Region &region, Region &limit,
                                 function_ref<void(OpOperand *)> callback);

/// Calls `callback` for each use of a value within any of the regions provided
/// that was defined in one of the ancestors.
void visitUsedValuesDefinedAbove(MutableArrayRef<Region> regions,
                                 function_ref<void(OpOperand *)> callback);

/// Fill `values` with a list of values defined at the ancestors of the `limit`
/// region and used within `region` or its descendants.
void getUsedValuesDefinedAbove(Region &region, Region &limit,
                               SetVector<Value> &values);

/// Fill `values` with a list of values used within any of the regions provided
/// but defined in one of the ancestors.
void getUsedValuesDefinedAbove(MutableArrayRef<Region> regions,
                               SetVector<Value> &values);

/// Run a set of structural simplifications over the given regions. This
/// includes transformations like unreachable block elimination, dead argument
/// elimination, as well as some other DCE. This function returns success if any
/// of the regions were simplified, failure otherwise. The provided rewriter is
/// used to notify callers of operation and block deletion.
LogicalResult simplifyRegions(RewriterBase &rewriter,
                              MutableArrayRef<Region> regions);

} // namespace mlir

#endif // MLIR_TRANSFORMS_REGIONUTILS_H_
