//===- AliasUtils.cpp - Utility functions for alias analysis -------------===//
//
// Copyright 2025 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/AliasUtils.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace rock {

LLVM::AliasScopeDomainAttr getDirectToLDSScopeDomain(MLIRContext *ctx) {
  Builder b(ctx);
  return b.getAttr<LLVM::AliasScopeDomainAttr>(
      b.getStringAttr("amdgpu.DirectToLDSLoads"),
      b.getStringAttr(
          "Domain to hold alias scopes to specify aliasing information for "
          "operations that load directly from global memory to LDS"));
}

LLVM::AliasScopeAttr getDirectToLDSLoadScope(MLIRContext *ctx) {
  Builder b(ctx);
  auto name = b.getStringAttr("amdgpu.DirectToLDSLoads");
  auto desc = b.getStringAttr(
      "Scope containing all operations that perform direct global-to-LDS loads");
  return b.getAttr<LLVM::AliasScopeAttr>(name, getDirectToLDSScopeDomain(ctx), desc);
}

void addDirectToLDSLoadAliasScope(LLVM::AliasAnalysisOpInterface op) {
  auto ctx = op->getContext();
  Builder b(ctx);
  op.setAliasScopes(b.getArrayAttr(getDirectToLDSLoadScope(ctx)));
}

} // namespace rock
} // namespace mlir

