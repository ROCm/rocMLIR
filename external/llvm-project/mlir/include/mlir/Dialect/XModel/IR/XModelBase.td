//===- XModelDialect.td - XModel dialect definition -----------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XMODEL_DIALECT
#define XMODEL_DIALECT

include "mlir/IR/OpBase.td"

def XModel_Dialect : Dialect {
  let name = "xmodel";
  let cppNamespace = "::mlir::xmodel";
  let useFoldAPI = kEmitFoldAdaptorFolder;

  let useDefaultAttributePrinterParser = 1;
}

#endif // XMODEL_DIALECT
