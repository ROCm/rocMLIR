//===- DxgmlBase.td - Dxgml dialect definitions -----------*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines base MLIR Dxgml dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DXGML_BASE
#define DXGML_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def Dxgml_Dialect : Dialect {
  let name = "dxgml";
  let cppNamespace = "::mlir::dxgml";
  
  let summary = "The DxGML dialect";
  let description = [{
    The DxGML dialect provides core operations for representing
    DirectX Machine Learning (DXGML) graphs in MLIR.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

// Base class for Dxgml dialect types
class Dxgml_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Dxgml_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Base class for Dxgml dialect attributes  
class Dxgml_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Dxgml_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // DXGML_BASE
