//===- MHALTypes.td - MHAL dialect types -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MHAL dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MHAL_IR_MHALTYPES
#define MLIR_DIALECT_MHAL_IR_MHALTYPES

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/MHAL/IR/MHALBase.td"

//===----------------------------------------------------------------------===//
// MHAL Types
//===----------------------------------------------------------------------===//

class MHAL_Type<string name, string typeMnemonic> : TypeDef<MHALDialect,
                                                             name> {
  let mnemonic = typeMnemonic;
}

def MHAL_TokenType : MHAL_Type<"Token", "token"> {
  let summary = "mhal token type";
  let description = [{
    `mhal.token` is a type returned by mhalhronous operations, and it becomes
    `available` when the mhalhronous operations that created it is completed.
  }];
}

class MHAL_I32Enum<string name, string summary, list<I32EnumAttrCase> cases>
  : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::mhal";

  let genSpecializedAttr = 0;
  let stringToSymbolFnName = "get" # name # "ForName";
  let symbolToStringFnName = "getNameFor" # name;
}

/// TargetObjectType
def ELFType : I32EnumAttrCase<"ELF", 0>;
def SPIRVType : I32EnumAttrCase<"SPIRV", 1>;
def LLVMIRType : I32EnumAttrCase<"LLVMIR", 2>;

def TargetObjectType : MHAL_I32Enum<"TargetObjectType",
    "The target object type",
    [ELFType, SPIRVType, LLVMIRType]>;

/// TargetType
def CPUType : I32EnumAttrCase<"CPU", 0>;
def GPUType : I32EnumAttrCase<"GPU", 1>;
def NPUType : I32EnumAttrCase<"NPU", 2>;
def ALTType : I32EnumAttrCase<"ALT", 3>;

def TargetType : MHAL_I32Enum<"TargetType",
    "The IP type for a kernel package/target object",
    [CPUType, GPUType, NPUType, ALTType]>;

#endif // MLIR_DIALECT_MHAL_IR_MHALTYPES
