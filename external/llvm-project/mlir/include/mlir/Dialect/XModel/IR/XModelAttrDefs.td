//===- XModelAttrDefs.td - XModel MLIR Attributes ---------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XMODEL_ATTRS
#define XMODEL_ATTRS

include "mlir/Dialect/XModel/IR/XModelBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SubElementInterfaces.td"

class XModel_I32Enum<string name, string summary, list<I32EnumAttrCase> cases>
  : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::xmodel";

  let genSpecializedAttr = 0;
  let stringToSymbolFnName = "get" # name # "ForName";
  let symbolToStringFnName = "getNameFor" # name;
}

class XModel_Attr<string name, list<Trait> traits = []> :
    AttrDef<XModel_Dialect, name, traits>;

/// TargetObjectType
def ELFType : I32EnumAttrCase<"ELF", 0>;
def SPIRVType : I32EnumAttrCase<"SPIRV", 1>;
def LLVMIRType : I32EnumAttrCase<"LLVMIR", 2>;

def TargetObjectType : XModel_I32Enum<"TargetObjectType",
    "The target object type",
    [ELFType, SPIRVType, LLVMIRType]>;

/// TargetObjectAttr
def XModel_TargetObjectAttr : XModel_Attr<"TargetObject"> {
    let mnemonic = "target_obj";

    let description = [{
        A container for a target binary (or IR) and build details.

        A target_obj captures target specific build objects that may be
        referenced from one or more target_pkg's'.

        Each target_obj is specified as an attribute with the form
        ```
        #xmodel.target_obj<<type> = <arch> {<attributes>} -> <binary>>
        ```
        where the <type> is one of 'ELF' or 'SPIRV'. And where <arch> 
        specifies what target-arch the object is compiled for in the case
        of an 'ELF' object. And where the <binary> is a string stored
        representation of the actual binary.
    }];

    let parameters = (ins
      AttrParameter<"::mlir::xmodel::TargetObjectType", "The target object type">:$type,
      StringRefParameter<"The architecture target">:$arch,
      AttrParameter<"DictionaryAttr", "The object type">:$attributes,
      StringRefParameter<"The object binary">:$binary
    );

    let genVerifyDecl = 0;
    let hasCustomAssemblyFormat = 1;
}

/// TargetType
def CPUType : I32EnumAttrCase<"CPU", 0>;
def GPUType : I32EnumAttrCase<"GPU", 1>;
def NPUType : I32EnumAttrCase<"NPU", 2>;
def ALTType : I32EnumAttrCase<"ALT", 3>;

def TargetType : XModel_I32Enum<"TargetType",
    "The IP type for a kernel package/target object",
    [CPUType, GPUType, NPUType, ALTType]>;

/// KernelPackageAttr
def XModel_KernelPackageAttr : XModel_Attr<"KernelPackage", [
    DeclareAttrInterfaceMethods<SubElementAttrInterface>
  ]> {
    let mnemonic = "kernel_pkg";

    let description = [{
        A package for a target-specific launch object and launch details.

        A target_pkg is a way to describe affine maps between coordinate spaces
        that easily allows one to "take their derivative" when emitting hot unrolled
         loops and that simplifies the process of describing these mappings.

        Each target_pkg is specified as an attribute with the form
        ```
        #xmodel.kernel_pkg<'type' = 'target' : 'kernel_name' ['launch_dims'] {attributes} -> #xmodel.target_obj<>>
        ```
        where the params block may be omitted for transforms that do not take parameters.

    }];

    let parameters = (ins
      AttrParameter<"::mlir::xmodel::TargetType", "The target IP type">:$type,
      StringRefParameter<"The target name, redundant with target_object">:$target,
      StringRefParameter<"The name of the function reference">:$kernel_name,
      ArrayRefParameter<"uint32_t", "The launch dimensions">:$launch_dims,
      AttrParameter<"DictionaryAttr", "attributes">:$attributes,
      AttrParameter<"::mlir::xmodel::TargetObjectAttr", "The object">:$object
    );

    let extraClassDeclaration = [{
      public:
      using iterator = iter_one<const xmodel::TargetObjectAttr>;
      iterator begin() const;
      iterator end() const;
      bool empty() const { return false; }
      void walkImmediateSubElements(function_ref<void(Attribute)> walkAttrsFn,
                                function_ref<void(Type)> walkTypesFn) const;
      Attribute replaceImmediateSubElements(ArrayRef<Attribute>, ArrayRef<Type>) const;
    }];

    let genVerifyDecl = 0;
    let hasCustomAssemblyFormat = 1;
}

#endif
