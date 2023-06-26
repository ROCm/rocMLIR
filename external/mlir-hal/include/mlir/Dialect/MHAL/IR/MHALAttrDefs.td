//===- MHALAttrDefs.td - MHAL MLIR Attributes ---------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MHAL_ATTRS
#define MHAL_ATTRS

include "mlir/Dialect/MHAL/IR/MHALBase.td"
include "mlir/Dialect/MHAL/IR/MHALTypes.td"
//include "mlir/IR/SubElementInterfaces.td"

class MHAL_Attr<string name, list<Trait> traits = []> :
    AttrDef<MHALDialect, name, traits>;

/// TargetObjectAttr
def MHAL_TargetObjectAttr : MHAL_Attr<"TargetObject"> {
    let mnemonic = "target_obj";

    let description = [{
        A container for a target binary (or IR) and build details.

        A target_obj captures target specific build objects that may be
        referenced from one or more target_pkg's'.

        Each target_obj is specified as an attribute with the form
        ```
        #mhal.target_obj<<type> = <arch> {<attributes>} -> <binary>>
        ```
        where the <type> is one of 'ELF' or 'SPIRV'. And where <arch> 
        specifies what target-arch the object is compiled for in the case
        of an 'ELF' object. And where the <binary> is a string stored
        representation of the actual binary.
    }];

    let parameters = (ins
      AttrParameter<"::mlir::mhal::TargetObjectType", "The target object type">:$type,
      StringRefParameter<"The architecture target">:$arch,
      AttrParameter<"DictionaryAttr", "The object type">:$attributes,
      StringRefParameter<"The object binary">:$binary
    );

    let genVerifyDecl = 0;
    let hasCustomAssemblyFormat = 1;
}

/// KernelPackageAttr
def MHAL_KernelPackageAttr : MHAL_Attr<"KernelPackage"> {
    let mnemonic = "kernel_pkg";

    let description = [{
        A package for a target-specific launch object and launch details.

        A target_pkg is a way to describe affine maps between coordinate spaces
        that easily allows one to "take their derivative" when emitting hot unrolled
         loops and that simplifies the process of describing these mappings.

        Each target_pkg is specified as an attribute with the form
        ```
        #mhal.kernel_pkg<'type' = 'target' : 'kernel_name' ['launch_dims'] {attributes} -> #mhal.target_obj<>>
        ```
        where the params block may be omitted for transforms that do not take parameters.

    }];

    let parameters = (ins
      AttrParameter<"::mlir::mhal::TargetType", "The target IP type">:$type,
      StringRefParameter<"The target name, redundant with target_object">:$target,
      StringRefParameter<"The name of the function reference">:$kernel_name,
      ArrayRefParameter<"uint32_t", "The launch dimensions">:$launch_dims,
      AttrParameter<"DictionaryAttr", "attributes">:$attributes,
      AttrParameter<"::mlir::mhal::TargetObjectAttr", "The object">:$object
    );

    let genVerifyDecl = 0;
    let hasCustomAssemblyFormat = 1;
}

#endif
