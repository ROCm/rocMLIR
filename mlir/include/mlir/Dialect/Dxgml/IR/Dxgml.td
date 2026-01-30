//===- Dxgml.td - Dxgml operation definitions -------------*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Dxgml core operations.
//
//===----------------------------------------------------------------------===//

#ifndef DXGML
#define DXGML

include "mlir/Dialect/Dxgml/IR/DxgmlBase.td"
include "mlir/Dialect/Dxgml/IR/DxgmlTypes.td"
include "mlir/Dialect/Dxgml/IR/DxgmlAttrs.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for Dxgml dialect ops
class Dxgml_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dxgml_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Module Operation
//===----------------------------------------------------------------------===//

def Dxgml_ModuleOp : Dxgml_Op<"module", [
    IsolatedFromAbove,
    NoTerminator,
    SingleBlock,
    NoRegionArguments,
    SymbolTable,
    Symbol,
    AffineScope
  ]> {
  let summary = "Top-level container operation";
  let description = [{
    A `dxgml.module` represents a top-level container for DXGML operations.
    It contains a single region with a single block.
    
    Example:
    ```mlir
    dxgml.module {
      dxgml.entry_point @main(...) -> ... {
        ...
      }
    }
    ```
  }];
  
  let arguments = (ins
    OptionalAttr<SymbolNameAttr>:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  
  let regions = (region SizedRegion<1>:$bodyRegion);
  
  let assemblyFormat = "($sym_name^)? attr-dict-with-keyword $bodyRegion";
  
  let builders = [OpBuilder<(ins
    CArg<"std::optional<StringRef>", "{}">:$name)
  >];
}

//===----------------------------------------------------------------------===//
// Entry Point Operation
//===----------------------------------------------------------------------===//

def Dxgml_EntryPointOp : Dxgml_Op<"entry_point", [
    AffineScope,
    AutomaticAllocationScope,
    FunctionOpInterface,
    IsolatedFromAbove,
    Symbol
  ]> {
  let summary = "Entry point function";
  let description = [{
    An entry point function in a DXGML module.
    
    Example:
    ```mlir
    dxgml.entry_point @main(%arg0: !dxgml.tensor<1x4x!dxgml.float16>) 
        -> !dxgml.tensor<1x4x!dxgml.float16> {
      dxgml.return %arg0 : !dxgml.tensor<1x4x!dxgml.float16>
    }
    ```
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  
  let regions = (region AnyRegion:$body);
  
  let builders = [OpBuilder<(ins
    "StringRef":$name,
    "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//
    
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { 
      return cast<FunctionType>(getFunctionType()).getInputs(); 
    }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { 
      return cast<FunctionType>(getFunctionType()).getResults(); 
    }
    
    /// Clone the function type with new inputs and results.
    Type cloneTypeWith(TypeRange inputs, TypeRange results) {
      return FunctionType::get(getContext(), inputs, results);
    }
    
    /// Returns the region on the function operation that is callable.
    Region *getCallableRegion() { return &getBody(); }
  }];
  
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Function Operation
//===----------------------------------------------------------------------===//

def Dxgml_FunctionOp : Dxgml_Op<"function", [
    AffineScope,
    AutomaticAllocationScope,
    FunctionOpInterface,
    IsolatedFromAbove,
    Symbol
  ]> {
  let summary = "Function operation";
  let description = [{
    A function in a DXGML module.
    
    Example:
    ```mlir
    dxgml.function @helper(%arg0: !dxgml.tensor<1x4x!dxgml.float16>) 
        -> !dxgml.tensor<1x4x!dxgml.float16> {
      dxgml.return %arg0 : !dxgml.tensor<1x4x!dxgml.float16>
    }
    ```
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  
  let regions = (region AnyRegion:$body);
  
  let builders = [OpBuilder<(ins
    "StringRef":$name,
    "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//
    
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { 
      return cast<FunctionType>(getFunctionType()).getInputs(); 
    }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { 
      return cast<FunctionType>(getFunctionType()).getResults(); 
    }
    
    /// Clone the function type with new inputs and results.
    Type cloneTypeWith(TypeRange inputs, TypeRange results) {
      return FunctionType::get(getContext(), inputs, results);
    }
    
    /// Returns the region on the function operation that is callable.
    Region *getCallableRegion() { return &getBody(); }
  }];
  
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Return Operation
//===----------------------------------------------------------------------===//

def Dxgml_ReturnOp : Dxgml_Op<"return", [
    Pure,
    ReturnLike,
    Terminator,
    HasParent<"EntryPointOp, FunctionOp">
  ]> {
  let summary = "Return operation";
  let description = [{
    The `dxgml.return` operation terminates a function or entry point
    and returns values.
    
    Example:
    ```mlir
    dxgml.return %0 : !dxgml.tensor<1x4x!dxgml.float16>
    ```
  }];
  
  let arguments = (ins Variadic<AnyType>:$operands);
  
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Invoke Operation  
//===----------------------------------------------------------------------===//

def Dxgml_InvokeOp : Dxgml_Op<"invoke", [
    CallOpInterface,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Invoke a function or subgraph";
  let description = [{
    Invokes a function or subgraph.
    
    Example:
    ```mlir
    %result = dxgml.invoke @function(%arg0) : (!dxgml.tensor<1x4x!dxgml.float16>) 
                                            -> !dxgml.tensor<1x4x!dxgml.float16>
    ```
  }];
  
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );
  
  let results = (outs Variadic<AnyType>);
  
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  
  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() { return getOperands(); }
    MutableOperandRange getArgOperandsMutable() { return getOperandsMutable(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCalleeAttr(); }
    
    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      setCalleeAttr(cast<FlatSymbolRefAttr>(callee.get<SymbolRefAttr>()));
    }
  }];
}

#endif // DXGML
