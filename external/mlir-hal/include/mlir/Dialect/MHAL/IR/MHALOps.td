//===- MHALOps.td - MHAL operations definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for MHAL dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MHAL_OPS
#define MHAL_OPS

include "mlir/Dialect/MHAL/IR/MHALBase.td"
include "mlir/Dialect/MHAL/IR/MHALAttrDefs.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"


//===----------------------------------------------------------------------===//
// MHAL op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class MHAL_Op<string mnemonic, list<Trait> traits = []> :
    Op<MHALDialect, mnemonic, traits>;

def MHAL_LaunchOp :
  MHAL_Op<"launch", [DeclareOpInterfaceMethods<CallOpInterface>,
                      DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                      AttrSizedOperandSegments]> {
  let summary = "asynchronous launch operation";
  let description = [{
    The `mhal.launch` operation encapsulates a call to a `kernel` func that
    may run mhalhronously from the caller. An `mhal.token` is returned to
    maintain ordered dependent execution. Subsequent instructions that depend
    on results from the `mhal.launch` must either be dependent on the token
    or be preceded by an `mhal.await` of that token.

    The actual concurrency semantics depends on the dialect lowering to the
    executable format. Fully sequential execution ("compute0" completes before
    "compute1" starts) is a completely legal execution.

    Because concurrent execution is not guaranteed, it is undefined behavior
    to create an implicit dependency from "compute1" to "compute0" (e.g. via
    shared global state). All dependencies must be made explicit with mhal
    launch arguments (i.e. `mhal.token`).

   `mhal.launch` operation takes `mhal.token` dependencies and operands
    separately, and starts execution of the kernel function only when all
    tokens become ready. In contrast to `mhal.execute` this operation does
    not depend on operand values as `mhal.value` requires.

    Example:

    ```mlir
    %dependency = ... : !mhal.token

    %token, %results =
      mhal.launch @compute0 [%dependency] (%value0, %value1) :
                 (some.type, some.type) -> some.type

    %1 = "compute1"(...) : !some.type
    ```

    In the example above mhalhronous execution starts only after dependency
    token becomes ready.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<MHAL_TokenType>:$dependencies,
                       Variadic<AnyType>:$launchOperands);

  let results = (outs MHAL_TokenType:$token,
                      Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins "func::FuncOp":$kernelFunc, "ValueRange":$dependencies,
                   "ValueRange":$kernelOperands)>,
    OpBuilder<(ins "FlatSymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$launchOperands), [{
      $_state.addOperands(launchOperands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
      auto operandSegmentSizes = $_builder.getDenseI32ArrayAttr(
        {0, static_cast<int32_t>(launchOperands.size())});
      $_state.addAttribute(getOperandSegmentSizesAttrName($_state.name),
                           operandSegmentSizes);

    }]>,
  ];

  let extraClassDeclaration = [{
    Operation::result_range getCallResults();
    Operation::result_type_range getCallResultTypes();
    void updateSegmentSizes(MLIRContext *);
  }];

  let hasVerifier = 1;
  let assemblyFormat = [{
    $callee (` ` `[` $dependencies^ `]`)? `(` $launchOperands `)` attr-dict
    `:` `(` type($launchOperands) `)` (`->` type($results)^)?
  }];

}

def MHAL_AwaitOp : MHAL_Op<"await"> {
  let summary = "waits for the argument to become ready";
  let description = [{
    The `mhal.await` operation waits until the argument becomes ready, and for

    Example:

    ```mlir
    %0 = ... : !mhal.token
    mhal.await %0 : !mhal.token

    ```
  }];

  let arguments = (ins MHAL_TokenType:$operand);
  let results = (outs Optional<AnyType>:$result);

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$operand,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = [{
    std::optional<Type> getResultType() {
      if (getResultTypes().empty()) return std::nullopt;
      return getResultTypes()[0];
    }
  }];

  let assemblyFormat = [{
    $operand `:` custom<AwaitResultType>(
      type($operand), type($result)
    ) attr-dict
  }];
}

#endif // MHAL_OPS
