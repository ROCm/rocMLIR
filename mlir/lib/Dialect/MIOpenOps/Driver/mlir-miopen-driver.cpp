//===- mlir-miopen-driver.cpp - MLIR MIOpen Dialect Driver ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-miopen-driver.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<std::string> filterLayout("f", cl::desc("Filter layout"),
                                              cl::value_desc("layout string"),
                                              cl::init("kcyx"));

static cl::opt<std::string> inputLayout("i", cl::desc("Input layout"),
                                              cl::value_desc("layout string"),
                                              cl::init("nchw"));

static cl::opt<std::string> outputLayout("d", cl::desc("Output layout"),
                                              cl::value_desc("layout string"),
                                              cl::init("nkhw"));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR MIOpen Dialect driver\n");

  // Construct a new ModuleOp.
  MLIRContext context;
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());

  // Construct a new FuncOp.
  auto argType = MemRefType::get({-1, -1, -1, -1}, builder.getF32Type());
  auto funcType = builder.getFunctionType({argType, argType, argType}, {});
  auto func = FuncOp::create(builder.getUnknownLoc(), "miopen_conv2d_" + filterLayout + "_" + inputLayout + "_" + outputLayout, funcType);
  module.push_back(func);

  // Construct a new Block.
  auto *block = func.addEntryBlock();

  // Construct a new Con2DOp.
  llvm::SmallVector<StringAttr, 4> filterLayoutSpec;
  llvm::SmallVector<StringAttr, 4> inputLayoutSpec;
  llvm::SmallVector<StringAttr, 4> outputLayoutSpec;
  for (size_t i = 0; i < 4; ++i) {
    filterLayoutSpec.push_back(builder.getStringAttr(StringRef(&filterLayout.getValue()[i], 1)));
    inputLayoutSpec.push_back(builder.getStringAttr((StringRef(&inputLayout.getValue()[i], 1) + "i").str()));
    outputLayoutSpec.push_back(builder.getStringAttr((StringRef(&outputLayout.getValue()[i], 1) + "o").str()));
  }

  auto convOp = builder.create<miopen::Conv2DOp>(
    builder.getUnknownLoc(),
    ArrayRef<Type>({}),
    ValueRange{block->getArgument(0), block->getArgument(1), block->getArgument(2)}, 
    ArrayRef<NamedAttribute>{
      builder.getNamedAttr("filter_layout", builder.getArrayAttr(ArrayRef<Attribute>(filterLayoutSpec.begin(), filterLayoutSpec.end()))),
      builder.getNamedAttr("input_layout", builder.getArrayAttr(ArrayRef<Attribute>(inputLayoutSpec.begin(), inputLayoutSpec.end()))),
      builder.getNamedAttr("output_layout", builder.getArrayAttr(ArrayRef<Attribute>(outputLayoutSpec.begin(), outputLayoutSpec.end()))),

      // TBD: support dilations / strides / padding.
      builder.getNamedAttr("dilations", builder.getArrayAttr({
                                          builder.getI32IntegerAttr(1),
                                          builder.getI32IntegerAttr(1),
                                        })),
      builder.getNamedAttr("strides", builder.getArrayAttr({
                                          builder.getI32IntegerAttr(1),
                                          builder.getI32IntegerAttr(1),
                                        })),
      builder.getNamedAttr("padding", builder.getArrayAttr({
                                          builder.getI32IntegerAttr(0),
                                          builder.getI32IntegerAttr(0),
                                        })),
    });
  block->push_back(convOp);

  // Construct a new ReturnOp.
  auto returnOp = builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  // Set up the output file.
  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  module.print(output->os());
  output->keep();

  return 0;
}
