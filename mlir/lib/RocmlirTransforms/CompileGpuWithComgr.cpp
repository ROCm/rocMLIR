//===- CompileGpuWithComgr.cpp - Compile kernelsto HSACO with COMGR --===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2023 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that serializes a gpu module into HSAco blob and
// adds that blob as a string attribute of the module. Unlike the main
// SeralizeToHsaco, this uses AMD's COMGR interface to call an external LLVM.
//
//===----------------------------------------------------------------------===//

#include "mlir/RocmlirTransforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#ifdef ROCMLIR_COMPILE_GPU_WITH_COMGR_PASS_ENABLE
#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

#include <memory>
#include <string>

#include <amd_comgr/amd_comgr.h>
#endif

namespace mlir {
namespace rocmlir {
#define GEN_PASS_DEF_COMPILEGPUWITHCOMGRPASS
#include "mlir/RocmlirTransforms/Passes.h.inc"
} // namespace rocmlir
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "compile-gpu-with-comgr"

namespace {
#ifdef ROCMLIR_COMPILE_GPU_WITH_COMGR_PASS_ENABLE
class CompileGpuWithComgrPass
    : public rocmlir::impl::CompileGpuWithComgrPassBase<
          CompileGpuWithComgrPass> {
public:
  using rocmlir::impl::CompileGpuWithComgrPassBase<
      CompileGpuWithComgrPass>::CompileGpuWithComgrPassBase;

  void runOnOperation() override;

private:
  void getDependentDialects(DialectRegistry &registry) const override;

  /// Adjusts the target string we've been passed to how comgr likes them,
  /// namely, instead of triple:cpu[:feature:feature], we go for
  // triple--cpu[:feature].
  FailureOr<std::string> getComgrTarget();

  /// Scan a module to see what device libraries are needed.
  /// Returns the target backend.
  amd_comgr_language_t getDeviceLibList(const llvm::Module &module,
                                        SmallVectorImpl<const char *> &libs);

  /// Whereas the comgr LLVM is behind the one we use to print the IR, use this
  /// callback to do any "downgrades" needed for the serialized form of the IR.
  void downgradeIrForBackcompat(SmallVectorImpl<char> &ir);

  LogicalResult compileModule(gpu::GPUModuleOp mlirMod,
                              SmallVectorImpl<char> &binary);
};
} // end anonymous namespace

void CompileGpuWithComgrPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerROCDLDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  rocmlir::impl::CompileGpuWithComgrPassBase<
      CompileGpuWithComgrPass>::getDependentDialects(registry);
}

FailureOr<std::string> CompileGpuWithComgrPass::getComgrTarget() {
  Location loc = getOperation()->getLoc();
  if (triple.empty() || chip.empty())
    return emitError(loc, "must specify a target triple and chipset");
  std::string retVal;
  llvm::raw_string_ostream ret(retVal);

  ret << triple << "--" << chip;
  // COMGR wants features clang-style (:bar+:foo-) features but SerializeToHsaco
  // wants llc-style (+bar,-foo). We expect llc-style features to be passed in,
  // but account for both.
  if (chipFeatures.empty()) {
    // do nothing
  } else if (chipFeatures.find(':') != std::string::npos)
    ret << ':' << chipFeatures;
  else {
    for (StringRef feature : llvm::split(chipFeatures, ',')) {
      ret << ':' << feature.substr(1) << feature.take_front(1);
    }
  }
  return retVal;
}

amd_comgr_language_t
CompileGpuWithComgrPass::getDeviceLibList(const llvm::Module &module,
                                          SmallVectorImpl<const char *> &libs) {
  amd_comgr_language_t ret = AMD_COMGR_LANGUAGE_LAST;

  // Walk the LLVM module in order to determine if we need to link in device
  // libs.
  bool needOpenCl = false;
  bool needOckl = false;
  bool needOcml = false;
  for (const llvm::Function &f : module.functions()) {
    if (f.hasExternalLinkage() && f.hasName() && !f.hasExactDefinition()) {
      StringRef funcName = f.getName();
      if ("printf" == funcName)
        needOpenCl = true;
      if (funcName.startswith("__ockl_"))
        needOckl = true;
      if (funcName.startswith("__ocml_"))
        needOcml = true;
    }
  }

  if (!(needOpenCl || needOcml || needOckl))
    // Return _NONE to indicate that no device libraries are needed.
    return AMD_COMGR_LANGUAGE_NONE;

  if (needOpenCl) {
    ret = AMD_COMGR_LANGUAGE_OPENCL_2_0;
    needOcml = needOckl = true;
  } else {
    ret = AMD_COMGR_LANGUAGE_HIP;
  }
  if (needOcml) {
    libs.push_back("correctly_rounded_sqrt");
  }
  if (needOcml || needOckl) {
    libs.push_back("wavefrontsize64");
    // This constant must always match the default code object ABI version
    // of the AMDGPU backend.
    libs.push_back("code_object_v4");
  }
  return ret;
}

static std::string subAll(const llvm::Regex &regex, StringRef rep,
                          std::string orig) {
  while (regex.match(orig)) {
    orig = regex.sub(rep, orig);
  }
  return orig;
}

void CompileGpuWithComgrPass::downgradeIrForBackcompat(
    SmallVectorImpl<char> &ir) {
  llvm::Regex memoryNone("memory\\(none\\)");
  std::string original(ir.data(), ir.size());
  std::string memoryNoneRep = subAll(memoryNone, "readnone", original);
  llvm::Regex memoryRead("memory\\(read\\)");
  std::string memoryReadRep = subAll(memoryRead, "readonly", memoryNoneRep);
  llvm::Regex memoryWrite("memory\\(write\\)");
  std::string memoryWriteRep = subAll(memoryWrite, "writeonly", memoryReadRep);
  ir.assign(memoryWriteRep.begin(), memoryWriteRep.end());
}

LogicalResult
CompileGpuWithComgrPass::compileModule(gpu::GPUModuleOp mlirMod,
                                       SmallVectorImpl<char> &binary) {
#define CHECK_COMGR_CALL(call)                                                 \
  do {                                                                         \
    status = (call);                                                           \
    if (AMD_COMGR_STATUS_SUCCESS != status) {                                  \
      const char *statusName;                                                  \
      amd_comgr_status_string(status, &statusName);                            \
      return mlirMod->emitOpError("Failed comgr call: " #call " with status ") \
             << statusName;                                                    \
    }                                                                          \
  } while (0)

  amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;
  FailureOr<std::string> maybeTargetIsa = getComgrTarget();
  if (failed(maybeTargetIsa))
    return failure();
  std::string targetIsa = std::move(*maybeTargetIsa);

  StringRef name = mlirMod.getNameAttr().getValue();
  if (name.empty())
    name = "GPUKernels";
  SmallString<32> pointableName(name);
  // needed to make the compiler do the right thing
  pointableName.append(".ll");

  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> llvmMod =
      translateModuleToLLVMIR(mlirMod, ctx, name);
  if (!llvmMod) {
    return mlirMod.emitOpError("Could not translate module to LLVM IR");
  }
  llvmMod->setTargetTriple(triple);

  SmallVector<const char *> libraries;
  amd_comgr_language_t inputLanguage = getDeviceLibList(*llvmMod, libraries);

  llvm::SmallVector<char, 0> irBuf;
  llvm::raw_svector_ostream irStream(irBuf);
  llvmMod->print(irStream, nullptr);
  downgradeIrForBackcompat(irBuf);

  LLVM_DEBUG({
    llvm::dbgs() << "LLVM IR for module " << name << "\n" << irBuf;
    llvm::dbgs().flush();
  });

  amd_comgr_action_info_t actionInfo;
  CHECK_COMGR_CALL(amd_comgr_create_action_info(&actionInfo));
  CHECK_COMGR_CALL(
      amd_comgr_action_info_set_isa_name(actionInfo, targetIsa.c_str()));

  amd_comgr_data_t irData;
  CHECK_COMGR_CALL(amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, &irData));
  CHECK_COMGR_CALL(amd_comgr_set_data(irData, irBuf.size(), irBuf.data()));
  CHECK_COMGR_CALL(amd_comgr_set_data_name(irData, pointableName.c_str()));

  amd_comgr_data_set_t inputIrSet, targetBcSet;
  CHECK_COMGR_CALL(amd_comgr_create_data_set(&inputIrSet));
  CHECK_COMGR_CALL(amd_comgr_create_data_set(&targetBcSet));
  CHECK_COMGR_CALL(amd_comgr_data_set_add(inputIrSet, irData));

  // HACK: "compile" needs a language set, and HIP will get us into offload-arch
  // so we claim to be OpenCL for a hot second.
  amd_comgr_action_info_t actionInfoHack;
  CHECK_COMGR_CALL(amd_comgr_create_action_info(&actionInfoHack));
  CHECK_COMGR_CALL(
      amd_comgr_action_info_set_isa_name(actionInfoHack, targetIsa.c_str()));
  CHECK_COMGR_CALL(amd_comgr_action_info_set_language(
      actionInfoHack, AMD_COMGR_LANGUAGE_OPENCL_2_0));
  const char *hackArgs[] = {"-x", "ir"};
  CHECK_COMGR_CALL(
      amd_comgr_action_info_set_option_list(actionInfoHack, hackArgs, 2));

  CHECK_COMGR_CALL(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                                       actionInfoHack, inputIrSet,
                                       targetBcSet));
  CHECK_COMGR_CALL(amd_comgr_destroy_action_info(actionInfoHack));
  CHECK_COMGR_CALL(amd_comgr_release_data(irData));
  CHECK_COMGR_CALL(amd_comgr_destroy_data_set(inputIrSet));

  if (!libraries.empty()) {
    // We do this down here (and temporarily) in order to prevent
    // issuse where setting the language affects some of how COMgr handles
    // setting up Clang, namely that setting the language to HIP implies that
    // architecture should be handled by way of offload-arch.
    CHECK_COMGR_CALL(
        amd_comgr_action_info_set_language(actionInfo, inputLanguage));

    amd_comgr_data_set_t withDeviceLibsSet, linkedBcSet;
    CHECK_COMGR_CALL(amd_comgr_create_data_set(&withDeviceLibsSet));
    CHECK_COMGR_CALL(amd_comgr_create_data_set(&linkedBcSet));

    CHECK_COMGR_CALL(amd_comgr_action_info_set_option_list(
        actionInfo, libraries.data(), libraries.size()));
    CHECK_COMGR_CALL(amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES,
                                         actionInfo, targetBcSet,
                                         withDeviceLibsSet));
    CHECK_COMGR_CALL(
        amd_comgr_action_info_set_option_list(actionInfo, nullptr, 0));
    CHECK_COMGR_CALL(amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC,
                                         actionInfo, withDeviceLibsSet,
                                         linkedBcSet));
    // Free data we're about to lose references to.
    CHECK_COMGR_CALL(amd_comgr_destroy_data_set(targetBcSet));
    CHECK_COMGR_CALL(amd_comgr_destroy_data_set(withDeviceLibsSet));
    targetBcSet = linkedBcSet;

    // Hack: go back to the NONE language to make sure arch flags are handled
    // correctly.
    CHECK_COMGR_CALL(amd_comgr_action_info_set_language(
        actionInfo, AMD_COMGR_LANGUAGE_NONE));
  }

  // Set up compiler options.
  llvm::SmallString<4> optOption;
  ("-O" + Twine(optLevel)).toVector(optOption);
  SmallVector<const char *, 0> options;
  options.push_back(optOption.c_str());
  CHECK_COMGR_CALL(amd_comgr_action_info_set_option_list(
      actionInfo, options.data(), options.size()));

  // Codegen (opt + llc) to .o files.
  amd_comgr_data_set_t objectsSet;
  amd_comgr_action_info_set_logging(actionInfo, true);
  CHECK_COMGR_CALL(amd_comgr_create_data_set(&objectsSet));
  CHECK_COMGR_CALL(
      amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                          actionInfo, targetBcSet, objectsSet));

  // Link into a loadable executable.
  amd_comgr_data_set_t hsacoSet;
  CHECK_COMGR_CALL(amd_comgr_create_data_set(&hsacoSet));
  CHECK_COMGR_CALL(
      amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                          actionInfo, objectsSet, hsacoSet));
  CHECK_COMGR_CALL(amd_comgr_destroy_data_set(objectsSet));

  size_t resultCount = 0;
  CHECK_COMGR_CALL(amd_comgr_action_data_count(
      hsacoSet, AMD_COMGR_DATA_KIND_EXECUTABLE, &resultCount));
  if (resultCount != 1)
    return mlirMod.emitOpError("couldn't compile LLVM IR");

  amd_comgr_data_t hsaco;
  CHECK_COMGR_CALL(amd_comgr_action_data_get_data(
      hsacoSet, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &hsaco));
  size_t numBytes = 0;
  CHECK_COMGR_CALL(amd_comgr_get_data(hsaco, &numBytes, nullptr));
  binary.resize_for_overwrite(numBytes);
  CHECK_COMGR_CALL(amd_comgr_get_data(hsaco, &numBytes, binary.data()));

  CHECK_COMGR_CALL(amd_comgr_release_data(hsaco));
  CHECK_COMGR_CALL(amd_comgr_destroy_data_set(hsacoSet));
  CHECK_COMGR_CALL(amd_comgr_destroy_action_info(actionInfo));

  return success();
}

void CompileGpuWithComgrPass::runOnOperation() {
  gpu::GPUModuleOp mlirMod = getOperation();
  SmallVector<char, 0> binary;

  if (failed(compileModule(mlirMod, binary)))
    return signalPassFailure();

  MLIRContext *ctx = mlirMod->getContext();
  auto binaryAttr =
      StringAttr::get(ctx, StringRef(binary.data(), binary.size()));
  // Hard-code this to avoid depending on SerializeToBlob, which we're trying to
  // avoid pulling in.
  mlirMod->setAttr("gpu.binary", binaryAttr);
}

#else
class CompileGpuWithComgrPass
    : public rocmlir::impl::CompileGpuWithComgrPassBase<
          CompileGpuWithComgrPass> {
  void runOnOperation() override {
    gpu::GPUModuleOp mod = getOperation();
    mod.emitOpError("Cannot use COMGR because it is not linked in. Try "
                    "serialize-to-hsaco instead\n");
    signalPassFailure();
  }
};
#endif
