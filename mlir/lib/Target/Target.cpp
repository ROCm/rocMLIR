//===- Target.cpp - MLIR LLVM ROCDL target compilation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines ROCDL target related functions including registration
// calls for the `#rocdl.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitRocMLIRTarget.h"

#include "mlir/Config/mlir-config.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::ROCDL;

namespace {
// Implementation of the `TargetAttrInterface` model.
class ROCDLTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<ROCDLTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

// Register the ROCDL dialect, the ROCDL translation and the target interface.
void mlir::registerRocTarget(DialectRegistry &registry) {
  registerLLVMDialectTranslation(registry);
  registerROCDLDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  registry.addExtension(+[](MLIRContext *ctx, ROCDL::ROCDLDialect *dialect) {
    ROCDLTargetAttr::attachInterface<ROCDLTargetAttrImpl>(*ctx);
  });
}

void mlir::registerRocTarget(MLIRContext &context) {
  DialectRegistry registry;
  registerRocTarget(registry);
  context.appendDialectRegistry(registry);
}

#if MLIR_ENABLE_ROCM_CONVERSIONS == 1
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"

LLD_HAS_DRIVER(elf)

#ifdef ROCMLIR_DEVICE_LIBS_PACKAGED
#include "AmdDeviceLibs.cpp.inc"
#endif

namespace {
class AMDGPUSerializer : public SerializeGPUModuleBase {
public:
  AMDGPUSerializer(Operation &module, ROCDLTargetAttr target,
                   const gpu::TargetOptions &targetOptions);

  /// Loads the bitcode files in `fileList`.
  virtual std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
  loadBitcodeFiles(llvm::Module &module) override;

  /// Compiles assembly to a binary.
  std::optional<SmallVector<char, 0>>
  compileToBinary(const std::string &serializedISA) override;

  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;

private:
  // Target options.
  gpu::TargetOptions targetOptions;
};
} // namespace

AMDGPUSerializer::AMDGPUSerializer(Operation &module, ROCDLTargetAttr target,
                                   const gpu::TargetOptions &targetOptions)
    : SerializeGPUModuleBase(module, target, targetOptions),
      targetOptions(targetOptions) {
  deviceLibs = AMDGCNLibraries::None;
}

std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
AMDGPUSerializer::loadBitcodeFiles(llvm::Module &module) {
  SmallVector<std::unique_ptr<llvm::Module>> bcFiles;
  // Return if there are no libs to load.
  if (deviceLibs == AMDGCNLibraries::None && fileList.empty())
    return bcFiles;
  SmallVector<std::pair<StringRef, AMDGCNLibraries>, 3> libraries;
  AMDGCNLibraries libs = deviceLibs;
  if (any(libs & AMDGCNLibraries::Ocml))
    libraries.push_back({"ocml.bc", AMDGCNLibraries::Ocml});
  if (any(libs & AMDGCNLibraries::Ockl))
    libraries.push_back({"ockl.bc", AMDGCNLibraries::Ockl});
  if (any(libs & AMDGCNLibraries::OpenCL))
    libraries.push_back({"opencl.bc", AMDGCNLibraries::OpenCL});
#ifdef ROCMLIR_DEVICE_LIBS_PACKAGED
  const llvm::StringMap<StringRef> &packagedLibs = getDeviceLibraries();
  for (auto [file, lib] : libraries) {
    std::unique_ptr<llvm::Module> library;
    if (packagedLibs.contains(file)) {
      std::unique_ptr<llvm::MemoryBuffer> fileBc =
          llvm::MemoryBuffer::getMemBuffer(packagedLibs.at(file), file);

      llvm::SMDiagnostic error;
      library =
          llvm::getLazyIRModule(std::move(fileBc), error, module.getContext());
      if (!library) {
        getOperation().emitError("Error loading library: " + file +
                                 ", error message:" + error.getMessage());
        return std::nullopt;
      }
      // Unset the lib so we don't add it with `appendStandardLibs`.
      libs = libs & ~lib;
      bcFiles.push_back(std::move(library));
    } else {
      getOperation().emitWarning("Trying to find " + Twine(file) +
                                 " at runtime since it wasn't packaged");
    }
  }
#endif
  // Try to append any remaining standard device libs.
  if (failed(appendStandardLibs(libs)))
    return std::nullopt;
  if (failed(loadBitcodeFilesFromList(module.getContext(), fileList, bcFiles,
                                      true)))
    return std::nullopt;
  return std::move(bcFiles);
}

std::optional<SmallVector<char, 0>>
AMDGPUSerializer::compileToBinary(const std::string &serializedISA) {
  // Assemble the ISA.
  std::optional<SmallVector<char, 0>> isaBinary = assembleIsa(serializedISA);

  if (!isaBinary) {
    getOperation().emitError() << "Failed during ISA assembling.";
    return std::nullopt;
  }

  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel%%", "o", tempIsaBinaryFd,
                                         tempIsaBinaryFilename)) {
    getOperation().emitError()
        << "Failed to create a temporary file for dumping the ISA binary.";
    return std::nullopt;
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);
  {
    llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, true);
    tempIsaBinaryOs << StringRef(isaBinary->data(), isaBinary->size());
    tempIsaBinaryOs.flush();
  }

  // Create a temp file for HSA code object.
  SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco",
                                         tempHsacoFilename)) {
    getOperation().emitError()
        << "Failed to create a temporary file for the HSA code object.";
    return std::nullopt;
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  static llvm::sys::Mutex mutex;
  {
    const llvm::sys::ScopedLock lock(mutex);
    // Invoke lld. Expect a true return value from lld.
    if (!lld::elf::link({"ld.lld", "-shared", tempIsaBinaryFilename.c_str(),
                         "-o", tempHsacoFilename.c_str()},
                        llvm::outs(), llvm::errs(), false, false)) {
      getOperation().emitError() << "lld invocation error";
      return std::nullopt;
    }
    lld::CommonLinkerContext::destroy();
  }

  // Load the HSA code object.
  auto hsacoFile =
      llvm::MemoryBuffer::getFile(tempHsacoFilename, /*IsText=*/false);
  if (!hsacoFile) {
    getOperation().emitError()
        << "Failed to read the HSA code object from the temp file.";
    return std::nullopt;
  }

  StringRef buffer = (*hsacoFile)->getBuffer();

  return SmallVector<char, 0>(buffer.begin(), buffer.end());
}

std::optional<SmallVector<char, 0>>
AMDGPUSerializer::moduleToObject(llvm::Module &llvmModule) {
  return moduleToObjectImpl(targetOptions, llvmModule);
}
#endif // MLIR_ENABLE_ROCM_CONVERSIONS

std::optional<SmallVector<char, 0>> ROCDLTargetAttrImpl::serializeToObject(
    Attribute attribute, Operation *module,
    const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }
#if MLIR_ENABLE_ROCM_CONVERSIONS
  AMDGPUSerializer serializer(*module, cast<ROCDLTargetAttr>(attribute),
                              options);
  serializer.init();
  return serializer.run();
#else
  module->emitError("The `AMDGPU` target was not built. Please enable it when "
                    "building LLVM.");
  return std::nullopt;
#endif // MLIR_ENABLE_ROCM_CONVERSIONS
}

static gpu::KernelTableAttr getRockKernelMetadata(Operation *gpuModule,
                                                  ArrayRef<char> elfData) {
  auto module = cast<gpu::GPUModuleOp>(gpuModule);
  Builder builder(module.getContext());
  SmallVector<gpu::KernelMetadataAttr> kernels;
  std::optional<DenseMap<StringAttr, NamedAttrList>> mdMapOrNull =
      getAMDHSAKernelsELFMetadata(builder, elfData);
  auto getMD = [&](StringAttr attr) -> NamedAttrList {
    if (!mdMapOrNull)
      return NamedAttrList();
    return mdMapOrNull->lookup(attr);
  };
  auto addFuncAttr = [](Operation *op, NamedAttrList &attrs, StringRef key) {
    if (Attribute attr = op->getAttr(key))
      attrs.append(key, attr);
  };
  for (auto funcOp : module.getBody()->getOps<LLVM::LLVMFuncOp>()) {
    if (!funcOp->getDiscardableAttr("rocdl.kernel"))
      continue;
    NamedAttrList attrs = getMD(funcOp.getNameAttr());
    addFuncAttr(funcOp, attrs, "rock.shared_buffer_size");
    addFuncAttr(funcOp, attrs, "grid_size");
    addFuncAttr(funcOp, attrs, "block_size");
    addFuncAttr(funcOp, attrs, "original_func");
    kernels.push_back(gpu::KernelMetadataAttr::get(funcOp, builder.getDictionaryAttr(attrs)));
  }
  SmallVector<gpu::KernelMetadataAttr> sortedKernels(kernels);
  llvm::array_pod_sort(sortedKernels.begin(), sortedKernels.end());
  return gpu::KernelTableAttr::get(module.getContext(), sortedKernels, true);
}

Attribute
ROCDLTargetAttrImpl::createObject(Attribute attribute, Operation *module,
                                  const SmallVector<char, 0> &object,
                                  const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  // If format is `fatbin` transform it to binary as `fatbin` is not yet
  // supported.
  if (format > gpu::CompilationTarget::Binary)
    format = gpu::CompilationTarget::Binary;

  DictionaryAttr properties{};
  Builder builder(attribute.getContext());

  gpu::KernelTableAttr metadata{};
  // Only add properties if the format is an ELF binary.
  if (gpu::CompilationTarget::Binary == format) {
    NamedAttrList props;
    // Collect kernel metadata
    metadata = getRockKernelMetadata(module, object);
    // TODO(fmora): MHAL appears to be storing prefill attributes in the module,
    // switch them to be function attributes as they are present in
    // `ROCDLObjectMDAttr`.
    for (auto kernel : metadata) {
      auto key = kernel.getName();
      if (auto attr = module->getDiscardableAttr(key))
        props.append(key, attr);
    }
    properties = builder.getDictionaryAttr(props);
  }

  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(StringRef(object.data(), object.size())),
      properties, metadata);
}
