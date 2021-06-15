//===- BackendUtils.cpp - MLIR to C++ option parsing ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mlir rocm backend utils
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/ROCm/BackendUitls.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetRegistry.h"

// MC headers.
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"

// lld headers.
#include "lld/Common/Driver.h"

#include <mutex>

using namespace mlir;

static constexpr const char kRunnerProgram[] = "mlir-rocm-runner";
static constexpr const char kRocmAgentEnumerator[] = "rocm_agent_enumerator";

BackendUtils::BackendUtils(const std::string &defaultTriple,
                           const std::string &defaultChip,
                           const std::string &defaultFeatures)
    : triple(defaultTriple), chip(defaultChip), features(defaultFeatures) {
  setupDefaults(chip, features, triple);
}

BackendUtils::BackendUtils() : BackendUtils("", "", "") {}

LogicalResult BackendUtils::assembleIsa(const std::string isa, StringRef name,
                                        Blob &result,
                                        const std::string &tripleName,
                                        const std::string &targetChip,
                                        const std::string &features) {
  llvm::raw_svector_ostream os(result);

  std::string error;
  llvm::Triple theTriple(llvm::Triple::normalize(tripleName));
  const llvm::Target *theTarget =
      llvm::TargetRegistry::lookupTarget(theTriple.normalize(), error);
  if (!theTarget) {
    llvm::WithColor::error(llvm::errs(), name) << error;
    return failure();
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(isa),
                            llvm::SMLoc());

  const llvm::MCTargetOptions mcOptions;
  std::unique_ptr<llvm::MCRegisterInfo> mri(
      theTarget->createMCRegInfo(tripleName));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      theTarget->createMCAsmInfo(*mri, tripleName, mcOptions));
  mai->setRelaxELFRelocations(true);

  llvm::MCObjectFileInfo mofi;
  llvm::MCContext ctx(mai.get(), mri.get(), &mofi, &srcMgr, &mcOptions);
  mofi.InitMCObjectFileInfo(theTriple, false, ctx, false);

  SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  std::unique_ptr<llvm::MCStreamer> mcStreamer;
  std::unique_ptr<llvm::MCInstrInfo> mcii(theTarget->createMCInstrInfo());
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      theTarget->createMCSubtargetInfo(tripleName, targetChip, features));

  llvm::MCCodeEmitter *ce = theTarget->createMCCodeEmitter(*mcii, *mri, ctx);
  llvm::MCAsmBackend *mab =
      theTarget->createMCAsmBackend(*sti, *mri, mcOptions);
  mcStreamer.reset(theTarget->createMCObjectStreamer(
      theTriple, ctx, std::unique_ptr<llvm::MCAsmBackend>(mab),
      mab->createObjectWriter(os), std::unique_ptr<llvm::MCCodeEmitter>(ce),
      *sti, mcOptions.MCRelaxAll, mcOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));
  mcStreamer->setUseAssemblerInfoForParsing(true);

  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  std::unique_ptr<llvm::MCTargetAsmParser> tap(
      theTarget->createMCAsmParser(*sti, *parser, *mcii, mcOptions));

  if (!tap) {
    llvm::WithColor::error(llvm::errs(), name)
        << "assembler initialization error.\n";
    return failure();
  }

  parser->setTargetParser(*tap);
  parser->Run(false);

  return success();
}

LogicalResult BackendUtils::assembleIsa(const std::string isa, StringRef name,
                                        Blob &result) {
  return assembleIsa(isa, name, result, triple, chip, features);
}

static std::mutex mutex;
LogicalResult BackendUtils::createHsaco(const Blob &isaBlob, StringRef name,
                                        Blob &hsacoBlob) {
  const std::lock_guard<std::mutex> lock(mutex);
  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  std::error_code ec = llvm::sys::fs::createTemporaryFile(
      "kernel", "o", tempIsaBinaryFd, tempIsaBinaryFilename);
  if (ec) {
    llvm::WithColor::error(llvm::errs(), name)
        << "temporary file for ISA binary creation error.\n";
    return failure();
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);
  llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, true);
  tempIsaBinaryOs << isaBlob;
  tempIsaBinaryOs.close();

  // Create a temp file for HSA code object.
  int tempHsacoFD = -1;
  SmallString<128> tempHsacoFilename;
  ec = llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFD,
                                          tempHsacoFilename);
  if (ec) {
    llvm::WithColor::error(llvm::errs(), name)
        << "temporary file for HSA code object creation error.\n";
    return failure();
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  // Invoke lld. Expect a true return value from lld.
  bool ret = lld::elf::link({"ld.lld", "-shared", tempIsaBinaryFilename.c_str(),
                             "-o", tempHsacoFilename.c_str()},
                            /*canEarlyExit=*/false, llvm::outs(), llvm::errs());
  if (!ret) {
    llvm::WithColor::error(llvm::errs(), name) << "lld invocation error.\n";
    return failure();
  }

  // Load the HSA code object.
  auto hsacoFile = mlir::openInputFile(tempHsacoFilename);
  if (!hsacoFile) {
    llvm::WithColor::error(llvm::errs(), name)
        << "read HSA code object from temp file error.\n";
    return failure();
  }
  hsacoBlob.assign(hsacoFile->getBuffer().begin(),
                   hsacoFile->getBuffer().end());

  return success();
}

OwnedBlob BackendUtils::compileISAToHsaco(const std::string isa, Location loc,
                                          StringRef name) {
  // ISA -> ISA in binary form via MC.
  // Use lld to create HSA code object.
  Blob isaBlob;
  Blob hsacoBlob;

  if (succeeded(assembleIsa(isa, name, isaBlob)) &&
      succeeded(createHsaco(isaBlob, name, hsacoBlob)))
    return std::make_unique<std::vector<char>>(hsacoBlob.begin(),
                                               hsacoBlob.end());

  llvm::WithColor::error(llvm::errs(), name)
      << "producing HSA code object error.\n";
  return {};
}

std::unique_ptr<llvm::Module> BackendUtils::compileModuleToROCDLIR(
    Operation *m, llvm::LLVMContext &llvmContext, llvm::StringRef name) {
  StringRef amdgcnDataLayout =
      "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-"
      "v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:"
      "1024-v2048:2048-n32:64-S32-A5-ni:7";

  auto llvmModule = translateModuleToROCDLIR(m, llvmContext);
  llvmModule->setTargetTriple(triple);
  llvmModule->setDataLayout(amdgcnDataLayout);

  // TODO(whchung): Link with ROCm-Device-Libs in case needed (ex: the Module
  // depends on math functions).
  return llvmModule;
}

void BackendUtils::configTargetChip(std::string &targetChip) {
  // Locate rocm_agent_enumerator.
  llvm::ErrorOr<std::string> rocmAgentEnumerator = llvm::sys::findProgramByName(
      kRocmAgentEnumerator, {__ROCM_PATH__ "/bin"});
  std::error_code ec = rocmAgentEnumerator.getError();
  if (ec) {
    llvm::WithColor::warning(llvm::errs(), kRunnerProgram)
        << kRocmAgentEnumerator << " couldn't be located under "
        << __ROCM_PATH__ << ", set target as " << chip << "\n";
    return;
  }

  // Prepare temp file to hold the outputs.
  int tempFd = -1;
  SmallString<128> tempFilename;
  ec = llvm::sys::fs::createTemporaryFile("rocm_agent", "txt", tempFd,
                                          tempFilename);
  if (ec) {
    llvm::WithColor::warning(llvm::errs(), kRunnerProgram)
        << "temporary file for " << kRocmAgentEnumerator
        << " creation error, set target as " << chip << "\n";
    return;
  }
  llvm::FileRemover cleanup(tempFilename);

  // Invoke rocm_agent_enumerator.
  std::string errorMessage;
  SmallVector<StringRef, 2> args{"-t", "GPU"};
  Optional<StringRef> redirects[3] = {{""}, tempFilename.str(), {""}};
  int result =
      llvm::sys::ExecuteAndWait(rocmAgentEnumerator.get(), args, llvm::None,
                                redirects, 0, 0, &errorMessage);
  if (result) {
    llvm::WithColor::warning(llvm::errs(), kRunnerProgram)
        << kRocmAgentEnumerator << " invocation error: " << errorMessage
        << ", set target as " << chip << "\n";
    return;
  }

  // Load and parse the result.
  auto gfxIsaList = mlir::openInputFile(tempFilename);
  if (!gfxIsaList) {
    llvm::WithColor::error(llvm::errs(), kRunnerProgram)
        << "read ROCm agent list temp file error, set target as " << chip
        << "\n";
    return;
  }
  for (llvm::line_iterator lines(*gfxIsaList); !lines.is_at_end(); ++lines) {
    // Skip the line with content "gfx000".
    if (*lines == "gfx000")
      continue;
    // Use the first ISA version found.
    targetChip = lines->str();
    break;
  }
}

void BackendUtils::setupDefaults(std::string &chip, std::string &features,
                                 std::string &triple) {
  // Configure target chip ISA version if it has not been specified.
  if (!chip.size())
    configTargetChip(chip);

  if (!triple.size()) {
    triple = "amdgcn-amd-amdhsa";
  }

  // Configure target features per ROCm version, and target GPU.
  configTargetFeatures(chip, triple, features);
}

void BackendUtils::configTargetFeatures(const std::string &chip,
                                        const std::string &triple,
                                        std::string &features) {
  // For gfx908, add +sramecc by default to be compatible with ROCm 4.1+.
  if (chip == "gfx908") {
    features += "+sramecc";
  }
}
