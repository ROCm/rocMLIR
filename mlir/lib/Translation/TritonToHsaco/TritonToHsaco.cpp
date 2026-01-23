//===- TritonToHsaco.cpp - Convert Triton LLVM IR to HSACO binary --------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides:
// 1. A translation that converts Triton LLVM dialect IR to HSACO binary format
// 2. A pass wrapper that can be used in pipelines
//
// It implements the functionality from Triton's make_llir(), make_amdgcn(),
// and make_hsaco() in compiler.py.
//
//===----------------------------------------------------------------------===//

#include "mlir/Translation/TritonToHsaco.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "mlir/Pass/Pass.h"

#include <array>
#include <mutex>
#include <unordered_set>

// LLD for linking
#if MLIR_ENABLE_ROCM_CONVERSIONS
#include "lld/Common/Driver.h"
LLD_HAS_DRIVER(elf)
#endif

#define DEBUG_TYPE "triton-to-hsaco"

// Forward declaration for Triton's BreakStructPhiNodesPass (same as llvm.cc)
// Implementation is in lib/Target/LLVMIR/LLVMIRBreakPhiStruct.cpp
namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

// Forward declaration for Triton's scalarize pass
namespace mlir::triton::AMD {
void runScalarizePackedFOpsPass(llvm::Function &F);
}

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Initialize LLVM targets (call once) - from init_targets in llvm.cc
void initializeLLVMTargets() {
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  });
}

/// Create LLVM target machine - from createTargetMachine in llvm.cc
std::unique_ptr<llvm::TargetMachine> createTargetMachine(llvm::Module &module,
                                                         llvm::Triple &triple,
                                                         StringRef archStr,
                                                         StringRef features,
                                                         bool enableFpFusion) {
  std::string error;
  auto *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    llvm::errs() << "Target lookup failed: " << error << "\n";
    return nullptr;
  }

  llvm::TargetOptions opt;
  if (enableFpFusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;

  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple, archStr, features, opt, llvm::Reloc::PIC_, std::nullopt,
      llvm::CodeGenOptLevel::Aggressive));
}

/// Add control constant to module (for device library compatibility)
void addControlConstant(llvm::Module &module, const char *name, int bitwidth,
                        int value) {
  llvm::Type *type =
      llvm::IntegerType::getIntNTy(module.getContext(), bitwidth);
  auto *gv = new llvm::GlobalVariable(
      module, type, /*isConstant=*/true,
      llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
      llvm::ConstantInt::get(type, value), name, nullptr,
      llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, /*AddressSpace=*/4);
  gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
  gv->setVisibility(llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
  gv->setAlignment(llvm::MaybeAlign(bitwidth / 8));
}

/// Set ISA version control constant
void setISAVersion(llvm::Module &module, StringRef archStr) {
  llvm::AMDGPU::IsaVersion version = llvm::AMDGPU::getIsaVersion(archStr);
  int isaVersion =
      version.Major * 1000 + version.Minor * 100 + version.Stepping;
  addControlConstant(module, "__oclc_ISA_version", /*bitwidth=*/32, isaVersion);
}

/// Set ABI version control constant
void setABIVersion(llvm::Module &module, int version) {
  llvm::Type *i32Ty = llvm::Type::getInt32Ty(module.getContext());
  auto *gv = new llvm::GlobalVariable(
      module, i32Ty, /*isConstant=*/true,
      llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
      llvm::ConstantInt::get(i32Ty, version), "__oclc_ABI_version", nullptr,
      llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, /*AddressSpace=*/4);
  gv->setVisibility(llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
  gv->setAlignment(llvm::MaybeAlign(4));
  gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);

  module.addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                       version);
}

/// Set kernel function attributes
void setKernelAttributes(llvm::Module &module, StringRef archStr,
                         StringRef features, int numWarps, int wavesPerEU,
                         bool allowFlushDenorm, bool enableAsan,
                         StringRef scheduleHint) {
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(archStr);
  int waveSize = archInfo.waveSize;
  int totalThreads = numWarps * waveSize;

  llvm::Function *kernelFn = nullptr;
  for (llvm::Function &fn : module) {
    if (!fn.isDeclaration() && fn.hasExternalLinkage()) {
      kernelFn = &fn;
      break;
    }
  }

  if (!kernelFn)
    return;

  kernelFn->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  kernelFn->addFnAttr("amdgpu-flat-work-group-size",
                      "1," + std::to_string(totalThreads));

  // memory-bound-attention schedule hint enables iterative-ilp scheduler
  // (compiler.py lines 387-388)
  // TODO(roctriton): set this in ToBlockwise? or somewhere
  if (scheduleHint.contains("memory-bound-attention")) {
    kernelFn->addFnAttr("amdgpu-sched-strategy", "iterative-ilp");
  }

  kernelFn->addFnAttr("uniform-work-group-size", "true");

  if (wavesPerEU > 0) {
    std::string wavesStr =
        std::to_string(wavesPerEU) + ", " + std::to_string(wavesPerEU);
    kernelFn->addFnAttr("amdgpu-waves-per-eu", wavesStr);
  }

  std::string denormalMode = allowFlushDenorm ? "preserve-sign" : "ieee";
  kernelFn->addFnAttr("denormal-fp-math-f32", denormalMode);

  kernelFn->addFnAttr("target-features", features);
  // ASan support
  if (enableAsan) {
    kernelFn->addFnAttr(llvm::Attribute::SanitizeAddress);
  }

  // set_all_fn_arg_inreg in compiler.py
  if (!archStr.starts_with("gfx1250")) {
    for (llvm::Argument &arg : kernelFn->args()) {
      if (!arg.hasByRefAttr() && !arg.hasNestAttr()) {
        arg.addAttr(llvm::Attribute::InReg);
      }
    }
  }
}

/// Check if architecture has architected SGPRs
bool hasArchitectedSGPRs(llvm::Triple &triple, StringRef archStr) {
  std::string error;
  auto *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    return false;

  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(triple, archStr, ""));
  return sti && sti->checkFeatures("+architected-sgprs");
}

/// Link external device libraries (ocml, ockl, etc.)
bool linkExternLibs(llvm::Module &module,
                    const std::vector<std::string> &paths) {
  if (paths.empty())
    return true;

  llvm::LLVMContext &ctx = module.getContext();
  llvm::Linker linker(module);

  for (const std::string &path : paths) {
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> libMod = llvm::parseIRFile(path, err, ctx);
    if (!libMod) {
      llvm::errs() << "Failed to parse library at " << path << "\n";
      return false;
    }
    libMod->setTargetTriple(llvm::Triple(module.getTargetTriple()));
    libMod->setDataLayout(module.getDataLayout());

    std::unordered_set<std::string> externalFns;
    for (llvm::Function &fn : libMod->functions()) {
      if (!fn.isDeclaration())
        externalFns.insert(fn.getName().str());
    }

    if (linker.linkInModule(std::move(libMod),
                            llvm::Linker::Flags::LinkOnlyNeeded)) {
      llvm::errs() << "Failed to link library at " << path << "\n";
      return false;
    }

    // Mark linked-in functions as internal
    for (llvm::Function &fn : module.functions()) {
      if (externalFns.count(fn.getName().str())) {
        fn.setLinkage(llvm::GlobalValue::InternalLinkage);
      }
    }
  }
  return true;
}

static std::optional<llvm::OptimizationLevel> mapToLevel(unsigned optLevel) {
  switch (optLevel) {
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 2:
    return llvm::OptimizationLevel::O2;
  case 3:
    return llvm::OptimizationLevel::O3;
  }
  return std::nullopt;
}

/// Run LLVM optimization passes - matches optimize_module in llvm.cc
void optimizeModule(llvm::Module &module, llvm::TargetMachine *tm,
                    StringRef arch, llvm::OptimizationLevel optLevel,
                    bool enableAsan) {
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PipelineTuningOptions tuningOptions;
  tuningOptions.LoopUnrolling = true;
  tuningOptions.LoopInterleaving = true;
  tuningOptions.LoopVectorization = true;
  tuningOptions.SLPVectorization = true;

  llvm::PassBuilder pb(tm, tuningOptions);

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager mpm;

  // Register callback to add BreakStructPhiNodesPass before vectorization
  // This matches llvm.cc's registerVectorizerStartEPCallback
  pb.registerVectorizerStartEPCallback(
      [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
        // Triton generates large structure of scalars which may pessimise
        // optimizations, we run a pass to break up phi of struct to make
        // sure all the struct are removed for the following passes.
        fpm.addPass(llvm::BreakStructPhiNodesPass());
        fpm.addPass(llvm::InstCombinePass());
      });

  // Add address sanitizer if enabled
  if (enableAsan) {
    llvm::AddressSanitizerOptions asanOpts;
    mpm.addPass(llvm::AddressSanitizerPass(asanOpts));
  }

  mpm.addPass(pb.buildPerModuleDefaultPipeline(optLevel));
  mpm.run(module, mam);
}

/// Clean up metadata (cleanup_bitcode_metadata in compiler.py)
void cleanupBitcodeMetadata(llvm::Module &module) {
  if (auto *ident = module.getNamedMetadata("llvm.ident"))
    module.eraseNamedMetadata(ident);
  if (auto *openclVersion = module.getNamedMetadata("opencl.ocl.version"))
    module.eraseNamedMetadata(openclVersion);
}

/// Disable inlining of print related functions (disable_print_inline)
void disablePrintInline(llvm::Module &module) {
  // List of functions name prefixes we want to forbid inline.
  std::array<const char *, 2> prefixes = {"__ockl_fprintf", "__ockl_printf"};

  for (llvm::Function &f : module) {
    if (!f.hasName())
      continue;
    llvm::StringRef name = f.getName();

    auto isNamePrefixed = [&name](const char *prefix) {
      return name.starts_with(prefix);
    };

    if (llvm::any_of(prefixes, isNamePrefixed))
      f.addFnAttr(llvm::Attribute::NoInline);
  }
}

//===----------------------------------------------------------------------===//
// make_amdgcn - LLVM IR to AMDGCN assembly (compiler.py lines 452-473)
//===----------------------------------------------------------------------===//

std::string translateLLVMIRToASM(llvm::Module &module,
                                 llvm::TargetMachine *machine) {
  using namespace mlir;

  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager pass;
    // emit
    machine->addPassesToEmitFile(pass, pstream, nullptr,
                                 llvm::CodeGenFileType::AssemblyFile);
    pass.run(module);
  }
  return result;
}

/// Translate LLVM IR module to AMDGCN assembly string
std::string makeAMDGCN(llvm::Module &module, llvm::TargetMachine *tm) {
  return translateLLVMIRToASM(module, tm);
}

//===----------------------------------------------------------------------===//
// make_hsaco - AMDGCN assembly to HSACO binary (compiler.py lines 476-488)
//===----------------------------------------------------------------------===//

/// Assemble AMDGCN assembly to object code (amd.assemble_amdgcn)
std::optional<SmallVector<char, 0>> assembleAMDGCN(StringRef assembly,
                                                   llvm::Triple &triple,
                                                   StringRef archStr,
                                                   StringRef features) {
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    llvm::errs() << "Target lookup error: " << error << "\n";
    return std::nullopt;
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(assembly),
                            llvm::SMLoc());

  llvm::MCTargetOptions mcOptions;
  std::unique_ptr<llvm::MCRegisterInfo> mri(target->createMCRegInfo(triple));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, triple, mcOptions));
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(triple, archStr, features));

  llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                      &mcOptions);
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(
      target->createMCObjectFileInfo(ctx, /*PIC=*/false,
                                     /*LargeCodeModel=*/false));
  ctx.setObjectFileInfo(mofi.get());

  llvm::SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  llvm::SmallVector<char, 0> result;
  llvm::raw_svector_ostream svos(result);

  std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());
  std::unique_ptr<llvm::MCCodeEmitter> ce(
      target->createMCCodeEmitter(*mcii, ctx));
  std::unique_ptr<llvm::MCAsmBackend> mab(
      target->createMCAsmBackend(*sti, *mri, mcOptions));
  std::unique_ptr<llvm::MCObjectWriter> ow(mab->createObjectWriter(svos));
  std::unique_ptr<llvm::MCStreamer> mcStreamer(target->createMCObjectStreamer(
      triple, ctx, std::move(mab), std::move(ow), std::move(ce), *sti));

  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  std::unique_ptr<llvm::MCTargetAsmParser> tap(
      target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));
  if (!tap) {
    llvm::errs() << "Assembler initialization error\n";
    return std::nullopt;
  }

  parser->setTargetParser(*tap);
  parser->Run(/*NoInitialTextSection=*/false);

  return SmallVector<char, 0>(result.begin(), result.end());
}

/// Invoke LLD to link object file to HSACO - matches triton_amd.cc lldInvoke
static std::optional<std::string> lldInvoke(const char *inPath,
                                            const char *outPath) {
#if MLIR_ENABLE_ROCM_CONVERSIONS
  // Workaround: Disable parallelism to avoid hangs caused by LLVM's thread pool
  // when the following code is executed in a forked child process.
  // Context: lld::elf::LinkerDriver::link uses parallelFor which uses the
  // LLVM's thread pool. During cleanup at ~TaskGroup() the child process hangs
  // waiting.
  //
  // IMPORTANT: LLD's CommonLinkerContext uses a global static pointer (not
  // thread_local) to store the linker context. This means lldMain is NOT
  // thread-safe - concurrent calls will race on the global context pointer.
  // We must serialize all LLD invocations with a mutex.
  static std::mutex lldMutex;
  std::lock_guard<std::mutex> lock(lldMutex);

  std::array<const char *, 6> args = {"ld.lld", "--threads=1", "-shared",
                                      inPath,   "-o",          outPath};
  std::string errString;
  llvm::raw_string_ostream errStream(errString);
  auto lldRes = lld::lldMain(args, llvm::outs(), llvm::errs(),
                             {{lld::Gnu, &lld::elf::link}});
  bool noErrors = (!lldRes.retCode && lldRes.canRunAgain);
  if (!noErrors) {
    errStream.flush();
    return errString;
  }
  return std::nullopt;
#else
  return "ROCM conversions not enabled";
#endif
}

/// Link object file to HSACO using LLD (amd.link_hsaco)
std::optional<SmallVector<char, 0>> linkHSACO(ArrayRef<char> objectCode) {
#if MLIR_ENABLE_ROCM_CONVERSIONS
  int tempObjFd = -1;
  llvm::SmallString<128> tempObjFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "o", tempObjFd,
                                         tempObjFilename)) {
    llvm::errs() << "Failed to create temporary object file\n";
    return std::nullopt;
  }
  llvm::FileRemover cleanupObj(tempObjFilename);
  {
    llvm::raw_fd_ostream tempObjOs(tempObjFd, true);
    tempObjOs << StringRef(objectCode.data(), objectCode.size());
    tempObjOs.flush();
  }

  llvm::SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco",
                                         tempHsacoFilename)) {
    llvm::errs() << "Failed to create temporary HSACO file\n";
    return std::nullopt;
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  // Use lldMain for safe re-entry support (matches triton_amd.cc)
  auto errOpt = lldInvoke(tempObjFilename.c_str(), tempHsacoFilename.c_str());
  if (errOpt) {
    llvm::errs() << "LLD invocation error: " << *errOpt << "\n";
    return std::nullopt;
  }

  auto hsacoFile =
      llvm::MemoryBuffer::getFile(tempHsacoFilename, /*IsText=*/false);
  if (!hsacoFile) {
    llvm::errs() << "Failed to read HSACO file\n";
    return std::nullopt;
  }

  StringRef buffer = (*hsacoFile)->getBuffer();
  return SmallVector<char, 0>(buffer.begin(), buffer.end());
#else
  llvm::errs() << "ROCM conversions not enabled. Rebuild with "
                  "MLIR_ENABLE_ROCM_CONVERSIONS=1\n";
  return std::nullopt;
#endif
}

/// Convert AMDGCN assembly to HSACO binary - make_hsaco in compiler.py
std::optional<SmallVector<char, 0>> makeHSACO(StringRef amdgcnAsm,
                                              llvm::Triple &triple,
                                              StringRef archStr,
                                              StringRef features) {
  // Assemble to object code
  auto objectCode = assembleAMDGCN(amdgcnAsm, triple, archStr, features);
  if (!objectCode) {
    return std::nullopt;
  }

  // Link to HSACO
  return linkHSACO(*objectCode);
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

namespace mlir {
namespace rock {

FailureOr<llvm::SmallVector<char, 0>>
translateTritonToHsaco(ModuleOp module, const TritonToHsacoOptions &options) {
  initializeLLVMTargets();

  // Note: Translation interfaces must be registered before running the pass
  // pipeline. They are registered in:
  // 1. registerTritonToHsacoTranslation() for standalone translation use
  // 2. InitRocMLIRDialects.h for rocmlir-driver and other tools

  // Translate MLIR to LLVM IR (llvm.to_module in compiler.py)
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate module to LLVM IR\n";
    return failure();
  }

  StringRef arch = options.arch;
  std::string features = options.features;
  bool enableAsan = (StringRef(options.features).contains("+xnack"));

  auto triple = llvm::Triple(options.triple);
  // Set target triple and data layout (attach_target_triple in compiler.py)
  llvmModule->setTargetTriple(triple);

  // attach_datalayout in compiler.py
  auto tm = createTargetMachine(*llvmModule, triple, arch, features,
                                options.enableFpFusion);
  if (!tm) {
    return failure();
  }
  llvmModule->setDataLayout(tm->createDataLayout());

  // Set AMD-specific control constants
  setISAVersion(*llvmModule, arch);
  setABIVersion(*llvmModule, 500);

  AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  int waveSize = archInfo.waveSize;
  addControlConstant(*llvmModule, "__oclc_finite_only_opt", 8, 0);
  addControlConstant(*llvmModule, "__oclc_correctly_rounded_sqrt32", 8, 1);
  addControlConstant(*llvmModule, "__oclc_unsafe_math_opt", 8, 0);
  addControlConstant(*llvmModule, "__oclc_wavefrontsize64", 8, waveSize == 64);

  int numWarps = options.numWarps;
  if(auto totalNumWarps = module->getAttrOfType<IntegerAttr>("ttg.total-num-warps")) {
    if(numWarps != totalNumWarps.getInt()) {
      LLVM_DEBUG(llvm::dbgs() << "ttg.total-num-warps != rock.num_waves ("<<totalNumWarps.getInt()<<" != "<<numWarps<<")\n");
      LLVM_DEBUG(llvm::dbgs() << "This can happen due to warp-specialization\n");
    }
    numWarps = totalNumWarps.getInt();
  }

  // Set kernel attributes (including schedule_hint for memory-bound-attention)
  setKernelAttributes(*llvmModule, arch, features, numWarps,
                      options.wavesPerEU, options.allowFlushDenorm,
                      enableAsan, options.scheduleHint);

  // Link external device libraries (ocml.bc, ockl.bc, asanrtl.bc, etc.)
  // compiler.py lines 412-423
  if (!options.externLibPaths.empty()) {
    if (!linkExternLibs(*llvmModule, options.externLibPaths)) {
      llvm::errs() << "Failed to link external libraries\n";
      return failure();
    }
  }

  std::optional<llvm::OptimizationLevel> optLevel =
      mapToLevel(options.optLevel);
  if (!optLevel.has_value()) {
    llvm::errs() << "Invalid optimization level: " << options.optLevel << "\n";
    return failure();
  }

  // optimize_module in llvm.cc
  optimizeModule(*llvmModule, tm.get(), arch, optLevel.value(),
                 enableAsan);

  // Handle architected SGPRs (compiler.py lines 427-434)
  if (hasArchitectedSGPRs(triple, arch)) {
    for (llvm::Function &fn : *llvmModule) {
      if (!fn.isDeclaration() && fn.hasExternalLinkage()) {
        fn.removeFnAttr("amdgpu-no-workgroup-id-x");
        fn.removeFnAttr("amdgpu-no-workgroup-id-y");
        fn.removeFnAttr("amdgpu-no-workgroup-id-z");
        break;
      }
    }
  }

  // scalarize_packed_fops (compiler.py line 436-437)
  if (options.scalarizePackedFops) {
    for (llvm::Function &fn : *llvmModule) {
      if (!fn.isDeclaration() && fn.hasExternalLinkage()) {
        mlir::triton::AMD::runScalarizePackedFOpsPass(fn);
        break;
      }
    }
  }

  // cleanup_bitcode_metadata in compiler.py
  cleanupBitcodeMetadata(*llvmModule);

  // disable_print_inline in compiler.py
  disablePrintInline(*llvmModule);

  // make_amdgcn (compiler.py lines 452-473)
  // Get features for assembly
  std::string asmFeatures(features);
  if (arch.contains("gfx11")) {
    asmFeatures += "-real-true16";
  }
  // Recreate target machine with proper features for codegen
  auto tmAsm = createTargetMachine(*llvmModule, triple, arch, asmFeatures,
                                   options.enableFpFusion);
  if (!tmAsm) {
    return failure();
  }

  std::string amdgcnAsm = makeAMDGCN(*llvmModule, tmAsm.get());
  if (amdgcnAsm.empty()) {
    llvm::errs() << "Failed to generate AMDGCN assembly\n";
    return failure();
  }

  // make_hsaco (compiler.py lines 476-488)
  auto hsaco = makeHSACO(amdgcnAsm, triple, arch, features);
  if (!hsaco) {
    return failure();
  }

  return llvm::SmallVector<char, 0>(hsaco->begin(), hsaco->end());
}

void registerTritonToHsacoTranslation() {
  TranslateFromMLIRRegistration registration(
      "triton-to-hsaco", "Translate Triton LLVM IR to HSACO binary",
      [](ModuleOp module, raw_ostream &output) {
        // Default options - in practice these would come from command line
        TritonToHsacoOptions options;
        auto hsacoOrErr = translateTritonToHsaco(module, options);
        if (failed(hsacoOrErr))
          return failure();
        output.write(hsacoOrErr->data(), hsacoOrErr->size());
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect>();
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerGPUDialectTranslation(registry);
        mlir::registerROCDLDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
      });
}

} // namespace rock
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pass Wrapper
//===----------------------------------------------------------------------===//

namespace mlir {
namespace rock {

#define GEN_PASS_DEF_TRITONTOHSACOPASS
#include "mlir/Dialect/Rock/Passes.h.inc"

namespace {

/// Pass wrapper that calls the TritonToHsaco translation.
/// This allows the translation to be used in pass pipelines.
class TritonToHsacoPass
    : public impl::TritonToHsacoPassBase<TritonToHsacoPass> {
public:
  using TritonToHsacoPassBase::TritonToHsacoPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Build options from pass parameters
    TritonToHsacoOptions options;
    options.triple = triple.getValue();
    options.arch = arch.getValue();
    options.features = features.getValue();
    options.optLevel = optLevel.getValue();
    options.numWarps = numWarps.getValue();
    options.wavesPerEU = wavesPerEU.getValue();
    options.enableFpFusion = enableFpFusion.getValue();
    options.allowFlushDenorm = allowFlushDenorm.getValue();
    options.scalarizePackedFops = scalarizePackedFops.getValue();
    options.scheduleHint = scheduleHint.getValue();

    // Call the translation
    auto hsacoOrErr = translateTritonToHsaco(module, options);
    if (failed(hsacoOrErr)) {
      signalPassFailure();
      return;
    }

    // Store the HSACO binary as a module attribute
    llvm::SmallVector<char, 0> &hsaco = *hsacoOrErr;
    auto hsacoAttr = StringAttr::get(module.getContext(),
                                     StringRef(hsaco.data(), hsaco.size()));
    module->setAttr("triton.hsaco", hsacoAttr);
  }
};

} // namespace
} // namespace rock
} // namespace mlir
