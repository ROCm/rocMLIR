/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2003-2017 University of Illinois at Urbana-Champaign.
 * Modifications (c) 2018 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of the LLVM Team, University of Illinois at
 *       Urbana-Champaign, nor the names of its contributors may be used to
 *       endorse or promote products derived from this Software without specific
 *       prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "comgr-compiler.h"
#include "comgr-device-libs.h"
#include "comgr-env.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/OffloadBundler.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/FrontendTool/Utils.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
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
#include "llvm/Object/Archive.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Host.h"

#include "time-stat/ts-interface.h"

#include <csignal>

LLD_HAS_DRIVER(elf)

using namespace llvm;
using namespace llvm::opt;
using namespace llvm::sys;
using namespace clang;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace TimeStatistics;

namespace COMGR {

namespace {
static constexpr llvm::StringLiteral LinkerJobName = "amdgpu::Linker";

/// \brief Helper class for representing a single invocation of the assembler.
struct AssemblerInvocation {
  /// @name Target Options
  /// @{

  /// The name of the target triple to assemble for.
  std::string Triple;

  /// If given, the name of the target CPU to determine which instructions
  /// are legal.
  std::string CPU;

  /// The list of target specific features to enable or disable -- this should
  /// be a list of strings starting with '+' or '-'.
  std::vector<std::string> Features;

  /// The list of symbol definitions.
  std::vector<std::string> SymbolDefs;

  /// @}
  /// @name Language Options
  /// @{

  std::vector<std::string> IncludePaths;
  unsigned NoInitialTextSection : 1;
  unsigned SaveTemporaryLabels : 1;
  unsigned GenDwarfForAssembly : 1;
  unsigned RelaxELFRelocations : 1;
  unsigned DwarfVersion;
  std::string DwarfDebugFlags;
  std::string DwarfDebugProducer;
  std::string DebugCompilationDir;
  llvm::DebugCompressionType CompressDebugSections =
      llvm::DebugCompressionType::None;
  std::string MainFileName;

  /// @}
  /// @name Frontend Options
  /// @{

  std::string InputFile;
  std::vector<std::string> LLVMArgs;
  std::string OutputPath;
  enum FileType {
    FT_Asm,  ///< Assembly (.s) output, transliterate mode.
    FT_Null, ///< No output, for timing purposes.
    FT_Obj   ///< Object file output.
  };
  FileType OutputType;
  unsigned ShowHelp : 1;
  unsigned ShowVersion : 1;

  /// @}
  /// @name Transliterate Options
  /// @{

  unsigned OutputAsmVariant;
  unsigned ShowEncoding : 1;
  unsigned ShowInst : 1;

  /// @}
  /// @name Assembler Options
  /// @{

  unsigned RelaxAll : 1;
  unsigned NoExecStack : 1;
  unsigned FatalWarnings : 1;
  unsigned IncrementalLinkerCompatible : 1;

  /// The name of the relocation model to use.
  std::string RelocationModel;

  /// @}

public:
  AssemblerInvocation() {
    Triple = "";
    NoInitialTextSection = 0;
    InputFile = "-";
    OutputPath = "-";
    OutputType = FT_Asm;
    OutputAsmVariant = 0;
    ShowInst = 0;
    ShowEncoding = 0;
    RelaxAll = 0;
    NoExecStack = 0;
    FatalWarnings = 0;
    IncrementalLinkerCompatible = 0;
    DwarfVersion = 0;
  }

  static bool createFromArgs(AssemblerInvocation &Res,
                             ArrayRef<const char *> Argv,
                             DiagnosticsEngine &Diags);
};
} // namespace

bool AssemblerInvocation::createFromArgs(AssemblerInvocation &Opts,
                                         ArrayRef<const char *> Argv,
                                         DiagnosticsEngine &Diags) {
  bool Success = true;

  // Parse the arguments.
  const OptTable &OptTbl = getDriverOptTable();

  const unsigned IncludedFlagsBitmask = options::CC1AsOption;
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList Args = OptTbl.ParseArgs(Argv, MissingArgIndex, MissingArgCount,
                                       IncludedFlagsBitmask);

  // Check for missing argument error.
  if (MissingArgCount) {
    Diags.Report(diag::err_drv_missing_argument)
        << Args.getArgString(MissingArgIndex) << MissingArgCount;
    Success = false;
  }

  // Issue errors on unknown arguments.
  for (const Arg *A : Args.filtered(OPT_UNKNOWN)) {
    auto ArgString = A->getAsString(Args);
    std::string Nearest;
    if (OptTbl.findNearest(ArgString, Nearest, IncludedFlagsBitmask) > 1) {
      Diags.Report(diag::err_drv_unknown_argument) << ArgString;
    } else {
      Diags.Report(diag::err_drv_unknown_argument_with_suggestion)
          << ArgString << Nearest;
    }
    Success = false;
  }

  // Construct the invocation.

  // Target Options
  Opts.Triple = llvm::Triple::normalize(Args.getLastArgValue(OPT_triple));
  Opts.CPU = std::string(Args.getLastArgValue(OPT_target_cpu));
  Opts.Features = Args.getAllArgValues(OPT_target_feature);

  // Use the default target triple if unspecified.
  if (Opts.Triple.empty()) {
    Opts.Triple = llvm::sys::getDefaultTargetTriple();
  }

  // Language Options
  Opts.IncludePaths = Args.getAllArgValues(OPT_I);
  Opts.NoInitialTextSection = Args.hasArg(OPT_n);
  Opts.SaveTemporaryLabels = Args.hasArg(OPT_msave_temp_labels);
  // Any DebugInfoKind implies GenDwarfForAssembly.
  Opts.GenDwarfForAssembly = Args.hasArg(OPT_debug_info_kind_EQ);

  if (const Arg *A = Args.getLastArg(OPT_compress_debug_sections,
                                     OPT_compress_debug_sections_EQ)) {
    if (A->getOption().getID() == OPT_compress_debug_sections) {
      // TODO: be more clever about the compression type auto-detection
      Opts.CompressDebugSections = llvm::DebugCompressionType::Zlib;
    } else {
      Opts.CompressDebugSections =
          llvm::StringSwitch<llvm::DebugCompressionType>(A->getValue())
              .Case("none", llvm::DebugCompressionType::None)
              .Case("zlib", llvm::DebugCompressionType::Zlib)
              .Default(llvm::DebugCompressionType::None);
    }
  }

  Opts.RelaxELFRelocations = !Args.hasArg(OPT_mrelax_relocations_no);
  Opts.DwarfVersion = getLastArgIntValue(Args, OPT_dwarf_version_EQ, 2, Diags);
  Opts.DwarfDebugFlags =
      std::string(Args.getLastArgValue(OPT_dwarf_debug_flags));
  Opts.DwarfDebugProducer =
      std::string(Args.getLastArgValue(OPT_dwarf_debug_producer));
  Opts.DebugCompilationDir =
      std::string(Args.getLastArgValue(OPT_fdebug_compilation_dir));
  Opts.MainFileName = std::string(Args.getLastArgValue(OPT_main_file_name));

  // Frontend Options
  if (Args.hasArg(OPT_INPUT)) {
    bool First = true;
    for (const Arg *A : Args.filtered(OPT_INPUT)) {
      if (First) {
        Opts.InputFile = A->getValue();
        First = false;
      } else {
        Diags.Report(diag::err_drv_unknown_argument) << A->getAsString(Args);
        Success = false;
      }
    }
  }
  Opts.LLVMArgs = Args.getAllArgValues(OPT_mllvm);
  Opts.OutputPath = std::string(Args.getLastArgValue(OPT_o));
  if (Arg *A = Args.getLastArg(OPT_filetype)) {
    StringRef Name = A->getValue();
    unsigned OutputType = StringSwitch<unsigned>(Name)
                              .Case("asm", FT_Asm)
                              .Case("null", FT_Null)
                              .Case("obj", FT_Obj)
                              .Default(~0U);
    if (OutputType == ~0U) {
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.OutputType = FileType(OutputType);
    }
  }
  Opts.ShowHelp = Args.hasArg(OPT_help);
  Opts.ShowVersion = Args.hasArg(OPT_version);

  // Transliterate Options
  Opts.OutputAsmVariant =
      getLastArgIntValue(Args, OPT_output_asm_variant, 0, Diags);
  Opts.ShowEncoding = Args.hasArg(OPT_show_encoding);
  Opts.ShowInst = Args.hasArg(OPT_show_inst);

  // Assemble Options
  Opts.RelaxAll = Args.hasArg(OPT_mrelax_all);
  Opts.NoExecStack = Args.hasArg(OPT_mno_exec_stack);
  Opts.FatalWarnings = Args.hasArg(OPT_massembler_fatal_warnings);
  Opts.RelocationModel =
      std::string(Args.getLastArgValue(OPT_mrelocation_model, "pic"));
  Opts.IncrementalLinkerCompatible =
      Args.hasArg(OPT_mincremental_linker_compatible);
  Opts.SymbolDefs = Args.getAllArgValues(OPT_defsym);

  return Success;
}

static std::unique_ptr<raw_fd_ostream>
getOutputStream(AssemblerInvocation &Opts, DiagnosticsEngine &Diags,
                bool Binary) {
  if (Opts.OutputPath.empty()) {
    Opts.OutputPath = "-";
  }

  // Make sure that the Out file gets unlinked from the disk if we get a
  // SIGINT.
  if (Opts.OutputPath != "-") {
    sys::RemoveFileOnSignal(Opts.OutputPath);
  }

  std::error_code EC;
  auto Out = std::make_unique<raw_fd_ostream>(
      Opts.OutputPath, EC, (Binary ? sys::fs::OF_None : sys::fs::OF_Text));
  if (EC) {
    Diags.Report(diag::err_fe_unable_to_open_output)
        << Opts.OutputPath << EC.message();
    return nullptr;
  }

  return Out;
}

static bool executeAssemblerImpl(AssemblerInvocation &Opts,
                                 DiagnosticsEngine &Diags, raw_ostream &LogS) {
  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(Opts.Triple, Error);
  if (!TheTarget) {
    return Diags.Report(diag::err_target_unknown_triple) << Opts.Triple;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      MemoryBuffer::getFileOrSTDIN(Opts.InputFile);

  if (std::error_code EC = Buffer.getError()) {
    Error = EC.message();
    return Diags.Report(diag::err_fe_error_reading) << Opts.InputFile;
  }

  SourceMgr SrcMgr;
  SrcMgr.setDiagHandler(
      [](const SMDiagnostic &SMDiag, void *LogS) {
        SMDiag.print("", *(raw_ostream *)LogS, /* ShowColors */ false);
      },
      &LogS);

  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  SrcMgr.AddNewSourceBuffer(std::move(*Buffer), SMLoc());

  // Record the location of the include directories so that the lexer can find
  // it later.
  SrcMgr.setIncludeDirs(Opts.IncludePaths);

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(Opts.Triple));
  assert(MRI && "Unable to create target register info!");

  llvm::MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, Opts.Triple, MCOptions));
  assert(MAI && "Unable to create target asm info!");

  // Ensure MCAsmInfo initialization occurs before any use, otherwise sections
  // may be created with a combination of default and explicit settings.
  MAI->setCompressDebugSections(Opts.CompressDebugSections);

  MAI->setRelaxELFRelocations(Opts.RelaxELFRelocations);

  bool IsBinary = Opts.OutputType == AssemblerInvocation::FT_Obj;
  std::unique_ptr<raw_fd_ostream> FDOS = getOutputStream(Opts, Diags, IsBinary);
  if (!FDOS) {
    return true;
  }

  // Build up the feature string from the target feature list.
  std::string FS;
  if (!Opts.Features.empty()) {
    FS = Opts.Features[0];
    for (unsigned I = 1, E = Opts.Features.size(); I != E; ++I) {
      FS += "," + Opts.Features[I];
    }
  }

  std::unique_ptr<MCObjectFileInfo> MOFI(new MCObjectFileInfo());
  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(Opts.Triple, Opts.CPU, FS));

  MCContext Ctx(Triple(Opts.Triple), MAI.get(), MRI.get(),
                STI.get(), &SrcMgr);
  Ctx.setObjectFileInfo(MOFI.get());

  bool PIC = false;
  if (Opts.RelocationModel == "static") {
    PIC = false;
  } else if (Opts.RelocationModel == "pic") {
    PIC = true;
  } else {
    assert(Opts.RelocationModel == "dynamic-no-pic" && "Invalid PIC model!");
    PIC = false;
  }

  MOFI->initMCObjectFileInfo(Ctx, PIC);
  if (Opts.SaveTemporaryLabels) {
    Ctx.setAllowTemporaryLabels(false);
  }
  if (Opts.GenDwarfForAssembly) {
    Ctx.setGenDwarfForAssembly(true);
  }
  if (!Opts.DwarfDebugFlags.empty()) {
    Ctx.setDwarfDebugFlags(StringRef(Opts.DwarfDebugFlags));
  }
  if (!Opts.DwarfDebugProducer.empty()) {
    Ctx.setDwarfDebugProducer(StringRef(Opts.DwarfDebugProducer));
  }
  if (!Opts.DebugCompilationDir.empty()) {
    Ctx.setCompilationDir(Opts.DebugCompilationDir);
  }
  if (!Opts.MainFileName.empty()) {
    Ctx.setMainFileName(StringRef(Opts.MainFileName));
  }
  Ctx.setDwarfVersion(Opts.DwarfVersion);

  std::unique_ptr<MCStreamer> Str;
  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());

  raw_pwrite_stream *Out = FDOS.get();
  std::unique_ptr<buffer_ostream> BOS;

  // FIXME: There is a bit of code duplication with addPassesToEmitFile.
  if (Opts.OutputType == AssemblerInvocation::FT_Asm) {
    MCInstPrinter *IP = TheTarget->createMCInstPrinter(
        llvm::Triple(Opts.Triple), Opts.OutputAsmVariant, *MAI, *MCII, *MRI);
    std::unique_ptr<MCCodeEmitter> MCE;
    std::unique_ptr<MCAsmBackend> MAB;
    if (Opts.ShowEncoding) {
      MCE.reset(TheTarget->createMCCodeEmitter(*MCII, Ctx));
      MCTargetOptions Options;
      MAB.reset(TheTarget->createMCAsmBackend(*STI, *MRI, Options));
    }
    auto FOut = std::make_unique<formatted_raw_ostream>(*Out);
    Str.reset(TheTarget->createAsmStreamer(
        Ctx, std::move(FOut), /*asmverbose*/ true,
        /*useDwarfDirectory*/ true, IP, std::move(MCE), std::move(MAB),
        Opts.ShowInst));
  } else if (Opts.OutputType == AssemblerInvocation::FT_Null) {
    Str.reset(createNullStreamer(Ctx));
  } else {
    assert(Opts.OutputType == AssemblerInvocation::FT_Obj &&
           "Invalid file type!");
    if (!FDOS->supportsSeeking()) {
      BOS = std::make_unique<buffer_ostream>(*FDOS);
      Out = BOS.get();
    }

    MCCodeEmitter *CE = TheTarget->createMCCodeEmitter(*MCII, Ctx);
    MCTargetOptions Options;
    MCAsmBackend *MAB = TheTarget->createMCAsmBackend(*STI, *MRI, Options);
    Triple T(Opts.Triple);
    Str.reset(TheTarget->createMCObjectStreamer(
        T, Ctx, std::unique_ptr<MCAsmBackend>(MAB),
        MAB->createObjectWriter(*Out), std::unique_ptr<MCCodeEmitter>(CE), *STI,
        Opts.RelaxAll, Opts.IncrementalLinkerCompatible,
        /*DWARFMustBeAtTheEnd*/ true));
    Str.get()->initSections(Opts.NoExecStack, *STI);
  }

  bool Failed = false;

  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(SrcMgr, Ctx, *Str.get(), *MAI));

  // FIXME: init MCTargetOptions from sanitizer flags here.
  MCTargetOptions Options;
  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(*STI, *Parser, *MCII, Options));
  if (!TAP) {
    Failed = Diags.Report(diag::err_target_unknown_triple) << Opts.Triple;
  }

  // Set values for symbols, if any.
  for (auto &S : Opts.SymbolDefs) {
    auto Pair = StringRef(S).split('=');
    auto Sym = Pair.first;
    auto Val = Pair.second;
    int64_t Value;
    // We have already error checked this in the driver.
    if (!Val.getAsInteger(0, Value)) {
      Ctx.setSymbolValue(Parser->getStreamer(), Sym, Value);
    }
  }

  if (!Failed) {
    Parser->setTargetParser(*TAP.get());
    Failed = Parser->Run(Opts.NoInitialTextSection);
  }

  return Failed;
}

static bool executeAssembler(AssemblerInvocation &Opts,
                             DiagnosticsEngine &Diags, raw_ostream &LogS) {
  bool Failed = executeAssemblerImpl(Opts, Diags, LogS);

  // Delete output file if there were errors.
  if (Failed && Opts.OutputPath != "-") {
    sys::fs::remove(Opts.OutputPath);
  }

  return Failed;
}

static SmallString<128> getFilePath(DataObject *Object, StringRef Dir) {
  SmallString<128> Path(Dir);
  path::append(Path, Object->Name);

  // Create directories specified in the File Path so that the in-process driver
  // can successfully execute clang commands that use this file path as an
  // output argument
  if (fs::create_directories(path::parent_path(Path))) {
    return SmallString<128>();
  }

  return Path;
}

static amd_comgr_status_t inputFromFile(DataObject *Object, StringRef Path) {
  ProfilePoint Point("FileIO");
  auto BufOrError = MemoryBuffer::getFile(Path);
  if (std::error_code EC = BufOrError.getError()) {
    return AMD_COMGR_STATUS_ERROR;
  }
  Object->setData(BufOrError.get()->getBuffer());
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t outputToFile(StringRef Data, StringRef Path) {
  SmallString<128> DirPath = Path;
  path::remove_filename(DirPath);
  {
    ProfilePoint Point("CreateDir");
    if (fs::create_directories(DirPath)) {
      return AMD_COMGR_STATUS_ERROR;
    }
  }
  std::error_code EC;
  ProfilePoint Point("FileIO");
  raw_fd_ostream OS(Path, EC, fs::OF_None);
  if (EC) {
    return AMD_COMGR_STATUS_ERROR;
  }
  OS << Data;
  OS.close();
  if (OS.has_error()) {
    return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t outputToFile(DataObject *Object, StringRef Path) {
  return outputToFile(StringRef(Object->Data, Object->Size), Path);
}

static void initializeCommandLineArgs(SmallVectorImpl<const char *> &Args) {
  // Workaround for flawed Driver::BuildCompilation(...) implementation,
  // which eliminates 1st argument, cause it actually awaits argv[0].
  Args.clear();
  Args.push_back("");
}

// Parse -mllvm options
static amd_comgr_status_t
parseLLVMOptions(const std::vector<std::string> &Options) {
  std::vector<const char *> LLVMArgs;
  for (auto Option : Options) {
    LLVMArgs.push_back("");
    LLVMArgs.push_back(Option.c_str());
    if (!cl::ParseCommandLineOptions(LLVMArgs.size(), &LLVMArgs[0],
                                     "-mllvm options parsing")) {
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
    LLVMArgs.clear();
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t linkWithLLD(llvm::ArrayRef<const char *> Args,
                                      llvm::raw_ostream &LogS,
                                      llvm::raw_ostream &LogE) {
  ArgStringList LLDArgs(llvm::iterator_range<ArrayRef<const char *>::iterator>(
      Args.begin(), Args.end()));
  LLDArgs.insert(LLDArgs.begin(), "ld.lld");
  LLDArgs.push_back("--threads=1");

  ArrayRef<const char *> ArgRefs = llvm::ArrayRef(LLDArgs);
  lld::Result LLDRet =
      lld::lldMain(ArgRefs, LogS, LogE, {{lld::Gnu, &lld::elf::link}});
  lld::CommonLinkerContext::destroy();
  if (LLDRet.retCode || !LLDRet.canRunAgain) {
    return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static void logArgv(raw_ostream &OS, StringRef ProgramName,
                    ArrayRef<const char *> Argv) {
  OS << "     Driver Job Args: " << ProgramName;
  for (size_t I = 0; I < Argv.size(); ++I) {
    // Skip the first argument, which we replace with ProgramName, and the last
    // argument, which is a null terminator.
    if (I && Argv[I]) {
      OS << " \"" << Argv[I] << '\"';
    }
  }
  OS << '\n';
  OS.flush();
}

amd_comgr_status_t
AMDGPUCompiler::executeInProcessDriver(ArrayRef<const char *> Args) {
  // A DiagnosticsEngine is required at several points:
  //  * By the Driver in order to diagnose option parsing.
  //  * By the CompilerInvocation in order to diagnose option parsing.
  //  * By the CompilerInstance in order to diagnose everything else.
  // It is a chicken-and-egg problem in that you need some form of diagnostics
  // in order to diagnose options which further influence diagnostics. The code
  // here is mostly copy-and-pasted from driver.cpp/cc1_main.cpp/various Clang
  // tests to try to approximate the same behavior as running the `clang`
  // executable.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions);
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList ArgList = getDriverOptTable().ParseArgs(
      Args.slice(1), MissingArgIndex, MissingArgCount);
  // We ignore MissingArgCount and the return value of ParseDiagnosticArgs. Any
  // errors that would be diagnosed here will also be diagnosed later, when the
  // DiagnosticsEngine actually exists.
  (void)ParseDiagnosticArgs(*DiagOpts, ArgList);
  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(LogS, &*DiagOpts);
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs);
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);
  ProcessWarningOptions(Diags, *DiagOpts, /*ReportDiags=*/false);
  Driver TheDriver("", "", Diags);
  TheDriver.setTitle("AMDGPU Code Object Manager");
  TheDriver.setCheckInputsExist(false);

  // Log arguments used to build compilation
  if (env::shouldEmitVerboseLogs()) {
    LogS << "    Compilation Args: " ;
    for (size_t i = 1; i < Args.size(); ++i) {
      if (Args[i]) {
        LogS << " \"" << Args[i] << '\"';
      }
    }
    LogS << '\n';
    LogS.flush();
  }

  std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(Args));
  if (!C) {
    return C->containsError() ? AMD_COMGR_STATUS_ERROR
                              : AMD_COMGR_STATUS_SUCCESS;
  }
  for (auto &Job : C->getJobs()) {
    auto Arguments = Job.getArguments();
    SmallVector<const char *, 128> Argv;
    initializeCommandLineArgs(Argv);
    Argv.append(Arguments.begin(), Arguments.end());
    Argv.push_back(nullptr);

    // By default clang driver will ask CC1 to leak memory.
    auto *IT = find(Argv, StringRef("-disable-free"));
    if (IT != Argv.end()) {
      Argv.erase(IT);
    }

    clearLLVMOptions();

    if (Argv[1] == StringRef("-cc1")) {
      if (env::shouldEmitVerboseLogs()) {
        logArgv(LogS, "clang", Argv);
      }

      std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());
      Clang->setVerboseOutputStream(LogS);
      if (!Argv.back()) {
        Argv.pop_back();
      }
      if (!CompilerInvocation::CreateFromArgs(Clang->getInvocation(), Argv,
                                              Diags)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      // Internally this call refers to the invocation created above, so at
      // this point the DiagnosticsEngine should accurately reflect all user
      // requested configuration from Argv.
      Clang->createDiagnostics(DiagClient, /* ShouldOwnClient */ false);
      if (!Clang->hasDiagnostics()) {
        return AMD_COMGR_STATUS_ERROR;
      }
      if (!ExecuteCompilerInvocation(Clang.get())) {
        return AMD_COMGR_STATUS_ERROR;
      }
    } else if (Argv[1] == StringRef("-cc1as")) {
      if (env::shouldEmitVerboseLogs()) {
        logArgv(LogS, "clang", Argv);
      }
      Argv.erase(Argv.begin() + 1);
      if (!Argv.back()) {
        Argv.pop_back();
      }
      AssemblerInvocation Asm;
      if (!AssemblerInvocation::createFromArgs(Asm, Argv, Diags)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      if (auto Status = parseLLVMOptions(Asm.LLVMArgs)) {
        return Status;
      }
      if (executeAssembler(Asm, Diags, LogS)) {
        return AMD_COMGR_STATUS_ERROR;
      }
    } else if (Job.getCreator().getName() == LinkerJobName) {
      if (env::shouldEmitVerboseLogs()) {
        logArgv(LogS, "lld", Argv);
      }
      if (auto Status = linkWithLLD(Arguments, LogS, LogS)) {
        return Status;
      }
    } else {
      return AMD_COMGR_STATUS_ERROR;
    }
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::createTmpDirs() {
  ProfilePoint Point("CreateDir");
  if (fs::createUniqueDirectory("comgr", TmpDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  InputDir = TmpDir;
  path::append(InputDir, "input");
  if (fs::create_directory(InputDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  OutputDir = TmpDir;
  path::append(OutputDir, "output");
  if (fs::create_directory(OutputDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  IncludeDir = TmpDir;
  path::append(IncludeDir, "include");
  if (fs::create_directory(IncludeDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

// On windows fs::remove_directories takes huge time so use fs::remove.
amd_comgr_status_t RemoveDirectory(const StringRef DirName) {
  std::error_code EC;
  for (fs::directory_iterator Dir(DirName, EC), DirEnd;
       Dir != DirEnd && !EC; Dir.increment(EC)) {
    const StringRef Path = Dir->path();

    fs::file_status Status;
    EC = fs::status(Path, Status);
    if (EC) {
      return AMD_COMGR_STATUS_ERROR;
    }

    switch (Status.type()) {
    case fs::file_type::regular_file:
      if (fs::remove(Path)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      break;
    case fs::file_type::directory_file:
      if (RemoveDirectory(Path)) {
        return AMD_COMGR_STATUS_ERROR;
      }

      if (fs::remove(Path)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      break;
    default:
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  if (fs::remove(DirName)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::removeTmpDirs() {
  if (TmpDir.empty()) {
    return AMD_COMGR_STATUS_SUCCESS;
  }
  ProfilePoint Point("RemoveDir");
#ifndef _WIN32
  if (fs::remove_directories(TmpDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
#else
  return RemoveDirectory(TmpDir);
#endif
}

amd_comgr_status_t AMDGPUCompiler::executeOutOfProcessHIPCompilation(
    llvm::ArrayRef<const char *> Args) {
  std::string Exec = (Twine(env::getHIPPath()) + "/bin/hipcc").str();
  std::vector<StringRef> ArgsV;
  ArgsV.push_back(Exec);
  for (unsigned I = 0, E = Args.size(); I != E; ++I) {
    if (strcmp(Args[I], "-hip-path") == 0) {
      ++I;
      if (I == E) {
        LogS << "Error: -hip-path option misses argument.\n";
        return AMD_COMGR_STATUS_ERROR;
      }
      Exec = (Twine(Args[I]) + "/bin/hipcc").str();
      ArgsV[0] = Exec;

    } else {
      ArgsV.push_back(Args[I]);
    }
  }

  ArgsV.push_back("--genco");

  if (env::shouldEmitVerboseLogs()) {
    LogS << "\t    hipcc Command: ";
    for (auto A : ArgsV)
      LogS << A << " ";
    LogS << "\n";
  }

  llvm::ArrayRef<std::optional<StringRef>> Redirects;
  std::string ErrMsg;
  int RC = sys::ExecuteAndWait(Exec, ArgsV,
                               /*env=*/std::nullopt, Redirects, /*secondsToWait=*/0,
                               /*memoryLimit=*/0, &ErrMsg);
  LogS << ErrMsg;
  return RC ? AMD_COMGR_STATUS_ERROR : AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::processFile(const char *InputFilePath,
                                               const char *OutputFilePath) {
  SmallVector<const char *, 128> Argv;

  for (auto &Arg : Args) {
    Argv.push_back(Arg);
  }

  for (auto &Option : ActionInfo->getOptions()) {
    Argv.push_back(Option.c_str());
    if (Option.rfind("--rocm-path", 0) == 0) {
      NoGpuLib = false;
    }
  }

  // The ROCm device library should be provided via --rocm-path. Otherwise
  // we can pass -nogpulib to build without the ROCm device library
  if (NoGpuLib) {
    Argv.push_back("-nogpulib");
  }

  if (getLanguage() == AMD_COMGR_LANGUAGE_HIP && env::shouldSaveTemps()) {
    std::string save_tmps = "-save-temps=" + OutputDir.str().str();
    Argv.push_back(strdup(save_tmps.c_str()));
  }

  Argv.push_back(InputFilePath);

  Argv.push_back("-o");
  Argv.push_back(OutputFilePath);

  // For HIP OOP compilation, we launch a process.
  if (CompileOOP && getLanguage() == AMD_COMGR_LANGUAGE_HIP) {
    return executeOutOfProcessHIPCompilation(Argv);
  }

  return executeInProcessDriver(Argv);
}

amd_comgr_status_t
AMDGPUCompiler::processFiles(amd_comgr_data_kind_t OutputKind,
                             const char *OutputSuffix) {
  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_INCLUDE) {
      continue;
    }
    auto IncludeFilePath = getFilePath(Input, IncludeDir);
    if (auto Status = outputToFile(Input, IncludeFilePath)) {
      return Status;
    }
  }

  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_SOURCE &&
        Input->DataKind != AMD_COMGR_DATA_KIND_BC &&
        Input->DataKind != AMD_COMGR_DATA_KIND_RELOCATABLE &&
        Input->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE) {
      continue;
    }

    auto InputFilePath = getFilePath(Input, InputDir);
    if (auto Status = outputToFile(Input, InputFilePath)) {
      return Status;
    }

    amd_comgr_data_t OutputT;
    if (auto Status = amd_comgr_create_data(OutputKind, &OutputT)) {
      return Status;
    }
    ScopedDataObjectReleaser SDOR(OutputT);

    DataObject *Output = DataObject::convert(OutputT);
    Output->setName(std::string(Input->Name) + OutputSuffix);

    auto OutputFilePath = getFilePath(Output, OutputDir);

    if (auto Status =
            processFile(InputFilePath.c_str(), OutputFilePath.c_str())) {
      return Status;
    }

    if (auto Status = inputFromFile(Output, OutputFilePath)) {
      return Status;
    }

    if (auto Status = amd_comgr_data_set_add(OutSetT, OutputT)) {
      return Status;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::addIncludeFlags() {
  if (ActionInfo->Path) {
    Args.push_back("-I");
    Args.push_back(ActionInfo->Path);
  }

  Args.push_back("-I");
  Args.push_back(IncludeDir.c_str());

  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER) {
      continue;
    }
    PrecompiledHeaders.push_back(getFilePath(Input, IncludeDir));
    auto &PrecompiledHeaderPath = PrecompiledHeaders.back();
    if (auto Status = outputToFile(Input, PrecompiledHeaderPath)) {
      return Status;
    }
    Args.push_back("-include-pch");
    Args.push_back(PrecompiledHeaderPath.c_str());
    Args.push_back("-Xclang");
    Args.push_back("-fno-validate-pch");
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t
AMDGPUCompiler::addTargetIdentifierFlags(llvm::StringRef IdentStr,
                                         bool SrcToBC = false) {
  TargetIdentifier Ident;
  if (auto Status = parseTargetIdentifier(IdentStr, Ident)) {
    return Status;
  }
  Triple = (Twine(Ident.Arch) + "-" + Ident.Vendor + "-" + Ident.OS).str();

  GPUArch = Twine(Ident.Processor).str();
  if (!Ident.Features.empty()) {
    GPUArch += ":" + join(Ident.Features, ":");
  }

  if (SrcToBC && getLanguage() == AMD_COMGR_LANGUAGE_HIP) {
    OffloadArch = (Twine("--offload-arch=") + GPUArch).str();
    Args.push_back(OffloadArch.c_str());
  } else {
    CPU = (Twine("-mcpu=") + GPUArch).str();
    Args.push_back("-target");
    Args.push_back(Triple.c_str());
    Args.push_back(CPU.c_str());
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::addCompilationFlags() {
  HIPIncludePath = (Twine(env::getHIPPath()) + "/include").str();
  // HIP headers depend on hsa.h which is in ROCM_DIR/include.
  ROCMIncludePath = (Twine(env::getROCMPath()) + "/include").str();
  ClangIncludePath =
      (Twine(env::getLLVMPath()) + "/lib/clang/" + CLANG_VERSION_STRING).str();
  ClangIncludePath2 = (Twine(env::getLLVMPath()) + "/lib/clang/" +
                       CLANG_VERSION_STRING + "/include")
                          .str();

  Args.push_back("-x");

  switch (ActionInfo->Language) {
  case AMD_COMGR_LANGUAGE_LLVM_IR:
    Args.push_back("ir");
    break;
  case AMD_COMGR_LANGUAGE_OPENCL_1_2:
    Args.push_back("cl");
    Args.push_back("-std=cl1.2");
    Args.push_back("-cl-no-stdinc");
    break;
  case AMD_COMGR_LANGUAGE_OPENCL_2_0:
    Args.push_back("cl");
    Args.push_back("-std=cl2.0");
    Args.push_back("-cl-no-stdinc");
    break;
  case AMD_COMGR_LANGUAGE_HIP:
    Args.push_back("hip");
    Args.push_back("-std=c++11");
    Args.push_back("-target");
    Args.push_back("x86_64-unknown-linux-gnu");
    Args.push_back("--cuda-device-only");
    Args.push_back("-isystem");
    Args.push_back(ROCMIncludePath.c_str());
    Args.push_back("-isystem");
    Args.push_back(HIPIncludePath.c_str());
    Args.push_back("-isystem");
    Args.push_back(ClangIncludePath.c_str());
    Args.push_back("-isystem");
    Args.push_back(ClangIncludePath2.c_str());
    break;
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::addDeviceLibraries() {
  llvm::SmallString<128> FakeRocmDir = TmpDir;
  path::append(FakeRocmDir, "rocm");
  llvm::SmallString<128> DeviceLibsDir = FakeRocmDir;
  path::append(DeviceLibsDir, "amdgcn", "bitcode");
  if (fs::create_directory(InputDir)) {
    return AMD_COMGR_STATUS_ERROR;
  }
  Args.push_back(Saver.save(Twine("--rocm-path=") + FakeRocmDir).data());
  NoGpuLib = false;

  for (auto DeviceLib : getDeviceLibraries()) {
    llvm::SmallString<128> DeviceLibPath = DeviceLibsDir;
    path::append(DeviceLibPath, std::get<0>(DeviceLib));
    if (auto Status = outputToFile(std::get<1>(DeviceLib), DeviceLibPath)) {
      return Status;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMDGPUCompiler::preprocessToSource() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName)) {
      return Status;
    }
  }

  if (auto Status = addIncludeFlags()) {
    return Status;
  }

  if (auto Status = addCompilationFlags()) {
    return Status;
  }

  Args.push_back("-E");

  return processFiles(AMD_COMGR_DATA_KIND_SOURCE, ".i");
}

amd_comgr_status_t AMDGPUCompiler::compileToBitcode(bool WithDeviceLibs) {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName, true)) {
      return Status;
    }
  }

  if (auto Status = addIncludeFlags()) {
    return Status;
  }

  if (auto Status = addCompilationFlags()) {
    return Status;
  }

  Args.push_back("-c");
  Args.push_back("-emit-llvm");

#if _WIN32
  Args.push_back("-fshort-wchar");
#endif

  if (WithDeviceLibs) {
    if (auto Status = addDeviceLibraries()) {
      return Status;
    }
  }

  return processFiles(AMD_COMGR_DATA_KIND_BC, ".bc");
}

amd_comgr_status_t AMDGPUCompiler::compileToExecutable() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName, true)) {
      return Status;
    }
  }

  if (auto Status = addIncludeFlags()) {
    return Status;
  }

  if (auto Status = addCompilationFlags()) {
    return Status;
  }

#if _WIN32
  Args.push_back("-fshort-wchar");
#endif

  if (auto Status = addDeviceLibraries()) {
    return Status;
  }

  return processFiles(AMD_COMGR_DATA_KIND_EXECUTABLE, ".so");
}

amd_comgr_status_t AMDGPUCompiler::compileToRelocatable() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->Language != AMD_COMGR_LANGUAGE_HIP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName, true)) {
      return Status;
    }
  }

  Args.push_back("-c");
  Args.push_back("-fhip-emit-relocatable");
  Args.push_back("-mllvm");
  Args.push_back("-amdgpu-internalize-symbols");

  if (auto Status = addIncludeFlags()) {
    return Status;
  }

  if (auto Status = addCompilationFlags()) {
    return Status;
  }

#if _WIN32
  Args.push_back("-fshort-wchar");
#endif

  if (auto Status = addDeviceLibraries()) {
    return Status;
  }

  return processFiles(AMD_COMGR_DATA_KIND_RELOCATABLE, ".o");
}

amd_comgr_status_t AMDGPUCompiler::compileToFatBin() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->Language != AMD_COMGR_LANGUAGE_HIP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // This is a workaround to support HIP OOP Fatbin Compilation
  CompileOOP = true;
  auto Status = processFiles(AMD_COMGR_DATA_KIND_FATBIN, ".fatbin");
  CompileOOP = false;

  return Status;
}

amd_comgr_status_t AMDGPUCompiler::linkBitcodeToBitcode() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  SMDiagnostic SMDiag;
  LLVMContext Context;
  Context.setDiagnosticHandler(
    std::make_unique<AMDGPUCompilerDiagnosticHandler>(this), true);

  auto Composite = std::make_unique<llvm::Module>("llvm-link", Context);
  Linker L(*Composite);
  unsigned ApplicableFlags = Linker::Flags::None;

  // Collect bitcode memory buffers from bitcodes, bundles, and archives
  for (auto *Input : InSet->DataObjects) {

    if (!strcmp(Input->Name, "")) {
      // If the calling API doesn't provide a DataObject name, generate a random
      // string to assign. This string is used when the DataObject is written
      // to the file system via SAVE_TEMPS, or if the object is a bundle which
      // also needs a file system write for unpacking

      char *buf = (char *) malloc(sizeof(char) * 30);
      sprintf(buf,"comgr-anon-bitcode-%d.bc", std::rand() % 10000);

      Input->Name = buf;
    }

    if (env::shouldSaveTemps()) {
      if (auto Status = outputToFile(Input, getFilePath(Input, InputDir))) {
        return Status;
      }
    }

    if (Input->DataKind == AMD_COMGR_DATA_KIND_BC) {
      // The data in Input outlives Mod, and the linker destructs Mod after
      // linking it into composite (i.e. ownership is not transferred to the
      // composite) so MemoryBuffer::getMemBuffer is sufficient.

      if (env::shouldEmitVerboseLogs()) {
        LogS << "\t     Linking Bitcode: " << InputDir << "/" << Input->Name
             << "\n";
      }

      auto Mod =
        getLazyIRModule(MemoryBuffer::getMemBuffer(
            StringRef(Input->Data, Input->Size), "", false),
          SMDiag, Context, true);

      if (!Mod) {
        SMDiag.print(Input->Name, LogS, /* ShowColors */ false);
        return AMD_COMGR_STATUS_ERROR;
      }
      if (verifyModule(*Mod, &LogS))
        return AMD_COMGR_STATUS_ERROR;
      if (L.linkInModule(std::move(Mod), ApplicableFlags))
        return AMD_COMGR_STATUS_ERROR;
    }
    else if (Input->DataKind == AMD_COMGR_DATA_KIND_BC_BUNDLE) {
      if (env::shouldEmitVerboseLogs()) {
        LogS << "      Linking Bundle: " << InputDir << "/" << Input->Name
          << "\n";
      }

      // Determine desired bundle entry ID
      if (!ActionInfo->IsaName)
        return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

      std::string isa_name = ActionInfo->IsaName;
      size_t index = isa_name.find("gfx");
      std::string bundle_entry_id = "hip-amdgcn-amd-amdhsa--gfx" +
        isa_name.substr(index + 3);

      // Write data to file system so that Offload Bundler can process, assuming
      // we didn't already write due to save-temps above
      // TODO: Switch write to VFS
      if (!env::shouldSaveTemps()) {
        if (auto Status = outputToFile(Input, getFilePath(Input, InputDir))) {
          return Status;
        }
      }

      // Configure Offload Bundler
      OffloadBundlerConfig BundlerConfig;
      BundlerConfig.AllowMissingBundles = true;
      BundlerConfig.FilesType = "bc";

      BundlerConfig.TargetNames.push_back(bundle_entry_id);
      std::string input_file_path = getFilePath(Input, InputDir).str().str();
      BundlerConfig.InputFileNames.push_back(input_file_path);

      // Generate prefix for output files
      std::string output_prefix = std::string(Input->Name);
      index = output_prefix.find_last_of(".");
      output_prefix = output_prefix.substr(0, index);
      std::string output_file_name = output_prefix + '-' + bundle_entry_id +
        ".bc";

      // ISA name may contain ':', which is an invalid character in file names
      // on Windows. Replace with '_'
      std::replace(output_file_name.begin(), output_file_name.end(), ':', '_');

      std::string output_file_path = OutputDir.str().str() + "/" +
        output_file_name;
      BundlerConfig.OutputFileNames.push_back(output_file_path);

      OffloadBundler Bundler(BundlerConfig);

      // Execute unbundling
      if (env::shouldEmitVerboseLogs()) {
        LogS << "Extracting Bitcode Bundle:\n"
          << "\t  Bundle Entry ID: " << BundlerConfig.TargetNames[0] << "\n"
          << "\t   Input Filename: " << BundlerConfig.InputFileNames[0] << "\n"
          << "\t  Output Filename: " << BundlerConfig.OutputFileNames[0]
          << "\n";
        LogS << "\t          Command: clang-offload-bundler -unbundle -type=bc"
          " -targets=" << bundle_entry_id <<
          " -input="   << input_file_path <<
          " -output="  << output_file_path << "\n";
        LogS.flush();
      }


      llvm::Error Err = Bundler.UnbundleFiles();
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "UnbundleFiles error: ");

      // Read unbundled bitcode from file system in order to pass to linker
      amd_comgr_data_t ResultT;
      if (auto Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, &ResultT))
        return Status;

      DataObject *Result = DataObject::convert(ResultT);
      if (auto Status = inputFromFile(Result, StringRef(output_file_path)))
        return Status;

      Result->Name = strdup(output_file_name.c_str());

      auto Mod =
        getLazyIRModule(MemoryBuffer::getMemBuffer(
            StringRef(Result->Data, Result->Size), "", false),
          SMDiag, Context, true);

      if (!Mod) {
        SMDiag.print(Result->Name, LogS, /* ShowColors */ false);
        return AMD_COMGR_STATUS_ERROR;
      }
      if (verifyModule(*Mod, &LogS))
        return AMD_COMGR_STATUS_ERROR;
      if (L.linkInModule(std::move(Mod), ApplicableFlags))
        return AMD_COMGR_STATUS_ERROR;

      Result->release();
    }
    // Unbundle bitcode archive
    else if (Input->DataKind == AMD_COMGR_DATA_KIND_AR_BUNDLE) {
      if (env::shouldEmitVerboseLogs()) {
        LogS << "\t     Linking Archive: " << InputDir << "/" << Input->Name
             << "\n";
      }

      // Determine desired bundle entry ID
      if (!ActionInfo->IsaName)
        return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

      std::string isa_name = ActionInfo->IsaName;
      size_t index = isa_name.find("gfx");
      std::string bundle_entry_id = "hip-amdgcn-amd-amdhsa--gfx" +
        isa_name.substr(index + 3);

      // Write data to file system so that Offload Bundler can process, assuming
      // we didn't already write due to save-temps above
      // TODO: Switch write to VFS
      if (!env::shouldSaveTemps()) {
        if (auto Status = outputToFile(Input, getFilePath(Input, InputDir))) {
          return Status;
        }
      }

      // Configure Offload Bundler
      OffloadBundlerConfig BundlerConfig;
      BundlerConfig.AllowMissingBundles = true;
      BundlerConfig.FilesType = "a";
      BundlerConfig.HipOpenmpCompatible = 1;
      BundlerConfig.AllowNoHost = 1;

      BundlerConfig.TargetNames.push_back(bundle_entry_id);
      std::string input_file_path = getFilePath(Input, InputDir).str().str();
      BundlerConfig.InputFileNames.push_back(input_file_path);

      // Generate prefix for output files
      std::string output_prefix = std::string(Input->Name);
      index = output_prefix.find_last_of(".");
      output_prefix = output_prefix.substr(0, index);

      std::string output_file_name = output_prefix + '-' + bundle_entry_id +
        ".a";

      // ISA name may contain ':', which is an invalid character in file names
      // on Windows. Replace with '_'
      std::replace(output_file_name.begin(), output_file_name.end(), ':', '_');

      std::string output_file_path = OutputDir.str().str() + "/" +
        output_file_name;
      BundlerConfig.OutputFileNames.push_back(output_file_path);

      OffloadBundler Bundler(BundlerConfig);

      // Execute unbundling
      if (env::shouldEmitVerboseLogs()) {
        LogS << "    Extracting Bitcode Archive:\n"
          << "\t  Bundle Entry ID: " << BundlerConfig.TargetNames[0] << "\n"
          << "\t   Input Filename: " << BundlerConfig.InputFileNames[0] << "\n"
          << "\t  Output Filename: " << BundlerConfig.OutputFileNames[0]
          << "\n";
        LogS << "\t          Command: clang-offload-bundler -unbundle -type=a "
          " -targets=" << bundle_entry_id <<
          " -input="   << input_file_path <<
          " -output="  << output_file_path << "\n";
        LogS.flush();
      }
      llvm::Error Err = Bundler.UnbundleArchive();
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "UnbundleArchive error: ");

      // Read archive back into Comgr
      amd_comgr_data_t ResultT;
      if (auto Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_AR, &ResultT))
        return Status;

      DataObject *Result = DataObject::convert(ResultT);
      if (auto Status = inputFromFile(Result, StringRef(output_file_path)))
        return Status;

      // Get memory buffer for each bitcode in archive file
      //   Modeled after static loadArFile in llvm-link.cpp
      std::string ArchiveName = "comgr.ar";
      llvm::StringRef ArchiveBuf = StringRef(Result->Data, Result->Size);
      auto ArchiveOrError = object::Archive::create(
        MemoryBufferRef(ArchiveBuf, ArchiveName));

      if (!ArchiveOrError) {
        llvm::logAllUnhandledErrors(ArchiveOrError.takeError(),
                                    llvm::errs(), "Unpack Archives error: ");
        return AMD_COMGR_STATUS_ERROR;
      }

      auto Archive = std::move(ArchiveOrError.get());

      Err = Error::success();
      for (const object::Archive::Child &C : Archive->children(Err)) {

        // Get child name
        Expected<StringRef> Ename = C.getName();
        if (Error E = Ename.takeError()) {
          errs() << ": ";
          WithColor::error() << " failed to read name of archive member"
            << ArchiveName << "'\n";
          return AMD_COMGR_STATUS_ERROR;
        }
        std::string ChildName = Ename.get().str();

        // Get memory buffer
        SMDiagnostic ParseErr;
        Expected<MemoryBufferRef> MemBuf = C.getMemoryBufferRef();
        if (Error E = MemBuf.takeError()) {
          errs() << ": ";
          WithColor::error() << " loading memory for member '"
            << "' of archive library failed'" << ArchiveName
            << "'\n";
          return AMD_COMGR_STATUS_ERROR;
        };

        // Link memory buffer into composite
        auto Mod =
          getLazyIRModule(MemoryBuffer::getMemBuffer(MemBuf.get()),
                          SMDiag, Context, true);

        if (!Mod) {
          SMDiag.print(ChildName.c_str(), LogS, /* ShowColors */ false);
          return AMD_COMGR_STATUS_ERROR;
        }
        if (verifyModule(*Mod, &LogS))
          return AMD_COMGR_STATUS_ERROR;
        if (L.linkInModule(std::move(Mod), ApplicableFlags))
          return AMD_COMGR_STATUS_ERROR;
      }

      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Unpack Archives error: ");

      Result->release();
    }
    else
      continue;
  }

  if (verifyModule(*Composite, &LogS)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  SmallString<0> OutBuf;
  BitcodeWriter Writer(OutBuf);
  Writer.writeModule(*Composite, false, nullptr, false, nullptr);
  Writer.writeSymtab();
  Writer.writeStrtab();

  amd_comgr_data_t OutputT;
  if (auto Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, &OutputT)) {
    return Status;
  }
  ScopedDataObjectReleaser SDOR(OutputT);

  DataObject *Output = DataObject::convert(OutputT);
  Output->setName("linked.bc");
  Output->setData(OutBuf);

  return amd_comgr_data_set_add(OutSetT, OutputT);
}

amd_comgr_status_t AMDGPUCompiler::codeGenBitcodeToRelocatable() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName)) {
      return Status;
    }
  }

  Args.push_back("-c");

  Args.push_back("-mllvm");
  Args.push_back("-amdgpu-internalize-symbols");

  return processFiles(AMD_COMGR_DATA_KIND_RELOCATABLE, ".o");
}

amd_comgr_status_t AMDGPUCompiler::codeGenBitcodeToAssembly() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName)) {
      return Status;
    }
  }

  Args.push_back("-S");

  return processFiles(AMD_COMGR_DATA_KIND_SOURCE, ".s");
}

amd_comgr_status_t AMDGPUCompiler::assembleToRelocatable() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName)) {
      return Status;
    }
  }

  if (auto Status = addIncludeFlags()) {
    return Status;
  }

  Args.push_back("-c");
  Args.push_back("-x");
  Args.push_back("assembler");

  return processFiles(AMD_COMGR_DATA_KIND_RELOCATABLE, ".o");
}

amd_comgr_status_t AMDGPUCompiler::linkToRelocatable() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  for (auto &Option : ActionInfo->getOptions()) {
    Args.push_back(Option.c_str());
  }

  SmallVector<SmallString<128>, 128> Inputs;
  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_RELOCATABLE) {
      continue;
    }

    Inputs.push_back(getFilePath(Input, InputDir));
    if (auto Status = outputToFile(Input, Inputs.back())) {
      return Status;
    }
    Args.push_back(Inputs.back().c_str());
  }

  amd_comgr_data_t OutputT;
  if (auto Status =
          amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &OutputT)) {
    return Status;
  }
  ScopedDataObjectReleaser SDOR(OutputT);

  DataObject *Output = DataObject::convert(OutputT);
  Output->setName("a.o");
  auto OutputFilePath = getFilePath(Output, OutputDir);
  Args.push_back("-o");
  Args.push_back(OutputFilePath.c_str());

  Args.push_back("-r");

  if (auto Status = linkWithLLD(Args, LogS, LogS)) {
    return Status;
  }

  if (auto Status = inputFromFile(Output, OutputFilePath)) {
    return Status;
  }

  return amd_comgr_data_set_add(OutSetT, OutputT);
}

amd_comgr_status_t AMDGPUCompiler::linkToExecutable() {
  if (auto Status = createTmpDirs()) {
    return Status;
  }

  if (ActionInfo->IsaName) {
    if (auto Status = addTargetIdentifierFlags(ActionInfo->IsaName)) {
      return Status;
    }
  }

  for (auto &Option : ActionInfo->getOptions()) {
    Args.push_back(Option.c_str());
  }

  SmallVector<SmallString<128>, 128> Inputs;
  for (auto *Input : InSet->DataObjects) {
    if (Input->DataKind != AMD_COMGR_DATA_KIND_RELOCATABLE) {
      continue;
    }

    Inputs.push_back(getFilePath(Input, InputDir));
    if (auto Status = outputToFile(Input, Inputs.back())) {
      return Status;
    }
    Args.push_back(Inputs.back().c_str());
  }

  amd_comgr_data_t OutputT;
  if (auto Status =
          amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &OutputT)) {
    return Status;
  }
  ScopedDataObjectReleaser SDOR(OutputT);

  DataObject *Output = DataObject::convert(OutputT);
  Output->setName("a.so");
  auto OutputFilePath = getFilePath(Output, OutputDir);
  Args.push_back("-o");
  Args.push_back(OutputFilePath.c_str());

  if (auto Status = executeInProcessDriver(Args)) {
    return Status;
  }

  if (auto Status = inputFromFile(Output, OutputFilePath)) {
    return Status;
  }

  return amd_comgr_data_set_add(OutSetT, OutputT);
}

AMDGPUCompiler::AMDGPUCompiler(DataAction *ActionInfo, DataSet *InSet,
                               DataSet *OutSet, raw_ostream &LogS)
    : ActionInfo(ActionInfo), InSet(InSet), OutSetT(DataSet::convert(OutSet)),
      LogS(LogS) {
  initializeCommandLineArgs(Args);
}

AMDGPUCompiler::~AMDGPUCompiler() {
  if (!env::shouldSaveTemps()) {
    removeTmpDirs();
  }
}

} // namespace COMGR
