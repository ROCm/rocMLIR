//===- AMDGPUOpenMP.h - AMDGPUOpenMP ToolChain Implementation -*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDGPUOPENMP_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDGPUOPENMP_H

#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Tool.h"
#include "AMDGPU.h"

namespace clang {
namespace driver {

/// Is -Ofast used?
bool isOFastUsed(const llvm::opt::ArgList &Args);

/// Is -fopenmp-target-fast or -Ofast used
bool isTargetFastUsed(const llvm::opt::ArgList &Args);

/// Ignore possibility of environment variables if either
/// -fopenmp-target-fast or -Ofast is used.
bool shouldIgnoreEnvVars(const llvm::opt::ArgList &Args);

namespace toolchains {
class AMDGPUOpenMPToolChain;
}

namespace tools {

namespace AMDGCN {
  // Construct command for creating HIP fatbin.
  void constructHIPFatbinCommand(Compilation &C, const JobAction &JA,
                  StringRef OutputFileName, const InputInfoList &Inputs,
                  const llvm::opt::ArgList &TCArgs, const Tool& T);

// Runs llvm-link/opt/llc/lld, which links multiple LLVM bitcode, together with
// device library, then compiles it to ISA in a shared object.
class LLVM_LIBRARY_VISIBILITY OpenMPLinker : public Tool {
public:
  OpenMPLinker(const ToolChain &TC)
      : Tool("AMDGCN::OpenMPLinker", "amdgcn-link", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

private:
  /// \return output file name from build-select, prelink, and preopt
  const char *constructOmpExtraCmds(Compilation &C, const JobAction &JA,
                                    const InputInfoList &Inputs,
                                    const llvm::opt::ArgList &Args,
                                    llvm::StringRef TargetID,
                                    llvm::StringRef OutputFilePrefix) const;
};

} // end namespace AMDGCN
} // end namespace tools

namespace toolchains {
class LLVM_LIBRARY_VISIBILITY AMDGPUOpenMPToolChain final
    : public ROCMToolChain {
public:
  AMDGPUOpenMPToolChain(const Driver &D, const llvm::Triple &Triple,
                        const ToolChain &HostTC,
                        const llvm::opt::ArgList &Args,
                        const Action::OffloadKind OK);

  const llvm::Triple *getAuxTriple() const override {
    return &HostTC.getTriple();
  }

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;
  void
  addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                        llvm::opt::ArgStringList &CC1Args,
                        Action::OffloadKind DeviceOffloadKind) const override;

  bool useIntegratedAs() const override { return true; }
  bool isCrossCompiling() const override { return true; }
  bool isPICDefault() const override { return false; }
  bool isPIEDefault(const llvm::opt::ArgList &Args) const override { return false; }
  bool isPICDefaultForced() const override { return false; }
  bool SupportsProfiling() const override { return false; }
  bool IsMathErrnoDefault() const override { return false; }

  void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const override;
  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;
  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &Args,
      llvm::opt::ArgStringList &CC1Args) const override;
  void AddIAMCUIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                           llvm::opt::ArgStringList &CC1Args) const override;
  void
  AddFlangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &FlangArgs) const override;

  SanitizerMask getSupportedSanitizers() const override;

  StringRef getAsanRTLPath() const {
    return RocmInstallation->getAsanRTLPath();
  }

  VersionTuple
  computeMSVCVersion(const Driver *D,
                     const llvm::opt::ArgList &Args) const override;

  unsigned GetDefaultDwarfVersion() const override { return 5; }
  llvm::SmallVector<BitCodeLibraryInfo, 12>
  getDeviceLibs(const llvm::opt::ArgList &Args) const override;

  const ToolChain &HostTC;

private:
  const Action::OffloadKind OK;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDGPUOPENMP_H
