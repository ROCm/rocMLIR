//===- LLDLibrary.cpp - Use LLD as a library ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include <cstdlib>

using namespace lld;
using namespace llvm;
using namespace llvm::sys;

enum Flavor {
  Invalid,
  Gnu,     // -flavor gnu
  MinGW,   // -flavor gnu MinGW
  WinLink, // -flavor link
  Darwin,  // -flavor darwin
  Wasm,    // -flavor wasm
};

static void err(const Twine &s) { llvm::errs() << s << "\n"; }

static Flavor getFlavor(StringRef s) {
  return StringSwitch<Flavor>(s)
      .CasesLower("ld", "ld.lld", "gnu", Gnu)
      .CasesLower("wasm", "ld-wasm", Wasm)
      .CaseLower("link", WinLink)
      .CasesLower("ld64", "ld64.lld", "darwin", Darwin)
      .Default(Invalid);
}

static cl::TokenizerCallback getDefaultQuotingStyle() {
  if (Triple(sys::getProcessTriple()).getOS() == Triple::Win32)
    return cl::TokenizeWindowsCommandLine;
  return cl::TokenizeGNUCommandLine;
}

static bool isPETargetName(StringRef s) {
  return s == "i386pe" || s == "i386pep" || s == "thumb2pe" || s == "arm64pe";
}

static std::optional<bool> isPETarget(llvm::ArrayRef<const char *> args) {
  for (auto it = args.begin(); it + 1 != args.end(); ++it) {
    if (StringRef(*it) != "-m")
      continue;
    return isPETargetName(*(it + 1));
  }

  // Expand response files (arguments in the form of @<filename>)
  // to allow detecting the -m argument from arguments in them.
  SmallVector<const char *, 256> expandedArgs(args.data(),
                                              args.data() + args.size());
  BumpPtrAllocator a;
  StringSaver saver(a);
  cl::ExpansionContext ectx(saver.getAllocator(), getDefaultQuotingStyle());
  if (ectx.expandResponseFiles(expandedArgs) == false) {
    return std::nullopt;
  }

  for (auto it = expandedArgs.begin(); it + 1 != expandedArgs.end(); ++it) {
    if (StringRef(*it) != "-m")
      continue;
    return isPETargetName(*(it + 1));
  }

#ifdef LLD_DEFAULT_LD_LLD_IS_MINGW
  return true;
#else
  return false;
#endif
}

static Flavor parseProgname(StringRef progname) {
  // Use GNU driver for "ld" by default.
  if (progname == "ld")
    return Gnu;

  // Progname may be something like "lld-gnu". Parse it.
  SmallVector<StringRef, 3> v;
  progname.split(v, "-");
  for (StringRef s : v)
    if (Flavor f = getFlavor(s))
      return f;
  return Invalid;
}

static Flavor parseFlavorWithoutMinGW(llvm::ArrayRef<const char *> args) {
  // Parse -flavor option.
  if (args.size() > 1 && args[1] == StringRef("-flavor")) {
    if (args.size() <= 2) {
      err("missing arg value for '-flavor'");
      return Invalid;
    }
    Flavor f = getFlavor(args[2]);
    if (f == Invalid) {
      err("Unknown flavor: " + StringRef(args[2]));
      return Invalid;
    }
    return f;
  }

  // Deduct the flavor from argv[0].
  StringRef arg0 = path::filename(args[0]);
  if (arg0.endswith_insensitive(".exe"))
    arg0 = arg0.drop_back(4);
  Flavor f = parseProgname(arg0);
  if (f == Invalid) {
    err("lld is a generic driver.\n"
        "Invoke ld.lld (Unix), ld64.lld (macOS), lld-link (Windows), wasm-ld"
        " (WebAssembly) instead");
    return Invalid;
  }
  return f;
}

static Flavor parseFlavor(llvm::ArrayRef<const char *> args) {
  Flavor f = parseFlavorWithoutMinGW(args);
  if (f == Gnu) {
    auto isPE = isPETarget(args);
    if (!isPE)
      return Invalid;
    if (*isPE)
      return MinGW;
  }
  return f;
}

static bool invalidLink(llvm::ArrayRef<const char *> args,
                        llvm::raw_ostream &stdoutOS,
                        llvm::raw_ostream &stderrOS, bool exitEarly,
                        bool disableOutput) {
  return false;
}

namespace lld {
bool inTestOutputDisabled = false;

/// Universal linker main(). This linker emulates the gnu, darwin, or
/// windows linker based on the argv[0] or -flavor option.
int lldMain(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
            llvm::raw_ostream &stderrOS, bool exitEarly) {
  Flavor f = parseFlavor(args);
  auto link = [&]() {
    if (f == MinGW)
      return mingw::link;
    else if (f == Gnu)
      return elf::link;
    else if (f == WinLink)
      return coff::link;
    else if (f == Darwin)
      return macho::link;
    else if (f == Wasm)
      return lld::wasm::link;
    else
      return &invalidLink;
  }();

  // Run the driver. If an error occurs, false will be returned.
  int r = !link(args, stdoutOS, stderrOS, exitEarly, inTestOutputDisabled);

  // Call exit() if we can to avoid calling destructors.
  if (exitEarly)
    exitLld(r);

  // Delete the global context and clear the global context pointer, so that it
  // cannot be accessed anymore.
  CommonLinkerContext::destroy();

  return r;
}

// Similar to lldMain except that exceptions are caught.
SafeReturn safeLldMain(llvm::ArrayRef<const char *> args,
                       llvm::raw_ostream &stdoutOS,
                       llvm::raw_ostream &stderrOS) {
  int r = 0;
  {
    // The crash recovery is here only to be able to recover from arbitrary
    // control flow when fatal() is called (through setjmp/longjmp or
    // __try/__except).
    llvm::CrashRecoveryContext crc;
    if (!crc.RunSafely([&]() {
          r = lldMain(args, stdoutOS, stderrOS, /*exitEarly=*/false);
        }))
      return {crc.RetCode, /*canRunAgain=*/false};
  }

  // Cleanup memory and reset everything back in pristine condition. This path
  // is only taken when LLD is in test, or when it is used as a library.
  llvm::CrashRecoveryContext crc;
  if (!crc.RunSafely([&]() { CommonLinkerContext::destroy(); })) {
    // The memory is corrupted beyond any possible recovery.
    return {r, /*canRunAgain=*/false};
  }
  return {r, /*canRunAgain=*/true};
}
} // namespace lld
