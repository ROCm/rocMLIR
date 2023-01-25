//===- tools/lld/lld.cpp - Linker Driver Dispatcher -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function of the lld executable. The main
// function is a thin wrapper which dispatches to the platform specific
// driver.
//
// lld is a single executable that contains four different linkers for ELF,
// COFF, WebAssembly and Mach-O. The main function dispatches according to
// argv[0] (i.e. command name). The most common name for each target is shown
// below:
//
//  - ld.lld:    ELF (Unix)
//  - ld64:      Mach-O (macOS)
//  - lld-link:  COFF (Windows)
//  - ld-wasm:   WebAssembly
//
// lld can be invoked as "lld" along with "-flavor" option. This is for
// backward compatibility and not recommended.
//
//===----------------------------------------------------------------------===//

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
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/Process.h"
#include <cstdlib>
#include <optional>

using namespace lld;
using namespace llvm;
using namespace llvm::sys;

namespace lld {
extern bool inTestOutputDisabled;

int lldMain(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
            llvm::raw_ostream &stderrOS, bool exitEarly);
} // namespace lld

// When in lit tests, tells how many times the LLD tool should re-execute the
// main loop with the same inputs. When not in test, returns a value of 0 which
// signifies that LLD shall not release any memory after execution, to speed up
// process destruction.
static unsigned inTestVerbosity() {
  unsigned v = 0;
  StringRef(getenv("LLD_IN_TEST")).getAsInteger(10, v);
  return v;
}

int lld_main(int argc, char **argv) {
  InitLLVM x(argc, argv);
  sys::Process::UseANSIEscapeCodes(true);

  if (::getenv("FORCE_LLD_DIAGNOSTICS_CRASH")) {
    llvm::errs()
        << "crashing due to environment variable FORCE_LLD_DIAGNOSTICS_CRASH\n";
    LLVM_BUILTIN_TRAP;
  }

  ArrayRef<const char *> args(argv, argv + argc);

  // Not running in lit tests, just take the shortest codepath with global
  // exception handling and no memory cleanup on exit.
  if (!inTestVerbosity())
    return lldMain(args, llvm::outs(), llvm::errs(), /*exitEarly=*/true);

  std::optional<int> mainRet;
  CrashRecoveryContext::Enable();

  for (unsigned i = inTestVerbosity(); i > 0; --i) {
    // Disable stdout/stderr for all iterations but the last one.
    inTestOutputDisabled = (i != 1);

    // Execute one iteration.
    auto r = safeLldMain(args, llvm::outs(), llvm::errs());
    if (!r.canRunAgain)
      exitLld(r.ret); // Exit now, can't re-execute again.

    if (!mainRet) {
      mainRet = r.ret;
    } else if (r.ret != *mainRet) {
      // Exit now, to fail the tests if the result is different between runs.
      return r.ret;
    }
  }
  return *mainRet;
}
