//===- lld/Common/Driver.h - Linker Driver Emulator -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COMMON_DRIVER_H
#define LLD_COMMON_DRIVER_H

#include "lld/Common/CommonLinkerContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
struct SafeReturn {
  int ret;
  bool canRunAgain;
};

// Generic entry point when using LLD as a library, safe for re-entry, supports
// crash recovery. Returns a general completion code and a boolean telling
// whether it can be called again. In some cases, a crash could corrupt memory
// and re-entry would not be possible anymore. Use exitLld() in that case to
// properly exit your application and avoid intermittent crashes on exit caused
// by cleanup.
SafeReturn safeLldMain(llvm::ArrayRef<const char *> args,
                       llvm::raw_ostream &stdoutOS,
                       llvm::raw_ostream &stderrOS);

namespace coff {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

namespace mingw {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

namespace elf {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

namespace macho {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

namespace wasm {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}
} // namespace lld

// It is also possible to selectively use a subset of the LLD drivers in
// applications that use LLD as a library. Simply link the required libs, such
// as lldELF and lldCommon; then use the macro below at global scope to force
// definition of symbols for the missing drivers.
#define LLD_IMPLEMENT_SHALLOW_DRIVER(driver)                                   \
  namespace lld {                                                              \
  namespace driver {                                                           \
  bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,    \
            llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) { \
    llvm::errs() << "The LLD " #driver                                         \
                    " driver is not available in this build.\n";               \
    return false;                                                              \
  }                                                                            \
  } /* namespace driver */                                                     \
  } /* namespace lld */

#endif
