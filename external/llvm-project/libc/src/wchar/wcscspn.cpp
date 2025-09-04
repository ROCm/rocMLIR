//===-- Implementation of wcscspn -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcscspn.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
<<<<<<< HEAD

namespace LIBC_NAMESPACE_DECL {

bool check(wchar_t c, const wchar_t *s2) {
  for (int n = 0; s2[n]; ++n) {
    if (s2[n] == c)
      return false;
  }
  return true;
}
LLVM_LIBC_FUNCTION(size_t, wcscspn, (const wchar_t *s1, const wchar_t *s2)) {
  size_t i = 0;
  for (; s1[i]; ++i) {
    if (!check(s1[i], s2))
      return i;
  }
  return i;
=======
#include "src/__support/macros/null_check.h"
#include "wchar_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcscspn, (const wchar_t *s1, const wchar_t *s2)) {
  LIBC_CRASH_ON_NULLPTR(s1);
  LIBC_CRASH_ON_NULLPTR(s2);
  return internal::wcsspn(s1, s2, /*not_match_set=*/true);
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
}

} // namespace LIBC_NAMESPACE_DECL
