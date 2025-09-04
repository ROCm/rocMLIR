//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<<< HEAD:external/llvm-project/libcxx/test/extensions/gnu/hash_map/const_iterator.verify.cpp
// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated

#include <ext/hash_map>

int main(int, char**) {
  __gnu_cxx::hash_map<int, int> m;
  m[1]                                    = 1;
  const __gnu_cxx::hash_map<int, int>& cm = m;
  cm.find(1)->second = 2; // expected-error {{cannot assign to return value because function 'operator->' returns a const value}}

  return 0;
========
#include <clc/opencl/workitem/get_max_sub_group_size.h>
#include <clc/workitem/clc_get_max_sub_group_size.h>

_CLC_OVERLOAD _CLC_DEF uint get_max_sub_group_size() {
  return __clc_get_max_sub_group_size();
>>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a:external/llvm-project/libclc/opencl/lib/ptx-nvidiacl/workitem/get_max_sub_group_size.cl
}
