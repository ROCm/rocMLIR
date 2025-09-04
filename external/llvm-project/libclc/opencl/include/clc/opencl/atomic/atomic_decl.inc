//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<< HEAD
#define __CLC_DECLARE_ATOMIC(ADDRSPACE, TYPE)                                  \
  _CLC_OVERLOAD _CLC_DECL TYPE FUNCTION(volatile ADDRSPACE TYPE *, TYPE);
=======
#ifdef __CLC_SCALAR
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

#if defined(__opencl_c_fp64) && (defined(cl_khr_int64_base_atomics) &&         \
                                 defined(cl_khr_int64_extended_atomics))
#define __CLC_HAVE_64_ATOMIC
#endif
#if defined(__CLC_FPSIZE) &&                                                   \
    (__CLC_FPSIZE < 64 || defined(__CLC_HAVE_64_ATOMIC))
#define __CLC_HAVE_FP_ATOMIC
#endif
#if defined(__CLC_GENSIZE) &&                                                  \
    ((__CLC_GENSIZE == 32) ||                                                  \
     (__CLC_GENSIZE == 64 && defined(__CLC_HAVE_64_ATOMIC)))
#define __CLC_HAVE_INT_ATOMIC
#endif
#if defined(__CLC_HAVE_FP_ATOMIC) || defined(__CLC_HAVE_INT_ATOMIC)

#define __CLC_ATOMIC_GENTYPE __CLC_XCONCAT(atomic_, __CLC_GENTYPE)

#ifdef __CLC_NO_VALUE_ARG
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(                        \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr);
#elif defined(__CLC_RETURN_VOID)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DECL void __CLC_FUNCTION(                                 \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr, __CLC_GENTYPE Value);
#elif defined(__CLC_COMPARE_EXCHANGE)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(                        \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr,                            \
      ADDRSPACE __CLC_GENTYPE *Expected, __CLC_GENTYPE Desired);
#else
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(                        \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr, __CLC_GENTYPE Value);
#endif

<<<<<<< HEAD
#undef FUNCTION
=======
__CLC_DEFINE_ATOMIC(global)
__CLC_DEFINE_ATOMIC(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC()
#endif

#undef __CLC_DEFINE_ATOMIC

#endif // __CLC_HAVE_FP_ATOMIC || __CLC_HAVE_INT_ATOMIC

#undef __CLC_HAVE_INT_ATOMIC
#undef __CLC_HAVE_FP_ATOMIC
#undef __CLC_HAVE_64_ATOMIC

#endif // __CLC_SCALAR
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
