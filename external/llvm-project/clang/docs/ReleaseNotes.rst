.. If you want to modify sections/contents permanently, you should modify both
   ReleaseNotes.rst and ReleaseNotesTemplate.txt.

===========================================
Clang |release| |ReleaseNotesTitle|
===========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming Clang |version| release.
     Release notes for previous releases can be found on
     `the Releases Page <https://llvm.org/releases/>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. For the libc++ release notes,
see `this page <https://libcxx.llvm.org/ReleaseNotes.html>`_. All LLVM releases
may be downloaded from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Potentially Breaking Changes
============================

- Clang will now emit a warning if the auto-detected GCC installation
  directory (i.e. the one with the largest version number) does not
  contain libstdc++ include directories although a "complete" GCC
  installation directory containing the include directories is
  available. It is planned to change the auto-detection to prefer the
  "complete" directory in the future.  The warning will disappear if
  the libstdc++ include directories are either installed or removed
  for all GCC installation directories considered by the
  auto-detection; see the output of ``clang -v`` for a list of those
  directories. If the GCC installations cannot be modified and
  maintaining the current choice of the auto-detection is desired, the
  GCC installation directory can be selected explicitly using the
  ``--gcc-install-dir`` command line argument. This will silence the
  warning. It can also be disabled using the
  ``-Wno-gcc-install-dir-libstdcxx`` command line flag.

C/C++ Language Potentially Breaking Changes
-------------------------------------------

C++ Specific Potentially Breaking Changes
-----------------------------------------
- For C++20 modules, the Reduced BMI mode will be the default option. This may introduce
  regressions if your build system supports two-phase compilation model but haven't support
  reduced BMI or it is a compiler bug or a bug in users code.

<<<<<<< HEAD
- The type trait builtin ``__is_referenceable`` has been removed, since it has
  very few users and all the type traits that could benefit from it in the
  standard library already have their own bespoke builtins.
- A workaround for libstdc++4.7 has been removed. Note that 4.8.3 remains the oldest
  supported libstdc++ version.
- Added ``!nonnull/!align`` metadata to load of references for better codegen.
- Checking for integer to enum conversions in constant expressions is more
  strict; in particular, ``const E x = (E)-1;`` is not treated as a constant
  if it's out of range. The Boost numeric_conversion library is impacted by
  this; it was fixed in Boost 1.81. (#GH143034)

- Fully implemented `CWG400 Using-declarations and the `
  `"struct hack" <https://wg21.link/CWG400>`_. Invalid member using-declaration
  whose nested-name-specifier doesn't refer to a base class such as
  ``using CurrentClass::Foo;`` is now rejected in C++98 mode.
=======
- Clang now correctly diagnoses during constant expression evaluation undefined behavior due to member
  pointer access to a member which is not a direct or indirect member of the most-derived object
  of the accessed object but is instead located directly in a sibling class to one of the classes
  along the inheritance hierarchy of the most-derived object as ill-formed.
  Other scenarios in which the member is not member of the most derived object were already
  diagnosed previously. (#GH150709)

  .. code-block:: c++

    struct A {};
    struct B : A {};
    struct C : A { constexpr int foo() const { return 1; } };
    constexpr A a;
    constexpr B b;
    constexpr C c;
    constexpr auto mp = static_cast<int(A::*)() const>(&C::foo);
    static_assert((a.*mp)() == 1); // continues to be rejected
    static_assert((b.*mp)() == 1); // newly rejected
    static_assert((c.*mp)() == 1); // accepted
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

ABI Changes in This Version
---------------------------

AST Dumping Potentially Breaking Changes
----------------------------------------
- How nested name specifiers are dumped and printed changes, keeping track of clang AST changes.

Clang Frontend Potentially Breaking Changes
-------------------------------------------
- Members of anonymous unions/structs are now injected as ``IndirectFieldDecl``
  into the enclosing record even if their names conflict with other names in the
  scope. These ``IndirectFieldDecl`` are marked invalid.

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------
- TypeKind ``ELABORATED`` is not used anymore, per clang AST changes removing
  ElaboratedTypes. The value becomes unused, and all the existing users should
  expect the former underlying type to be reported instead.

What's New in Clang |release|?
==============================

C++ Language Changes
--------------------

- A new family of builtins ``__builtin_*_synthesises_from_spaceship`` has been added. These can be queried to know
  whether the ``<`` (``lt``), ``>`` (``gt``), ``<=`` (``le``), or ``>=`` (``ge``) operators are synthesised from a
  ``<=>``. This makes it possible to optimize certain facilities by using the ``<=>`` operation directly instead of
  doing multiple comparisons.

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

<<<<<<< HEAD
- Implemented `P1061R10 Structured Bindings can introduce a Pack <https://wg21.link/P1061R10>`_.
- Implemented `P2786R13 Trivial Relocatability <https://wg21.link/P2786R13>`_.


- Implemented `P0963R3 Structured binding declaration as a condition <https://wg21.link/P0963R3>`_.

- Implemented `P2719R4 Type-aware allocation and deallocation functions <https://wg21.link/P2719>`_.

- Implemented `P3618R0 Allow attaching main to the global module <https://wg21.link/P3618>`_.

=======
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++17 Feature Support
^^^^^^^^^^^^^^^^^^^^^

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C Language Changes
------------------

<<<<<<< HEAD
- Clang now allows an ``inline`` specifier on a typedef declaration of a
  function type in Microsoft compatibility mode. #GH124869
- Clang now allows ``restrict`` qualifier for array types with pointer elements (#GH92847).
- Clang now diagnoses ``const``-qualified object definitions without an
  initializer. If the object is a variable or field which is zero-initialized,
  it will be diagnosed under the new warning ``-Wdefault-const-init-var`` or
  ``-Wdefault-const-init-field``, respectively. Similarly, if the variable or
  field is not zero-initialized, it will be diagnosed under the new diagnostic
  ``-Wdefault-const-init-var-unsafe`` or ``-Wdefault-const-init-field-unsafe``,
  respectively. The unsafe diagnostic variants are grouped under a new
  diagnostic ``-Wdefault-const-init-unsafe``, which itself is grouped under the
  new diagnostic ``-Wdefault-const-init``. Finally, ``-Wdefault-const-init`` is
  grouped under ``-Wc++-compat`` because these constructs are not compatible
  with C++. #GH19297
- Added ``-Wimplicit-void-ptr-cast``, grouped under ``-Wc++-compat``, which
  diagnoses implicit conversion from ``void *`` to another pointer type as
  being incompatible with C++. (#GH17792)
- Added ``-Wc++-keyword``, grouped under ``-Wc++-compat``, which diagnoses when
  a C++ keyword is used as an identifier in C. (#GH21898)
- Added ``-Wc++-hidden-decl``, grouped under ``-Wc++-compat``, which diagnoses
  use of tag types which are visible in C but not visible in C++ due to scoping
  rules. e.g.,

  .. code-block:: c

    struct S {
      struct T {
        int x;
      } t;
    };
    struct T t; // Invalid C++, valid C, now diagnosed
- Added ``-Wimplicit-int-enum-cast``, grouped under ``-Wc++-compat``, which
  diagnoses implicit conversion from integer types to an enumeration type in C,
  which is not compatible with C++. #GH37027
- Split "implicit conversion from enum type to different enum type" diagnostic
  from ``-Wenum-conversion`` into its own diagnostic group,
  ``-Wimplicit-enum-enum-cast``, which is grouped under both
  ``-Wenum-conversion`` and ``-Wimplicit-int-enum-cast``. This conversion is an
  int-to-enum conversion because the enumeration on the right-hand side is
  promoted to ``int`` before the assignment.
- Added ``-Wtentative-definition-compat``, grouped under ``-Wc++-compat``,
  which diagnoses tentative definitions in C with multiple declarations as
  being incompatible with C++. e.g.,

  .. code-block:: c

    // File scope
    int i;
    int i; // Vaild C, invalid C++, now diagnosed
- Added ``-Wunterminated-string-initialization``, grouped under ``-Wextra``,
  which diagnoses an initialization from a string literal where only the null
  terminator cannot be stored. e.g.,

  .. code-block:: c


    char buf1[3] = "foo"; // -Wunterminated-string-initialization
    char buf2[3] = "flarp"; // -Wexcess-initializers
    char buf3[3] = "fo\0";  // This is fine, no warning.

  This diagnostic can be suppressed by adding the new ``nonstring`` attribute
  to the field or variable being initialized. #GH137705
- Added ``-Wc++-unterminated-string-initialization``, grouped under
  ``-Wc++-compat``, which also diagnoses the same cases as
  ``-Wunterminated-string-initialization``. However, this diagnostic is not
  silenced by the ``nonstring`` attribute as these initializations are always
  incompatible with C++.
- Added ``-Wjump-misses-init``, which is off by default and grouped under
  ``-Wc++-compat``. It diagnoses when a jump (``goto`` to its label, ``switch``
  to its ``case``) will bypass the initialization of a local variable, which is
  invalid in C++.
- Added the existing ``-Wduplicate-decl-specifier`` diagnostic, which is on by
  default, to ``-Wc++-compat`` because duplicated declaration specifiers are
  not valid in C++.
- The ``[[clang::assume()]]`` attribute is now correctly recognized in C. The
  ``__attribute__((assume()))`` form has always been supported, so the fix is
  specific to the attribute syntax used.
- The ``clang-cl`` driver now recognizes ``/std:clatest`` and sets the language
  mode to C23. (#GH147233)

=======
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
C2y Feature Support
^^^^^^^^^^^^^^^^^^^
- Clang now supports `N3355 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3355.htm>`_ Named Loops.

C23 Feature Support
^^^^^^^^^^^^^^^^^^^
<<<<<<< HEAD
- Clang now accepts ``-std=iso9899:2024`` as an alias for C23.
- Added ``__builtin_c23_va_start()`` for compatibility with GCC and to enable
  better diagnostic behavior for the ``va_start()`` macro in C23 and later.
  This also updates the definition of ``va_start()`` in ``<stdarg.h>`` to use
  the new builtin. Fixes #GH124031.
- Implemented `WG14 N2819 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2819.pdf>`_
  which clarified that a compound literal used within a function prototype is
  treated as if the compound literal were within the body rather than at file
  scope.
- Fixed a bug where you could not cast a null pointer constant to type
  ``nullptr_t``. Fixes #GH133644.
- Implemented `WG14 N3037 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3037.pdf>`_
  which allows tag types to be redefined within the same translation unit so
  long as both definitions are structurally equivalent (same tag types, same
  tag names, same tag members, etc). As a result of this paper, ``-Wvisibility``
  is no longer diagnosed in C23 if the parameter is a complete tag type (it
  does still fire when the parameter is an incomplete tag type as that cannot
  be completed).
- Fixed a failed assertion with an invalid parameter to the ``#embed``
  directive. Fixes #GH126940.
- Fixed a crash when a declaration of a ``constexpr`` variable with an invalid
  type. Fixes #GH140887
- Documented `WG14 N3006 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3006.htm>`_
  which clarified how Clang is handling underspecified object declarations.
- Clang now accepts single variadic parameter in type-name. It's a part of
  `WG14 N2975 <https://open-std.org/JTC1/SC22/WG14/www/docs/n2975.pdf>`_
- Fixed a bug with handling the type operand form of ``typeof`` when it is used
  to specify a fixed underlying type for an enumeration. #GH146351

C11 Feature Support
^^^^^^^^^^^^^^^^^^^
- Implemented `WG14 N1285 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1285.htm>`_
  which introduces the notion of objects with a temporary lifetime. When an
  expression resulting in an rvalue with structure or union type and that type
  contains a member of array type, the expression result is an automatic storage
  duration object with temporary lifetime which begins when the expression is
  evaluated and ends at the evaluation of the containing full expression. This
  functionality is also implemented for earlier C language modes because the
  C99 semantics will never be implemented (it would require dynamic allocations
  of memory which leaks, which users would not appreciate).
=======
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

Non-comprehensive list of changes in this release
-------------------------------------------------
- Added ``__builtin_elementwise_fshl`` and ``__builtin_elementwise_fshr``.

- ``__builtin_elementwise_abs`` can now be used in constant expression.

- Added ``__builtin_elementwise_minnumnum`` and ``__builtin_elementwise_maxnumnum``.

- Trapping UBSan (e.g. ``-fsanitize=undefined -fsanitize-trap=undefined``) now
  emits a string describing the reason for trapping into the generated debug
  info. This feature allows debuggers (e.g. LLDB) to display the reason for
  trapping if the trap is reached. The string is currently encoded in the debug
  info as an artificial frame that claims to be inlined at the trap location.
  The function used for the artificial frame is an artificial function whose
  name encodes the reason for trapping. The encoding used is currently the same
  as ``__builtin_verbose_trap`` but might change in the future. This feature is
  enabled by default but can be disabled by compiling with
  ``-fno-sanitize-debug-trap-reasons``. The feature has a ``basic`` and
  ``detailed`` mode (the default). The ``basic`` mode emits a hard-coded string
  per trap kind (e.g. ``Integer addition overflowed``) and the ``detailed`` mode
  emits a more descriptive string describing each individual trap (e.g. ``signed
  integer addition overflow in 'a + b'``). The ``detailed`` mode produces larger
  debug info than ``basic`` but is more helpful for debugging. The
  ``-fsanitize-debug-trap-reasons=`` flag can be used to switch between the
  different modes or disable the feature entirely. Note due to trap merging in
  optimized builds (i.e. in each function all traps of the same kind get merged
  into the same trap instruction) the trap reasons might be removed. To prevent
  this build without optimizations (i.e. use `-O0` or use the `optnone` function
  attribute) or use the `fno-sanitize-merge=` flag in optimized builds.

- ``__builtin_elementwise_max`` and ``__builtin_elementwise_min`` functions for integer types can
  now be used in constant expressions.

- A vector of booleans is now a valid condition for the ternary ``?:`` operator.
  This binds to a simple vector select operation.

- Added ``__builtin_masked_load``, ``__builtin_masked_expand_load``,
  ``__builtin_masked_store``, ``__builtin_masked_compress_store`` for
  conditional memory loads from vectors. Binds to the LLVM intrinsics of the
  same name.

- The ``__builtin_popcountg``, ``__builtin_ctzg``, and ``__builtin_clzg``
  functions now accept fixed-size boolean vectors.

- Use of ``__has_feature`` to detect the ``ptrauth_qualifier`` and ``ptrauth_intrinsics``
  features has been deprecated, and is restricted to the arm64e target only. The
  correct method to check for these features is to test for the ``__PTRAUTH__``
  macro.

- Added a new builtin, ``__builtin_dedup_pack``, to remove duplicate types from a parameter pack.
  This feature is particularly useful in template metaprogramming for normalizing type lists.
  The builtin produces a new, unexpanded parameter pack that can be used in contexts like template
  argument lists or base specifiers.

  .. code-block:: c++

    template <typename...> struct TypeList;

    // The resulting type is TypeList<int, double, char>
    using MyTypeList = TypeList<__builtin_dedup_pack<int, double, int, char, double>...>;

  Currently, the use of ``__builtin_dedup_pack`` is limited to template arguments and base
  specifiers, it also must be used within a template context.

<<<<<<< HEAD
- Support parsing the `cc` operand modifier and alias it to the `c` modifier (#GH127719).
- Added `__builtin_elementwise_exp10`.
- For AMDPGU targets, added `__builtin_v_cvt_off_f32_i4` that maps to the `v_cvt_off_f32_i4` instruction.
- Added `__builtin_elementwise_minnum` and `__builtin_elementwise_maxnum`.
- No longer crashing on invalid Objective-C categories and extensions when
  dumping the AST as JSON. (#GH137320)
- Clang itself now uses split stacks instead of threads for allocating more
  stack space when running on Apple AArch64 based platforms. This means that
  stack traces of Clang from debuggers, crashes, and profilers may look
  different than before.
- Fixed a crash when a VLA with an invalid size expression was used within a
  ``sizeof`` or ``typeof`` expression. (#GH138444)
- ``__builtin_invoke`` has been added to improve the compile time of ``std::invoke``.
- Deprecation warning is emitted for the deprecated ``__reference_binds_to_temporary`` intrinsic.
  ``__reference_constructs_from_temporary`` should be used instead. (#GH44056)
- Added `__builtin_get_vtable_pointer` to directly load the primary vtable pointer from a
  polymorphic object.
- ``libclang`` receives a family of new bindings to query basic facts about
  GCC-style inline assembly blocks, including whether the block is ``volatile``
  and its template string following the LLVM IR ``asm`` format. (#GH143424)
- Clang no longer rejects reinterpret_cast conversions between indirect
  ARC-managed pointers and other pointer types. The prior behavior was overly
  strict and inconsistent with the ARC specification.
=======
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

New Compiler Flags
------------------
- New option ``-fno-sanitize-debug-trap-reasons`` added to disable emitting trap reasons into the debug info when compiling with trapping UBSan (e.g. ``-fsanitize-trap=undefined``).
- New option ``-fsanitize-debug-trap-reasons=`` added to control emitting trap reasons into the debug info when compiling with trapping UBSan (e.g. ``-fsanitize-trap=undefined``).


Lanai Support
^^^^^^^^^^^^^^
- The option ``-mcmodel={small,medium,large}`` is supported again.

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------
- The `-gkey-instructions` compiler flag is now enabled by default when DWARF is emitted for plain C/C++ and optimizations are enabled. (#GH149509)

Removed Compiler Flags
-------------------------

Attribute Changes in Clang
--------------------------

Improvements to Clang's diagnostics
-----------------------------------

- Improve the diagnostics for deleted default constructor errors for C++ class
  initializer lists that don't explicitly list a class member and thus attempt
  to implicitly default construct that member.
- The ``-Wunique-object-duplication`` warning has been added to warn about objects
  which are supposed to only exist once per program, but may get duplicated when
  built into a shared library.
- Fixed a bug where Clang's Analysis did not correctly model the destructor behavior of ``union`` members (#GH119415).
- A statement attribute applied to a ``case`` label no longer suppresses
  'bypassing variable initialization' diagnostics (#84072).
- The ``-Wunsafe-buffer-usage`` warning has been updated to warn
  about unsafe libc function calls.  Those new warnings are emitted
  under the subgroup ``-Wunsafe-buffer-usage-in-libc-call``.
- Diagnostics on chained comparisons (``a < b < c``) are now an error by default. This can be disabled with
  ``-Wno-error=parentheses``.
- Similarly, fold expressions over a comparison operator are now an error by default.
- Clang now better preserves the sugared types of pointers to member.
- Clang now better preserves the presence of the template keyword with dependent
  prefixes.
- Clang now in more cases avoids printing 'type-parameter-X-X' instead of the name of
  the template parameter.
- Clang now respects the current language mode when printing expressions in
  diagnostics. This fixes a bunch of `bool` being printed as `_Bool`, and also
  a bunch of HLSL types being printed as their C++ equivalents.
- Clang now consistently quotes expressions in diagnostics.
- When printing types for diagnostics, clang now doesn't suppress the scopes of
  template arguments contained within nested names.
- The ``-Wshift-bool`` warning has been added to warn about shifting a boolean. (#GH28334)
- Fixed diagnostics adding a trailing ``::`` when printing some source code
  constructs, like base classes.
- The :doc:`ThreadSafetyAnalysis` now supports ``-Wthread-safety-pointer``,
  which enables warning on passing or returning pointers to guarded variables
  as function arguments or return value respectively. Note that
  :doc:`ThreadSafetyAnalysis` still does not perform alias analysis. The
  feature will be default-enabled with ``-Wthread-safety`` in a future release.
- The :doc:`ThreadSafetyAnalysis` now supports reentrant capabilities.
- Clang will now do a better job producing common nested names, when producing
  common types for ternary operator, template argument deduction and multiple return auto deduction.
- The ``-Wsign-compare`` warning now treats expressions with bitwise not(~) and minus(-) as signed integers
  except for the case where the operand is an unsigned integer
  and throws warning if they are compared with unsigned integers (##18878).
- The ``-Wunnecessary-virtual-specifier`` warning (included in ``-Wextra``) has
  been added to warn about methods which are marked as virtual inside a
  ``final`` class, and hence can never be overridden.

- Improve the diagnostics for chained comparisons to report actual expressions and operators (#GH129069).

- Improve the diagnostics for shadows template parameter to report correct location (#GH129060).

- Improve the ``-Wundefined-func-template`` warning when a function template is not instantiated due to being unreachable in modules.

- When diagnosing an unused return value of a type declared ``[[nodiscard]]``, the type
  itself is now included in the diagnostic.

- Clang will now prefer the ``[[nodiscard]]`` declaration on function declarations over ``[[nodiscard]]``
  declaration on the return type of a function. Previously, when both have a ``[[nodiscard]]`` declaration attached,
  the one on the return type would be preferred. This may affect the generated warning message:

  .. code-block:: c++

    struct [[nodiscard("Reason 1")]] S {};
    [[nodiscard("Reason 2")]] S getS();
    void use()
    {
      getS(); // Now diagnoses "Reason 2", previously diagnoses "Reason 1"
    }

- Fixed an assertion when referencing an out-of-bounds parameter via a function
  attribute whose argument list refers to parameters by index and the function
  is variadic. e.g.,

  .. code-block:: c

    __attribute__ ((__format_arg__(2))) void test (int i, ...) { }

  Fixes #GH61635

- Split diagnosing base class qualifiers from the ``-Wignored-Qualifiers`` diagnostic group into a new ``-Wignored-base-class-qualifiers`` diagnostic group (which is grouped under ``-Wignored-qualifiers``). Fixes #GH131935.

- ``-Wc++98-compat`` no longer diagnoses use of ``__auto_type`` or
  ``decltype(auto)`` as though it was the extension for ``auto``. (#GH47900)
- Clang now issues a warning for missing return in ``main`` in C89 mode. (#GH21650)

- Now correctly diagnose a tentative definition of an array with static
  storage duration in pedantic mode in C. (#GH50661)
- No longer diagnosing idiomatic function pointer casts on Windows under
  ``-Wcast-function-type-mismatch`` (which is enabled by ``-Wextra``). Clang
  would previously warn on this construct, but will no longer do so on Windows:

  .. code-block:: c

    typedef void (WINAPI *PGNSI)(LPSYSTEM_INFO);
    HMODULE Lib = LoadLibrary("kernel32");
    PGNSI FnPtr = (PGNSI)GetProcAddress(Lib, "GetNativeSystemInfo");


- An error is now emitted when a ``musttail`` call is made to a function marked with the ``not_tail_called`` attribute. (#GH133509).

- ``-Whigher-precision-for-complex-divison`` warns when:

  -	The divisor is complex.
  -	When the complex division happens in a higher precision type due to arithmetic promotion.
  -	When using the divide and assign operator (``/=``).

  Fixes #GH131127

- ``-Wuninitialized`` now diagnoses when a class does not declare any
  constructors to initialize their non-modifiable members. The diagnostic is
  not new; being controlled via a warning group is what's new. Fixes #GH41104

- Analysis-based diagnostics (like ``-Wconsumed`` or ``-Wunreachable-code``)
  can now be correctly controlled by ``#pragma clang diagnostic``. #GH42199

- Improved Clang's error recovery for invalid function calls.

- Improved bit-field diagnostics to consider the type specified by the
  ``preferred_type`` attribute. These diagnostics are controlled by the flags
  ``-Wpreferred-type-bitfield-enum-conversion`` and
  ``-Wpreferred-type-bitfield-width``. These warnings are on by default as they
  they're only triggered if the authors are already making the choice to use
  ``preferred_type`` attribute.

- ``-Winitializer-overrides`` and ``-Wreorder-init-list`` are now grouped under
  the ``-Wc99-designator`` diagnostic group, as they also are about the
  behavior of the C99 feature as it was introduced into C++20. Fixes #GH47037
- ``-Wreserved-identifier`` now fires on reserved parameter names in a function
  declaration which is not a definition.
- Clang now prints the namespace for an attribute, if any,
  when emitting an unknown attribute diagnostic.

- ``-Wvolatile`` now warns about volatile-qualified class return types
  as well as volatile-qualified scalar return types. Fixes #GH133380

- Several compatibility diagnostics that were incorrectly being grouped under
  ``-Wpre-c++20-compat`` are now part of ``-Wc++20-compat``. (#GH138775)

- Improved the ``-Wtautological-overlap-compare`` diagnostics to warn about overlapping and non-overlapping ranges involving character literals and floating-point literals.
  The warning message for non-overlapping cases has also been improved (#GH13473).

- Fixed a duplicate diagnostic when performing typo correction on function template
  calls with explicit template arguments. (#GH139226)

- Explanatory note is printed when ``assert`` fails during evaluation of a
  constant expression. Prior to this, the error inaccurately implied that assert
  could not be used at all in a constant expression (#GH130458)

- A new off-by-default warning ``-Wms-bitfield-padding`` has been added to alert to cases where bit-field
  packing may differ under the MS struct ABI (#GH117428).

- ``-Watomic-access`` no longer fires on unreachable code. e.g.,

  .. code-block:: c

    _Atomic struct S { int a; } s;
    void func(void) {
      if (0)
        s.a = 12; // Previously diagnosed with -Watomic-access, now silenced
      s.a = 12; // Still diagnosed with -Watomic-access
      return;
      s.a = 12; // Previously diagnosed, now silenced
    }


- A new ``-Wcharacter-conversion`` warns where comparing or implicitly converting
  between different Unicode character types (``char8_t``, ``char16_t``, ``char32_t``).
  This warning only triggers in C++ as these types are aliases in C. (#GH138526)

- Fixed a crash when checking a ``__thread``-specified variable declaration
  with a dependent type in C++. (#GH140509)

- Clang now suggests corrections for unknown attribute names.

- ``-Wswitch`` will now diagnose unhandled enumerators in switches also when
  the enumerator is deprecated. Warnings about using deprecated enumerators in
  switch cases have moved behind a new ``-Wdeprecated-declarations-switch-case``
  flag.

  For example:

  .. code-block:: c

    enum E {
      Red,
      Green,
      Blue [[deprecated]]
    };
    void example(enum E e) {
      switch (e) {
      case Red:   // stuff...
      case Green: // stuff...
      }
    }

  will result in a warning about ``Blue`` not being handled in the switch.

  The warning can be fixed either by adding a ``default:``, or by adding
  ``case Blue:``. Since the enumerator is deprecated, the latter approach will
  trigger a ``'Blue' is deprecated`` warning, which can be turned off with
  ``-Wno-deprecated-declarations-switch-case``.

- Split diagnosis of implicit integer comparison on negation to a new
  diagnostic group ``-Wimplicit-int-comparison-on-negation``, grouped under
  ``-Wimplicit-int-conversion``, so user can turn it off independently.

- Improved the FixIts for unused lambda captures.

- Delayed typo correction was removed from the compiler; immediate typo
  correction behavior remains the same. Delayed typo correction facilities were
  fragile and unmaintained, and the removal closed the following issues:
  #GH142457, #GH139913, #GH138850, #GH137867, #GH137860, #GH107840, #GH93308,
  #GH69470, #GH59391, #GH58172, #GH46215, #GH45915, #GH45891, #GH44490,
  #GH36703, #GH32903, #GH23312, #GH69874.

<<<<<<< HEAD
=======
- Clang no longer emits a spurious -Wdangling-gsl warning in C++23 when
  iterating over an element of a temporary container in a range-based
  for loop.(#GH109793, #GH145164)

>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
- Fixed false positives in ``-Wformat-truncation`` and ``-Wformat-overflow``
  diagnostics when floating-point numbers had both width field and plus or space
  prefix specified. (#GH143951)

- A warning is now emitted when ``main`` is attached to a named module,
  which can be turned off with ``-Wno-main-attached-to-named-module``. (#GH146247)

- Clang now avoids issuing `-Wreturn-type` warnings in some cases where
  the final statement of a non-void function is a `throw` expression, or
  a call to a function that is trivially known to always throw (i.e., its
  body consists solely of a `throw` statement). This avoids certain
  false positives in exception-heavy code, though only simple patterns
  are currently recognized.

- Clang now accepts ``@tparam`` comments on variable template partial
  specializations. (#GH144775)
<<<<<<< HEAD
=======

- Fixed a bug that caused diagnostic line wrapping to not function correctly on
  some systems. (#GH139499)

- Clang now tries to avoid printing file paths that contain ``..``, instead preferring
  the canonical file path if it ends up being shorter.

- Improve the diagnostics for placement new expression when const-qualified
  object was passed as the storage argument. (#GH143708)

- Clang now does not issue a warning about returning from a function declared with
  the ``[[noreturn]]`` attribute when the function body is ended with a call via
  pointer, provided it can be proven that the pointer only points to
  ``[[noreturn]]`` functions.

- Added a separate diagnostic group ``-Wfunction-effect-redeclarations``, for the more pedantic
  diagnostics for function effects (``[[clang::nonblocking]]`` and ``[[clang::nonallocating]]``).
  Moved the warning for a missing (though implied) attribute on a redeclaration into this group.
  Added a new warning in this group for the case where the attribute is missing/implicit on
  an override of a virtual method.
- Fixed fix-it hint for fold expressions. Clang now correctly places the suggested right
  parenthesis when diagnosing malformed fold expressions. (#GH151787)
- Added fix-it hint for when scoped enumerations require explicit conversions for binary operations. (#GH24265)

- Fixed an issue where emitted format-signedness diagnostics were not associated with an appropriate
  diagnostic id. Besides being incorrect from an API standpoint, this was user visible, e.g.:
  "format specifies type 'unsigned int' but the argument has type 'int' [-Wformat]"
  "signedness of format specifier 'u' is incompatible with 'c' [-Wformat]"
  This was misleading, because even though -Wformat is required in order to emit the diagnostics,
  the warning flag the user needs to concerned with here is -Wformat-signedness, which is also
  required and is not enabled by default. With the change you'll now see:
  "format specifies type 'unsigned int' but the argument has type 'int', which differs in signedness [-Wformat-signedness]"
  "signedness of format specifier 'u' is incompatible with 'c' [-Wformat-signedness]"
  and the API-visible diagnostic id will be appropriate.

- Fixed false positives in ``-Waddress-of-packed-member`` diagnostics when
  potential misaligned members get processed before they can get discarded.
  (#GH144729)

- Clang now emits dignostic with correct message in case of assigning to const reference captured in lambda. (#GH105647)

- Fixed false positive in ``-Wmissing-noreturn`` diagnostic when it was requiring the usage of
  ``[[noreturn]]`` on lambdas before C++23 (#GH154493).

- Clang now diagnoses the use of ``#`` and ``##`` preprocessor tokens in
  attribute argument lists in C++ when ``-pedantic`` is enabled. The operators
  can be used in macro replacement lists with the usual preprocessor semantics,
  however, non-preprocessor use of tokens now triggers a pedantic warning in C++.
  Compilation in C mode is unchanged, and still permits these tokens to be used. (#GH147217)

- Clang now diagnoses misplaced array bounds on declarators for template
  specializations in th same way as it already did for other declarators.
  (#GH147333)

- A new warning ``-Walloc-size`` has been added to detect calls to functions
  decorated with the ``alloc_size`` attribute don't allocate enough space for
  the target pointer type.

- The :doc:`ThreadSafetyAnalysis` attributes ``ACQUIRED_BEFORE(...)`` and
  ``ACQUIRED_AFTER(...)`` have been moved to the stable feature set and no
  longer require ``-Wthread-safety-beta`` to be used.
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------
<<<<<<< HEAD

- Clang now outputs correct values when #embed data contains bytes with negative
  signed char values (#GH102798).
- Fixed a crash when merging named enumerations in modules (#GH114240).
- Fixed rejects-valid problem when #embed appears in std::initializer_list or
  when it can affect template argument deduction (#GH122306).
- Fix crash on code completion of function calls involving partial order of function templates
  (#GH125500).
- Fixed clang crash when #embed data does not fit into an array
  (#GH128987).
- Non-local variable and non-variable declarations in the first clause of a ``for`` loop in C are no longer incorrectly
  considered an error in C23 mode and are allowed as an extension in earlier language modes.

- Remove the ``static`` specifier for the value of ``_FUNCTION_`` for static functions, in MSVC compatibility mode.
- Fixed a modules crash where exception specifications were not propagated properly (#GH121245, relanded in #GH129982)
- Fixed a problematic case with recursive deserialization within ``FinishedDeserializing()`` where
  ``PassInterestingDeclsToConsumer()`` was called before the declarations were safe to be passed. (#GH129982)
- Fixed a modules crash where an explicit Constructor was deserialized. (#GH132794)
- Defining an integer literal suffix (e.g., ``LL``) before including
  ``<stdint.h>`` in a freestanding build no longer causes invalid token pasting
  when using the ``INTn_C`` macros. (#GH85995)
- Fixed an assertion failure in the expansion of builtin macros like ``__has_embed()`` with line breaks before the
  closing paren. (#GH133574)
- Fixed a crash in error recovery for expressions resolving to templates. (#GH135621)
- Clang no longer accepts invalid integer constants which are too large to fit
  into any (standard or extended) integer type when the constant is unevaluated.
  Merely forming the token is sufficient to render the program invalid. Code
  like this was previously accepted and is now rejected (#GH134658):
  .. code-block:: c

    #if 1 ? 1 : 999999999999999999999
    #endif
- ``#embed`` directive now diagnoses use of a non-character file (device file)
  such as ``/dev/urandom`` as an error. This restriction may be relaxed in the
  future. See (#GH126629).
- Fixed a clang 20 regression where diagnostics attached to some calls to member functions
  using C++23 "deducing this" did not have a diagnostic location (#GH135522)

- Fixed a crash when a ``friend`` function is redefined as deleted. (#GH135506)
- Fixed a crash when ``#embed`` appears as a part of a failed constant
  evaluation. The crashes were happening during diagnostics emission due to
  unimplemented statement printer. (#GH132641)
- Fixed visibility calculation for template functions. (#GH103477)
- Fixed a bug where an attribute before a ``pragma clang attribute`` or
  ``pragma clang __debug`` would cause an assertion. Instead, this now diagnoses
  the invalid attribute location appropriately. (#GH137861)
- Fixed a crash when a malformed ``_Pragma`` directive appears as part of an
  ``#include`` directive. (#GH138094)
- Fixed a crash during constant evaluation involving invalid lambda captures
  (#GH138832)
- Fixed compound literal is not constant expression inside initializer list
  (#GH87867)
- Fixed a crash when instantiating an invalid dependent friend template specialization.
  (#GH139052)
- Fixed a crash with an invalid member function parameter list with a default
  argument which contains a pragma. (#GH113722)
- Fixed assertion failures when generating name lookup table in modules. (#GH61065, #GH134739)
- Fixed an assertion failure in constant compound literal statements. (#GH139160)
- Fix crash due to unknown references and pointer implementation and handling of
  base classes. (GH139452)
- Fixed an assertion failure in serialization of constexpr structs containing unions. (#GH140130)
- Fixed duplicate entries in TableGen that caused the wrong attribute to be selected. (GH#140701)
- Fixed type mismatch error when 'builtin-elementwise-math' arguments have different qualifiers, this should be well-formed. (#GH141397)
- Constant evaluation now correctly runs the destructor of a variable declared in
  the second clause of a C-style ``for`` loop. (#GH139818)
- Fixed a bug with constexpr evaluation for structs containing unions in case of C++ modules. (#GH143168)
- Fixed incorrect token location when emitting diagnostics for tokens expanded from macros. (#GH143216)
- Fixed an infinite recursion when checking constexpr destructors. (#GH141789)
- Fixed a crash when a malformed using declaration appears in a ``constexpr`` function. (#GH144264)
- Fixed a bug when use unicode character name in macro concatenation. (#GH145240)
=======
- Fix a crash when marco name is empty in ``#pragma push_macro("")`` or
  ``#pragma pop_macro("")``. (#GH149762).
- Fix a crash in variable length array (e.g. ``int a[*]``) function parameter type
  being used in ``_Countof`` expression. (#GH152826).
- ``-Wunreachable-code`` now diagnoses tautological or contradictory
  comparisons such as ``x != 0 || x != 1.0`` and ``x == 0 && x == 1.0`` on
  targets that treat ``_Float16``/``__fp16`` as native scalar types. Previously
  the warning was silently lost because the operands differed only by an implicit
  cast chain. (#GH149967).
- Fix crash in ``__builtin_function_start`` by checking for invalid
  first parameter. (#GH113323).
- Fixed a crash with incompatible pointer to integer conversions in designated
  initializers involving string literals. (#GH154046)
- Clang now emits a frontend error when a function marked with the `flatten` attribute
  calls another function that requires target features not enabled in the caller. This
  prevents a fatal error in the backend.
- Fixed scope of typedefs present inside a template class. (#GH91451)
- Builtin elementwise operators now accept vector arguments that have different
  qualifiers on their elements. For example, vector of 4 ``const float`` values
  and vector of 4 ``float`` values. (#GH155405)
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fix an ambiguous reference to the builtin `type_info` (available when using
  `-fms-compatibility`) with modules. (#GH38400)

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``[[nodiscard]]`` is now respected on Objective-C and Objective-C++ methods
  (#GH141504) and on types returned from indirect calls (#GH142453).
- Fixes some late parsed attributes, when applied to function definitions, not being parsed
  in function try blocks, and some situations where parsing of the function body
  is skipped, such as error recovery and code completion. (#GH153551)
- Using ``[[gnu::cleanup(some_func)]]`` where some_func is annotated with
  ``[[gnu::error("some error")]]`` now correctly triggers an error. (#GH146520)

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^
<<<<<<< HEAD

- Clang now supports implicitly defined comparison operators for friend declarations. (#GH132249)
- Clang now diagnoses copy constructors taking the class by value in template instantiations. (#GH130866)
- Clang is now better at keeping track of friend function template instance contexts. (#GH55509)
- Clang now prints the correct instantiation context for diagnostics suppressed
  by template argument deduction.
- Errors that occur during evaluation of certain type traits and builtins are
  no longer incorrectly emitted when they are used in an SFINAE context. The
  type traits are:

  - ``__is_constructible`` and variants,
  - ``__is_convertible`` and variants,
  - ``__is_assignable`` and variants,
  - ``__reference_binds_to_temporary``,
    ``__reference_constructs_from_temporary``,
    ``__reference_converts_from_temporary``,
  - ``__is_trivially_equality_comparable``.

  The builtin is ``__builtin_common_type``. (#GH132044)
- Clang is now better at instantiating the function definition after its use inside
  of a constexpr lambda. (#GH125747)
- Fixed a local class member function instantiation bug inside dependent lambdas. (#GH59734), (#GH132208)
- Clang no longer crashes when trying to unify the types of arrays with
  certain differences in qualifiers (this could happen during template argument
  deduction or when building a ternary operator). (#GH97005)
- Fixed type alias CTAD issues involving default template arguments. (#GH134471)
- Fixed CTAD issues when initializing anonymous fields with designated initializers. (#GH67173)
- The initialization kind of elements of structured bindings
  direct-list-initialized from an array is corrected to direct-initialization.
- Clang no longer crashes when a coroutine is declared ``[[noreturn]]``. (#GH127327)
- Clang now uses the parameter location for abbreviated function templates in ``extern "C"``. (#GH46386)
- Clang will emit an error instead of crash when use co_await or co_yield in
  C++26 braced-init-list template parameter initialization. (#GH78426)
- Improved fix for an issue with pack expansions of type constraints, where this
  now also works if the constraint has non-type or template template parameters.
  (#GH131798)
- Fixes to partial ordering of non-type template parameter packs. (#GH132562)
- Fix crash when evaluating the trailing requires clause of generic lambdas which are part of
  a pack expansion.
- Fixes matching of nested template template parameters. (#GH130362)
- Correctly diagnoses template template parameters which have a pack parameter
  not in the last position.
- Disallow overloading on struct vs class on dependent types, which is IFNDR, as
  this makes the problem diagnosable.
- Improved preservation of the presence or absence of typename specifier when
  printing types in diagnostics.
- Clang now correctly parses ``if constexpr`` expressions in immediate function context. (#GH123524)
- Fixed an assertion failure affecting code that uses C++23 "deducing this". (#GH130272)
- Clang now properly instantiates destructors for initialized members within non-delegating constructors. (#GH93251)
- Correctly diagnoses if unresolved using declarations shadows template parameters (#GH129411)
- Fixed C++20 aggregate initialization rules being incorrectly applied in certain contexts. (#GH131320)
- Clang was previously coalescing volatile writes to members of volatile base class subobjects.
  The issue has been addressed by propagating qualifiers during derived-to-base conversions in the AST. (#GH127824)
- Correctly propagates the instantiated array type to the ``DeclRefExpr`` that refers to it. (#GH79750), (#GH113936), (#GH133047)
- Fixed a Clang regression in C++20 mode where unresolved dependent call expressions were created inside non-dependent contexts (#GH122892)
- Clang now emits the ``-Wunused-variable`` warning when some structured bindings are unused
  and the ``[[maybe_unused]]`` attribute is not applied. (#GH125810)
- Declarations using class template argument deduction with redundant
  parentheses around the declarator are no longer rejected. (#GH39811)
- Fixed a crash caused by invalid declarations of ``std::initializer_list``. (#GH132256)
- Clang no longer crashes when establishing subsumption between some constraint expressions. (#GH122581)
- Clang now issues an error when placement new is used to modify a const-qualified variable
  in a ``constexpr`` function. (#GH131432)
- Fixed an incorrect TreeTransform for calls to ``consteval`` functions if a conversion template is present. (#GH137885)
- Clang now emits a warning when class template argument deduction for alias templates is used in C++17. (#GH133806)
- Fixed a missed initializer instantiation bug for variable templates. (#GH134526), (#GH138122)
- Fix a crash when checking the template template parameters of a dependent lambda appearing in an alias declaration.
  (#GH136432), (#GH137014), (#GH138018)
- Fixed an assertion when trying to constant-fold various builtins when the argument
  referred to a reference to an incomplete type. (#GH129397)
- Fixed a crash when a cast involved a parenthesized aggregate initialization in dependent context. (#GH72880)
- No longer crashes when instantiating invalid variable template specialization
  whose type depends on itself. (#GH51347), (#GH55872)
- Improved parser recovery of invalid requirement expressions. In turn, this
  fixes crashes from follow-on processing of the invalid requirement. (#GH138820)
- Fixed the handling of pack indexing types in the constraints of a member function redeclaration. (#GH138255)
- Clang now correctly parses arbitrary order of ``[[]]``, ``__attribute__`` and ``alignas`` attributes for declarations (#GH133107)
- Fixed a crash when forming an invalid function type in a dependent context. (#GH138657) (#GH115725) (#GH68852)
- Fixed a function declaration mismatch that caused inconsistencies between concepts and variable template declarations. (#GH139476)
- Clang no longer segfaults when there is a configuration mismatch between modules and their users (http://crbug.com/400353616).
- Fix an incorrect deduction when calling an explicit object member function template through an overload set address.
- Fixed bug in constant evaluation that would allow using the value of a
  reference in its own initializer in C++23 mode (#GH131330).
- Clang could incorrectly instantiate functions in discarded contexts (#GH140449)
- Fix instantiation of default-initialized variable template specialization. (#GH140632) (#GH140622)
- Clang modules now allow a module and its user to differ on TrivialAutoVarInit*
- Fixed an access checking bug when initializing non-aggregates in default arguments (#GH62444), (#GH83608)
- Fixed a pack substitution bug in deducing class template partial specializations. (#GH53609)
- Fixed a crash when constant evaluating some explicit object member assignment operators. (#GH142835)
- Fixed an access checking bug when substituting into concepts (#GH115838)
- Fix a bug where private access specifier of overloaded function not respected. (#GH107629)
- Correctly handle allocations in the condition of a ``if constexpr``.(#GH120197) (#GH134820)
- Fixed a crash when handling invalid member using-declaration in C++20+ mode. (#GH63254)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed type checking when a statement expression ends in an l-value of atomic type. (#GH106576)
- Fixed uninitialized use check in a lambda within CXXOperatorCallExpr. (#GH129198)
- Fixed a malformed printout of ``CXXParenListInitExpr`` in certain contexts.
- Fixed a malformed printout of certain calling convention function attributes. (#GH143160)
- Fixed dependency calculation for TypedefTypes (#GH89774)
- The ODR checker now correctly hashes the names of conversion operators. (#GH143152)
- Fixed the right parenthesis source location of ``CXXTemporaryObjectExpr``. (#GH143711)
- Fixed a crash when performing an ``IgnoreUnlessSpelledInSource`` traversal of ASTs containing ``catch(...)`` statements. (#GH146103)
=======
- Diagnose binding a reference to ``*nullptr`` during constant evaluation. (#GH48665)
- Suppress ``-Wdeprecated-declarations`` in implicitly generated functions. (#GH147293)
- Fix a crash when deleting a pointer to an incomplete array (#GH150359).
- Fixed a mismatched lambda scope bug when propagating up ``consteval`` within nested lambdas. (#GH145776)
- Fix an assertion failure when expression in assumption attribute
  (``[[assume(expr)]]``) creates temporary objects.
- Fix the dynamic_cast to final class optimization to correctly handle
  casts that are guaranteed to fail (#GH137518).
- Fix bug rejecting partial specialization of variable templates with auto NTTPs (#GH118190).
- Fix a crash if errors "member of anonymous [...] redeclares" and
  "intializing multiple members of union" coincide (#GH149985).
- Fix a crash when using ``explicit(bool)`` in pre-C++11 language modes. (#GH152729)
- Fix the parsing of variadic member functions when the ellipis immediately follows a default argument.(#GH153445)
- Fixed a bug that caused ``this`` captured by value in a lambda with a dependent explicit object parameter to not be
  instantiated properly. (#GH154054)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fix incorrect name qualifiers applied to alias CTAD. (#GH136624)
- Fixed ElaboratedTypes appearing within NestedNameSpecifier, which was not a
  legal representation. This is fixed because ElaboratedTypes don't exist anymore. (#GH43179) (#GH68670) (#GH92757)
- Fix unrecognized html tag causing undesirable comment lexing (#GH152944)
- Fix comment lexing of special command names (#GH152943)
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

- Bump the default code object version to 6. ROCm 6.3 is required to run any program compiled with COV6.
- Introduced a new target specific builtin ``__builtin_amdgcn_processor_is``,
  a late / deferred query for the current target processor
- Introduced a new target specific builtin ``__builtin_amdgcn_is_invocable``,
  which enables fine-grained, per-builtin, feature availability

NVPTX Support
^^^^^^^^^^^^^^

X86 Support
^^^^^^^^^^^
- More SSE, AVX and AVX512 intrinsics, including initializers and general
  arithmetic can now be used in C++ constant expressions.
- Some SSE, AVX and AVX512 intrinsics have been converted to wrap
  generic __builtin intrinsics.
- NOTE: Please avoid use of the __builtin_ia32_* intrinsics - these are not
  guaranteed to exist in future releases, or match behaviour with previous
  releases of clang or other compilers.

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

<<<<<<< HEAD
- Implementation of modal 8-bit floating point intrinsics in accordance with
  the Arm C Language Extensions (ACLE)
  `as specified here <https://github.com/ARM-software/acle/blob/main/main/acle.md#modal-8-bit-floating-point-extensions>`_
  is now available.
- Support has been added for the following processors (command-line identifiers in parentheses):
  - Arm Cortex-A320 (``cortex-a320``)
- For ARM targets, cc1as now considers the FPU's features for the selected CPU or Architecture.
- The ``+nosimd`` attribute is now fully supported for ARM. Previously, this had no effect when being used with
  ARM targets, however this will now disable NEON instructions being generated. The ``simd`` option is
  also now printed when the ``--print-supported-extensions`` option is used.
- When a feature that depends on NEON (``simd``) is used, NEON is now automatically enabled.
- When NEON is disabled (``+nosimd``), all features that depend on NEON will now be disabled.

-  Support for __ptrauth type qualifier has been added.

- For AArch64, added support for generating executable-only code sections by using the
  ``-mexecute-only`` or ``-mpure-code`` compiler flags. (#GH125688)
- Added ``-msve-streaming-vector-bits=`` flag, which allows specifying the
  SVE vector width in streaming mode.

=======
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

- Add support for OHOS on loongarch64.

- Add target attribute support for function. Supported formats include:
  * `arch=<arch>` strings - specifies architecture features for a function (equivalent to `-march=<arch>`).
  * `tune=<cpu>` strings - specifies the tune CPU for a function (equivalent to `-mtune`).
  * `<feature>`/`no-<feature>` - enables/disables specific features.

- Add support for the `_Float16` type. And fix incorrect ABI lowering of `_Float16`
  in the case of structs containing fp16 that are eligible for passing via `GPR+FPR`
  or `FPR+FPR`. Also fix `int16` -> `__fp16` conversion code gen, which uses generic LLVM
  IR rather than `llvm.convert.to.fp16` intrinsics.

- Add support for the `__bf16` type.

- Fix incorrect _BitInt(N>64) alignment. Now consistently uses 16-byte alignment for all
  `_BitInt(N)` where N > 64.

RISC-V Support
^^^^^^^^^^^^^^

- Add support for `__attribute__((interrupt("rnmi")))` to be used with the `Smrnmi` extension.
  With this the `Smrnmi` extension is fully supported.

- Add `-march=unset` to clear any previous `-march=` value. This ISA string will
  be computed from `-mcpu` or the platform default.

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Support
^^^^^^^^^^^^

AIX Support
^^^^^^^^^^^

NetBSD Support
^^^^^^^^^^^^^^

WebAssembly Support
^^^^^^^^^^^^^^^^^^^

AVR Support
^^^^^^^^^^^

DWARF Support in Clang
----------------------

Floating Point Support in Clang
-------------------------------

Fixed Point Support in Clang
----------------------------

AST Matchers
------------
- Removed elaboratedType matchers, and related nested name specifier changes,
  following the corresponding changes in the clang AST.
- Ensure ``hasBitWidth`` doesn't crash on bit widths that are dependent on template
  parameters.

- Add a boolean member ``IgnoreSystemHeaders`` to ``MatchFinderOptions``. This
  allows it to ignore nodes in system headers when traversing the AST.

- ``hasConditionVariableStatement`` now supports ``for`` loop, ``while`` loop
  and ``switch`` statements.

clang-format
------------
- Add ``SpaceInEmptyBraces`` option and set it to ``Always`` for WebKit style.

libclang
--------

Code Completion
---------------

Static Analyzer
---------------
- The Clang Static Analyzer now handles parenthesized initialization.
  (#GH148875)
- ``__datasizeof`` (C++) and ``_Countof`` (C) no longer cause a failed assertion
  when given an operand of VLA type. (#GH151711)

New features
^^^^^^^^^^^^

<<<<<<< HEAD
- A new flag - `-static-libclosure` was introduced to support statically linking
  the runtime for the Blocks extension on Windows. This flag currently only
  changes the code generation, and even then, only on Windows. This does not
  impact the linker behaviour like the other `-static-*` flags.
- OpenACC support, enabled via `-fopenacc` has reached a level of completeness
  to finally be at least notionally usable. Currently, the OpenACC 3.4
  specification has been completely implemented for Sema and AST creation, so
  nodes will show up in the AST after having been properly checked. Lowering is
  currently a work in progress, with compute, loop, and combined constructs
  partially implemented, plus a handful of data and executable constructs
  implemented. Lowering will only work in Clang-IR mode (so only with a compiler
  built with Clang-IR enabled, and with `-fclangir` used on the command line).
  However, note that the Clang-IR implementation status is also quite partial,
  so frequent 'not yet implemented' diagnostics should be expected.  Also, the
  ACC MLIR dialect does not currently implement any lowering to LLVM-IR, so no
  code generation is possible for OpenACC.

=======
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
Crash and bug fixes
^^^^^^^^^^^^^^^^^^^
- Fixed a crash in the static analyzer that when the expression in an
  ``[[assume(expr)]]`` attribute was enclosed in parentheses.  (#GH151529)
- Fixed a crash when parsing ``#embed`` parameters with unmatched closing brackets. (#GH152829)

- Fixed a crash in ``UnixAPIMisuseChecker`` and ``MallocChecker`` when analyzing
  code with non-standard ``getline`` or ``getdelim`` function signatures. (#GH144884)

Improvements
^^^^^^^^^^^^

Moved checkers
^^^^^^^^^^^^^^

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------
- Exposed `clang_getCursorLanguage` via `Cursor.language`.

OpenMP Support
--------------
- Added parsing and semantic analysis support for the ``need_device_addr``
  modifier in the ``adjust_args`` clause.
- Allow array length to be omitted in array section subscript expression.
- Fixed non-contiguous strided update in the ``omp target update`` directive with the ``from`` clause.
- Properly handle array section/assumed-size array privatization in C/C++.

Improvements
^^^^^^^^^^^^

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the `Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_.
