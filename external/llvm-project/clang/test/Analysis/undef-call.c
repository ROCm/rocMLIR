<<<<<<< HEAD
// RUN: rm -rf %T/ctudir
// RUN: mkdir %T/ctudir
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -analyzer-config experimental-enable-naive-ctu-analysis=true -analyzer-config ctu-dir=%T/ctudir -verify %s
=======
// RUN: rm -rf %t.dir/ctudir
// RUN: mkdir -p %t.dir/ctudir
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -analyzer-config experimental-enable-naive-ctu-analysis=true -analyzer-config ctu-dir=%t.dir/ctudir -verify %s
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
// expected-no-diagnostics

struct S {
  void (*fp)(void);
};

int main(void) {
  struct S s;
  // This will cause the analyzer to look for a function definition that has
  // no FunctionDecl. It used to cause a crash in AnyFunctionCall::getRuntimeDefinition.
  // It would only occur when CTU analysis is enabled.
  s.fp();
}
