// REQUIRES: x86-registered-target
<<<<<<< HEAD
// RUN: %clang -target x86_64-windows-gnu -x c++-module -std=gnu++23 -c -o /dev/null -Xclang -disable-llvm-passes %s
=======
// RUN: %clang -target x86_64-windows-gnu -x c++-module -std=gnu++23 -fno-modules-reduced-bmi \
// RUN:     -c -o /dev/null -Xclang -disable-llvm-passes %s
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

// Make sure the command succeeds and doesn't break on the -exception-model flag in cc1.
export module empty;
