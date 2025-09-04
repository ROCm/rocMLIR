// RUN: %clang -std=c23 -O0 %s -o %t && %run %t
<<<<<<< HEAD
// UNSUPPORTED: asan, hwasan, rtsan, ubsan
=======
// UNSUPPORTED: asan, hwasan, ubsan
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

#include <stddef.h>
#include <stdlib.h>

extern void free_sized(void *p, size_t size);

int main() {
  volatile void *p = malloc(64);
  free_sized((void *)p, 64);
  return 0;
}
