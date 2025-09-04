// RUN: %clang -std=c23 -O0 %s -o %t && %run %t
<<<<<<< HEAD
// UNSUPPORTED: asan, hwasan, rtsan, ubsan
=======
// UNSUPPORTED: asan, hwasan, ubsan
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

#include <stddef.h>
#include <stdlib.h>

extern void *aligned_alloc(size_t alignment, size_t size);

extern void free_aligned_sized(void *p, size_t alignment, size_t size);

int main() {
  volatile void *p = aligned_alloc(128, 1024);
  free_aligned_sized((void *)p, 128, 1024);
  return 0;
}
