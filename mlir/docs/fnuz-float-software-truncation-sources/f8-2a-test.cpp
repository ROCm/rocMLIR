#include <cstdint>
#include <cstdio>
#include <limits>

uint8_t e4m3, e5m2;

extern void foo(float v);

// infinity, NaN, max-f32, max-e5m2, max-e4m3, 1, 0, min-f32, min-e5m2,
// min-e4m3, and the negatives of each.

int main(int argc, char *argv[]) {
  foo(std::numeric_limits<float>::quiet_NaN());
  std::printf("quiet NaN\n");
  std::printf("e4m3: %02x\n", e4m3);
  std::printf("e5m2: %02x\n", e5m2);

  foo(std::numeric_limits<float>::signaling_NaN());
  std::printf("\nsignaling NaN\n");
  std::printf("e4m3: %02x\n", e4m3);
  std::printf("e5m2: %02x\n", e5m2);

  foo(std::numeric_limits<float>::infinity());
  std::printf("\ninfinity\n");
  std::printf("e4m3: %02x\n", e4m3);
  std::printf("e5m2: %02x\n", e5m2);

  foo(std::numeric_limits<float>::max());
  std::printf("\nmax f32\n");
  std::printf("e4m3: %02x\n", e4m3);
  std::printf("e5m2: %02x\n", e5m2);

  // ...

  foo(2.0);
  std::printf("\n2.0\n");
  std::printf("e4m3: %02x\n", e4m3);
  std::printf("e5m2: %02x\n", e5m2);

  foo(2.5);
  std::printf("\n2.5\n");
  std::printf("e4m3: %02x\n", e4m3);
  std::printf("e5m2: %02x\n", e5m2);

  foo(-6.0);
  std::printf("\n-6.0\n");
  std::printf("e4m3: %02x\n", e4m3);
  std::printf("e5m2: %02x\n", e5m2);

  // ...

  foo(std::numeric_limits<float>::min());
  std::printf("\nmin f32\n");
  std::printf("e4m3: %02x\n", e4m3);
  std::printf("e5m2: %02x\n", e5m2);

  // ...
}
