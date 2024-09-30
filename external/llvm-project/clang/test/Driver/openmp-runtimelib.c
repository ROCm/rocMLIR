// REQUIRES: amdgpu-registered-target

// Asan-Debug: /lib-debug/asan
// Asan-Devel: /lib/asan
// Asan-Perf: /lib-perf/asan

// RUN:  %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a -fopenmp-runtimelib=lib-debug %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Debug %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a -fopenmp-runtimelib=lib-perf %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Perf %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a -fopenmp-runtimelib=lib %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Devel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a -fopenmp-target-fast %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Default %s

// RUN: not %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a -fopenmp-runtimelib=oopsy %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Error %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a:xnack+ -fopenmp-runtimelib=lib-debug -fsanitize=address -shared-libasan %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefix=Asan-Debug %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a:xnack+ -fopenmp-runtimelib=lib -fsanitize=address -shared-libasan %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefix=Asan-Devel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a:xnack+ -fopenmp-runtimelib=lib-perf -fsanitize=address -shared-libasan %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefix=Asan-Perf %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a:xnack+ -fopenmp-target-fast -fsanitize=address -shared-libasan %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefix=Asan-Devel %s

// Debug: /lib-debug
// Perf: /lib-perf
// Devel: /../lib
// Default: /../lib
// Error: clang: error: unsupported argument 'oopsy' to option '-fopenmp-runtimelib='
