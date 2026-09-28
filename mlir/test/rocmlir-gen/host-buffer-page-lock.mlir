// The host harness page-locks the buffers it hands to the kernel before they are
// copied to the device. HIP can silently drop small asynchronous host-to-device
// copies made out of pageable memory, which leaves the kernel reading zeros from
// an input that never arrived, with no error reported by any HIP call.

// Every buffer is cast to an unranked memref and registered, and all of that
// happens before the kernel is called.
// RUN: rocmlir-gen --arch gfx1101 --operation gemm -p -ph --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=REGISTER
// REGISTER-LABEL: func.func @main
// REGISTER: %[[BUF:.*]] = memref.cast %{{.*}} to memref<*xf32>
// REGISTER-NEXT: gpu.host_register %[[BUF]] : memref<*xf32>
// REGISTER-COUNT-2: gpu.host_register {{.*}} : memref<*xf32>
// REGISTER: call @rock_gemm{{.*}}_gpu

// Under -pv_with_gpu the reference runs on the device as well, so its buffers
// make the same trip and are registered too.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 1 -m 64 -k 64 -n 64 -pv_with_gpu | FileCheck %s --check-prefix=GPUVAL
// GPUVAL-COUNT-6: gpu.host_register {{.*}} : memref<*xf32>
// GPUVAL: call @rock_gemm_gpu
// GPUVAL: call @rock_gemm_ver_gpu

// An f32 GEMM on a non-accelerated target validates on the host instead, so only
// the three buffers the kernel receives are registered.
// RUN: rocmlir-gen --arch gfx1101 --operation gemm -g 1 -m 64 -k 64 -n 64 -pv_with_gpu | FileCheck %s --check-prefix=CPUVAL
// CPUVAL-COUNT-3: gpu.host_register {{.*}} : memref<*xf32>
// CPUVAL-NOT: gpu.host_register

// A scaled GEMM registers its f32 output and its two 8-bit scale operands.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 1 -m 1024 -k 768 -n 1024 -t f4E2M1FN -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU -out_dtype f32 --scaledGemm -pv | FileCheck %s --check-prefix=SCALED
// SCALED: gpu.host_register {{.*}} : memref<*xf32>
// SCALED-COUNT-2: gpu.host_register {{.*}} : memref<*xf8E8M0FNU>
// SCALED: call @rock_gemm_gpu

// Its f4E2M1FN operands are skipped, because a sub-byte memref cannot be cast to
// an unranked memref. This prefix carries only a CHECK-NOT, so that it applies to
// the whole harness rather than to whatever follows an earlier match.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 1 -m 1024 -k 768 -n 1024 -t f4E2M1FN -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU -out_dtype f32 --scaledGemm -pv | FileCheck %s --check-prefix=NO-SUBBYTE
// NO-SUBBYTE-NOT: memref<*xf4E2M1FN>
