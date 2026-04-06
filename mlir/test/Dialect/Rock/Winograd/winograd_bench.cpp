// Winograd kernel benchmark using HIP C API.
// Assembles a Rage v4_9 kernel, loads it, packs arguments per the V2 ABI
// (232-byte kernarg segment), launches, and measures performance.
//
// Build:
//   hipcc -o winograd_bench winograd_bench.cpp -lhsa-runtime64
//
// Usage:
//   ROCMLIR_WINOGRAD_KERNEL_DIR=<path> ./winograd_bench [dtype] [N C H W K]

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err = (expr);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %d at %s:%d: %s\n", err, __FILE__, __LINE__, \
              hipGetErrorString(err));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// V2 ABI kernel argument struct (232 bytes, matching Rage metadata.inc)
#ifdef _MSC_VER
#pragma pack(push, 1)
struct WinoV2Args {
#else
struct __attribute__((packed)) WinoV2Args {
#endif
  uint32_t N;           // 0
  uint32_t C;           // 4
  uint32_t H;           // 8
  uint32_t W;           // 12
  uint32_t K;           // 16
  uint32_t n_groups;    // 20
  uint64_t flags64;     // 24
  void *data_addr;      // 32
  void *filter_addr;    // 40
  void *output_addr;    // 48
  uint64_t reserved0;   // 56
  uint32_t R;           // 64
  uint32_t S;           // 68
  int32_t pad_h;        // 72
  int32_t pad_w;        // 76
  uint32_t out_h;       // 80
  uint32_t out_w;       // 84
  void *bias_addr;      // 88
  float alpha;          // 96
  float beta;           // 100
  uint64_t d_offset;    // 104
  uint64_t f_offset;    // 112
  uint64_t o_offset;    // 120
  uint64_t b_offset;    // 128
  uint32_t d_N_stride;  // 136
  uint32_t d_C_stride;  // 140
  uint32_t d_H_stride;  // 144
  uint32_t d_W_stride;  // 148 (reserved in Rage, but present)
  uint32_t f_K_stride;  // 152
  uint32_t f_C_stride;  // 156
  uint32_t f_R_stride;  // 160
  uint32_t f_S_stride;  // 164 (reserved in Rage)
  uint32_t o_N_stride;  // 168
  uint32_t o_K_stride;  // 172
  uint32_t o_H_stride;  // 176
  uint32_t o_W_stride;  // 180 (reserved in Rage)
  uint32_t G;           // 184
  uint32_t d_G_stride;  // 188
  uint32_t f_G_stride;  // 192
  uint32_t o_G_stride;  // 196
  uint8_t activation_mode; // 200
  uint8_t sync_limit;     // 201
  uint8_t sync_period;    // 202
  uint8_t reserved1;      // 203
  uint32_t reserved2;     // 204
  void *sync_addr;        // 208
  void *acc_addr;         // 216
  uint64_t a_offset;      // 224
};
#ifdef _MSC_VER
#pragma pack(pop)
#endif
static_assert(sizeof(WinoV2Args) == 232, "V2 ABI must be 232 bytes");

static std::string getRocmBinDir() {
  const char *rocm = std::getenv("ROCM_PATH");
#ifdef _WIN32
  std::string base = rocm ? rocm : "C:\\Program Files\\AMD\\ROCm";
  return base + "\\llvm\\bin\\";
#else
  std::string base = rocm ? rocm : "/opt/rocm";
  return base + "/llvm/bin/";
#endif
}

static std::string getTempFile(const char *name) {
#ifdef _WIN32
  char tmp[260];
  GetTempPathA(sizeof(tmp), tmp);
  return std::string(tmp) + name;
#else
  return std::string("/tmp/") + name;
#endif
}

static std::string assembleKernel(const char *arch, const char *kernelDir,
                                  const char *kernelFile) {
  std::string binDir = getRocmBinDir();
  std::string srcPath = std::string(kernelDir) + "/" + kernelFile;
  std::string objPath = getTempFile("wino_bench.o");
  std::string hsacoPath = getTempFile("wino_bench.hsaco");

  char cmd[2048];
  snprintf(cmd, sizeof(cmd),
           "%sclang -x assembler -target amdgcn-amd-amdhsa "
           "-mcpu=%s -I%s -c %s -o %s 2>&1",
           binDir.c_str(), arch, kernelDir, srcPath.c_str(), objPath.c_str());
  if (system(cmd) != 0) {
    fprintf(stderr, "Assembly failed\n");
    exit(1);
  }
  snprintf(cmd, sizeof(cmd),
           "%sld.lld -shared %s -o %s 2>&1",
           binDir.c_str(), objPath.c_str(), hsacoPath.c_str());
  if (system(cmd) != 0) {
    fprintf(stderr, "Linking failed\n");
    exit(1);
  }

  FILE *f = fopen(hsacoPath.c_str(), "rb");
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::string binary(sz, '\0');
  fread(&binary[0], 1, sz, f);
  fclose(f);
  return binary;
}

int main(int argc, char **argv) {
  const char *dtype = (argc > 1) ? argv[1] : "f16";
  int N = (argc > 2) ? atoi(argv[2]) : 1;
  int C = (argc > 3) ? atoi(argv[3]) : 64;
  int H = (argc > 4) ? atoi(argv[4]) : 56;
  int W = (argc > 5) ? atoi(argv[5]) : 56;
  int K = (argc > 6) ? atoi(argv[6]) : 64;
  int pad_h = (argc > 7) ? atoi(argv[7]) : 1;
  int pad_w = pad_h;

  const int R = 3, S = 3;
  const int out_h = H + 2 * pad_h - R + 1;
  const int out_w = W + 2 * pad_w - S + 1;
  const int group = 1;

  const char *kernelDir = getenv("ROCMLIR_WINOGRAD_KERNEL_DIR");
  if (!kernelDir) {
    fprintf(stderr, "Set ROCMLIR_WINOGRAD_KERNEL_DIR\n");
    return 1;
  }

  // Select kernel file and compute element size
  const char *kernelFile;
  const char *kernelName;
  int elemSize;
  if (strcmp(dtype, "f16") == 0) {
    kernelFile = "Conv_Winograd_Rage_v4_9_0_fp16_fp32acc_f2x3_stride1.s";
    kernelName = "miopenSp3AsmConvRage_v4_9_0_gfx9_fp16_fp32acc_f2x3_stride1";
    elemSize = 2;
  } else if (strcmp(dtype, "f32") == 0) {
    kernelFile = "Conv_Winograd_Rage_v4_9_0_fp32_fp32acc_f2x3_stride1.s";
    kernelName = "miopenSp3AsmConvRage_v4_9_0_gfx9_fp32_fp32acc_f2x3_stride1";
    elemSize = 4;
  } else {
    fprintf(stderr, "Unsupported dtype: %s\n", dtype);
    return 1;
  }

  // Get arch
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  char arch[64];
  snprintf(arch, sizeof(arch), "gfx%d%x%x",
           prop.gcnArchName[3] - '0',
           prop.gcnArchName[4] >= 'a' ? prop.gcnArchName[4] - 'a' + 10
                                      : prop.gcnArchName[4] - '0',
           prop.gcnArchName[5] >= 'a' ? prop.gcnArchName[5] - 'a' + 10
                                      : prop.gcnArchName[5] - '0');
  // Use gcnArchName directly
  strncpy(arch, prop.gcnArchName, sizeof(arch) - 1);
  // Strip features like ":sramecc+:xnack-"
  char *colon = strchr(arch, ':');
  if (colon) *colon = '\0';
  printf("GPU: %s (%s), CUs: %d\n", prop.name, arch, prop.multiProcessorCount);

  int numCU = prop.multiProcessorCount;
  // n_groups must satisfy: K/32 <= n_groups
  // MIOpen's Rage model tries n_dispatches 1..8 and picks best WTI
  // Use numCU for n_groups (1 dispatch), matching MIOpen default
  int n_groups = numCU;
  // Ensure minimum constraint
  int minNG = (K + 31) / 32;
  if (n_groups < minNG) n_groups = minNG;

  // Assemble
  printf("Assembling %s for %s...\n", kernelFile, arch);
  std::string hsaco = assembleKernel(arch, kernelDir, kernelFile);
  printf("HSACO: %zu bytes\n", hsaco.size());

  // Load module
  hipModule_t module;
  HIP_CHECK(hipModuleLoadData(&module, hsaco.data()));

  hipFunction_t func;
  HIP_CHECK(hipModuleGetFunction(&func, module, kernelName));
  printf("Kernel loaded: %s\n", kernelName);

  // Allocate device buffers
  size_t inputBytes = (size_t)N * C * H * W * elemSize;
  size_t filterBytes = (size_t)K * C * R * S * elemSize;
  size_t outputBytes = (size_t)N * K * out_h * out_w * elemSize;

  void *d_input, *d_filter, *d_output;
  HIP_CHECK(hipMalloc(&d_input, inputBytes));
  HIP_CHECK(hipMalloc(&d_filter, filterBytes));
  HIP_CHECK(hipMalloc(&d_output, outputBytes));
  HIP_CHECK(hipMemset(d_input, 0, inputBytes));
  HIP_CHECK(hipMemset(d_filter, 0, filterBytes));
  HIP_CHECK(hipMemset(d_output, 0, outputBytes));

  // Fill kernel arguments
  WinoV2Args args;
  memset(&args, 0, sizeof(args));
  args.N = N;
  args.C = C;
  args.H = H;
  args.W = W;
  args.K = K;
  args.n_groups = n_groups;

  // Flags matching MIOpen's Rage solver exactly:
  // F_DENORMS_RND_ENABLE(3) | F_NKCHR_STRIDES(9) | F_TENSOR_OFFSETS(13) |
  // F_USE_ACTIVATION_MODE(14) | F_USE_EXTENDED_FLAGS_64(15)
  args.flags64 = (1ULL << 3) | (1ULL << 9) | (1ULL << 13) | (1ULL << 14) | (1ULL << 15);

  args.data_addr = d_input;
  args.filter_addr = d_filter;
  args.output_addr = d_output;
  args.R = R;
  args.S = S;
  args.pad_h = pad_h;
  args.pad_w = pad_w;
  args.out_h = out_h;
  args.out_w = out_w;
  args.alpha = 1.0f;
  args.beta = 0.0f;

  // NCHW strides in ELEMENTS (the kernel handles element-to-byte conversion)
  args.d_H_stride = W;
  args.d_C_stride = H * W;
  args.d_N_stride = C * H * W;
  args.d_G_stride = C * H * W;

  // KCRS strides in ELEMENTS
  args.f_R_stride = S;
  args.f_C_stride = R * S;
  args.f_K_stride = C * R * S;
  args.f_G_stride = K * C * R * S;

  // Output NKHW strides in ELEMENTS
  args.o_H_stride = out_w;
  args.o_K_stride = out_h * out_w;
  args.o_N_stride = K * out_h * out_w;
  args.o_G_stride = K * out_h * out_w;

  args.G = group;
  args.activation_mode = 0; // identity

  printf("Args struct size: %zu bytes\n", sizeof(args));
  printf("Config: N=%d C=%d H=%dx%d K=%d R=%dx%d pad=%d out=%dx%d\n",
         N, C, H, W, K, R, S, pad_h, out_h, out_w);

  // Launch parameters
  int blockSize = 768;
  int gridX = n_groups * group;
  // The kernel descriptor already specifies group_segment_fixed_size = 65536.
  // Pass 0 for dynamic shared memory -- the fixed LDS is handled by the KD.
  size_t sharedMem = 0;
  size_t argSize = sizeof(args);
  void *config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
      HIP_LAUNCH_PARAM_END};

  printf("Launching: grid=(%d,1,1) block=(%d,1,1) shared=%zuKB\n",
         gridX, blockSize, sharedMem / 1024);

  // Debug: print key arg offsets
  printf("Arg offsets: N=%zu C=%zu flags64=%zu data=%zu filter=%zu output=%zu\n",
         offsetof(WinoV2Args, N), offsetof(WinoV2Args, C),
         offsetof(WinoV2Args, flags64), offsetof(WinoV2Args, data_addr),
         offsetof(WinoV2Args, filter_addr), offsetof(WinoV2Args, output_addr));
  printf("  R=%zu pad_h=%zu d_N_stride=%zu G=%zu activation=%zu\n",
         offsetof(WinoV2Args, R), offsetof(WinoV2Args, pad_h),
         offsetof(WinoV2Args, d_N_stride), offsetof(WinoV2Args, G),
         offsetof(WinoV2Args, activation_mode));
  printf("  data_ptr=%p filter_ptr=%p output_ptr=%p\n",
         args.data_addr, args.filter_addr, args.output_addr);
  printf("  argSize=%zu, &args=%p\n", argSize, (void*)&args);
  fflush(stdout);

  // Try with just 1 launch first
  printf("Launching single kernel...\n"); fflush(stdout);
  hipError_t launchErr = hipModuleLaunchKernel(
      func, gridX, 1, 1, blockSize, 1, 1,
      sharedMem, nullptr, nullptr, config);
  printf("Launch returned: %d\n", launchErr); fflush(stdout);
  if (launchErr != hipSuccess) {
    printf("Launch failed: %s\n", hipGetErrorString(launchErr));
    // Try with smaller grid
    printf("Retrying with grid=1...\n"); fflush(stdout);
    launchErr = hipModuleLaunchKernel(
        func, 1, 1, 1, blockSize, 1, 1,
        sharedMem, nullptr, nullptr, config);
    printf("Retry returned: %d\n", launchErr); fflush(stdout);
  }
  hipError_t syncErr = hipDeviceSynchronize();
  printf("Sync returned: %d (%s)\n", syncErr, hipGetErrorString(syncErr));
  fflush(stdout);
  if (syncErr != hipSuccess) {
    printf("Kernel execution failed. Exiting.\n");
    hipFree(d_input); hipFree(d_filter); hipFree(d_output);
    hipModuleUnload(module);
    return 1;
  }
  printf("Warmup done\n");

  // Benchmark
  const int nIter = 100;
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start, nullptr));
  for (int i = 0; i < nIter; i++) {
    HIP_CHECK(hipModuleLaunchKernel(func, gridX, 1, 1, blockSize, 1, 1,
                                     sharedMem, nullptr, nullptr, config));
  }
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));

  float elapsedMs;
  HIP_CHECK(hipEventElapsedTime(&elapsedMs, start, stop));
  double avgUs = elapsedMs * 1000.0 / nIter;

  double flops = 2.0 * N * C * K * out_h * out_w * R * S;
  double tflops = flops / (avgUs * 1e-6) / 1e12;

  printf("\n=== Winograd Rage v4.9 %s Results ===\n", dtype);
  printf("  Avg time: %.2f us\n", avgUs);
  printf("  TFlops:   %.2f\n", tflops);

  // Cleanup
  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_filter));
  HIP_CHECK(hipFree(d_output));
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return 0;
}
