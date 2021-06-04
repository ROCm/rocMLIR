#include "Miir.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>
#include <string>

using namespace llvm;
static cl::opt<std::string> args(
    "args", cl::desc("Convolution args"),
    cl::value_desc("Igemm convolution args string"),
    cl::init(
        R"( --operation conv2d --arch gfx906 --num_cu 64 --fil_layout NCHW )"
        R"(--in_layout NCHW --out_layout NCHW --batchsize 128 --in_channels 8 )"
        R"(--out_channels 128 --in_h 32 --in_w 32 --out_h 30 --out_w 30 )"
        R"(--fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 )"
        R"(--conv_stride_w 1 --padding_h 0 --padding_w 0)"));

static cl::opt<std::string>
    option("option", cl::desc("Code gen options: source/header/cflags"),
           cl::value_desc("Igemm convolution option string"),
           cl::init("source"));

int main(int argc, char **argv) {
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR MIOpen Dialect driver\n");

  MiirStatus status = MIIR_SUCCESS;
  // save args
  std::string parameters = args.getValue();

  MiirHandle handle = miirCreateHandle(args.getValue().c_str());

  // Cpp backend source/header/cflags generation
  if ((option.getValue() == "source") || (option.getValue() == "header") ||
      (option.getValue() == "cflags")) {
    status = miirLowerCpp(handle);
    if (status == MIIR_SUCCESS) {
      if (option.getValue() == "source") {
        std::string source = miirGenIgemmSource(handle);
        std::cout << source << std::endl;
      } else if (option.getValue() == "header") {
        std::string header = miirGenIgemmHeader(handle);
        std::cout << header << std::endl;
      } else if (option.getValue() == "cflags") {
        std::string cflags = miirGenIgemmCflags(handle);
        std::cout << cflags << std::endl;
      }
    }
    // Bin backend binary generation
  } else if (option.getValue() == "tuningparams") {
    status = miirLowerTuningParams(handle);
    if (status != MIIR_SUCCESS) {
      return status;
    }

    size_t globalSize, localSize;
    status = miirGetExecutionDims(handle, &globalSize, &localSize);
    if (status != MIIR_SUCCESS) {
      return status;
    }
    std::cout << "ExecutionDims - globalSize=" << globalSize
              << ", localSize=" << localSize << std::endl;

  } else if (option.getValue() == "bin") {
    int count = miirGetKernelCount(handle);
    for (int i = 0; i < count; i++) {
      auto arguments = parameters + " --kernel_id " + std::to_string(i);

      MiirHandle newHandle = miirCreateHandle(arguments.c_str());

      status = miirLowerBin(newHandle);
      if (status != MIIR_SUCCESS) {
        return status;
      }

      size_t size = 0;
      status = miirBufferGet(newHandle, nullptr, &size);
      if (status != MIIR_SUCCESS) {
        return status;
      }
      std::vector<char> buffer(size);
      status = miirBufferGet(newHandle, buffer.data(), &size);
      if (status != MIIR_SUCCESS) {
        return status;
      }
      std::for_each(buffer.begin(), buffer.end(),
                    [](char &c) { std::cout << c; });
      std::cout << std::endl;

      size_t globalSize, localSize;
      status = miirGetExecutionDims(newHandle, &globalSize, &localSize);
      if (status != MIIR_SUCCESS) {
        return status;
      }
      std::cout << "ExecutionDims - globalSize=" << globalSize
                << ", localSize=" << localSize << ", kernelId = " << i
                << std::endl;
      miirDestroyHandle(newHandle);
    }
  }

  miirDestroyHandle(handle);

  return status;
}
