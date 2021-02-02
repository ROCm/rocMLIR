#include "mlir-miopen-lib.hpp"
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

  mlir::MlirHandle handle = mlir::CreateMlirHandle(args.getValue().c_str());

  // Cpp backend source/header/cflags generation
  if ((option.getValue() == "source") || (option.getValue() == "header") ||
      (option.getValue() == "cflags")) {
    mlir::MlirLowerCpp(handle);
    if (option.getValue() == "source") {
      std::string source = mlir::MlirGenIgemmSource(handle);
      std::cout << source << std::endl;
    } else if (option.getValue() == "header") {
      std::string header = mlir::MlirGenIgemmHeader(handle);
      std::cout << header << std::endl;
    } else if (option.getValue() == "cflags") {
      std::string cflags = mlir::MlirGenIgemmCflags(handle);
      std::cout << cflags << std::endl;
    }
    // Bin backend binary generation
  } else if (option.getValue() == "bin") {
    char tmp[] = "";
    char *buffer = &tmp[0];
    size_t size = 0;
    mlir::MlirLowerBin(handle);
    mlir::MlirGenIgemmBin(handle, &buffer, &size);
    std::string res(buffer, size);
    std::cout << res << std::endl;
  }

  mlir::DestroyMlirHandle(handle);
}
