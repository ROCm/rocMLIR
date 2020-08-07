#include "mlir-miopen-lib.hpp"
#include "llvm/Support/CommandLine.h"
#include <iostream>
#include <string>

using namespace llvm;
static cl::opt<std::string> args(
    "args", cl::desc("Convolution args"),
    cl::value_desc("Igemm convolution args string"),
    cl::init(
        R"( --operation conv2d_bwd_weight --fil_layout kcyx )"
        R"(--in_layout nchw --out_layout nkhw --batchsize 64 --in_channels 1024 )"
        R"(--out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 )"
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

  mlir::DestroyMlirHandle(handle);
}
