#include "MlirParse.h"
#include "mlir/Dialect/MIOpen/LowerMIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/Target/MIOpenCPP.h"

#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace mlir {
namespace {
struct MlirHandle_s {
  MlirHandle_s() {
    OpBuilder builder(&context);
    module = ModuleOp::create(builder.getUnknownLoc());
  }
  MLIRContext context;
  ModuleOp module;
};

static void strToTokens(const std::string &arguments,
                        std::map<std::string, std::string> &argMap) {
  std::istringstream iss(arguments);
  std::string token;
  std::string argKey;
  std::string argVal;
  while (std::getline(iss, token, ' ')) {
    auto pos = token.find("--");
    if (pos != std::string::npos) {
      argKey = token.substr(pos + 2, token.size());
    } else {
      argVal = token;
      if (argKey.empty() || argVal.empty())
        continue;
      argMap[argKey] = argVal;
    }
  }
}
} // namespace

typedef void *MlirHandle;

extern "C" MlirHandle CreateMlirHandle(const char *arguments) {
  registerDialect<miopen::MIOpenDialect>();
  registerDialect<StandardOpsDialect>();

  MlirHandle_s *handle = new MlirHandle_s();
  OpBuilder builder(&(handle->context));

  std::map<std::string, std::string> argMap;
  strToTokens(arguments, argMap);

  auto strToLong = [&argMap](std::string argKey) {
    return std::stoul(argMap[argKey]);
  };

  SmallString<128> kernelName;
  populateConvolutionLogic(
      argMap["operation"], argMap["in_layout"], argMap["out_layout"],
      argMap["fil_layout"], strToLong("batchsize"), strToLong("in_channels"),
      strToLong("in_h"), strToLong("in_w"), strToLong("out_channels"),
      strToLong("out_h"), strToLong("out_w"), strToLong("fil_w"),
      strToLong("fil_h"), strToLong("dilation_h"), strToLong("dilation_w"),
      strToLong("conv_stride_h"), strToLong("conv_stride_w"),
      strToLong("padding_h"), strToLong("padding_w"), handle->module, builder,
      kernelName);

  PassManager pm(handle->module.getContext());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
  pm.run(handle->module);

  return handle;
}

extern "C" void DestroyMlirHandle(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  delete handle;
}

extern "C" const char *MlirGenIgemmSource(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  auto sourceCode = translateModuleToMIOpenCpp(handle->module);
  return sourceCode->data();
  ;
}

extern "C" const char *MlirGenIgemmHeader(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  auto headerCode = translateModuleToMIOpenHeader(handle->module);
  return headerCode->data();
}

extern "C" const char *MlirGenIgemmCflags(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  auto cflagsTxt = translateModuleToMIOpenCFlags(handle->module);
  return cflagsTxt->data();
}
} // namespace mlir

// int main(){
//  std::string mimic = R"( --operation conv2d_bwd_weight --fil_layout kcyx )"
//  R"(--in_layout nchw --out_layout nkhw --batchsize 64 --in_channels 1024 )"
//  R"(--out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 )"
//  R"(--fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 )"
//  R"(--conv_stride_w 1 --padding_h 0 --padding_w 0)";
//  mlir::MlirHandle handle =
//    mlir::CreateMlirHandle(mimic.c_str());
//  std::string source = mlir::MlirGenIgemmSource(handle);
//  std::string header = mlir::MlirGenIgemmHeader(handle);
//  std::string cflags = mlir::MlirGenIgemmCflags(handle);
//  mlir::DestroyMlirHandle(handle);
//}
