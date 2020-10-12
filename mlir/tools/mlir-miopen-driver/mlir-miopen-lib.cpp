#include "mlir-miopen-lib.hpp"
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
  mlir::ModuleOp getModule() { return module.get(); }
  MLIRContext context;
  mlir::OwningModuleRef module;
  std::string genTxt;
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

  auto isValid = [&argMap]() {
    std::vector<std::string> validKeys = {
        "operation",     "in_layout",   "out_layout", "fil_layout",
        "batchsize",     "in_channels", "in_h",       "in_w",
        "out_channels",  "out_h",       "out_w",      "fil_w",
        "fil_h",         "dilation_h",  "dilation_w", "conv_stride_h",
        "conv_stride_w", "padding_h",   "padding_w"};
    return std::all_of(
        validKeys.begin(), validKeys.end(),
        [&argMap](std::string &key) { return argMap.count(key) > 0; });
  };

  // Proceed only if we have a valid argMap. Otherwise leave the handle to be
  // empty
  if (isValid()) {
    auto strToLong = [&argMap](std::string argKey) {
      return std::stoul(argMap[argKey]);
    };

    auto strToInt = [&argMap](std::string argKey) {
      return std::stoi(argMap[argKey]);
    };

    SmallString<128> kernelName;
    ModuleOp module = handle->getModule();
    populateConvolutionLogic(
        argMap["arch"], strToInt("num_cu"), argMap["operation"],
        argMap["in_layout"], argMap["out_layout"], argMap["fil_layout"],
        strToLong("batchsize"), strToLong("in_channels"), strToLong("in_h"),
        strToLong("in_w"), strToLong("out_channels"), strToLong("out_h"),
        strToLong("out_w"), strToLong("fil_w"), strToLong("fil_h"),
        strToInt("dilation_h"), strToInt("dilation_w"),
        strToInt("conv_stride_h"), strToInt("conv_stride_w"),
        strToInt("padding_h"), strToInt("padding_w"), module, builder,
        kernelName, mlir::FloatType::getF32(&(handle->context)), false);

    PassManager pm(module.getContext());
    pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
    pm.run(module);
  }

  return handle;
}

extern "C" void DestroyMlirHandle(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  delete handle;
}

extern "C" const char *MlirGenIgemmSource(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  handle->genTxt = "";
  translateModuleToMIOpenCpp(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" const char *MlirGenIgemmHeader(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  handle->genTxt = "";
  translateModuleToMIOpenHeader(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" const char *MlirGenIgemmCflags(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  handle->genTxt = "";
  translateModuleToMIOpenCFlags(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}
} // namespace mlir
