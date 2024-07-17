//===- IREEGen.cpp - Use IREE for code gen --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass uses IREE for code gen.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

using namespace mlir;
using namespace mlir::migraphx;

#define LINE_NUM STRINGIFY_HELPER(__LINE__)
#define STRINGIFY_HELPER(x) STRINGIFY(x)
#define STRINGIFY(x) #x

// #define CHECK_ASSERT
#define CHECK(conditionMacro, msgMacro)                                        \
  [](bool condition, bool debug, StringRef msg) -> bool {                      \
    if (debug && condition)                                                    \
      llvm::errs() << "Error[iree-gen:" LINE_NUM "]: " << msg << "\n";         \
    return condition;                                                          \
  }(conditionMacro, debug, msgMacro)

/// Search for the IREE_BIN path environment variable.
static StringRef getIREEPathVar() {
  if (const char *var = std::getenv("IREE_BIN"))
    return var;
  return "";
}

/// Get the IREE DEBUG environment variable.
static bool getIREEDebug() {
  if (const char *var = std::getenv("IREE_DEBUG"))
    return std::atoi(var) != 0;
  return false;
}

/// Get the IREE_DISABLE environment variable.
static bool getIREEDisable() {
  if (const char *var = std::getenv("IREE_DISABLE"))
    return std::atoi(var) != 0;
  return false;
}

/// Search for the `iree-export` tool.
static std::string getIREEExport(StringRef pathHint) {
  SmallString<128> path;
  auto getPath = [&path](StringRef pathHint) -> StringRef {
    if (pathHint.empty())
      return "";
    path.assign(pathHint);
    llvm::sys::path::append(path, "iree-export");
    if (llvm::sys::fs::exists(path) && llvm::sys::fs::can_execute(path))
      return path.str();
    return "";
  };
  if (StringRef path = getPath(pathHint); !path.empty())
    return path.str();
  if (StringRef path = getPath(getIREEPathVar()); !path.empty())
    return path.str();
  return "";
}

using TmpFile = std::pair<llvm::SmallString<128>, llvm::FileRemover>;
/// Create a temporary file.
std::optional<TmpFile> static createTemp(StringRef name, StringRef suffix) {
  llvm::SmallString<128> filename;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile(name, suffix, filename);
  if (ec)
    return std::nullopt;
  return TmpFile(filename, llvm::FileRemover(filename.c_str()));
}

/// Returns the JSON value as an object or `nullptr`.
static llvm::json::Object *getAsObject(llvm::json::Value *value) {
  if (!value)
    return nullptr;
  return value->getAsObject();
}

/// Returns the JSON value as an array or `nullptr`.
static llvm::json::Array *getAsArray(llvm::json::Value *value) {
  if (!value)
    return nullptr;
  return value->getAsArray();
}

/// Returns the JSON value as a string or `""`.
static StringRef getAsString(llvm::json::Value *value) {
  if (!value)
    return "";
  auto str = value->getAsString();
  return str ? *str : "";
}

/// Gets one or more numbers from a JSON value.
static LogicalResult getInts(llvm::json::Value *value,
                             SmallVectorImpl<int64_t> &values) {
  if (!value)
    return failure();
  std::optional<int64_t> iv = value->getAsInteger();
  if (iv) {
    values.push_back(*iv);
    return success();
  }
  auto array = getAsArray(value);
  if (!array)
    return failure();
  for (auto value : *array) {
    std::optional<int64_t> iv = value.getAsInteger();
    if (iv)
      values.push_back(*iv);
    else
      return failure();
  }
  return success();
}

namespace mlir {
namespace migraphx {
#define GEN_PASS_DEF_MIGRAPHXIREEGEN
#include "mlir/Dialect/MIGraphX/Passes.h.inc"
} // namespace migraphx
} // namespace mlir

namespace {
struct MIGraphXIREEGen
    : public migraphx::impl::MIGraphXIREEGenBase<MIGraphXIREEGen> {
public:
  using Base::Base;

  /// Generate the `iree.export` attribute. This function invokes the
  /// `iree-export` tool and parses the generated JSON.
  Attribute getAttr(ModuleOp module, Builder &builder, StringRef pathHint,
                    StringRef arch, StringRef features, bool debug);

  void runOnOperation() override;
};
} // namespace

Attribute MIGraphXIREEGen::getAttr(ModuleOp module, Builder &builder,
                                   StringRef pathHint, StringRef arch,
                                   StringRef features, bool debug) {
  // Search for the `iree-export` tool.
  std::string ireeExport = getIREEExport(pathHint);
  if (CHECK(ireeExport.empty(), "couldn't find `iree-export`"))
    return nullptr;

  // Create temporary files to store the input and output of `iree-export`.
  std::optional<TmpFile> moduleMLIR = createTemp("tosa-moduleMLIR", "mlir");
  if (CHECK(!moduleMLIR, "couldn't create MLIR tmp file"))
    return nullptr;
  std::optional<TmpFile> ireeJSON = createTemp("iree-bin", "json");
  if (CHECK(!ireeJSON, "couldn't create JSON tmp file"))
    return nullptr;

  // Dump the MLIR module to a temp file.
  {
    std::error_code ec;
    llvm::raw_fd_ostream stream(moduleMLIR->first.str(), ec);
    if (CHECK(static_cast<bool>(ec), "couldn't open the MLIR tmp file"))
      return nullptr;
    module.print(stream);
    if (CHECK(stream.has_error(), "couldn't dump the MLIR tmp file"))
      return nullptr;
    stream.flush();
  }

  // Redirects.
  std::optional<StringRef> redirects[] = {
      std::nullopt,
      std::nullopt,
      std::nullopt,
  };

  std::string message;
  // Invoke the tool.
  SmallVector<StringRef, 12> ireeArgs({StringRef("iree-export"),
                                       StringRef("-a"), arch, StringRef("-f"),
                                       features, moduleMLIR->first.str(),
                                       StringRef("-o"), ireeJSON->first.str()});
  int status = llvm::sys::ExecuteAndWait(ireeExport, ireeArgs,
                                         /*Env=*/std::nullopt,
                                         /*Redirects=*/redirects,
                                         /*SecondsToWait=*/60,
                                         /*MemoryLimit=*/0,
                                         /*ErrMsg=*/&message);
  if (CHECK(status != 0, "`iree-export` invocation failed"))
    return nullptr;

  // Read the JSON file back.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> jsonFile =
      llvm::MemoryBuffer::getFile(ireeJSON->first);
  if (CHECK((!jsonFile || !jsonFile->get()), "failed reading the JSON output"))
    return nullptr;
  StringRef buff = jsonFile->get()->getBuffer();
  // Parse the JSON.
  llvm::Expected<llvm::json::Value> jsonOrErr = llvm::json::parse(buff);
  if (!jsonOrErr) {
    llvm::consumeError(jsonOrErr.takeError());
    CHECK(true, "failed parsing the JSON output");
    return nullptr;
  }
  // Retrieve the JSON data.
  llvm::json::Object *topDict = getAsObject(&jsonOrErr.get());
  if (CHECK(!topDict, "invalid JSON"))
    return nullptr;
  // Check if the tool failed with a message.
  if (llvm::json::Value *error = topDict->get("error")) {
    StringRef errMsg = "IREE invocation failed";
    if (auto err = error->getAsString())
      errMsg = *err;
    CHECK(true, errMsg);
    return nullptr;
  }
  // Get the binary.
  llvm::json::Array *binaries = getAsArray(topDict->get("binaries"));
  if (CHECK(!binaries, "invalid JSON, expected `binaries` field") ||
      CHECK(binaries->size() > 1, "more than one binary was found") ||
      CHECK(binaries->empty(), "expected at least one binary"))
    return nullptr;
  llvm::json::Object *binaryDict = getAsObject(&(*binaries)[0]);
  if (CHECK(!binaryDict, "expected a dict for each binary"))
    return nullptr;
  // Get the ELF.
  StringRef elf = getAsString(binaryDict->get("$elf"));
  if (CHECK(elf.empty(), "expected a valid ELF"))
    return nullptr;
  // Get the kernel name.
  llvm::json::Array *kernelTable = getAsArray(binaryDict->get("$kernels"));
  if (CHECK(!kernelTable || kernelTable->size() != 1,
            "expected a valid kernel table"))
    return nullptr;
  StringRef kernelName = getAsString(&(*kernelTable)[0]);
  if (CHECK(kernelName.empty(), "expected a valid kernel name"))
    return nullptr;
  // Get the kernel properties.
  llvm::json::Object *kernelProperties =
      getAsObject(&(*binaryDict)[kernelName]);
  if (CHECK(!kernelProperties, "invalid kernel properties"))
    return nullptr;
  SmallVector<int64_t> tmp;
  // Get the workgroup sizes.
  if (CHECK(failed(getInts(&(*kernelProperties)["workgroup_sizes"], tmp)),
            "expected a valid `workgroup_sizes` field") ||
      CHECK(tmp.size() != 3, "expected a valid `workgroup_sizes` field"))
    return nullptr;
  DenseI64ArrayAttr workgroupSizes = builder.getDenseI64ArrayAttr(tmp);
  tmp.clear();
  // Get the workgroup count.
  if (CHECK(failed(getInts(&(*kernelProperties)["workgroup_count"], tmp)),
            "expected a valid `workgroup_count` field") ||
      CHECK(tmp.size() != 3, "expected a valid `workgroup_count` field"))
    return nullptr;
  DenseI64ArrayAttr workgroupCount = builder.getDenseI64ArrayAttr(tmp);
  tmp.clear();
  // Get the workgroup shared mem size.
  if (CHECK(failed(getInts(&(*kernelProperties)["workgroup_memory"], tmp)),
            "expected a valid `workgroup_memory` field") ||
      CHECK(tmp.size() != 1, "expected a valid `workgroup_memory` field"))
    return nullptr;
  IntegerAttr workgroupMem = builder.getI64IntegerAttr(tmp[0]);
  tmp.clear();
  // Build the properties dict.
  NamedAttrList attrs;
  attrs.append("kernel", builder.getStringAttr(kernelName));
  attrs.append("workgroup_sizes", workgroupSizes);
  attrs.append("workgroup_count", workgroupCount);
  attrs.append("workgroup_memory", workgroupMem);
  // Create the GPU object with the IREE binary.
  return builder.getAttr<gpu::ObjectAttr>(
      builder.getAttr<ROCDL::ROCDLTargetAttr>(3, "amdgcn-amd-amdhsa", arch),
      gpu::CompilationTarget::Binary, builder.getStringAttr(llvm::fromHex(elf)),
      builder.getDictionaryAttr(attrs), nullptr);
}

void MIGraphXIREEGen::runOnOperation() {
  // Don't run if `IREE_DISABLE` was set.
  if (getIREEDisable())
    return;
  ModuleOp op = getOperation();
  Builder builder(op.getContext());
  bool debug = getIREEDebug();
  if (CHECK(arch.empty(), "expected a valid target architecture"))
    return;
  // Get the `iree-export` attribute.
  Attribute attr = getAttr(op, builder, "", arch, features, debug);
  // If `getAttr` failed, create a generic error for `iree-export`.
  if (!attr)
    attr = builder.getStringAttr("[iree-error]: invocation failed");
  // Set the attribute.
  op->setAttr("iree.export", attr);
}
