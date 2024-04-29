/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef SRC_HIPBIN_AMD_H_
#define SRC_HIPBIN_AMD_H_

#include "hipBin_base.h"
#include "hipBin_util.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <cassert>


// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

// Known Features
 std::unordered_set
 <std::string> knownFeatures =  { "sramecc-" , "sramecc+",
                                  "xnack-", "xnack+" };

class HipBinAmd : public HipBinBase {
 private:
  HipBinUtil* hipBinUtilPtr_;
  string hipClangPath_ = "";
  string roccmPathEnv_, hipRocclrPathEnv_, hsaPathEnv_;
  PlatformInfo platformInfoAMD_;
  string hipCFlags_, hipCXXFlags_, hipLdFlags_;
  void constructRocclrHomePath();
  void constructHsaPath();

 public:
  HipBinAmd();
  virtual ~HipBinAmd() = default;
  virtual bool detectPlatform();
  virtual void constructCompilerPath();
  virtual const string& getCompilerPath() const;
  virtual const PlatformInfo& getPlatformInfo() const;
  virtual string getCppConfig();
  virtual void printFull();
  virtual void printCompilerInfo() const;
  virtual string getCompilerVersion();
  virtual void checkHipconfig();
  virtual string getDeviceLibPath() const;
  virtual string getHipLibPath() const;
  virtual string getHipCC() const;
  virtual string getHipInclude() const;
  virtual void initializeHipCXXFlags();
  virtual void initializeHipCFlags();
  virtual void initializeHipLdFlags();
  virtual const string& getHipCXXFlags() const;
  virtual const string& getHipCFlags() const;
  virtual const string& getHipLdFlags() const;
  virtual void executeHipCCCmd(vector<string> argv);
  // non virtual functions
  const string& getHsaPath() const;
  const string& getRocclrHomePath() const;
  const bool isWindows() const;
};

HipBinAmd::HipBinAmd() {
  PlatformInfo platformInfo;
  platformInfo.os = getOSInfo();
  platformInfo.platform = amd;
  platformInfo.runtime = rocclr;
  platformInfo.compiler = clang;
  platformInfoAMD_ = platformInfo;

  // Base class calls readEnvVariables, but we need to make sure we set rocm_path and hip_path, so that we can set hipClangPath
  constructHipPath();
  constructRoccmPath();
  constructCompilerPath();
  readHipVersion();
}

// returns the Rocclr Home path
void HipBinAmd::constructRocclrHomePath() {
  fs::path full_path(fs::current_path());
  fs::path hipvars_dir = full_path;
  fs::path bitcode = hipvars_dir;
  string rocclrHomePath = getEnvVariables().hipRocclrPathEnv_;
  if (rocclrHomePath.empty()) {
    bitcode /= "../lib/bitcode";
    if (!fs::exists(bitcode)) {
      rocclrHomePath = getHipPath();
    } else {
      hipvars_dir /= "..";
      rocclrHomePath = hipvars_dir.string();
    }
  }
  hipRocclrPathEnv_ = rocclrHomePath;
}


// construct hsa Path
void HipBinAmd::constructHsaPath() {
  fs::path hsaPathfs;
  string hsaPath = getEnvVariables().hsaPathEnv_;
  if (hsaPath.empty()) {
    hsaPath = getRoccmPath();
    hsaPathfs = hsaPath;
    hsaPathfs /= "hsa";
    hsaPath = hsaPathfs.string();
    hsaPathEnv_ = hsaPath;
  } else {
    hsaPathEnv_ = hsaPath;
  }
}

// returns the Rocclr Home path
const string& HipBinAmd::getRocclrHomePath() const {
  return hipRocclrPathEnv_;
}

// returns hsa Path
const string& HipBinAmd::getHsaPath() const {
  // return variables_.hsaPathEnv_;
  return hsaPathEnv_;
}


const string& HipBinAmd::getHipCFlags() const {
  return hipCFlags_;
}


const string& HipBinAmd::getHipLdFlags() const {
  return hipLdFlags_;
}


void HipBinAmd::initializeHipLdFlags() {
  string hipLdFlags;
  const string& hipClangPath = getCompilerPath();
  // If $HIPCC clang++ is not compiled, use clang instead
  string hipCC = "\"" + hipClangPath + "/clang++";
  if (!fs::exists(hipCC)) {
    hipLdFlags = "--driver-mode=g++";
  }
  hipLdFlags_ = hipLdFlags;
}

void HipBinAmd::initializeHipCFlags() {
}

const string& HipBinAmd::getHipCXXFlags() const {
  return hipCXXFlags_;
}


string HipBinAmd::getHipInclude() const {
  const string& rocclrHomePath = getRocclrHomePath();
  fs::path hipIncludefs = rocclrHomePath;
  hipIncludefs /= "include";
  if (hipIncludefs.string().empty()) {
    const string& hipPath = getHipPath();
    hipIncludefs = hipPath;
    hipIncludefs /= "include";
  }
  string hipInclude = hipIncludefs.string();
  return hipInclude;
}


void HipBinAmd::initializeHipCXXFlags() {
  string hipCXXFlags;
  const OsType& os = getOSInfo();
  const EnvVariables& var = getEnvVariables();
  // Allow __fp16 as function parameter and return type.
  if (var.hipClangHccCompactModeEnv_.compare("1") == 0) {
    hipCXXFlags +=
    " -Xclang -fallow-half-arguments-and-returns -D__HIP_HCC_COMPAT_MODE__=1";
  }

  hipCXXFlags_ = hipCXXFlags;
}

// populates clang path.
void HipBinAmd::constructCompilerPath() {
  string compilerPath;
  const EnvVariables& envVariables = getEnvVariables();
  if (envVariables.hipClangPathEnv_.empty()) {
    fs::path hipClangPath;
    if (isWindows()) {
      compilerPath = getHipPath();
      hipClangPath = compilerPath;
      hipClangPath /= "bin";
    } else {
      compilerPath = getRoccmPath();
      hipClangPath = compilerPath;
      hipClangPath /= "llvm/bin";
    }

    compilerPath = hipClangPath.string();
  } else {
    compilerPath = envVariables.hipClangPathEnv_;
  }
  hipClangPath_ = compilerPath;
}

// returns clang path.
const string& HipBinAmd::getCompilerPath() const {
  return hipClangPath_;
}

void HipBinAmd::printCompilerInfo() const {
  const string& hipClangPath = getCompilerPath();
  const string& hipPath = getHipPath();
  if (isWindows()) {
    string cmd = hipClangPath + "/clang++ --version";
    system(cmd.c_str());  // hipclang version
    cout << "llc-version :" << endl;
    cmd = hipClangPath + "/llc --version";
    system(cmd.c_str());  // llc version
    cout << "hip-clang-cxxflags :" << endl;
    cmd = hipPath + "/bin/hipcc  --cxxflags";
    system(cmd.c_str());  // cxx flags
    cout << endl << "hip-clang-ldflags :" << endl;
    cmd = hipPath + "/bin/hipcc --ldflags";
    system(cmd.c_str());  // ld flags
    cout << endl;
  } else {
    string cmd = hipClangPath + "/clang++ --version";
    system(cmd.c_str());  // hipclang version
    cmd = hipClangPath + "/llc --version";
    system(cmd.c_str());  // llc version
    cout << "hip-clang-cxxflags :" << endl;
    cmd = hipPath + "/bin/hipcc --cxxflags";
    system(cmd.c_str());  // cxx flags
    cout << endl << "hip-clang-ldflags :" << endl;
    cmd = hipPath + "/bin/hipcc --ldflags";
    system(cmd.c_str());  // ldflags version
    cout << endl;
  }
}

string HipBinAmd::getCompilerVersion() {
  string out, compilerVersion;
  const string& hipClangPath = getCompilerPath();
  fs::path cmdAmd = hipClangPath;
  cmdAmd /= "clang++";
  if (canRunCompiler(cmdAmd.string(), out) || canRunCompiler("amdclang++", out)) {
    regex regexp("([0-9.]+)");
    smatch m;
    if (regex_search(out, m, regexp)) {
      if (m.size() > 1) {
        // get the index =1 match, 0=whole match we ignore
        std::ssub_match sub_match = m[1];
        compilerVersion = sub_match.str();
      }
    }
  } else {
    std::cerr << "Hip Clang Compiler not found" << endl;
  }
  return compilerVersion;
}



const PlatformInfo& HipBinAmd::getPlatformInfo() const {
  return platformInfoAMD_;
}


string HipBinAmd::getCppConfig() {
  string cppConfig = " -D__HIP_PLATFORM_HCC__= -D__HIP_PLATFORM_AMD__=";

  string compilerVersion;
  compilerVersion = getCompilerVersion();

  fs::path hipPathInclude, cppConfigFs;
  const string& hipPath = getHipPath();
  hipPathInclude = hipPath;
  hipPathInclude /= "include";
  if (isWindows()) {
    cppConfig += " -I" + hipPathInclude.string();
    cppConfigFs = cppConfig;
    cppConfigFs /= "/";
  } else {
    const string& hsaPath = getHsaPath();
    cppConfig += " -I" + hipPathInclude.string() +
                 " -I" + hsaPath;
    cppConfigFs = cppConfig;
    cppConfigFs /= "include";
    cppConfig = cppConfigFs.string();
  }
  return cppConfig;
}

string HipBinAmd::getDeviceLibPath() const {
  const EnvVariables& var = getEnvVariables();
  const string& rocclrHomePath = getRocclrHomePath();
  const string& roccmPath = getRoccmPath();
  fs::path bitCodePath = rocclrHomePath;
  bitCodePath /= "lib/bitcode";
  string deviceLibPath = var.deviceLibPathEnv_;
  if (deviceLibPath.empty() && fs::exists(bitCodePath)) {
    deviceLibPath = bitCodePath.string();
  }

  if (deviceLibPath.empty()) {
    fs::path amdgcnBitcode = roccmPath;
    amdgcnBitcode /= "amdgcn/bitcode";
    if (fs::exists(amdgcnBitcode)) {
      deviceLibPath = amdgcnBitcode.string();
    } else {
      // This path is to support an older build of the device library
      // TODO(hipcc): To be removed in the future.
      fs::path lib = roccmPath;
      lib /= "lib";
      deviceLibPath = lib.string();
    }
  }
  return deviceLibPath;
}


bool HipBinAmd::detectPlatform() {
  string out;
  constructCompilerPath();
  const string& hipClangPath = getCompilerPath();
  fs::path cmdAmd = hipClangPath;
  cmdAmd /= "clang++";
  const EnvVariables& var = getEnvVariables();
  bool detected = false;
  if (var.hipPlatformEnv_.empty()) {
    if (canRunCompiler(cmdAmd.string(), out)){
      detected = true;
    }
  } else {
    if (var.hipPlatformEnv_ == "amd" ||
        var.hipPlatformEnv_ == "hcc") {
      detected = true;
      if (var.hipPlatformEnv_ == "hcc")
        std::cerr <<
        "Warning: HIP_PLATFORM=hcc is deprecated."<<
        "Please use HIP_PLATFORM=amd." << endl;
    }
  }
  return detected;
}

string HipBinAmd::getHipLibPath() const {
  string hipLibPath;
  const EnvVariables& env = getEnvVariables();
  if (!env.hipLibPathEnv_.empty()) {
    hipLibPath = env.hipLibPathEnv_;
  }
  else if (!env.hipPathEnv_.empty()) {
    fs::path p = env.hipLibPathEnv_;
    p /= "lib";
    hipLibPath = p.string();
  }
  return hipLibPath;
}

string HipBinAmd::getHipCC() const {
  string hipCC;
  const string& hipClangPath = getCompilerPath();
  fs::path compiler = hipClangPath;
  if (isWindows())
    compiler /= "clang.exe";
  else
    compiler /= "clang++";

  if (!fs::exists(compiler)) {
    fs::path compiler = hipClangPath;
    compiler /= "clang";
  }
  hipCC = compiler.string();

  if (isWindows()) // wrap hipcc (clang) command in escaped double quotes.
    hipCC = "\"" + hipCC + "\" ";
  return hipCC;
}

void HipBinAmd::checkHipconfig() {
  printFull();
  cout << endl << "Check system installation: " << endl;
  cout << "check hipconfig in PATH..." << endl;
  if (system("which hipconfig > /dev/null 2>&1") != 0) {
    std::cerr << "FAIL " << endl;
  } else {
    cout << "good" << endl;
  }
  string ldLibraryPath;
  const EnvVariables& env = getEnvVariables();
  ldLibraryPath = env.ldLibraryPathEnv_;
  const string& hsaPath = getHsaPath();
  cout << "check LD_LIBRARY_PATH (" << ldLibraryPath <<
          ") contains HSA_PATH (" << hsaPath << ")..." << endl;
  if (ldLibraryPath.find(hsaPath) == string::npos) {
    std::cerr << "FAIL" << endl;
  } else {
    cout << "good" << endl;
  }
}

void HipBinAmd::printFull() {
  const string& hipVersion = getHipVersion();
  const string& hipPath = getHipPath();
  const string& roccmPath = getRoccmPath();
  const PlatformInfo& platformInfo = getPlatformInfo();
  const string& ccpConfig = getCppConfig();
  const string& hsaPath = getHsaPath();
  const string& hipClangPath = getCompilerPath();

  cout << "HIP version: " << hipVersion << endl;
  cout << endl << "==hipconfig" << endl;
  cout << "HIP_PATH           :" << hipPath << endl;
  cout << "ROCM_PATH          :" << roccmPath << endl;
  cout << "HIP_COMPILER       :" << CompilerTypeStr(
                                    platformInfo.compiler) << endl;
  cout << "HIP_PLATFORM       :" << PlatformTypeStr(
                                    platformInfo.platform) << endl;
  cout << "HIP_RUNTIME        :" << RuntimeTypeStr(
                                    platformInfo.runtime) << endl;
  cout << "CPP_CONFIG         :" << ccpConfig << endl;

  cout << endl << "==hip-clang" << endl;
  cout << "HIP_CLANG_PATH     :" << hipClangPath << endl;
  printCompilerInfo();
  cout << endl << "== Environment Variables" << endl;
  printEnvironmentVariables();
  getSystemInfo();
  if (fs::exists("/usr/bin/lsb_release"))
    system("/usr/bin/lsb_release -a");
  cout << endl;
}

const bool HipBinAmd::isWindows() const {
    const OsType& osInfo = getOSInfo();
    return (osInfo == windows);
}

void HipBinAmd::executeHipCCCmd(vector<string> argv) {
  if (argv.size() < 2) {
    cout<< "No Arguments passed, exiting ...\n";
    exit(EXIT_SUCCESS);
  }
  const EnvVariables& var = getEnvVariables();
  int verbose = 0;
  if (!var.verboseEnv_.empty())
    verbose = stoi(var.verboseEnv_);

  // Verbose: 0x1=commands, 0x2=paths, 0x4=hipcc args
  // set if user explicitly requests -stdlib=libc++
  // (else we default to libstdc++ for better interop with g++)
  bool setStdLib = 0;
  bool default_amdgpu_target = 1;
  bool compileOnly = 0;
  bool needCXXFLAGS = 0;  // need to add CXX flags to compile step
  bool needCFLAGS = 0;    // need to add C flags to compile step
  bool needLDFLAGS = 1;   // need to add LDFLAGS to compile step.
  bool fileTypeFlag = 0;  // to see if -x flag is mentioned
  bool hasOMPTargets = 0;  // If OMP targets is mentioned
  bool hasC = 0;          // options contain a c-style file
  // options contain a cpp-style file (NVCC must force recognition as GPU file)
  bool hasCXX = 0;
  // options contain a hip-style file (HIP-Clang must pass offloading options)
  bool hasHIP = 0;
  bool printHipVersion = 0;    // print HIP version
  bool printCXXFlags = 0;      // print HIPCXXFLAGS
  bool printLDFlags = 0;       // print HIPLDFLAGS
  bool runCmd = 1;
  bool buildDeps = 0;
  string hsacoVersion;
  bool funcSupp = 1;      // enable function support
  bool rdc = 0;           // whether -fgpu-rdc is on

  string prevArg;  //  previous argument
  // TODO(hipcc): convert toolArgs to an array rather than a string
  string toolArgs;   // arguments to pass to the clang or nvcc tool
  string optArg;     // -O args
  vector<string> options, inputs;

  // TODO(hipcc): hipcc uses --amdgpu-target for historical reasons.
  // It should be replaced
  // by clang option --offload-arch.
  vector<string> targetOpts = {"--offload-arch=", "--amdgpu-target="};
  string targetsStr;
  // file followed by -o should not contibute in picking compiler flags
  bool skipOutputFile = false;

  const OsType& os = getOSInfo();
  string hip_compile_cxx_as_hip;
  if (var.hipCompileCxxAsHipEnv_.empty()) {
    hip_compile_cxx_as_hip = "1";
  } else {
    hip_compile_cxx_as_hip = var.hipCompileCxxAsHipEnv_;
  }

  string HIPLDARCHFLAGS;
  string HIPCXXFLAGS, HIPCFLAGS, HIPLDFLAGS;

  // ARGV Processing Loop
  // TODO(hipcc): create a proper Options Processing function/routine
  for (unsigned int argcount = 1; argcount < argv.size(); argcount++) {
    // Save $arg, it can get changed in the loop.
    string arg = argv.at(argcount);
    // TODO(hipcc): figure out why this space removal is wanted.
    // TODO(hipcc): If someone has gone to the effort of
    // quoting the spaces to the shell
    // TODO(hipcc): why are we removing it here?
    regex toRemove("\\s+");
    // Remove whitespace
    string trimarg = hipBinUtilPtr_->replaceRegex(arg, toRemove, "");
    bool swallowArg = false;
    bool escapeArg = true;
    if (arg == "-c" || arg == "--genco" || arg == "-E") {
      compileOnly = true;
      needLDFLAGS  = false;
    }

    if (skipOutputFile) {
      // TODO(hipcc): handle filename with shell metacharacters
      toolArgs += " \"" + arg +"\"";
      prevArg = arg;
      skipOutputFile = 0;
      continue;
    }

    if (arg == "-o") {
      needLDFLAGS = 1;
      skipOutputFile = 1;
    }

    if ((trimarg == "-stdlib=libc++") && (setStdLib == 0)) {
      HIPCXXFLAGS += " -stdlib=libc++";
      setStdLib = 1;
    }

    // Process --rocm-path option
    const string& rocmPathOption = "--rocm-path=";
    if (arg.compare(0,rocmPathOption.length(),rocmPathOption) == 0)
    	rocm_pathOption_ = arg.substr(rocmPathOption.length());
    // Process --hip-path option
    const string& hipPathOption = "--hip-path=";
    if (arg.compare(0,hipPathOption.length(),hipPathOption) == 0)
    	hip_pathOption_ = arg.substr(hipPathOption.length());

    // Check target selection option: --offload-arch= and --amdgpu-target=...
    for (unsigned int i = 0; i <targetOpts.size(); i++) {
      string targetOpt = targetOpts.at(i);
      // match arg with the starting of targetOpt
      string pattern = "^" + targetOpt + ".*";
      if (hipBinUtilPtr_->stringRegexMatch(arg, pattern))  {
        if (targetOpt == "--amdgpu-target=") {
          std::cerr << "Warning: The --amdgpu-target option has been deprecated and will be removed in the future."
                    << "  Use --offload-arch instead.\n";
        }
        // If targets string is not empty,
        // add a comma before adding new target option value.
        targetsStr.size() >0 ? targetsStr += ",": targetsStr += "";
        targetsStr += arg.substr(targetOpt.size());  // argument of targetOpts
        default_amdgpu_target = 0;
        // Collect the GPU arch options and pass them to clang later.
        swallowArg = 1;
      }
    }  // end of for targetOpts for loop

    if (hipBinUtilPtr_->substringPresent(arg, "--genco")) {
      arg = "--cuda-device-only";
    }

    if (trimarg == "--version") {
      printHipVersion = 1;
    }
    if (trimarg == "--short-version") {
      printHipVersion = 1;
      runCmd = 0;
    }
    if (trimarg == "--cxxflags") {
      printCXXFlags = 1;
      runCmd = 0;
    }
    if (trimarg == "--ldflags") {
      printLDFlags = 1;
      runCmd = 0;
    }
    if (trimarg == "-M") {
      compileOnly = 1;
      buildDeps = 1;
    }
    if ((trimarg == "-use-staticlib")) {
      std::cerr << "Warning: The -use-staticlib option has been deprecated and is no longer needed.\n";
      swallowArg = true;
    }
    if ((trimarg == "-use-sharedlib")) {
      std::cerr << "Warning: The -use-sharedlib option has been deprecated and is no longer needed.\n";
      swallowArg = true;
    }
    if (hipBinUtilPtr_->stringRegexMatch(arg, "^-O.*")) {
      optArg = arg;
    }
    if (hipBinUtilPtr_->substringPresent(
        arg, "--amdhsa-code-object-version=")) {
      std::cerr << "Warning: The --amdhsa-code-object-version option has been deprecated and will be removed in the future."
                << "  Use -mllvm -mcode-object-version instead.\n";
      arg = hipBinUtilPtr_->replaceStr(
            arg, "--amdhsa-code-object-version=", "");
      hsacoVersion = arg;
      swallowArg = 1;
    }

    if (arg == "-x") {
        fileTypeFlag = 1;
    } else if ((arg == "c" && prevArg == "-x") || (arg == "-xc")) {
        fileTypeFlag = 1;
        hasC = 1;
        hasCXX = 0;
        hasHIP = 0;
    } else if ((arg == "c++" && prevArg == "-x") || (arg == "-xc++")) {
        fileTypeFlag = 1;
        hasC = 0;
        hasCXX = 1;
        hasHIP = 0;
    } else if ((arg == "hip" && prevArg == "-x") || (arg == "-xhip")) {
        fileTypeFlag = 1;
        hasC = 0;
        hasCXX = 0;
        hasHIP = 1;
    } else if (hipBinUtilPtr_->substringPresent(arg, "-fopenmp-targets=")) {
        hasOMPTargets = 1;
      // options start with -
    } else if (hipBinUtilPtr_->stringRegexMatch(arg, "^-.*")) {
        if  (arg == "-fgpu-rdc") {
          rdc = 1;
        } else if (arg == "-fno-gpu-rdc") {
          rdc = 0;
        }
        //# Process HIPCC options here:
        if (hipBinUtilPtr_->stringRegexMatch(arg, "^--hipcc.*")) {
          swallowArg = 1;
          if (arg == "--hipcc-func-supp") {
            std::cerr << "Warning: The --hipcc-func-supp option has been deprecated and will be removed in the future.\n";
            funcSupp = 1;
          } else if (arg == "--hipcc-no-func-supp") {
            std::cerr << "Warning: The --hipcc-no-func-supp option has been deprecated and will be removed in the future.\n";
            funcSupp = 0;
          }
        } else {
          options.push_back(arg);
        }
      // print "O: <$arg>\n";
    } else if (prevArg != "-o") {
    // input files and libraries
    // Skip guessing if `-x {c|c++|hip}` is already specified.
    // Add proper file extension before each file type
    // File Extension                 -> Flag
    // .c                             -> -x c
    // .cpp/.cxx/.cc/.cu/.cuh/.hip    -> -x hip

    if (fileTypeFlag == 0) {
      if (hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.c$")) {
        hasC = 1;
        needCFLAGS = 1;
        toolArgs += " -x c";
      } else if ((hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.cpp$")) ||
                 (hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.cxx$")) ||
                 (hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.cc$")) ||
                 (hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.C$"))) {
        needCXXFLAGS = 1;
        if (hip_compile_cxx_as_hip == "0" || hasOMPTargets == 1) {
          hasCXX = 1;
        } else {
          hasHIP = 1;
          toolArgs += " -x hip";
        }
      } else if (((hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.cu$") ||
                   hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.cuh$")) &&
                   hip_compile_cxx_as_hip != "0") ||
                  (hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.hip$"))) {
        needCXXFLAGS = 1;
        hasHIP = 1;
        toolArgs += " -x hip";
      }
    }
    if (hasC) {
      needCFLAGS = 1;
    } else if (hasCXX || hasHIP) {
      needCXXFLAGS = 1;
    }
    if (isWindows())
      arg = "\"" + arg + "\"";

    inputs.push_back(arg);
    // print "I: <$arg>\n";
    }
    // Produce a version of $arg where characters significant to the shell are
    // quoted. One could quote everything of course but don't bother for
    // common characters such as alphanumerics.
    // Do the quoting here because sometimes the $arg is changed in the loop
    // Important to have all of '-Xlinker' in the set of unquoted characters.
    // Windows needs different quoting, ignore for now
    if (!isWindows() && escapeArg) {
      regex reg("[^-a-zA-Z0-9_=+,.\\/]");
      arg = regex_replace(arg, reg, "\\$&");
    }
    if (!swallowArg)
      toolArgs += " " + arg;
    prevArg = arg;
  }  // end of ARGV Processing Loop

  // now construct Paths ...
  constructHipPath();           // constructs HIP Path
  constructRoccmPath();         // constructs Roccm Path
  readHipVersion();             // stores the hip version
  constructCompilerPath();
  constructRocclrHomePath();
  constructHsaPath();

  initializeHipCXXFlags();
  initializeHipCFlags();
  initializeHipLdFlags();
  HIPCFLAGS = getHipCFlags();
  HIPCXXFLAGS = getHipCXXFlags();
  HIPLDFLAGS = getHipLdFlags();

  string hipLibPath;
  string hipIncludePath, deviceLibPath;
  hipLibPath = getHipLibPath();
  const string& roccmPath = getRoccmPath();
  const string& hipPath = getHipPath();
  const PlatformInfo& platformInfo = getPlatformInfo();
  const string& rocclrHomePath = getRocclrHomePath();
  const string& hipClangPath = getCompilerPath();
  hipIncludePath = getHipInclude();
  deviceLibPath = getDeviceLibPath();
  const string& hipVersion = getHipVersion();
  if (verbose & 0x2) {
    cout << "HIP_PATH=" << hipPath << endl;
    cout << "HIP_PLATFORM=" <<  PlatformTypeStr(platformInfo.platform) <<endl;
    cout << "HIP_COMPILER=" << CompilerTypeStr(platformInfo.compiler) <<endl;
    cout << "HIP_RUNTIME=" << RuntimeTypeStr(platformInfo.runtime) <<endl;
    cout << "ROCM_PATH=" << roccmPath << endl;
    cout << "HIP_ROCCLR_HOME="<< rocclrHomePath << endl;
    cout << "HIP_CLANG_PATH=" << hipClangPath <<endl;
    cout << "HIP_INCLUDE_PATH="<< hipIncludePath  <<endl;
    cout << "HIP_LIB_PATH="<< hipLibPath <<endl;
    cout << "DEVICE_LIB_PATH="<< deviceLibPath <<endl;
  }

  if (verbose & 0x4) {
    cout <<  "hipcc-args: ";
    for (unsigned int i = 1; i< argv.size(); i++) {
      cout <<  argv.at(i) << " ";
    }
    cout << endl;
  }

 // No AMDGPU target specified at commandline. So look for HCC_AMDGPU_TARGET
  if (default_amdgpu_target == 1) {
    if (!var.hccAmdGpuTargetEnv_.empty()) {
      targetsStr = var.hccAmdGpuTargetEnv_;
    } else if (os != windows) {
      // Else try using rocm_agent_enumerator
      string ROCM_AGENT_ENUM;
      ROCM_AGENT_ENUM = roccmPath + "/bin/rocm_agent_enumerator";
      targetsStr = ROCM_AGENT_ENUM +" -t GPU";
      SystemCmdOut sysOut = hipBinUtilPtr_->exec(targetsStr.c_str());
      regex toReplace("\n+");
      targetsStr = hipBinUtilPtr_->replaceRegex(sysOut.out, toReplace, ",");
    }
    default_amdgpu_target = 0;
  }
  // Parse the targets collected in targetStr
  // and set corresponding compiler options.
  vector<string> targets = hipBinUtilPtr_->splitStr(targetsStr, ',');
  string GPU_ARCH_OPT = " --offload-arch=";

  for (auto &val : targets) {
    // Ignore 'gfx000' target reported by rocm_agent_enumerator.
    if (val != "gfx000") {
      vector<string> procAndFeatures = hipBinUtilPtr_->splitStr(val, ':');
      size_t len = procAndFeatures.size();
      // proc and features
      assertm(procAndFeatures.size() >= 1, "Pass the correct device/feature");
      for (size_t i = 1; i < len; i++) {
          // fixme: currently it checks only for validity of the feature string.
          // does not check if the device supports the feature or not
          // e.g. vega10 does not support sramecc
          if (knownFeatures.find(procAndFeatures.at(i)) == knownFeatures.end()) {
            std::cerr <<  "Warning: The Feature: "<< procAndFeatures.at(i) <<
                     " is unknown. Correct compilation is not guaranteed.\n";
          }
      }
      string GPU_ARCH_ARG;
      GPU_ARCH_ARG = GPU_ARCH_OPT + val;

      HIPLDARCHFLAGS += GPU_ARCH_ARG;
      if (hasHIP) {
        HIPCXXFLAGS += GPU_ARCH_ARG;
      }
    }  // end of val != "gfx000"
  }  // end of targets for loop
  string HCC_EXTRA_LIBRARIES;
  if (hsacoVersion.size() > 0) {
    if (compileOnly == 0) {
      HIPLDFLAGS += " -mcode-object-version=" + hsacoVersion;
    } else {
      HIPCXXFLAGS += " -mcode-object-version=" + hsacoVersion;
    }
  }

  // rocm_agent_enumerator failed! Throw an error and die if linking is required
  if (default_amdgpu_target == 1 && compileOnly == 0) {
    // TODO(agunashe) exit from function
    std::cerr <<  "No valid AMD GPU target was either specified or found."
              << "Please specify a valid target using --offload-arch=<target>.\n";
  }
  HCC_EXTRA_LIBRARIES ="\n";  // TODO(agunashe) write to env

  if (buildDeps) {
    HIPCXXFLAGS += " --cuda-host-only";
  }

  // hipcc currrently requires separate compilation of source files,
  // ie it is not possible to pass
  // CPP files combined with .O files
  // Reason is that NVCC uses the file extension to determine
  // whether to compile in CUDA mode or
  // pass-through CPP mode.
  // Set default optimization level to -O3 for hip-clang.
  if (optArg.empty()) {
    HIPCXXFLAGS += " -O3";
    HIPCFLAGS += " -O3";
    HIPLDFLAGS += " -O3";
  }

  if (!funcSupp && optArg != "-O0" && hasHIP) {
    HIPCXXFLAGS +=
    " -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false";
    if (needLDFLAGS && !needCXXFLAGS) {
      HIPLDFLAGS +=
      " -mllvm -amdgpu-early-inline-all=true"
      " -mllvm -amdgpu-function-calls=false";
    }
  }

  if (hasHIP) {
    fs::path bitcodeFs = roccmPath;
    bitcodeFs /= "amdgcn/bitcode";
    if (deviceLibPath != bitcodeFs.string()) {
      string hip_device_lib_str = " --hip-device-lib-path=\""
                                  + deviceLibPath + "\"";
      HIPCXXFLAGS += hip_device_lib_str;
    }
  }

  // to avoid using dk linker or MSVC linker
  if (isWindows()) {
    HIPLDFLAGS += " -fuse-ld=lld --ld-path=\"" + hipClangPath + "/lld-link.exe\"";
  }

  if (!compileOnly) {
    string hip_path = getHipLibPath();
    if (!hip_path.empty()) {
      HIPLDFLAGS += " -L" + hip_path;
    }
    HIPLDFLAGS += " --hip-link";
    if (rdc) {
      HIPLDFLAGS += HIPLDARCHFLAGS;
    }
    if (!windows) {
      HIPLDFLAGS += "  --rtlib=compiler-rt -unwindlib=libgcc";
    }
  }

  if (!var.hipccCompileFlagsAppendEnv_.empty()) {
    HIPCXXFLAGS += " " + var.hipccCompileFlagsAppendEnv_ + " ";
    HIPCFLAGS += " " + var.hipccCompileFlagsAppendEnv_ + " ";
  }
  if (!var.hipccLinkFlagsAppendEnv_.empty()) {
    HIPLDFLAGS += " " + var.hipccLinkFlagsAppendEnv_ + " ";
  }
  // TODO(hipcc): convert CMD to an array rather than a string
  string compiler;
  compiler = getHipCC();
  string CMD = compiler;
  if (needCFLAGS) {
    CMD += " " + HIPCFLAGS;
  }

  if (needCXXFLAGS) {
    CMD += " " + HIPCXXFLAGS;
  }

  if (needLDFLAGS && !compileOnly) {
    CMD += " " + HIPLDFLAGS;
  }

  CMD += " " + toolArgs;
  if (verbose & 0x1) {
    cout << "hipcc-cmd: " <<  CMD << "\n";
  }

  if (printHipVersion) {
    if (runCmd) {
      cout <<  "HIP version: ";
    }
    cout << hipVersion << endl;
  }
  if (printCXXFlags) {
    cout << HIPCXXFLAGS;
  }
  if (printLDFlags) {
    cout << HIPLDFLAGS;
  }
  if (runCmd) {
    if (isWindows())
      CMD = "\"" + CMD + "\"";

    SystemCmdOut sysOut;
    sysOut = hipBinUtilPtr_->exec(CMD.c_str(), true);
    string cmdOut = sysOut.out;
    int CMD_EXIT_CODE = sysOut.exitCode;
    if (CMD_EXIT_CODE !=0) {
       std::cerr <<  "failed to execute:"  << CMD << std::endl;
    }
    exit(CMD_EXIT_CODE);
  }  // end of runCmd section
}   // end of function

#endif  // SRC_HIPBIN_AMD_H_
