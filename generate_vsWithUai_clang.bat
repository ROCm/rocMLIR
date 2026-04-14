@echo off
REM Build rocMLIR + DXGML using UAI tool and generate Visual Studio solution (ClangCL)
REM This script configures with UAI then generates VS solution with Clang toolset

setlocal enabledelayedexpansion

echo ==========================================
echo rocMLIR + DXGML Visual Studio Solution (ClangCL)
echo ==========================================
echo.

REM Check for help flag
if "%1"=="--help" goto ShowHelp
if "%1"=="-h" goto ShowHelp
if "%1"=="/?" goto ShowHelp
if "%2"=="--help" goto ShowHelp
if "%2"=="-h" goto ShowHelp
if "%2"=="/?" goto ShowHelp
if "%3"=="--help" goto ShowHelp
if "%3"=="-h" goto ShowHelp
if "%3"=="/?" goto ShowHelp

REM Check if uai command exists
where uai >nul 2>&1
if errorlevel 1 (
    echo Error: UAI tool not found in PATH
    echo.
    echo UAI tool location: C:\uaitool
    echo.
    echo Please ensure UAI is in your PATH or run from UAI folder
    exit /b 1
)

echo UAI tool found
echo.

REM Check if clang-cl is available
where clang-cl >nul 2>&1
if errorlevel 1 (
    echo Error: clang-cl not found in PATH
    echo.
    echo Expected ROCm clang location: C:\opt\rocm\bin\clang-cl.exe
    echo.
    echo Please ensure clang-cl is installed and in PATH before running this script.
    exit /b 1
)

echo clang-cl found
echo.

REM Auto-detect Windows ROCm installation
set ROCM_PATH=
set ENABLE_ROCM=OFF
set ROCM_TEST_CHIPSET=
set ROCM_AGENT_ENUMERATOR=

if exist "C:\opt\rocm\bin\hipconfig.exe" (
    set ROCM_PATH=C:\opt\rocm
    set ENABLE_ROCM=ON
    goto RocmFound
)

for /d %%d in ("C:\Program Files\AMD\ROCm\*") do (
    if exist "%%d\bin\hipconfig.exe" (
        set ROCM_PATH=%%d
        set ENABLE_ROCM=ON
        goto RocmFound
    )
)

:RocmFound
if "%ENABLE_ROCM%"=="ON" (
    if exist "%ROCM_PATH%\bin\rocm_agent_enumerator.exe" (
        set ROCM_AGENT_ENUMERATOR=%ROCM_PATH%\bin\rocm_agent_enumerator.exe
    ) else (
        set ROCM_TEST_CHIPSET=gfx1200
    )
    echo Found Windows ROCm at: %ROCM_PATH%
    if defined ROCM_AGENT_ENUMERATOR (
        echo ROCm agent enumerator: %ROCM_AGENT_ENUMERATOR%
    ) else (
        echo ROCm agent enumerator not found; using ROCM_TEST_CHIPSET=%ROCM_TEST_CHIPSET%
    )
) else (
    echo Warning: Windows ROCm not found.
    echo   Expected locations checked: C:\opt\rocm and C:\Program Files\AMD\ROCm\*
)
echo.

REM Choose build configuration
set CONFIG=Debug
set SKIP_CONFIGURE=false
set SKIP_BUILD=false
set BUILD_RUNNER=false
set ARG1=%~1
set ARG2=%~2
set ARG3=%~3
set OPT1=
set OPT2=
set OPT3=

if /I "%ARG1%"=="Debug" (
    set CONFIG=%ARG1%
    set OPT1=%ARG2%
    set OPT2=%ARG3%
) else if /I "%ARG1%"=="Release" (
    set CONFIG=%ARG1%
    set OPT1=%ARG2%
    set OPT2=%ARG3%
) else if /I "%ARG1%"=="RelWithDebInfo" (
    set CONFIG=%ARG1%
    set OPT1=%ARG2%
    set OPT2=%ARG3%
) else if /I "%ARG1%"=="MinSizeRel" (
    set CONFIG=%ARG1%
    set OPT1=%ARG2%
    set OPT2=%ARG3%
) else (
    set OPT1=%ARG1%
    set OPT2=%ARG2%
    set OPT3=%ARG3%
)

if "%OPT1%"=="--skip-configure" set SKIP_CONFIGURE=true
if "%OPT1%"=="--build-only" set SKIP_CONFIGURE=true
if "%OPT2%"=="--skip-configure" set SKIP_CONFIGURE=true
if "%OPT2%"=="--build-only" set SKIP_CONFIGURE=true
if "%OPT3%"=="--skip-configure" set SKIP_CONFIGURE=true
if "%OPT3%"=="--build-only" set SKIP_CONFIGURE=true

if "%OPT1%"=="--skip-build" set SKIP_BUILD=true
if "%OPT1%"=="--no-build" set SKIP_BUILD=true
if "%OPT2%"=="--skip-build" set SKIP_BUILD=true
if "%OPT2%"=="--no-build" set SKIP_BUILD=true
if "%OPT3%"=="--skip-build" set SKIP_BUILD=true
if "%OPT3%"=="--no-build" set SKIP_BUILD=true

if "%OPT1%"=="--build-runner" set BUILD_RUNNER=true
if "%OPT2%"=="--build-runner" set BUILD_RUNNER=true
if "%OPT3%"=="--build-runner" set BUILD_RUNNER=true

if "%SKIP_CONFIGURE%"=="true" goto SkipConfigure

echo Build Configuration: %CONFIG%
echo NOTE: Default is Debug to match Visual Studio GUI default and avoid runtime library mismatch errors
echo      To use Release, run: generate_vsWithUai_clang.bat Release
echo      To skip configure and build only: generate_vsWithUai_clang.bat %CONFIG% --skip-configure
echo      To generate solution only (no build): generate_vsWithUai_clang.bat %CONFIG% --skip-build
if "%BUILD_RUNNER%"=="true" echo      To build runner-related targets only: generate_vsWithUai_clang.bat %CONFIG% --build-runner
echo.

echo ==========================================
echo Generate Visual Studio Solution (ClangCL)
echo ==========================================
echo.

echo Generating VS2022 ClangCL solution in build_vs_clang folder...
echo.

REM Create build_vs_clang directory if it doesn't exist
if not exist build_vs_clang mkdir build_vs_clang

cd build_vs_clang

REM Build appropriate CMake flags based on ROCm detection
set CMAKE_ROCM_FLAGS=
if "%ENABLE_ROCM%"=="ON" (
    set ROCM_PATH_CMAKE=%ROCM_PATH:\=/%
    set CMAKE_ROCM_FLAGS=-DROCM_PATH=%ROCM_PATH_CMAKE% -DHIP_PATH=%ROCM_PATH_CMAKE%
    if defined ROCM_TEST_CHIPSET (
        set CMAKE_ROCM_FLAGS=%CMAKE_ROCM_FLAGS% -DROCM_TEST_CHIPSET=%ROCM_TEST_CHIPSET%
    )
)

REM Generate Visual Studio 2022 solution with ClangCL toolset
cmake -G "Visual Studio 17 2022" ^
  -A x64 ^
  -T ClangCL ^
  -DCMAKE_BUILD_TYPE=%CONFIG% ^
  %CMAKE_ROCM_FLAGS% ^
  -DBUILD_FAT_LIBROCKCOMPILER=ON ^
  -DMLIR_ENABLE_ROCM_RUNNER=ON ^
  -DMHAL_ENABLE_HOST_RUNNER=ON ^
  -DMLIR_INCLUDE_TESTS=OFF ^
  -DLLVM_DISABLE_ASSEMBLY_FILES=ON ^
  -DROCMLIR_USE_BINSKIM_COMPLIANT_COMPILE_FLAGS=ON ^
  ..

if errorlevel 1 (
    echo.
    echo ==========================================
    echo VS ClangCL Solution Generation Failed!
    echo ==========================================
    echo.
    echo Check the error output above.
    cd ..
    exit /b 1
)

echo.
echo ==========================================
echo Visual Studio ClangCL Solution Generated!
echo ==========================================
echo.

REM Find .sln file
for %%f in (*.sln) do (
    echo Solution file: %%f
    echo Location: build_vs_clang\%%f
    echo.
    echo You can now open this in Visual Studio 2022:
    echo   start %%f
    echo.
    echo Or build from command line:
    echo   cmake --build . --config %CONFIG%
    echo.
    set SOLUTION=%%f
)

goto BuildStep

:SkipConfigure
echo.
echo ==========================================
echo Skipping CMake Configure Step
echo ==========================================
echo.
echo Using existing solution in build_vs_clang folder...
if not exist build_vs_clang (
    echo ERROR: build_vs_clang folder does not exist!
    echo You must run without --skip-configure first to generate the solution.
    exit /b 1
)
cd build_vs_clang
for %%f in (*.sln) do set SOLUTION=%%f
echo Build Configuration: %CONFIG%
echo.

:BuildStep
REM If skip-build is set, exit after configure/skip-configure
if "%SKIP_BUILD%"=="true" (
    echo.
    echo ==========================================
    echo Skipping Build Step
    echo ==========================================
    echo.
    echo Visual Studio ClangCL solution generated successfully!
    echo Open the solution to build manually:
    echo   cd build_vs_clang
    if defined SOLUTION (
      echo   start %SOLUTION%
    ) else (
      for %%f in (*.sln) do echo   start %%f
    )
    echo.
    cd ..
    exit /b 0
)

if "%BUILD_RUNNER%"=="true" goto RunnerBuildStep

echo.
echo ==========================================
echo Building Solution (ClangCL)
echo ==========================================
echo.
echo Building with configuration: %CONFIG%
echo NOTE: Building in %CONFIG% mode to match the configuration setting
echo      This ensures runtime library compatibility when opening in Visual Studio
echo This may take several minutes...
echo.

cmake --build . --config %CONFIG% --parallel

if errorlevel 1 (
    echo.
    echo ==========================================
    echo Build Failed!
    echo ==========================================
    echo.
    echo The build encountered errors. Check the output above for details.
    echo.
    echo You can:
    echo 1. Open the solution in Visual Studio for detailed error messages:
    echo    cd build_vs_clang
    echo    start %SOLUTION%
    echo.
    echo 2. Try building again with verbose output:
    echo    cd build_vs_clang
    echo    cmake --build . --config %CONFIG% --verbose
    echo.
    cd ..
    exit /b 1
)

echo.
echo ==========================================
echo Build Successful!
echo ==========================================
echo.
echo Solution built in:
echo   build_vs_clang\%CONFIG%
echo.
echo IMPORTANT: When building in Visual Studio GUI, use %CONFIG% configuration
echo            to avoid runtime library mismatch errors.
echo.

echo ==========================================
echo Next Steps
echo ==========================================
echo.
echo 1. Run tests from mlir\test directories
echo.
echo 2. Open solution in Visual Studio for development:
echo    cd build_vs_clang
echo    start %SOLUTION%
echo.
echo 3. Rebuild a specific configuration:
echo    cd build_vs_clang
echo    cmake --build . --config Debug --parallel
echo    cmake --build . --config Release --parallel
echo.
echo 4. Or continue with UAI build workflow:
echo    uai bootstrap -p %CONFIG% --skip-git --skip-hip-sdk --skip-prep
echo.
echo ==========================================

exit /b 0

:RunnerBuildStep
echo.
echo ==========================================
echo Building Runner Targets (ClangCL)
echo ==========================================
echo.
echo Building runner runtime dependencies with configuration: %CONFIG%
echo.

cmake --build . --config %CONFIG% --target ^
  external/llvm-project/llvm/tools/mlir/lib/ExecutionEngine/mlir_runner_utils ^
  external/llvm-project/llvm/tools/mlir/lib/ExecutionEngine/mlir_c_runner_utils ^
  external/llvm-project/llvm/tools/mlir/lib/ExecutionEngine/mlir_async_runtime ^
  external/llvm-project/llvm/tools/mlir/lib/ExecutionEngine/mlir_rocm_runtime ^
  external/llvm-project/llvm/utils/FileCheck/FileCheck ^
  mlir/lib/ExecutionEngine/conv-validation-wrappers ^
  --parallel

if errorlevel 1 (
    echo.
    echo ==========================================
    echo Runner Dependency Build Failed!
    echo ==========================================
    echo.
    echo The runtime dependency targets encountered errors. Check the output above for details.
    cd ..
    exit /b 1
)

echo.
echo Building xmir-runner support targets...
echo.

cmake --build . --config %CONFIG% --target MLIRRocmExecutionEngineUtils xmir-runner --parallel

if errorlevel 1 (
    echo.
    echo ==========================================
    echo xmir-runner Build Failed!
    echo ==========================================
    echo.
    echo The xmir-runner targets encountered errors. Check the output above for details.
    cd ..
    exit /b 1
)

echo.
echo Building rocMLIR command-line tools...
echo.

cmake --build . --config %CONFIG% --target rocmlir-driver rocmlir-gen --parallel

if errorlevel 1 (
    echo.
    echo ==========================================
    echo Tool Build Failed!
    echo ==========================================
    echo.
    echo The rocMLIR tool targets encountered errors. Check the output above for details.
    cd ..
    exit /b 1
)

echo.
echo ==========================================
echo Runner Build Successful!
echo ==========================================
echo.
echo Built runner-related targets in:
echo   build_vs_clang\%CONFIG%
echo.
echo Included targets:
echo   - MLIR runner runtimes and FileCheck
echo   - MLIRRocmExecutionEngineUtils and xmir-runner
echo   - rocmlir-driver and rocmlir-gen
echo.

cd ..

echo ==========================================
echo Next Steps
echo ==========================================
echo.
echo 1. Run xmir-based examples or tests
echo.
echo 2. Open solution in Visual Studio for development:
echo    cd build_vs_clang
echo    start %SOLUTION%
echo.
echo 3. Rebuild runner-only targets after code changes:
echo    generate_vsWithUai_clang.bat %CONFIG% --build-runner --skip-configure
echo.
echo 4. Or continue with UAI build workflow:
echo    uai bootstrap -p %CONFIG% --skip-git --skip-hip-sdk --skip-prep
echo.
echo ==========================================

exit /b 0

:ShowHelp
echo.
echo ==========================================
echo HELP - Command Options
echo ==========================================
echo.
echo USAGE:
echo   generate_vsWithUai_clang.bat [CONFIG] [OPTIONS]
echo.
echo PARAMETERS:
echo   CONFIG              Build configuration (default: Debug)
echo                       Options: Debug, Release, RelWithDebInfo, MinSizeRel
echo.
echo OPTIONS:
echo   --help, -h, /?      Show this help message
echo   --skip-build        Generate VS solution only, skip automatic build
echo   --no-build          Alias for --skip-build
echo   --skip-configure    Skip CMake configure, build existing solution only
echo   --build-only        Alias for --skip-configure
echo   --build-runner      Build runner-related targets instead of full solution
echo.
echo EXAMPLES:
echo   Full workflow (configure + build):
echo     generate_vsWithUai_clang.bat
echo     generate_vsWithUai_clang.bat Debug
echo     generate_vsWithUai_clang.bat Release
echo.
echo   Generate solution only (for manual build in VS):
echo     generate_vsWithUai_clang.bat --skip-build
echo     generate_vsWithUai_clang.bat Release --skip-build
echo     generate_vsWithUai_clang.bat Debug --no-build
echo.
echo   Skip configure, build only (faster for code changes):
echo     generate_vsWithUai_clang.bat --skip-configure
echo     generate_vsWithUai_clang.bat Release --skip-configure
echo     generate_vsWithUai_clang.bat Debug --build-only
echo.
echo   Build runner-related targets only:
echo     generate_vsWithUai_clang.bat --build-runner
echo     generate_vsWithUai_clang.bat Debug --build-runner
echo     generate_vsWithUai_clang.bat Release --build-runner --skip-configure
echo.
echo WORKFLOW SCENARIOS:
echo   First-time setup:
echo     generate_vsWithUai_clang.bat Debug
echo     (Generates solution and builds)
echo.
echo   Want to build in Visual Studio:
echo     generate_vsWithUai_clang.bat Debug --skip-build
echo     cd build_vs_clang
echo     start rocMLIR.sln
echo.
echo   Quick rebuild after code changes:
echo     generate_vsWithUai_clang.bat Debug --skip-configure
echo     (Faster - skips CMake configure step)
echo.
echo   Build xmir-runner and related utilities:
echo     generate_vsWithUai_clang.bat Debug --build-runner
echo     (Builds runtime libraries, FileCheck, xmir-runner, rocmlir-driver, rocmlir-gen)
echo.
echo   Switch configurations:
echo     generate_vsWithUai_clang.bat Release --skip-configure
echo     (Build Release using existing solution)
echo.
echo NOTES:
echo   - Toolset uses ClangCL for Visual Studio 2022
echo   - MLIR_ENABLE_ROCM_RUNNER is ON so xmir-runner target is generated
echo   - LLVM assembly files are disabled to avoid MASM/Clang flag conflicts on Windows
echo   - Default configuration is Debug (matches VS GUI default)
echo   - UAI tool and clang-cl must be in PATH
echo   - Solution generated in build_vs_clang\ directory
echo.
echo ==========================================
exit /b 0
