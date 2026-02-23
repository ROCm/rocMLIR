@echo off
REM Build rocMLIR + DXGML using UAI tool and generate Visual Studio solution
REM This script configures with UAI then generates VS solution

setlocal enabledelayedexpansion

echo ==========================================
echo rocMLIR + DXGML Visual Studio Solution
echo ==========================================
echo.

REM Check for help flag
if "%1"=="--help" goto ShowHelp
if "%1"=="-h" goto ShowHelp
if "%1"=="/?" goto ShowHelp
if "%2"=="--help" goto ShowHelp
if "%2"=="-h" goto ShowHelp
if "%2"=="/?" goto ShowHelp

REM Check if uai command exists
where uai >nul 2>&1
if errorlevel 1 (
    echo Error: UAI tool not found in PATH
    echo.
    echo UAI tool location: C:\Users\hisha\Documents\RocM\uaitool
    echo.
    echo Please ensure UAI is in your PATH or run from UAI folder
    pause
    exit /b 1
)

echo UAI tool found
echo.

REM Choose build configuration
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Debug

REM Check for skip-configure flag
set SKIP_CONFIGURE=%2
if "%SKIP_CONFIGURE%"=="--skip-configure" goto SkipConfigure
if "%SKIP_CONFIGURE%"=="--build-only" goto SkipConfigure

REM Check for skip-build flag
set SKIP_BUILD=%2
if "%SKIP_BUILD%"=="--skip-build" set SKIP_BUILD=true
if "%SKIP_BUILD%"=="--no-build" set SKIP_BUILD=true

echo Build Configuration: %CONFIG%
echo NOTE: Default is Debug to match Visual Studio GUI default and avoid runtime library mismatch errors
echo      To use Release, run: build_vs_with_uai.bat Release
echo      To skip configure and build only: build_vs_with_uai.bat %CONFIG% --skip-configure
echo      To generate solution only (no build): build_vs_with_uai.bat %CONFIG% --skip-build
echo.

echo ==========================================
echo Generate Visual Studio Solution
echo ==========================================
echo.

echo Generating VS2022 solution in build_vs folder...
echo.

REM Create build_vs directory if it doesn't exist
if not exist build_vs mkdir build_vs

cd build_vs

REM Generate Visual Studio 2022 solution
cmake -G "Visual Studio 17 2022" ^
  -A x64 ^
  -T host=x64 ^
  -DCMAKE_BUILD_TYPE=%CONFIG% ^
  -DBUILD_FAT_LIBROCKCOMPILER=ON ^
  -DMLIR_INCLUDE_TESTS=OFF ^
  -DROCMLIR_USE_BINSKIM_COMPLIANT_COMPILE_FLAGS=ON ^
  ..

if errorlevel 1 (
    echo.
    echo ==========================================
    echo VS Solution Generation Failed!
    echo ==========================================
    echo.
    echo Check the error output above.
    cd ..
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Visual Studio Solution Generated!
echo ==========================================
echo.

REM Find .sln file
for %%f in (*.sln) do (
    echo Solution file: %%f
    echo Location: build_vs\%%f
    echo.
    echo You can now open this in Visual Studio 2022:
    echo   start %%f
    echo.
    echo Or build from command line:
    echo   cmake --build . --config %CONFIG%
    echo.
    set SOLUTION=%%f
)

:SkipConfigure
REM If skip-build is set, exit after configure
if "%SKIP_BUILD%"=="true" (
    echo.
    echo ==========================================
    echo Skipping Build Step
    echo ==========================================
    echo.
    echo Visual Studio solution generated successfully!
    echo Open the solution to build manually:
    echo   cd build_vs
    for %%f in (*.sln) do echo   start %%f
    echo.
    cd ..
    pause
    exit /b 0
)

REM Check if build_vs directory exists when skipping configure
if "%SKIP_CONFIGURE%"=="--skip-configure" (
    echo.
    echo ==========================================
    echo Skipping CMake Configure Step
    echo ==========================================
    echo.
    echo Using existing solution in build_vs folder...
    if not exist build_vs (
        echo ERROR: build_vs folder does not exist!
        echo You must run without --skip-configure first to generate the solution.
        pause
        exit /b 1
    )
    cd build_vs
    echo Build Configuration: %CONFIG%
    echo.
)

if "%SKIP_CONFIGURE%"=="--build-only" (
    echo.
    echo ==========================================
    echo Skipping CMake Configure Step
    echo ==========================================
    echo.
    echo Using existing solution in build_vs folder...
    if not exist build_vs (
        echo ERROR: build_vs folder does not exist!
        echo You must run without --build-only first to generate the solution.
        pause
        exit /b 1
    )
    cd build_vs
    echo Build Configuration: %CONFIG%
    echo.
)

echo.
echo ==========================================
echo Building Solution
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
    echo    cd build_vs
    echo    start %SOLUTION%
    echo.
    echo 2. Try building again with verbose output:
    echo    cd build_vs
    echo    cmake --build . --config %CONFIG% --verbose
    echo.
    cd ..
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Build Successful!
echo ==========================================
echo.
echo Solution built in:
echo   build_vs\%CONFIG%
echo.
echo IMPORTANT: When building in Visual Studio GUI, use %CONFIG% configuration
echo            to avoid runtime library mismatch errors.
echo.

cd ..

echo ==========================================
echo Next Steps
echo ==========================================
echo.
echo 1. Run tests from mlir\test directories
echo.
echo 2. Open solution in Visual Studio for development:
echo    cd build_vs
echo    start %SOLUTION%
echo.
echo 3. Rebuild a specific configuration:
echo    cd build_vs
echo    cmake --build . --config Debug --parallel
echo    cmake --build . --config Release --parallel
echo.
echo 4. Or continue with UAI build workflow:
echo    uai bootstrap -p %CONFIG% --skip-git --skip-hip-sdk --skip-prep
echo.
echo ==========================================

pause
exit /b 0

:ShowHelp
echo.
echo ==========================================
echo HELP - Command Options
echo ==========================================
echo.
echo USAGE:
echo   build_vs_with_uai.bat [CONFIG] [OPTIONS]
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
echo.
echo EXAMPLES:
echo   Full workflow (configure + build):
echo     build_vs_with_uai.bat
echo     build_vs_with_uai.bat Debug
echo     build_vs_with_uai.bat Release
echo.
echo   Generate solution only (for manual build in VS):
echo     build_vs_with_uai.bat --skip-build
echo     build_vs_with_uai.bat Release --skip-build
echo     build_vs_with_uai.bat Debug --no-build
echo.
echo   Skip configure, build only (faster for code changes):
echo     build_vs_with_uai.bat --skip-configure
echo     build_vs_with_uai.bat Release --skip-configure
echo     build_vs_with_uai.bat Debug --build-only
echo.
echo WORKFLOW SCENARIOS:
echo   First-time setup:
echo     build_vs_with_uai.bat Debug
echo     (Generates solution and builds)
echo.
echo   Want to build in Visual Studio:
echo     build_vs_with_uai.bat Debug --skip-build
echo     cd build_vs
echo     start rocMLIR.sln
echo.
echo   Quick rebuild after code changes:
echo     build_vs_with_uai.bat Debug --skip-configure
echo     (Faster - skips CMake configure step)
echo.
echo   Switch configurations:
echo     build_vs_with_uai.bat Release --skip-configure
echo     (Build Release using existing solution)
echo.
echo NOTES:
echo   - Default configuration is Debug (matches VS GUI default)
echo   - This avoids runtime library mismatch errors
echo   - UAI tool must be in PATH or run from UAI folder
echo   - Solution generated in build_vs\ directory
echo.
echo ==========================================
pause
exit /b 0
