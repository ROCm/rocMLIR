@echo off
REM Build rocMLIR with DXGML dialect using UAI tool
REM This is the recommended build method

setlocal enabledelayedexpansion

echo ==========================================
echo rocMLIR + DXGML Build with UAI Tool
echo ==========================================
echo.

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
if "%CONFIG%"=="" set CONFIG=WinDebug

echo Build Configuration: %CONFIG%
echo.

echo ==========================================
echo Stage 1: Bootstrap Build
echo ==========================================
echo.

REM Run UAI bootstrap
echo Running: uai bootstrap -p %CONFIG% --skip-git --skip-hip-sdk
echo.
echo This will:
echo  1. Configure CMake
echo  2. Build all LLVM/MLIR components
echo  3. Build rocMLIR components including DXGML
echo.
echo Build may take 30-60 minutes...
echo.

uai bootstrap -p %CONFIG% --skip-git --skip-hip-sdk

if errorlevel 1 (
    echo.
    echo ==========================================
    echo Build Failed!
    echo ==========================================
    echo.
    echo Check the error output above.
    echo Common issues:
    echo  - LLVM Support library build errors
    echo  - ROCm execution engine warnings (can be ignored)
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Build Successful!
echo ==========================================
echo.

REM Check if DXGML libraries were built
set BUILD_DIR=build\%CONFIG%
echo Checking for DXGML dialect libraries...
echo.

if exist "%BUILD_DIR%\lib\MLIRDxgmlDialect.lib" (
    echo [OK] MLIRDxgmlDialect.lib found
) else (
    echo [MISSING] MLIRDxgmlDialect.lib
)

if exist "%BUILD_DIR%\lib\MLIRDxgmlOpDialect.lib" (
    echo [OK] MLIRDxgmlOpDialect.lib found
) else (
    echo [MISSING] MLIRDxgmlOpDialect.lib
)

if exist "%BUILD_DIR%\bin\rocmlir-driver.exe" (
    echo [OK] rocmlir-driver.exe found
) else (
    echo [MISSING] rocmlir-driver.exe
)

echo.
echo ==========================================
echo Next Steps
echo ==========================================
echo.
echo To run DXGML tests:
echo   cd mlir\test\Dialect\Dxgml
echo   run_tests.bat all %CONFIG%
echo.
echo To test with your models:
echo   %BUILD_DIR%\bin\rocmlir-opt.exe ^
echo     C:\Users\hisha\Documents\shared_drive\DxML\DXGML-Drop3.7\Models\model1\model.mlir
echo.
echo ==========================================

pause
