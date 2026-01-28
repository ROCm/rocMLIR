@echo off
REM Script to run MIGraphX dialect examples with rocmlir-driver on Windows
REM Usage: run_examples.bat [example_name] [arch] [pipeline]
REM
REM Examples:
REM   run_examples.bat simple gfx1150 gpu
REM   run_examples.bat conv gfx1150 full
REM   run_examples.bat simple gfx1150 validate

setlocal enabledelayedexpansion

REM Default values
set EXAMPLE=%1
if "%EXAMPLE%"=="" set EXAMPLE=simple

set ARCH=%2
if "%ARCH%"=="" set ARCH=gfx1150

set PIPELINE=%3
if "%PIPELINE%"=="" set PIPELINE=gpu

REM Determine build directory (assume we're in examples\migraphx-dialect)
set SCRIPT_DIR=%~dp0
set BUILD_DIR=%SCRIPT_DIR%..\..\build

REM Check if build directory exists
if not exist "%BUILD_DIR%" (
    echo Error: Build directory not found at %BUILD_DIR%
    echo Please build rocMLIR first.
    exit /b 1
)

REM Check if rocmlir-driver exists
if not exist "%BUILD_DIR%\bin\rocmlir-driver.exe" (
    echo Error: rocmlir-driver.exe not found
    echo Please build rocMLIR first: cd build ^&^& ninja
    exit /b 1
)

REM Map example name to file
if /i "%EXAMPLE%"=="simple" (
    set EXAMPLE_FILE=migraphx_simple_example.mlir
    set EXAMPLE_DESC=Simple GEMM + ReLU
) else if /i "%EXAMPLE%"=="gemm" (
    set EXAMPLE_FILE=migraphx_simple_example.mlir
    set EXAMPLE_DESC=Simple GEMM + ReLU
) else if /i "%EXAMPLE%"=="conv" (
    set EXAMPLE_FILE=migraphx_convolution_example.mlir
    set EXAMPLE_DESC=Convolution + BatchNorm + ReLU + Pooling
) else if /i "%EXAMPLE%"=="convolution" (
    set EXAMPLE_FILE=migraphx_convolution_example.mlir
    set EXAMPLE_DESC=Convolution + BatchNorm + ReLU + Pooling
) else (
    echo Unknown example: %EXAMPLE%
    echo Available examples: simple, conv
    exit /b 1
)

set EXAMPLE_PATH=%SCRIPT_DIR%\%EXAMPLE_FILE%

REM Check if example file exists
if not exist "%EXAMPLE_PATH%" (
    echo Error: Example file not found: %EXAMPLE_PATH%
    exit /b 1
)

echo ======================================
echo MIGraphX Dialect Example Runner
echo ======================================
echo Example:    %EXAMPLE_DESC%
echo File:       %EXAMPLE_FILE%
echo Arch:       %ARCH%
echo Pipeline:   %PIPELINE%
echo ======================================
echo.

REM Change to build directory
cd /d "%BUILD_DIR%"

REM Run based on pipeline selection
if /i "%PIPELINE%"=="validate" goto validate
if /i "%PIPELINE%"=="parse" goto validate
if /i "%PIPELINE%"=="gpu" goto gpu
if /i "%PIPELINE%"=="rocdl" goto rocdl
if /i "%PIPELINE%"=="binary" goto binary
if /i "%PIPELINE%"=="full" goto full
if /i "%PIPELINE%"=="all" goto full
if /i "%PIPELINE%"=="view" goto view
if /i "%PIPELINE%"=="inspect" goto view
if /i "%PIPELINE%"=="debug" goto debug

echo Unknown pipeline: %PIPELINE%
echo Available pipelines:
echo   - validate : Parse and validate MLIR
echo   - gpu      : Lower to GPU dialect
echo   - rocdl    : Lower to ROCDL
echo   - binary   : Generate binary
echo   - full     : Full compilation
echo   - view     : View intermediate IR
echo   - debug    : Debug output
exit /b 1

:validate
echo Step 1: Parsing and validating MLIR...
bin\rocmlir-opt.exe "%EXAMPLE_PATH%"
if errorlevel 1 (
    echo Validation failed!
    exit /b 1
)
echo [OK] Validation successful!
goto end

:gpu
echo Step 1: Lowering to GPU dialect...
bin\rocmlir-driver.exe "%EXAMPLE_PATH%" --kernel-pipeline=gpu --arch=%ARCH% --verify-passes
if errorlevel 1 (
    echo GPU lowering failed!
    exit /b 1
)
echo [OK] GPU lowering successful!
goto end

:rocdl
echo Step 1: Lowering to GPU dialect...
echo Step 2: Lowering to ROCDL...
bin\rocmlir-driver.exe "%EXAMPLE_PATH%" --kernel-pipeline=gpu,rocdl --arch=%ARCH% --verify-passes
if errorlevel 1 (
    echo ROCDL lowering failed!
    exit /b 1
)
echo [OK] ROCDL lowering successful!
goto end

:binary
echo Step 1: Lowering to GPU dialect...
echo Step 2: Lowering to ROCDL...
echo Step 3: Generating binary...
bin\rocmlir-driver.exe "%EXAMPLE_PATH%" --kernel-pipeline=gpu,binary --arch=%ARCH% --verify-passes
if errorlevel 1 (
    echo Binary generation failed!
    exit /b 1
)
echo [OK] Binary generation successful!
goto end

:full
echo Running full compilation pipeline...
bin\rocmlir-driver.exe "%EXAMPLE_PATH%" -c --arch=%ARCH% --verify-passes
if errorlevel 1 (
    echo Full compilation failed!
    exit /b 1
)
echo [OK] Full compilation successful!
goto end

:view
echo Lowering to GPU dialect and viewing IR...
bin\rocmlir-driver.exe "%EXAMPLE_PATH%" --kernel-pipeline=gpu --arch=%ARCH% | bin\rocmlir-opt.exe
goto end

:debug
echo Running with debug output...
bin\rocmlir-driver.exe "%EXAMPLE_PATH%" -c --arch=%ARCH% --debug-only=serialize-to-blob 2>&1 | more
echo (Use 'more' to view output page by page)
goto end

:end
echo.
echo Done!
